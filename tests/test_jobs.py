import threading
import time

import numpy as np
import pytest

from ssa.api.jobs import JobManager
from ssa.api.shared import MemoryJobStore, MemoryRateLimiter, SQLiteJobStore, SQLiteKV, SQLiteRateLimiter


def _wait(jm, jid, status, n=200):
    for _ in range(n):
        j = jm.get(jid)
        if j and j["status"] == status:
            return j
        time.sleep(0.02)
    return jm.get(jid)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_job_lifecycle_and_errors(backend, tmp_path):
    store = SQLiteJobStore(tmp_path / "s.sqlite3") if backend == "sqlite" else MemoryJobStore()
    jm = JobManager(store, workers=2)

    def ok(progress):
        progress("step one", {"source": "computed"})
        return {"value": np.float64(1.0), "bad": float("nan"), "arr": np.arange(2)}

    def boom(progress):
        raise RuntimeError("bad")

    a, b = jm.submit("ok", ok), jm.submit("boom", boom)
    ja, jb = _wait(jm, a["id"], "done"), _wait(jm, b["id"], "error")
    assert ja["result"] == {"value": 1.0, "bad": None, "arr": [0, 1]} and ja["steps"][0]["step"] == "step one"
    assert jb["error"]["type"] == "RuntimeError"


def test_sqlite_jobs_are_visible_to_another_worker(tmp_path):
    path = tmp_path / "shared.sqlite3"
    worker_a, worker_b = JobManager(SQLiteJobStore(path)), JobManager(SQLiteJobStore(path))
    gate = threading.Event()

    def slow(progress):
        progress("first agent", {"source": "llm"})
        gate.wait(5)
        return {"answer": 42}

    jid = worker_a.submit("analysis", slow)["id"]
    for _ in range(200):  # the record is marked running just before the first step is saved
        running = worker_b.get(jid)
        if running["steps"]:
            break
        time.sleep(0.02)
    assert running["status"] == "running" and running["steps"][0]["source"] == "llm"
    gate.set()
    assert _wait(worker_b, jid, "done")["result"] == {"answer": 42}
    assert worker_b.get("no-such-job") is None


def test_lost_job_is_reported(tmp_path):
    store = SQLiteJobStore(tmp_path / "s.sqlite3")
    store.put({"id": "x", "kind": "analysis", "status": "running", "steps": [], "result": None, "error": None,
               "created": time.time() - 2000, "updated": time.time() - 1000})
    store.ttl_s = 3600
    j = JobManager(store, stale_s=900).get("x")
    assert j["status"] == "error" and j["error"]["type"] == "JobLost"


def test_sqlite_rate_limit_is_shared_between_workers(tmp_path):
    path = tmp_path / "s.sqlite3"
    a, b = SQLiteRateLimiter(path, 3), SQLiteRateLimiter(path, 3)
    results = [a.allow("1.2.3.4")[0], b.allow("1.2.3.4")[0], a.allow("1.2.3.4")[0], b.allow("1.2.3.4")]
    assert results[:3] == [True, True, True]
    ok, retry = results[3]
    assert not ok and retry > 0
    assert a.allow("5.6.7.8")[0]  # other clients unaffected
    mem = MemoryRateLimiter(1)
    assert mem.allow("k")[0] and not mem.allow("k")[0]


def test_sqlite_kv_bounded_and_shared(tmp_path):
    path = tmp_path / "s.sqlite3"
    a, b = SQLiteKV(path, "recent", 2), SQLiteKV(path, "recent", 2)
    a.put("r1", {"x": np.int64(1)})
    assert b.get("r1") == {"x": 1}
    a.put("r2", {}), time.sleep(0.01), a.put("r3", {})
    assert b.get("r1") is None and b.get("r3") == {}
