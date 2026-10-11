import time

from ssa.api.jobs import JobManager


def test_job_lifecycle_and_errors():
    jm = JobManager(workers=2)

    def ok(progress):
        progress("step one", {"source": "computed"})
        return {"value": 1}

    def boom(progress):
        raise RuntimeError("bad")

    a, b = jm.submit("ok", ok), jm.submit("boom", boom)
    for _ in range(100):
        if jm.get(a["id"])["status"] == "done" and jm.get(b["id"])["status"] == "error":
            break
        time.sleep(0.02)
    ja, jb = jm.get(a["id"]), jm.get(b["id"])
    assert ja["result"] == {"value": 1} and ja["steps"][0]["step"] == "step one"
    assert jb["error"]["type"] == "RuntimeError"
