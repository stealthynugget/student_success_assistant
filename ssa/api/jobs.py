"""Background jobs for slow, LLM-heavy operations (analysis, briefing, copilot).

The UI submits a job, then polls ``GET /api/jobs/{id}``; each agent step is appended to ``steps`` as it completes,
so advisors see the multi-agent workflow progress live. A job runs in a thread pool of the worker that accepted it; its
record lives in a job store (``ssa.api.shared``): in-process by default, or SQLite so any worker on the host can answer
the poll. A record that stops updating for ``stale_s`` while queued or running is reported as lost (its worker died)."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from ssa.api.shared import MemoryJobStore
from ssa.utils import get_logger

log = get_logger("ssa.jobs")


class JobManager:
    def __init__(self, store=None, workers: int = 4, stale_s: int = 900):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ssa-job")
        self.store = store or MemoryJobStore()
        self.stale_s = stale_s
        self._lock = threading.Lock()

    def submit(self, kind: str, fn) -> dict:
        jid = uuid.uuid4().hex
        now = time.time()
        job = {"id": jid, "kind": kind, "status": "queued", "steps": [], "result": None, "error": None, "created": now, "updated": now}
        self.store.put(job)

        def save(**changes):
            with self._lock:
                job.update(changes, updated=time.time())
                self.store.put(job)

        def progress(name, payload=None):
            src = payload.get("source") if isinstance(payload, dict) else None
            save(steps=job["steps"] + [{"step": name, "source": src, "t": round(time.time() - job["created"], 2)}])

        def run():
            save(status="running")
            try:
                result = fn(progress)
            except Exception as exc:
                log.error(f"job {jid} failed: {traceback.format_exc()}")
                save(status="error", error={"type": type(exc).__name__, "message": str(exc)[:300]})
                return
            try:
                save(status="done", result=result)
            except Exception as exc:  # e.g. the store could not be written
                log.error(f"job {jid} result not stored: {traceback.format_exc()}")
                save(status="error", error={"type": type(exc).__name__, "message": "result could not be stored"})

        self._pool.submit(run)
        return self.public(job, include_result=False)

    def get(self, jid: str) -> dict | None:
        j = self.store.get(jid)
        if not j:
            return None
        if j["status"] in ("queued", "running") and time.time() - j["updated"] > self.stale_s:
            j["status"] = "error"
            j["error"] = {"type": "JobLost", "message": "the worker running this job stopped responding; run it again"}
        return self.public(j)

    @staticmethod
    def public(j: dict, include_result: bool = True) -> dict:
        out = {k: j[k] for k in ("id", "kind", "status", "steps", "error")}
        out["elapsed_s"] = round(j["updated"] - j["created"], 2)
        if include_result:
            out["result"] = j["result"]
        return out
