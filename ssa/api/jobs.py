"""In-process background jobs for slow, LLM-heavy operations (analysis, briefing, copilot).

The UI submits a job, then polls ``GET /api/jobs/{id}``; each agent step is appended to ``steps`` as it completes,
so advisors see the multi-agent workflow progress live. Jobs expire after ``ttl_s``; the store is bounded.
For multi-worker deployments, run one worker or move jobs to a shared queue (documented in DEPLOYMENT.md)."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from ssa.utils import get_logger

log = get_logger("ssa.jobs")


class JobManager:
    def __init__(self, workers: int = 4, max_jobs: int = 500, ttl_s: int = 1800):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ssa-job")
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self.max_jobs, self.ttl_s = max_jobs, ttl_s

    def _gc(self):
        now = time.time()
        for k in [k for k, j in self._jobs.items() if now - j["created"] > self.ttl_s]:
            self._jobs.pop(k, None)
        while len(self._jobs) > self.max_jobs:
            self._jobs.popitem(last=False)

    def submit(self, kind: str, fn) -> dict:
        jid = uuid.uuid4().hex
        job = {"id": jid, "kind": kind, "status": "queued", "steps": [], "result": None, "error": None,
               "created": time.time(), "updated": time.time()}
        with self._lock:
            self._gc()
            self._jobs[jid] = job

        def progress(name, payload=None):
            src = payload.get("source") if isinstance(payload, dict) else None
            with self._lock:
                job["steps"].append({"step": name, "source": src, "t": round(time.time() - job["created"], 2)})
                job["updated"] = time.time()

        def run():
            job["status"] = "running"
            try:
                job["result"] = fn(progress)
                job["status"] = "done"
            except Exception as exc:
                log.error(f"job {jid} failed: {traceback.format_exc()}")
                job["status"], job["error"] = "error", {"type": type(exc).__name__, "message": str(exc)[:300]}
            job["updated"] = time.time()

        self._pool.submit(run)
        return self.public(job, include_result=False)

    def get(self, jid: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(jid)
            return self.public(j) if j else None

    @staticmethod
    def public(j: dict, include_result: bool = True) -> dict:
        out = {k: j[k] for k in ("id", "kind", "status", "steps", "error")}
        out["elapsed_s"] = round(j["updated"] - j["created"], 2)
        if include_result:
            out["result"] = j["result"]
        return out
