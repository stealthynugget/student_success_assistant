"""State the API shares between requests: background-job records, rate-limit buckets and recent analyses.

``SHARED_STATE=memory`` (default) keeps it in the process: right for one worker per container.
``SHARED_STATE=sqlite`` keeps it in one SQLite file (WAL) under ``SSA_HOME``, so several uvicorn workers on the same host
(``--workers N``, same volume) see the same jobs, limits and analyses: a job submitted to one worker can be polled
through any other. Across hosts, put the limits in the gateway and use a shared queue (see docs/DEPLOYMENT.md).

A job still runs in the worker that accepted it; only its record is shared. If that worker dies, the record stops
updating and is reported as lost after ``stale_s``.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created);
CREATE TABLE IF NOT EXISTS rate_buckets (key TEXT PRIMARY KEY, tokens REAL NOT NULL, last REAL NOT NULL);
CREATE TABLE IF NOT EXISTS kv (ns TEXT NOT NULL, key TEXT NOT NULL, body TEXT NOT NULL, ts REAL NOT NULL, PRIMARY KEY (ns, key));
CREATE INDEX IF NOT EXISTS idx_kv_ts ON kv(ns, ts);
"""


# ------------------------------------------------------------------ job records
class MemoryJobStore:
    def __init__(self, max_jobs: int = 500, ttl_s: int = 1800):
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self.max_jobs, self.ttl_s = max_jobs, ttl_s

    def put(self, job: dict) -> None:
        with self._lock:
            now = time.time()
            for k in [k for k, j in self._jobs.items() if now - j["created"] > self.ttl_s]:
                self._jobs.pop(k, None)
            self._jobs[job["id"]] = json.loads(_dumps(job))
            self._jobs.move_to_end(job["id"])
            while len(self._jobs) > self.max_jobs:
                self._jobs.popitem(last=False)

    def get(self, jid: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(jid)
            return json.loads(json.dumps(j)) if j else None


class SQLiteJobStore:
    def __init__(self, path: Path, max_jobs: int = 500, ttl_s: int = 1800):
        self.db = _DB(path)
        self.max_jobs, self.ttl_s = max_jobs, ttl_s

    def put(self, job: dict) -> None:
        body = _dumps(job)
        with self.db.tx() as c:
            c.execute("INSERT INTO jobs(id, body, created, updated) VALUES (?, ?, ?, ?) "
                      "ON CONFLICT(id) DO UPDATE SET body = excluded.body, updated = excluded.updated",
                      (job["id"], body, job["created"], job["updated"]))
            if job["status"] == "queued":  # prune on submit only
                c.execute("DELETE FROM jobs WHERE created < ?", (time.time() - self.ttl_s,))
                c.execute("DELETE FROM jobs WHERE id NOT IN (SELECT id FROM jobs ORDER BY created DESC LIMIT ?)", (self.max_jobs,))

    def get(self, jid: str) -> dict | None:
        with self.db.tx() as c:
            row = c.execute("SELECT body FROM jobs WHERE id = ? AND created >= ?", (jid, time.time() - self.ttl_s)).fetchone()
        return json.loads(row[0]) if row else None


# ------------------------------------------------------------------ rate limiting (token bucket per client)
class MemoryRateLimiter:
    def __init__(self, rate_per_minute: int, max_keys: int = 10_000):
        self.capacity = max(1, rate_per_minute)
        self.refill = self.capacity / 60.0
        self.max_keys = max_keys
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.pop(key, (float(self.capacity), now))
            tokens, ok, retry = _take(tokens, last, now, self.capacity, self.refill)
            self._buckets[key] = (tokens, now)
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
        return ok, retry


class SQLiteRateLimiter:
    """Same bucket, shared by every worker using the file. Uses wall-clock time, since monotonic clocks differ per process."""

    def __init__(self, path: Path, rate_per_minute: int, max_keys: int = 10_000):
        self.db = _DB(path)
        self.capacity = max(1, rate_per_minute)
        self.refill = self.capacity / 60.0
        self.max_keys = max_keys
        self._n = 0

    def allow(self, key: str) -> tuple[bool, float]:
        now = time.time()
        with self.db.tx(immediate=True) as c:
            row = c.execute("SELECT tokens, last FROM rate_buckets WHERE key = ?", (key,)).fetchone()
            tokens, last = row if row else (float(self.capacity), now)
            tokens, ok, retry = _take(tokens, last, now, self.capacity, self.refill)
            c.execute("INSERT INTO rate_buckets(key, tokens, last) VALUES (?, ?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET tokens = excluded.tokens, last = excluded.last", (key, tokens, now))
            self._n += 1
            if self._n % 500 == 0:  # buckets idle for a minute are full again: forget them
                c.execute("DELETE FROM rate_buckets WHERE last < ?", (now - 60,))
        return ok, retry


def _take(tokens: float, last: float, now: float, capacity: float, refill: float) -> tuple[float, bool, float]:
    tokens = min(capacity, tokens + max(0.0, now - last) * refill)
    ok = tokens >= 1.0
    if ok:
        tokens -= 1.0
    return tokens, ok, 0.0 if ok else (1.0 - tokens) / refill


# ------------------------------------------------------------------ small bounded key-value cache
class MemoryKV:
    def __init__(self, ns: str, max_items: int = 500):
        self.n = max_items
        self._d: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key: str, value) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.n:
                self._d.popitem(last=False)

    def get(self, key: str):
        with self._lock:
            return self._d.get(key)


class SQLiteKV:
    def __init__(self, path: Path, ns: str, max_items: int = 500):
        self.db, self.ns, self.n = _DB(path), ns, max_items

    def put(self, key: str, value) -> None:
        with self.db.tx() as c:
            c.execute("INSERT INTO kv(ns, key, body, ts) VALUES (?, ?, ?, ?) ON CONFLICT(ns, key) DO UPDATE SET body = excluded.body, "
                      "ts = excluded.ts", (self.ns, key, _dumps(value), time.time()))
            c.execute("DELETE FROM kv WHERE ns = ? AND key NOT IN (SELECT key FROM kv WHERE ns = ? ORDER BY ts DESC LIMIT ?)",
                      (self.ns, self.ns, self.n))

    def get(self, key: str):
        with self.db.tx() as c:
            row = c.execute("SELECT body FROM kv WHERE ns = ? AND key = ?", (self.ns, key)).fetchone()
        return json.loads(row[0]) if row else None


# ------------------------------------------------------------------ plumbing
class _DB:
    _init_lock = threading.Lock()
    _ready: set[str] = set()

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _DB._init_lock:
            if str(self.path) not in _DB._ready:
                c = sqlite3.connect(self.path, timeout=10)
                try:
                    c.execute("PRAGMA journal_mode=WAL")
                    c.executescript(SCHEMA)  # manages its own transaction
                finally:
                    c.close()
                _DB._ready.add(str(self.path))

    def tx(self, immediate: bool = False):
        return _Tx(self.path, immediate)


class _Tx:
    def __init__(self, path: Path, immediate: bool):
        self.path, self.immediate = path, immediate

    def __enter__(self) -> sqlite3.Connection:
        self.c = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.execute("BEGIN IMMEDIATE" if self.immediate else "BEGIN")
        return self.c

    def __exit__(self, exc_type, exc, tb):
        try:
            self.c.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.c.close()


def jsonable(o):
    """Plain JSON: numpy scalars/arrays to Python, NaN/inf to null (browsers reject NaN)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if hasattr(o, "tolist"):
        return jsonable(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _dumps(o) -> str:
    return json.dumps(jsonable(o), ensure_ascii=False, default=str)


# ------------------------------------------------------------------ factories
def _backend(settings) -> str:
    b = settings.shared_state
    if b not in ("memory", "sqlite"):
        raise ValueError(f"SHARED_STATE must be 'memory' or 'sqlite', not {b!r}")
    return b


def job_store(settings):
    return SQLiteJobStore(settings.shared_state_db) if _backend(settings) == "sqlite" else MemoryJobStore()


def kv_store(settings, ns: str, max_items: int = 500):
    return SQLiteKV(settings.shared_state_db, ns, max_items) if _backend(settings) == "sqlite" else MemoryKV(ns, max_items)


def rate_limiter(settings):
    """None when RATE_LIMIT_PER_MINUTE=0 (limits enforced by the gateway instead)."""
    if settings.rate_limit_per_minute <= 0:
        return None
    if _backend(settings) == "sqlite":
        return SQLiteRateLimiter(settings.shared_state_db, settings.rate_limit_per_minute)
    return MemoryRateLimiter(settings.rate_limit_per_minute)
