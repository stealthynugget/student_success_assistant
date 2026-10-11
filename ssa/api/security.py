"""API security primitives.

* API key required on every endpoint except ``/api/health`` unless ``SSA_DEV_MODE=1``
  (v4 shipped with authentication disabled by default). Constant-time comparison.
* Thread-safe token-bucket rate limiter with bounded memory (LRU eviction). Client identity
  uses ``X-Forwarded-For`` only when ``TRUSTED_PROXY_COUNT`` > 0 (otherwise spoofable).
  For multi-worker / multi-host deployments put a shared limiter (e.g. nginx / API gateway /
  Redis) in front - documented in docs/DEPLOYMENT.md.
* CORS only for an explicit allow-list (``*`` is never echoed).
* Errors return a generic message + request id; stack traces go to the server log only.
"""
from __future__ import annotations

import hmac
import threading
import time
import uuid
from collections import OrderedDict



class TokenBucketLimiter:
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
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            ok = tokens >= 1.0
            if ok:
                tokens -= 1.0
            self._buckets[key] = (tokens, now)
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
            retry_after = 0.0 if ok else (1.0 - tokens) / self.refill
        return ok, retry_after


def client_ip(req, trusted_proxy_count: int) -> str:
    """Works with Starlette/FastAPI requests (``request.client.host``) and WSGI-style ``remote_addr``."""
    if trusted_proxy_count > 0:
        chain = [p.strip() for p in req.headers.get("X-Forwarded-For", "").split(",") if p.strip()]
        if len(chain) >= trusted_proxy_count:
            return chain[-trusted_proxy_count]
    client = getattr(req, "client", None)
    return (getattr(client, "host", None) if client else None) or getattr(req, "remote_addr", None) or "unknown"


def api_key_ok(provided: str | None, expected: str | None) -> bool:
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())


def new_request_id(req) -> str:
    rid = req.headers.get("X-Request-ID", "")
    if rid and len(rid) <= 64 and rid.replace("-", "").isalnum():
        return rid
    return uuid.uuid4().hex
