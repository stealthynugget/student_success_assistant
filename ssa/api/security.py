"""API security primitives.

* API key required on every endpoint except ``/api/health`` unless ``SSA_DEV_MODE=1``
  (v4 shipped with authentication disabled by default). Constant-time comparison.
* Token-bucket rate limiting per client (``ssa.api.shared``: in-process, or SQLite shared by the workers on a host).
  Client identity uses ``X-Forwarded-For`` only when ``TRUSTED_PROXY_COUNT`` > 0 (otherwise spoofable).
  Across hosts, enforce limits in the gateway and set ``RATE_LIMIT_PER_MINUTE=0`` (docs/DEPLOYMENT.md).
* CORS only for an explicit allow-list (``*`` is never echoed).
* Errors return a generic message + request id; stack traces go to the server log only.
"""
from __future__ import annotations

import hmac
import uuid

from ssa.api.shared import MemoryRateLimiter

TokenBucketLimiter = MemoryRateLimiter  # kept for callers of the old name


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
