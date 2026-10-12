"""FastAPI application (served by uvicorn / gunicorn with uvicorn workers).

Thin HTTP layer over :class:`ssa.api.handlers.Handlers`. Interactive docs: ``/api/docs`` (OpenAPI at
``/api/openapi.json``). The built React console (``frontend/dist``) is served at ``/`` when present, so the whole
product runs as one microservice.

Security: API key on every route except ``/api/health`` (``X-API-Key``), admin key for applying feedback-loop
changes (``X-Admin-Key``), token-bucket rate limiting, body-size limit, strict validation, explicit CORS allow-list,
security headers and generic error bodies with a request id.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ssa.api import schemas as S
from ssa.api.handlers import Handlers
from ssa.api.security import api_key_ok, client_ip, new_request_id
from ssa.api.shared import rate_limiter
from ssa.config import PROJECT_ROOT, Settings, get_settings
from ssa.feedback.store import UnknownRecommendation
from ssa.service import AlreadyWithdrawn, NotFound
from ssa.utils import get_logger

log = get_logger("ssa.api")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


class SafeJSONResponse(JSONResponse):
    """Never emits NaN/Infinity; understands numpy scalars and arrays."""

    def render(self, content) -> bytes:
        return json.dumps(_clean(content), ensure_ascii=False, allow_nan=False, default=str, separators=(",", ":")).encode("utf-8")


def _err(request: Request, status: int, message: str, details=None):
    body = {"error": message, "request_id": getattr(request.state, "request_id", None)}
    if details is not None:
        body["details"] = details
    return SafeJSONResponse(body, status_code=status)


def create_app(settings: Settings | None = None, service=None, llm=None) -> FastAPI:
    s = (settings or get_settings()).at(None)
    if not s.api_key and not s.dev_mode:
        log.warning("STUDENT_ASSISTANT_API_KEY not set and SSA_DEV_MODE off: protected endpoints return 503")
    h = Handlers(s, service=service, llm=llm)
    limiter = rate_limiter(s)
    workers = os.getenv("WEB_CONCURRENCY", "1")
    if workers.isdigit() and int(workers) > 1 and s.shared_state == "memory":
        log.warning(f"WEB_CONCURRENCY={workers} with SHARED_STATE=memory: job polls and rate limits are per worker; "
                    "set SHARED_STATE=sqlite (docs/DEPLOYMENT.md)")
    app = FastAPI(title="Uplift: Student Success Assistant API", version="6.0.0", default_response_class=SafeJSONResponse,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json",
                  description="Early warning, historical-case retrieval, multi-agent support planning, copilot and evaluation over OULAD.")
    app.state.handlers = h
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    if s.allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(s.allowed_origins), allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-API-Key", "X-Admin-Key", "X-Request-ID"], allow_credentials=False)

    @app.middleware("http")
    async def envelope(request: Request, call_next):
        request.state.request_id = new_request_id(request)
        t0 = time.time()
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > s.max_request_bytes:
            return _err(request, 413, "request body too large")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Frame-Options"] = "DENY"
        log.info(f"{request.method} {request.url.path} {response.status_code} {round(time.time() - t0, 3)}s rid={request.state.request_id}")
        return response

    # ------------------------------------------------------------ dependencies
    def auth(request: Request):
        if not s.dev_mode:
            if not s.api_key:
                raise _Http(503, "API key not configured on server")
            if not api_key_ok(request.headers.get("X-API-Key"), s.api_key):
                raise _Http(401, "unauthorized")
        if limiter is None:
            return
        ok, retry = limiter.allow(client_ip(request, s.trusted_proxy_count))
        if not ok:
            raise _Http(429, "rate limit exceeded", headers={"Retry-After": str(int(retry) + 1)})

    def admin(request: Request):
        if s.dev_mode and not s.admin_key:
            return
        if not s.admin_key or not api_key_ok(request.headers.get("X-Admin-Key"), s.admin_key):
            raise _Http(403, "admin key required")

    # ------------------------------------------------------------ errors
    @app.exception_handler(_Http)
    async def on_http(request, e: "_Http"):
        r = _err(request, e.status, e.message)
        for k, v in (e.headers or {}).items():
            r.headers[k] = v
        return r

    @app.exception_handler(RequestValidationError)
    async def on_req_validation(request, e: RequestValidationError):
        return _err(request, 400, "validation error", [{"loc": ".".join(str(x) for x in err.get("loc", ()) if x != "body"),
                                                         "msg": err.get("msg"), "type": err.get("type")} for err in e.errors()])

    @app.exception_handler(ValidationError)
    async def on_validation(request, e: ValidationError):
        return _err(request, 400, "validation error", [{"loc": ".".join(str(x) for x in err.get("loc", ())), "msg": err.get("msg"),
                                                         "type": err.get("type")} for err in e.errors()])

    @app.exception_handler(NotFound)
    async def on_nf(request, e):
        return _err(request, 404, str(e))

    @app.exception_handler(AlreadyWithdrawn)
    async def on_wd(request, e):
        return _err(request, 409, "student had already unregistered before this checkpoint; early-warning scoring does not apply", e.info)

    @app.exception_handler(UnknownRecommendation)
    async def on_rec(request, e):
        return _err(request, 404, "unknown recommendation_id")

    @app.exception_handler(FileNotFoundError)
    async def on_missing(request, e):
        log.error(f"missing artifact: {e}")
        return _err(request, 503, "service not ready: build artifacts first (python -m ssa.cli build)")

    @app.exception_handler(ValueError)
    async def on_value(request, e):
        return _err(request, 400, str(e)[:300])

    @app.exception_handler(StarletteHTTPException)
    async def on_starlette(request, e):
        return _err(request, e.status_code, str(e.detail).lower() if e.detail else "error")

    @app.exception_handler(Exception)
    async def on_error(request, e):
        log.error(f"unhandled rid={getattr(request.state, 'request_id', '')}: {traceback.format_exc()}")
        return _err(request, 500, "internal error")

    # ------------------------------------------------------------ routes
    @app.get("/api/health", tags=["system"])
    def health():
        return h.health()

    r = APIRouter(prefix="/api", dependencies=[Depends(auth)])
    CK = Query(None, ge=0.05, le=0.95, description="Course checkpoint (share of planned length)")

    r.get("/ready", tags=["system"])(lambda: h.ready())
    r.get("/config", tags=["system"])(lambda: h.config())

    @r.get("/overview", tags=["cohort"])
    def overview(checkpoint: float | None = CK):
        return h.overview(checkpoint)

    @r.get("/cohort", tags=["cohort"])
    def cohort(code_module: str, code_presentation: str, checkpoint: float | None = CK):
        return h.cohort(code_module, code_presentation, checkpoint)

    @r.get("/students", tags=["cohort"])
    def students(code_module: str, code_presentation: str, checkpoint: float | None = CK, tier: str | None = None,
                 search: str | None = Query(None, max_length=10), sort: str = "withdrawal_risk", descending: bool = True,
                 limit: int = Query(200, ge=1, le=5000), offset: int = Query(0, ge=0)):
        return h.students(code_module, code_presentation, checkpoint, tier, search, sort, descending, limit, offset)

    @r.get("/students/{id_student}", tags=["student"])
    def snapshot(id_student: int, code_module: str, code_presentation: str, checkpoint: float | None = CK):
        return h.snapshot(id_student, code_module, code_presentation, checkpoint)

    @r.post("/predict-risk", tags=["student"])
    def predict(req: S.PredictRequest):
        return h.predict(req)

    @r.post("/students/timeline", tags=["student"])
    def timeline(req: S.TimelineRequest):
        return h.timeline(req)

    @r.post("/trends/student", tags=["student"])
    def trend(req: S.PredictRequest):
        return h.trend(req)

    @r.post("/analyze", tags=["agents"])
    def analyze(req: S.AnalyzeRequestV2):
        return h.analyze(req)

    @r.post("/jobs/analyze", tags=["agents"], status_code=202)
    def analyze_job(req: S.AnalyzeRequestV2):
        return h.analyze_job(req)

    @r.post("/jobs/briefing", tags=["agents"], status_code=202)
    def briefing_job(req: S.BriefingRequest):
        return h.briefing_job(req)

    @r.post("/jobs/copilot", tags=["agents"], status_code=202)
    def copilot_job(req: S.CopilotRequest):
        return h.copilot_job(req)

    @r.get("/jobs/{job_id}", tags=["agents"])
    def job(job_id: str):
        return h.job(job_id)

    @r.post("/cohort/briefing", tags=["agents"])
    def briefing(req: S.BriefingRequest):
        return h.briefing(req)

    @r.post("/copilot", tags=["agents"])
    def copilot(req: S.CopilotRequest):
        return h.copilot(req)

    @r.post("/copilot/stream", tags=["agents"], response_class=StreamingResponse,
            responses={200: {"content": {"text/event-stream": {}}, "description": "Server-sent events (see Handlers.copilot_stream)"}})
    def copilot_stream(req: S.CopilotRequest):
        return StreamingResponse(h.copilot_stream(req), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @r.post("/outreach/variants", tags=["agents"])
    def outreach(req: S.OutreachRequest):
        return h.outreach(req)

    @r.post("/search/parse", tags=["retrieval"])
    def parse(req: S.ParseQueryRequest):
        return h.parse_query(req)

    @r.post("/retrieve-similar", tags=["retrieval"])
    def retrieve(req: S.RetrievalRequestV2):
        return h.retrieve(req)

    @r.post("/prioritize", tags=["cohort"])
    def prioritize(req: S.PrioritizeRequestV2):
        return h.prioritize(req)

    @r.get("/trends/course", tags=["insights"])
    def course_trends(code_module: str | None = None):
        return h.course_trends(code_module)

    r.get("/insights/factors", tags=["insights"])(lambda: h.factors())

    @r.get("/insights/assessments", tags=["insights"])
    def assessments(code_module: str | None = None, code_presentation: str | None = None):
        return h.assessments(code_module, code_presentation)

    r.get("/insights/checkpoints", tags=["insights"])(lambda: h.checkpoints())

    @r.get("/metrics", tags=["evaluation"])
    def metrics(checkpoint: float | None = CK):
        return h.metrics(checkpoint)

    r.get("/evaluations/llm", tags=["evaluation"])(lambda: h.llm_eval())

    @r.get("/evaluations/online", tags=["evaluation"])
    def online(kind: str | None = None, limit: int = Query(100, ge=1, le=500)):
        return h.online_evals(kind, limit)

    @r.post("/evaluations/judge", tags=["evaluation"])
    def judge(req: S.JudgeRequest):
        return h.judge_recommendation(req)

    r.get("/evaluations/human-review", tags=["evaluation"])(lambda: h.human_review())

    @r.post("/evaluations/human-labels", tags=["evaluation"])
    def human_label(req: S.HumanLabelRequest):
        return h.human_label(req)

    @r.get("/llm/usage", tags=["evaluation"])
    def usage(since_hours: float | None = Query(None, gt=0, le=24 * 365)):
        return h.llm_usage(since_hours)

    @r.post("/feedback", tags=["feedback"], status_code=201)
    def feedback(req: S.FeedbackRequest):
        return h.feedback(req)

    r.get("/feedback/summary", tags=["feedback"])(lambda: h.feedback_summary())

    @r.get("/feedback", tags=["feedback"])
    def feedback_list(limit: int = Query(50, ge=1, le=500)):
        return h.feedback_list(limit)

    r.get("/feedback/themes", tags=["feedback"])(lambda: h.feedback_themes())
    r.get("/improvement-plan", tags=["feedback"])(lambda: h.improvement_plan())

    @r.post("/feedback-loop/run", tags=["feedback"], dependencies=[Depends(admin)])
    def loop(req: S.FeedbackLoopRequest):
        return h.run_feedback_loop(req)

    app.include_router(r)

    dist = PROJECT_ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/", _console_files(dist), name="console")
    return app


def _console_files(dist):
    """The built console. Client-side routes (e.g. /students/BBB/2014J/123) get index.html; missing files and /api stay 404."""
    from fastapi.staticfiles import StaticFiles

    class SPAFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            try:
                resp = await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code != 404 or not _is_client_route(path):
                    raise
                return await super().get_response("index.html", scope)
            if resp.status_code == 404 and _is_client_route(path):
                return await super().get_response("index.html", scope)
            return resp

    return SPAFiles(directory=str(dist), html=True)


def _is_client_route(path: str) -> bool:
    last = path.rstrip("/").rsplit("/", 1)[-1]
    return not path.startswith("api") and "." not in last


class _Http(Exception):
    def __init__(self, status: int, message: str, headers: dict | None = None):
        self.status, self.message, self.headers = status, message, headers


def app_factory() -> FastAPI:  # uvicorn --factory ssa.api.app:app_factory
    return create_app()


def main():  # pragma: no cover
    import uvicorn
    s = get_settings()
    uvicorn.run(create_app(s), host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()

