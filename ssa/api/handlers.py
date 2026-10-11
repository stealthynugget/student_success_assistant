"""Framework-independent API handlers.

Every endpoint's logic lives here as a plain method taking validated pydantic models / primitives and returning
JSON-serialisable dicts. ``ssa.api.app`` (FastAPI) is a thin HTTP layer over this class, and the test-suite exercises
the handlers directly as well as through FastAPI's TestClient.
"""
from __future__ import annotations

import json
import random
import threading

from ssa.agents.copilot import Copilot
from ssa.agents.genai import briefing_evidence, cohort_briefing, note_themes, outreach_variant, understand_query
from ssa.api import schemas as S
from ssa.api.jobs import JobManager
from ssa.api.shared import job_store, jsonable, kv_store
from ssa.appdb import AppDB
from ssa.config import Settings, get_settings
from ssa.evaluation.judge import LLMJudge, analysis_evidence, judge_text
from ssa.feedback.loop import run_loop
from ssa.feedback.store import FeedbackStore
from ssa.service import StudentSuccessService
from ssa.utils import get_logger, read_json

log = get_logger("ssa.api")


_jsonable = jsonable


class Handlers:
    def __init__(self, settings: Settings | None = None, service: StudentSuccessService | None = None, llm=None):
        self.settings = (settings or get_settings()).at(None)
        self.svc = service or StudentSuccessService(self.settings, llm=llm)
        self.store = FeedbackStore(self.settings)
        self.db = AppDB(self.settings)
        self.jobs = JobManager(job_store(self.settings))
        self.recent = kv_store(self.settings, "recent_analyses", 500)  # recommendation_id -> analysis (for on-demand judging)

    # ---------------------------------------------------------------- helpers
    def _rp(self, inp, checkpoint=None, allow_withdrawn=False):
        if inp.student is not None:
            return self.svc.resolve(ref=inp.student.model_dump(), features=getattr(inp, "what_if", None),
                                    allow_withdrawn=allow_withdrawn, checkpoint=checkpoint)
        return self.svc.resolve(features=inp.profile.model_dump(exclude_none=True), checkpoint=checkpoint)

    def _llm(self, use_llm: bool):
        llm = self.svc.llm() if use_llm else None
        return llm if (llm is not None and llm.available) else None

    def _maybe_online_eval(self, kind: str, ref_id: str, output, evidence):
        rate = self.settings.online_eval_sample_rate
        if rate <= 0 or random.random() >= rate:
            return
        judge = LLMJudge(self.settings)
        if not judge.llm.available:
            return

        def work(progress):
            if kind == "analysis":
                j = judge.evaluate(output, evidence)
            else:
                j = judge_text(judge.llm, kind, output, evidence)
            self.db.add_eval(kind, ref_id, j.get("verdict"), j.get("overall"), j.get("scores"), j.get("judge_model"),
                             {"checks": j.get("deterministic_checks"), "rationale": j.get("rationale"), "sampled": True})
            return j
        self.jobs.submit(f"online_eval:{kind}", work)

    # ---------------------------------------------------------------- system
    def health(self):
        return {"status": "ok"}

    def ready(self):
        return self.svc.readiness()

    def config(self):
        return self.svc.config()

    # ---------------------------------------------------------------- cohort
    def overview(self, checkpoint=None):
        return self.svc.overview(checkpoint)

    def cohort(self, module, presentation, checkpoint=None):
        ref = S.StudentRef.model_validate({"id_student": 1, "code_module": module, "code_presentation": presentation})
        return self.svc.cohort(ref.code_module, ref.code_presentation, checkpoint)

    def students(self, module, presentation, checkpoint=None, tier=None, search=None, sort="withdrawal_risk", descending=True,
                 limit=200, offset=0):
        ref = S.StudentRef.model_validate({"id_student": 1, "code_module": module, "code_presentation": presentation})
        if tier not in (None, "", "High", "Medium", "Low"):
            raise ValueError("tier must be High, Medium or Low")
        allowed_sort = {"withdrawal_risk", "p_poor_outcome", "id_student", "weighted_score", "score_pct", "missed_rate", "clicks_pct",
                        "days_since_last_activity"}
        if sort not in allowed_sort:
            raise ValueError(f"sort must be one of {sorted(allowed_sort)}")
        return self.svc.list_students(ref.code_module, ref.code_presentation, checkpoint, tier or None, search, sort, descending,
                                      max(1, min(int(limit), 5000)), max(0, int(offset)))

    def briefing(self, req: S.BriefingRequest, progress=None):
        c = self.svc.cohort(req.code_module, req.code_presentation, req.checkpoint)
        if progress:
            progress("cohort", {"source": "computed"})
        rk = self.svc.prioritize(code_module=req.code_module, code_presentation=req.code_presentation, capacity=req.capacity,
                                 limit=12, checkpoint=req.checkpoint)
        if progress:
            progress("ranking", {"source": "computed"})
        b = cohort_briefing(self._llm(True), c, rk, [])
        if progress:
            progress("briefing", b)
        self._maybe_online_eval("cohort_briefing", f"{req.code_module}-{req.code_presentation}", b, briefing_evidence(c, rk))
        return {"briefing": b, "cohort": {k: c[k] for k in ("code_module", "code_presentation", "snapshot_week", "tiers", "in_scope")}}

    def prioritize(self, req: S.PrioritizeRequestV2):
        return self.svc.prioritize(code_module=req.code_module, code_presentation=req.code_presentation,
                                   refs=[x.model_dump() for x in req.students] if req.students else None,
                                   profiles=[x.model_dump(exclude_none=True) for x in req.profiles] if req.profiles else None,
                                   capacity=req.capacity, limit=req.limit, checkpoint=req.checkpoint)

    # ---------------------------------------------------------------- student
    def snapshot(self, id_student, module, presentation, checkpoint=None):
        ref = S.StudentRef.model_validate({"id_student": id_student, "code_module": module, "code_presentation": presentation})
        return self.svc.snapshot(ref.model_dump(), checkpoint)

    def predict(self, req: S.PredictRequest):
        return self.svc.predict(self._rp(req, req.checkpoint))

    def timeline(self, req: S.TimelineRequest):
        return self.svc.timeline(req.student.model_dump())

    def trend(self, req: S.PredictRequest):
        return self.svc.student_trend(self._rp(req, req.checkpoint, allow_withdrawn=True))

    def analyze(self, req: S.AnalyzeRequestV2, progress=None):
        rp = self._rp(req, req.checkpoint)
        res = self.svc.analyze(rp, use_llm=req.use_llm, judge=req.judge, on_step=progress, synthesize=req.synthesize)
        self.store.register(res, model_version=str(self.svc.bundle(rp.checkpoint).risk.art.metadata.get("engine")))
        rid = res["workflow"]["recommendation_id"]
        self.recent.put(rid, res)
        self._maybe_online_eval("analysis", rid, res["workflow"], analysis_evidence(res))
        return res

    def analyze_job(self, req: S.AnalyzeRequestV2):
        self._rp(req, req.checkpoint)  # validate synchronously so errors surface as HTTP errors, not job failures
        return self.jobs.submit("analysis", lambda progress: self.analyze(req, progress))

    def briefing_job(self, req: S.BriefingRequest):
        self.svc.cohort(req.code_module, req.code_presentation, req.checkpoint)
        return self.jobs.submit("briefing", lambda progress: self.briefing(req, progress))

    def copilot(self, req: S.CopilotRequest, progress=None, on_event=None):
        cp = Copilot(self.svc, self._llm(req.use_llm), req.checkpoint)
        res = cp.ask(req.question, [t.model_dump() for t in req.history], on_event=on_event)
        if progress:
            progress("answer", {"source": res["source"]})
        if res["source"] == "llm":
            self._maybe_online_eval("copilot", "copilot", {"question": req.question, "answer": res["answer"]},
                                    {"tool_results": [t["result"] for t in res["tool_calls"]]})
        return res

    def copilot_stream(self, req: S.CopilotRequest):
        """Server-sent events for one copilot answer: ``tool_call`` / ``tool_result`` as tools run, ``text`` deltas (unverified
        until ``done``), ``text_reset``, ``revision``, then ``done`` with the full checked result (or ``error``)."""
        import queue
        q: queue.Queue = queue.Queue()

        def work():
            try:
                res = self.copilot(req, on_event=lambda kind, data: q.put((kind, data)))
                q.put(("done", res))
            except Exception as exc:  # generic message to the client; details in the log
                log.error(f"copilot stream failed: {type(exc).__name__}: {exc}")
                q.put(("error", {"error": "the copilot could not answer; try again"}))
            q.put(None)

        threading.Thread(target=work, daemon=True, name="copilot-stream").start()
        while (item := q.get()) is not None:
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(_jsonable(data), ensure_ascii=False, default=str)}\n\n"

    def copilot_job(self, req: S.CopilotRequest):
        return self.jobs.submit("copilot", lambda progress: self.copilot(req, progress))

    def job(self, jid: str):
        j = self.jobs.get(jid)
        if j is None:
            from ssa.service import NotFound
            raise NotFound("unknown or expired job")
        return j

    def outreach(self, req: S.OutreachRequest):
        return outreach_variant(self._llm(True), req.message, req.channel, req.tone, req.language)

    # ---------------------------------------------------------------- retrieval
    def parse_query(self, req: S.ParseQueryRequest):
        return understand_query(self._llm(True), req.query)

    def retrieve(self, req: S.RetrievalRequestV2):
        rp = None
        if req.student is not None:
            rp = self.svc.resolve(ref=req.student.model_dump(), allow_withdrawn=True, checkpoint=req.checkpoint)
        elif req.profile is not None:
            rp = self.svc.resolve(features=req.profile.model_dump(exclude_none=True), checkpoint=req.checkpoint)
        filters = req.filters.model_dump(exclude_none=True) if req.filters else None
        interpretation = None
        descriptors = list(req.descriptors or [])
        if req.query and req.understand:
            interpretation = understand_query(self._llm(True), req.query)
            merged = dict(interpretation["filters"])
            merged.update(filters or {})  # explicit filters win over interpreted ones
            filters = merged or None
            descriptors += interpretation["descriptors"]
        res = self.svc.similar(rp, text=req.query, filters=filters, top_k=req.top_k, mode=req.mode,
                               historical_only=req.historical_only, checkpoint=req.checkpoint, descriptors=descriptors or None,
                               parser=interpretation["source"] if interpretation else None)
        res["interpretation"] = interpretation
        res["applied_filters"] = filters
        return res

    # ---------------------------------------------------------------- insights & evaluation
    def factors(self):
        d = self.svc.factor_insights()
        if not d:
            raise FileNotFoundError("factor_analysis.json")
        return d

    def assessments(self, module=None, presentation=None):
        return self.svc.assessment_insights(module, presentation)

    def course_trends(self, module=None):
        return self.svc.course_trends(module)

    def checkpoints(self):
        return self.svc.checkpoint_metrics()

    def metrics(self, checkpoint=None):
        return self.svc.metrics(checkpoint)

    def llm_eval(self):
        return read_json(self.settings.global_metrics_dir / "llm_evaluation.json", {"status": "not run; python -m ssa.cli llm-eval"})

    def online_evals(self, kind=None, limit=100):
        return {"summary": self.db.eval_summary(), "items": self.db.list_evals(kind, max(1, min(int(limit), 500)))}

    def judge_recommendation(self, req: S.JudgeRequest):
        res = self.recent.get(req.recommendation_id)
        if res is None:
            from ssa.feedback.store import UnknownRecommendation
            raise UnknownRecommendation(req.recommendation_id)
        judge = LLMJudge(self.settings)
        j = judge.evaluate(res["workflow"], analysis_evidence(res))
        if j.get("scores"):
            self.db.add_eval("analysis", req.recommendation_id, j["verdict"], j.get("overall"), j["scores"], j.get("judge_model"),
                             {"checks": j.get("deterministic_checks"), "rationale": j.get("rationale"), "sampled": False})
        return j

    def llm_usage(self, since_hours: float | None = None):
        return self.db.usage_summary(since_s=since_hours * 3600 if since_hours else None)

    # ---------------------------------------------------------------- feedback
    def feedback(self, req: S.FeedbackRequest):
        fb = req.model_dump()
        return {"status": "recorded", "feedback_id": self.store.add_feedback(fb)}

    def feedback_summary(self):
        return self.store.summary()

    def feedback_list(self, limit=50):
        rows = sorted(self.store.joined(), key=lambda r: -r["created_at"])[: max(1, min(int(limit), 500))]
        for r in rows:
            r.pop("similar_cases", None)
        return {"items": rows}

    def feedback_themes(self):
        rows = [{"id": r["id"], "notes": r.get("notes") or "", "rating": r.get("rating")} for r in self.store.joined()]
        return note_themes(self._llm(True), rows)

    def improvement_plan(self):
        return read_json(self.settings.improvement_plan_path, {"status": "not generated yet"})

    def run_feedback_loop(self, req: S.FeedbackLoopRequest):
        plan = run_loop(self.settings, apply=req.apply, approved_by=req.approved_by)
        if req.apply and plan["applied"]:
            # applied artifacts change scores / thresholds: rebuild cached scores lazily
            for b in self.svc._bundles.values():
                b.scores = None
        return plan
