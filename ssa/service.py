"""Application service: one place that wires data, models, retrieval, agents and generative features.

The API calls this layer; nothing here is HTTP-specific.

* **Checkpoints.** Models, features and indexes exist for several points in the course (e.g. 15/25/40/60 % of the
  planned length). Every call takes an optional ``checkpoint``; the primary checkpoint is the default.
* **Profile resolution.** By reference (real snapshot from the feature store - default), with what-if overrides,
  or from supplied features (imputed from module medians, imputed fields and out-of-distribution values reported).
* **Complete records.** Historical precedents carry what happened *after* the checkpoint: activity trajectory,
  withdrawal week, final coursework and exam scores, final result.
"""
from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ssa.agents.llm import LLMClient
from ssa.agents.workflow import SupportWorkflow
from ssa.config import Settings, course_week, get_settings
from ssa.data import vocab
from ssa.data.pipeline import load_course_level, load_features, load_weekly
from ssa.models.features import FRIENDLY, NUMERIC, SENSITIVE
from ssa.models.outcome import OutcomeModel
from ssa.models.prioritization import Prioritizer
from ssa.models.risk import WithdrawalRiskModel, tier_of
from ssa.retrieval.hybrid import HistoricalCaseIndex
from ssa.utils import read_json

KEY_SIGNALS = ["weighted_score", "missed_rate", "total_clicks", "days_since_last_activity", "engagement_momentum",
               "assessments_due", "active_days"]
HIDDEN = set(SENSITIVE) | {"_key", "final_result", "withdrawn", "poor_outcome", "date_unregistration", "withdrawn_before_start"}


class NotFound(LookupError):
    pass


class AlreadyWithdrawn(Exception):
    def __init__(self, info: dict):
        super().__init__("student had already unregistered before the checkpoint")
        self.info = info


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _same_value(new, stored) -> bool:
    """True when a what-if value equals the stored snapshot value (numerically, within float noise)."""
    try:
        return stored is not None and abs(float(new) - float(stored)) <= 1e-6 * max(1.0, abs(float(stored)))
    except (TypeError, ValueError):
        return new == stored


@dataclass
class ResolvedProfile:
    profile: dict
    source: str
    imputed_fields: list[str]
    out_of_distribution: list[dict]
    completeness: float
    checkpoint: float

    @property
    def confidence(self) -> str:
        if self.source == "feature_store":
            return "normal"
        return "low" if self.completeness < 0.5 or len(self.out_of_distribution) > 2 else "normal"

    def quality(self) -> dict:
        return {"source": self.source, "imputed_fields": self.imputed_fields, "out_of_distribution": self.out_of_distribution,
                "key_signal_completeness": round(self.completeness, 2), "confidence": self.confidence,
                "checkpoint": self.checkpoint}


class Bundle:
    """Everything for one checkpoint, loaded lazily and shared across requests."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.risk = WithdrawalRiskModel(settings)
        self.outcome = OutcomeModel(settings)
        self.prioritizer = Prioritizer(settings)
        self.index = HistoricalCaseIndex(settings)
        self.features: pd.DataFrame | None = None
        self.scores: pd.DataFrame | None = None
        self._weekly = None
        self._lock = threading.Lock()

    def warm(self):
        with self._lock:
            if self.risk.art is None:
                self.risk.load()
            if self.outcome.engine is None:
                self.outcome.load()
            if self.prioritizer.weights is None:
                self.prioritizer.load()
            if self.index.meta is None and self.index.ready:
                self.index.load()
            if self.features is None:
                f = load_features(self.settings)
                f["_key"] = f["code_module"].astype(str) + "|" + f["code_presentation"].astype(str) + "|" + f["id_student"].astype(str)
                self.features = f.set_index("_key", drop=False)
            if self.scores is None:
                scope = self.features[self.features["in_scope"] == 1]
                p = self.risk.predict_proba(scope)
                pp = self.outcome.p_poor(scope)
                thr = self.risk.thresholds()
                self.scores = pd.DataFrame({"withdrawal_risk": p, "p_poor_outcome": pp,
                                            "risk_tier": [tier_of(float(x), thr) for x in p]}, index=scope.index)
        return self

    def weekly(self):
        if self._weekly is None:
            keys, weekly, _ = load_weekly(self.settings)
            self._weekly = (pd.Series(np.arange(len(keys)), index=keys), weekly)
        return self._weekly


class StudentSuccessService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, settings: Settings | None = None, llm=None):
        self.settings = (settings or get_settings()).at(None)
        self._bundles: dict[str, Bundle] = {}
        self._llm = llm
        self._course = None
        self._block = threading.Lock()

    @classmethod
    def instance(cls) -> "StudentSuccessService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------ plumbing
    def bundle(self, checkpoint: float | None = None) -> Bundle:
        sc = self.settings.at(checkpoint)
        with self._block:
            if sc.tag not in self._bundles:
                self._bundles[sc.tag] = Bundle(sc)
        return self._bundles[sc.tag].warm()

    def warm(self):
        self.bundle(None)
        return self

    # Back-compat helpers used by evaluation code
    @property
    def risk(self):
        return self.bundle().risk

    @property
    def outcome(self):
        return self.bundle().outcome

    @property
    def index(self):
        return self.bundle().index

    @property
    def features(self) -> pd.DataFrame:
        return self.bundle().features

    def llm(self):
        if self._llm is None:
            self._llm = LLMClient(self.settings)
        return self._llm

    def course(self):
        if self._course is None:
            det, keys, weekly = load_course_level(self.settings)
            self._course = (det.set_index("enrollment_key"), pd.Series(np.arange(len(keys)), index=keys), weekly)
        return self._course

    def checkpoint_list(self) -> list[dict]:
        return [{"checkpoint": c, "tag": self.settings.at(c).tag, "approx_week": course_week(c),
                 "primary": abs(c - self.settings.primary_checkpoint) < 1e-9,
                 "available": (self.settings.at(c).model_dir / "withdrawal_model.joblib").exists()} for c in self.settings.checkpoints]

    def readiness(self) -> dict:
        s = self.settings
        checks = {
            "features": s.features_path.exists(), "course_level_records": s.outcome_details_path.exists(),
            "withdrawal_model": (s.model_dir / "withdrawal_model.joblib").exists(),
            "outcome_model": (s.model_dir / "outcome_model.joblib").exists(),
            "prioritization": (s.model_dir / "prioritization.json").exists(),
            "retrieval_index": (s.index_dir / "manifest.json").exists(),
        }
        return {"ready": all(checks.values()), "checks": checks, "checkpoints": self.checkpoint_list(),
                "index": read_json(s.index_dir / "manifest.json", {}), "llm_configured": s.llm_configured,
                "llm_provider": s.llm_provider if s.llm_configured else None, "llm_model": s.llm_model if s.llm_configured else None,
                "fast_model": s.fast_model if s.llm_configured else None, "judge_model": s.judge_model if s.llm_configured else None}

    def config(self) -> dict:
        b = self.bundle()
        f = b.features
        pres = {m: sorted(g.unique().tolist(), key=vocab.presentation_sort_key) for m, g in f.groupby("code_module")["code_presentation"]}
        return {
            "checkpoints": self.checkpoint_list(), "modules": list(vocab.MODULES), "presentations_by_module": pres,
            "latest_presentation": sorted(f["code_presentation"].unique(), key=vocab.presentation_sort_key)[-1],
            "vocab": {"regions": list(vocab.REGIONS), "education": list(vocab.EDUCATION), "age_bands": list(vocab.AGE_BANDS),
                      "outcomes": list(vocab.OUTCOMES)},
            "features": {k: FRIENDLY.get(k, k) for k in NUMERIC},
            "llm": {"configured": self.settings.llm_configured, "enabled": self.settings.llm_enabled, "provider": self.settings.llm_provider,
                    "model": self.settings.llm_model, "fast_model": self.settings.fast_model, "judge_model": self.settings.judge_model,
                    "online_eval_sample_rate": self.settings.online_eval_sample_rate},
            "dense_retrieval": b.index.manifest.get("dense_channel") if b.index.ready else None,
            # Which presentations the models learned from, were tuned on, and never saw (scores there are out-of-sample).
            "split": {k: v for k, v in read_json(b.settings.metrics_dir / "withdrawal_model.json", {}).get("split", {}).items()
                      if k in ("method", "train", "validation", "test")},
        }

    # -------------------------------------------------------------- lookup
    def get_row(self, id_student: int, code_module: str, code_presentation: str, checkpoint=None) -> pd.Series:
        f = self.bundle(checkpoint).features
        key = f"{code_module}|{code_presentation}|{id_student}"
        if key not in f.index:
            raise NotFound(f"no enrolment for student {id_student} in {code_module} {code_presentation}")
        row = f.loc[key]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    def snapshot(self, ref: dict, checkpoint=None) -> dict:
        row = self.get_row(int(ref["id_student"]), ref["code_module"], ref["code_presentation"], checkpoint)
        data = {k: _clean(v) for k, v in row.to_dict().items() if k not in HIDDEN}
        data["already_withdrawn"] = int(row["in_scope"]) == 0
        return {"snapshot": data, "labels": {k: FRIENDLY.get(k, k) for k in data if k in FRIENDLY},
                "note": "Outcome labels and sensitive attributes are not exposed."}

    def resolve(self, ref: dict | None = None, features: dict | None = None, allow_withdrawn: bool = False,
                checkpoint=None) -> ResolvedProfile:
        b = self.bundle(checkpoint)
        c = b.settings.cutoff_fraction
        if ref:
            row = self.get_row(int(ref["id_student"]), ref["code_module"], ref["code_presentation"], checkpoint)
            if int(row["in_scope"]) == 0 and not allow_withdrawn:
                raise AlreadyWithdrawn({"id_student": int(row["id_student"]), "date_unregistration": _clean(row["date_unregistration"]),
                                        "snapshot_day": int(row["snapshot_day"]), "checkpoint": c})
            prof = {k: _clean(v) for k, v in row.to_dict().items() if k != "_key"}
            if features:
                given = {k: v for k, v in features.items() if v is not None}
                changed = {k for k, v in given.items() if not _same_value(v, prof.get(k))}
                prof.update(given)
                ood = [o for o in self._ood(prof, b) if o["field"] in changed]  # only flag values the user actually changed
                return ResolvedProfile(prof, "feature_store+overrides", [], ood, 1.0, c)
            return ResolvedProfile(prof, "feature_store", [], [], 1.0, c)
        if not features:
            raise ValueError("provide a student reference or a feature profile")
        return self._impute(dict(features), b)

    def _impute(self, prof: dict, b: Bundle) -> ResolvedProfile:
        med = b.risk.art.module_medians
        base = med.get(prof.get("code_module"), med["__global__"])
        imputed = []
        supplied = [k for k in KEY_SIGNALS if prof.get(k) is not None]
        for col in NUMERIC:
            if prof.get(col) is None:
                val = base.get(col, med["__global__"].get(col))
                prof[col] = None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
                imputed.append(col)
        if prof.get("weighted_score") is None and prof.get("mean_score") is not None:
            prof["weighted_score"] = prof["mean_score"]
        prof.setdefault("in_scope", 1)
        prof.setdefault("id_student", 0)
        return ResolvedProfile(prof, "supplied_profile", imputed, self._ood(prof, b), len(supplied) / len(KEY_SIGNALS),
                               b.settings.cutoff_fraction)

    @staticmethod
    def _ood(prof: dict, b: Bundle) -> list[dict]:
        out = []
        for col, (lo, hi) in b.risk.art.feature_bounds.items():
            v = prof.get(col)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v < lo or v > hi:
                out.append({"field": col, "label": FRIENDLY.get(col, col), "value": v, "training_range_p0_5_p99_5": [lo, hi]})
        return out

    @staticmethod
    def _ident(p: dict) -> dict:
        return {"id_student": int(p.get("id_student") or 0), "code_module": p.get("code_module"),
                "code_presentation": p.get("code_presentation")}

    # ------------------------------------------------------- cohort views
    def overview(self, checkpoint=None) -> dict:
        b = self.bundle(checkpoint)
        f = b.features
        sc = b.scores
        rows = []
        for (m, p), g in f.groupby(["code_module", "code_presentation"]):
            s = sc.reindex(g.index[g["in_scope"] == 1])
            rows.append({"code_module": m, "code_presentation": p, "enrolled": int(len(g)), "in_scope": int(len(s)),
                         "already_withdrawn": int((g["in_scope"] == 0).sum()),
                         "high": int((s["risk_tier"] == "High").sum()), "medium": int((s["risk_tier"] == "Medium").sum()),
                         "low": int((s["risk_tier"] == "Low").sum()),
                         "expected_withdrawals": round(float(s["withdrawal_risk"].sum()), 1),
                         "expected_fail_or_withdraw": round(float(s["p_poor_outcome"].sum()), 1),
                         "snapshot_day": int(g["snapshot_day"].iloc[0])})
        rows.sort(key=lambda r: (r["code_module"], vocab.presentation_sort_key(r["code_presentation"])))
        tot = {k: sum(r[k] for r in rows) for k in ("enrolled", "in_scope", "already_withdrawn", "high", "medium", "low")}
        return {"checkpoint": b.settings.cutoff_fraction, "presentations": rows, "totals": tot, "thresholds": b.risk.thresholds()}

    def cohort(self, code_module: str, code_presentation: str, checkpoint=None) -> dict:
        b = self.bundle(checkpoint)
        f = b.features
        g = f[(f["code_module"] == code_module) & (f["code_presentation"] == code_presentation)]
        if g.empty:
            raise NotFound(f"no cohort {code_module} {code_presentation}")
        scope = g[g["in_scope"] == 1]
        s = b.scores.reindex(scope.index)
        hist, edges = np.histogram(s["withdrawal_risk"], bins=20, range=(0, 1))
        high = scope[s["risk_tier"] == "High"]
        driver_counts: dict[str, int] = {}
        if len(high):
            for drivers in b.risk.explain(high.head(400), top_n=3):
                for d in drivers:
                    if d["direction"] == "increases_risk":
                        driver_counts[d["label"]] = driver_counts.get(d["label"], 0) + 1
        row_of, weekly = b.weekly()
        M = weekly[row_of.reindex(scope["enrollment_key"].astype(np.int64)).dropna().astype(int).to_numpy()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # weeks beyond the checkpoint are all-NaN by design
            med = np.nanmedian(M, 0) if len(M) else np.array([])
        assessments = [a for a in read_json(self.settings.analysis_dir / "assessment_analytics.json", {}).get("assessments", [])
                       if a["code_module"] == code_module and a["code_presentation"] == code_presentation]
        snap_day = int(g["snapshot_day"].iloc[0])
        return {
            "code_module": code_module, "code_presentation": code_presentation, "checkpoint": b.settings.cutoff_fraction,
            "snapshot_day": snap_day, "snapshot_week": snap_day // 7,
            "course_length_days": int(round(snap_day / b.settings.cutoff_fraction)),
            "enrolled": int(len(g)), "in_scope": int(len(scope)), "already_withdrawn": int((g["in_scope"] == 0).sum()),
            "tiers": {k: int(v) for k, v in s["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).items()},
            "expected_withdrawals": round(float(s["withdrawal_risk"].sum()), 1),
            "expected_fail_or_withdraw": round(float(s["p_poor_outcome"].sum()), 1),
            "risk_histogram": {"counts": hist.tolist(), "edges": edges.round(3).tolist()},
            "thresholds": b.risk.thresholds(),
            "common_high_tier_drivers": sorted(([k, v] for k, v in driver_counts.items()), key=lambda x: -x[1])[:8],
            "cohort_median_weekly_clicks": [_clean(x) for x in med],
            "medians": {k: _clean(scope[k].median()) for k in ["weighted_score", "missed_rate", "total_clicks", "days_since_last_activity"]},
            "assessments": [{**a, "upcoming": a["deadline_day"] is not None and a["deadline_day"] > snap_day} for a in assessments],
        }

    def list_students(self, code_module: str, code_presentation: str, checkpoint=None, tier: str | None = None, search: str | None = None,
                      sort: str = "withdrawal_risk", descending: bool = True, limit: int = 200, offset: int = 0,
                      include_withdrawn: bool = False) -> dict:
        b = self.bundle(checkpoint)
        f = b.features
        sel = f[(f["code_module"] == code_module) & (f["code_presentation"] == code_presentation)]
        if not include_withdrawn:
            sel = sel[sel["in_scope"] == 1]
        cols = ["id_student", "weighted_score", "score_pct", "missed_rate", "clicks_pct", "days_since_last_activity",
                "engagement_momentum", "num_of_prev_attempts", "studied_credits", "in_scope"]
        out = sel[cols].join(b.scores, how="left")
        if tier:
            out = out[out["risk_tier"] == tier]
        if search:
            out = out[out["id_student"].astype(str).str.contains(str(search).strip(), regex=False)]
        if sort in out.columns:
            out = out.sort_values(sort, ascending=not descending, na_position="last")
        page = out.iloc[offset: offset + limit]
        return {"total": int(len(out)), "offset": offset, "limit": limit, "checkpoint": b.settings.cutoff_fraction,
                "students": [{k: _clean(v) for k, v in r.items()} for r in page.to_dict(orient="records")]}

    # --------------------------------------------------------- per student
    def predict(self, rp: ResolvedProfile, explain: bool = True) -> dict:
        b = self.bundle(rp.checkpoint)
        df = pd.DataFrame([rp.profile])
        return {"student": self._ident(rp.profile), "checkpoint": rp.checkpoint, "withdrawal_risk": b.risk.score(df, explain=explain)[0],
                "outcome_prediction": b.outcome.predict(df)[0], "input_quality": rp.quality()}

    def timeline(self, ref: dict) -> dict:
        """Risk for the same enrolment at every checkpoint where the student was still enrolled."""
        pts = []
        for cp in self.checkpoint_list():
            if not cp["available"]:
                continue
            b = self.bundle(cp["checkpoint"])
            try:
                row = self.get_row(int(ref["id_student"]), ref["code_module"], ref["code_presentation"], cp["checkpoint"])
            except NotFound:
                continue
            entry = {"checkpoint": cp["checkpoint"], "snapshot_week": int(row["snapshot_day"]) // 7, "in_scope": int(row["in_scope"]) == 1}
            if entry["in_scope"]:
                sc = b.scores.loc[row["_key"]]
                entry.update({"withdrawal_risk": round(float(sc["withdrawal_risk"]), 4), "risk_tier": sc["risk_tier"],
                              "p_poor_outcome": round(float(sc["p_poor_outcome"]), 4),
                              "weighted_score": _clean(row["weighted_score"]), "clicks_pct": _clean(row["clicks_pct"]),
                              "missed_rate": _clean(row["missed_rate"])})
            pts.append(entry)
        return {"student": {k: ref[k] for k in ("id_student", "code_module", "code_presentation")}, "points": pts,
                "note": "Each point uses only information available at that checkpoint."}

    def student_trend(self, rp: ResolvedProfile) -> dict:
        b = self.bundle(rp.checkpoint)
        p = rp.profile
        row_of, weekly = b.weekly()
        series = cohort_median = None
        key = p.get("enrollment_key")
        if key is not None and int(key) in row_of.index:
            series = [_clean(x) for x in weekly[row_of[int(key)]]]
            f = b.features
            peers = f[(f["presentation_id"] == p["presentation_id"]) & (f["in_scope"] == 1)]["enrollment_key"]
            M = weekly[row_of.reindex(peers.astype(np.int64)).dropna().astype(int).to_numpy()]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cohort_median = [_clean(x) for x in np.nanmedian(M, 0)]
        deadlines = [{"week": a["deadline_week"], "type": a["assessment_type"], "weight": a["weight"], "id": a["id_assessment"]}
                     for a in read_json(self.settings.analysis_dir / "assessment_analytics.json", {}).get("assessments", [])
                     if a["code_module"] == p.get("code_module") and a["code_presentation"] == p.get("code_presentation")
                     and a["deadline_week"] is not None]
        signals = []
        for cond, name in [((p.get("engagement_momentum") or 0) < -0.3, "recent_engagement_drop"),
                           ((p.get("engagement_slope") or 0) < 0, "declining_weekly_trend"),
                           ((p.get("inactive_weeks_last3") or 0) >= 2, "recently_inactive"),
                           ((p.get("score_slope") or 0) < -3, "falling_scores"), ((p.get("missed_rate") or 0) > 0, "missed_assessments")]:
            if cond:
                signals.append(name)
        return {"student": self._ident(p), "checkpoint": rp.checkpoint, "weekly_clicks": series, "cohort_median_weekly_clicks": cohort_median,
                "snapshot_week": int(p.get("snapshot_day") or 0) // 7, "assessment_deadlines": deadlines,
                "engagement": {k: _clean(p.get(k)) for k in ["engagement_slope", "engagement_slope_rel", "engagement_momentum",
                                                             "engagement_drop_ratio", "longest_inactive_streak", "inactive_weeks_last3",
                                                             "clicks_pct", "planned_coverage"]},
                "academic": {k: _clean(p.get(k)) for k in ["weighted_score", "score_pct", "score_slope", "min_score", "max_score", "last_score",
                                                           "tma_mean_score", "cma_mean_score", "missed_rate", "late_rate"]},
                "signals": signals, "note": "Observed behaviour up to the checkpoint; descriptive, not causal."}

    # ------------------------------------------------------- precedents
    def precedent_stories(self, cases: list[dict], checkpoint: float) -> list[dict]:
        if not self.settings.outcome_details_path.exists():
            return cases
        det, row_of, weekly = self.course()
        out = []
        for c in cases:
            k = int(c["case_id"])
            if k not in det.index:
                out.append(c)
                continue
            d = det.loc[k]
            sw = int(round(float(d["length"]) * checkpoint)) // 7
            w = weekly[row_of[k]] if k in row_of.index else None
            before = after = None
            if w is not None:
                before = float(np.nanmean(w[: sw + 1]))
                end_w = int(d["withdrawal_week"]) if pd.notna(d["withdrawal_week"]) else len(w)
                seg = w[sw + 1: max(sw + 1, end_w)]
                after = float(np.nanmean(seg)) if len(seg) and not np.all(np.isnan(seg)) else 0.0
            change = (after - before) / before if before and before > 0 and after is not None else None
            wd = _clean(d["date_unregistration"])
            story = []
            if d["final_result"] == "Withdrawn" and wd is not None:
                story.append(f"withdrew in week {int(wd // 7)} ({max(0, int(wd // 7) - sw)} weeks after this point)")
            else:
                story.append(f"finished with {str(d['final_result']).lower()}")
            if change is not None:
                story.append(f"weekly activity {'fell' if change < 0 else 'rose'} {abs(round(100 * change))}% after week {sw}")
            if pd.notna(d["final_coursework_score"]):
                story.append(f"final coursework average {round(float(d['final_coursework_score']))}")
            if pd.notna(d["exam_score"]):
                story.append(f"exam score {round(float(d['exam_score']))}")
            out.append({**c, "story": {"snapshot_week": sw, "final_result": d["final_result"],
                                       "withdrawal_week": None if wd is None else int(wd // 7),
                                       "weeks_after_checkpoint_to_withdrawal": None if wd is None else max(0, int(wd // 7) - sw),
                                       "mean_weekly_clicks_before": None if before is None else round(before, 1),
                                       "mean_weekly_clicks_after": None if after is None else round(after, 1),
                                       "activity_change_after_checkpoint": None if change is None else round(change, 3),
                                       "final_coursework_score": _clean(d["final_coursework_score"]), "exam_score": _clean(d["exam_score"]),
                                       "total_course_clicks": _clean(d["total_course_clicks"]),
                                       "trajectory": [_clean(x) for x in w] if w is not None else None,
                                       "text": "; ".join(story) + "."}})
        return out

    def similar(self, rp: ResolvedProfile | None = None, text: str | None = None, filters: dict | None = None, top_k: int = 10,
                mode: str = "hybrid", historical_only: bool = True, checkpoint=None, stories: bool = True,
                descriptors: list[str] | None = None, parser: str | None = None) -> dict:
        cp = rp.checkpoint if rp else checkpoint
        b = self.bundle(cp)
        prof = rp.profile if rp else None
        if prof is not None and filters is None:
            filters = {"code_module": prof.get("code_module")}
        res = b.index.search(profile=prof, text=text, filters=filters, top_k=top_k, mode=mode, historical_only=historical_only,
                             extra_tokens=descriptors, parser=parser)
        if stories:
            res["cases"] = self.precedent_stories(res["cases"], b.settings.cutoff_fraction)
            res["aftermath"] = self.aftermath_summary(res["cases"])
        res["checkpoint"] = b.settings.cutoff_fraction
        return res

    @staticmethod
    def aftermath_summary(cases: list[dict]) -> dict | None:
        st = [c["story"] for c in cases if c.get("story")]
        if not st:
            return None
        wd = [s for s in st if s["final_result"] == "Withdrawn"]
        changes = [s["activity_change_after_checkpoint"] for s in st if s["activity_change_after_checkpoint"] is not None]
        lag = [s["weeks_after_checkpoint_to_withdrawal"] for s in wd if s["weeks_after_checkpoint_to_withdrawal"] is not None]
        cw = [s["final_coursework_score"] for s in st if s["final_coursework_score"] is not None]
        return {"n": len(st), "outcomes": {k: int(v) for k, v in pd.Series([s["final_result"] for s in st]).value_counts().items()},
                "median_weeks_to_withdrawal": float(np.median(lag)) if lag else None,
                "median_activity_change_after_checkpoint": float(np.median(changes)) if changes else None,
                "share_with_activity_drop_over_50pct": float(np.mean([c < -0.5 for c in changes])) if changes else None,
                "median_final_coursework_score": float(np.median(cw)) if cw else None}

    def module_context(self, p: dict, checkpoint=None) -> dict:
        b = self.bundle(checkpoint)
        med = b.risk.art.module_medians.get(p.get("code_module"), b.risk.art.module_medians["__global__"])
        keys = ["weighted_score", "missed_rate", "total_clicks", "active_days", "days_since_last_activity"]
        return {"module": p.get("code_module"),
                "historical_medians": {k: (None if med.get(k) is None or np.isnan(med.get(k)) else round(float(med[k]), 2)) for k in keys}}

    def analyze(self, rp: ResolvedProfile, use_llm: bool = True, judge: bool = False, on_step=None, synthesize: bool = True) -> dict:
        pred = self.predict(rp)
        if on_step:
            on_step("models", {"source": "computed"})
        sim = self.similar(rp, top_k=6)
        if on_step:
            on_step("retrieval", {"source": "computed"})
        llm = self.llm() if use_llm else None
        wf = SupportWorkflow(llm, self.settings)
        result = wf.run(profile=rp.profile, risk=pred["withdrawal_risk"], outcome=pred["outcome_prediction"],
                        neighbours=sim["outcome_summary"], similar_cases=sim["cases"],
                        module_context=self.module_context(rp.profile, rp.checkpoint), on_step=on_step)
        out = {**pred, "similar_cases": sim, "workflow": result}
        if synthesize:
            from ssa.agents.genai import synthesize_precedents
            out["precedent_synthesis"] = synthesize_precedents(llm, rp.profile, pred, sim)
            if on_step:
                on_step("precedent_synthesis", out["precedent_synthesis"])
        if judge:
            from ssa.evaluation.judge import LLMJudge
            out["judge"] = LLMJudge(self.settings).evaluate(result, evidence={
                "observations": result["observations"], "risk": pred["withdrawal_risk"], "outcome": pred["outcome_prediction"],
                "neighbours": sim["outcome_summary"]})
            if on_step:
                on_step("judge", out["judge"])
        return out

    # ------------------------------------------------------- prioritise
    def prioritize(self, *, code_module: str | None = None, code_presentation: str | None = None, refs: list[dict] | None = None,
                   profiles: list[dict] | None = None, capacity: float | None = 0.2, limit: int = 100, checkpoint=None) -> dict:
        b = self.bundle(checkpoint)
        quality = None
        if code_module and code_presentation:
            f = b.features
            cohort = f[(f["code_module"] == code_module) & (f["code_presentation"] == code_presentation) & (f["in_scope"] == 1)]
            source = "feature_store_presentation"
        elif refs:
            rows, skipped = [], []
            for r in refs:
                try:
                    rows.append(self.resolve(ref=r, checkpoint=checkpoint).profile)
                except (NotFound, AlreadyWithdrawn) as exc:
                    skipped.append({"ref": r, "reason": type(exc).__name__})
            cohort = pd.DataFrame(rows)
            source, quality = "feature_store_refs", {"skipped": skipped}
        elif profiles:
            resolved = [self._impute(dict(p), b) for p in profiles]
            cohort = pd.DataFrame([r.profile for r in resolved])
            source = "supplied_profiles"
            quality = {"low_confidence_ids": [int(r.profile.get("id_student") or 0) for r in resolved if r.confidence == "low"]}
        else:
            raise ValueError("provide code_module+code_presentation, refs or profiles")
        if cohort.empty:
            raise NotFound("no in-scope students for this selection")
        ranking = b.prioritizer.rank(cohort, b.risk, b.outcome, capacity=capacity, limit=limit)
        return {"source": source, "checkpoint": b.settings.cutoff_fraction, "cohort_size": len(cohort), "capacity": capacity,
                "weights": b.prioritizer.weights, "weights_provenance": b.prioritizer.metadata.get("selection"),
                "ranking": ranking, "input_quality": quality}

    # ------------------------------------------------------- insights
    def course_trends(self, code_module: str | None = None) -> dict:
        t = read_json(self.settings.analysis_dir / "trend_analysis.json", {})
        return {"module": code_module, **t.get("modules", {}).get(code_module, {})} if code_module else t

    def assessment_insights(self, code_module: str | None = None, code_presentation: str | None = None) -> dict:
        a = read_json(self.settings.analysis_dir / "assessment_analytics.json", {"assessments": []})
        rows = [r for r in a["assessments"] if (not code_module or r["code_module"] == code_module)
                and (not code_presentation or r["code_presentation"] == code_presentation)]
        return {"note": a.get("note"), "assessments": rows}

    def factor_insights(self) -> dict:
        return read_json(self.settings.analysis_dir / "factor_analysis.json", {})

    def checkpoint_metrics(self) -> dict:
        return read_json(self.settings.global_metrics_dir / "checkpoint_summary.json", {"checkpoints": []})

    def metrics(self, checkpoint=None) -> dict:
        sc = self.settings.at(checkpoint)
        names = ["data_build", "withdrawal_model", "outcome_model", "ablations", "fairness", "prioritization", "retrieval_benchmark"]
        out = {n: read_json(sc.metrics_dir / f"{n}.json") for n in names}
        for n in ("llm_evaluation", "quality_gate", "checkpoint_summary"):
            out[n] = read_json(self.settings.global_metrics_dir / f"{n}.json")
        out["checkpoint"] = sc.cutoff_fraction
        return out
