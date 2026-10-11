"""Durable recommendation registry + advisor feedback (SQLite, WAL, foreign keys).

Every recommendation the system produces is registered with its evidence fingerprint, model and
prompt versions, chosen actions and the per-channel ranks of the similar cases shown. Feedback
must reference a registered recommendation (v4 accepted any free-form id) and can rate the plan,
individual actions, individual similar cases, and later record the real outcome.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from ssa.config import Settings, get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
  id TEXT PRIMARY KEY, created_at REAL NOT NULL, id_student INTEGER NOT NULL, code_module TEXT, code_presentation TEXT,
  risk_score REAL, risk_tier TEXT, p_poor REAL, mode TEXT, plan_source TEXT, action_ids TEXT, similar_cases TEXT,
  prompt_version TEXT, model_version TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recommendation_id TEXT NOT NULL REFERENCES recommendations(id),
  created_at REAL NOT NULL, advisor_id TEXT, rating INTEGER CHECK (rating BETWEEN 1 AND 5), plan_accepted INTEGER,
  risk_assessment_agrees INTEGER, action_feedback TEXT, case_feedback TEXT, observed_outcome TEXT,
  intervention_effective INTEGER, notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_rec ON feedback(recommendation_id);
"""


class UnknownRecommendation(LookupError):
    pass


class FeedbackStore:
    def __init__(self, settings: Settings | None = None, path: Path | None = None):
        self.settings = settings or get_settings()
        self.path = Path(path or self.settings.feedback_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.row_factory = sqlite3.Row
        return c

    def register(self, analysis: dict, model_version: str = "") -> str:
        wf = analysis["workflow"]
        st = analysis["student"]
        cases = [{"case_id": c["case_id"], "channel_ranks": c["channel_ranks"], "final_result": c["final_result"]}
                 for c in analysis["similar_cases"]["cases"]]
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO recommendations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                wf["recommendation_id"], time.time(), st["id_student"], st["code_module"], st["code_presentation"],
                analysis["withdrawal_risk"]["withdrawal_risk"], analysis["withdrawal_risk"]["risk_tier"],
                analysis["outcome_prediction"]["p_poor_outcome"], wf["intervention_plan"]["mode"],
                wf["intervention_plan"]["source"], json.dumps([a["id"] for a in wf["intervention_plan"]["actions"]]),
                json.dumps(cases), wf["prompt_version"], model_version))
        return wf["recommendation_id"]

    def get_recommendation(self, rec_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM recommendations WHERE id=?", (rec_id,)).fetchone()
        return dict(row) if row else None

    def add_feedback(self, fb: dict) -> int:
        rec = self.get_recommendation(fb["recommendation_id"])
        if rec is None:
            raise UnknownRecommendation(fb["recommendation_id"])
        if int(fb["id_student"]) != int(rec["id_student"]):
            raise ValueError("id_student does not match the recommendation")
        allowed_actions = set(json.loads(rec["action_ids"]))
        unknown = set((fb.get("action_feedback") or {})) - allowed_actions
        if unknown:
            raise ValueError(f"action_feedback refers to actions not in this recommendation: {sorted(unknown)}")
        allowed_cases = {str(x["case_id"]) for x in json.loads(rec["similar_cases"])}
        unknown = {str(k) for k in (fb.get("case_feedback") or {})} - allowed_cases
        if unknown:
            raise ValueError(f"case_feedback refers to cases not shown in this recommendation: {sorted(unknown)}")
        b = lambda v: None if v is None else int(bool(v))  # noqa: E731
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO feedback (recommendation_id, created_at, advisor_id, rating, plan_accepted, risk_assessment_agrees, "
                "action_feedback, case_feedback, observed_outcome, intervention_effective, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (fb["recommendation_id"], time.time(), fb.get("advisor_id"), fb.get("rating"), b(fb.get("plan_accepted")),
                 b(fb.get("risk_assessment_agrees")), json.dumps(fb.get("action_feedback") or {}),
                 json.dumps({str(k): v for k, v in (fb.get("case_feedback") or {}).items()}), fb.get("observed_outcome"),
                 b(fb.get("intervention_effective")), fb.get("notes", "")))
            return int(cur.lastrowid)

    def joined(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT f.*, r.risk_score, r.risk_tier, r.p_poor, r.action_ids, r.similar_cases, r.prompt_version, "
                             "r.plan_source FROM feedback f JOIN recommendations r ON r.id=f.recommendation_id").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("action_feedback", "case_feedback", "action_ids", "similar_cases"):
                d[k] = json.loads(d[k]) if d.get(k) else {}
            out.append(d)
        return out

    def summary(self) -> dict:
        rows = self.joined()
        if not rows:
            return {"count": 0}
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return round(sum(xs) / len(xs), 3) if xs else None
        return {"count": len(rows), "average_rating": mean([r["rating"] for r in rows]),
                "plan_acceptance_rate": mean([r["plan_accepted"] for r in rows]),
                "risk_agreement_rate": mean([r["risk_assessment_agrees"] for r in rows]),
                "effectiveness_rate": mean([r["intervention_effective"] for r in rows]),
                "with_observed_outcome": sum(r["observed_outcome"] is not None for r in rows)}
