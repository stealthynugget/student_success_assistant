"""Closed feedback loop: advisor feedback -> evidence-backed, versioned changes.

Each learner produces a *proposal* with its evidence and minimum-sample guard. ``apply=True``
(CLI: ``python -m ssa.cli feedback-loop --apply``) writes the approved artifacts that the
running system reads on the next request:

1. ``action_stats.json``   - Beta-Binomial acceptance per action; low performers are demoted in
   the catalog shown to the intervention agent and the deterministic planner.
2. ``retrieval_weights.json`` - RRF channel weights re-learned from advisors' "useful case" labels
   using the per-channel ranks recorded for each shown case.
3. ``calibration_override.json`` - Platt recalibration of the withdrawal score on observed
   outcomes (+ re-learned tier thresholds) when enough outcomes exist and Brier improves in CV.
4. Drift / retraining / prompt-quality monitors that raise actions for humans.

v4's "feedback loop" appended JSONL and printed generic advice; nothing changed.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from ssa.config import Settings, get_settings
from ssa.feedback.store import FeedbackStore
from ssa.models.risk import PlattCalibrator, learn_thresholds
from ssa.utils import read_json, write_json

PRIOR_A, PRIOR_B = 2.0, 2.0
MIN_ACTION_N = 10
MIN_RETRIEVAL_ROWS = 30
MIN_OUTCOMES = 200


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def action_learner(rows: list[dict]) -> dict:
    counts: dict[str, list[int]] = {}
    for r in rows:
        for aid, v in (r.get("action_feedback") or {}).items():
            accepted = v.get("accepted") if isinstance(v, dict) else v
            if accepted is None:
                continue
            counts.setdefault(aid, []).append(int(bool(accepted)))
    actions = {}
    for aid, xs in counts.items():
        k, n = sum(xs), len(xs)
        mean = (PRIOR_A + k) / (PRIOR_A + PRIOR_B + n)
        actions[aid] = {"n": n, "accepted": k, "acceptance_posterior_mean": round(mean, 4),
                        "demoted": bool(n >= MIN_ACTION_N and mean < 0.35)}
    return {"type": "action_catalog_ranking", "status": "proposed" if actions else "insufficient_data",
            "evidence": {"actions_with_feedback": len(actions)}, "payload": {"actions": actions}}


def retrieval_learner(rows: list[dict], current: dict) -> dict:
    data = []
    for r in rows:
        labels = r.get("case_feedback") or {}
        if not labels:
            continue
        cases = {str(c["case_id"]): c.get("channel_ranks", {}) for c in (r.get("similar_cases") or [])}
        items = [(cases[cid], 1.0 if (v.get("useful") if isinstance(v, dict) else v) else 0.0)
                 for cid, v in labels.items() if cid in cases]
        if len(items) >= 2 and any(x[1] for x in items):
            data.append(items)
    if len(data) < MIN_RETRIEVAL_ROWS:
        return {"type": "retrieval_fusion_weights", "status": "insufficient_data", "evidence": {"labelled_rows": len(data),
                "required": MIN_RETRIEVAL_ROWS}}

    def ndcg(weights):
        vals = []
        for items in data:
            fused = [sum(weights.get(ch, 0) / (60 + rk) for ch, rk in ranks.items()) for ranks, _ in items]
            gains = [g for _, g in sorted(zip(fused, [g for _, g in items]), key=lambda x: -x[0])]
            disc = 1 / np.log2(np.arange(2, len(gains) + 2))
            ideal = sorted(gains, reverse=True)
            vals.append(float(np.dot(gains, disc) / max(np.dot(ideal, disc), 1e-9)))
        return float(np.mean(vals))

    grid = [{"behavioural": 1.0, "lexical": a, "dense": b} for a, b in itertools.product((0, .25, .5, 1, 2), (0, .5, 1))]
    scored = sorted(((ndcg(w), w) for w in grid), key=lambda x: -x[0])
    base = ndcg(current)
    best_score, best = scored[0]
    status = "proposed" if best_score > base + 0.01 else "no_change"
    return {"type": "retrieval_fusion_weights", "status": status,
            "evidence": {"labelled_rows": len(data), "current_ndcg": base, "proposed_ndcg": best_score, "current": current},
            "payload": {"weights": best}}


def calibration_learner(rows: list[dict]) -> dict:
    data = [(r["risk_score"], 1 if r["observed_outcome"] == "Withdrawn" else 0) for r in rows
            if r.get("observed_outcome") and r.get("risk_score") is not None]
    if len(data) < MIN_OUTCOMES:
        return {"type": "risk_recalibration", "status": "insufficient_data",
                "evidence": {"outcomes": len(data), "required": MIN_OUTCOMES}}
    p, y = np.array([d[0] for d in data]), np.array([d[1] for d in data])
    if y.min() == y.max():
        return {"type": "risk_recalibration", "status": "insufficient_data", "evidence": {"reason": "single class"}}
    rng = np.random.default_rng(0)
    folds = rng.integers(0, 5, len(y))
    before, after = [], []
    for k in range(5):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        cal = PlattCalibrator().fit(_logit(p[tr]), y[tr])
        before.append(np.mean((p[te] - y[te]) ** 2))
        after.append(np.mean((cal.transform(_logit(p[te])) - y[te]) ** 2))
    cal = PlattCalibrator().fit(_logit(p), y)
    p_new = cal.transform(_logit(p))
    improve = float(np.mean(before) - np.mean(after))
    return {"type": "risk_recalibration", "status": "proposed" if improve > 0.002 else "no_change",
            "evidence": {"outcomes": len(y), "observed_rate": float(y.mean()), "mean_predicted": float(p.mean()),
                         "cv_brier_before": float(np.mean(before)), "cv_brier_after": float(np.mean(after))},
            "payload": {"a": cal.a, "b": cal.b, "thresholds": learn_thresholds(y, p_new)}}


def monitors(rows: list[dict], settings: Settings) -> list[dict]:
    out = []
    with_outcome = [r for r in rows if r.get("observed_outcome")]
    metrics = read_json(settings.metrics_dir / "withdrawal_model.json", {})
    expected = metrics.get("test_tiers", {})
    for tier in ("High", "Medium", "Low"):
        obs = [1 if r["observed_outcome"] == "Withdrawn" else 0 for r in with_outcome if r["risk_tier"] == tier]
        exp = expected.get(tier, {}).get("observed_withdrawal_rate")
        if len(obs) >= 50 and exp is not None:
            rate = float(np.mean(obs))
            se = np.sqrt(max(exp * (1 - exp), 1e-6) / len(obs))
            drift = abs(rate - exp) > 2.5 * se
            out.append({"type": "tier_drift", "tier": tier, "status": "alert" if drift else "ok",
                        "evidence": {"n": len(obs), "observed": rate, "expected_from_test": exp}})
    if len(with_outcome) >= 500:
        out.append({"type": "retrain", "status": "recommended", "evidence": {"outcome_labels": len(with_outcome)},
                    "action": "Add the new presentation's OULAD-format extracts and run `python -m ssa.cli build`."})
    by_src: dict[str, list[int]] = {}
    for r in rows:
        if r.get("rating") is not None:
            by_src.setdefault(str(r.get("plan_source")), []).append(r["rating"])
    llm = [v for k, v in by_src.items() if k.startswith("llm") for v in v]
    det = by_src.get("deterministic", [])
    if len(llm) >= 20 and len(det) >= 20:
        gap = float(np.mean(det) - np.mean(llm))
        out.append({"type": "prompt_quality", "status": "alert" if gap > 0.5 else "ok",
                    "evidence": {"llm_mean_rating": float(np.mean(llm)), "deterministic_mean_rating": float(np.mean(det))},
                    "action": "Review prompt version / consider disabling the LLM intervention agent." if gap > 0.5 else None})
    return out


def run_loop(settings: Settings | None = None, apply: bool = False, approved_by: str | None = None) -> dict:
    s = settings or get_settings()
    rows = FeedbackStore(s).joined()
    current_w = read_json(s.retrieval_weights_path, {}).get("weights") or read_json(s.index_dir / "fusion_weights.json", {}).get(
        "weights") or {"behavioural": 1.0, "lexical": 0.5, "dense": 0.5}
    proposals = [action_learner(rows), retrieval_learner(rows, current_w), calibration_learner(rows)]
    applied = []
    if apply:
        stamp = {"applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "approved_by": approved_by or "cli"}
        for prop in proposals:
            if prop["status"] != "proposed":
                continue
            if prop["type"] == "action_catalog_ranking":
                write_json(s.action_stats_path, {**prop["payload"], **stamp})
            elif prop["type"] == "retrieval_fusion_weights":
                write_json(s.retrieval_weights_path, {**prop["payload"], **stamp, "evidence": prop["evidence"]})
            elif prop["type"] == "risk_recalibration":
                write_json(s.artifact_dir / "feedback" / "calibration_override.json", {**prop["payload"], **stamp, "evidence": prop["evidence"]})
            prop["status"] = "applied"
            applied.append(prop["type"])
    plan = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "feedback_rows": len(rows), "proposals": proposals,
            "monitors": monitors(rows, s), "applied": applied,
            "process": ["advisors rate plans, actions and similar cases; record real outcomes",
                        "learners produce proposals with evidence and minimum-sample guards",
                        "a human approves with --apply; artifacts are versioned with timestamp and approver",
                        "the API reads applied artifacts on the next request; offline evaluation re-runs"]}
    write_json(s.improvement_plan_path, plan)
    return plan
