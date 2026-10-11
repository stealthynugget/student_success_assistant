"""Capacity-aware student prioritisation for academic support.

v4 used a hand-typed formula (0.5 risk + 0.25 academic + 0.15 engagement + 0.10 decline) on
absolute click counts, double-counting signals already inside the risk score and never
validated. Here:

* Target = *poor outcome* (Fail or Withdrawn) - the population that needs academic support.
* Signals are bounded and module-relative (cohort percentiles), so a click count means the
  same thing in every module.
* Weights are learned on the **validation presentation** by maximising mean precision at
  advisor capacities of 10/20/30 % (exhaustive search over a simplex grid), then evaluated
  once on the untouched test presentation against single-signal baselines and the v4 formula.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ssa.config import Settings, get_settings
from ssa.evaluation.metrics import at_capacity
from ssa.models.outcome import OutcomeModel
from ssa.models.risk import WithdrawalRiskModel, tier_of
from ssa.models.splits import temporal_split
from ssa.utils import read_json, write_json

SIGNALS = ["p_withdraw", "p_poor", "academic_gap", "engagement_gap", "missed_work", "recent_decline"]
SIGNAL_LABELS = {
    "p_withdraw": "Predicted withdrawal risk", "p_poor": "Predicted fail-or-withdraw probability",
    "academic_gap": "Low score relative to cohort", "engagement_gap": "Low activity relative to cohort",
    "missed_work": "Missed due assessments", "recent_decline": "Recent engagement decline",
}
CAPACITIES = (0.10, 0.20, 0.30)
FILE = "prioritization.json"


def compute_signals(df: pd.DataFrame, risk: WithdrawalRiskModel, outcome: OutcomeModel) -> pd.DataFrame:
    sig = pd.DataFrame(index=df.index)
    sig["p_withdraw"] = risk.predict_proba(df)
    sig["p_poor"] = outcome.p_poor(df)
    score_pct = pd.to_numeric(df.get("score_pct"), errors="coerce")
    # No score yet counts as a moderate academic gap (unknown), not as "fine".
    sig["academic_gap"] = (1 - score_pct).fillna(0.6).clip(0, 1)
    sig["engagement_gap"] = (1 - pd.to_numeric(df.get("clicks_pct"), errors="coerce")).fillna(0.5).clip(0, 1)
    sig["missed_work"] = pd.to_numeric(df.get("missed_rate"), errors="coerce").fillna(0).clip(0, 1)
    momentum = pd.to_numeric(df.get("engagement_momentum"), errors="coerce").fillna(0)
    sig["recent_decline"] = (-momentum).clip(0, 1)
    return sig


def _rank_normalise(sig: pd.DataFrame) -> pd.DataFrame:
    """Map each signal to [0,1] by within-cohort rank so weights are scale-free."""
    return sig.rank(pct=True, method="average")


def _objective(y, score) -> float:
    return float(np.mean([at_capacity(y, score, c)["precision"] for c in CAPACITIES]))


def simplex_grid(n: int, step: float = 0.1):
    k = int(round(1 / step))
    for combo in itertools.product(range(k + 1), repeat=n - 1):
        if sum(combo) <= k:
            yield np.array(list(combo) + [k - sum(combo)], dtype=float) / k


class Prioritizer:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.weights: dict | None = None
        self.metadata: dict = {}

    def fit(self, df: pd.DataFrame, risk: WithdrawalRiskModel, outcome: OutcomeModel) -> dict:
        s = self.settings
        work = df[df["in_scope"] == 1].reset_index(drop=True)
        split = temporal_split(work, "poor_outcome", s.random_state)
        val, test = work.iloc[split.val].reset_index(drop=True), work.iloc[split.test].reset_index(drop=True)

        def ranked(frame):
            sig = compute_signals(frame, risk, outcome)
            return sig.groupby(frame["code_presentation"].values).transform(lambda c: c.rank(pct=True))

        rv, rt = ranked(val), ranked(test)
        yv, yt = val["poor_outcome"].to_numpy(int), test["poor_outcome"].to_numpy(int)
        best_w, best_obj = None, -1.0
        Rv = rv[SIGNALS].to_numpy()
        for w in simplex_grid(len(SIGNALS), 0.1):
            obj = _objective(yv, Rv @ w)
            if obj > best_obj + 1e-9:
                best_w, best_obj = w, obj
        self.weights = dict(zip(SIGNALS, best_w.round(3).tolist()))

        Rt = rt[SIGNALS].to_numpy()
        v4 = 0.5 * rt["p_withdraw"] + 0.25 * rt["academic_gap"] + 0.15 * rt["engagement_gap"] + 0.10 * rt["recent_decline"]
        rng = np.random.default_rng(s.random_state)
        candidates = {"learned_weights": Rt @ best_w, "v4_hand_formula": v4.to_numpy(),
                      "random": rng.random(len(yt))}
        for sname in SIGNALS:
            candidates[f"only_{sname}"] = rt[sname].to_numpy()
        test_eval = {name: {"objective_mean_precision": _objective(yt, sc),
                            "at_capacity": [at_capacity(yt, sc, c) for c in CAPACITIES]} for name, sc in candidates.items()}
        self.metadata = {
            "target": "poor_outcome (Fail or Withdrawn)", "population": "in_scope", "weights": self.weights,
            "validation_objective": best_obj, "objective": "mean precision@{10,20,30}% capacity",
            "selection": "exhaustive simplex grid (step 0.1) on the validation presentation",
            "test_comparison": test_eval, "split": split.info, "signal_labels": SIGNAL_LABELS,
        }
        write_json(s.model_dir / FILE, self.metadata)
        write_json(s.metrics_dir / "prioritization.json", self.metadata)
        return self.metadata

    def load(self):
        meta = read_json(self.settings.model_dir / FILE)
        if not meta:
            raise FileNotFoundError("prioritization weights missing; run `python -m ssa.cli train`")
        self.metadata, self.weights = meta, meta["weights"]
        return self

    def rank(self, cohort: pd.DataFrame, risk: WithdrawalRiskModel, outcome: OutcomeModel,
             capacity: float | None = None, top_drivers: bool = True, limit: int | None = None) -> list[dict]:
        """Score the whole cohort; compute (comparatively expensive) per-student explanations only for the
        ``limit`` students that will actually be returned."""
        if self.weights is None:
            self.load()
        cohort = cohort.reset_index(drop=True)
        sig = compute_signals(cohort, risk, outcome)
        ranked = _rank_normalise(sig) if len(cohort) > 1 else sig.clip(0, 1)
        w = np.array([self.weights[k] for k in SIGNALS])
        score = ranked[SIGNALS].to_numpy() @ w
        order = np.argsort(-score, kind="stable")
        shown = order[: limit] if limit else order
        thr = risk.thresholds()
        drivers = {}
        if top_drivers and len(shown):
            expl = risk.explain(cohort.iloc[shown], top_n=3)
            drivers = {int(i): d for i, d in zip(shown, expl)}
        k = int(np.ceil(len(cohort) * capacity)) if capacity else None
        out = []
        for rank, i in enumerate(shown, 1):
            contrib = {sname: round(float(ranked.iloc[i][sname] * self.weights[sname]), 4) for sname in SIGNALS}
            # Reason codes describe observable signals in the student's cohort-relative top quartile, so they stay
            # informative even when the learned weights concentrate on a model probability.
            observable = ["missed_work", "academic_gap", "engagement_gap", "recent_decline"]
            strong = sorted((s_ for s_ in observable if ranked.iloc[i][s_] >= 0.75 and sig.iloc[i][s_] > 0),
                            key=lambda s_: -ranked.iloc[i][s_])
            reasons = [SIGNAL_LABELS[x] for x in strong][:3] or \
                [SIGNAL_LABELS[x] for x, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v > 0][:2]
            p = float(sig.iloc[i]["p_withdraw"])
            out.append({
                "priority_rank": rank, "id_student": int(cohort.iloc[i]["id_student"]),
                "code_module": str(cohort.iloc[i]["code_module"]),
                "code_presentation": str(cohort.iloc[i].get("code_presentation", "")),
                "priority_score": round(float(score[i]), 4),
                "within_capacity": (rank <= k) if k else None,
                "withdrawal_risk": round(p, 4), "risk_tier": tier_of(p, thr),
                "p_poor_outcome": round(float(sig.iloc[i]["p_poor"]), 4),
                "signals": {sname: round(float(sig.iloc[i][sname]), 4) for sname in SIGNALS},
                "weighted_contributions": contrib, "reasons": reasons,
                "top_drivers": drivers.get(int(i), []),
            })
        return out
