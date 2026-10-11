"""Offline analysis artifacts: feature-group ablations, fairness audit, factor analysis
(PRD question 2) and engagement/performance trend analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ssa.config import Settings, get_settings
from ssa.evaluation.metrics import binary_summary, wilson_interval
from ssa.models.boosting import GBMConfig, GBMEngine, resolve_engine
from ssa.models.features import ACADEMIC, ENGAGEMENT, ENROLMENT, FRIENDLY, NUMERIC, SENSITIVE, FeatureEncoder
from ssa.models.risk import WithdrawalRiskModel, tier_of
from ssa.models.splits import temporal_split
from ssa.utils import write_json


def ablations(df: pd.DataFrame, settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    work = df[df["in_scope"] == 1].reset_index(drop=True)
    split = temporal_split(work, "withdrawn", s.random_state)
    hist = np.sort(np.concatenate([split.train, split.val]))
    y = work["withdrawn"].to_numpy(int)
    engine = resolve_engine(s.model_engine)
    variants = {
        "enrolment_only": FeatureEncoder(numeric=ENROLMENT),
        "academic_only": FeatureEncoder(numeric=ENROLMENT + ACADEMIC),
        "engagement_only": FeatureEncoder(numeric=ENROLMENT + ENGAGEMENT),
        "full_without_sensitive (production)": FeatureEncoder(include_sensitive=False),
        "full_with_sensitive": FeatureEncoder(include_sensitive=True),
    }
    out = {}
    for name, enc in variants.items():
        X = enc.transform(work)
        eng = GBMEngine(2, GBMConfig(0.03, 5, 80, 5.0), engine, s.random_state).fit(X[hist], y[hist])
        out[name] = binary_summary(y[split.test], 1 / (1 + np.exp(-eng.margin(X[split.test]))))
        out[name].pop("ece_10", None)
        out[name].pop("brier", None)
        out[name].pop("log_loss", None)
    full, sens = out["full_without_sensitive (production)"], out["full_with_sensitive"]
    result = {"split": split.info, "population": "in_scope", "metric_note": "uncalibrated scores; compare ranking metrics only",
              "variants": out,
              "cost_of_excluding_sensitive_attributes": {"roc_auc": sens["roc_auc"] - full["roc_auc"],
                                                         "pr_auc": sens["pr_auc"] - full["pr_auc"]}}
    write_json(s.metrics_dir / "ablations.json", result)
    return result


def fairness_audit(df: pd.DataFrame, model: WithdrawalRiskModel, settings: Settings | None = None, min_n: int = 50) -> dict:
    s = settings or get_settings()
    work = df[df["in_scope"] == 1].reset_index(drop=True)
    split = temporal_split(work, "withdrawn", s.random_state)
    test = work.iloc[split.test].reset_index(drop=True)
    p = model.predict_proba(test)
    thr = model.thresholds()
    flagged = np.array([tier_of(v, thr) == "High" for v in p])
    y = test["withdrawn"].to_numpy(int)
    report = {}
    for col in SENSITIVE:
        groups = {}
        for value, idx in test.groupby(col).groups.items():
            idx = np.asarray(list(idx))
            yy, pp, ff = y[idx], p[idx], flagged[idx]
            entry = {"n": int(len(idx)), "observed_rate": float(yy.mean()), "mean_predicted": float(pp.mean()),
                     "selection_rate_high": float(ff.mean()),
                     "tpr_high": float(ff[yy == 1].mean()) if (yy == 1).any() else None,
                     "fpr_high": float(ff[yy == 0].mean()) if (yy == 0).any() else None,
                     "ppv_high": float(yy[ff].mean()) if ff.any() else None,
                     "sufficient_sample": bool(len(idx) >= min_n and yy.sum() >= 10)}
            groups[str(value)] = entry
        ok = {k: v for k, v in groups.items() if v["sufficient_sample"]}
        tprs = [v["tpr_high"] for v in ok.values() if v["tpr_high"] is not None]
        sel = [v["selection_rate_high"] for v in ok.values()]
        calib = [abs(v["mean_predicted"] - v["observed_rate"]) for v in ok.values()]
        report[col] = {
            "groups": groups,
            "equal_opportunity_gap": float(max(tprs) - min(tprs)) if len(tprs) > 1 else None,
            "selection_rate_ratio": float(min(sel) / max(sel)) if len(sel) > 1 and max(sel) > 0 else None,
            "max_calibration_gap": float(max(calib)) if calib else None,
        }
    result = {
        "population": "in_scope test presentation", "threshold": "High tier",
        "model_uses_sensitive_attributes": bool(model.art.encoder.include_sensitive),
        "attributes": report,
        "interpretation": "Descriptive audit. Gaps can arise from different base rates even when sensitive attributes "
                          "are excluded from the model (proxies). Review groups with equal_opportunity_gap > 0.15 or "
                          "selection_rate_ratio < 0.8 (four-fifths rule) before deployment.",
    }
    write_json(s.metrics_dir / "fairness.json", result)
    return result


def factor_analysis(df: pd.DataFrame, model: WithdrawalRiskModel, settings: Settings | None = None) -> dict:
    """Which factors are associated with poor performance / withdrawal? (historical, in-scope)."""
    s = settings or get_settings()
    work = df[df["in_scope"] == 1].reset_index(drop=True)
    split = temporal_split(work, "withdrawn", s.random_state)
    hist = work.iloc[np.sort(np.concatenate([split.train, split.val]))]
    base_w, base_p = hist["withdrawn"].mean(), hist["poor_outcome"].mean()
    numeric_report = []
    for col in NUMERIC:
        x = pd.to_numeric(hist[col], errors="coerce")
        if x.nunique(dropna=True) < 3:
            continue
        bins = pd.qcut(x.rank(method="first"), q=5, labels=False)
        rows = []
        for b in range(5):
            m = bins == b
            if m.sum() == 0:
                continue
            nw, n = int(hist.loc[m, "withdrawn"].sum()), int(m.sum())
            npoor = int(hist.loc[m, "poor_outcome"].sum())
            rows.append({"quintile": b + 1, "range": [float(x[m].min()), float(x[m].max())], "n": n,
                         "withdrawal_rate": nw / n, "withdrawal_ci": wilson_interval(nw, n),
                         "poor_outcome_rate": npoor / n, "poor_outcome_ci": wilson_interval(npoor, n)})
        missing = x.isna()
        rho = spearmanr(x[~missing], hist.loc[~missing, "poor_outcome"]).statistic if (~missing).sum() > 10 else np.nan
        spread = max(r["poor_outcome_rate"] for r in rows) - min(r["poor_outcome_rate"] for r in rows)
        numeric_report.append({"feature": col, "label": FRIENDLY.get(col, col), "spearman_with_poor_outcome": float(rho),
                               "poor_outcome_rate_spread": float(spread), "quintiles": rows,
                               "missing_share": float(missing.mean())})
    numeric_report.sort(key=lambda r: -r["poor_outcome_rate_spread"])

    categorical_report = {}
    for col in ["code_module", "num_of_prev_attempts"] + list(SENSITIVE):
        rows = []
        for value, g in hist.groupby(col):
            n = len(g)
            if n < 30:
                continue
            rows.append({"value": str(value), "n": n, "withdrawal_rate": float(g["withdrawn"].mean()),
                         "poor_outcome_rate": float(g["poor_outcome"].mean()),
                         "poor_outcome_ci": wilson_interval(int(g["poor_outcome"].sum()), n)})
        categorical_report[col] = {"descriptive_only": col in SENSITIVE, "levels": rows}

    imp = model.art.importance
    result = {
        "population": "historical in-scope enrolments", "n": int(len(hist)),
        "base_rates": {"withdrawal": float(base_w), "poor_outcome": float(base_p)},
        "model_importance_mean_abs_logodds": {FRIENDLY.get(k, k): v for k, v in imp.items()},
        "numeric_factors_ranked": numeric_report[:25],
        "categorical_factors": categorical_report,
        "caveat": "Associations, not causal effects. Sensitive attributes are reported descriptively for equity "
                  "monitoring and are never used as reasons for intervention.",
    }
    write_json(s.analysis_dir / "factor_analysis.json", result)
    return result


def trend_analysis(settings: Settings | None = None) -> dict:
    """Engagement trajectories by final outcome per module and the week the at-risk groups diverge."""
    s = settings or get_settings()
    if not s.course_trends_path.exists():
        return {"available": False}
    t = pd.read_csv(s.course_trends_path)
    out = {}
    for module, g in t.groupby("code_module"):
        curves = (g.groupby(["final_result", "week_full"])["mean_clicks_per_student"].mean().unstack(0).sort_index())
        curves = curves[curves.index <= 39]
        divergence = {}
        if "Pass" in curves:
            for grp in ["Fail", "Withdrawn"]:
                if grp in curves:
                    ratio = ((curves[grp] + 1) / (curves["Pass"] + 1)).rolling(2).max()  # sustained 2 weeks
                    weeks = ratio[ratio < 0.6].index
                    divergence[grp] = int(weeks.min()) - 1 if len(weeks) else None
        out[module] = {"weekly_mean_clicks_by_outcome": {c: [round(float(v), 2) for v in curves[c].fillna(0)] for c in curves},
                       "weeks": [int(w) for w in curves.index], "first_week_below_60pct_of_pass": divergence}
    result = {"modules": out, "note": "Mean weekly VLE clicks per student still enrolled at the snapshot, over the full course, by final outcome "
                                       "(historical descriptive analysis; not used as model input)."}
    write_json(s.analysis_dir / "trend_analysis.json", result)
    return result
