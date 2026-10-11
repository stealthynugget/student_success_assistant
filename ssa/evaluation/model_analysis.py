"""Offline analysis artifacts: feature-group ablations, fairness audit, factor analysis
(PRD question 2) and engagement/performance trend analysis."""
from __future__ import annotations

import warnings

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


MIN_GROUP_POSITIVES = 30  # a TPR from fewer withdrawals has a 95% interval wider than +/-0.18: too noisy to compare


def _eo_gap(flag_pos: np.ndarray, groups_pos: np.ndarray, keep: list) -> float | None:
    tprs = [flag_pos[groups_pos == g].mean() for g in keep if (groups_pos == g).any()]
    return float(max(tprs) - min(tprs)) if len(tprs) > 1 else None


def _proxy_analysis(X: np.ndarray, names: list[str], member: np.ndarray, seed: int) -> dict:
    """How well the model's own inputs identify membership of one group (cross-validated ROC-AUC), and which inputs
    differ most between that group and everyone else (standardised mean difference)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    clf = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_leaf_nodes=15, random_state=seed)
    score = cross_val_predict(clf, X, member, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    a, b = X[member == 1], X[member == 0]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pooled = np.sqrt((np.nanvar(a, axis=0) + np.nanvar(b, axis=0)) / 2)
        smd = (np.nanmean(a, axis=0) - np.nanmean(b, axis=0)) / pooled
    smd = np.where(np.isfinite(smd), smd, 0.0)
    top = np.argsort(-np.abs(smd))[:4]
    return {"membership_auc_from_model_inputs": float(roc_auc_score(member, score)),
            "most_different_inputs": [{"feature": names[i], "label": FRIENDLY.get(names[i], names[i]),
                                       "standardised_mean_difference": float(smd[i])} for i in top]}


def _gap_verdict(gap: float, null: dict) -> str:
    if null["p_value"] >= 0.05:
        return "within chance: random group labels give a gap this large at these group sizes"
    if null["within_module_p_value"] >= 0.05:
        return "beyond chance overall, but explained by which modules the groups study"
    return "beyond chance, also within modules: review"


def _threshold_review(p: np.ndarray, y: np.ndarray, g: np.ndarray, keep: list, thr_high: float) -> dict:
    """For each group: the High-tier cut-off that would give it the overall true-positive rate, and what that would cost.
    Descriptive only: applying it would use a sensitive attribute at decision time, which this service does not do."""
    flagged = p >= thr_high
    target = float(flagged[y == 1].mean())
    rows = {}
    for v in keep:
        m = g == v
        pos = np.sort(p[m & (y == 1)])
        k = int(np.ceil(target * len(pos)))  # flag at least this many of the group's withdrawn students
        t = float(pos[len(pos) - k]) if k else float(thr_high)
        fl = p[m] >= t
        rows[str(v)] = {"current_tpr": float(flagged[m & (y == 1)].mean()), "current_selection_rate": float(flagged[m].mean()),
                        "current_ppv": float(y[m][flagged[m]].mean()) if flagged[m].any() else None,
                        "equalising_threshold": t, "tpr_at_equalising": float((pos >= t).mean()),
                        "selection_rate_at_equalising": float(fl.mean()),
                        "ppv_at_equalising": float(y[m][fl].mean()) if fl.any() else None,
                        "change_in_flagged_students": int(fl.sum() - flagged[m].sum())}
    return {"current_threshold": float(thr_high), "target_tpr": target, "groups": rows}


def fairness_audit(df: pd.DataFrame, model: WithdrawalRiskModel, settings: Settings | None = None, min_n: int = 50,
                   n_perm: int = 1000) -> dict:
    s = settings or get_settings()
    work = df[df["in_scope"] == 1].reset_index(drop=True)
    split = temporal_split(work, "withdrawn", s.random_state)
    test = work.iloc[split.test].reset_index(drop=True)
    p = model.predict_proba(test)
    thr = model.thresholds()
    flagged = np.array([tier_of(v, thr) == "High" for v in p])
    y = test["withdrawn"].to_numpy(int)
    enc = model.art.encoder
    X = enc.transform(test)
    rng = np.random.default_rng(s.random_state)
    mods_pos = test["code_module"].astype(str).to_numpy()[y == 1]
    report = {}
    for col in SENSITIVE:
        groups = {}
        gcol = test[col].astype(str).to_numpy()
        for value, idx in test.groupby(col).groups.items():
            idx = np.asarray(list(idx))
            yy, pp, ff = y[idx], p[idx], flagged[idx]
            npos, tp = int(yy.sum()), int(ff[yy == 1].sum())
            entry = {"n": int(len(idx)), "positives": npos, "observed_rate": float(yy.mean()), "mean_predicted": float(pp.mean()),
                     "mean_predicted_among_withdrawn": float(pp[yy == 1].mean()) if npos else None,
                     "selection_rate_high": float(ff.mean()),
                     "tpr_high": tp / npos if npos else None,
                     "tpr_high_ci95": list(wilson_interval(tp, npos)) if npos else None,
                     "fpr_high": float(ff[yy == 0].mean()) if (yy == 0).any() else None,
                     "ppv_high": float(yy[ff].mean()) if ff.any() else None,
                     "sufficient_sample": bool(len(idx) >= min_n and npos >= MIN_GROUP_POSITIVES
                                               and len(idx) - npos >= MIN_GROUP_POSITIVES)}
            groups[str(value)] = entry
        ok = {k: v for k, v in groups.items() if v["sufficient_sample"]}
        keep = list(ok)
        sel = [v["selection_rate_high"] for v in ok.values()]
        calib = [abs(v["mean_predicted"] - v["observed_rate"]) for v in ok.values()]
        fpos, gpos = flagged[y == 1], gcol[y == 1]
        gap = _eo_gap(fpos, gpos, keep)
        entry = {
            "groups": groups,
            "equal_opportunity_gap": gap,
            "selection_rate_ratio": float(min(sel) / max(sel)) if len(sel) > 1 and max(sel) > 0 else None,
            "max_calibration_gap": float(max(calib)) if calib else None,
            "groups_excluded_small_sample": [k for k, v in groups.items() if not v["sufficient_sample"]],
        }
        if gap is not None:
            # the max-min gap that group labels unrelated to flagging would produce at these group sizes
            null = np.array([_eo_gap(fpos, rng.permutation(gpos), keep) for _ in range(n_perm)])
            # same, but shuffling only within each module: is the gap explained by which modules the groups study?
            strata = [np.where(mods_pos == m)[0] for m in np.unique(mods_pos)]
            null_mod = []
            for _ in range(n_perm):
                gp = gpos.copy()
                for ix in strata:
                    gp[ix] = rng.permutation(gpos[ix])
                null_mod.append(_eo_gap(fpos, gp, keep))
            null_mod = np.array(null_mod)
            tprs = {k: v["tpr_high"] for k, v in ok.items()}
            lo, hi = min(tprs, key=tprs.get), max(tprs, key=tprs.get)
            entry["gap_null"] = {"method": f"{n_perm} permutations of group labels among withdrawn students",
                                 "median": float(np.median(null)), "p95": float(np.quantile(null, 0.95)),
                                 "p_value": float((1 + (null >= gap).sum()) / (1 + n_perm)),
                                 "within_module_median": float(np.median(null_mod)),
                                 "within_module_p_value": float((1 + (null_mod >= gap).sum()) / (1 + n_perm))}
            entry["verdict"] = _gap_verdict(gap, entry["gap_null"])
            entry["lowest_tpr_group"], entry["highest_tpr_group"] = lo, hi
            entry["proxies"] = {g: _proxy_analysis(X, enc.feature_names, (gcol == g).astype(int), s.random_state) for g in (lo, hi)}
            entry["threshold_review"] = _threshold_review(p, y, gcol, keep, thr["high"])
        report[col] = entry
    result = {
        "population": "in_scope test presentation", "threshold": "High tier",
        "model_uses_sensitive_attributes": bool(model.art.encoder.include_sensitive),
        "sufficient_sample_rule": f"n >= {min_n}, at least {MIN_GROUP_POSITIVES} withdrawn and {MIN_GROUP_POSITIVES} not withdrawn",
        "attributes": report,
        "interpretation": "Descriptive audit. Gaps can arise from different base rates even when sensitive attributes "
                          "are excluded from the model (proxies). Review groups with equal_opportunity_gap > 0.15 or "
                          "selection_rate_ratio < 0.8 (four-fifths rule) before deployment. gap_null is the gap that random "
                          "group labels give at the same group sizes; a gap inside it is not evidence of a group difference. "
                          "threshold_review is for policy review only: group-specific cut-offs would use a sensitive "
                          "attribute at decision time, so they are not applied.",
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
