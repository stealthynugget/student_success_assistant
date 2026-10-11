"""Withdrawal early-warning model.

Population: students still enrolled at the snapshot (``in_scope == 1``). Target: final_result
== Withdrawn (i.e. withdrawal *after* the snapshot - the only withdrawals an early-warning
system can still prevent).

Pipeline: temporal split -> small hyper-parameter search selected on the validation
presentation (PR-AUC) -> Platt calibration fitted on validation margins -> two risk thresholds
learned on validation -> one final scoring of the untouched test presentation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ssa.config import Settings, get_settings
from ssa.evaluation.metrics import at_capacity, binary_summary, bootstrap_auc_ci, reliability_table
from ssa.models.boosting import SEARCH_SPACE, GBMEngine, resolve_engine
from ssa.models.features import ACADEMIC, ENGAGEMENT, ENROLMENT, FRIENDLY, NUMERIC, SENSITIVE, FeatureEncoder
from ssa.models.splits import temporal_split
from ssa.utils import get_logger, write_json

log = get_logger("ssa.risk")
MODEL_FILE = "withdrawal_model.joblib"


class PlattCalibrator:
    """Monotone logistic recalibration of GBM margins (fitted on validation data only)."""

    def __init__(self):
        self.a, self.b = 1.0, 0.0

    def fit(self, margin, y):
        lr = LogisticRegression(C=1e4, max_iter=1000)
        lr.fit(np.asarray(margin).reshape(-1, 1), np.asarray(y))
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, margin):
        z = self.a * np.asarray(margin, float) + self.b
        return 1.0 / (1.0 + np.exp(-z))


def learn_thresholds(y, p, medium_recall: float = 0.80, min_high_recall: float = 0.30) -> dict:
    """Learn both tier thresholds on validation data.

    medium = highest threshold that still captures ``medium_recall`` of eventual withdrawals
    (the "watch list"); high = F1-optimal threshold above medium with at least
    ``min_high_recall`` recall. v4 learned only "high" and left "medium" as a static env value,
    which collapsed the Medium band to 0.35-0.37.
    """
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok = np.where(rec >= medium_recall)[0]
    medium = float(thr[ok[-1]]) if len(ok) else float(np.quantile(p, 0.5))
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    cand = np.where((thr > medium) & (rec >= min_high_recall))[0]
    if len(cand):
        high = float(thr[cand[np.argmax(f1[cand])]])
        method = "validation_f1_above_medium"
    else:
        high = float(np.quantile(p, 0.90))
        method = "fallback_top_decile"
    if high <= medium:
        high = float(np.quantile(p[p > medium], 0.5)) if (p > medium).any() else medium + 1e-3
        method += "+separated"

    def tier_stats(t):
        pred = p >= t
        return {"threshold": t, "precision": float(y[pred].mean()) if pred.any() else None,
                "recall": float(pred[y == 1].mean()) if (y == 1).any() else None, "flag_rate": float(pred.mean())}

    return {"medium": medium, "high": high, "method": method, "medium_recall_target": medium_recall,
            "validation": {"medium": tier_stats(medium), "high": tier_stats(high)}}


def tier_of(p: float, thresholds: dict) -> str:
    if p >= thresholds["high"]:
        return "High"
    if p >= thresholds["medium"]:
        return "Medium"
    return "Low"


@dataclass
class RiskArtifacts:
    encoder: FeatureEncoder
    engine: GBMEngine
    calibrator: PlattCalibrator
    thresholds: dict
    reference: np.ndarray
    module_medians: dict
    feature_bounds: dict
    importance: dict
    metadata: dict = field(default_factory=dict)


class WithdrawalRiskModel:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.art: RiskArtifacts | None = None

    # ---------------------------------------------------------------- train
    def fit(self, df: pd.DataFrame) -> dict:
        """Temporal protocol.

        1. Hyper-parameters: fit on train presentations, select by PR-AUC on the validation
           presentation.
        2. Calibration + thresholds: leave-one-presentation-out (LOPO) out-of-fold margins over
           all historical presentations (train + validation), so the calibrator never sees
           predictions on data its model was trained on.
        3. Final model: refit on all historical presentations (recency matters: adding the most
           recent presentation is worth ~+0.03 ROC-AUC on OULAD).
        4. One final score on the untouched test presentation.
        """
        s = self.settings
        t0 = time.time()
        work = df[df["in_scope"] == 1].reset_index(drop=True)
        split = temporal_split(work, "withdrawn", s.random_state)
        enc = FeatureEncoder(include_sensitive=s.include_sensitive_features)
        X = enc.transform(work)
        y = work["withdrawn"].to_numpy(int)
        engine_name = resolve_engine(s.model_engine)

        search = []
        best = None
        for cfg in SEARCH_SPACE:
            eng = GBMEngine(2, cfg, engine_name, s.random_state).fit(X[split.train], y[split.train])
            ap = float(average_precision_score(y[split.val], eng.margin(X[split.val])))
            search.append({"config": cfg.as_dict(), "val_pr_auc": ap, "iterations": eng.best_iteration})
            if best is None or ap > best[0]:
                best = (ap, cfg)
        cfg = best[1]

        hist = np.sort(np.concatenate([split.train, split.val]))
        oof_margin = self._oof_margins(work, X, y, hist, cfg, engine_name)
        calibrator = PlattCalibrator().fit(oof_margin, y[hist])
        p_oof = calibrator.transform(oof_margin)
        thresholds = learn_thresholds(y[hist], p_oof)

        engine = GBMEngine(2, cfg, engine_name, s.random_state).fit(X[hist], y[hist])
        p_test = calibrator.transform(engine.margin(X[split.test]))

        Xh = X[hist]
        reference = np.nanmedian(Xh, axis=0)
        reference = np.where(np.isnan(reference), 0.0, reference)
        mod_cols = [i for i, n in enumerate(enc.feature_names) if n.startswith("code_module=")]
        if mod_cols:  # reference module = most frequent historical module
            reference[mod_cols] = 0.0
            reference[mod_cols[int(np.argmax(np.nansum(Xh[:, mod_cols], 0)))]] = 1.0
        hist_df = work.iloc[hist]
        module_medians = {m: g[NUMERIC].median(numeric_only=True).to_dict() for m, g in hist_df.groupby("code_module")}
        module_medians["__global__"] = hist_df[NUMERIC].median(numeric_only=True).to_dict()
        bounds = {c: [float(hist_df[c].quantile(0.005)), float(hist_df[c].quantile(0.995))]
                  for c in NUMERIC if hist_df[c].notna().any()}
        importance = engine.global_importance(X[split.test], enc.group_index, reference)

        self.art = RiskArtifacts(enc, engine, calibrator, thresholds, reference, module_medians, bounds, importance)
        metrics = self._evaluate(work, split, X, y, hist, p_oof, p_test)
        metrics.update({"split": split.info, "engine": engine_name, "selected_config": cfg.as_dict(),
                        "hyperparameter_search": search, "train_seconds": round(time.time() - t0, 1),
                        "protocol": "config on validation; LOPO out-of-fold calibration+thresholds; refit on all historical presentations",
                        "global_importance_test": {FRIENDLY.get(k, k): v for k, v in list(importance.items())[:20]}})
        metrics["population_audit"] = self._population_audit(df, split)
        self.art.metadata = {"metrics": metrics, "thresholds": thresholds, "feature_contract": enc.to_dict(),
                             "engine": engine_name, "target": "withdrawn_after_snapshot", "population": "in_scope",
                             "trained_on_presentations": sorted(hist_df["code_presentation"].unique().tolist())}
        self.save()
        write_json(self.settings.metrics_dir / "withdrawal_model.json", metrics)
        return metrics

    def _oof_margins(self, work, X, y, hist, cfg, engine_name) -> np.ndarray:
        pres = work["code_presentation"].to_numpy()[hist]
        groups = np.unique(pres)
        oof = np.zeros(len(hist))
        if len(groups) < 2:  # tiny fixtures: fall back to 3-fold stratified OOF
            from sklearn.model_selection import StratifiedKFold
            folds = StratifiedKFold(3, shuffle=True, random_state=self.settings.random_state).split(hist, y[hist])
        else:
            folds = ((np.where(pres != g)[0], np.where(pres == g)[0]) for g in groups)
        for tr, te in folds:
            eng = GBMEngine(2, cfg, engine_name, self.settings.random_state).fit(X[hist[tr]], y[hist[tr]])
            oof[te] = eng.margin(X[hist[te]])
        return oof

    def _evaluate(self, work, split, X, y, hist, p_oof, p_test) -> dict:
        yte = y[split.test]
        thr = self.art.thresholds
        tiers = np.array([tier_of(v, thr) for v in p_test])
        tier_table = {}
        for t in ["High", "Medium", "Low"]:
            m = tiers == t
            tier_table[t] = {"n": int(m.sum()), "share": float(m.mean()),
                             "observed_withdrawal_rate": float(yte[m].mean()) if m.any() else None,
                             "share_of_all_withdrawals": float(yte[m].sum() / max(yte.sum(), 1))}
        flag_high = p_test >= thr["high"]
        flag_med = p_test >= thr["medium"]

        # Transparent baseline: regularised logistic regression on the same features.
        base = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000))
        base.fit(X[hist], y[hist])
        p_base = base.predict_proba(X[split.test])[:, 1]

        summary = binary_summary(yte, p_test)
        summary["roc_auc_95ci"] = bootstrap_auc_ci(yte, p_test)
        return {
            "test": summary,
            "historical_out_of_fold": binary_summary(y[hist], p_oof),
            "thresholds": thr,
            "test_at_high_threshold": {"precision": float(yte[flag_high].mean()) if flag_high.any() else None,
                                       "recall": float(flag_high[yte == 1].mean()),
                                       "f1": float(f1_score(yte, flag_high)), "flag_rate": float(flag_high.mean())},
            "test_at_medium_threshold": {"precision": float(yte[flag_med].mean()) if flag_med.any() else None,
                                         "recall": float(flag_med[yte == 1].mean()), "flag_rate": float(flag_med.mean())},
            "test_tiers": tier_table,
            "test_capacity": [at_capacity(yte, p_test, f) for f in (0.05, 0.10, 0.20, 0.30)],
            "test_reliability": reliability_table(yte, p_test),
            "baseline_logistic_regression_test": binary_summary(yte, p_base),
        }

    def _population_audit(self, df_all: pd.DataFrame, split) -> dict:
        """Quantify v4's leakage: the same model scored on the full test presentation including
        students who had *already* unregistered before the snapshot."""
        test_pres = split.info.get("test", [None])[0]
        if test_pres is None:
            return {"available": False}
        full = df_all[df_all["code_presentation"] == test_pres]
        p = self.predict_proba(full)
        y = full["withdrawn"].to_numpy(int)
        scoped = full["in_scope"].to_numpy() == 1
        return {
            "available": True, "test_presentation": test_pres,
            "all_enrolments_including_already_withdrawn": binary_summary(y, p),
            "in_scope_only": binary_summary(y[scoped], p[scoped]),
            "already_withdrawn_share_of_test_withdrawals": float(((~scoped) & (y == 1)).sum() / max(y.sum(), 1)),
            "note": "Scoring students who already left inflates ROC-AUC; only the in-scope figure is a valid early-warning estimate.",
        }

    # ------------------------------------------------------------- persist
    def save(self):
        self.settings.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.art, self.settings.model_dir / MODEL_FILE)

    def load(self):
        path = self.settings.model_dir / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run `python -m ssa.cli train`.")
        self.art = joblib.load(path)
        return self

    @property
    def ready(self) -> bool:
        return self.art is not None or (self.settings.model_dir / MODEL_FILE).exists()

    def _ensure(self):
        if self.art is None:
            self.load()

    # -------------------------------------------------------------- serve
    def _override(self) -> dict | None:
        """Feedback-loop recalibration (applied only after human approval)."""
        path = self.settings.artifact_dir / "feedback" / "calibration_override.json"
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        if getattr(self, "_ov_mtime", None) != mtime:
            from ssa.utils import read_json
            self._ov, self._ov_mtime = read_json(path), mtime
        return self._ov

    def thresholds(self) -> dict:
        self._ensure()
        ov = self._override()
        return ov["thresholds"] if ov and "thresholds" in ov else self.art.thresholds

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure()
        X = self.art.encoder.transform(df)
        p = self.art.calibrator.transform(self.art.engine.margin(X))
        ov = self._override()
        if ov:
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
            p = 1 / (1 + np.exp(-(ov["a"] * z + ov["b"])))
        return p

    def explain(self, df: pd.DataFrame, top_n: int = 6) -> list[list[dict]]:
        self._ensure()
        X = self.art.encoder.transform(df)
        contrib, method = self.art.engine.contributions(X, self.art.encoder.group_index, self.art.reference)
        names = list(self.art.encoder.group_index)
        out = []
        for i in range(len(df)):
            row = df.iloc[i]
            order = np.argsort(-np.abs(contrib[i]))
            drivers = []
            for j in order:
                g = names[j]
                if g in SENSITIVE:  # never surface protected attributes as reasons
                    continue
                if abs(contrib[i, j]) < 1e-4:
                    continue
                val = row.get(g) if g in row else None
                drivers.append({
                    "feature": g, "label": FRIENDLY.get(g, g),
                    "value": None if val is None or (isinstance(val, float) and np.isnan(val)) else (float(val) if isinstance(val, (int, float, np.number)) else str(val)),
                    "contribution_logodds": round(float(contrib[i, j]), 4),
                    "direction": "increases_risk" if contrib[i, j] > 0 else "reduces_risk",
                })
                if len(drivers) >= top_n:
                    break
            out.append([{**d, "method": method} for d in drivers])
        return out

    def score(self, df: pd.DataFrame, explain: bool = True) -> list[dict]:
        self._ensure()
        p = self.predict_proba(df)
        drivers = self.explain(df) if explain else [[] for _ in range(len(df))]
        thr = self.thresholds()
        return [{
            "withdrawal_risk": round(float(pi), 4),
            "risk_tier": tier_of(float(pi), thr),
            "thresholds": {"medium": round(thr["medium"], 4), "high": round(thr["high"], 4)},
            "drivers": d,
            "calibration": "platt_lopo" + ("+feedback_override" if self._override() else ""),
            "interpretation": "Calibrated probability of withdrawing after the snapshot among students still enrolled. "
                              "An early-warning estimate, not a diagnosis or a causal claim.",
        } for pi, d in zip(p, drivers)]

    def feature_groups(self) -> dict:
        return {"academic": ACADEMIC, "engagement": ENGAGEMENT, "enrolment": ENROLMENT}
