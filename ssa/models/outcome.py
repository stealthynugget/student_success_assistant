"""Final-outcome (Distinction / Pass / Fail / Withdrawn) model for in-scope students.

Fixes vs v4: class weighting is *selected* on the validation presentation by macro-F1 rather
than assumed (v4 shipped a weighted model that was worse than its own baseline on macro-F1,
accuracy and weighted-F1); probabilities are temperature-scaled on out-of-fold margins;
majority-class and multinomial-logistic baselines are reported; ``P(poor) = P(Fail)+P(Withdrawn)``
feeds academic-support prioritisation.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ssa.config import Settings, get_settings
from ssa.data.vocab import OUTCOMES
from ssa.evaluation.metrics import binary_summary
from ssa.models.boosting import GBMConfig, GBMEngine, resolve_engine
from ssa.models.features import FeatureEncoder
from ssa.models.splits import temporal_split
from ssa.utils import write_json

MODEL_FILE = "outcome_model.joblib"
WEIGHTINGS = {"none": 0.0, "sqrt_inverse": 0.5, "inverse": 1.0}


def class_weights(y: np.ndarray, power: float, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = (counts.sum() / np.maximum(counts, 1) / n_classes) ** power
    return w[y]


class OutcomeModel:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.encoder = None
        self.engine = None
        self.temperature = 1.0
        self.log_w = None  # log class weights used in training (for prior correction)
        self.metadata: dict = {}

    def fit(self, df: pd.DataFrame) -> dict:
        s = self.settings
        t0 = time.time()
        work = df[(df["in_scope"] == 1) & df["final_result"].isin(OUTCOMES)].reset_index(drop=True)
        y = work["final_result"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy(int)
        work["_y"] = y
        split = temporal_split(work, "_y", s.random_state)
        self.encoder = FeatureEncoder(include_sensitive=s.include_sensitive_features)
        X = self.encoder.transform(work)
        engine_name = resolve_engine(s.model_engine)
        cfg = GBMConfig(0.05, 5, 60, 2.0)
        K = len(OUTCOMES)

        selection = {}
        for name, power in WEIGHTINGS.items():
            eng = GBMEngine(K, cfg, engine_name, s.random_state).fit(
                X[split.train], y[split.train], sample_weight=class_weights(y[split.train], power, K))
            pred = eng.predict_proba(X[split.val]).argmax(1)
            selection[name] = {"val_macro_f1": float(f1_score(y[split.val], pred, average="macro")),
                               "val_accuracy": float(accuracy_score(y[split.val], pred))}
        chosen = max(selection, key=lambda k: selection[k]["val_macro_f1"])
        power = WEIGHTINGS[chosen]

        hist = np.sort(np.concatenate([split.train, split.val]))
        # Temperature on out-of-fold margins of the validation presentation (model fit on train only).
        eng_tr = GBMEngine(K, cfg, engine_name, s.random_state).fit(
            X[split.train], y[split.train], sample_weight=class_weights(y[split.train], power, K))
        counts_tr = np.bincount(y[split.train], minlength=K).astype(float)
        w_tr = (counts_tr.sum() / np.maximum(counts_tr, 1) / K) ** power
        # Weighted training estimates p_w(k|x) ∝ w_k p(k|x); subtract log w_k to recover calibrated
        # probabilities, then temperature-scale. Decisions still use the balanced (weighted) argmax.
        val_margin = np.asarray(eng_tr.margin(X[split.val])) - np.log(w_tr)
        yv = y[split.val]
        res = minimize_scalar(lambda T: -np.mean(log_softmax(val_margin / T, axis=1)[np.arange(len(yv)), yv]),
                              bounds=(0.3, 5.0), method="bounded")
        self.temperature = float(res.x)

        self.engine = GBMEngine(K, cfg, engine_name, s.random_state).fit(
            X[hist], y[hist], sample_weight=class_weights(y[hist], power, K))
        counts_h = np.bincount(y[hist], minlength=K).astype(float)
        self.log_w = np.log((counts_h.sum() / np.maximum(counts_h, 1) / K) ** power)
        probs = self.predict_proba_matrix(X[split.test])
        yt = y[split.test]
        pred = self.decide(X[split.test])

        base = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=3000))
        base.fit(X[hist], y[hist], logisticregression__sample_weight=class_weights(y[hist], power, K))
        pb = base.predict(X[split.test])
        majority = np.full_like(yt, np.bincount(y[hist]).argmax())
        poor_true = np.isin(yt, [OUTCOMES.index("Fail"), OUTCOMES.index("Withdrawn")]).astype(int)
        p_poor = probs[:, OUTCOMES.index("Fail")] + probs[:, OUTCOMES.index("Withdrawn")]

        metrics = {
            "engine": engine_name, "split": split.info, "classes": list(OUTCOMES),
            "class_weighting_selection": selection, "chosen_weighting": chosen, "temperature": self.temperature,
            "test": {
                "accuracy": float(accuracy_score(yt, pred)), "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
                "macro_f1": float(f1_score(yt, pred, average="macro")), "weighted_f1": float(f1_score(yt, pred, average="weighted")),
                "log_loss": float(log_loss(yt, probs, labels=list(range(K)))),
                "per_class": {k: v for k, v in classification_report(yt, pred, labels=list(range(K)), target_names=list(OUTCOMES),
                                                                     output_dict=True, zero_division=0).items() if k in OUTCOMES},
                "confusion_matrix": confusion_matrix(yt, pred, labels=list(range(K))).tolist(),
                "n": int(len(yt)),
            },
            "poor_outcome_from_p_fail_plus_p_withdrawn_test": binary_summary(poor_true, p_poor),
            "baselines_test": {
                "majority_class": {"accuracy": float(accuracy_score(yt, majority)), "macro_f1": float(f1_score(yt, majority, average="macro"))},
                "multinomial_logistic": {"accuracy": float(accuracy_score(yt, pb)), "macro_f1": float(f1_score(yt, pb, average="macro")),
                                         "balanced_accuracy": float(balanced_accuracy_score(yt, pb))},
            },
            "population": "in_scope (enrolled at snapshot)", "train_seconds": round(time.time() - t0, 1),
        }
        self.metadata = {"metrics": metrics, "feature_contract": self.encoder.to_dict()}
        self.save()
        write_json(self.settings.metrics_dir / "outcome_model.json", metrics)
        return metrics

    def predict_proba_matrix(self, X) -> np.ndarray:
        """Calibrated class probabilities (prior-corrected + temperature-scaled)."""
        m = np.asarray(self.engine.margin(X))
        if self.log_w is not None:
            m = m - self.log_w
        return softmax(m / self.temperature, axis=1)

    def decide(self, X) -> np.ndarray:
        """Balanced decision rule (argmax of class-weighted scores), selected for macro-F1."""
        return np.asarray(self.engine.margin(X)).argmax(1)

    def save(self):
        self.settings.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"encoder": self.encoder, "engine": self.engine, "temperature": self.temperature,
                     "log_w": self.log_w, "metadata": self.metadata}, self.settings.model_dir / MODEL_FILE)

    def load(self):
        path = self.settings.model_dir / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run `python -m ssa.cli train`.")
        c = joblib.load(path)
        self.encoder, self.engine, self.temperature, self.metadata = c["encoder"], c["engine"], c["temperature"], c["metadata"]
        self.log_w = c.get("log_w")
        return self

    @property
    def ready(self) -> bool:
        return self.engine is not None or (self.settings.model_dir / MODEL_FILE).exists()

    def predict(self, df: pd.DataFrame) -> list[dict]:
        if self.engine is None:
            self.load()
        X = self.encoder.transform(df)
        P = self.predict_proba_matrix(X)
        decisions = self.decide(X)
        out = []
        for row, d in zip(P, decisions):
            probs = {o: round(float(v), 4) for o, v in zip(OUTCOMES, row)}
            out.append({"predicted_outcome": OUTCOMES[int(d)], "probabilities": probs,
                        "decision_rule": "balanced_argmax (macro-F1 optimised); probabilities are calibrated",
                        "p_poor_outcome": round(probs["Fail"] + probs["Withdrawn"], 4)})
        return out

    def p_poor(self, df: pd.DataFrame) -> np.ndarray:
        if self.engine is None:
            self.load()
        P = self.predict_proba_matrix(self.encoder.transform(df))
        return P[:, OUTCOMES.index("Fail")] + P[:, OUTCOMES.index("Withdrawn")]
