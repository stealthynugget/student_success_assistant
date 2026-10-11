"""Gradient-boosting engine abstraction.

* ``xgboost`` (preferred): early stopping on the *temporal validation presentation* and exact
  TreeSHAP attributions via ``pred_contribs=True`` - no dependency on the ``shap`` package,
  which v4 relied on and silently fell back to global feature importances when missing.
* ``sklearn`` HistGradientBoosting fallback (used automatically when xgboost is not
  installed): native NaN handling and internal early stopping. Attributions use exact
  *baseline substitution* (replace a feature group by its training reference value and
  measure the change in log-odds), which is model-agnostic and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def resolve_engine(preference: str = "auto") -> str:
    if preference == "xgboost":
        if not xgboost_available():
            raise RuntimeError("MODEL_ENGINE=xgboost but xgboost is not installed")
        return "xgboost"
    if preference == "sklearn":
        return "sklearn"
    return "xgboost" if xgboost_available() else "sklearn"


@dataclass
class GBMConfig:
    learning_rate: float = 0.05
    depth: int = 5
    min_leaf: int = 40
    l2: float = 1.0
    subsample: float = 0.85
    colsample: float = 0.8
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"learning_rate": self.learning_rate, "depth": self.depth, "min_leaf": self.min_leaf, "l2": self.l2,
                "subsample": self.subsample, "colsample": self.colsample}


SEARCH_SPACE = [
    GBMConfig(0.05, 4, 40, 1.0),
    GBMConfig(0.05, 6, 60, 2.0),
    GBMConfig(0.03, 5, 80, 5.0),
    GBMConfig(0.08, 3, 20, 0.5),
]


class GBMEngine:
    def __init__(self, n_classes: int = 2, config: GBMConfig | None = None, engine: str = "auto", seed: int = 42):
        self.n_classes = n_classes
        self.config = config or GBMConfig()
        self.engine = resolve_engine(engine)
        self.seed = seed
        self.model = None
        self.best_iteration: int | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None, val_weight=None):
        c = self.config
        if self.engine == "xgboost":
            import xgboost as xgb

            params = dict(
                n_estimators=3000, learning_rate=c.learning_rate, max_depth=c.depth, min_child_weight=max(1, c.min_leaf / 10),
                subsample=c.subsample, colsample_bytree=c.colsample, reg_lambda=c.l2, tree_method="hist",
                random_state=self.seed, n_jobs=-1, early_stopping_rounds=100,
            )
            if self.n_classes > 2:
                params.update(objective="multi:softprob", num_class=self.n_classes, eval_metric="mlogloss")
            else:
                params.update(objective="binary:logistic", eval_metric="logloss")
            self.model = xgb.XGBClassifier(**params)
            if X_val is None:  # carve an internal early-stopping split (mirrors sklearn's behaviour)
                rng = np.random.default_rng(self.seed)
                m = rng.random(len(y)) < 0.15
                X, X_val, y, y_val = X[~m], X[m], np.asarray(y)[~m], np.asarray(y)[m]
                if sample_weight is not None:
                    sample_weight, val_weight = np.asarray(sample_weight)[~m], np.asarray(sample_weight)[m]
            sw_eval = [val_weight] if val_weight is not None else None
            self.model.fit(X, y, sample_weight=sample_weight, eval_set=[(X_val, y_val)], sample_weight_eval_set=sw_eval,
                           verbose=False)
            self.best_iteration = int(getattr(self.model, "best_iteration", params["n_estimators"] - 1))
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model = HistGradientBoostingClassifier(
                learning_rate=c.learning_rate, max_iter=1500, max_depth=c.depth, max_leaf_nodes=min(63, 2 ** c.depth),
                min_samples_leaf=c.min_leaf, l2_regularization=c.l2, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=60, random_state=self.seed,
            )
            self.model.fit(X, y, sample_weight=sample_weight)
            self.best_iteration = int(self.model.n_iter_)
        return self

    # -------------------------------------------------------------- predict
    def _iter_range(self):
        return (0, self.best_iteration + 1) if self.best_iteration is not None else None

    def margin(self, X) -> np.ndarray:
        if self.engine == "xgboost":
            return self.model.predict(X, output_margin=True, iteration_range=self._iter_range())
        return self.model.decision_function(X)

    def predict_proba(self, X) -> np.ndarray:
        if self.engine == "xgboost":
            return self.model.predict_proba(X, iteration_range=self._iter_range())
        return self.model.predict_proba(X)

    # ------------------------------------------------------- attributions
    def contributions(self, X, group_index: dict[str, list[int]], reference: np.ndarray) -> tuple[np.ndarray, str]:
        """Per-feature-group contributions to the binary log-odds, shape (n, n_groups)."""
        groups = list(group_index)
        if self.engine == "xgboost":
            import xgboost as xgb

            raw = self.model.get_booster().predict(xgb.DMatrix(X, missing=np.nan), pred_contribs=True,
                                                   iteration_range=self._iter_range())
            raw = raw[:, :-1]  # drop bias column
            out = np.stack([raw[:, idx].sum(1) for idx in group_index.values()], axis=1)
            return out, "tree_shap"
        base = self.margin(X)
        out = np.zeros((X.shape[0], len(groups)), dtype=float)
        for g, (name, idx) in enumerate(group_index.items()):
            Xr = X.copy()
            Xr[:, idx] = reference[idx]
            out[:, g] = base - self.margin(Xr)
        return out, "baseline_substitution"

    def global_importance(self, X, group_index, reference, max_rows: int = 3000, seed: int = 0) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        rows = rng.choice(X.shape[0], size=min(max_rows, X.shape[0]), replace=False)
        contrib, _ = self.contributions(X[rows], group_index, reference)
        vals = np.abs(contrib).mean(0)
        return dict(sorted(zip(group_index, vals.tolist()), key=lambda kv: -kv[1]))
