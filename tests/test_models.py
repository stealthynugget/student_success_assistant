import json

import numpy as np

from ssa.models.features import SENSITIVE
from ssa.models.risk import learn_thresholds, tier_of
from ssa.utils import read_json


def test_learned_thresholds_are_ordered_and_meaningful():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.12
    p = np.clip(0.12 + 0.25 * y + rng.normal(0, 0.1, 4000), 0.001, 0.999)
    t = learn_thresholds(y, p)
    assert t["medium"] < t["high"]
    assert t["validation"]["medium"]["recall"] >= 0.8 - 1e-9
    assert tier_of(t["high"], t) == "High" and tier_of(t["medium"], t) == "Medium" and tier_of(0.0, t) == "Low"


def test_withdrawal_model_trained_on_in_scope_with_honest_metrics(built):
    m = read_json(built.metrics_dir / "withdrawal_model.json")
    assert m["thresholds"]["medium"] < m["thresholds"]["high"]
    assert m["population_audit"]["available"]
    assert 0 <= m["test"]["ece_10"] <= 1
    assert "baseline_logistic_regression_test" in m
    assert m["split"]["method"] == "temporal" and m["split"]["test"] == ["2014J"]


def test_sensitive_attributes_never_model_inputs_or_drivers(service, features):
    assert not service.risk.art.encoder.include_sensitive
    assert not set(SENSITIVE) & set(service.risk.art.encoder.groups)
    sample = features[features["in_scope"] == 1].head(25)
    for drivers in service.risk.explain(sample):
        assert all(d["feature"] not in SENSITIVE for d in drivers)


def test_scores_and_outcomes_are_valid_probabilities(service, features):
    sample = features[features["in_scope"] == 1].head(30)
    risk = service.risk.score(sample)
    assert all(0 <= r["withdrawal_risk"] <= 1 and r["risk_tier"] in {"High", "Medium", "Low"} for r in risk)
    out = service.outcome.predict(sample)
    for o in out:
        assert abs(sum(o["probabilities"].values()) - 1) < 1e-3
        assert abs(o["p_poor_outcome"] - o["probabilities"]["Fail"] - o["probabilities"]["Withdrawn"]) < 2e-4


def test_outcome_weighting_selected_on_validation(built):
    m = read_json(built.metrics_dir / "outcome_model.json")
    sel = m["class_weighting_selection"]
    assert m["chosen_weighting"] == max(sel, key=lambda k: sel[k]["val_macro_f1"])
    assert m["test"]["macro_f1"] > m["baselines_test"]["majority_class"]["macro_f1"]


def test_metrics_json_is_strict(built):
    for p in built.metrics_dir.glob("*.json"):
        json.loads(p.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
