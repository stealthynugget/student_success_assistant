import numpy as np

from ssa.evaluation.model_analysis import _eo_gap, _gap_verdict, _threshold_review
from ssa.utils import read_json


def test_threshold_review_reaches_target_tpr():
    rng = np.random.default_rng(0)
    n = 2000
    g = np.where(rng.random(n) < 0.5, "a", "b")
    y = (rng.random(n) < 0.2).astype(int)
    # group b's withdrawals are scored lower, so one shared cut-off catches fewer of them
    p = np.clip(0.1 + 0.3 * y * np.where(g == "a", 1.0, 0.6) + 0.2 * rng.random(n), 0, 1)
    rev = _threshold_review(p, y, g, ["a", "b"], thr_high=0.35)
    a, b = rev["groups"]["a"], rev["groups"]["b"]
    assert a["current_tpr"] > b["current_tpr"]
    for r in (a, b):
        assert r["tpr_at_equalising"] >= rev["target_tpr"] - 1e-9
    assert b["equalising_threshold"] < 0.35 and b["change_in_flagged_students"] > 0


def test_gap_and_verdict():
    flag = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    grp = np.array(list("aaaabbbb"))
    assert abs(_eo_gap(flag, grp, ["a", "b"]) - 0.5) < 1e-9
    assert _eo_gap(flag, grp, ["a"]) is None
    assert _gap_verdict(0.3, {"p_value": 0.2, "within_module_p_value": 0.2}).startswith("within chance")
    assert "modules" in _gap_verdict(0.3, {"p_value": 0.01, "within_module_p_value": 0.4})
    assert _gap_verdict(0.3, {"p_value": 0.01, "within_module_p_value": 0.01}).endswith("review")


def test_audit_artifact_shape(built):
    f = read_json(built.metrics_dir / "fairness.json")
    assert "sufficient_sample_rule" in f and not f["model_uses_sensitive_attributes"]
    for rep in f["attributes"].values():
        for grp in rep["groups"].values():
            assert {"positives", "tpr_high_ci95", "mean_predicted_among_withdrawn"} <= set(grp)
            if not grp["sufficient_sample"]:
                assert grp in [rep["groups"][k] for k in rep["groups_excluded_small_sample"]]
        if rep["equal_opportunity_gap"] is not None:
            assert {"gap_null", "verdict", "proxies", "threshold_review"} <= set(rep)
