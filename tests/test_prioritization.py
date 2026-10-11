import numpy as np

from ssa.models.prioritization import SIGNALS, _fast_objective, _objective, select_weights, simplex_grid
from ssa.utils import read_json


def test_simplex_grid_is_valid():
    grid = list(simplex_grid(3, 0.5))
    assert all(abs(w.sum() - 1) < 1e-9 and (w >= 0).all() for w in grid)
    assert len(grid) == 6


def test_fast_objective_matches_reference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 503)
    score = np.round(rng.random(503), 2)  # ties exercise the stable sort
    ks = [max(1, int(round(503 * c))) for c in (0.1, 0.2, 0.3)]
    assert abs(_fast_objective(y, score, ks) - _objective(y, score)) < 1e-12


def test_one_se_rule_prefers_a_blend_among_equivalent_weightings():
    rng = np.random.default_rng(0)
    n = 400
    latent = rng.random(n)
    y = (latent + 0.3 * rng.random(n) > 0.9).astype(int)
    # two nearly identical informative signals and one noise signal
    R = np.column_stack([latent + 0.01 * rng.random(n), latent + 0.01 * rng.random(n), rng.random(n)])
    arg = select_weights(R, y, step=0.1, rule="argmax")
    blend = select_weights(R, y, step=0.1, rule="one_se_blend")
    assert np.allclose(arg["chosen"], arg["argmax"])
    assert blend["chosen_objective"] >= blend["argmax_objective"] - blend["bootstrap_se"] - 1e-12
    assert (blend["chosen"] > 0).sum() >= 2  # spreads weight rather than putting 1.0 on one signal
    assert abs(blend["chosen"].sum() - 1) < 1e-9


def test_learned_weights_and_comparisons(built):
    m = read_json(built.metrics_dir / "prioritization.json")
    assert abs(sum(m["weights"].values()) - 1) < 1e-6
    assert set(m["weights"]) == set(SIGNALS)
    assert "argmax_weights" in m["selection_detail"]
    for name in ("learned_weights", "validation_argmax", "v4_hand_formula", "random"):
        assert name in m["test_comparison"]


def test_rank_presentation_with_capacity(service):
    res = service.prioritize(code_module="AAA", code_presentation="2014J", capacity=0.2, limit=1000)
    ranks = res["ranking"]
    scores = [r["priority_score"] for r in ranks]
    assert scores == sorted(scores, reverse=True)
    assert sum(r["within_capacity"] for r in ranks) == int(np.ceil(res["cohort_size"] * 0.2))
    assert all(r["reasons"] for r in ranks[:5])
