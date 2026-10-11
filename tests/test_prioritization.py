import numpy as np

from ssa.models.prioritization import SIGNALS, simplex_grid
from ssa.utils import read_json


def test_simplex_grid_is_valid():
    grid = list(simplex_grid(3, 0.5))
    assert all(abs(w.sum() - 1) < 1e-9 and (w >= 0).all() for w in grid)
    assert len(grid) == 6


def test_learned_weights_and_comparisons(built):
    m = read_json(built.metrics_dir / "prioritization.json")
    assert abs(sum(m["weights"].values()) - 1) < 1e-6
    assert set(m["weights"]) == set(SIGNALS)
    for name in ("learned_weights", "v4_hand_formula", "random"):
        assert name in m["test_comparison"]


def test_rank_presentation_with_capacity(service):
    res = service.prioritize(code_module="AAA", code_presentation="2014J", capacity=0.2, limit=1000)
    ranks = res["ranking"]
    scores = [r["priority_score"] for r in ranks]
    assert scores == sorted(scores, reverse=True)
    assert sum(r["within_capacity"] for r in ranks) == int(np.ceil(res["cohort_size"] * 0.2))
    assert all(r["reasons"] for r in ranks[:5])
