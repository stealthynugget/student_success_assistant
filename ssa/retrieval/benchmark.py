"""Retrieval benchmark.

v4's nDCG divided each list by its *own* best ordering, so a random list scored 0.854 (above
its hybrid retriever). This benchmark is task-grounded and non-circular:

* **Queries** = in-scope students from the latest (test) presentation. **Corpus** = earlier
  presentations only (``historical_only``), exactly as in production.
* **Neighbour-vote AUC**: the share of the k retrieved historical cases that withdrew (or
  failed/withdrew) is used as a predictor of the query student's real outcome. This directly
  answers PRD Q1 - "has a student with similar patterns previously struggled or withdrawn?" -
  and measures whether retrieval surfaces *informative* precedents.
* **Outcome agreement@k**: share of neighbours whose poor/non-poor outcome matches the query's.
* **Reference nDCG@k with a global ideal**: relevance from an *unweighted* standardised distance
  over the full numeric feature set; IDCG uses the true top-k of the whole eligible corpus.
  (Partially aligned with the behavioural channel by construction - reported as a secondary
  diagnostic only.)
* A **random** within-module baseline is always reported; the quality gate requires the
  hybrid retriever to beat it by a clear margin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ssa.config import Settings, get_settings
from ssa.data.vocab import presentation_sort_key
from ssa.models.features import NUMERIC
from ssa.retrieval.hybrid import MODES, HistoricalCaseIndex
from ssa.utils import write_json


def _ndcg(gains, ideal):
    disc = 1 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((np.asarray(gains) * disc).sum())
    idcg = float((np.asarray(ideal) * disc[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def _evaluate(df, index, query_presentation, specs, n_queries, k, seed):
    """specs: list of (name, mode, weights|None); mode 'random' = random within-module baseline."""
    index._ensure()
    scoped = df[df["in_scope"] == 1].reset_index(drop=True)
    pool = scoped[scoped["code_presentation"] == query_presentation]
    queries = pool.sample(min(n_queries, len(pool)), random_state=seed)
    ref_cols = [c for c in NUMERIC if c in scoped]
    R = scoped[ref_cols].apply(pd.to_numeric, errors="coerce")
    med, iqr = R.median(), (R.quantile(0.75) - R.quantile(0.25)).replace(0, 1).fillna(1)
    Rz = ((R - med) / iqr).clip(-5, 5).fillna(0).to_numpy()
    key_to_row = pd.Series(np.arange(len(scoped)), index=scoped["enrollment_key"].to_numpy())
    corpus_keys = index.meta["enrollment_key"].to_numpy()
    corpus_rows = key_to_row.reindex(corpus_keys).to_numpy()
    key_pos = {ck: i for i, ck in enumerate(corpus_keys)}
    rng = np.random.default_rng(seed)
    acc = {name: {"vote_w": [], "vote_p": [], "agree": [], "ndcg": []} for name, _, _ in specs}
    y_w, y_p = [], []
    for _, q in queries.iterrows():
        prof = q.to_dict()
        filt = {"code_module": q["code_module"]}
        elig = np.where(index._allowed(filt, int(q["id_student"]), str(q["code_presentation"])))[0]
        if len(elig) < k:
            continue
        rows_e = corpus_rows[elig]
        d_ref = np.sqrt(((Rz[rows_e] - Rz[key_to_row[q["enrollment_key"]]]) ** 2).sum(1))
        rel = 1.0 / (1.0 + d_ref)
        rel_of = dict(zip(elig, rel))
        ideal = np.sort(rel)[::-1][:k]
        y_w.append(int(q["withdrawn"])), y_p.append(int(q["poor_outcome"]))
        for name, mode, weights in specs:
            if mode == "random":
                docs = rng.choice(elig, k, replace=False)
            else:
                res = index.search(profile=prof, filters=filt, top_k=k, mode=mode, weights=weights)
                docs = np.array([key_pos[c["case_id"]] for c in res["cases"]])
            sub = index.meta.iloc[docs]
            acc[name]["vote_w"].append(sub["withdrawn"].mean())
            acc[name]["vote_p"].append(sub["poor_outcome"].mean())
            acc[name]["agree"].append(float((sub["poor_outcome"] == q["poor_outcome"]).mean()))
            acc[name]["ndcg"].append(_ndcg([rel_of[d] for d in docs], ideal))
    y_w, y_p = np.array(y_w), np.array(y_p)
    out = {}
    for name, a in acc.items():
        out[name] = {
            "neighbour_vote_auc_withdrawal": float(roc_auc_score(y_w, a["vote_w"])) if len(set(y_w)) > 1 else None,
            "neighbour_vote_auc_poor_outcome": float(roc_auc_score(y_p, a["vote_p"])) if len(set(y_p)) > 1 else None,
            f"outcome_agreement_at_{k}": float(np.mean(a["agree"])),
            f"reference_ndcg_at_{k}_global_ideal": float(np.mean(a["ndcg"])),
        }
    return out, y_w, y_p


def tune_fusion_weights(df: pd.DataFrame, index: HistoricalCaseIndex, n_queries: int = 300, k: int = 10,
                        settings: Settings | None = None, seed: int = 7) -> dict:
    """Select RRF channel weights on the *validation* presentation (queries) vs older presentations."""
    s = settings or get_settings()
    pres = sorted(df.loc[df["in_scope"] == 1, "code_presentation"].unique(), key=presentation_sort_key)
    if len(pres) < 3:
        return {"status": "skipped", "reason": "needs >= 3 presentations"}
    grid = [{"behavioural": 1.0, "lexical": lw, "dense": dw} for lw in (0.0, 0.25, 0.5, 1.0, 2.0)
            for dw in ((0.0, 0.5, 1.0) if index.dense is not None else (0.0,))]
    specs = [(f"w{i}", "hybrid", w) for i, w in enumerate(grid)]
    res, _, _ = _evaluate(df, index, pres[-2], specs, n_queries, k, seed)
    def objective(r):
        return 0.5 * r["neighbour_vote_auc_poor_outcome"] + 0.5 * r[f"reference_ndcg_at_{k}_global_ideal"]
    scored = sorted(((objective(res[f"w{i}"]), w) for i, w in enumerate(grid)), key=lambda x: -x[0])
    best = scored[0][1]
    out = {"status": "tuned", "validation_presentation": pres[-2], "objective": "0.5*vote_AUC(poor) + 0.5*nDCG",
           "weights": best, "grid": [{"weights": w, "objective": o} for o, w in scored]}
    write_json(s.index_dir / "fusion_weights.json", out)
    return out


def run_benchmark(df: pd.DataFrame, index: HistoricalCaseIndex, n_queries: int = 600, k: int = 10,
                  settings: Settings | None = None, seed: int = 42) -> dict:
    s = settings or get_settings()
    index._ensure()
    pres = sorted(df.loc[df["in_scope"] == 1, "code_presentation"].unique(), key=presentation_sort_key)
    modes = [m for m in MODES if m != "dense" or index.dense is not None]
    specs = [(m, m, None) for m in modes] + [("random", "random", None)]
    out, y_w, y_p = _evaluate(df, index, pres[-1], specs, n_queries, k, seed)
    result = {
        "queries": int(len(y_w)), "k": k, "query_presentation": pres[-1], "corpus": "earlier presentations, same module",
        "query_positive_rates": {"withdrawn": float(y_w.mean()), "poor_outcome": float(y_p.mean())},
        "modes": out,
        "hybrid_minus_random": {key: (out["hybrid"][key] - out["random"][key]) if out["hybrid"][key] is not None else None
                                for key in out["hybrid"]},
        "notes": [
            "Neighbour-vote AUC and outcome agreement are non-circular: relevance comes from real future outcomes.",
            "Reference nDCG uses an unweighted distance over all numeric features with a global ideal ranking; "
            "it partially overlaps with the behavioural channel and is a secondary diagnostic.",
            "Hybrid uses fusion weights tuned on the validation presentation (artifacts/index/fusion_weights.json).",
        ],
        "index": index.manifest,
    }
    write_json(s.metrics_dir / "retrieval_benchmark.json", result)
    return result
