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

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ssa.config import Settings, get_settings
from ssa.data.vocab import presentation_sort_key
from ssa.models.features import NUMERIC
from ssa.retrieval.hybrid import MODES, HistoricalCaseIndex
from ssa.utils import read_json, write_json


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


# --------------------------------------------------------------------------- typed (free-text) search
TEXT_MODES = ("lexical", "dense", "hybrid")
TEXT_DENSE_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)  # dense weight relative to lexical = 1 (RRF order depends only on the ratio)


def _parsers(llm=None) -> dict:
    from ssa.agents.genai import rule_parse, understand_query
    out = {"rules": rule_parse}
    if llm is not None:
        out["llm"] = lambda q: understand_query(llm, q)
    return out


def _text_relevance(index: HistoricalCaseIndex, q: dict, token_sets: list[set]) -> tuple[np.ndarray, np.ndarray]:
    """Per corpus document: (meets every expected filter, shows every expected behaviour)."""
    filt_ok = index._allowed(q["expected"], None, None)
    desc_ok = np.ones(len(token_sets), dtype=bool)
    for group in q.get("descriptors") or []:
        g = set(group)
        desc_ok &= np.array([bool(ts & g) for ts in token_sets])
    return filt_ok, desc_ok


def _score_text_queries(index: HistoricalCaseIndex, queries: list[dict], parse, modes: dict, k: int) -> dict:
    """modes: name -> (mode, weights|None). Returns per-mode metrics plus the two exact random baselines."""
    index._ensure()
    token_sets = [set(t) for t in index.tokens]
    key_pos = {int(ck): i for i, ck in enumerate(index.meta["enrollment_key"].to_numpy())}
    acc = {m: {"p": [], "f": [], "d": [], "empty": 0} for m in modes}
    acc_kind: dict[str, dict[str, list]] = {}
    base_corpus, base_filtered, parsed_exact = [], [], 0
    for q in queries:
        interp = parse(q["query"])
        pf, pd_ = interp["filters"] or None, interp.get("descriptors") or None
        filt_ok, desc_ok = _text_relevance(index, q, token_sets)
        rel = filt_ok & desc_ok
        allowed = index._allowed(pf, None, None)
        base_corpus.append(float(rel.mean()))
        base_filtered.append(float(rel[allowed].mean()) if allowed.any() else 0.0)
        parsed_exact += int(interp["filters"] == q["expected"])
        for name, (mode, weights) in modes.items():
            try:
                res = index.search(text=q["query"], filters=pf, top_k=k, mode=mode, weights=weights, extra_tokens=pd_)
                docs = np.array([key_pos[c["case_id"]] for c in res["cases"]], dtype=int)
            except ValueError:
                docs = np.array([], dtype=int)
            if len(docs) == 0:
                acc[name]["empty"] += 1
                p = f = d = 0.0
            else:
                p, f, d = float(rel[docs].mean()), float(filt_ok[docs].mean()), float(desc_ok[docs].mean())
            acc[name]["p"].append(p)
            acc[name]["f"].append(f)
            if q.get("descriptors"):
                acc[name]["d"].append(d)
            acc_kind.setdefault(q.get("kind", "other"), {}).setdefault(name, []).append(p)
    out = {name: {f"precision_at_{k}": float(np.mean(a["p"])), f"filter_match_at_{k}": float(np.mean(a["f"])),
                  f"behaviour_match_at_{k}": float(np.mean(a["d"])) if a["d"] else None, "empty_results": a["empty"]}
           for name, a in acc.items()}
    out["random_corpus"] = {f"precision_at_{k}": float(np.mean(base_corpus))}
    out["random_within_parsed_filters"] = {f"precision_at_{k}": float(np.mean(base_filtered))}
    by_kind = {kind: {name: float(np.mean(v)) for name, v in d.items()} for kind, d in acc_kind.items()}
    return {"modes": out, "precision_by_query_kind": by_kind, "parser_exact_filter_match": parsed_exact / max(len(queries), 1)}


def tune_text_weights(index: HistoricalCaseIndex, settings: Settings | None = None, k: int = 10) -> dict:
    """Choose the dense:lexical fusion ratio for typed queries on a separate templated tuning set (rule parser).
    Stored next to the profile weights; adopted only if it beats lexical alone on the tuning set."""
    from ssa.evaluation.search_queries import text_tuning_queries
    s = settings or get_settings()
    index._ensure()
    if index.dense is None:
        return {"status": "skipped", "reason": "dense channel not built"}
    grid = [{"lexical": 1.0, "dense": dw} for dw in TEXT_DENSE_GRID]
    modes = {f"w{i}": ("hybrid", w) for i, w in enumerate(grid)}
    res = _score_text_queries(index, text_tuning_queries(), _parsers()["rules"], modes, k)["modes"]
    scored = sorted(((res[f"w{i}"][f"precision_at_{k}"], -w["dense"], w) for i, w in enumerate(grid)), key=lambda x: (-x[0], -x[1]))
    best_p, _, best = scored[0]
    lexical_only = res["w0"][f"precision_at_{k}"]
    path = s.index_dir / "fusion_weights.json"
    doc = read_json(path, {})
    doc["text_weights"] = best
    doc["text_tuning"] = {"queries": "text_tuning_queries(): 30 templated (seed 101) + 10 tuning-only paraphrases; disjoint from the reported set",
                          "objective": f"precision@{k}", "parser": "rules", "lexical_only": lexical_only, "best": best_p,
                          "grid": [{"weights": w, f"precision_at_{k}": p} for p, _, w in scored]}
    write_json(path, doc)
    return {"status": "tuned", "weights": best, "precision": best_p, "lexical_only": lexical_only}


def run_text_benchmark(index: HistoricalCaseIndex, settings: Settings | None = None, llm=None, k: int = 10) -> dict:
    """Typed-search benchmark on the ~40 labelled plain-English queries, per parser (rules, and LLM when given)."""
    from ssa.evaluation.search_queries import text_search_queries
    index._ensure()
    queries = text_search_queries()
    modes = {"lexical": ("lexical", None), "hybrid": ("hybrid", None)}
    if index.dense is not None:
        modes["dense"] = ("dense", None)
    out = {"queries": len(queries), "k": k, "corpus": "all indexed enrolments (typed search has no query student)",
           "relevance": "a returned case is relevant when it meets every expected filter and shows every expected behaviour",
           "hybrid_text_weights": index.fusion_weights(text_only=True),
           "notes": ["random_corpus = a random case from the whole corpus; random_within_parsed_filters = a random case among "
                     "those passing the parser's filters (isolates what the ranking adds beyond filtering)."],
           "parsers": {}}
    for name, parse in _parsers(llm).items():
        t0 = time.time()
        out["parsers"][name] = {**_score_text_queries(index, queries, parse, modes, k), "seconds": round(time.time() - t0, 1)}
    return out
