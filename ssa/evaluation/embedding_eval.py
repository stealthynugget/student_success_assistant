"""Embedding model evaluation for the dense (narrative) retrieval channel.

Each candidate sentence-embedding model encodes the same case narratives of one checkpoint index and is scored on:

* **typed search** (the dense channel's main job): the 40 labelled plain-English queries of ``search_queries``, dense
  alone and hybrid (lexical + dense, with the dense:lexical ratio tuned per model on the separate tuning queries,
  never on the reported ones), precision@10 and behaviour match@10, per query parser;
* **student-to-student precedents**: dense-only neighbours of real query students (test presentation, earlier
  presentations as corpus), neighbour-vote AUC for fail/withdraw and outcome agreement@10, as in the retrieval benchmark;
* **efficiency**: parameters, embedding size, index size, corpus encoding throughput and query latency (CPU).

Nothing is written to the production index: vectors are held in memory (NumPy store, same results as the database).
Results: ``artifacts/metrics/embedding_eval.json``, summarised in the model card.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ssa.config import Settings, get_settings
from ssa.data.vocab import presentation_sort_key
from ssa.utils import get_logger, write_json

log = get_logger("ssa.embedding_eval")


@dataclass(frozen=True)
class Candidate:
    name: str
    hf_id: str
    query_prefix: str = ""
    passage_prefix: str = ""
    note: str = ""


CANDIDATES = [
    Candidate("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", note="earlier default; 6 layers"),
    Candidate("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", note="larger general-purpose sentence model"),
    Candidate("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: ",
              note="retrieval-tuned; instruction on short queries"),
    Candidate("e5-small-v2", "intfloat/e5-small-v2", query_prefix="query: ", passage_prefix="passage: ",
              note="retrieval-tuned; query/passage prefixes; production default since this evaluation"),
    Candidate("gte-small", "thenlper/gte-small", note="retrieval-tuned"),
]


def _encoder(c: Candidate):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(c.hf_id, device="cpu")


def evaluate_model(c: Candidate, settings: Settings, df, n_profile_queries: int = 300, llm=None, k: int = 10) -> dict:
    from ssa.evaluation.search_queries import text_search_queries, text_tuning_queries
    from ssa.retrieval.benchmark import TEXT_DENSE_GRID, _evaluate, _parsers, _score_text_queries
    from ssa.retrieval.hybrid import HistoricalCaseIndex
    from ssa.retrieval.vector_store import NumpyVectorStore

    idx = HistoricalCaseIndex(settings).load()
    texts = idx.meta["narrative"].fillna("").astype(str).tolist()
    model = _encoder(c)
    params = int(sum(p.numel() for p in model.parameters()))

    t0 = time.time()
    D = np.asarray(model.encode([c.passage_prefix + t for t in texts], batch_size=256, normalize_embeddings=True,
                                show_progress_bar=False), dtype=np.float32)
    encode_s = time.time() - t0
    beh = idx.store.vectors("behavioural", list(range(len(texts))))
    idx.store = NumpyVectorStore({"behavioural": beh, "dense": D})

    def use_prefix(prefix: str):
        cache: dict[str, np.ndarray] = {}

        def enc(text: str, kind: str | None = None) -> np.ndarray:  # this evaluation sets the prefix itself
            if text not in cache:
                cache[text] = np.asarray(model.encode([prefix + text], normalize_embeddings=True, show_progress_bar=False)[0], np.float32)
            return cache[text]
        idx._encode_query = enc  # instance attribute shadows the method: this model, this prefix

    out = {"model": c.name, "hf_id": c.hf_id, "note": c.note, "parameters_millions": round(params / 1e6, 1),
           "dimensions": int(D.shape[1]), "index_mb": round(D.nbytes / 2**20, 1), "corpus_docs": len(texts),
           "encode_seconds": round(encode_s, 1), "encode_docs_per_second": round(len(texts) / encode_s, 1),
           "query_prefix": c.query_prefix or None, "passage_prefix": c.passage_prefix or None, "typed_search": {}}

    # latency of one fresh typed query: encode + filtered exact search
    use_prefix(c.query_prefix)
    lat = []
    for q in text_search_queries()[:20]:
        t = time.time()
        idx.search(text=q["query"], top_k=k, mode="dense")  # each query is encoded for the first time here
        lat.append(time.time() - t)
    out["query_latency_ms"] = round(1000 * float(np.median(lat)), 1)

    for parser, parse in _parsers(llm).items():
        use_prefix(c.query_prefix)
        grid = {f"w{i}": ("hybrid", {"lexical": 1.0, "dense": dw}) for i, dw in enumerate(TEXT_DENSE_GRID)}
        tune = _score_text_queries(idx, text_tuning_queries(), parse, grid, k)["modes"]
        best = max(range(len(TEXT_DENSE_GRID)), key=lambda i: (tune[f"w{i}"][f"precision_at_{k}"], -TEXT_DENSE_GRID[i]))
        weights = {"lexical": 1.0, "dense": TEXT_DENSE_GRID[best]}
        rep = _score_text_queries(idx, text_search_queries(), parse,
                                  {"dense": ("dense", None), "hybrid": ("hybrid", weights)}, k)["modes"]
        out["typed_search"][parser] = {
            "dense_precision_at_10": rep["dense"][f"precision_at_{k}"],
            "dense_behaviour_match_at_10": rep["dense"][f"behaviour_match_at_{k}"],
            "hybrid_precision_at_10": rep["hybrid"][f"precision_at_{k}"],
            "hybrid_dense_weight_tuned": weights["dense"]}

    use_prefix(c.passage_prefix)  # profile -> case is passage-to-passage similarity
    pres = sorted(df.loc[df["in_scope"] == 1, "code_presentation"].unique(), key=presentation_sort_key)
    prof, _, _ = _evaluate(df, idx, pres[-1], [("dense", "dense", None)], n_profile_queries, k, seed=42)
    out["profile_dense"] = {"vote_auc_poor_outcome": prof["dense"]["neighbour_vote_auc_poor_outcome"],
                            "vote_auc_withdrawal": prof["dense"]["neighbour_vote_auc_withdrawal"],
                            "outcome_agreement_at_10": prof["dense"][f"outcome_agreement_at_{k}"]}
    log.info(f"{c.name}: {out}")
    return out


def run(settings: Settings | None = None, names: list[str] | None = None, llm=None, n_profile_queries: int = 300) -> dict:
    from ssa.data.pipeline import load_features
    s = settings or get_settings()
    df = load_features(s)
    cands = [c for c in CANDIDATES if not names or c.name in names]
    rows = []
    for c in cands:
        try:
            rows.append(evaluate_model(c, s, df, n_profile_queries=n_profile_queries, llm=llm))
        except Exception as exc:  # a model that cannot be downloaded or run is reported, not fatal
            log.warning(f"{c.name} failed: {exc}")
            rows.append({"model": c.name, "hf_id": c.hf_id, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    result = {"checkpoint": s.cutoff_fraction, "device": "cpu", "k": 10,
              "typed_queries": "40 labelled queries (text_search_queries); hybrid ratio tuned per model on the 40 tuning queries",
              "profile_queries": f"{n_profile_queries} in-scope students of the test presentation, earlier presentations as corpus",
              "current_model": get_settings().embedding_model_name, "models": rows}
    write_json(s.metrics_dir / "embedding_eval.json", result)
    return result
