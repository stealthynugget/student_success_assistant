"""Hybrid historical-case retrieval.

Channels
--------
1. **behavioural** (primary "semantic" similarity for profiles): exact weighted-Euclidean kNN
   over robust-scaled snapshot features. Feature weights are *supervised*: proportional to the
   withdrawal model's mean |attribution| per feature, so "similar" means similar in the ways
   that matter for outcomes.
2. **lexical**: BM25 (sparse, vectorised) over discretised descriptor tokens - exact keyword
   matching that is actually meaningful ("missed_most", "engagement_declining").
3. **dense** (optional): sentence-embedding similarity over qualitative narratives, enabling
   free-text queries such as "students who stopped logging in after a failed assignment".

Fusion is weighted Reciprocal Rank Fusion; weights can be learned from advisor feedback
(``artifacts/feedback/retrieval_weights.json``).

Storage: both vector channels live in a vector database (LanceDB, embedded, one table per checkpoint index with the
filterable metadata; ``VECTOR_STORE=numpy`` keeps plain arrays). Searches run there with the metadata filter pushed
down as a pre-filter and exact kNN; the lexical channel (BM25) stays in memory. See ``ssa/retrieval/vector_store.py``.

Guards
------
* ``historical_only`` (default): when the query belongs to presentation P, only presentations
  that finished before P are searchable - the current cohort's outcomes are unknown and later
  cohorts are the future.
* The query student's own enrolments are always excluded.
* Index is built in a temp directory and swapped atomically; a manifest records version, size
  and checksum. Nothing in the test-suite can touch a production index (paths are isolated).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ssa.config import Settings, get_settings
from ssa.data.vocab import presentation_sort_key
from ssa.evaluation.metrics import wilson_interval
from ssa.retrieval.descriptors import build_corpus_text, descriptor_tokens, narrative, parse_query
from ssa.retrieval.vector_store import Filter as VectorFilter
from ssa.retrieval.vector_store import LanceVectorStore, NumpyVectorStore, VectorStore, filter_sql
from ssa.utils import get_logger, read_json, write_json

log = get_logger("ssa.retrieval")
INDEX_VERSION = "5.0"
_LOAD_LOCK = __import__("threading").RLock()

BEHAVIOURAL_FEATURES = [
    "score_pct", "weighted_score", "min_score", "score_slope", "missed_rate", "late_rate", "mean_days_early",
    "no_scores_yet", "clicks_pct", "active_days_pct", "log_total_clicks", "days_since_last_activity",
    "engagement_slope_rel", "engagement_momentum", "engagement_drop_ratio", "longest_inactive_streak",
    "inactive_weeks_last3", "weeks_active", "share_forum", "share_quiz", "share_content", "pre_start_clicks",
    "num_of_prev_attempts", "studied_credits", "registration_days_before_start", "late_registration",
]
META_COLUMNS = [
    "enrollment_key", "id_student", "code_module", "code_presentation", "region", "highest_education", "age_band",
    "imd_band", "studied_credits", "num_of_prev_attempts", "final_result", "withdrawn", "poor_outcome",
    "weighted_score", "missed_rate", "clicks_pct", "score_pct", "engagement_momentum", "days_since_last_activity",
]
FILTERABLE = {"code_module", "code_presentation", "region", "highest_education", "age_band", "imd_band",
              "studied_credits", "num_of_prev_attempts", "final_result"}
DEFAULT_WEIGHTS = {"behavioural": 1.0, "lexical": 0.5, "dense": 0.5}
MODES = ("hybrid", "behavioural", "lexical", "dense")
LANCE_DIR_NAME = "vectors.lance"


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
        rows, cols, vals = [], [], []
        for i, d in enumerate(docs):
            for t, c in pd.Series(d).value_counts().items():
                rows.append(i), cols.append(self.vocab[t]), vals.append(float(c))
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)))
        dl = np.asarray(tf.sum(1)).ravel()
        avgdl = dl.mean() if len(dl) else 1.0
        df = np.asarray((tf > 0).sum(0)).ravel()
        n = len(docs)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)
        weights = tf.data * (k1 + 1) / denom * self.idf[tf.col]
        self.matrix = sp.csr_matrix((weights, (tf.row, tf.col)), shape=(n, len(self.vocab)))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        idx = [self.vocab[t] for t in set(query_tokens) if t in self.vocab]
        if not idx:
            return np.zeros(self.matrix.shape[0])
        return np.asarray(self.matrix[:, idx].sum(1)).ravel()


class HistoricalCaseIndex:
    def __init__(self, settings: Settings | None = None, index_dir: Path | None = None):
        self.settings = settings or get_settings()
        self.dir = Path(index_dir or self.settings.index_dir)
        self.meta: pd.DataFrame | None = None
        self.store: VectorStore | None = None
        self.bm25: BM25 | None = None
        self.tokens: list[list[str]] = []
        self.scaler: dict = {}
        self.manifest: dict = {}
        self._encoder = None
        self._qcache: OrderedDict[str, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------ build
    @staticmethod
    def feature_weights(importance: dict | None) -> np.ndarray:
        if not importance:
            return np.ones(len(BEHAVIOURAL_FEATURES))
        raw = np.array([float(importance.get(f, 0.0)) for f in BEHAVIOURAL_FEATURES])
        if raw.sum() <= 0:
            return np.ones(len(BEHAVIOURAL_FEATURES))
        w = np.sqrt(raw / raw.max())  # temper dominance of a single feature
        return np.maximum(w, 0.1)

    def build(self, df: pd.DataFrame, importance: dict | None = None, with_dense: bool | None = None) -> dict:
        t0 = time.time()
        corpus = df[(df["in_scope"] == 1) & df["final_result"].notna()].reset_index(drop=True)
        X = corpus[BEHAVIOURAL_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        med = np.nanmedian(X, 0)
        q75, q25 = np.nanpercentile(X, 75, 0), np.nanpercentile(X, 25, 0)
        iqr = np.where((q75 - q25) > 1e-9, q75 - q25, np.nanstd(X, 0) + 1e-9)
        w = self.feature_weights(importance)
        Z = self._scale(X, med, iqr, w)
        tokens, texts = build_corpus_text(corpus)

        dense = None
        use_dense = self.settings.enable_text_embeddings if with_dense is None else with_dense
        dense_status = "disabled"
        if use_dense:
            try:
                dense = self._encode(texts)
                dense_status = "built"
            except Exception as exc:  # sentence-transformers optional
                dense_status = f"unavailable: {type(exc).__name__}"
                log.warning(f"dense channel unavailable: {exc}")

        tmp = self.dir.parent / f".{self.dir.name}.building-{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        meta = corpus[[c for c in META_COLUMNS if c in corpus]].copy()
        arrays = {"behavioural": Z.astype(np.float32), **({"dense": dense.astype(np.float32)} if dense is not None else {})}
        backend = self._write_vectors(tmp, meta, arrays)
        meta["tokens"] = [" ".join(t) for t in tokens]
        meta["narrative"] = texts
        meta.to_csv(tmp / "meta.csv.gz", index=False, compression="gzip")
        scaler = {"median": med.tolist(), "iqr": iqr.tolist(), "weights": w.tolist(), "features": BEHAVIOURAL_FEATURES}
        write_json(tmp / "scaler.json", scaler)
        digest = hashlib.sha256(Z.astype(np.float32).tobytes()).hexdigest()[:16]
        manifest = {"index_version": INDEX_VERSION, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_docs": int(len(meta)),
                    "presentations": sorted(meta["code_presentation"].unique().tolist(), key=presentation_sort_key),
                    "dense_channel": dense_status, "embedding_model": self.settings.embedding_model_name if dense is not None else None,
                    "vector_checksum": digest, "vector_store": backend, "supervised_feature_weights": bool(importance),
                    "build_seconds": round(time.time() - t0, 1)}
        write_json(tmp / "manifest.json", manifest)
        # atomic swap
        old = self.dir.parent / f".{self.dir.name}.old-{os.getpid()}"
        if self.dir.exists():
            os.replace(self.dir, old)
        os.replace(tmp, self.dir)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        self.load()
        log.info(f"index built: {manifest['n_docs']:,} docs, dense={dense_status}")
        return manifest

    def _write_vectors(self, directory: Path, meta: pd.DataFrame, arrays: dict[str, np.ndarray]) -> str:
        if self.settings.vector_store == "lancedb":
            LanceVectorStore.write(directory, meta, arrays)
            return "lancedb"
        NumpyVectorStore.write(directory, arrays)
        return "numpy"

    def migrate_vectors(self, remove_arrays: bool = False) -> dict:
        """Move an index built with ``.npy`` arrays into the vector database without re-encoding anything."""
        if LanceVectorStore.exists(self.dir):
            return {"status": "already in the vector database", "dir": str(self.dir)}
        arrays = NumpyVectorStore.load(self.dir).arrays
        if not arrays:
            raise FileNotFoundError(f"no vector arrays to migrate in {self.dir}")
        meta = pd.read_csv(self.dir / "meta.csv.gz", low_memory=False)
        tmp = self.dir / f".{LANCE_DIR_NAME}.tmp-{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        LanceVectorStore.write(tmp, meta, arrays)
        os.replace(tmp / LANCE_DIR_NAME, self.dir / LANCE_DIR_NAME)
        shutil.rmtree(tmp, ignore_errors=True)
        man = read_json(self.dir / "manifest.json")
        man["vector_store"] = "lancedb"
        write_json(self.dir / "manifest.json", man)
        if remove_arrays:
            for f in ("vectors.npy", "dense.npy"):
                (self.dir / f).unlink(missing_ok=True)
        self.meta = None  # reload on next use
        return {"status": "migrated", "dir": str(self.dir), "channels": sorted(arrays), "rows": int(len(meta))}

    @staticmethod
    def _scale(X, med, iqr, w):
        Z = (X - med) / iqr
        Z = np.where(np.isnan(Z), 0.0, Z)  # missing -> cohort median (no_scores_yet flag carries the info)
        Z = np.clip(Z, -5, 5)
        return Z * w

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.settings.embedding_model_name)
        return np.asarray(self._encoder.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False))

    def _encode_query(self, text: str) -> np.ndarray:
        """Query embeddings are cached: tuning and benchmarks re-rank the same query under many fusion weightings."""
        with _LOAD_LOCK:
            hit = self._qcache.get(text)
            if hit is not None:
                self._qcache.move_to_end(text)
                return hit
        v = self._encode([text])[0]
        with _LOAD_LOCK:
            self._qcache[text] = v
            while len(self._qcache) > 2048:
                self._qcache.popitem(last=False)
        return v

    # ------------------------------------------------------------------- load
    def load(self):
        """Load atomically: all attributes are assigned together under a lock, so concurrent
        requests never observe a half-loaded index."""
        with _LOAD_LOCK:
            man = self.dir / "manifest.json"
            if not man.exists():
                raise FileNotFoundError(f"retrieval index missing at {self.dir}; run `python -m ssa.cli index`")
            manifest = read_json(man)
            if manifest.get("index_version") != INDEX_VERSION:
                raise RuntimeError(f"index version {manifest.get('index_version')} != {INDEX_VERSION}; rebuild the index")
            meta = pd.read_csv(self.dir / "meta.csv.gz", low_memory=False)
            store = self._open_store()
            if len(meta) != manifest["n_docs"] or store.count() != len(meta):
                raise RuntimeError("index files are inconsistent with manifest")
            tokens = [str(t).split() for t in meta["tokens"].fillna("")]
            self.manifest, self.store, self.scaler, self.tokens = manifest, store, read_json(self.dir / "scaler.json"), tokens
            self.bm25 = BM25(tokens)
            self._pres_rank = meta["code_presentation"].map(lambda p: presentation_sort_key(p)).tolist()
            self.meta = meta  # assigned last: readiness flag for _ensure()
        return self

    def _open_store(self) -> VectorStore:
        want_lance = self.settings.vector_store == "lancedb"
        if LanceVectorStore.exists(self.dir) and (want_lance or not (self.dir / "vectors.npy").exists()):
            return LanceVectorStore(self.dir)
        if (self.dir / "vectors.npy").exists():
            if want_lance:
                log.warning(f"index at {self.dir} still uses .npy arrays; move it into the vector database with "
                            "`python -m ssa.cli index --migrate-vectors`")
            return NumpyVectorStore.load(self.dir)
        raise FileNotFoundError(f"no vectors in {self.dir}; run `python -m ssa.cli index`")

    @property
    def has_dense(self) -> bool:
        self._ensure()
        return self.store.has("dense")

    @property
    def ready(self) -> bool:
        return self.meta is not None or (self.dir / "manifest.json").exists()

    def _ensure(self):
        if self.meta is None:
            self.load()

    # ----------------------------------------------------------------- search
    def _allowed(self, filters: dict | None, exclude_student: int | None, before_presentation: str | None) -> np.ndarray:
        m = np.ones(len(self.meta), dtype=bool)
        if exclude_student is not None:
            m &= self.meta["id_student"].to_numpy() != int(exclude_student)
        if before_presentation:
            cutoff = presentation_sort_key(before_presentation)
            m &= np.array([pr < cutoff for pr in self._pres_rank])
        for key, cond in (filters or {}).items():
            if cond in (None, "", [], {}):
                continue
            if key not in FILTERABLE:
                raise ValueError(f"unsupported filter '{key}'. Allowed: {sorted(FILTERABLE)}")
            col = self.meta[key]
            if isinstance(cond, dict):
                if "eq" in cond:
                    m &= (col == cond["eq"]).to_numpy()
                if "ne" in cond:
                    m &= (col != cond["ne"]).to_numpy()
                if "in" in cond:
                    m &= col.isin(list(cond["in"])).to_numpy()
                if "min" in cond:
                    m &= (pd.to_numeric(col, errors="coerce") >= float(cond["min"])).to_numpy()
                if "max" in cond:
                    m &= (pd.to_numeric(col, errors="coerce") <= float(cond["max"])).to_numpy()
            elif isinstance(cond, (list, tuple)):
                m &= col.isin(list(cond)).to_numpy()
            else:
                m &= (col == cond).to_numpy()
        return m

    def query_vector(self, profile: dict) -> np.ndarray:
        X = np.array([[pd.to_numeric(profile.get(f), errors="coerce") for f in BEHAVIOURAL_FEATURES]], dtype=float)
        return self._scale(X, np.array(self.scaler["median"]), np.array(self.scaler["iqr"]), np.array(self.scaler["weights"]))[0]

    def search(self, *, profile: dict | None = None, text: str | None = None, filters: dict | None = None,
               top_k: int = 10, mode: str = "hybrid", historical_only: bool = True, weights: dict | None = None,
               rrf_k: int = 60, candidate_pool: int = 200, outcome_neighbourhood: int = 50,
               extra_tokens: list[str] | None = None, parser: str | None = None) -> dict:
        self._ensure()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if profile is None and not text:
            raise ValueError("provide a student profile or a text query")
        exclude = int(profile["id_student"]) if profile and profile.get("id_student") is not None else None
        before = str(profile.get("code_presentation")) if (profile and historical_only and profile.get("code_presentation")) else None
        allowed = self._allowed(filters, exclude, before)
        idx = np.where(allowed)[0]
        if len(idx) == 0:
            return {"cases": [], "outcome_summary": None, "channels_used": [], "candidates": 0}
        w = self.fusion_weights(text_only=profile is None, parser=parser)
        w.update(weights or {})
        flt = VectorFilter(allowed, filter_sql(filters, exclude, before)
                           if self.store.backend == "lancedb" else None)

        ranks: dict[str, np.ndarray] = {}
        qv, dist_of = None, {}
        if profile is not None and mode in ("hybrid", "behavioural"):
            qv = self.query_vector(profile)
            docs, dist = self.store.search("behavioural", qv, flt, candidate_pool)
            ranks["behavioural"] = docs
            dist_of = {int(d): float(x) for d, x in zip(docs, dist, strict=True)}
        q_tokens = []
        if mode in ("hybrid", "lexical"):
            q_tokens = (descriptor_tokens(profile, include_outcome=False) if profile is not None else []) + (parse_query(text) if text else [])
            q_tokens = list(dict.fromkeys(q_tokens + list(extra_tokens or [])))
            if q_tokens:
                sc = self.bm25.scores(q_tokens)[idx]
                ranks["lexical"] = idx[np.argsort(-sc, kind="stable")]
        if mode in ("hybrid", "dense") and self.store.has("dense") and (text or profile is not None):
            q_text = (text or "") + (" " + narrative(profile, include_outcome=False) if profile is not None else "")
            try:
                qe = self._encode_query(q_text.strip())
                ranks["dense"] = self.store.search("dense", qe, flt, candidate_pool)[0]
            except Exception as exc:
                log.warning(f"dense query failed: {exc}")
        if not ranks:
            raise ValueError(f"mode '{mode}' is unavailable for this query (dense channel: {self.manifest.get('dense_channel')})")

        fused: dict[int, float] = {}
        per_channel_rank: dict[int, dict] = {}
        for ch, order in ranks.items():
            for r, doc in enumerate(order[:candidate_pool]):
                fused[doc] = fused.get(doc, 0.0) + w.get(ch, 0.0) / (rrf_k + r + 1)
                per_channel_rank.setdefault(doc, {})[ch] = r + 1
        ordered = sorted(fused, key=lambda d: -fused[d])
        top = ordered[:top_k]
        q_set = set(q_tokens)
        missing = [int(d) for d in top if qv is not None and int(d) not in dist_of]
        if missing:  # a fused top case outside the behavioural candidates: compute its distance directly
            V = self.store.vectors("behavioural", missing)
            dist_of.update({d: float(np.sqrt(((v - qv) ** 2).sum())) for d, v in zip(missing, V, strict=True)})
        cases = []
        for d in top:
            row = self.meta.iloc[d]
            shared = [t for t in self.tokens[d] if t in q_set and not t.startswith("outcome_")]
            cases.append({
                "case_id": int(row["enrollment_key"]), "id_student": int(row["id_student"]),
                "code_module": row["code_module"], "code_presentation": row["code_presentation"],
                "final_result": row["final_result"], "fusion_score": round(fused[d], 6),
                "behavioural_distance": round(dist_of[d], 4) if d in dist_of else None,
                "channel_ranks": per_channel_rank.get(d, {}), "shared_descriptors": shared[:8],
                "summary": row["narrative"],
                "snapshot": {k: (None if pd.isna(row.get(k)) else (float(row[k]) if isinstance(row[k], (int, float, np.number)) else row[k]))
                             for k in ["weighted_score", "missed_rate", "clicks_pct", "score_pct", "engagement_momentum",
                                       "days_since_last_activity", "studied_credits", "num_of_prev_attempts"]},
                "metadata": {k: row[k] for k in ["region", "highest_education", "age_band"] if k in row},
            })
        neighbourhood = ordered[:outcome_neighbourhood]
        summary = self.outcome_summary(neighbourhood)
        return {"cases": cases, "outcome_summary": summary, "channels_used": list(ranks), "fusion_weights": w,
                "candidates": int(len(idx)), "historical_only_before": before, "query_descriptors": q_tokens}

    def fusion_weights(self, text_only: bool = False, parser: str | None = None) -> dict:
        """Precedence: advisor-feedback-learned weights > validation-tuned weights > defaults. Typed queries (no query
        student) use their own tuned ``text_weights`` when present: only the lexical and dense channels rank them.
        ``parser`` ("rules" or "llm": who interpreted the typed query) selects that parser's tuned weights if any."""
        w = dict(DEFAULT_WEIGHTS)
        tuned = read_json(self.dir / "fusion_weights.json", {})
        w.update(tuned.get("weights", {}))
        if text_only:
            w.update(tuned.get("text_weights", {}))
            if parser:
                w.update(tuned.get("text_weights_by_parser", {}).get(parser, {}))
        w.update(read_json(self.settings.retrieval_weights_path, {}).get("weights", {}))
        return w

    def outcome_summary(self, docs: list[int]) -> dict | None:
        if not docs:
            return None
        res = self.meta.iloc[docs]["final_result"].value_counts()
        n = len(docs)
        out = {"n_neighbours": n, "counts": {k: int(v) for k, v in res.items()}}
        for label, cols in [("withdrawn", ["Withdrawn"]), ("fail_or_withdrawn", ["Fail", "Withdrawn"])]:
            k = int(sum(res.get(c, 0) for c in cols))
            lo, hi = wilson_interval(k, n)
            out[f"{label}_rate"] = round(k / n, 4)
            out[f"{label}_ci95"] = [round(lo, 4), round(hi, 4)]
        return out

    def stats(self) -> dict:
        self._ensure()
        return dict(self.manifest)


def manifest_summary(settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    return read_json(s.index_dir / "manifest.json", {"status": "missing"})


def dumps(obj) -> str:  # pragma: no cover - debugging helper
    return json.dumps(obj, indent=2, default=str)
