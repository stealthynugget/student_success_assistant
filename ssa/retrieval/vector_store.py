"""Vector store for the case index: LanceDB (embedded vector database) or in-memory NumPy arrays.

The historical-case index has two vector channels per enrolment:

* ``behavioural``: robust-scaled, outcome-weighted snapshot features (26 dims), searched by Euclidean distance;
* ``dense``: sentence embeddings of the case narrative (384 dims, unit length), searched by cosine (dot product).

With ``VECTOR_STORE=lancedb`` (default) both live in one LanceDB table (``vectors.lance`` inside the index directory)
next to the filterable metadata, and every search runs in the database: the metadata filter is pushed down as a
pre-filter (so filtering never drops true neighbours) and the k-nearest-neighbour search is exact (flat scan; at
~25k enrolments per checkpoint an approximate index would only cost recall). ``VECTOR_STORE=numpy`` keeps the
earlier ``.npy`` arrays and NumPy scan. Both backends return the same neighbours; ``tests/test_vector_store.py``
checks it, including every filter shape the API accepts.

Search results are cached per (channel, query vector, filter, limit), so tuning and benchmarks that re-rank the same
query under many fusion weightings pay for each vector search once.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

TABLE = "cases"
LANCE_DIR = "vectors.lance"
NPY = {"behavioural": "vectors.npy", "dense": "dense.npy"}
METRIC = {"behavioural": "l2", "dense": "dot"}
TIE_MARGIN = 64
TIE_DECIMALS = 5  # distances equal to 1e-5 are ties; float32 kernels differ by ~1e-7 between backends


def _ordered(docs: np.ndarray, dist: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Best first; ties (to TIE_DECIMALS) broken by index position, so every backend returns the same order."""
    order = np.lexsort((docs, np.round(dist, TIE_DECIMALS)))[:limit]
    return docs[order], dist[order]


class VectorStore:
    """Common interface. ``search`` returns (doc positions, distances) best first; distances are Euclidean for
    ``behavioural`` and 1 - cosine for ``dense``."""

    backend = "base"

    def __init__(self, cache_size: int = 4096):
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def has(self, channel: str) -> bool:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def search(self, channel: str, query: np.ndarray, flt: "Filter", limit: int) -> tuple[np.ndarray, np.ndarray]:
        key = (channel, np.asarray(query, np.float32).tobytes(), flt.sql, flt.key, int(limit))
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        res = self._search(channel, np.asarray(query, np.float32), flt, int(limit))
        with self._lock:
            self._cache[key] = res
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return res

    def _search(self, channel, query, flt, limit):
        raise NotImplementedError

    def vectors(self, channel: str, docs: list[int]) -> np.ndarray:
        raise NotImplementedError


class NumpyVectorStore(VectorStore):
    backend = "numpy"

    def __init__(self, arrays: dict[str, np.ndarray]):
        super().__init__()
        self.arrays = arrays

    @classmethod
    def load(cls, directory: Path) -> "NumpyVectorStore":
        return cls({ch: np.load(directory / f) for ch, f in NPY.items() if (directory / f).exists()})

    @staticmethod
    def write(directory: Path, arrays: dict[str, np.ndarray]) -> None:
        for ch, a in arrays.items():
            np.save(directory / NPY[ch], a.astype(np.float32))

    def has(self, channel):
        return channel in self.arrays

    def count(self):
        return len(self.arrays["behavioural"]) if "behavioural" in self.arrays else 0

    def _search(self, channel, query, flt, limit):
        idx = np.where(flt.mask)[0]
        A = self.arrays[channel][idx]
        dist = np.sqrt(((A - query) ** 2).sum(1)) if channel == "behavioural" else 1.0 - A @ query
        return _ordered(idx, dist, limit)

    def vectors(self, channel, docs):
        return self.arrays[channel][np.asarray(docs, dtype=int)]


class LanceVectorStore(VectorStore):
    backend = "lancedb"

    def __init__(self, directory: Path):
        super().__init__()
        import lancedb
        self.db = lancedb.connect(str(directory / LANCE_DIR))
        self.table = self.db.open_table(TABLE)
        self.channels = {f.name for f in self.table.schema if f.name in METRIC}

    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / LANCE_DIR).exists()

    @staticmethod
    def write(directory: Path, meta: pd.DataFrame, arrays: dict[str, np.ndarray]) -> None:
        """One row per indexed enrolment: position (doc_id), the filterable metadata and the vectors."""
        import lancedb
        import pyarrow as pa

        from ssa.retrieval.hybrid import FILTERABLE
        cols = {"doc_id": pa.array(np.arange(len(meta), dtype=np.int32)), "id_student": pa.array(meta["id_student"].astype("int64")),
                "pres_rank": pa.array(np.asarray([pres_ordinal(p) for p in meta["code_presentation"]], dtype=np.int64))}
        for c in sorted(FILTERABLE):
            if c not in meta:
                continue
            s = meta[c]
            if c in ("studied_credits", "num_of_prev_attempts"):
                cols[c] = pa.array(pd.to_numeric(s, errors="coerce").astype("float64"), from_pandas=True)
            else:
                cols[c] = pa.array([None if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v) for v in s], pa.string())
        for ch, a in arrays.items():
            a = np.ascontiguousarray(a, dtype=np.float32)
            cols[ch] = pa.FixedSizeListArray.from_arrays(pa.array(a.ravel(), pa.float32()), a.shape[1])
        db = lancedb.connect(str(directory / LANCE_DIR))
        table = db.create_table(TABLE, data=pa.table(cols), mode="overwrite")
        LanceVectorStore.index_filters(table)

    @staticmethod
    def index_filters(table) -> None:
        """Scalar indexes on the filter columns: the pre-filter then reads an index instead of scanning every row
        (measured at c40: filtered dense search 65 -> 16 ms). Bitmap for low-cardinality text, B-tree for numbers.
        No vector index: exact search over ~25k rows is fast enough and keeps recall at 100%."""
        import warnings
        names = set(table.schema.names)
        with warnings.catch_warnings():  # create_scalar_index is deprecated in favour of an API that mis-routes columns in 0.40
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.filterwarnings("ignore", message=".*create_scalar_index.*")
            for col in ("code_module", "code_presentation", "region", "highest_education", "age_band", "imd_band", "final_result"):
                if col in names:
                    table.create_scalar_index(col, index_type="BITMAP")
            for col in ("pres_rank", "id_student", "studied_credits", "num_of_prev_attempts", "doc_id"):
                if col in names:
                    table.create_scalar_index(col, index_type="BTREE")

    def has(self, channel):
        return channel in self.channels

    def count(self):
        return self.table.count_rows()

    def _search(self, channel, query, flt, limit):
        # fetch a margin past the cut so that tied distances at the boundary resolve exactly as in the NumPy scan
        q = (self.table.search(query, vector_column_name=channel).distance_type(METRIC[channel])
             .limit(limit + TIE_MARGIN).select(["doc_id", "_distance"]))
        if flt.sql:
            q = q.where(flt.sql, prefilter=True)
        res = q.to_arrow()
        docs = res["doc_id"].to_numpy().astype(int)
        dist = res["_distance"].to_numpy().astype(float)
        if channel == "behavioural":
            dist = np.sqrt(np.maximum(dist, 0.0))  # LanceDB reports squared L2; "dot" is already 1 - cosine
        return _ordered(docs, dist, limit)

    def vectors(self, channel, docs):
        if not len(docs):
            return np.zeros((0, 0), np.float32)
        ids = ", ".join(str(int(d)) for d in docs)
        res = self.table.search().where(f"doc_id IN ({ids})").select(["doc_id", channel]).limit(len(docs)).to_arrow()
        by_id = dict(zip(res["doc_id"].to_pylist(), res[channel].to_pylist(), strict=True))
        return np.asarray([by_id[int(d)] for d in docs], dtype=np.float32)


class Filter:
    """The allowed set of documents, as a boolean mask (NumPy backend) and as an SQL predicate (LanceDB).
    ``key`` identifies the mask for caching."""

    def __init__(self, mask: np.ndarray, sql: str | None):
        self.mask, self.sql = mask, sql
        self.key = None if sql is not None else np.packbits(mask).tobytes()


def _lit(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return "'" + str(v).replace("'", "''") + "'"


def pres_ordinal(code: str) -> int:
    """Presentation order as an integer for SQL: 2013B -> 20130, 2013J -> 20131 (same order as presentation_sort_key)."""
    from ssa.data.vocab import presentation_sort_key
    year, term = presentation_sort_key(code)
    return year * 10 + term


def filter_sql(filters: dict | None, exclude_student: int | None, before_presentation: str | None) -> str | None:
    """SQL twin of ``HistoricalCaseIndex._allowed`` (same semantics, including missing values: a missing value never
    equals, is never in a list and never meets a bound, but does count as "not equal")."""
    from ssa.retrieval.hybrid import FILTERABLE
    numeric = {"studied_credits", "num_of_prev_attempts"}
    parts = []
    if exclude_student is not None:
        parts.append(f"id_student != {int(exclude_student)}")
    if before_presentation:
        parts.append(f"pres_rank < {pres_ordinal(before_presentation)}")
    for key, cond in (filters or {}).items():
        if cond in (None, "", [], {}):
            continue
        if key not in FILTERABLE:
            raise ValueError(f"unsupported filter '{key}'. Allowed: {sorted(FILTERABLE)}")
        val = (lambda v: _lit(float(v))) if key in numeric else _lit

        def eq(v, key=key, val=val):
            return f"{key} = {val(v)}"

        if isinstance(cond, dict):
            if "eq" in cond:
                parts.append(eq(cond["eq"]))
            if "ne" in cond:
                parts.append(f"({key} IS NULL OR {key} != {val(cond['ne'])})")
            if "in" in cond:
                vals = list(cond["in"])
                parts.append(f"{key} IN ({', '.join(val(v) for v in vals)})" if vals else "FALSE")
            if "min" in cond:
                parts.append(f"{key} >= {_lit(float(cond['min']))}")
            if "max" in cond:
                parts.append(f"{key} <= {_lit(float(cond['max']))}")
        elif isinstance(cond, (list, tuple)):
            vals = list(cond)
            parts.append(f"{key} IN ({', '.join(val(v) for v in vals)})" if vals else "FALSE")
        else:
            parts.append(eq(cond))
    return " AND ".join(parts) if parts else ""
