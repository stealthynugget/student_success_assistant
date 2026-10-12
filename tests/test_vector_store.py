"""The vector database (LanceDB) must return exactly what the NumPy scan returns, for every filter shape the API accepts."""
import numpy as np
import pandas as pd
import pytest

from ssa.retrieval.vector_store import Filter, LanceVectorStore, NumpyVectorStore, filter_sql


@pytest.fixture(scope="module")
def stores(tmp_path_factory, built):
    from ssa.retrieval.hybrid import HistoricalCaseIndex
    idx = HistoricalCaseIndex(built).load()
    meta = idx.meta.copy()
    meta.loc[meta.index[:7], "imd_band"] = np.nan  # missing values must behave like the pandas mask
    rng = np.random.default_rng(0)
    beh = idx.store.vectors("behavioural", list(range(len(meta))))
    dense = rng.normal(size=(len(meta), 32)).astype(np.float32)
    dense[5] = dense[4]  # exact ties
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    arrays = {"behavioural": beh, "dense": dense}
    d = tmp_path_factory.mktemp("vs")
    LanceVectorStore.write(d, meta, arrays)
    idx.meta = meta
    return idx, NumpyVectorStore(arrays), LanceVectorStore(d)


FILTERS = [
    None,
    {"code_module": "AAA"},
    {"code_module": {"in": ["AAA", "BBB"]}, "studied_credits": {"min": 60, "max": 120}},
    {"num_of_prev_attempts": {"min": 1}},
    {"num_of_prev_attempts": {"max": 0}, "age_band": ["0-35"]},
    {"imd_band": {"ne": "0-10%"}},
    {"imd_band": "10-20"},
    {"region": {"eq": "Scotland"}, "highest_education": {"in": ["A Level or Equivalent", "HE Qualification"]}},
    {"final_result": "Withdrawn", "code_presentation": {"in": ["2013J", "2014B"]}},
    {"code_module": {"in": []}},
]


@pytest.mark.parametrize("filters", FILTERS)
@pytest.mark.parametrize("exclude,before", [(None, None), ("first", "2014J")])
def test_sql_filter_matches_pandas_mask(stores, filters, exclude, before):
    idx, _, lance = stores
    ex = int(idx.meta["id_student"].iloc[0]) if exclude == "first" else None
    mask = idx._allowed(filters, ex, before)
    sql = filter_sql(filters, ex, before)
    q = lance.table.search().select(["doc_id"]).limit(len(idx.meta) + 1)
    if sql:
        q = q.where(sql)
    got = set(q.to_arrow()["doc_id"].to_pylist())
    assert got == set(np.where(mask)[0].tolist())


@pytest.mark.parametrize("channel", ["behavioural", "dense"])
@pytest.mark.parametrize("filters", FILTERS[:6])
def test_same_neighbours_as_numpy(stores, channel, filters):
    idx, npy, lance = stores
    mask = idx._allowed(filters, None, None)
    for qi in (0, 4, 17):
        q = npy.arrays[channel][qi]
        a_docs, a_dist = npy.search(channel, q, Filter(mask, None), 50)
        b_docs, b_dist = lance.search(channel, q, Filter(mask, filter_sql(filters, None, None)), 50)
        assert a_docs.tolist() == b_docs.tolist()
        assert np.allclose(a_dist, b_dist, atol=1e-4)


def test_vectors_round_trip_and_cache(stores):
    _, npy, lance = stores
    docs = [3, 0, 11]
    assert np.allclose(lance.vectors("dense", docs), npy.arrays["dense"][docs])
    flt = Filter(np.ones(lance.count(), bool), "")
    first = lance.search("dense", npy.arrays["dense"][0], flt, 10)
    assert lance.search("dense", npy.arrays["dense"][0], flt, 10) is first  # cached


def test_index_uses_the_vector_database_by_default(built):
    from ssa.retrieval.hybrid import HistoricalCaseIndex
    idx = HistoricalCaseIndex(built).load()
    assert idx.store.backend == "lancedb" and idx.manifest["vector_store"] == "lancedb"
    assert not (built.index_dir / "vectors.npy").exists()


def test_migrate_numpy_index(tmp_path, built, monkeypatch):
    import shutil
    from ssa.retrieval.hybrid import HistoricalCaseIndex
    src = HistoricalCaseIndex(built).load()
    d = tmp_path / "index"
    shutil.copytree(built.index_dir, d, ignore=shutil.ignore_patterns("vectors.lance"))
    np.save(d / "vectors.npy", src.store.vectors("behavioural", list(range(len(src.meta)))))
    old = HistoricalCaseIndex(built, index_dir=d).load()
    assert old.store.backend == "numpy"
    profile = src.meta.iloc[0].to_dict()
    before = old.search(profile={**profile, "code_presentation": "2014J"}, top_k=5, mode="behavioural")
    res = old.migrate_vectors(remove_arrays=True)
    assert res["status"] == "migrated" and not (d / "vectors.npy").exists()
    new = HistoricalCaseIndex(built, index_dir=d).load()
    assert new.store.backend == "lancedb"
    after = new.search(profile={**profile, "code_presentation": "2014J"}, top_k=5, mode="behavioural")
    assert [c["case_id"] for c in before["cases"]] == [c["case_id"] for c in after["cases"]]
    assert pd.Series([c["behavioural_distance"] for c in after["cases"]]).notna().all()
