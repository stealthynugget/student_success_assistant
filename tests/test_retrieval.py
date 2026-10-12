import pytest

from ssa.data.vocab import presentation_sort_key
from ssa.retrieval.descriptors import parse_query
from ssa.retrieval.hybrid import BM25, HistoricalCaseIndex
from ssa.utils import read_json


def test_bm25_ranks_matching_documents_first():
    bm = BM25([["a", "b"], ["c"], ["a", "a", "c"]])
    s = bm.scores(["a"])
    assert s[1] == 0 and s[0] > 0 and s[2] > 0


def test_query_parsing_maps_language_to_descriptors():
    toks = parse_query("students with declining engagement who missed assessments and have low scores in BBB")
    assert {"engagement_declining", "missed_most", "score_low", "module_bbb"} <= set(toks)


def test_index_is_isolated_complete_and_versioned(built, features):
    man = read_json(built.index_dir / "manifest.json")
    assert str(built.index_dir).startswith(str(built.home))
    assert man["n_docs"] == int((features["in_scope"] == 1).sum())


def test_historical_only_excludes_self_and_future(service, features):
    q = features[(features["in_scope"] == 1) & (features["code_presentation"] == "2014B")].iloc[0].to_dict()
    res = service.index.search(profile=q, filters={"code_module": q["code_module"]}, top_k=20)
    assert res["cases"]
    for c in res["cases"]:
        assert presentation_sort_key(c["code_presentation"]) < presentation_sort_key("2014B")
        assert c["id_student"] != q["id_student"]
        assert c["code_module"] == q["code_module"]
    assert res["outcome_summary"]["n_neighbours"] > 0


def test_range_and_membership_filters(service, features):
    res = service.index.search(text="low engagement and missed assessments",
                               filters={"studied_credits": {"min": 60, "max": 90}, "final_result": ["Fail", "Withdrawn"]}, top_k=15)
    assert res["cases"]
    for c in res["cases"]:
        assert 60 <= c["snapshot"]["studied_credits"] <= 90
        assert c["final_result"] in {"Fail", "Withdrawn"}
    with pytest.raises(ValueError):
        service.index.search(text="low scores", filters={"gender": "F"})


def test_benchmark_reports_random_baseline(built):
    b = read_json(built.metrics_dir / "retrieval_benchmark.json")
    assert "random" in b["modes"] and "hybrid" in b["modes"]
    assert b["modes"]["hybrid"]["neighbour_vote_auc_poor_outcome"] > b["modes"]["random"]["neighbour_vote_auc_poor_outcome"]


def test_index_manifest_mismatch_detected(built, tmp_path):
    import shutil
    d = tmp_path / "idx"
    shutil.copytree(built.index_dir, d)
    m = read_json(d / "manifest.json")
    m["index_version"] = "0.0"
    (d / "manifest.json").write_text(__import__("json").dumps(m))
    with pytest.raises(RuntimeError):
        HistoricalCaseIndex(built, index_dir=d).load()


def test_typed_search_benchmark_runs_and_beats_random(service):
    from ssa.evaluation.search_queries import text_search_queries, text_tuning_queries
    from ssa.retrieval.benchmark import run_text_benchmark
    qs = text_search_queries()
    assert len(qs) == 40 and all("descriptors" in q and "expected" in q for q in qs)
    assert not {q["query"] for q in qs} & {q["query"] for q in text_tuning_queries()}  # tuning never sees reported queries
    res = run_text_benchmark(service.bundle(None).index)
    m = res["parsers"]["rules"]["modes"]
    assert {"lexical", "hybrid", "random_corpus", "random_within_parsed_filters"} <= set(m)
    assert all(0 <= v["precision_at_10"] <= 1 for v in m.values())
    assert m["hybrid"]["precision_at_10"] > m["random_corpus"]["precision_at_10"]


def test_text_weights_apply_only_to_typed_queries(service, monkeypatch):
    idx = service.bundle(None).index
    tuned = {"weights": {"lexical": 0.5, "dense": 0.0}, "text_weights": {"lexical": 1.0, "dense": 2.0}}
    monkeypatch.setattr("ssa.retrieval.hybrid.read_json", lambda p, d=None: tuned if p.name == "fusion_weights.json" else (d or {}))
    assert idx.fusion_weights()["dense"] == 0.0
    assert idx.fusion_weights(text_only=True)["dense"] == 2.0


def test_text_weights_follow_the_parser(service, monkeypatch):
    idx = service.bundle(None).index
    tuned = {"weights": {"lexical": 0.5, "dense": 0.0}, "text_weights": {"lexical": 1.0, "dense": 2.0},
             "text_weights_by_parser": {"rules": {"lexical": 1.0, "dense": 2.0}, "llm": {"lexical": 1.0, "dense": 0.0}}}
    monkeypatch.setattr("ssa.retrieval.hybrid.read_json", lambda p, d=None: tuned if p.name == "fusion_weights.json" else (d or {}))
    assert idx.fusion_weights(text_only=True, parser="llm")["dense"] == 0.0
    assert idx.fusion_weights(text_only=True, parser="rules")["dense"] == 2.0
    assert idx.fusion_weights(text_only=True, parser="other")["dense"] == 2.0  # unknown parser: the default text weights
    assert idx.fusion_weights(parser="llm")["dense"] == 0.0 and idx.fusion_weights(parser="llm")["lexical"] == 0.5  # profiles unaffected


def test_queries_use_the_index_embedding_spec(service):
    idx = service.bundle(None).index
    old = dict(idx.manifest)
    try:
        idx.manifest = {**old, "embedding_model": "intfloat/e5-small-v2", "embedding_query_prefix": "query: ",
                        "embedding_passage_prefix": "passage: "}
        seen = []
        idx._encode = lambda texts: seen.extend(texts) or __import__("numpy").zeros((len(texts), 4))
        idx._qcache.clear()
        idx._encode_query("students who stopped submitting")
        idx._encode_query("a narrative", kind="passage")
        assert seen == ["query: students who stopped submitting", "passage: a narrative"]
        idx.manifest = {k: v for k, v in old.items() if not k.startswith("embedding")} | {"embedding_model": "all-MiniLM-L6-v2"}
        assert idx._embedding_spec() == {"model": "all-MiniLM-L6-v2", "query_prefix": "", "passage_prefix": ""}  # older index
    finally:
        idx.manifest = old
        del idx._encode
        idx._qcache.clear()


def test_dense_vectors_are_reused_for_unchanged_narratives(built):
    import numpy as np

    from ssa.retrieval.hybrid import HistoricalCaseIndex
    idx = HistoricalCaseIndex(built)
    calls = []
    idx._encode = lambda texts: (calls.append(list(texts)), np.ones((len(texts), 3), np.float32))[1]
    old = {1: ("same text", np.full(3, 7, np.float32)), 2: ("old text", np.full(3, 9, np.float32))}
    vecs, reused = idx._encode_corpus(np.array([1, 2, 3]), ["same text", "new text", "third"], "passage: ", old)
    assert reused == 1 and calls == [["passage: new text", "passage: third"]]
    assert (vecs[0] == 7).all() and (vecs[1] == 1).all() and (vecs[2] == 1).all()
    assert idx._reusable_dense({"model": "some-other-model", "query_prefix": "", "passage_prefix": ""}) == {}  # model changed
