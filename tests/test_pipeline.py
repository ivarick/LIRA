import json
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd

import lira_retrieval
import lira_pipeline
from lira import cases as case_module
from lira import core as core_module
from lira import statutes as statute_module


class DummyCountModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame):
        return [self.value for _ in range(len(frame))]


class DummySparseIndex:
    def __init__(self, doc_ids):
        self.doc_ids = list(doc_ids)

    def search(self, query_weights, top_k):
        return []


class DummyBM25:
    def search(self, text, top_k=50):
        return []


def make_minimal_resources():
    return SimpleNamespace(
        config=lira_pipeline.V44Config(
            laws_dense_k=1,
            laws_sparse_k=1,
            laws_bm25_k=1,
            laws_shortlist_k=1,
            laws_rerank_k=1,
            case_profile_shortlist_k=5,
            case_profile_rerank_k=3,
            case_dense_k=1,
            case_sparse_k=1,
            case_bm25_k=1,
            case_backproj_k=1,
        ),
        runtime_cache={},
        runtime_diagnostics=defaultdict(list),
        citation_universe=lira_retrieval.CitationUniverse(
            law_exact_universe={"Art. 1 ZGB"},
            case_exact_universe={"4A_42/2015 E. 5.5"},
            case_base_universe={"4A_42/2015"},
        ),
        laws_embeddings=np.asarray([[0.95]], dtype=np.float32),
        laws_citations=["Art. 1 ZGB"],
        laws_text={"Art. 1 ZGB": "Civil code rule"},
        laws_title={"Art. 1 ZGB": "ZGB"},
        laws_code={"Art. 1 ZGB": "ZGB"},
        laws_sparse_index=DummySparseIndex(["Art. 1 ZGB"]),
        laws_bm25=DummyBM25(),
        reranker_tokenizer=None,
        reranker_model=None,
        reranker_device="cpu",
        reranker_batch_size=4,
        require_reranker=False,
        co_citation_index={"bge_art": {}, "art_art": {}},
        official_enrichment=lira_pipeline.empty_official_enrichment(),
        citation_doc_freq=Counter({"4A_42/2015 E. 5.5": 2, "Art. 1 ZGB": 3}),
        total_cases=10,
    )


def test_statute_candidates_include_article_parent_and_case_graph_targets():
    resources = make_minimal_resources()
    resources.config.laws_rerank_k = 0
    resources.config.laws_graph_k = 2
    resources.config.laws_union_pool_cap = 10
    resources.laws_citations = ["Art. 1 Abs. 1 ZGB"]
    resources.laws_sparse_index = DummySparseIndex(resources.laws_citations)
    resources.laws_text = {"Art. 1 Abs. 1 ZGB": "Civil code rule"}
    resources.laws_title = {"Art. 1 Abs. 1 ZGB": "ZGB"}
    resources.laws_code = {"Art. 1 Abs. 1 ZGB": "ZGB"}
    resources.citation_universe.law_exact_universe = {
        "Art. 1 Abs. 1 ZGB", "Art. 1 ZGB", "Art. 2 ZGB"
    }
    profile = lira_pipeline.ScoredCaseProfile(
        base_case_id="BGE 100 I 1", rrf_score=0.8,
        cited_statutes=[("Art. 2 ZGB", 3)],
    )
    encoded_views = {
        name: {"text": "civil dispute", "dense": np.asarray([1.0], dtype=np.float32), "sparse": {}}
        for name in lira_pipeline.QUERY_VIEW_NAMES
    }
    candidates, _, _ = lira_pipeline.build_statute_candidates(
        resources, "q1", "civil dispute", {"Art. 1 ZGB", "Art. 2 ZGB"},
        mode="support_only", encoded_views=encoded_views,
        case_evidence={"profile_candidates": [profile]},
    )
    by_citation = candidates.set_index("citation")
    assert by_citation.loc["Art. 1 ZGB", "parent_alias_hit"] == 1
    assert by_citation.loc["Art. 1 ZGB", "original_dense_hit"] == 1
    assert by_citation.loc["Art. 1 ZGB", "label"] == 1
    assert by_citation.loc["Art. 2 ZGB", "graph_hit"] == 1
    assert by_citation.loc["Art. 2 ZGB", "label"] == 1


def test_court_backprojection_prioritizes_retrieved_exact_consideration(monkeypatch):
    resources = make_minimal_resources()
    resources.config.enable_case_bm25 = False
    resources.config.dense_min_sim = 0.0
    profile_index, court_index = object(), object()
    resources.case_profile_faiss = profile_index
    resources.case_profile_dense = np.asarray([[1.0]], dtype=np.float32)
    resources.case_profile_ids = ["BGE 100 I 1"]
    resources.case_profile_sparse_index = DummySparseIndex(["BGE 100 I 1"])
    resources.case_profiles = {
        "BGE 100 I 1": lira_pipeline.CaseProfile(
            base_case_id="BGE 100 I 1",
            exact_case_ids=["BGE 100 I 1 E. 1"],
            cited_statutes=[], cited_cases=[], snippets=[],
            profile_text="case profile", total_considerations=2,
        )
    }
    resources.court_faiss = court_index
    resources.court_embeddings = np.asarray([[1.0], [1.0]], dtype=np.float32)
    resources.court_case_ids = ["BGE 100 I 1 E. 1", "BGE 100 I 1 E. 7"]

    def fake_search(index, embeddings, query, search_k, final_k):
        return [(0, 0.8)] if index is profile_index else [(1, 0.95), (0, 0.8)]

    monkeypatch.setattr(case_module, "faiss_search", fake_search)
    views = {
        name: {"text": "issue", "dense": np.asarray([1.0], dtype=np.float32), "sparse": {}}
        for name in lira_pipeline.QUERY_VIEW_NAMES
    }
    profiles = lira_pipeline.retrieve_case_profile_candidates(resources, "q1", "issue", views)
    assert profiles[0].exact_case_ids[:2] == ["BGE 100 I 1 E. 7", "BGE 100 I 1 E. 1"]


def test_case_candidate_pruning_preserves_base_diversity():
    resources = make_minimal_resources()
    resources.config.case_candidate_cap = 6
    resources.config.case_diversity_per_base = 3
    resources.config.case_indirect_profile_limit = 0
    profiles = []
    for base, rrf in (("BGE 100 I 1", 1.0), ("BGE 101 I 1", 0.8)):
        exact_ids = [f"{base} E. {i}" for i in range(1, 9)]
        resources.citation_universe.case_exact_universe.update(exact_ids)
        resources.citation_universe.case_base_universe.add(base)
        profiles.append(
            lira_pipeline.ScoredCaseProfile(
                base_case_id=base, rrf_score=rrf, reranker_score=rrf,
                exact_case_ids=exact_ids,
            )
        )
    candidates = lira_pipeline.build_case_candidate_frame(resources, "legal issue", profiles)
    bases = {lira_pipeline.get_case_group_id(citation) for citation in candidates["citation"]}
    assert len(candidates) == 6
    assert bases == {"BGE 100 I 1", "BGE 101 I 1"}


def test_macro_f1_matches_official_empty_behavior_v44():
    preds = [[], ["Art. 1 ZGB"], []]
    golds = [[], ["Art. 1 ZGB"], ["Art. 2 ZGB"]]
    score = lira_pipeline.macro_f1_from_sets(preds, golds)
    assert abs(score - ((1.0 + 1.0 + 0.0) / 3.0)) < 1e-9


def test_deterministic_query_views_is_stable():
    query = "Art. 221 Abs. 1 StPO and detention review after violent assault with urgent appeal."
    left = lira_pipeline.deterministic_query_views(query)
    right = lira_pipeline.deterministic_query_views(query)
    assert left.issue_view == right.issue_view
    assert left.precedent_view == right.precedent_view
    assert left.original == query


def test_merge_predictions_prefers_exact_case_over_base():
    merged = lira_pipeline.merge_predictions(
        ["Art. 1 ZGB"],
        ["4A_42/2015", "4A_42/2015 E. 5.5", "BGE 131 III 209", "BGE 131 III 209 E. 5"],
    )
    assert "4A_42/2015 E. 5.5" in merged
    assert "BGE 131 III 209 E. 5" in merged
    assert "4A_42/2015" not in merged
    assert "BGE 131 III 209" not in merged


def test_compute_neighbor_support_detects_adjacent_articles():
    support = lira_pipeline.compute_neighbor_support([
        "Art. 1 OR",
        "Art. 2 OR",
        "Art. 5 ZGB",
        "Art. 7 ZGB",
    ])
    assert support["Art. 1 OR"] == 1
    assert support["Art. 2 OR"] == 1
    assert "Art. 5 ZGB" not in support


def test_top_k_indices_matches_full_sort_for_distinct_scores():
    scores = np.asarray([0.4, 0.9, -0.1, 0.6, 0.3], dtype=np.float32)
    assert lira_pipeline.top_k_indices(scores, 3).tolist() == [1, 3, 0]
    assert lira_pipeline.top_k_indices(scores, 0).size == 0
    assert lira_pipeline.top_k_indices(scores, 10).tolist() == [1, 3, 0, 4, 2]


def test_cv_validation_reporting_is_explicit():
    parser = lira_pipeline.build_parser()
    assert parser.parse_args(["cv"]).report_val is False
    assert parser.parse_args(["cv", "--report-val"]).report_val is True


def test_model_snapshot_resolves_cache_or_override(tmp_path, monkeypatch):
    snapshots = tmp_path / "snapshots"
    cached = snapshots / "revision-a"
    cached.mkdir(parents=True)
    assert lira_pipeline.resolve_model_snapshot(snapshots, "LIRA_TEST_MODEL_PATH") == cached
    override = tmp_path / "custom-model"
    monkeypatch.setenv("LIRA_TEST_MODEL_PATH", str(override))
    assert lira_pipeline.resolve_model_snapshot(snapshots, "LIRA_TEST_MODEL_PATH") == override


def test_get_statute_feature_columns_excludes_metadata_columns():
    frame = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "citation": "Art. 1 ZGB",
                "label": 1,
                "fold": 0,
                "raw_score": 0.7,
                "prob": 0.8,
                "heuristic_score": 1.2,
                "rrf_score": 0.9,
            }
        ]
    )

    cols = lira_pipeline.get_statute_feature_columns(frame)

    assert "fold" not in cols
    assert "raw_score" not in cols
    assert "prob" not in cols
    assert "rrf_score" in cols


def test_case_budget_respects_query_mode():
    case_df = pd.DataFrame(
        [
            {"citation": "BGE 1", "case_prob": 0.40, "max_profile_reranker": 0.8, "support_profiles": 4, "regex_hit": 1, "source_direct_exact": 1},
            {"citation": "BGE 2", "case_prob": 0.35, "max_profile_reranker": 0.7, "support_profiles": 3, "regex_hit": 0, "source_direct_exact": 1},
            {"citation": "BGE 3", "case_prob": 0.10, "max_profile_reranker": 0.2, "support_profiles": 1, "regex_hit": 0, "source_direct_exact": 0},
        ]
    )
    params = {
        "case_rerank_threshold": 0.5,
        "case_regex_bonus": 1.0,
    }
    statute_df = pd.DataFrame(columns=["prob"])
    config = lira_pipeline.V44Config(case_count_clip=(0, 20))
    case_heavy = lira_pipeline.compute_case_budget(case_df, statute_df, params, config, "case-heavy")
    statute_heavy = lira_pipeline.compute_case_budget(case_df, statute_df, params, config, "statute-heavy")
    assert 0 <= case_heavy <= 20
    assert 0 <= statute_heavy <= 20
    assert case_heavy > statute_heavy


def test_case_budget_can_cover_evidence_rich_queries():
    case_df = pd.DataFrame(
        [
            {"case_prob": 0.06, "max_profile_reranker": 0.9,
             "source_direct_exact": 1, "support_profiles": 3, "regex_hit": 0}
            for _ in range(15)
        ]
    )
    params = lira_pipeline.default_selector_params(lira_pipeline.V44Config())
    budget = lira_pipeline.compute_case_budget(
        case_df, pd.DataFrame(columns=["prob"]), params,
        lira_pipeline.V44Config(), "case-heavy"
    )
    assert 5 < budget <= 25


def test_build_case_graph_support_handles_dict_shaped_bge_art():
    resources = make_minimal_resources()
    resources.co_citation_index = {
        "bge_art": {
            "4A_42/2015 E. 5.5": {
                "Art. 1 ZGB": 9,
                "Art. 2 ZGB": 3,
            }
        },
        "art_art": {},
    }
    profile = lira_pipeline.ScoredCaseProfile(
        base_case_id="4A_42/2015",
        rrf_score=1.0,
        reranker_score=0.8,
        exact_case_ids=["4A_42/2015 E. 5.5"],
        cited_cases=[],
        cited_statutes=[],
    )

    support = lira_pipeline.build_case_graph_support_from_profiles(resources, [profile])

    assert support["Art. 1 ZGB"] > 0


def test_build_case_graph_support_uses_official_case_statute_graph():
    resources = make_minimal_resources()
    resources.official_enrichment["case_statute_graph"] = {
        "4A_42/2015": {"Art. 1 ZGB": 2},
    }
    profile = lira_pipeline.ScoredCaseProfile(
        base_case_id="4A_42/2015",
        rrf_score=1.0,
        reranker_score=0.8,
        exact_case_ids=["4A_42/2015 E. 5.5"],
        cited_cases=[],
        cited_statutes=[],
    )

    support = lira_pipeline.build_case_graph_support_from_profiles(resources, [profile])

    assert support["Art. 1 ZGB"] > 0


def test_select_predictions_dedupes_exact_base_collisions():
    statute_df = pd.DataFrame(
        [
            {"citation": "Art. 1 ZGB", "prob": 0.8, "regex_hit": 0, "graph_case_support": 0.0},
            {"citation": "Art. 2 ZGB", "prob": 0.5, "regex_hit": 0, "graph_case_support": 0.0},
        ]
    )
    case_df = pd.DataFrame(
        [
            {"citation": "4A_42/2015", "case_prob": 0.40, "max_profile_reranker": 0.9, "support_profiles": 3, "regex_hit": 0, "source_direct_exact": 0},
            {"citation": "4A_42/2015 E. 5.5", "case_prob": 0.55, "max_profile_reranker": 0.95, "support_profiles": 4, "regex_hit": 1, "source_direct_exact": 1},
        ]
    )
    bundle = {
        "statute_count_model": DummyCountModel(1.0),
        "statute_count_feature_cols": ["candidate_count"],
        "selector_params": {
            "statute_multiplier": 1.0,
            "statute_floor": 0.1,
            "case_rerank_threshold": 0.1,
            "case_floor": 0.1,
            "case_regex_bonus": 1.0,
        },
        "statute_clip": (0, 5),
        "case_clip": (0, 5),
    }
    preds = lira_pipeline.select_predictions(statute_df, case_df, "contract dispute", bundle)
    assert "Art. 1 ZGB" in preds
    assert "4A_42/2015 E. 5.5" in preds
    assert "4A_42/2015" not in preds


def test_statute_selection_uses_probability_not_candidate_order():
    statute_df = pd.DataFrame(
        [
            {"citation": "Art. 1 ZGB", "prob": 0.05, "heuristic_score": 9.0},
            {"citation": "Art. 2 ZGB", "prob": 0.90, "heuristic_score": 1.0},
        ]
    )
    assert lira_pipeline.select_statute_predictions(statute_df, floor=0.01, count=1) == ["Art. 2 ZGB"]
    bundle = {
        "statute_count_model": DummyCountModel(1.0),
        "statute_count_feature_cols": ["candidate_count"],
        "selector_params": {
            "statute_multiplier": 1.0,
            "statute_floor": 0.01,
            "case_rerank_threshold": 0.1,
            "case_floor": 0.1,
            "case_regex_bonus": 0.5,
        },
        "statute_clip": (0, 5),
        "case_clip": (0, 5),
    }
    preds = lira_pipeline.select_predictions(
        statute_df, lira_pipeline.empty_case_frame(scored=True), "civil issue", bundle
    )
    assert preds == ["Art. 2 ZGB"]


def test_score_statute_frame_succeeds_without_fold_when_schema_is_clean(monkeypatch):
    candidate_df = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "citation": "Art. 1 ZGB",
                "label": 0,
                "heuristic_score": 1.0,
                "rrf_score": 0.8,
            }
        ]
    )
    seen = {}

    def fake_predict(df, model, feature_cols):
        seen["columns"] = list(df.columns)
        seen["feature_cols"] = list(feature_cols)
        return np.array([0.25], dtype=np.float32)

    monkeypatch.setattr(statute_module, "predict_with_ranker", fake_predict)
    monkeypatch.setattr(statute_module, "apply_calibrator", lambda df, calibrator: np.array([0.6], dtype=np.float32))

    scored = lira_pipeline.score_statute_frame(candidate_df, object(), None, ["rrf_score"])

    assert seen["feature_cols"] == ["rrf_score"]
    assert "fold" not in seen["columns"]
    assert float(scored.loc[0, "raw_score"]) == 0.25
    assert abs(float(scored.loc[0, "prob"]) - 0.6) < 1e-6


def test_score_statute_frame_rejects_stale_feature_schema_with_fold():
    candidate_df = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "citation": "Art. 1 ZGB",
                "label": 0,
                "heuristic_score": 1.0,
                "rrf_score": 0.8,
            }
        ]
    )

    try:
        lira_pipeline.score_statute_frame(candidate_df, None, None, ["fold", "rrf_score"])
    except RuntimeError as exc:
        assert "Stale statute feature schema" in str(exc)
    else:
        raise AssertionError("Expected stale feature schema failure")


def test_to_v43_config_returns_real_v43_config():
    cfg = lira_pipeline.V44Config()
    adapted = lira_pipeline.to_v43_config(cfg)
    assert isinstance(adapted, lira_retrieval.V43Config)
    assert adapted.encoder_path == cfg.encoder_path
    assert adapted.court_dense_path == cfg.court_dense_path
    assert adapted.sparse_doc_top_tokens == cfg.sparse_doc_top_tokens


def test_load_official_enrichment_returns_empty_defaults_when_missing(tmp_path):
    cfg = lira_pipeline.V44Config(official_enrichment_path=tmp_path / "missing.pkl")
    payload = lira_pipeline.load_official_enrichment(cfg)
    assert payload["statute_aliases"] == {}
    assert payload["case_statute_graph"] == {}


def test_maybe_load_reranker_respects_explicit_disable(monkeypatch):
    monkeypatch.setenv("LIRA_V44_DISABLE_RERANKER", "1")
    tokenizer, model, device, batch_size = lira_pipeline.maybe_load_reranker(lira_pipeline.V44Config())
    assert tokenizer is None
    assert model is None
    assert device in {"cpu", "cuda"}
    assert batch_size > 0


def test_ensure_v43_base_artifacts_auto_builds_missing_court_faiss(monkeypatch, tmp_path):
    dense_path = tmp_path / "court.npy"
    np.save(dense_path, np.zeros((2, 2), dtype=np.float32))
    meta_path = tmp_path / "court_meta.pkl"
    lira_pipeline.save_pickle(meta_path, {"case_ids": ["A", "B"], "case_citations": [[], []]})
    folds_path = tmp_path / "folds.pkl"
    lira_pipeline.save_pickle(folds_path, {"q1": 0})
    court_index = tmp_path / "court.index"
    court_meta_json = tmp_path / "court.meta.json"

    monkeypatch.setattr(core_module, "FOLDS_PATH", folds_path)
    monkeypatch.setattr(core_module, "COURT_FAISS_PATH", court_index)
    monkeypatch.setattr(lira_retrieval, "COURT_FAISS_META_PATH", court_meta_json)

    calls = {"laws": 0, "faiss": 0}

    def fake_ensure_laws_sparse_postings(cfg):
        calls["laws"] += 1

    def fake_build_faiss_index(cfg):
        calls["faiss"] += 1
        court_index.write_bytes(b"index")
        court_meta_json.write_text(json.dumps({"rows": 2}), encoding="utf-8")

    monkeypatch.setattr(lira_retrieval, "ensure_laws_sparse_postings", fake_ensure_laws_sparse_postings)
    monkeypatch.setattr(lira_retrieval, "build_faiss_index", fake_build_faiss_index)

    cfg = lira_pipeline.V44Config(court_dense_path=dense_path, court_meta_path=meta_path)
    lira_pipeline.ensure_v43_base_artifacts(cfg)

    assert calls["laws"] == 1
    assert calls["faiss"] == 1
    assert court_index.exists()
    assert court_meta_json.exists()


def test_support_only_mode_skips_case_reranker(monkeypatch):
    resources = make_minimal_resources()

    monkeypatch.setattr(
        statute_module,
        "encode_query_views_for_query",
        lambda resources, query_id, query: {name: {"text": query, "dense": np.asarray([1.0], dtype=np.float32), "sparse": {}} for name in lira_pipeline.QUERY_VIEW_NAMES},
    )
    monkeypatch.setattr(
        statute_module,
        "retrieve_case_profile_candidates",
        lambda resources, query_id, query, encoded_views=None: [
            lira_pipeline.ScoredCaseProfile(
                base_case_id="4A_42/2015",
                rrf_score=1.0,
                exact_case_ids=["4A_42/2015 E. 5.5"],
                cited_cases=[],
                cited_statutes=[("Art. 1 ZGB", 2)],
            )
        ],
    )
    monkeypatch.setattr(statute_module, "rerank_case_profiles", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reranker should not run")))

    statute_df, case_df, profiles = lira_pipeline.build_statute_candidates(resources, "q1", "contract issue", mode="support_only")

    assert not statute_df.empty
    assert case_df.empty
    assert profiles[0].base_case_id == "4A_42/2015"


def test_full_case_mode_prefers_exact_case_over_base(monkeypatch):
    resources = make_minimal_resources()

    monkeypatch.setattr(
        statute_module,
        "encode_query_views_for_query",
        lambda resources, query_id, query: {name: {"text": query, "dense": np.asarray([1.0], dtype=np.float32), "sparse": {}} for name in lira_pipeline.QUERY_VIEW_NAMES},
    )
    monkeypatch.setattr(
        statute_module,
        "retrieve_case_profile_candidates",
        lambda resources, query_id, query, encoded_views=None: [
            lira_pipeline.ScoredCaseProfile(
                base_case_id="4A_42/2015",
                rrf_score=1.0,
                exact_case_ids=["4A_42/2015 E. 5.5"],
                cited_cases=[("4A_42/2015", 2)],
                cited_statutes=[("Art. 1 ZGB", 2)],
                late_score=0.6,
            )
        ],
    )
    monkeypatch.setattr(
        statute_module,
        "rerank_case_profiles",
        lambda resources, encoded_views, profiles: [
            lira_pipeline.ScoredCaseProfile(
                base_case_id=profiles[0].base_case_id,
                rrf_score=profiles[0].rrf_score,
                exact_case_ids=profiles[0].exact_case_ids,
                cited_cases=profiles[0].cited_cases,
                cited_statutes=profiles[0].cited_statutes,
                late_score=profiles[0].late_score,
                reranker_score=0.9,
            )
        ],
    )

    _, case_df, _ = lira_pipeline.build_statute_candidates(
        resources,
        "q1",
        "contract issue 4A_42/2015 E. 5.5",
        mode="full_case",
        case_evidence_params=lira_pipeline.default_case_evidence_params(resources.config),
    )

    assert "4A_42/2015 E. 5.5" in case_df["citation"].tolist()
    selected = lira_pipeline.select_case_predictions(
        case_df, lira_pipeline.default_selector_params(resources.config), case_budget=2
    )
    assert "4A_42/2015 E. 5.5" in selected
    assert "4A_42/2015" not in selected


def test_case_probability_fallback_avoids_all_ones():
    frame = pd.DataFrame(
        [
            {
                "citation": "BGE 1",
                "rrf_score": 1.0,
                "reranker_score": 0.0,
                "late_score": 0.0,
                "support_profiles": 2,
                "direct_profile_hits": 2,
                "source_direct_exact": 1,
                "source_indirect_cited": 0,
                "regex_hit": 1,
                "idf": 1.0,
                "frequency_weight": 1.0,
                "is_exact_case": 1,
                "is_base_case": 0,
                "max_profile_reranker": 0.0,
            },
            {
                "citation": "BGE 2",
                "rrf_score": 1.0,
                "reranker_score": 0.0,
                "late_score": 0.0,
                "support_profiles": 1,
                "direct_profile_hits": 0,
                "source_direct_exact": 0,
                "source_indirect_cited": 1,
                "regex_hit": 0,
                "idf": 1.0,
                "frequency_weight": 1.0,
                "is_exact_case": 1,
                "is_base_case": 0,
                "max_profile_reranker": 0.0,
            },
        ]
    )

    scored = lira_pipeline.apply_case_evidence_model(frame, lira_pipeline.default_case_evidence_params(lira_pipeline.V44Config()))

    probs = scored["case_prob"].to_numpy(dtype=np.float32)
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    assert not np.allclose(probs, np.ones_like(probs))
    assert float(scored.iloc[0]["case_prob"]) > float(scored.iloc[1]["case_prob"])


def test_case_candidate_frame_respects_v444_pruning_caps():
    resources = make_minimal_resources()
    resources.citation_universe = lira_retrieval.CitationUniverse(
        law_exact_universe={"Art. 1 ZGB"},
        case_exact_universe={*(f"4A_{idx}/2015 E. 1.1" for idx in range(100)), *(f"BGE 140 III {idx}" for idx in range(100))},
        case_base_universe={*(f"4A_{idx}/2015" for idx in range(100)), *(f"BGE 140 III {idx}" for idx in range(100))},
    )
    resources.citation_doc_freq = Counter({f"4A_{idx}/2015 E. 1.1": 1 for idx in range(100)})
    resources.citation_doc_freq.update({f"BGE 140 III {idx}": 1 for idx in range(100)})
    resources.total_cases = 200

    profiles = []
    for idx in range(80):
        profiles.append(
            lira_pipeline.ScoredCaseProfile(
                base_case_id=f"4A_{idx}/2015",
                rrf_score=1.0 / (idx + 1),
                reranker_score=max(0.0, 1.0 - idx / 100.0),
                exact_case_ids=[f"4A_{idx}/2015 E. 1.1"],
                cited_cases=[(f"BGE 140 III {j}", 1) for j in range(10)],
                cited_statutes=[],
            )
        )

    case_df = lira_pipeline.build_case_candidate_frame(resources, "query", profiles)
    stats = case_df.attrs["candidate_stats"]

    assert stats["direct_exact_count"] <= 60
    assert stats["post_prune_count"] <= 300
    assert len(case_df) <= 300


def test_require_live_reranker_raises_when_quality_mode_missing_reranker():
    resources = SimpleNamespace(require_reranker=True, reranker_tokenizer=None, reranker_model=None)

    try:
        lira_pipeline.require_live_reranker(resources, "unit-test")
    except RuntimeError as exc:
        assert "required" in str(exc).lower()
    else:
        raise AssertionError("Expected reranker-required failure")


def test_write_and_validate_kaggle_manifest(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    base_asset = tmp_path / "asset.bin"
    bundle_path = tmp_path / "bundle.pkl"
    base_asset.write_bytes(b"asset")
    bundle_path.write_bytes(b"bundle")

    monkeypatch.setattr(core_module, "V44_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(core_module, "V44_MODEL_BUNDLE_PATH", bundle_path)
    monkeypatch.setattr(
        core_module,
        "collect_v44_asset_paths",
        lambda config=None, include_bundle=True: {"asset": base_asset, **({"model_bundle": bundle_path} if include_bundle else {})},
    )

    manifest = lira_pipeline.write_kaggle_manifest(include_bundle=True)
    validated = lira_pipeline.validate_kaggle_manifest(require_bundle=True)

    assert manifest_path.exists()
    assert manifest["pipeline_version"] == lira_pipeline.V44_PIPELINE_VERSION
    assert "asset" in validated["entries"]


def test_build_case_extracted_cits_from_meta_aggregates_by_base_case():
    extracted = lira_pipeline.build_case_extracted_cits_from_meta(
        [
            "4A_42/2015 E. 5.5",
            "4A_42/2015 E. 5.5",
            "BGE 131 III 209 E. 3",
        ],
        [
            ["Art. 1 ZGB", "BGE 100 Ib 1"],
            ["Art. 1 ZGB"],
            ["Art. 2 ZGB"],
        ],
    )

    assert extracted["4A_42/2015"]["Art. 1 ZGB"] == 2
    assert extracted["4A_42/2015"]["BGE 100 Ib 1"] == 1
    assert extracted["BGE 131 III 209"]["Art. 2 ZGB"] == 1


def test_load_case_profile_bm25_rebuilds_in_memory_on_memory_error(monkeypatch):
    profiles = {
        "4A_42/2015": lira_pipeline.CaseProfile(
            base_case_id="4A_42/2015",
            exact_case_ids=["4A_42/2015 E. 5.5"],
            cited_statutes=[("Art. 1 ZGB", 1)],
            cited_cases=[],
            snippets=["contract dispute"],
            profile_text="Case: 4A_42/2015\nTop statutes: Art. 1 ZGB",
            total_considerations=1,
        )
    }

    def fake_load_pickle(path):
        if path == lira_pipeline.V44_CASE_BM25_PATH:
            raise MemoryError("simulated")
        raise AssertionError(f"Unexpected pickle load: {path}")

    monkeypatch.setattr(core_module, "load_pickle", fake_load_pickle)
    monkeypatch.setattr(core_module, "build_bm25_from_docs", lambda doc_ids, texts: {"doc_ids": list(doc_ids), "texts": list(texts)})

    rebuilt = lira_pipeline.load_case_profile_bm25(["4A_42/2015"], profiles)

    assert rebuilt["doc_ids"] == ["4A_42/2015"]
    assert "Art. 1 ZGB" in rebuilt["texts"][0]


def test_load_case_profile_bm25_disables_channel_if_rebuild_still_oom(monkeypatch):
    profiles = {
        "4A_42/2015": lira_pipeline.CaseProfile(
            base_case_id="4A_42/2015",
            exact_case_ids=["4A_42/2015 E. 5.5"],
            cited_statutes=[],
            cited_cases=[],
            snippets=["contract dispute"],
            profile_text="Case: 4A_42/2015",
            total_considerations=1,
        )
    }

    monkeypatch.setattr(core_module, "load_pickle", lambda path: (_ for _ in ()).throw(MemoryError("simulated")))
    monkeypatch.setattr(core_module, "build_bm25_from_docs", lambda doc_ids, texts: (_ for _ in ()).throw(MemoryError("simulated")))

    rebuilt = lira_pipeline.load_case_profile_bm25(["4A_42/2015"], profiles)

    assert isinstance(rebuilt, lira_pipeline.EfficientBM25)
    assert rebuilt.doc_ids == []


def test_load_cached_frame_removes_corrupted_pickle(tmp_path):
    path = tmp_path / "broken.pkl"
    path.write_bytes(b"not-a-valid-pickle")

    frame = lira_pipeline.load_cached_frame(path, cache_kind="fold_statute", required_columns={"query_id"})

    assert frame is None
    assert not path.exists()


def test_load_cached_frame_removes_stale_schema_payload(tmp_path):
    path = tmp_path / "stale.pkl"
    pd.to_pickle(
        {
            "meta": {
                "pipeline_version": "old",
                "cache_kind": "fold_statute",
                "schema_version": lira_pipeline.STATUTE_CANDIDATE_SCHEMA_VERSION,
            },
            "frame": pd.DataFrame([{"query_id": "q1", "citation": "Art. 1 ZGB", "label": 1, "heuristic_score": 1.0, "fold": 0}]),
        },
        path,
    )

    frame = lira_pipeline.load_cached_frame(path, cache_kind="fold_statute", required_columns=lira_pipeline.STATUTE_REQUIRED_CANDIDATE_COLUMNS)

    assert frame is None
    assert not path.exists()


def test_load_resources_missing_shared_faiss_is_actionable(monkeypatch, tmp_path):
    missing_index = tmp_path / "missing.index"
    missing_meta = tmp_path / "missing.meta.json"
    monkeypatch.setattr(core_module, "COURT_FAISS_PATH", missing_index)
    monkeypatch.setattr(lira_retrieval, "COURT_FAISS_META_PATH", missing_meta)
    cfg = lira_pipeline.V44Config()
    try:
        lira_pipeline.load_resources(cfg)
    except RuntimeError as exc:
        assert "shared court FAISS index" in str(exc)
    else:
        raise AssertionError("Expected actionable RuntimeError for missing shared FAISS artifact")



