from collections import Counter

import pandas as pd
import pytest

import lira_retrieval


def test_parent_statute_from_paragraph_and_universe():
    laws = pd.DataFrame({"citation": ["Art. 41 Abs. 1 OR", "Art. 41 Abs. 2 OR", "Art. 7 Abs. 1 220"]})
    universe = lira_retrieval.build_citation_universe(laws, [], [], {})
    assert lira_retrieval.parent_statute_citation("Art. 41 Abs. 1 OR") == "Art. 41 OR"
    assert lira_retrieval.parent_statute_citation("Art. 7 Abs. 1 220") == "Art. 7 220"
    assert lira_retrieval.parent_statute_citation("Art. 41 OR") is None
    assert lira_retrieval.get_valid_citation("Art. 41 OR", universe).citation == "Art. 41 OR"
    assert "Art. 99 OR" not in universe.law_exact_universe


def test_macro_f1_rejects_dropped_predictions():
    with pytest.raises(ValueError, match="counts differ"):
        lira_retrieval.macro_f1_from_sets([[], []], [[]])


def test_macro_f1_matches_official_empty_behavior():
    preds = [[], ["Art. 1 ZGB"], []]
    golds = [[], ["Art. 1 ZGB"], ["Art. 2 ZGB"]]
    score = lira_retrieval.macro_f1_from_sets(preds, golds)
    assert abs(score - ((1.0 + 1.0 + 0.0) / 3.0)) < 1e-9


def test_get_case_group_id_strips_e_suffix():
    assert lira_retrieval.get_case_group_id("1C_344/2011 15.03.2012 E. 6") == "1C_344/2011"
    assert lira_retrieval.get_case_group_id("BGE 139 I 2 E. 5.1") == "BGE 139 I 2"


def test_exact_case_citations_are_preserved_when_present():
    universe = lira_retrieval.CitationUniverse(
        law_exact_universe={"Art. 1 ZGB"},
        case_exact_universe={"4A_42/2015 E. 5.5", "BGE 131 III 209 E. 5"},
        case_base_universe={"4A_42/2015", "BGE 131 III 209"},
    )
    exact_docket = lira_retrieval.get_valid_citation("4A_42/2015 E. 5.5", universe)
    exact_bge = lira_retrieval.get_valid_citation("ATF 131 III 209 E. 5", universe)
    assert exact_docket is not None and exact_docket.citation == "4A_42/2015 E. 5.5"
    assert exact_docket.case_match == "exact"
    assert exact_bge is not None and exact_bge.citation == "BGE 131 III 209 E. 5"
    assert exact_bge.case_match == "exact"


def test_case_base_fallback_only_happens_when_exact_case_is_absent():
    universe = lira_retrieval.CitationUniverse(
        law_exact_universe=set(),
        case_exact_universe=set(),
        case_base_universe={"5A_561/2020", "BGE 130 III 585"},
    )
    docket = lira_retrieval.get_valid_citation("5A_561/2020 E. 5.1.1", universe)
    bge = lira_retrieval.get_valid_citation("BGE 130 III 585 E. 2.1", universe)
    assert docket is not None and docket.citation == "5A_561/2020"
    assert docket.case_match == "base"
    assert bge is not None and bge.citation == "BGE 130 III 585"
    assert bge.case_match == "base"


def test_statute_validation_preserves_normalization_logic():
    universe = lira_retrieval.CitationUniverse(
        law_exact_universe={"Art. 934 Abs. 1 ZGB"},
        case_exact_universe=set(),
        case_base_universe=set(),
    )
    match = lira_retrieval.get_valid_citation("art 934 abs 1 ZGB", universe)
    assert match is not None
    assert match.citation == "Art. 934 Abs. 1 ZGB"
    assert match.case_match == ""


def test_fold_labels_have_min_support():
    train_df = pd.read_csv("data/train.csv").dropna(subset=["query", "gold_citations"]).reset_index(drop=True)
    labels = lira_retrieval.make_bucket_labels(train_df)
    counts = Counter(labels)
    assert min(counts.values()) >= 5


