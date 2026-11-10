import numpy as np

from scripts.setup.training_utils import split_triplets_by_query, top_non_gold_indices


def test_training_split_keeps_queries_disjoint():
    triplets = [
        {"anchor": query, "positive": f"p{index}", "negative": "n"}
        for query in ("q1", "q2", "q3", "q4")
        for index in range(3)
    ]
    train, validation = split_triplets_by_query(triplets, validation_fraction=0.25, seed=7)
    assert {item["anchor"] for item in train}.isdisjoint(
        {item["anchor"] for item in validation}
    )
    assert len(train) + len(validation) == len(triplets)
    assert split_triplets_by_query(triplets, 0.25, 7) == (train, validation)


def test_hard_negatives_exclude_gold_article_paragraphs():
    citations = ["Art. 41 Abs. 1 OR", "Art. 41 Abs. 2 OR", "Art. 42 OR", "Art. 43 OR"]
    scores = np.asarray([0.99, 0.90, 0.85, 0.70], dtype=np.float32)
    indices = top_non_gold_indices(scores, citations, {"Art. 41 OR"}, limit=2)
    assert [citations[index] for index in indices] == ["Art. 42 OR", "Art. 43 OR"]
    sibling_indices = top_non_gold_indices(
        scores, citations, {"Art. 41 Abs. 1 OR"}, limit=2
    )
    assert [citations[index] for index in sibling_indices] == ["Art. 42 OR", "Art. 43 OR"]
