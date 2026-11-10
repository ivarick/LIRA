"""Deterministic data preparation for embedding fine-tuning."""

from random import Random
from typing import Sequence

import numpy as np

from lira_retrieval import parent_statute_citation


def split_triplets_by_query(
    triplets: Sequence[dict[str, str]], validation_fraction: float = 0.05, seed: int = 42
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep every triplet for a query on the same side of the split."""
    queries = sorted({str(item["anchor"]) for item in triplets})
    if len(queries) < 2:
        raise ValueError("At least two distinct queries are required for a validation split")
    Random(seed).shuffle(queries)
    validation_count = min(len(queries) - 1, max(1, round(len(queries) * validation_fraction)))
    validation_queries = set(queries[:validation_count])
    train = [item for item in triplets if str(item["anchor"]) not in validation_queries]
    validation = [item for item in triplets if str(item["anchor"]) in validation_queries]
    return train, validation


def top_non_gold_indices(
    scores: np.ndarray, citations: Sequence[str], gold: set[str], limit: int
) -> list[int]:
    """Get high-scoring negatives while excluding gold articles and their paragraphs."""
    if limit <= 0 or scores.size == 0:
        return []
    if len(scores) != len(citations):
        raise ValueError("Score and citation counts differ")
    gold_articles = {parent_statute_citation(citation) or citation for citation in gold}
    pool = min(len(scores), max(100, limit * 10))
    while True:
        indices = np.argpartition(-scores, pool - 1)[:pool]
        ranked = indices[np.argsort(-scores[indices], kind="stable")]
        negatives = [
            int(index)
            for index in ranked
            if (parent_statute_citation(citations[index]) or citations[index])
            not in gold_articles
        ]
        if len(negatives) >= limit or pool == len(scores):
            return negatives[:limit]
        pool = min(len(scores), pool * 2)
