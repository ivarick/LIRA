# LIRA status — 10 November 2026

## Current implementation

Pipeline `v448` adds article-level statute candidates derived from retrieved paragraphs, direct case-to-statute graph candidates, query-relevant exact court consideration hits, and a diversity pass when pruning case candidates. Statute output now follows calibrated probability order in both tuning and prediction. The case budget can represent up to 25 citations for evidence-rich queries. Dense statute retrieval selects top results with a partial sort. Embedding fine-tuning now uses query-disjoint validation and filters gold-article paragraphs from hard negatives.

## Evidence checked locally

| Check | Result | Meaning |
| --- | --- | --- |
| Active unit suite | 45 passed | Includes retrieval, selection, and training-data tests |
| Ruff | Passed | Static lint checks on the pipeline, setup scripts, and tests |
| Train gold statute forms available in the corpus-derived universe | 70.9% before; 97.3% after | Article-level aliases remove a label coverage ceiling; this is not candidate recall or F1 |
| Validation gold statute forms available | 100% before and after | This change does not raise the validation universe ceiling |
| Exact case forms present in court corpus | 57/57 train; 102/102 validation | Case retrieval and ranking remain the limiting stages |
| Previous five-case output cap on validation labels | At most 43 of 102 cases could be emitted | The new 25-case cap removes this mechanical ceiling; it does not establish precision |

The saved 12-query CPU smoke report (`output/fast_v44_dev_report.json`) is from `v447`: selector F1 `0.2036`, statute candidate recall `0.5963`, statute top-50 recall `0.2370`, and `promote_to_full_cv: false`. It has not been rerun with `v448`.

## Remaining validation

The `cache/` model and index assets are absent, so `prepare`, full cross-validation, and Kaggle-style offline prediction cannot currently run. The new retrieval paths and training data helpers are verified with synthetic tests; their effect on full-split F1, runtime, and memory is unknown. Rebuild the artifacts, run the smoke harness and full OOF evaluation, inspect statute candidate recall and exact-case recall, then report a new score. The ten-query validation labels have informed development, so that split is not an independent final test.

Retired versions and logs remain outside the working tree at `C:\Users\Ivarick\Desktop\lira_retired_2026-11-10`.

**Related:** [Problem and solution](../README.md) · [Technical architecture](ARCHITECTURE.md) · [Training and evaluation](TRAINING_AND_EVALUATION.md) · [Operations](OPERATIONS.md)


