# Local data

| File | Purpose |
| --- | --- |
| `train.csv` | 1,139 labeled training questions |
| `val.csv` | 10 labeled report questions |
| `test.csv` | 40 unlabeled questions |
| `laws_de.csv` | Law texts and citation metadata |
| `court_considerations.csv` | Court texts and citation metadata |
| `sample_submission.csv` | Output column example |

Queries use `query_id` and `query`; labeled splits also include semicolon-separated `gold_citations`. Predictions use `query_id` and `predicted_citations`. Large data files are local and excluded from Git.

