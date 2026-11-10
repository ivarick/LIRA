#!/usr/bin/env python3
"""
Evaluate a submission file against a solution file using Citation-level Macro F1.

Use train.csv or val.csv as the solution file for local evaluation.
The test set solution is not publicly available.

Usage:
    # Evaluate against validation set (default)
    python scripts/evaluate_submission.py submission.csv

    # Evaluate against training set
    python scripts/evaluate_submission.py submission.csv --split train

    # Evaluate against a custom solution file
    python scripts/evaluate_submission.py submission.csv --solution path/to/solution.csv

    # Verbose mode (shows per-query F1 scores)
    python scripts/evaluate_submission.py submission.csv -v
"""

import argparse
import math
import re
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd

# Default paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class ParticipantVisibleError(Exception):
    """Errors raised with this type will be shown to participants."""
    pass


_WS_RE = re.compile(r"\s+")


def _canonicalize_citation(c: str) -> str:
    """Conservative normalization: strip + collapse whitespace."""
    c = c.strip()
    c = _WS_RE.sub(" ", c)
    return c


def _parse_citation_field(value: object, sep: str, max_items: int, max_chars: int) -> Set[str]:
    """
    Parse a semicolon-separated citation string into a canonicalized set.
    """
    # Robust missing-value handling (covers None, np.nan, pd.NA, etc.)
    if pd.isna(value):
        return set()

    # Be permissive: cast non-strings to string instead of failing validation
    if not isinstance(value, str):
        value = str(value)

    if len(value) > max_chars:
        raise ParticipantVisibleError(
            f"predicted_citations field too long ({len(value)} chars). "
            f"Please limit to <= {max_chars} characters per query."
        )

    parts = value.split(sep)
    if len(parts) > max_items:
        raise ParticipantVisibleError(
            f"Too many citations in a single row ({len(parts)}). "
            f"Please limit to <= {max_items} citations per query."
        )

    out: Set[str] = set()
    for p in parts:
        canon = _canonicalize_citation(p)
        if canon:
            out.add(canon)
    return out


def _f1_for_sets(pred: Set[str], gold: Set[str]) -> float:
    """Compute F1 score between two sets."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred.intersection(gold))
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    citation_separator: str = ";",
    max_citations_per_row: int = 200,
    max_chars_per_row: int = 10_000,
) -> float:
    """
    Citation-level Macro F1 (set-based) for legal source retrieval.

    Participants submit one row per query with a semicolon-separated list of citations.
    We compute per-query F1 between the predicted citation set and the gold citation set,
    then average across queries (Macro F1).

    Expected submission format (one non-ID column):
      query_id,predicted_citations
      q_001,"SR 210 Art. 1;BGE 116 Ia 56"
      q_002,"SR 311.0 Art. 117"

    The solution file must contain the same non-ID column name and hold the gold citations.
    """

    # Basic structural checks
    if row_id_column_name not in solution.columns:
        raise ParticipantVisibleError(f"Solution is missing id column '{row_id_column_name}'.")
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Submission is missing id column '{row_id_column_name}'.")

    # Align submission to solution by row_id
    solution = solution.set_index(row_id_column_name)
    submission = submission.set_index(row_id_column_name)

    # Check that all solution IDs are present in submission
    missing_ids = set(solution.index) - set(submission.index)
    if missing_ids:
        raise ParticipantVisibleError(
            f"Submission is missing {len(missing_ids)} query IDs: {sorted(missing_ids)[:5]}..."
        )

    # Reindex submission to match solution order
    submission = submission.loc[solution.index]

    # Require exactly one prediction column (simple + avoids ambiguity)
    if len(submission.columns) != 1:
        raise ParticipantVisibleError(
            f"Submission must have exactly 1 prediction column (found {len(submission.columns)})."
        )

    pred_col = submission.columns[0]

    # Find gold citations column in solution
    # Support both 'gold_citations' (train/val) and 'predicted_citations' (submission format)
    if "gold_citations" in solution.columns:
        gold_col = "gold_citations"
    elif pred_col in solution.columns:
        gold_col = pred_col
    elif len(solution.columns) == 1:
        gold_col = solution.columns[0]
    else:
        raise ParticipantVisibleError(
            "Solution file must have a 'gold_citations' column or match the submission column name. "
            f"Found columns: {list(solution.columns)}"
        )

    # Parse citation strings into sets
    gold_series = solution[gold_col]
    pred_series = submission[pred_col]

    if len(gold_series) != len(pred_series):
        raise ParticipantVisibleError("Solution and submission have different number of rows after alignment.")

    f1s: List[float] = []
    for g, p in zip(gold_series.tolist(), pred_series.tolist()):
        gold_set = _parse_citation_field(g, citation_separator, max_citations_per_row, max_chars_per_row)
        pred_set = _parse_citation_field(p, citation_separator, max_citations_per_row, max_chars_per_row)
        f1s.append(_f1_for_sets(pred_set, gold_set))

    if not f1s:
        raise ParticipantVisibleError("No rows to score.")

    result = float(sum(f1s) / len(f1s))

    # Must be finite, non-null
    if not math.isfinite(result):
        raise ParticipantVisibleError("Score is not finite. Please check submission format.")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a submission file using Citation-level Macro F1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate against validation set (default)
    python scripts/evaluate_submission.py submission.csv

    # Evaluate against training set
    python scripts/evaluate_submission.py submission.csv --split train

    # Evaluate against a custom solution file
    python scripts/evaluate_submission.py submission.csv --solution path/to/solution.csv

    # Verbose mode
    python scripts/evaluate_submission.py submission.csv -v
        """
    )
    parser.add_argument(
        "submission",
        type=Path,
        help="Path to the submission CSV file"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Which data split to evaluate against: 'train' or 'val' (default: val)"
    )
    parser.add_argument(
        "--solution", "-s",
        type=Path,
        dest="solution_path",
        help="Path to a custom solution CSV file (overrides --split)"
    )
    parser.add_argument(
        "--row-id", "-r",
        type=str,
        default="query_id",
        help="Name of the row ID column (default: query_id)"
    )
    parser.add_argument(
        "--separator", "-sep",
        type=str,
        default=";",
        help="Citation separator character (default: ;)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-query F1 scores"
    )

    args = parser.parse_args()

    # Resolve solution path
    if args.solution_path is not None:
        solution_path = args.solution_path
    else:
        # Use default split file from data directory
        solution_path = DEFAULT_DATA_DIR / f"{args.split}.csv"

    # Validate file paths
    if not args.submission.exists():
        print(f"Error: Submission file not found: {args.submission}", file=sys.stderr)
        sys.exit(1)
    if not solution_path.exists():
        print(f"Error: Solution file not found: {solution_path}", file=sys.stderr)
        sys.exit(1)

    # Load files
    try:
        submission_df = pd.read_csv(args.submission)
        solution_df = pd.read_csv(solution_path)
    except Exception as e:
        print(f"Error reading CSV files: {e}", file=sys.stderr)
        sys.exit(1)

    # Print info
    print(f"Submission: {args.submission} ({len(submission_df)} rows)")
    print(f"Solution: {solution_path} ({len(solution_df)} rows)")
    print(f"Row ID column: {args.row_id}")
    print()

    # Compute score
    try:
        macro_f1 = score(
            solution=solution_df,
            submission=submission_df,
            row_id_column_name=args.row_id,
            citation_separator=args.separator,
        )
        print(f"Macro F1 Score: {macro_f1:.6f}")

        if args.verbose:
            # Re-compute for per-query details
            print("\nPer-query F1 scores:")
            solution_df = solution_df.set_index(args.row_id)
            submission_df = submission_df.set_index(args.row_id)
            submission_df = submission_df.loc[solution_df.index]

            pred_col = submission_df.columns[0]
            # Find gold column
            if "gold_citations" in solution_df.columns:
                gold_col = "gold_citations"
            elif pred_col in solution_df.columns:
                gold_col = pred_col
            else:
                gold_col = solution_df.columns[0]

            for idx in solution_df.index:
                gold_set = _parse_citation_field(solution_df.loc[idx, gold_col], args.separator, 200, 10_000)
                pred_set = _parse_citation_field(submission_df.loc[idx, pred_col], args.separator, 200, 10_000)
                f1 = _f1_for_sets(pred_set, gold_set)
                print(f"  {idx}: F1={f1:.4f} (pred={len(pred_set)}, gold={len(gold_set)})")

    except ParticipantVisibleError as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error computing score: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


