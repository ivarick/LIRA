#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lira_pipeline  # noqa: E402 - repository path is added above


def parse_multi(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = [item.strip() for item in text.replace("|", ";").split(";")]
    return [item for item in parts if item]


def add_weighted_edges(target: dict[str, Counter], source: str, raw_values: object) -> None:
    counter = target.setdefault(str(source), Counter())
    for value in parse_multi(raw_values):
        counter[str(value)] += 1


def build_enrichment(input_dir: Path) -> dict[str, object]:
    enrichment = lira_pipeline.empty_official_enrichment()
    source_meta = {}

    articles_path = input_dir / "fedlex_articles.csv"
    if articles_path.exists():
        articles_df = pd.read_csv(articles_path)
        statute_aliases = defaultdict(list)
        code_aliases = defaultdict(list)
        article_neighbors = defaultdict(list)
        for _, row in articles_df.iterrows():
            citation = str(row.get("citation", "")).strip()
            if not citation:
                continue
            for alias in parse_multi(row.get("aliases", "")):
                if alias not in statute_aliases[citation]:
                    statute_aliases[citation].append(alias)
            code = str(row.get("code", "")).strip()
            if code:
                for alias in parse_multi(row.get("code_aliases", "")):
                    if alias not in code_aliases[code]:
                        code_aliases[code].append(alias)
            article_neighbors[citation].extend(parse_multi(row.get("neighbors", "")))
        enrichment["statute_aliases"] = dict(statute_aliases)
        enrichment["code_aliases"] = dict(code_aliases)
        enrichment["article_neighbors"] = {key: sorted(set(values)) for key, values in article_neighbors.items()}
        source_meta["fedlex_articles"] = int(len(articles_df))

    cases_path = input_dir / "federal_cases.csv"
    if cases_path.exists():
        cases_df = pd.read_csv(cases_path)
        case_aliases = {}
        case_statute_graph: dict[str, Counter] = {}
        case_case_graph: dict[str, Counter] = {}
        for _, row in cases_df.iterrows():
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue
            for alias in parse_multi(row.get("aliases", "")):
                case_aliases[alias] = case_id
            add_weighted_edges(case_statute_graph, case_id, row.get("cited_statutes", ""))
            add_weighted_edges(case_case_graph, case_id, row.get("cited_cases", ""))
        enrichment["case_aliases"] = case_aliases
        enrichment["case_statute_graph"] = {key: dict(counter) for key, counter in case_statute_graph.items()}
        enrichment["case_case_graph"] = {key: dict(counter) for key, counter in case_case_graph.items()}
        source_meta["federal_cases"] = int(len(cases_df))

    enrichment["source_meta"] = source_meta
    return enrichment


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optional official Swiss enrichment artifacts for V44.")
    parser.add_argument("--input-dir", default="data/official_swiss", help="Directory containing official Swiss CSV exports.")
    parser.add_argument("--output", default=str(lira_pipeline.V44_OFFICIAL_ENRICHMENT_PATH), help="Output pickle path.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise RuntimeError(
            f"Missing official Swiss input directory: {input_dir}. "
            "Populate it with `fedlex_articles.csv` and/or `federal_cases.csv` before building enrichment."
        )

    enrichment = build_enrichment(input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lira_pipeline.save_pickle(output_path, enrichment)
    print(f"Saved official Swiss enrichment: {output_path}")
    print(f"Source counts: {enrichment.get('source_meta', {})}")


if __name__ == "__main__":
    main()


