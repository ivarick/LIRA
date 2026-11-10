#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lira_pipeline  # noqa: E402 - repository path is added above

FAST_CACHE_DIR = Path("cache/fast_v44_dev")


def stable_sort_key(value: str, seed: int) -> str:
    return hashlib.md5(f"{seed}|{value}".encode("utf-8")).hexdigest()


def stratified_sample_indices(train_df: pd.DataFrame, labels: Sequence[str], target_size: int, seed: int) -> List[int]:
    target_size = max(0, min(int(target_size), len(train_df)))
    if target_size == 0:
        return []
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[str(label)].append(idx)
    for label, indices in groups.items():
        indices.sort(key=lambda idx: stable_sort_key(str(train_df.iloc[idx]["query_id"]), seed))

    total = len(train_df)
    raw_quotas = {label: target_size * len(indices) / total for label, indices in groups.items()}
    taken = {label: min(len(indices), int(np.floor(quota))) for label, quota in raw_quotas.items()}
    current = sum(taken.values())
    remainders = sorted(
        groups.keys(),
        key=lambda label: (raw_quotas[label] - taken[label], len(groups[label]), label),
        reverse=True,
    )
    for label in remainders:
        if current >= target_size:
            break
        if taken[label] < len(groups[label]):
            taken[label] += 1
            current += 1

    selected: List[int] = []
    for label, count in taken.items():
        selected.extend(groups[label][:count])
    selected.sort(key=lambda idx: stable_sort_key(str(train_df.iloc[idx]["query_id"]), seed))
    return selected[:target_size]


def make_fast_dev_splits(
    train_df: pd.DataFrame,
    dev_size: int,
    tune_fraction: float,
    train_limit: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = lira_pipeline.make_bucket_labels(train_df)
    dev_indices = stratified_sample_indices(train_df, labels, dev_size, seed)
    dev_df = train_df.iloc[dev_indices].reset_index(drop=True)
    remaining_df = train_df.drop(train_df.index[dev_indices]).reset_index(drop=True)

    dev_labels = lira_pipeline.make_bucket_labels(dev_df)
    tune_size = int(round(len(dev_df) * float(tune_fraction)))
    tune_indices = stratified_sample_indices(dev_df, dev_labels, tune_size, seed + 1)
    tune_df = dev_df.iloc[tune_indices].reset_index(drop=True)
    eval_df = dev_df.drop(dev_df.index[tune_indices]).reset_index(drop=True)

    if train_limit > 0 and train_limit < len(remaining_df):
        rem_labels = lira_pipeline.make_bucket_labels(remaining_df)
        train_indices = stratified_sample_indices(remaining_df, rem_labels, train_limit, seed + 2)
        train_subset = remaining_df.iloc[train_indices].reset_index(drop=True)
    else:
        train_subset = remaining_df
    return train_subset, tune_df, eval_df


def fast_config(base: lira_pipeline.V44Config, mode: str) -> lira_pipeline.V44Config:
    if mode == "smoke":
        return replace(
            base,
            enable_case_bm25=False,
            enable_case_backproj=False,
            enable_case_rerank=False,
            case_dense_k=min(base.case_dense_k, 80),
            case_sparse_k=min(base.case_sparse_k, 80),
            case_bm25_k=0,
            case_backproj_k=0,
            case_profile_shortlist_k=min(base.case_profile_shortlist_k, 80),
            case_profile_rerank_k=0,
            case_direct_profile_limit=min(base.case_direct_profile_limit, 40),
            case_indirect_profile_limit=min(base.case_indirect_profile_limit, 20),
            case_indirect_citations_per_profile=min(base.case_indirect_citations_per_profile, 3),
            case_candidate_cap=min(base.case_candidate_cap, 120),
            laws_dense_k=min(base.laws_dense_k, 120),
            laws_sparse_k=min(base.laws_sparse_k, 120),
            laws_bm25_k=min(base.laws_bm25_k, 60),
            laws_union_pool_cap=min(base.laws_union_pool_cap, 500),
            laws_shortlist_k=min(base.laws_shortlist_k, 64),
            laws_rerank_k=min(base.laws_rerank_k, 32),
            case_count_clip=(0, 8),
            ranker_estimators=min(base.ranker_estimators, 80),
            count_model_estimators=min(base.count_model_estimators, 60),
        )
    if mode == "strong":
        return replace(
            base,
            case_dense_k=min(base.case_dense_k, 120),
            case_sparse_k=min(base.case_sparse_k, 120),
            case_bm25_k=min(base.case_bm25_k, 120),
            case_backproj_k=min(base.case_backproj_k, 120),
            case_profile_shortlist_k=min(base.case_profile_shortlist_k, 120),
            case_profile_rerank_k=min(base.case_profile_rerank_k, 32),
            case_direct_profile_limit=min(base.case_direct_profile_limit, 50),
            case_indirect_profile_limit=min(base.case_indirect_profile_limit, 25),
            case_indirect_citations_per_profile=min(base.case_indirect_citations_per_profile, 4),
            case_candidate_cap=min(base.case_candidate_cap, 160),
            laws_dense_k=min(base.laws_dense_k, 180),
            laws_sparse_k=min(base.laws_sparse_k, 180),
            laws_bm25_k=min(base.laws_bm25_k, 120),
            laws_union_pool_cap=min(base.laws_union_pool_cap, 900),
            laws_shortlist_k=min(base.laws_shortlist_k, 96),
            laws_rerank_k=min(base.laws_rerank_k, 96),
            case_count_clip=(0, 8),
            ranker_estimators=min(base.ranker_estimators, 120),
            count_model_estimators=min(base.count_model_estimators, 80),
        )
    return replace(
        base,
        case_dense_k=min(base.case_dense_k, 120),
        case_sparse_k=min(base.case_sparse_k, 120),
        case_bm25_k=min(base.case_bm25_k, 60),
        case_backproj_k=min(base.case_backproj_k, 80),
        case_profile_shortlist_k=min(base.case_profile_shortlist_k, 80),
        case_profile_rerank_k=min(base.case_profile_rerank_k, 16),
        case_direct_profile_limit=min(base.case_direct_profile_limit, 40),
        case_indirect_profile_limit=min(base.case_indirect_profile_limit, 20),
        case_indirect_citations_per_profile=min(base.case_indirect_citations_per_profile, 3),
        case_candidate_cap=min(base.case_candidate_cap, 120),
        laws_dense_k=min(base.laws_dense_k, 120),
        laws_sparse_k=min(base.laws_sparse_k, 120),
        laws_bm25_k=min(base.laws_bm25_k, 60),
        laws_union_pool_cap=min(base.laws_union_pool_cap, 600),
        laws_shortlist_k=min(base.laws_shortlist_k, 64),
        laws_rerank_k=min(base.laws_rerank_k, 64),
        case_count_clip=(0, 8),
        ranker_estimators=min(base.ranker_estimators, 100),
        count_model_estimators=min(base.count_model_estimators, 60),
    )


def build_cache_key(args: argparse.Namespace, cfg: lira_pipeline.V44Config) -> str:
    payload = {
        "pipeline_version": lira_pipeline.V44_PIPELINE_VERSION,
        "mode": args.mode,
        "dev_size": args.dev_size,
        "train_limit": args.train_limit,
        "tune_fraction": args.tune_fraction,
        "seed": args.seed,
        "reranker_device": lira_pipeline.resolve_reranker_device(),
        "reranker_batch_size": lira_pipeline.resolve_reranker_batch_size(lira_pipeline.resolve_reranker_device(), cfg),
        "config": {
            "case_dense_k": cfg.case_dense_k,
            "case_sparse_k": cfg.case_sparse_k,
            "case_bm25_k": cfg.case_bm25_k,
            "case_backproj_k": cfg.case_backproj_k,
            "case_profile_shortlist_k": cfg.case_profile_shortlist_k,
            "case_profile_rerank_k": cfg.case_profile_rerank_k,
            "case_direct_profile_limit": cfg.case_direct_profile_limit,
            "case_indirect_profile_limit": cfg.case_indirect_profile_limit,
            "case_indirect_citations_per_profile": cfg.case_indirect_citations_per_profile,
            "case_candidate_cap": cfg.case_candidate_cap,
            "laws_dense_k": cfg.laws_dense_k,
            "laws_sparse_k": cfg.laws_sparse_k,
            "laws_bm25_k": cfg.laws_bm25_k,
            "laws_union_pool_cap": cfg.laws_union_pool_cap,
            "laws_shortlist_k": cfg.laws_shortlist_k,
            "laws_rerank_k": cfg.laws_rerank_k,
            "enable_case_bm25": cfg.enable_case_bm25,
            "enable_case_backproj": cfg.enable_case_backproj,
            "enable_case_rerank": cfg.enable_case_rerank,
        },
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_cached_payload(cache_path: Path) -> Dict[str, object] | None:
    if not cache_path.exists():
        return None
    try:
        payload = lira_pipeline.load_pickle(cache_path)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("pipeline_version") != lira_pipeline.V44_PIPELINE_VERSION:
        return None
    return payload


def save_cached_payload(cache_path: Path, payload: Dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lira_pipeline.save_pickle(cache_path, payload)


def dominant_statute_features(row: pd.Series) -> List[str]:
    feature_scores = {
        "regex": 1.20 * float(row.get("regex_hit", 0.0)),
        "alias": 0.60 * float(row.get("official_alias_hit", 0.0)),
        "domain": 0.45 * float(row.get("domain_match", 0.0)),
        "dense": 0.90 * float(row.get("best_dense_score", 0.0)),
        "sparse": 1.35 * float(row.get("best_sparse_score", 0.0)),
        "bm25": 1.35 * float(row.get("best_bm25_score", 0.0)),
        "rrf": 0.55 * float(row.get("rrf_score", 0.0)),
        "graph": 0.30 * float(row.get("graph_case_support", 0.0)),
        "neighbor": 0.30 * float(row.get("official_neighbor_support", 0.0)),
        "coverage": 0.30 * float(row.get("channel_coverage", 0.0)) + 0.25 * float(row.get("view_coverage", 0.0)),
        "ranksum": 0.55 * float(row.get("reciprocal_rank_sum", 0.0)),
        "reranker": 1.00 * float(row.get("reranker_score", 0.0)),
    }
    ranked = [name for name, score in sorted(feature_scores.items(), key=lambda kv: (float(kv[1]), kv[0]), reverse=True) if score > 0.0]
    return ranked[:3]


def analyze_subset_prepare(resources: lira_pipeline.V44Resources, query_df: pd.DataFrame) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    statute_candidate_recall = []
    statute_top50_recall = []
    case_profile_overlap = []
    case_top50_recall = []
    union_sizes = []
    gold_total = 0
    gold_seen_dense = 0
    gold_seen_sparse = 0
    gold_seen_bm25 = 0
    gold_seen_regex = 0
    gold_seen_alias = 0
    misses: List[Dict[str, object]] = []
    for _, row in query_df.iterrows():
        golds = set(lira_pipeline.parse_citation_set(row["gold_citations"]))
        gold_statutes = {citation for citation in golds if lira_pipeline.get_citation_type(citation) == "statute"}
        gold_case_exact = {citation for citation in golds if lira_pipeline.get_citation_type(citation) in {"bge", "docket"}}
        gold_case_base = {lira_pipeline.get_case_group_id(citation) for citation in gold_case_exact}
        statute_df, _, profiles = lira_pipeline.build_statute_candidates(resources, str(row["query_id"]), str(row["query"]), golds, mode="support_only")
        statute_stats = dict(statute_df.attrs.get("statute_candidate_stats", {}))
        statute_cands = set(statute_df["citation"].tolist())
        statute_top50 = set(statute_df.head(50)["citation"].tolist())
        case_profile_top = {profile.base_case_id for profile in profiles[:200]}
        exact_profile_hits = {
            citation
            for profile in profiles[:50]
            for citation in profile.exact_case_ids
            if lira_pipeline.get_citation_type(citation) in {"bge", "docket"}
        }
        statute_candidate_recall.append(len(statute_cands & gold_statutes) / len(gold_statutes) if gold_statutes else 1.0)
        statute_top50_recall.append(len(statute_top50 & gold_statutes) / len(gold_statutes) if gold_statutes else 1.0)
        case_profile_overlap.append(len(case_profile_top & gold_case_base) / len(gold_case_base) if gold_case_base else 1.0)
        case_top50_recall.append(len(exact_profile_hits & gold_case_exact) / len(gold_case_exact) if gold_case_exact else 1.0)
        union_sizes.append(float(statute_stats.get("union_pool_size", len(statute_df))))
        statute_lookup = statute_df.set_index("citation", drop=False) if not statute_df.empty else pd.DataFrame()
        for citation in sorted(gold_statutes):
            gold_total += 1
            if citation in statute_lookup.index:
                stat_row = statute_lookup.loc[citation]
                gold_seen_dense += int(any(float(stat_row.get(f"{view}_dense_hit", 0.0)) > 0.0 for view in lira_pipeline.QUERY_VIEW_NAMES))
                gold_seen_sparse += int(any(float(stat_row.get(f"{view}_sparse_hit", 0.0)) > 0.0 for view in lira_pipeline.QUERY_VIEW_NAMES))
                gold_seen_bm25 += int(any(float(stat_row.get(f"{view}_bm25_hit", 0.0)) > 0.0 for view in lira_pipeline.QUERY_VIEW_NAMES))
                gold_seen_regex += int(float(stat_row.get("regex_hit", 0.0)) > 0.0)
                gold_seen_alias += int(float(stat_row.get("official_alias_hit", 0.0)) > 0.0)
            else:
                misses.append(
                    {
                        "query_id": str(row["query_id"]),
                        "query_text": str(row["query"]),
                        "missing_gold_statute": citation,
                        "stage": "union_pool",
                        "channel_hits": {"dense": False, "sparse": False, "bm25": False, "regex": False, "alias": False},
                        "top_false_positives": [],
                    }
                )
        top_false_positives = []
        for _, false_row in statute_df.head(10).iterrows():
            citation = str(false_row["citation"])
            if citation in gold_statutes:
                continue
            top_false_positives.append(
                {
                    "citation": citation,
                    "heuristic_score": float(false_row.get("heuristic_score", 0.0)),
                    "dominant_features": dominant_statute_features(false_row),
                }
            )
            if len(top_false_positives) >= 5:
                break
        for citation in sorted(gold_statutes - statute_top50):
            if citation not in statute_lookup.index:
                continue
            stat_row = statute_lookup.loc[citation]
            misses.append(
                {
                    "query_id": str(row["query_id"]),
                    "query_text": str(row["query"]),
                    "missing_gold_statute": citation,
                    "stage": "top50",
                    "channel_hits": {
                        "dense": any(float(stat_row.get(f"{view}_dense_hit", 0.0)) > 0.0 for view in lira_pipeline.QUERY_VIEW_NAMES),
                        "sparse": any(float(stat_row.get(f"{view}_sparse_hit", 0.0)) > 0.0 for view in lira_pipeline.QUERY_VIEW_NAMES),
                        "bm25": any(float(stat_row.get(f"{view}_bm25_hit", 0.0)) > 0.0 for view in lira_pipeline.QUERY_VIEW_NAMES),
                        "regex": float(stat_row.get("regex_hit", 0.0)) > 0.0,
                        "alias": float(stat_row.get("official_alias_hit", 0.0)) > 0.0,
                    },
                    "best_rank": int(stat_row.get("best_rank", 10**6)),
                    "pre_rerank_fusion": float(stat_row.get("pre_rerank_fusion", 0.0)),
                    "dominant_features": dominant_statute_features(stat_row),
                    "top_false_positives": top_false_positives,
                }
            )
    diag = {
        "statute_candidate_recall": float(np.mean(statute_candidate_recall)) if statute_candidate_recall else 0.0,
        "statute_top50_recall": float(np.mean(statute_top50_recall)) if statute_top50_recall else 0.0,
        "case_profile_top200_overlap": float(np.mean(case_profile_overlap)) if case_profile_overlap else 0.0,
        "case_exact_top50_recall": float(np.mean(case_top50_recall)) if case_top50_recall else 0.0,
        "mean_statute_union_size": float(np.mean(union_sizes)) if union_sizes else 0.0,
        "gold_seen_dense_fraction": float(gold_seen_dense / gold_total) if gold_total else 0.0,
        "gold_seen_sparse_fraction": float(gold_seen_sparse / gold_total) if gold_total else 0.0,
        "gold_seen_bm25_fraction": float(gold_seen_bm25 / gold_total) if gold_total else 0.0,
        "gold_seen_regex_fraction": float(gold_seen_regex / gold_total) if gold_total else 0.0,
        "gold_seen_alias_fraction": float(gold_seen_alias / gold_total) if gold_total else 0.0,
    }
    return diag, misses


def compute_subset_prepare_diag(resources: lira_pipeline.V44Resources, query_df: pd.DataFrame) -> Dict[str, float]:
    diag, _ = analyze_subset_prepare(resources, query_df)
    return diag


def build_training_candidates(resources: lira_pipeline.V44Resources, train_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for _, row in train_df.iterrows():
        golds = set(lira_pipeline.parse_citation_set(row["gold_citations"]))
        statute_df, _, _ = lira_pipeline.build_statute_candidates(resources, str(row["query_id"]), str(row["query"]), golds, mode="support_only")
        statute_df["fold"] = -1
        rows.append(statute_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def train_dev_bundle(resources: lira_pipeline.V44Resources, train_df: pd.DataFrame) -> Dict[str, object]:
    candidate_df = build_training_candidates(resources, train_df)
    if candidate_df.empty:
        raise RuntimeError("Training candidate table is empty in fast dev harness.")
    if "reranker_score" not in candidate_df.columns or float(candidate_df["reranker_score"].max()) <= 0.0:
        raise RuntimeError("Fast dev harness produced no active statute reranker scores.")
    feature_cols = lira_pipeline.get_statute_feature_columns(candidate_df)
    candidate_df = lira_pipeline.sanitize_statute_feature_frame(candidate_df, feature_cols)
    statute_model = lira_pipeline.train_ranker(candidate_df, feature_cols, resources.config)
    candidate_df["raw_score"] = lira_pipeline.predict_with_ranker(candidate_df, statute_model, feature_cols)
    statute_calibrator = lira_pipeline.fit_calibrator(candidate_df[[*feature_cols, "label"]].assign(raw_score=candidate_df["raw_score"]))
    candidate_df["prob"] = lira_pipeline.apply_calibrator(candidate_df.assign(raw_score=candidate_df["raw_score"]), statute_calibrator)
    count_model, count_feature_cols, query_count_df = lira_pipeline.train_statute_count_model(candidate_df, train_df, resources.config)
    return {
        "feature_cols": feature_cols,
        "statute_model": statute_model,
        "statute_calibrator": statute_calibrator,
        "statute_count_model": count_model,
        "statute_count_feature_cols": count_feature_cols,
        "train_query_count_summary": {
            "mean_gold_statute_count": float(query_count_df["gold_statute_count"].mean()),
            "mean_pred_statute_count": float(query_count_df["pred_statute_count"].mean()),
        },
    }


def build_scored_frames(resources: lira_pipeline.V44Resources, bundle: Dict[str, object], query_df: pd.DataFrame, desc: str) -> List[Dict[str, object]]:
    frames: List[Dict[str, object]] = []
    for _, row in lira_pipeline.maybe_tqdm(query_df.iterrows(), total=len(query_df), desc=desc):
        golds = lira_pipeline.parse_citation_set(row["gold_citations"])
        statute_df, case_feature_df, _ = lira_pipeline.build_statute_candidates(
            resources,
            str(row["query_id"]),
            str(row["query"]),
            set(),
            mode="full_case",
            case_evidence_params=None,
        )
        statute_df = lira_pipeline.score_statute_frame(
            statute_df,
            bundle["statute_model"],
            bundle["statute_calibrator"],
            bundle["feature_cols"],
        )
        frames.append(
            {
                "query_id": str(row["query_id"]),
                "query_text": str(row["query"]),
                "golds": golds,
                "gold_cases": [citation for citation in golds if lira_pipeline.get_citation_type(citation) in {"bge", "docket"}],
                "statute_df": statute_df,
                "case_feature_df": case_feature_df,
                "case_candidate_stats": dict(case_feature_df.attrs.get("candidate_stats", {})),
            }
        )
    return frames


def tune_statute_focus_selector(
    frames: Sequence[Dict[str, object]],
    bundle: Dict[str, object],
    config: lira_pipeline.V44Config,
    case_evidence_params: Dict[str, float],
) -> Tuple[Dict[str, object], float, Dict[str, float]]:
    base = lira_pipeline.default_selector_params(config)
    base["case_regex_bonus"] = 1.0
    grids = {
        "statute_multiplier": config.statute_multiplier_grid,
        "statute_floor": config.statute_floor_grid,
        "case_floor": (0.01, 0.02, 0.03),
        "case_rerank_threshold": (0.05, 0.10, 0.15),
    }
    best_params = dict(base)
    best_score, best_diag = lira_pipeline.evaluate_selector_on_frames(frames, bundle, config, best_params, case_evidence_params)
    for _ in range(2):
        improved = False
        for name, grid in grids.items():
            local_best_params = dict(best_params)
            local_best_score = float(best_score)
            local_best_diag = dict(best_diag)
            for value in grid:
                candidate = dict(best_params)
                candidate[name] = float(value)
                score, diag = lira_pipeline.evaluate_selector_on_frames(frames, bundle, config, candidate, case_evidence_params)
                if score > local_best_score:
                    local_best_params = candidate
                    local_best_score = float(score)
                    local_best_diag = dict(diag)
            if local_best_score > best_score:
                best_params = local_best_params
                best_score = local_best_score
                best_diag = local_best_diag
                improved = True
        if not improved:
            break
    return best_params, float(best_score), best_diag


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast directional V44 experiment harness.")
    parser.add_argument("--mode", choices=("smoke", "fast", "strong"), default="fast", help="Fast-harness preset.")
    parser.add_argument("--focus", choices=("statute", "balanced"), default="statute", help="Optimization focus for the harness.")
    parser.add_argument("--dev-size", type=int, default=12, help="Total train-dev queries to reserve for fast evaluation.")
    parser.add_argument("--train-limit", type=int, default=48, help="Max training queries used for the fast harness.")
    parser.add_argument("--tune-fraction", type=float, default=0.5, help="Fraction of dev queries used for tuning; rest are eval.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--report", type=Path, default=Path("output/fast_v44_dev_report.json"), help="Where to save the fast dev report.")
    parser.add_argument("--misses-report", type=Path, default=Path("output/fast_v44_dev_misses.json"), help="Where to save statute miss diagnostics.")
    args = parser.parse_args()

    start = time.perf_counter()
    cfg = fast_config(lira_pipeline.V44Config(), args.mode)
    cache_key = build_cache_key(args, cfg)
    cache_path = FAST_CACHE_DIR / f"{cache_key}.pkl"
    cached = load_cached_payload(cache_path)
    cache_hit = cached is not None

    if cached is None:
        lira_pipeline.prepare_artifacts(cfg)
        resources = lira_pipeline.load_resources(cfg, require_reranker=True)

        train_subset, tune_df, eval_df = make_fast_dev_splits(
            resources.train_df,
            dev_size=args.dev_size,
            tune_fraction=args.tune_fraction,
            train_limit=args.train_limit,
            seed=args.seed,
        )

        lira_pipeline.reset_runtime_diagnostics(resources)
        tune_prepare, tune_misses = analyze_subset_prepare(resources, tune_df)
        eval_prepare, eval_misses = analyze_subset_prepare(resources, eval_df)

        lira_pipeline.reset_runtime_diagnostics(resources)
        bundle = train_dev_bundle(resources, train_subset)
        support_runtime = lira_pipeline.summarize_runtime_diagnostics(resources)

        lira_pipeline.reset_runtime_diagnostics(resources)
        tune_frames = build_scored_frames(resources, bundle, tune_df, f"Fast V44 tune [{args.mode}]")
        tune_runtime = lira_pipeline.summarize_runtime_diagnostics(resources)

        lira_pipeline.reset_runtime_diagnostics(resources)
        eval_frames = build_scored_frames(resources, bundle, eval_df, f"Fast V44 eval [{args.mode}]")
        eval_runtime = lira_pipeline.summarize_runtime_diagnostics(resources)

        split_sizes = {
            "train_subset": int(len(train_subset)),
            "tune": int(len(tune_df)),
            "eval": int(len(eval_df)),
        }
        runtime_payload = {
            "support_only": support_runtime,
            "tune_frames": tune_runtime,
            "eval_frames": eval_runtime,
        }
        reranker_meta = {
            "reranker_device": resources.reranker_device,
            "reranker_batch_size": resources.reranker_batch_size,
        }
        cached = {
            "pipeline_version": lira_pipeline.V44_PIPELINE_VERSION,
            "cache_key": cache_key,
            "mode": args.mode,
            "bundle": bundle,
            "tune_frames": tune_frames,
            "eval_frames": eval_frames,
            "prepare_tune": tune_prepare,
            "prepare_eval": eval_prepare,
            "tune_misses": tune_misses,
            "eval_misses": eval_misses,
            "split_sizes": split_sizes,
            "runtime_payload": runtime_payload,
            "reranker_meta": reranker_meta,
        }
        save_cached_payload(cache_path, cached)
    else:
        bundle = cached["bundle"]
        tune_frames = cached["tune_frames"]
        eval_frames = cached["eval_frames"]
        tune_prepare = cached["prepare_tune"]
        eval_prepare = cached["prepare_eval"]
        tune_misses = cached.get("tune_misses", [])
        eval_misses = cached.get("eval_misses", [])
        split_sizes = cached["split_sizes"]
        runtime_payload = cached["runtime_payload"]
        reranker_meta = cached["reranker_meta"]

    tune_diag = lira_pipeline.compute_frame_diagnostics(tune_frames, bundle, cfg)
    if args.focus == "balanced":
        case_params, case_tune_score, case_tune_diag = lira_pipeline.tune_case_evidence_on_oof(tune_frames, cfg)
        if args.mode == "smoke":
            selector_params = lira_pipeline.default_selector_params(cfg)
            selector_tune_score, selector_tune_diag = lira_pipeline.evaluate_selector_on_frames(tune_frames, bundle, cfg, selector_params, case_params)
        else:
            selector_params, selector_tune_score, selector_tune_diag = lira_pipeline.tune_selector_on_oof(tune_frames, bundle, cfg, case_params)
    else:
        case_params = lira_pipeline.default_case_evidence_params(cfg)
        case_tune_score, case_tune_diag = lira_pipeline.evaluate_case_evidence_on_frames(tune_frames, cfg, case_params)
        if args.mode == "smoke":
            selector_params = lira_pipeline.default_selector_params(cfg)
            selector_tune_score, selector_tune_diag = lira_pipeline.evaluate_selector_on_frames(tune_frames, bundle, cfg, selector_params, case_params)
        else:
            selector_params, selector_tune_score, selector_tune_diag = tune_statute_focus_selector(tune_frames, bundle, cfg, case_params)

    eval_diag = lira_pipeline.compute_frame_diagnostics(eval_frames, bundle, cfg)
    case_eval_score, case_eval_diag = lira_pipeline.evaluate_case_evidence_on_frames(eval_frames, cfg, case_params)
    selector_eval_score, selector_eval_diag = lira_pipeline.evaluate_selector_on_frames(eval_frames, bundle, cfg, selector_params, case_params)

    if args.mode == "strong":
        acceptance = {
            "mean_pred_case_count_ok": float(selector_eval_diag["mean_pred_case_count"]) <= 8.0,
            "statute_candidate_recall_ok": float(eval_prepare["statute_candidate_recall"]) >= 0.78,
            "statute_top50_recall_ok": float(eval_prepare["statute_top50_recall"]) >= 0.50,
            "selector_f1_ok": float(selector_eval_score) >= 0.26,
        }
    else:
        acceptance = {
            "mean_pred_case_count_ok": float(selector_eval_diag["mean_pred_case_count"]) <= 8.0,
            "statute_candidate_recall_ok": float(eval_prepare["statute_candidate_recall"]) >= 0.72,
            "statute_top50_recall_ok": float(eval_prepare["statute_top50_recall"]) >= 0.45,
            "selector_f1_ok": float(selector_eval_score) >= 0.24,
        }
    acceptance["wall_clock_advisory_ok"] = float(time.perf_counter() - start) <= 5400.0
    acceptance["case_guard_ok"] = (
        float(eval_prepare["case_profile_top200_overlap"]) >= 0.95
        and float(eval_prepare["case_exact_top50_recall"]) >= 0.90
        and float(selector_eval_diag["mean_pred_case_count"]) <= 8.0
    )
    acceptance["promote_to_full_cv"] = all(
        acceptance[key]
        for key in ("mean_pred_case_count_ok", "statute_candidate_recall_ok", "statute_top50_recall_ok", "selector_f1_ok", "case_guard_ok")
    )

    report = {
        "pipeline_version": lira_pipeline.V44_PIPELINE_VERSION,
        "fast_config": {
            "mode": args.mode,
            "focus": args.focus,
            "dev_size": args.dev_size,
            "train_limit": args.train_limit,
            "tune_fraction": args.tune_fraction,
            "seed": args.seed,
            "reranker_device": reranker_meta["reranker_device"],
            "reranker_batch_size": reranker_meta["reranker_batch_size"],
            "case_profile_rerank_k": cfg.case_profile_rerank_k,
            "laws_union_pool_cap": cfg.laws_union_pool_cap,
            "laws_rerank_k": cfg.laws_rerank_k,
            "case_candidate_cap": cfg.case_candidate_cap,
            "case_count_clip": list(cfg.case_count_clip),
        },
        "split_sizes": split_sizes,
        "cache": {
            "cache_key": cache_key,
            "cache_path": str(cache_path),
            "cache_hit": cache_hit,
        },
        "prepare_tune": tune_prepare,
        "prepare_eval": eval_prepare,
        "prepare_misses": {
            "tune": tune_misses,
            "eval": eval_misses,
        },
        "train_summary": bundle["train_query_count_summary"],
        "tune": {
            "frame_diagnostics": tune_diag,
            "case_evidence_f1": float(case_tune_score),
            "case_evidence_diag": case_tune_diag,
            "selector_f1": float(selector_tune_score),
            "selector_diag": selector_tune_diag,
        },
        "eval": {
            "frame_diagnostics": eval_diag,
            "case_evidence_f1": float(case_eval_score),
            "case_evidence_diag": case_eval_diag,
            "selector_f1": float(selector_eval_score),
            "selector_diag": selector_eval_diag,
        },
        "runtimes": {**runtime_payload, "wall_clock_sec": float(time.perf_counter() - start)},
        "selector_params": selector_params,
        "case_evidence_params": case_params,
        "changed_knobs": {
            "enable_case_bm25": cfg.enable_case_bm25,
            "enable_case_backproj": cfg.enable_case_backproj,
            "enable_case_rerank": cfg.enable_case_rerank,
            "case_direct_profile_limit": cfg.case_direct_profile_limit,
            "case_indirect_profile_limit": cfg.case_indirect_profile_limit,
            "case_indirect_citations_per_profile": cfg.case_indirect_citations_per_profile,
            "laws_union_pool_cap": cfg.laws_union_pool_cap,
            "laws_shortlist_k": cfg.laws_shortlist_k,
            "laws_rerank_k": cfg.laws_rerank_k,
        },
        "acceptance": acceptance,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.misses_report.parent.mkdir(parents=True, exist_ok=True)
    args.misses_report.write_text(
        json.dumps(
            {
                "pipeline_version": lira_pipeline.V44_PIPELINE_VERSION,
                "focus": args.focus,
                "mode": args.mode,
                "tune": tune_misses,
                "eval": eval_misses,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Fast V44 report: {args.report}")
    print(
        f"Split sizes: train={split_sizes['train_subset']} tune={split_sizes['tune']} eval={split_sizes['eval']} "
        f"device={reranker_meta['reranker_device']} batch={reranker_meta['reranker_batch_size']} mode={args.mode} focus={args.focus}"
    )
    print(
        f"Prepare eval: statute_candidate_recall={eval_prepare['statute_candidate_recall']:.4f} "
        f"statute_top50={eval_prepare['statute_top50_recall']:.4f} "
        f"statute_union={eval_prepare['mean_statute_union_size']:.1f} "
        f"case_profile_top200={eval_prepare['case_profile_top200_overlap']:.4f} "
        f"case_exact_top50={eval_prepare['case_exact_top50_recall']:.4f}"
    )
    print(
        f"Tune: case_f1={case_tune_score:.4f} selector_f1={selector_tune_score:.4f} "
        f"emit_statutes={selector_tune_diag['mean_pred_statute_count']:.2f} "
        f"emit_cases={selector_tune_diag['mean_pred_case_count']:.2f}"
    )
    print(
        f"Eval: case_f1={case_eval_score:.4f} selector_f1={selector_eval_score:.4f} "
        f"emit_statutes={selector_eval_diag['mean_pred_statute_count']:.2f} "
        f"emit_cases={selector_eval_diag['mean_pred_case_count']:.2f}"
    )
    print(f"Wall clock: {report['runtimes']['wall_clock_sec']:.1f}s")
    print(
        f"Gates: statute_recall={acceptance['statute_candidate_recall_ok']} "
        f"statute_top50={acceptance['statute_top50_recall_ok']} "
        f"emit_cases={acceptance['mean_pred_case_count_ok']} "
        f"selector_f1={acceptance['selector_f1_ok']} "
        f"promote_to_full_cv={acceptance['promote_to_full_cv']}"
    )


if __name__ == "__main__":
    main()


