"""LIRA cli stage."""

from .core import (
    Counter,
    Dict,
    OUTPUT_PATH,
    STATUTE_CANDIDATE_SCHEMA_VERSION,
    STATUTE_NON_FEATURE_COLUMNS,
    V44Config,
    V44_MODEL_BUNDLE_PATH,
    V44_PIPELINE_VERSION,
    argparse,
    default_case_evidence_params,
    load_resources,
    log,
    macro_f1_from_sets,
    os,
    parse_citation_set,
    prepare_artifacts,
    reset_runtime_diagnostics,
    summarize_runtime_diagnostics,
    validate_kaggle_manifest,
)
from .training import (
    build_oof_case_frames,
    build_val_scored_frames,
    compute_frame_diagnostics,
    compute_prepare_diagnostics,
    evaluate_selector_on_frames,
    generate_full_statute_candidates,
    generate_oof_statute_candidates,
    load_model_bundle,
    predict_split,
    run_oof_statute_training,
    save_model_bundle,
    train_full_statute_model,
    tune_case_evidence_on_oof,
    tune_selector_on_oof,
)
from .selection import (
    default_selector_params,
)

def cmd_prepare(args: argparse.Namespace, config: V44Config) -> None:
    prepare_artifacts(config)
    resources = load_resources(config, require_reranker=False)
    diag = compute_prepare_diagnostics(resources)
    log(
        f"Prepare diagnostics: statute_candidate_recall={diag['statute_candidate_recall']:.4f} "
        f"statute_top50_recall={diag['statute_top50_recall']:.4f} "
        f"case_profile_top200_overlap={diag['case_profile_top200_overlap']:.4f} "
        f"case_exact_top50_recall={diag['case_exact_top50_recall']:.4f}"
    )
    if diag["statute_candidate_recall"] < 0.95 or diag["statute_top50_recall"] < 0.60 or diag["case_profile_top200_overlap"] < 0.80 or diag["case_exact_top50_recall"] < 0.35:
        raise RuntimeError("V44 prepare diagnostics below acceptance gates")


def cmd_cv(args: argparse.Namespace, config: V44Config) -> None:
    prepare_artifacts(config)
    resources = load_resources(config, require_reranker=True)
    reset_runtime_diagnostics(resources)
    oof_candidates = generate_oof_statute_candidates(resources)
    scored_oof_candidates, metadata = run_oof_statute_training(resources, oof_candidates)
    full_candidates = generate_full_statute_candidates(resources)
    full_bundle = train_full_statute_model(resources, full_candidates, metadata)
    support_runtime_diag = summarize_runtime_diagnostics(resources)
    selector_params = default_selector_params(config)
    case_evidence_params = default_case_evidence_params(config)
    selector_score = 0.0
    selector_diag = {"mean_pred_statute_count": 0.0, "mean_pred_case_count": 0.0}
    case_evidence_score = 0.0
    case_evidence_diag = {"mean_pred_case_count": 0.0}
    full_runtime_diag: Dict[str, float] = {}
    oof_frame_diag: Dict[str, float] = {}
    val_frame_diag: Dict[str, float] = {}
    val_report_score = 0.0
    oof_bundle_core = {
        "feature_cols": metadata["feature_cols"],
        "statute_count_model": metadata["statute_count_model_oof"],
        "statute_count_feature_cols": metadata["statute_count_feature_cols"],
    }
    full_bundle_core = {**full_bundle, "feature_cols": metadata["feature_cols"]}
    oof_frames, full_runtime_diag = build_oof_case_frames(resources, scored_oof_candidates)
    oof_frame_diag = compute_frame_diagnostics(oof_frames, oof_bundle_core, config)
    if oof_frame_diag["statute_reranker_active_fraction"] < 0.95:
        raise RuntimeError("Statute reranker is inactive on too many OOF queries")
    if oof_frame_diag["case_reranker_active_fraction"] < 0.95:
        raise RuntimeError("Case reranker is inactive on too many OOF queries")
    if oof_frame_diag["mean_case_candidates_after_prune"] > float(config.case_candidate_cap):
        raise RuntimeError("Pruned case candidate count exceeds the quality gate")
    case_evidence_params, case_evidence_score, case_evidence_diag = tune_case_evidence_on_oof(oof_frames, config)
    if case_evidence_score <= 0.0:
        raise RuntimeError("Case evidence tuning produced zero OOF Macro F1; selector tuning aborted")
    selector_params, selector_score, selector_diag = tune_selector_on_oof(oof_frames, oof_bundle_core, config, case_evidence_params)
    if selector_diag.get("mean_pred_case_count", 0.0) >= 20.0 and case_evidence_score <= 0.01:
        raise RuntimeError("Selector tuning produced a degenerate case budget under near-zero OOF case evidence quality")
    if args.report_val:
        val_frames, _ = build_val_scored_frames(resources, full_bundle_core, config)
        val_frame_diag = compute_frame_diagnostics(val_frames, full_bundle_core, config)
        val_report_score, _ = evaluate_selector_on_frames(val_frames, full_bundle_core, config, selector_params, case_evidence_params)
    bundle = {
        "bundle_version": V44_PIPELINE_VERSION,
        "feature_cols": metadata["feature_cols"],
        "statute_feature_schema": {
            "feature_cols": list(metadata["feature_cols"]),
            "excluded_columns": sorted(STATUTE_NON_FEATURE_COLUMNS),
            "pipeline_version": V44_PIPELINE_VERSION,
            "schema_version": STATUTE_CANDIDATE_SCHEMA_VERSION,
        },
        "statute_model": full_bundle["statute_model"],
        "statute_calibrator": full_bundle["statute_calibrator"],
        "statute_count_model": full_bundle["statute_count_model"],
        "statute_count_feature_cols": full_bundle["statute_count_feature_cols"],
        "case_evidence_params": case_evidence_params,
        "selector_params": selector_params,
        "case_evidence_oof_f1": float(case_evidence_score),
        "selector_tuned_oof_f1": float(selector_score),
        "selector_val_report_f1": float(val_report_score) if args.report_val else None,
        "case_evidence_diagnostics": case_evidence_diag,
        "selector_diagnostics": selector_diag,
        "statute_clip": config.statute_count_clip,
        "case_clip": config.case_count_clip,
        "oof_statute_macro_f1": metadata["oof_macro_f1"],
        "train_query_count_summary": full_bundle["train_query_count_summary"],
        "citation_universe": resources.citation_universe.to_metadata(),
        "query_views_meta": {"count": len(resources.query_views), "sources": dict(Counter(view.source for view in resources.query_views.values()))},
        "case_evidence_settings": {
            "rrf_k": config.rrf_k,
            "profile_shortlist_k": config.case_profile_shortlist_k,
            "profile_rerank_k": config.case_profile_rerank_k,
            "params": case_evidence_params,
        },
        "reranker_required": True,
        "reranker_device": resources.reranker_device,
        "reranker_batch_size": resources.reranker_batch_size,
        "case_pruning_config": {
            "direct_profile_limit": config.case_direct_profile_limit,
            "indirect_profile_limit": config.case_indirect_profile_limit,
            "indirect_citations_per_profile": config.case_indirect_citations_per_profile,
            "candidate_cap": config.case_candidate_cap,
        },
        "case_probability_mode": "softmax_or_rank_decay_v1",
        "case_runtime_gates": {
            "statute_reranker_active_fraction_min": 0.95,
            "case_reranker_active_fraction_min": 0.95,
            "mean_case_candidates_after_prune_max": float(config.case_candidate_cap),
        },
        "offline_contract": {
            "predict_rebuilds_artifacts": False,
            "predict_uses_manifest": True,
            "query_view_policy": "deterministic_default",
            "llm_required": False,
        },
        "oof_frame_diagnostics": oof_frame_diag,
        "val_frame_diagnostics": val_frame_diag,
        "runtime_diagnostics": {
            "support_only": support_runtime_diag,
            "full_case": full_runtime_diag,
        },
    }
    save_model_bundle(bundle)
    mean_gold = float(full_bundle["train_query_count_summary"]["mean_gold_statute_count"])
    mean_pred = float(full_bundle["train_query_count_summary"]["mean_pred_statute_count"])
    log(f"OOF statute Macro F1: {metadata['oof_macro_f1']:.4f}")
    log(f"Case evidence OOF Macro F1: {case_evidence_score:.4f}")
    log(f"Selector tuned OOF Macro F1: {selector_score:.4f}")
    if args.report_val:
        log(f"Frozen-bundle val report Macro F1: {val_report_score:.4f}")
    log(f"Train statute count summary: mean_gold={mean_gold:.2f} mean_pred={mean_pred:.2f}")
    log(
        "Support-only runtime: "
        f"query_view={support_runtime_diag.get('query_view_encode_sec', 0.0):.3f}s "
        f"case_retrieval={support_runtime_diag.get('case_candidate_retrieval_sec', 0.0):.3f}s "
        f"case_rerank={support_runtime_diag.get('case_rerank_sec', 0.0):.3f}s "
        f"statute_rerank={support_runtime_diag.get('statute_rerank_sec', 0.0):.3f}s "
        f"total={support_runtime_diag.get('build_total_sec', 0.0):.3f}s"
    )
    log(
        "OOF case-frame runtime: "
        f"query_view={full_runtime_diag.get('query_view_encode_sec', 0.0):.3f}s "
        f"case_retrieval={full_runtime_diag.get('case_candidate_retrieval_sec', 0.0):.3f}s "
        f"case_rerank={full_runtime_diag.get('case_rerank_sec', 0.0):.3f}s "
        f"statute_rerank={full_runtime_diag.get('statute_rerank_sec', 0.0):.3f}s "
        f"total={full_runtime_diag.get('build_total_sec', 0.0):.3f}s"
    )
    log(
        "Candidate summary: "
        f"mean_statutes={support_runtime_diag.get('statute_candidate_count', 0.0):.1f} "
        f"mean_cases_oof={full_runtime_diag.get('case_candidate_count_full', 0.0):.1f}"
    )
    log(
        "OOF frame diagnostics: "
        f"statute_reranker_active={oof_frame_diag.get('statute_reranker_active_fraction', 0.0):.2f} "
        f"case_reranker_active={oof_frame_diag.get('case_reranker_active_fraction', 0.0):.2f} "
        f"case_pre_prune={oof_frame_diag.get('mean_case_candidates_before_prune', 0.0):.1f} "
        f"case_post_prune={oof_frame_diag.get('mean_case_candidates_after_prune', 0.0):.1f} "
        f"case_direct={oof_frame_diag.get('mean_direct_exact_case_candidates', 0.0):.1f} "
        f"case_indirect={oof_frame_diag.get('mean_indirect_case_candidates', 0.0):.1f} "
        f"case_mass_top10={oof_frame_diag.get('mean_top10_case_mass', 0.0):.3f} "
        f"default_emit_statutes={oof_frame_diag.get('mean_emit_statutes_default', 0.0):.2f} "
        f"default_emit_cases={oof_frame_diag.get('mean_emit_cases_default', 0.0):.2f}"
    )
    if args.report_val:
        log(
            "Val frame diagnostics: "
            f"statute_reranker_active={val_frame_diag.get('statute_reranker_active_fraction', 0.0):.2f} "
            f"case_reranker_active={val_frame_diag.get('case_reranker_active_fraction', 0.0):.2f} "
            f"case_pre_prune={val_frame_diag.get('mean_case_candidates_before_prune', 0.0):.1f} "
            f"case_post_prune={val_frame_diag.get('mean_case_candidates_after_prune', 0.0):.1f} "
            f"case_direct={val_frame_diag.get('mean_direct_exact_case_candidates', 0.0):.1f} "
            f"case_indirect={val_frame_diag.get('mean_indirect_case_candidates', 0.0):.1f} "
            f"case_mass_top10={val_frame_diag.get('mean_top10_case_mass', 0.0):.3f} "
            f"default_emit_statutes={val_frame_diag.get('mean_emit_statutes_default', 0.0):.2f} "
            f"default_emit_cases={val_frame_diag.get('mean_emit_cases_default', 0.0):.2f}"
        )
    log(
        "Selector summary: "
        f"mean_emit_statutes={selector_diag.get('mean_pred_statute_count', 0.0):.2f} "
        f"mean_emit_cases={selector_diag.get('mean_pred_case_count', 0.0):.2f}"
    )
    if mean_gold > 0 and abs(mean_pred - mean_gold) / mean_gold > 0.20:
        raise RuntimeError("Mean predicted statute count deviates by more than 20% from train gold statute count")
    log(f"Saved model bundle: {V44_MODEL_BUNDLE_PATH}")


def cmd_val(args: argparse.Namespace, config: V44Config) -> None:
    safe_mode = os.getenv("LIRA_V44_DISABLE_RERANKER", "0").strip().lower() in {"1", "true", "yes", "on"}
    resources = load_resources(config, require_reranker=not safe_mode)
    reset_runtime_diagnostics(resources)
    bundle = load_model_bundle()
    pred_df = predict_split(resources, bundle, split="val")
    output_path = OUTPUT_PATH / "val_v44.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    gold_lookup = resources.val_df.set_index("query_id")
    preds = [parse_citation_set(value) for value in pred_df["predicted_citations"].tolist()]
    golds = [parse_citation_set(gold_lookup.loc[query_id, "gold_citations"]) for query_id in pred_df["query_id"].tolist()]
    score = macro_f1_from_sets(preds, golds)
    runtime_diag = summarize_runtime_diagnostics(resources)
    log(f"Saved predictions: {output_path}")
    log(f"val Macro F1: {score:.4f}")
    log(
        "Val runtime: "
        f"query_view={runtime_diag.get('query_view_encode_sec', 0.0):.3f}s "
        f"case_retrieval={runtime_diag.get('case_candidate_retrieval_sec', 0.0):.3f}s "
        f"case_rerank={runtime_diag.get('case_rerank_sec', 0.0):.3f}s "
        f"statute_rerank={runtime_diag.get('statute_rerank_sec', 0.0):.3f}s "
        f"total={runtime_diag.get('build_total_sec', 0.0):.3f}s"
    )


def cmd_predict(args: argparse.Namespace, config: V44Config) -> None:
    safe_mode = os.getenv("LIRA_V44_DISABLE_RERANKER", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not safe_mode:
        validate_kaggle_manifest(require_bundle=True)
    resources = load_resources(config, require_reranker=not safe_mode)
    bundle = load_model_bundle()
    pred_df = predict_split(resources, bundle, split="test")
    output_path = OUTPUT_PATH / "submission_v44.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    log(f"Saved predictions: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LIRA V44 split retrieval pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Build V44 artifacts and run retrieval diagnostics")
    prepare_parser.set_defaults(func=cmd_prepare)
    cv_parser = subparsers.add_parser("cv", help="Train V44 statute ranker and tune OOF selector")
    cv_parser.add_argument("--report-val", action="store_true", help="Also report the frozen bundle on validation")
    cv_parser.set_defaults(func=cmd_cv)
    val_parser = subparsers.add_parser("val", help="Run V44 on val split")
    val_parser.set_defaults(func=cmd_val)
    predict_parser = subparsers.add_parser("predict", help="Run V44 on test split")
    predict_parser.set_defaults(func=cmd_predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = V44Config()
    args.func(args, config)


