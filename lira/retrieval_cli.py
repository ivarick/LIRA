"""LIRA retrieval cli stage."""

from .retrieval_core import (
    DATA_PATH,
    FOLDS_PATH,
    OOF_SCORED_PATH,
    OUTPUT_PATH,
    Path,
    V43Config,
    argparse,
    build_court_sparse_index,
    build_faiss_index,
    build_folds,
    ensure_laws_sparse_postings,
    load_encoder,
    load_pickle,
    load_resources,
    log,
    macro_f1_from_sets,
    np,
    parse_citation_set,
    pd,
    random,
    save_pickle,
)
from .retrieval_training import (
    debug_candidate_recall,
    generate_predictions_for_split,
    load_model_bundle,
    run_oof_ranking,
    save_model_bundle,
    summarize_fold_balance,
    train_full_models,
)
from .retrieval_candidates import (
    generate_fold_candidate_tables,
    generate_full_train_candidate_table,
)

def cmd_prepare(args: argparse.Namespace, config: V43Config) -> None:
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    train_df = pd.read_csv(DATA_PATH / "train.csv").dropna(subset=["query", "gold_citations"]).reset_index(drop=True)
    if FOLDS_PATH.exists():
        folds = load_pickle(FOLDS_PATH)
        log(f"   Folds: {FOLDS_PATH.name}")
    else:
        folds = build_folds(train_df, config)
        save_pickle(FOLDS_PATH, folds)
        log(f"   Saved: {FOLDS_PATH}")
    summarize_fold_balance(train_df, folds, config)
    ensure_laws_sparse_postings(config)
    if not args.skip_faiss:
        build_faiss_index(config)
    if args.build_court_sparse:
        encoder = load_encoder(config)
        build_court_sparse_index(config, encoder)


def cmd_cv(args: argparse.Namespace, config: V43Config) -> None:
    resources = load_resources(config, require_court_sparse=args.require_court_sparse)
    debug_df, debug_diag = debug_candidate_recall(resources, split="val")
    log(
        "Debug val candidate recall: "
        f"statute={debug_diag['candidate_recall_statute']:.4f} "
        f"case={debug_diag['candidate_recall_case']:.4f} "
        f"top20_statute={debug_diag['top20_recall_statute']:.4f} "
        f"top20_case={debug_diag['top20_recall_case']:.4f} "
        f"case_overlap={debug_diag['case_gold_overlap_rate']:.4f}"
    )
    if debug_diag["candidate_recall_case"] <= 0.0:
        raise RuntimeError("Case candidate recall is still zero after citation-universe rewrite. Aborting retrain.")

    candidate_df = generate_fold_candidate_tables(resources)
    if candidate_df.empty:
        raise RuntimeError("Candidate table generation returned no rows.")
    oof_df, metadata = run_oof_ranking(candidate_df, resources)
    oof_df.to_pickle(OOF_SCORED_PATH)
    log(f"Saved OOF candidates: {OOF_SCORED_PATH}")
    log(f"OOF Macro F1: {metadata['oof_macro_f1']:.4f}")
    log(
        "OOF candidate diagnostics: "
        f"statute={metadata['oof_diagnostics']['candidate_recall_statute']:.4f} "
        f"case={metadata['oof_diagnostics']['candidate_recall_case']:.4f} "
        f"top20_statute={metadata['oof_diagnostics']['top20_recall_statute']:.4f} "
        f"top20_case={metadata['oof_diagnostics']['top20_recall_case']:.4f} "
        f"case_overlap={metadata['oof_diagnostics']['case_gold_overlap_rate']:.4f}"
    )
    log(
        "Selection params: "
        f"statute_floor={metadata['selection_params']['statute_floor']:.4f} "
        f"case_floor={metadata['selection_params']['case_floor']:.4f}"
    )
    log(
        "Selection diagnostics: "
        f"mean_predicted_statute_count={metadata['selection_diagnostics']['mean_predicted_statute_count']:.2f} "
        f"mean_predicted_case_count={metadata['selection_diagnostics']['mean_predicted_case_count']:.2f}"
    )

    if args.fit_final:
        full_train_candidates = generate_full_train_candidate_table(resources)
        bundle = train_full_models(full_train_candidates, metadata, resources)
        save_model_bundle(bundle)


def cmd_predict(args: argparse.Namespace, config: V43Config) -> None:
    bundle = load_model_bundle()
    resources = load_resources(config, require_court_sparse=args.require_court_sparse)
    pred_df = generate_predictions_for_split(resources, bundle, args.split)
    out_path = Path(args.output) if args.output else OUTPUT_PATH / f"{args.split}_v43.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    log(f"Saved predictions: {out_path}")

    if args.split in {"train", "val"}:
        gold_df = resources.train_df if args.split == "train" else resources.val_df
        pred_lookup = dict(zip(pred_df["query_id"], pred_df["predicted_citations"]))
        preds = [parse_citation_set(pred_lookup.get(qid, "")) for qid in gold_df["query_id"].tolist()]
        golds = [parse_citation_set(value) for value in gold_df["gold_citations"].tolist()]
        score = macro_f1_from_sets(preds, golds)
        log(f"{args.split} Macro F1: {score:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LIRA V43 retrieval + LambdaMART pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build folds and retrieval artifacts")
    prepare.add_argument("--skip-faiss", action="store_true", help="Skip building the court FAISS index")
    prepare.add_argument("--build-court-sparse", action="store_true", help="Build the court sparse postings artifact")

    cv = subparsers.add_parser("cv", help="Run out-of-fold candidate generation and ranking")
    cv.add_argument("--fit-final", action="store_true", help="Train and save final models after OOF tuning")
    cv.add_argument("--require-court-sparse", action="store_true", help="Abort if the court sparse artifact is missing")

    predict = subparsers.add_parser("predict", help="Generate predictions for a split")
    predict.add_argument("--split", choices=["train", "val", "test"], default="val")
    predict.add_argument("--output", type=str, default="")
    predict.add_argument("--require-court-sparse", action="store_true", help="Abort if the court sparse artifact is missing")

    val = subparsers.add_parser("val", help="Shortcut for predict --split val")
    val.add_argument("--output", type=str, default="")
    val.add_argument("--require-court-sparse", action="store_true", help="Abort if the court sparse artifact is missing")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = V43Config()

    if args.command == "prepare":
        cmd_prepare(args, config)
    elif args.command == "cv":
        cmd_cv(args, config)
    elif args.command == "predict":
        cmd_predict(args, config)
    elif args.command == "val":
        args.split = "val"
        cmd_predict(args, config)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")


