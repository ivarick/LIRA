"""LIRA retrieval training stage."""

from .retrieval_candidates import (
    COUNT_FEATURE_META_COLUMNS,
    V43CandidateGenerator,
    get_feature_columns,
)
from .retrieval_core import (
    Dict,
    List,
    LogisticRegression,
    MODEL_BUNDLE_PATH,
    Optional,
    Tuple,
    V43Config,
    V43Resources,
    V43_MODEL_PATH,
    V43_PIPELINE_VERSION,
    detect_domain_codes,
    get_citation_type,
    lgb,
    load_pickle,
    log,
    macro_f1_from_sets,
    maybe_tqdm,
    np,
    parse_citation_set,
    pd,
    re,
    save_pickle,
)

def train_ranker(df: pd.DataFrame, feature_cols: List[str], config: V43Config) -> Optional[lgb.LGBMRanker]:
    if df.empty:
        return None
    y = df["label"].astype(int)
    if y.nunique() < 2:
        return None
    ranked = df.sort_values(["query_id", "heuristic_score"], ascending=[True, False]).reset_index(drop=True)
    groups = ranked.groupby("query_id").size().tolist()
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=config.ranker_estimators,
        learning_rate=config.ranker_learning_rate,
        num_leaves=config.ranker_num_leaves,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.random_seed,
        verbosity=-1,
    )
    model.fit(ranked[feature_cols], ranked["label"].astype(int), group=groups)
    return model


def predict_with_ranker(df: pd.DataFrame, model: Optional[lgb.LGBMRanker], feature_cols: List[str]) -> np.ndarray:
    if df.empty:
        return np.zeros(0, dtype=np.float32)
    if model is None:
        return df["heuristic_score"].to_numpy(dtype=np.float32)
    return model.predict(df[feature_cols]).astype(np.float32)


def fit_calibrator(df: pd.DataFrame) -> Optional[LogisticRegression]:
    if df.empty or df["label"].nunique() < 2:
        return None
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(df[["raw_score"]], df["label"].astype(int))
    return model


def apply_calibrator(df: pd.DataFrame, calibrator: Optional[LogisticRegression]) -> np.ndarray:
    if df.empty:
        return np.zeros(0, dtype=np.float32)
    if calibrator is None:
        raw = df["raw_score"].to_numpy(dtype=np.float32)
        centered = raw - np.median(raw) if len(raw) else raw
        return 1.0 / (1.0 + np.exp(-centered))
    return calibrator.predict_proba(df[["raw_score"]])[:, 1].astype(np.float32)

def get_count_feature_columns(query_df: pd.DataFrame) -> List[str]:
    return [col for col in query_df.columns if col not in COUNT_FEATURE_META_COLUMNS]


def threshold_label(value: float) -> str:
    return str(value).replace(".", "p")


def build_query_count_features(group_df: pd.DataFrame, query_text: str, config: V43Config) -> Dict[str, float]:
    def features_for_type(frame: pd.DataFrame, prefix: str) -> Dict[str, float]:
        frame = frame.sort_values("prob", ascending=False)
        probs = frame["prob"].to_numpy(dtype=np.float32)
        supports = frame["support_total"].to_numpy(dtype=np.float32) if "support_total" in frame.columns else np.zeros(0, dtype=np.float32)
        top10 = frame.head(10)
        feats = {
            f"{prefix}_candidate_count": float(len(frame)),
            f"{prefix}_prob_at_1": float(probs[0]) if len(probs) >= 1 else 0.0,
            f"{prefix}_prob_at_3": float(probs[2]) if len(probs) >= 3 else 0.0,
            f"{prefix}_prob_at_5": float(probs[4]) if len(probs) >= 5 else 0.0,
            f"{prefix}_prob_sum_top5": float(probs[:5].sum()) if len(probs) else 0.0,
            f"{prefix}_prob_sum_top10": float(probs[:10].sum()) if len(probs) else 0.0,
            f"{prefix}_support_max_top10": float(supports[:10].max()) if len(supports) else 0.0,
            f"{prefix}_support_mean_top10": float(supports[:10].mean()) if len(supports) else 0.0,
            f"{prefix}_regex_hit_count": float(top10["regex_hit"].sum()) if "regex_hit" in top10.columns else 0.0,
        }
        for threshold in config.query_prob_thresholds:
            feats[f"{prefix}_count_ge_{threshold_label(threshold)}"] = float((probs >= threshold).sum())
        return feats

    statute_df = group_df[group_df["citation_type"] == "statute"]
    case_df = group_df[group_df["citation_type"] != "statute"]
    query_tokens = re.findall(r"\b\w+\b", str(query_text).lower())
    features = {
        "query_token_length": float(len(query_tokens)),
        "query_domain_code_count": float(len(detect_domain_codes(query_text))),
    }
    features.update(features_for_type(statute_df, "statute"))
    features.update(features_for_type(case_df, "case"))
    return features


def build_query_count_frame(candidate_df: pd.DataFrame, query_df: pd.DataFrame, config: V43Config) -> pd.DataFrame:
    query_info = query_df.set_index("query_id")
    rows: List[Dict[str, object]] = []
    for query_id, group in candidate_df.groupby("query_id", sort=False):
        query_text = str(query_info.loc[query_id, "query"])
        features = build_query_count_features(group, query_text, config)
        golds = parse_citation_set(query_info.loc[query_id, "gold_citations"])
        rows.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "gold_statute_count": sum(1 for citation in golds if get_citation_type(citation) == "statute"),
                "gold_case_count": sum(1 for citation in golds if get_citation_type(citation) != "statute"),
                **features,
            }
        )
    return pd.DataFrame(rows)


def train_count_model(query_df: pd.DataFrame, target_col: str, feature_cols: List[str], config: V43Config) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=config.count_model_estimators,
        learning_rate=config.count_model_learning_rate,
        num_leaves=config.count_model_num_leaves,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=config.random_seed,
        verbosity=-1,
    )
    model.fit(query_df[feature_cols], query_df[target_col].astype(np.float32))
    return model


def predict_count(model: lgb.LGBMRegressor, query_features: pd.DataFrame, clip_bounds: Tuple[int, int]) -> np.ndarray:
    predicted = model.predict(query_features).astype(np.float32)
    rounded = np.rint(np.clip(predicted, clip_bounds[0], clip_bounds[1]))
    return rounded.astype(np.int32)


def select_predictions_with_counts(
    group_df: pd.DataFrame,
    statute_count: int,
    case_count: int,
    selection_params: Dict[str, object],
) -> List[str]:
    statute_df = group_df[group_df["citation_type"] == "statute"].sort_values("prob", ascending=False)
    case_df = group_df[group_df["citation_type"] != "statute"].sort_values("prob", ascending=False)

    statute_preds = [
        citation
        for citation, prob in zip(statute_df["citation"], statute_df["prob"])
        if citation and float(prob) >= float(selection_params["statute_floor"])
    ][: int(statute_count)]
    case_preds = [
        citation
        for citation, prob in zip(case_df["citation"], case_df["prob"])
        if citation and float(prob) >= float(selection_params["case_floor"])
    ][: int(case_count)]

    if int(statute_count) == 0 and not statute_preds and not statute_df.empty:
        top_row = statute_df.iloc[0]
        if float(top_row["prob"]) >= float(selection_params["min_emit_if_high_prob"]):
            statute_preds = [str(top_row["citation"])]

    if int(case_count) == 0 and not case_preds and not case_df.empty:
        top_row = case_df.iloc[0]
        if float(top_row["prob"]) >= float(selection_params["min_emit_if_high_prob"]):
            case_preds = [str(top_row["citation"])]

    merged: List[str] = []
    seen = set()
    for citation in statute_preds + case_preds:
        if citation not in seen:
            merged.append(citation)
            seen.add(citation)
    return merged


def compute_candidate_diagnostics(candidate_df: pd.DataFrame, score_col: str = "heuristic_score", top_k: int = 20) -> Dict[str, float]:
    stats = []
    for _, group in candidate_df.groupby("query_id", sort=False):
        ranked = group.sort_values(score_col, ascending=False).head(top_k)
        gold_stat = set(group.loc[(group["label"] == 1) & (group["citation_type"] == "statute"), "citation"])
        gold_case = set(group.loc[(group["label"] == 1) & (group["citation_type"] != "statute"), "citation"])
        cand_stat = set(group.loc[group["citation_type"] == "statute", "citation"])
        cand_case = set(group.loc[group["citation_type"] != "statute", "citation"])
        top_stat = set(ranked.loc[ranked["citation_type"] == "statute", "citation"])
        top_case = set(ranked.loc[ranked["citation_type"] != "statute", "citation"])
        stats.append(
            {
                "candidate_recall_statute": len(cand_stat & gold_stat) / len(gold_stat) if gold_stat else 1.0,
                "candidate_recall_case": len(cand_case & gold_case) / len(gold_case) if gold_case else 1.0,
                "top20_recall_statute": len(top_stat & gold_stat) / len(gold_stat) if gold_stat else 1.0,
                "top20_recall_case": len(top_case & gold_case) / len(gold_case) if gold_case else 1.0,
                "case_gold_overlap_rate": 1.0 if not gold_case else float(len(cand_case & gold_case) > 0),
            }
        )
    diag_df = pd.DataFrame(stats)
    return {key: float(diag_df[key].mean()) for key in diag_df.columns}


def debug_candidate_recall(resources: V43Resources, split: str = "val") -> Tuple[pd.DataFrame, Dict[str, float]]:
    generator = V43CandidateGenerator(resources)
    query_df = resources.val_df if split == "val" else resources.train_df
    all_train_indices = np.arange(len(resources.train_df))
    rows: List[pd.DataFrame] = []
    for idx, row in maybe_tqdm(query_df.iterrows(), total=len(query_df), desc=f"Debug {split} candidates"):
        knn_pool = all_train_indices
        if split == "train":
            knn_pool = all_train_indices[all_train_indices != idx]
        candidate_df = generator.generate_query_candidates(
            query_id=str(row["query_id"]),
            query=str(row["query"]),
            gold_set=set(parse_citation_set(row["gold_citations"])),
            knn_pool_indices=knn_pool,
            preserve_gold_positives=False,
        )
        if candidate_df.empty:
            candidate_df = pd.DataFrame([generator._new_row(str(row["query_id"]), "", set(), detect_domain_codes(str(row["query"])))])
        rows.append(candidate_df)
    debug_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return debug_df, compute_candidate_diagnostics(debug_df, score_col="heuristic_score", top_k=20)


def train_query_count_models(
    scored_candidate_df: pd.DataFrame,
    query_df: pd.DataFrame,
    config: V43Config,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    query_count_df = build_query_count_frame(scored_candidate_df, query_df, config)
    feature_cols = get_count_feature_columns(query_count_df)
    statute_model = train_count_model(query_count_df, "gold_statute_count", feature_cols, config)
    case_model = train_count_model(query_count_df, "gold_case_count", feature_cols, config)
    query_count_df = query_count_df.copy()
    query_count_df["pred_statute_count"] = predict_count(statute_model, query_count_df[feature_cols], config.statute_count_clip)
    query_count_df["pred_case_count"] = predict_count(case_model, query_count_df[feature_cols], config.case_count_clip)
    return query_count_df, {
        "count_feature_cols": feature_cols,
        "statute_count_model": statute_model,
        "case_count_model": case_model,
    }


def tune_selection(
    candidate_df: pd.DataFrame,
    query_count_df: pd.DataFrame,
    config: V43Config,
) -> Tuple[Dict[str, object], float, Dict[str, float]]:
    query_groups = {query_id: group.copy() for query_id, group in candidate_df.groupby("query_id", sort=False)}
    gold_lookup = {
        query_id: group.loc[group["label"] == 1, "citation"].tolist()
        for query_id, group in query_groups.items()
    }
    count_lookup = query_count_df.set_index("query_id")[["pred_statute_count", "pred_case_count"]].to_dict("index")

    best_params = {
        "statute_floor": 0.01,
        "case_floor": 0.01,
        "statute_clip": config.statute_count_clip,
        "case_clip": config.case_count_clip,
        "min_emit_if_high_prob": config.min_emit_if_high_prob,
    }
    best_f1 = -1.0
    best_diag = {"mean_predicted_statute_count": 0.0, "mean_predicted_case_count": 0.0}
    for statute_floor in config.statute_floor_grid:
        for case_floor in config.case_floor_grid:
            params = {
                "statute_floor": float(statute_floor),
                "case_floor": float(case_floor),
                "statute_clip": config.statute_count_clip,
                "case_clip": config.case_count_clip,
                "min_emit_if_high_prob": config.min_emit_if_high_prob,
            }
            preds = []
            pred_stat_counts = []
            pred_case_counts = []
            for query_id, group in query_groups.items():
                counts = count_lookup[query_id]
                pred_stat_counts.append(int(counts["pred_statute_count"]))
                pred_case_counts.append(int(counts["pred_case_count"]))
                preds.append(
                    select_predictions_with_counts(
                        group,
                        statute_count=int(counts["pred_statute_count"]),
                        case_count=int(counts["pred_case_count"]),
                        selection_params=params,
                    )
                )
            golds = [gold_lookup[query_id] for query_id in query_groups.keys()]
            macro_f1 = macro_f1_from_sets(preds, golds)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_params = params
                best_diag = {
                    "mean_predicted_statute_count": float(np.mean(pred_stat_counts)) if pred_stat_counts else 0.0,
                    "mean_predicted_case_count": float(np.mean(pred_case_counts)) if pred_case_counts else 0.0,
                }
    return best_params, float(best_f1), best_diag


def run_oof_ranking(candidate_df: pd.DataFrame, resources: V43Resources) -> Tuple[pd.DataFrame, Dict[str, object]]:
    candidate_df = candidate_df.copy()
    feature_cols = get_feature_columns(candidate_df)
    candidate_df["raw_score"] = 0.0

    for fold in range(resources.config.n_splits):
        train_fold = candidate_df[candidate_df["fold"] != fold].copy()
        valid_fold = candidate_df[candidate_df["fold"] == fold].copy()
        statute_train = train_fold[train_fold["citation_type"] == "statute"].copy()
        case_train = train_fold[train_fold["citation_type"] != "statute"].copy()
        statute_valid = valid_fold[valid_fold["citation_type"] == "statute"].copy()
        case_valid = valid_fold[valid_fold["citation_type"] != "statute"].copy()

        statute_model = train_ranker(statute_train, feature_cols, resources.config)
        case_model = train_ranker(case_train, feature_cols, resources.config)

        if not statute_valid.empty:
            candidate_df.loc[statute_valid.index, "raw_score"] = predict_with_ranker(statute_valid, statute_model, feature_cols)
        if not case_valid.empty:
            candidate_df.loc[case_valid.index, "raw_score"] = predict_with_ranker(case_valid, case_model, feature_cols)

    statute_df = candidate_df[candidate_df["citation_type"] == "statute"].copy()
    case_df = candidate_df[candidate_df["citation_type"] != "statute"].copy()
    statute_calibrator = fit_calibrator(statute_df)
    case_calibrator = fit_calibrator(case_df)
    candidate_df["prob"] = 0.0
    if not statute_df.empty:
        candidate_df.loc[statute_df.index, "prob"] = apply_calibrator(statute_df, statute_calibrator)
    if not case_df.empty:
        candidate_df.loc[case_df.index, "prob"] = apply_calibrator(case_df, case_calibrator)

    query_count_df, count_models = train_query_count_models(candidate_df, resources.train_df, resources.config)
    selection_params, best_f1, selection_diag = tune_selection(candidate_df, query_count_df, resources.config)
    oof_diag = compute_candidate_diagnostics(candidate_df, score_col="prob", top_k=20)
    metadata = {
        "feature_cols": feature_cols,
        "statute_calibrator": statute_calibrator,
        "case_calibrator": case_calibrator,
        "selection_params": selection_params,
        "oof_macro_f1": best_f1,
        "query_count_df": query_count_df,
        "count_feature_cols": count_models["count_feature_cols"],
        "statute_count_model_oof": count_models["statute_count_model"],
        "case_count_model_oof": count_models["case_count_model"],
        "oof_diagnostics": oof_diag,
        "selection_diagnostics": selection_diag,
    }
    return candidate_df, metadata


def score_candidate_frame(candidate_df: pd.DataFrame, bundle: Dict[str, object]) -> pd.DataFrame:
    candidate_df = candidate_df.copy()
    feature_cols = bundle["feature_cols"]
    statute_mask = candidate_df["citation_type"] == "statute"
    case_mask = ~statute_mask
    candidate_df["raw_score"] = 0.0
    candidate_df.loc[statute_mask, "raw_score"] = predict_with_ranker(
        candidate_df.loc[statute_mask], bundle["statute_model"], feature_cols
    )
    candidate_df.loc[case_mask, "raw_score"] = predict_with_ranker(
        candidate_df.loc[case_mask], bundle["case_model"], feature_cols
    )
    candidate_df["prob"] = 0.0
    if statute_mask.any():
        candidate_df.loc[statute_mask, "prob"] = apply_calibrator(
            candidate_df.loc[statute_mask], bundle["statute_calibrator"]
        )
    if case_mask.any():
        candidate_df.loc[case_mask, "prob"] = apply_calibrator(
            candidate_df.loc[case_mask], bundle["case_calibrator"]
        )
    return candidate_df


def train_full_models(full_train_candidates: pd.DataFrame, metadata: Dict[str, object], resources: V43Resources) -> Dict[str, object]:
    feature_cols = list(metadata["feature_cols"])
    statute_df = full_train_candidates[full_train_candidates["citation_type"] == "statute"].copy()
    case_df = full_train_candidates[full_train_candidates["citation_type"] != "statute"].copy()
    statute_model = train_ranker(statute_df, feature_cols, resources.config)
    case_model = train_ranker(case_df, feature_cols, resources.config)

    full_train_candidates = full_train_candidates.copy()
    full_train_candidates["raw_score"] = 0.0
    if not statute_df.empty:
        full_train_candidates.loc[statute_df.index, "raw_score"] = predict_with_ranker(statute_df, statute_model, feature_cols)
    if not case_df.empty:
        full_train_candidates.loc[case_df.index, "raw_score"] = predict_with_ranker(case_df, case_model, feature_cols)

    full_statute_calibrator = fit_calibrator(full_train_candidates[full_train_candidates["citation_type"] == "statute"])
    full_case_calibrator = fit_calibrator(full_train_candidates[full_train_candidates["citation_type"] != "statute"])

    provisional_bundle = {
        "feature_cols": feature_cols,
        "statute_model": statute_model,
        "case_model": case_model,
        "statute_calibrator": full_statute_calibrator,
        "case_calibrator": full_case_calibrator,
    }
    scored_full_train = score_candidate_frame(full_train_candidates, provisional_bundle)
    full_query_count_df, count_models = train_query_count_models(scored_full_train, resources.train_df, resources.config)

    return {
        "bundle_version": V43_PIPELINE_VERSION,
        "feature_cols": feature_cols,
        "statute_model": statute_model,
        "case_model": case_model,
        "statute_calibrator": full_statute_calibrator,
        "case_calibrator": full_case_calibrator,
        "statute_count_model": count_models["statute_count_model"],
        "case_count_model": count_models["case_count_model"],
        "count_feature_cols": count_models["count_feature_cols"],
        "statute_floor": metadata["selection_params"]["statute_floor"],
        "case_floor": metadata["selection_params"]["case_floor"],
        "statute_clip": resources.config.statute_count_clip,
        "case_clip": resources.config.case_count_clip,
        "min_emit_if_high_prob": resources.config.min_emit_if_high_prob,
        "query_prob_thresholds": resources.config.query_prob_thresholds,
        "selection_params": metadata["selection_params"],
        "oof_macro_f1": metadata["oof_macro_f1"],
        "oof_diagnostics": metadata["oof_diagnostics"],
        "selection_diagnostics": metadata["selection_diagnostics"],
        "citation_universe": resources.citation_universe.to_metadata(),
        "train_query_count_summary": {
            "mean_gold_statute_count": float(full_query_count_df["gold_statute_count"].mean()),
            "mean_gold_case_count": float(full_query_count_df["gold_case_count"].mean()),
            "mean_pred_statute_count": float(full_query_count_df["pred_statute_count"].mean()),
            "mean_pred_case_count": float(full_query_count_df["pred_case_count"].mean()),
        },
    }


def select_predictions(group_df: pd.DataFrame, bundle: Dict[str, object], query_text: str) -> List[str]:
    config = V43Config(query_prob_thresholds=tuple(bundle.get("query_prob_thresholds", V43Config().query_prob_thresholds)))
    query_feature_frame = pd.DataFrame([build_query_count_features(group_df, query_text, config)])
    count_feature_cols = bundle["count_feature_cols"]
    statute_count = int(predict_count(bundle["statute_count_model"], query_feature_frame[count_feature_cols], tuple(bundle["statute_clip"]))[0])
    case_count = int(predict_count(bundle["case_count_model"], query_feature_frame[count_feature_cols], tuple(bundle["case_clip"]))[0])
    return select_predictions_with_counts(
        group_df,
        statute_count=statute_count,
        case_count=case_count,
        selection_params={
            "statute_floor": bundle["statute_floor"],
            "case_floor": bundle["case_floor"],
            "min_emit_if_high_prob": bundle["min_emit_if_high_prob"],
        },
    )


def generate_predictions_for_split(
    resources: V43Resources,
    bundle: Dict[str, object],
    split: str,
) -> pd.DataFrame:
    generator = V43CandidateGenerator(resources)
    if split == "train":
        df = resources.train_df
    elif split == "val":
        df = resources.val_df
    elif split == "test":
        df = resources.test_df
    else:
        raise RuntimeError(f"Unknown split: {split}")

    rows: List[Dict[str, object]] = []
    all_train_indices = np.arange(len(resources.train_df))
    for idx, row in maybe_tqdm(df.reset_index(drop=True).iterrows(), total=len(df), desc=f"Predict {split}"):
        knn_pool = all_train_indices
        if split == "train":
            knn_pool = all_train_indices[all_train_indices != idx]
        candidate_df = generator.generate_query_candidates(
            query_id=str(row["query_id"]),
            query=str(row["query"]),
            gold_set=set(parse_citation_set(row.get("gold_citations", ""))),
            knn_pool_indices=knn_pool,
            preserve_gold_positives=False,
        )
        if candidate_df.empty:
            predictions: List[str] = []
        else:
            scored = score_candidate_frame(candidate_df, bundle)
            predictions = select_predictions(scored, bundle, str(row["query"]))
        rows.append({"query_id": row["query_id"], "predicted_citations": ";".join(predictions)})
    return pd.DataFrame(rows)


def summarize_fold_balance(train_df: pd.DataFrame, folds: Dict[str, int], config: V43Config) -> None:
    stats = []
    for fold in range(config.n_splits):
        fold_df = train_df[train_df["query_id"].map(folds).astype(int) == fold]
        gold_lists = [parse_citation_set(value) for value in fold_df["gold_citations"].tolist()]
        mean_gold = float(np.mean([len(g) for g in gold_lists])) if gold_lists else 0.0
        has_case = int(
            sum(1 for gold in gold_lists if any(get_citation_type(cit) != "statute" for cit in gold))
        )
        stats.append((fold, len(fold_df), mean_gold, has_case))
    for fold, size, mean_gold, has_case in stats:
        log(f"   Fold {fold}: queries={size}, mean_gold={mean_gold:.2f}, with_case={has_case}")


def save_model_bundle(bundle: Dict[str, object]) -> None:
    V43_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    save_pickle(MODEL_BUNDLE_PATH, bundle)
    log(f"Saved model bundle: {MODEL_BUNDLE_PATH}")


def load_model_bundle() -> Dict[str, object]:
    if not MODEL_BUNDLE_PATH.exists():
        raise RuntimeError(f"Missing {MODEL_BUNDLE_PATH}. Run `python lira_retrieval.py cv --fit-final` first.")
    bundle = load_pickle(MODEL_BUNDLE_PATH)
    if bundle.get("bundle_version") != V43_PIPELINE_VERSION:
        raise RuntimeError(
            f"{MODEL_BUNDLE_PATH} is stale for pipeline {V43_PIPELINE_VERSION}. "
            "Run `python lira_retrieval.py cv --fit-final` again."
        )
    required = {
        "statute_count_model",
        "case_count_model",
        "statute_floor",
        "case_floor",
        "count_feature_cols",
        "statute_clip",
        "case_clip",
    }
    missing = sorted(required - set(bundle.keys()))
    if missing:
        raise RuntimeError(
            f"{MODEL_BUNDLE_PATH} is missing required V43.1 keys: {missing}. "
            "Run `python lira_retrieval.py cv --fit-final` again."
        )
    return bundle


