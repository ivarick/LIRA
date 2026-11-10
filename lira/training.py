"""LIRA training stage."""

from .core import (
    Dict,
    List,
    STATUTE_REQUIRED_CANDIDATE_COLUMNS,
    ScoredCaseProfile,
    Sequence,
    Tuple,
    V44Config,
    V44Resources,
    V44_CACHE_PATH,
    V44_FULL_STATUTE_PATH,
    V44_MODEL_BUNDLE_PATH,
    V44_MODEL_PATH,
    V44_OOF_STATUTE_PATH,
    V44_PIPELINE_VERSION,
    apply_calibrator,
    default_case_evidence_params,
    encode_query_views_for_query,
    fit_calibrator,
    get_case_group_id,
    get_citation_type,
    load_cached_frame,
    load_pickle,
    log,
    macro_f1_from_sets,
    maybe_tqdm,
    np,
    os,
    parse_citation_set,
    pd,
    predict_with_ranker,
    record_runtime_metric,
    reset_runtime_diagnostics,
    save_cached_frame,
    save_pickle,
    summarize_runtime_diagnostics,
    time,
    train_ranker,
    write_kaggle_manifest,
)
from .cases import (
    apply_case_evidence_model,
    build_case_candidate_frame,
    rerank_case_profiles,
    retrieve_case_profile_candidates,
)
from .statutes import (
    build_statute_candidates,
    build_statute_count_features,
    get_query_runtime_state,
    get_statute_feature_columns,
    sanitize_statute_feature_frame,
    score_statute_frame,
    train_statute_count_model,
    validate_statute_feature_cols,
)
from .selection import (
    classify_query_mode,
    compute_case_budget,
    default_selector_params,
    merge_predictions,
    select_case_predictions,
    select_predictions,
    select_statute_predictions,
)

def compute_prepare_diagnostics(resources: V44Resources) -> Dict[str, float]:
    statute_candidate_recall = []
    statute_top50_recall = []
    case_profile_overlap = []
    case_top50_recall = []
    for _, row in maybe_tqdm(resources.val_df.iterrows(), total=len(resources.val_df), desc="Prepare diagnostics"):
        golds = parse_citation_set(row["gold_citations"])
        gold_statutes = {citation for citation in golds if get_citation_type(citation) == "statute"}
        gold_case_exact = {citation for citation in golds if get_citation_type(citation) in {"bge", "docket"}}
        gold_case_base = {get_case_group_id(citation) for citation in gold_case_exact}
        statute_df, _, profiles = build_statute_candidates(resources, str(row["query_id"]), str(row["query"]), set(golds), mode="support_only")
        statute_cands = set(statute_df["citation"].tolist())
        statute_top50 = set(statute_df.head(50)["citation"].tolist())
        case_profile_top = {profile.base_case_id for profile in profiles[:200]}
        statute_candidate_recall.append(len(statute_cands & gold_statutes) / len(gold_statutes) if gold_statutes else 1.0)
        statute_top50_recall.append(len(statute_top50 & gold_statutes) / len(gold_statutes) if gold_statutes else 1.0)
        case_profile_overlap.append(len(case_profile_top & gold_case_base) / len(gold_case_base) if gold_case_base else 1.0)
        exact_profile_hits = {
            citation
            for profile in profiles[:50]
            for citation in profile.exact_case_ids
            if get_citation_type(citation) in {"bge", "docket"}
        }
        case_top50_recall.append(len(exact_profile_hits & gold_case_exact) / len(gold_case_exact) if gold_case_exact else 1.0)
    return {
        "statute_candidate_recall": float(np.mean(statute_candidate_recall)) if statute_candidate_recall else 0.0,
        "statute_top50_recall": float(np.mean(statute_top50_recall)) if statute_top50_recall else 0.0,
        "case_profile_top200_overlap": float(np.mean(case_profile_overlap)) if case_profile_overlap else 0.0,
        "case_exact_top50_recall": float(np.mean(case_top50_recall)) if case_top50_recall else 0.0,
    }


def generate_oof_statute_candidates(resources: V44Resources) -> pd.DataFrame:
    cached_oof = load_cached_frame(
        V44_OOF_STATUTE_PATH,
        cache_kind="oof_statute",
        required_columns=STATUTE_REQUIRED_CANDIDATE_COLUMNS,
    )
    if cached_oof is not None:
        return cached_oof
    rows: List[pd.DataFrame] = []
    for fold in range(resources.config.n_splits):
        fold_path = V44_CACHE_PATH / f"fold_{fold}_{V44_PIPELINE_VERSION}.pkl"
        cached_fold = load_cached_frame(
            fold_path,
            cache_kind="fold_statute",
            required_columns=STATUTE_REQUIRED_CANDIDATE_COLUMNS,
        )
        if cached_fold is not None:
            rows.append(cached_fold)
            log(f"   V44 fold {fold}: cached")
            continue
        heldout_mask = resources.train_df["query_id"].map(resources.folds).astype(int) == fold
        heldout_df = resources.train_df.loc[heldout_mask].reset_index(drop=True)
        fold_rows: List[pd.DataFrame] = []
        for _, row in maybe_tqdm(heldout_df.iterrows(), total=len(heldout_df), desc=f"V44 fold {fold}"):
            gold_set = set(parse_citation_set(row["gold_citations"]))
            statute_df, _, _ = build_statute_candidates(resources, str(row["query_id"]), str(row["query"]), gold_set, mode="support_only")
            statute_df["fold"] = fold
            fold_rows.append(statute_df)
        fold_df = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
        save_cached_frame(fold_df, fold_path, cache_kind="fold_statute")
        rows.append(fold_df)
    oof_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    save_cached_frame(oof_df, V44_OOF_STATUTE_PATH, cache_kind="oof_statute")
    return oof_df


def generate_full_statute_candidates(resources: V44Resources) -> pd.DataFrame:
    cached_full = load_cached_frame(
        V44_FULL_STATUTE_PATH,
        cache_kind="full_statute",
        required_columns=STATUTE_REQUIRED_CANDIDATE_COLUMNS,
    )
    if cached_full is not None:
        return cached_full
    rows: List[pd.DataFrame] = []
    for _, row in maybe_tqdm(resources.train_df.iterrows(), total=len(resources.train_df), desc="V44 full-train statutes"):
        gold_set = set(parse_citation_set(row["gold_citations"]))
        statute_df, _, _ = build_statute_candidates(resources, str(row["query_id"]), str(row["query"]), gold_set, mode="support_only")
        statute_df["fold"] = -1
        rows.append(statute_df)
    full_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    save_cached_frame(full_df, V44_FULL_STATUTE_PATH, cache_kind="full_statute")
    return full_df


def run_oof_statute_training(resources: V44Resources, candidate_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    candidate_df = candidate_df.copy()
    if resources.require_reranker and ("reranker_score" not in candidate_df.columns or float(candidate_df["reranker_score"].max()) <= 0.0):
        raise RuntimeError("OOF statute candidates do not contain active reranker scores")
    feature_cols = get_statute_feature_columns(candidate_df)
    validate_statute_feature_cols(feature_cols)
    if resources.require_reranker and "reranker_score" not in feature_cols:
        raise RuntimeError("Statute reranker feature is missing from the OOF training schema")
    candidate_df = sanitize_statute_feature_frame(candidate_df, feature_cols)
    candidate_df["raw_score"] = 0.0
    for fold in range(resources.config.n_splits):
        train_fold = sanitize_statute_feature_frame(candidate_df[candidate_df["fold"] != fold].copy(), feature_cols)
        valid_fold = sanitize_statute_feature_frame(candidate_df[candidate_df["fold"] == fold].copy(), feature_cols)
        model = train_ranker(train_fold, feature_cols, resources.config)
        if not valid_fold.empty:
            candidate_df.loc[valid_fold.index, "raw_score"] = predict_with_ranker(valid_fold, model, feature_cols)
    calibrator = fit_calibrator(candidate_df[[*feature_cols, "label"]].assign(raw_score=candidate_df["raw_score"]))
    candidate_df["prob"] = apply_calibrator(candidate_df.assign(raw_score=candidate_df["raw_score"]), calibrator)
    count_model, count_feature_cols, query_count_df = train_statute_count_model(candidate_df, resources.train_df, resources.config)
    oof_preds = []
    oof_golds = []
    for _, row in resources.train_df.iterrows():
        query_id = str(row["query_id"])
        group = candidate_df[candidate_df["query_id"] == query_id].sort_values("prob", ascending=False)
        query_features = pd.DataFrame([build_statute_count_features(group, str(row["query"]), resources.config)])
        pred_count = int(np.clip(np.rint(count_model.predict(query_features[count_feature_cols])[0]), resources.config.statute_count_clip[0], resources.config.statute_count_clip[1]))
        preds = [citation for citation, prob in zip(group["citation"], group["prob"]) if float(prob) >= 0.01][:pred_count]
        oof_preds.append(preds)
        oof_golds.append([citation for citation in parse_citation_set(row["gold_citations"]) if get_citation_type(citation) == "statute"])
    oof_macro_f1 = macro_f1_from_sets(oof_preds, oof_golds)
    return candidate_df, {
        "feature_cols": feature_cols,
        "calibrator": calibrator,
        "oof_macro_f1": float(oof_macro_f1),
        "statute_count_model_oof": count_model,
        "statute_count_feature_cols": count_feature_cols,
        "query_count_df": query_count_df,
    }


def train_full_statute_model(resources: V44Resources, full_df: pd.DataFrame, metadata: Dict[str, object]) -> Dict[str, object]:
    feature_cols = list(metadata["feature_cols"])
    validate_statute_feature_cols(feature_cols)
    if resources.require_reranker and ("reranker_score" not in full_df.columns or float(full_df["reranker_score"].max()) <= 0.0):
        raise RuntimeError("Full-train statute candidates do not contain active reranker scores")
    full_df = sanitize_statute_feature_frame(full_df.copy(), feature_cols)
    model = train_ranker(full_df, feature_cols, resources.config)
    full_df["raw_score"] = predict_with_ranker(full_df, model, feature_cols)
    calibrator = fit_calibrator(full_df[[*feature_cols, "label"]].assign(raw_score=full_df["raw_score"]))
    full_df["prob"] = apply_calibrator(full_df.assign(raw_score=full_df["raw_score"]), calibrator)
    count_model, count_feature_cols, query_count_df = train_statute_count_model(full_df, resources.train_df, resources.config)
    return {
        "statute_model": model,
        "statute_calibrator": calibrator,
        "statute_count_model": count_model,
        "statute_count_feature_cols": count_feature_cols,
        "train_query_count_summary": {
            "mean_gold_statute_count": float(query_count_df["gold_statute_count"].mean()),
            "mean_pred_statute_count": float(query_count_df["pred_statute_count"].mean()),
        },
    }

def build_case_feature_frame_for_query(
    resources: V44Resources,
    query_id: str,
    query: str,
) -> Tuple[pd.DataFrame, Dict[str, float], List[ScoredCaseProfile]]:
    total_start = time.perf_counter()
    query_state = get_query_runtime_state(resources, query_id, query)
    encoded_views = query_state.get("encoded_views")
    if encoded_views is None:
        stage_start = time.perf_counter()
        encoded_views = encode_query_views_for_query(resources, query_id, query)
        query_state["encoded_views"] = encoded_views
        record_runtime_metric(resources, "query_view_encode_sec", time.perf_counter() - stage_start)
    else:
        record_runtime_metric(resources, "query_view_encode_sec", 0.0)
    case_candidates = query_state.get("case_profile_candidates")
    if case_candidates is None:
        stage_start = time.perf_counter()
        case_candidates = retrieve_case_profile_candidates(resources, query_id, query, encoded_views=encoded_views)
        query_state["case_profile_candidates"] = case_candidates
        record_runtime_metric(resources, "case_candidate_retrieval_sec", time.perf_counter() - stage_start)
    else:
        record_runtime_metric(resources, "case_candidate_retrieval_sec", 0.0)
    scored_profiles = query_state.get("reranked_case_profiles")
    if scored_profiles is None:
        stage_start = time.perf_counter()
        scored_profiles = rerank_case_profiles(resources, encoded_views, case_candidates)
        query_state["reranked_case_profiles"] = scored_profiles
        record_runtime_metric(resources, "case_rerank_sec", time.perf_counter() - stage_start)
    else:
        record_runtime_metric(resources, "case_rerank_sec", 0.0)
    case_feature_df = query_state.get("case_feature_df")
    if case_feature_df is None:
        case_feature_df = build_case_candidate_frame(resources, query, scored_profiles)
        query_state["case_feature_df"] = case_feature_df
    case_stats = dict(case_feature_df.attrs.get("candidate_stats", {}))
    record_runtime_metric(resources, "case_candidate_pre_prune", float(case_stats.get("pre_prune_count", len(case_feature_df))))
    record_runtime_metric(resources, "case_candidate_post_prune", float(case_stats.get("post_prune_count", len(case_feature_df))))
    record_runtime_metric(resources, "case_direct_exact_candidates", float(case_stats.get("direct_exact_count", 0.0)))
    record_runtime_metric(resources, "case_indirect_candidates", float(case_stats.get("indirect_case_count", 0.0)))
    record_runtime_metric(resources, "case_candidate_count_full", float(len(case_feature_df)))
    record_runtime_metric(resources, "build_total_sec", time.perf_counter() - total_start)
    return case_feature_df.copy(), case_stats, list(scored_profiles)


def build_oof_case_frames(
    resources: V44Resources,
    scored_oof_statutes: pd.DataFrame,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    reset_runtime_diagnostics(resources)
    frames: List[Dict[str, object]] = []
    for _, row in maybe_tqdm(resources.train_df.iterrows(), total=len(resources.train_df), desc="Build OOF case frames"):
        query_id = str(row["query_id"])
        golds = parse_citation_set(row["gold_citations"])
        statute_df = scored_oof_statutes[scored_oof_statutes["query_id"] == query_id].sort_values("prob", ascending=False).reset_index(drop=True)
        case_feature_df, case_stats, _ = build_case_feature_frame_for_query(resources, query_id, str(row["query"]))
        frames.append(
            {
                "query_id": query_id,
                "query_text": str(row["query"]),
                "golds": golds,
                "gold_cases": [citation for citation in golds if get_citation_type(citation) in {"bge", "docket"}],
                "statute_df": statute_df,
                "case_feature_df": case_feature_df,
                "case_candidate_stats": case_stats,
            }
        )
    return frames, summarize_runtime_diagnostics(resources)


def build_val_scored_frames(resources: V44Resources, bundle: Dict[str, object], config: V44Config) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    reset_runtime_diagnostics(resources)
    frames: List[Dict[str, object]] = []
    for _, row in maybe_tqdm(resources.val_df.iterrows(), total=len(resources.val_df), desc="Build val frames"):
        golds = parse_citation_set(row["gold_citations"])
        statute_df, case_feature_df, _ = build_statute_candidates(
            resources,
            str(row["query_id"]),
            str(row["query"]),
            set(),
            mode="full_case",
            case_evidence_params=None,
        )
        statute_df = score_statute_frame(statute_df, bundle["statute_model"], bundle["statute_calibrator"], bundle["feature_cols"])
        frames.append(
            {
                "query_id": str(row["query_id"]),
                "query_text": str(row["query"]),
                "golds": golds,
                "gold_cases": [citation for citation in golds if get_citation_type(citation) in {"bge", "docket"}],
                "statute_df": statute_df,
                "case_feature_df": case_feature_df,
                "case_candidate_stats": dict(case_feature_df.attrs.get("candidate_stats", {})),
            }
        )
    return frames, summarize_runtime_diagnostics(resources)


def compute_frame_diagnostics(
    val_frames: Sequence[Dict[str, object]],
    bundle: Dict[str, object],
    config: V44Config,
) -> Dict[str, float]:
    statute_active = []
    case_active = []
    pre_prune = []
    post_prune = []
    direct_counts = []
    indirect_counts = []
    top10_case_mass = []
    emit_statutes = []
    emit_cases = []
    selector_defaults = default_selector_params(config)
    evidence_defaults = default_case_evidence_params(config)

    for frame in val_frames:
        statute_df = frame["statute_df"]
        case_df = apply_case_evidence_model(frame["case_feature_df"], evidence_defaults)
        stats = frame.get("case_candidate_stats", {})
        statute_active.append(1.0 if not statute_df.empty and float(statute_df["reranker_score"].max()) > 0.0 else 0.0)
        case_active.append(1.0 if not case_df.empty and float(case_df["max_profile_reranker"].max()) > 0.0 else 0.0)
        pre_prune.append(float(stats.get("pre_prune_count", len(case_df))))
        post_prune.append(float(stats.get("post_prune_count", len(case_df))))
        direct_counts.append(float(stats.get("direct_exact_count", 0.0)))
        indirect_counts.append(float(stats.get("indirect_case_count", 0.0)))
        top10_case_mass.append(float(case_df.head(10)["case_prob"].sum()) if not case_df.empty else 0.0)

        feature_frame = pd.DataFrame([build_statute_count_features(statute_df, frame["query_text"], config)])
        feature_cols = bundle["statute_count_feature_cols"]
        count_pred = float(bundle["statute_count_model"].predict(feature_frame[feature_cols])[0]) if feature_cols else 0.0
        statute_count = int(np.clip(np.rint(count_pred * float(selector_defaults["statute_multiplier"])), config.statute_count_clip[0], config.statute_count_clip[1]))
        mode = classify_query_mode(statute_df, case_df, selector_defaults)
        case_count = compute_case_budget(case_df, statute_df, selector_defaults, config, mode)
        emit_statutes.append(float(statute_count))
        emit_cases.append(float(case_count))

    return {
        "statute_reranker_active_fraction": float(np.mean(statute_active)) if statute_active else 0.0,
        "case_reranker_active_fraction": float(np.mean(case_active)) if case_active else 0.0,
        "mean_case_candidates_before_prune": float(np.mean(pre_prune)) if pre_prune else 0.0,
        "mean_case_candidates_after_prune": float(np.mean(post_prune)) if post_prune else 0.0,
        "mean_direct_exact_case_candidates": float(np.mean(direct_counts)) if direct_counts else 0.0,
        "mean_indirect_case_candidates": float(np.mean(indirect_counts)) if indirect_counts else 0.0,
        "mean_top10_case_mass": float(np.mean(top10_case_mass)) if top10_case_mass else 0.0,
        "mean_emit_statutes_default": float(np.mean(emit_statutes)) if emit_statutes else 0.0,
        "mean_emit_cases_default": float(np.mean(emit_cases)) if emit_cases else 0.0,
    }


def evaluate_case_evidence_on_frames(
    frames: Sequence[Dict[str, object]],
    config: V44Config,
    params: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    selector_defaults = default_selector_params(config)
    selector_defaults["case_rerank_threshold"] = float(params["case_rerank_threshold"])
    preds = []
    golds = []
    pred_counts = []
    for frame in frames:
        case_df = apply_case_evidence_model(frame["case_feature_df"], params)
        mode = classify_query_mode(frame["statute_df"], case_df, selector_defaults)
        case_budget = compute_case_budget(case_df, frame["statute_df"], selector_defaults, config, mode)
        pred_counts.append(case_budget)
        case_preds = select_case_predictions(case_df, params, case_budget)
        preds.append(case_preds)
        golds.append(frame["gold_cases"])
    return macro_f1_from_sets(preds, golds), {
        "mean_pred_case_count": float(np.mean(pred_counts)) if pred_counts else 0.0,
    }


def tune_case_evidence_on_oof(frames: Sequence[Dict[str, object]], config: V44Config) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    best_params = default_case_evidence_params(config)
    best_score, best_diag = evaluate_case_evidence_on_frames(frames, config, best_params)
    param_grids = {
        "rrf_weight": config.case_evidence_rrf_weight_grid,
        "reranker_weight": config.case_evidence_reranker_weight_grid,
        "late_weight": config.case_evidence_late_weight_grid,
        "support_weight": config.case_evidence_support_weight_grid,
        "direct_weight": config.case_evidence_direct_weight_grid,
        "regex_bonus": config.case_evidence_regex_bonus_grid,
        "exact_case_bonus": config.case_evidence_exact_bonus_grid,
        "base_case_penalty": config.case_evidence_base_penalty_grid,
        "temperature": config.case_evidence_temperature_grid,
        "case_rerank_threshold": config.case_rerank_threshold_grid,
    }
    for _ in range(2):
        improved = False
        for name, grid in param_grids.items():
            local_best_params = dict(best_params)
            local_best_score = best_score
            local_best_diag = dict(best_diag)
            for value in grid:
                candidate = dict(best_params)
                candidate[name] = float(value)
                score, diag = evaluate_case_evidence_on_frames(frames, config, candidate)
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


def evaluate_selector_on_frames(
    frames: Sequence[Dict[str, object]],
    bundle: Dict[str, object],
    config: V44Config,
    selector_params: Dict[str, object],
    case_evidence_params: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    preds = []
    golds = []
    stat_counts = []
    case_counts = []
    for frame in frames:
        statute_df = frame["statute_df"]
        case_df = apply_case_evidence_model(frame["case_feature_df"], case_evidence_params)
        feature_frame = pd.DataFrame([build_statute_count_features(statute_df, frame["query_text"], config)])
        feature_cols = bundle["statute_count_feature_cols"]
        count_pred = float(bundle["statute_count_model"].predict(feature_frame[feature_cols])[0]) if feature_cols else 0.0
        statute_count = int(np.clip(np.rint(count_pred * float(selector_params["statute_multiplier"])), config.statute_count_clip[0], config.statute_count_clip[1]))
        mode = classify_query_mode(statute_df, case_df, selector_params)
        case_count = compute_case_budget(case_df, statute_df, selector_params, config, mode)
        stat_counts.append(statute_count)
        case_counts.append(case_count)
        statute_preds = select_statute_predictions(
            statute_df, float(selector_params["statute_floor"]), statute_count
        )
        case_preds = select_case_predictions(case_df, selector_params, case_count)
        preds.append(merge_predictions(statute_preds, case_preds))
        golds.append(frame["golds"])
    return macro_f1_from_sets(preds, golds), {
        "mean_pred_statute_count": float(np.mean(stat_counts)) if stat_counts else 0.0,
        "mean_pred_case_count": float(np.mean(case_counts)) if case_counts else 0.0,
    }


def tune_selector_on_oof(
    frames: Sequence[Dict[str, object]],
    bundle: Dict[str, object],
    config: V44Config,
    case_evidence_params: Dict[str, float],
) -> Tuple[Dict[str, object], float, Dict[str, float]]:
    best_params = default_selector_params(config)
    best_score, best_diag = evaluate_selector_on_frames(frames, bundle, config, best_params, case_evidence_params)
    param_grids = {
        "statute_multiplier": config.statute_multiplier_grid,
        "statute_floor": config.statute_floor_grid,
        "case_rerank_threshold": config.case_rerank_threshold_grid,
        "case_floor": config.case_floor_grid,
        "case_regex_bonus": config.case_regex_bonus_grid,
    }
    for _ in range(2):
        improved = False
        for name, grid in param_grids.items():
            local_best_params = dict(best_params)
            local_best_score = best_score
            local_best_diag = dict(best_diag)
            for value in grid:
                candidate = dict(best_params)
                candidate[name] = float(value)
                score, diag = evaluate_selector_on_frames(frames, bundle, config, candidate, case_evidence_params)
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


def save_model_bundle(bundle: Dict[str, object]) -> None:
    V44_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    save_pickle(V44_MODEL_BUNDLE_PATH, bundle)
    write_kaggle_manifest(include_bundle=True, include_hashes=False)


def load_model_bundle() -> Dict[str, object]:
    bundle = load_pickle(V44_MODEL_BUNDLE_PATH)
    bundle_version = bundle.get("bundle_version")
    allow_legacy = os.getenv("LIRA_V44_ALLOW_LEGACY_BUNDLE", "0").strip().lower() in {"1", "true", "yes", "on"}
    if bundle_version != V44_PIPELINE_VERSION:
        required_keys = {
            "feature_cols",
            "statute_model",
            "statute_calibrator",
            "statute_count_model",
            "statute_count_feature_cols",
            "case_evidence_params",
            "selector_params",
            "statute_clip",
            "case_clip",
        }
        if not allow_legacy or not required_keys.issubset(bundle.keys()):
            raise RuntimeError(f"Model bundle version mismatch: {bundle_version} != {V44_PIPELINE_VERSION}")
    schema = bundle.get("statute_feature_schema") or {}
    feature_cols = schema.get("feature_cols") or bundle.get("feature_cols") or []
    validate_statute_feature_cols(feature_cols)
    return bundle


def predict_split(resources: V44Resources, bundle: Dict[str, object], split: str) -> pd.DataFrame:
    query_df = resources.val_df if split == "val" else resources.test_df
    rows = []
    for _, row in maybe_tqdm(query_df.iterrows(), total=len(query_df), desc=f"Predict {split}"):
        statute_df, case_df, _ = build_statute_candidates(
            resources,
            str(row["query_id"]),
            str(row["query"]),
            set(),
            mode="full_case",
            case_evidence_params=bundle["case_evidence_params"],
        )
        statute_df = score_statute_frame(statute_df, bundle["statute_model"], bundle["statute_calibrator"], bundle["feature_cols"])
        preds = select_predictions(statute_df, case_df, str(row["query"]), bundle)
        rows.append({"query_id": row["query_id"], "predicted_citations": ";".join(preds)})
    return pd.DataFrame(rows)


