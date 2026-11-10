"""LIRA statutes stage."""

from .core import (
    Dict,
    List,
    LogisticRegression,
    Optional,
    QUERY_VIEW_CHANNELS,
    QUERY_VIEW_NAMES,
    STATUTE_NON_FEATURE_COLUMNS,
    ScoredCaseProfile,
    Sequence,
    Tuple,
    V44Config,
    V44Resources,
    apply_calibrator,
    build_late_interaction_proxy,
    compute_neighbor_support,
    compute_official_neighbor_support,
    default_case_evidence_params,
    defaultdict,
    encode_query_views_for_query,
    extract_citations_regex,
    get_citation_code,
    get_citation_idf,
    get_citation_type,
    get_statute_view_weight,
    get_valid_citation,
    lgb,
    make_query_cache_key,
    normalize_scores,
    np,
    official_statute_alias_hit,
    parent_statute_citation,
    parse_citation_set,
    pd,
    predict_with_ranker,
    re,
    record_runtime_metric,
    require_live_reranker,
    rerank_pairs,
    rrf_merge,
    threshold_label,
    time,
    top_k_indices,
    v43_detect_domain_codes,
)
from .cases import (
    apply_case_evidence_model,
    build_case_candidate_frame,
    build_case_graph_support_from_profiles,
    empty_case_frame,
    rerank_case_profiles,
    retrieve_case_profile_candidates,
)

def initialize_statute_row(query_id: str, citation: str, gold_set: set, detected_codes: set, resources: V44Resources) -> Dict[str, object]:
    row = {
        "query_id": query_id,
        "citation": citation,
        "label": int(citation in gold_set),
        "heuristic_score": 0.0,
        "regex_hit": 0,
        "domain_match": int((resources.laws_code.get(citation) or get_citation_code(citation) or "") in detected_codes),
        "global_df": int(resources.citation_doc_freq.get(citation, 0)),
        "idf": float(get_citation_idf(citation, resources.citation_doc_freq, resources.total_cases)),
        "graph_case_support": 0.0,
        "graph_hit": 0,
        "parent_alias_hit": 0,
        "parent_child_count": 0,
        "neighbor_support": 0.0,
        "official_alias_hit": 0,
        "official_neighbor_support": 0.0,
        "rrf_score": 0.0,
        "colbert_score": 0.0,
        "reranker_score": 0.0,
        "best_dense_score": 0.0,
        "best_sparse_score": 0.0,
        "best_bm25_score": 0.0,
        "channel_coverage": 0.0,
        "view_coverage": 0.0,
        "reciprocal_rank_sum": 0.0,
        "best_rank": 10**6,
        "pre_rerank_fusion": 0.0,
    }
    for view_name in QUERY_VIEW_NAMES:
        for channel in QUERY_VIEW_CHANNELS:
            row[f"{view_name}_{channel}_hit"] = 0
            row[f"{view_name}_{channel}_rank"] = 10**6
            row[f"{view_name}_{channel}_score"] = 0.0
    return row


def get_query_runtime_state(resources: V44Resources, query_id: str, query: str) -> Dict[str, object]:
    cache_key = make_query_cache_key(query_id, query)
    return resources.runtime_cache.setdefault(cache_key, {})


def build_statute_candidates(
    resources: V44Resources,
    query_id: str,
    query: str,
    gold_set: Optional[set] = None,
    mode: str = "full_case",
    encoded_views: Optional[Dict[str, Dict[str, object]]] = None,
    case_evidence: Optional[Dict[str, object]] = None,
    case_evidence_params: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[ScoredCaseProfile]]:
    total_start = time.perf_counter()
    gold_set = gold_set or set()
    detected_codes = v43_detect_domain_codes(query)
    query_state = get_query_runtime_state(resources, query_id, query)

    if encoded_views is None:
        encoded_views = query_state.get("encoded_views")
        if encoded_views is None:
            stage_start = time.perf_counter()
            encoded_views = encode_query_views_for_query(resources, query_id, query)
            query_state["encoded_views"] = encoded_views
            record_runtime_metric(resources, "query_view_encode_sec", time.perf_counter() - stage_start)
        else:
            record_runtime_metric(resources, "query_view_encode_sec", 0.0)
    else:
        record_runtime_metric(resources, "query_view_encode_sec", 0.0)

    case_candidates = case_evidence.get("profile_candidates") if case_evidence else None
    if case_candidates is None:
        case_candidates = query_state.get("case_profile_candidates")
    if case_candidates is None:
        stage_start = time.perf_counter()
        case_candidates = retrieve_case_profile_candidates(resources, query_id, query, encoded_views=encoded_views)
        query_state["case_profile_candidates"] = case_candidates
        record_runtime_metric(resources, "case_candidate_retrieval_sec", time.perf_counter() - stage_start)
    else:
        record_runtime_metric(resources, "case_candidate_retrieval_sec", 0.0)

    if mode not in {"support_only", "full_case"}:
        raise RuntimeError(f"Unknown statute-build mode: {mode}")

    if mode == "support_only":
        scored_profiles = list(case_candidates)
        case_df = empty_case_frame(scored=False)
        record_runtime_metric(resources, "case_rerank_sec", 0.0)
    else:
        scored_profiles = case_evidence.get("reranked_profiles") if case_evidence else None
        if scored_profiles is None:
            scored_profiles = query_state.get("reranked_case_profiles")
        if scored_profiles is None:
            stage_start = time.perf_counter()
            scored_profiles = rerank_case_profiles(resources, encoded_views, case_candidates)
            query_state["reranked_case_profiles"] = scored_profiles
            record_runtime_metric(resources, "case_rerank_sec", time.perf_counter() - stage_start)
        else:
            record_runtime_metric(resources, "case_rerank_sec", 0.0)
        case_feature_df = case_evidence.get("case_feature_df") if case_evidence else None
        if case_feature_df is None:
            case_feature_df = query_state.get("case_feature_df")
        if case_feature_df is None:
            case_feature_df = build_case_candidate_frame(resources, query, scored_profiles)
            query_state["case_feature_df"] = case_feature_df
        case_stats = case_feature_df.attrs.get("candidate_stats", {})
        record_runtime_metric(resources, "case_candidate_pre_prune", float(case_stats.get("pre_prune_count", len(case_feature_df))))
        record_runtime_metric(resources, "case_candidate_post_prune", float(case_stats.get("post_prune_count", len(case_feature_df))))
        record_runtime_metric(resources, "case_direct_exact_candidates", float(case_stats.get("direct_exact_count", 0.0)))
        record_runtime_metric(resources, "case_indirect_candidates", float(case_stats.get("indirect_case_count", 0.0)))
        case_df = apply_case_evidence_model(case_feature_df, case_evidence_params or default_case_evidence_params(resources.config)) if case_evidence_params is not None else case_feature_df.copy()

    case_graph_support = build_case_graph_support_from_profiles(resources, scored_profiles)

    rows: Dict[str, Dict[str, object]] = {}
    channel_hits: Dict[str, set[str]] = defaultdict(set)
    view_hits: Dict[str, set[str]] = defaultdict(set)
    rank_maps: List[Dict[str, int]] = []
    forced_regex_statutes: List[str] = []

    for citation in extract_citations_regex(query):
        valid = get_valid_citation(citation, resources.citation_universe)
        if valid and get_citation_type(valid.citation) == "statute":
            row = rows.setdefault(valid.citation, initialize_statute_row(query_id, valid.citation, gold_set, detected_codes, resources))
            row["regex_hit"] = 1
            if valid.citation not in forced_regex_statutes:
                forced_regex_statutes.append(valid.citation)

    for view_name, payload in encoded_views.items():
        q_dense = payload["dense"]
        q_sparse = payload["sparse"]
        view_weight = get_statute_view_weight(view_name)
        dense_scores = q_dense @ resources.laws_embeddings.T
        top_dense_idx = top_k_indices(dense_scores, resources.config.laws_dense_k)
        dense_rank: Dict[str, int] = {}
        for rank, idx in enumerate(top_dense_idx):
            sim = float(dense_scores[idx])
            if sim < resources.config.dense_min_sim:
                break
            citation = resources.laws_citations[idx]
            valid = get_valid_citation(citation, resources.citation_universe)
            if not valid or get_citation_type(valid.citation) != "statute":
                continue
            row = rows.setdefault(valid.citation, initialize_statute_row(query_id, valid.citation, gold_set, detected_codes, resources))
            row[f"{view_name}_dense_hit"] = 1
            row[f"{view_name}_dense_rank"] = min(int(row[f"{view_name}_dense_rank"]), rank + 1)
            row[f"{view_name}_dense_score"] = max(float(row[f"{view_name}_dense_score"]), sim)
            row["best_dense_score"] = max(float(row["best_dense_score"]), sim)
            row["reciprocal_rank_sum"] += view_weight / float(rank + 1)
            row["best_rank"] = min(int(row["best_rank"]), rank + 1)
            channel_hits[valid.citation].add("dense")
            view_hits[valid.citation].add(view_name)
            dense_rank[valid.citation] = rank
        rank_maps.append(dense_rank)

        sparse_results = resources.laws_sparse_index.search(q_sparse, resources.config.laws_sparse_k)
        sparse_rank: Dict[str, int] = {}
        top_sparse = max((score for _, score in sparse_results), default=1.0)
        for rank, (doc_idx, score) in enumerate(sparse_results):
            citation = resources.laws_sparse_index.doc_ids[doc_idx]
            valid = get_valid_citation(citation, resources.citation_universe)
            if not valid or get_citation_type(valid.citation) != "statute":
                continue
            norm_score = float(score) / float(top_sparse or 1.0)
            row = rows.setdefault(valid.citation, initialize_statute_row(query_id, valid.citation, gold_set, detected_codes, resources))
            row[f"{view_name}_sparse_hit"] = 1
            row[f"{view_name}_sparse_rank"] = min(int(row[f"{view_name}_sparse_rank"]), rank + 1)
            row[f"{view_name}_sparse_score"] = max(float(row[f"{view_name}_sparse_score"]), norm_score)
            row["best_sparse_score"] = max(float(row["best_sparse_score"]), norm_score)
            row["reciprocal_rank_sum"] += (1.15 * view_weight) / float(rank + 1)
            row["best_rank"] = min(int(row["best_rank"]), rank + 1)
            channel_hits[valid.citation].add("sparse")
            view_hits[valid.citation].add(view_name)
            sparse_rank[valid.citation] = rank
        rank_maps.append(sparse_rank)

        bm25_results = resources.laws_bm25.search(payload["text"], top_k=resources.config.laws_bm25_k)
        bm25_rank: Dict[str, int] = {}
        top_bm25 = max((score for _, score in bm25_results), default=1.0)
        for rank, (citation, score) in enumerate(bm25_results):
            valid = get_valid_citation(citation, resources.citation_universe)
            if not valid or get_citation_type(valid.citation) != "statute":
                continue
            norm_score = float(score) / float(top_bm25 or 1.0)
            row = rows.setdefault(valid.citation, initialize_statute_row(query_id, valid.citation, gold_set, detected_codes, resources))
            row[f"{view_name}_bm25_hit"] = 1
            row[f"{view_name}_bm25_rank"] = min(int(row[f"{view_name}_bm25_rank"]), rank + 1)
            row[f"{view_name}_bm25_score"] = max(float(row[f"{view_name}_bm25_score"]), norm_score)
            row["best_bm25_score"] = max(float(row["best_bm25_score"]), norm_score)
            row["reciprocal_rank_sum"] += (1.15 * view_weight) / float(rank + 1)
            row["best_rank"] = min(int(row["best_rank"]), rank + 1)
            channel_hits[valid.citation].add("bm25")
            view_hits[valid.citation].add(view_name)
            bm25_rank[valid.citation] = rank
        rank_maps.append(bm25_rank)

    forced_graph_statutes: List[str] = []
    for citation, support in case_graph_support.most_common(resources.config.laws_graph_k):
        if support <= 0:
            continue
        valid = get_valid_citation(citation, resources.citation_universe)
        if not valid or get_citation_type(valid.citation) != "statute":
            continue
        row = rows.setdefault(
            valid.citation,
            initialize_statute_row(query_id, valid.citation, gold_set, detected_codes, resources),
        )
        row["graph_hit"] = 1
        forced_graph_statutes.append(valid.citation)

    # Corpus rows are often paragraph-level while gold labels name the article.
    # Let an article-level candidate inherit the evidence of its retrieved paragraphs.
    for child_citation, child_row in list(rows.items()):
        parent = parent_statute_citation(child_citation)
        if parent is None or parent not in resources.citation_universe.law_exact_universe:
            continue
        parent_row = rows.setdefault(
            parent,
            initialize_statute_row(query_id, parent, gold_set, detected_codes, resources),
        )
        parent_row["parent_alias_hit"] = 1
        parent_row["parent_child_count"] = min(5, int(parent_row["parent_child_count"]) + 1)
        for feature_name in ("regex_hit", "graph_hit", "best_dense_score", "best_sparse_score", "best_bm25_score"):
            parent_row[feature_name] = max(parent_row[feature_name], child_row[feature_name])
        parent_row["reciprocal_rank_sum"] = max(
            float(parent_row["reciprocal_rank_sum"]), float(child_row["reciprocal_rank_sum"])
        )
        parent_row["best_rank"] = min(int(parent_row["best_rank"]), int(child_row["best_rank"]))
        parent_row["graph_case_support"] = max(
            float(parent_row["graph_case_support"]), float(case_graph_support.get(child_citation, 0.0))
        )
        channel_hits[parent].update(channel_hits.get(child_citation, set()))
        view_hits[parent].update(view_hits.get(child_citation, set()))
        for view_name in QUERY_VIEW_NAMES:
            for channel in QUERY_VIEW_CHANNELS:
                prefix = f"{view_name}_{channel}"
                parent_row[f"{prefix}_hit"] = max(parent_row[f"{prefix}_hit"], child_row[f"{prefix}_hit"])
                parent_row[f"{prefix}_rank"] = min(parent_row[f"{prefix}_rank"], child_row[f"{prefix}_rank"])
                parent_row[f"{prefix}_score"] = max(parent_row[f"{prefix}_score"], child_row[f"{prefix}_score"])

    for rank_map in rank_maps:
        for citation, rank in list(rank_map.items()):
            parent = parent_statute_citation(citation)
            if parent in rows:
                rank_map[parent] = min(rank_map.get(parent, 10**6), rank)

    fused_rrf = rrf_merge(rank_maps, resources.config.rrf_k)
    for citation, row in rows.items():
        row["rrf_score"] = float(fused_rrf.get(citation, 0.0))
        row["graph_case_support"] = max(
            float(row["graph_case_support"]), float(case_graph_support.get(citation, 0.0))
        )
        row["official_alias_hit"] = official_statute_alias_hit(query, citation, resources.official_enrichment)
        row["channel_coverage"] = float(len(channel_hits.get(citation, set())))
        row["view_coverage"] = float(len(view_hits.get(citation, set())))
        row["pre_rerank_fusion"] = (
            0.90 * float(row["best_dense_score"])
            + 1.35 * float(row["best_sparse_score"])
            + 1.35 * float(row["best_bm25_score"])
            + 0.55 * float(row["rrf_score"])
            + 1.20 * float(row["regex_hit"])
            + 0.60 * float(row["official_alias_hit"])
            + 0.45 * float(row["domain_match"])
            + 0.30 * float(row["graph_case_support"])
            + 0.30 * float(row["channel_coverage"])
            + 0.25 * float(row["view_coverage"])
            + 0.55 * float(row["reciprocal_rank_sum"])
        )

    ordered_union = [
        citation
        for citation, row in sorted(
            rows.items(),
            key=lambda kv: (
                float(kv[1]["pre_rerank_fusion"]),
                float(kv[1]["reciprocal_rank_sum"]),
                -float(kv[1]["best_rank"]),
                str(kv[0]),
            ),
            reverse=True,
        )
    ]
    union_pool: List[str] = []
    seen_union = set()
    for citation in forced_regex_statutes + forced_graph_statutes + ordered_union:
        if citation in seen_union:
            continue
        seen_union.add(citation)
        union_pool.append(citation)
        if len(union_pool) >= resources.config.laws_union_pool_cap:
            break

    neighbor_support = compute_neighbor_support(union_pool)
    official_neighbor_support = compute_official_neighbor_support(union_pool, resources.official_enrichment)
    for citation in union_pool:
        row = rows[citation]
        row["neighbor_support"] = float(neighbor_support.get(citation, 0.0))
        row["official_neighbor_support"] = float(official_neighbor_support.get(citation, 0.0))
        row["pre_rerank_fusion"] = (
            0.90 * float(row["best_dense_score"])
            + 1.35 * float(row["best_sparse_score"])
            + 1.35 * float(row["best_bm25_score"])
            + 0.55 * float(row["rrf_score"])
            + 1.20 * float(row["regex_hit"])
            + 0.60 * float(row["official_alias_hit"])
            + 0.45 * float(row["domain_match"])
            + 0.30 * float(row["official_neighbor_support"])
            + 0.30 * float(row["graph_case_support"])
            + 0.30 * float(row["channel_coverage"])
            + 0.25 * float(row["view_coverage"])
            + 0.55 * float(row["reciprocal_rank_sum"])
        )
        row["heuristic_score"] = float(row["pre_rerank_fusion"])
    rerank_shortlist = [
        citation
        for citation, row in sorted(
            ((citation, rows[citation]) for citation in union_pool),
            key=lambda kv: (
                float(kv[1]["pre_rerank_fusion"]),
                float(kv[1]["reciprocal_rank_sum"]),
                -float(kv[1]["best_rank"]),
                str(kv[0]),
            ),
            reverse=True,
        )[: resources.config.laws_rerank_k]
    ]
    rerank_queries: List[str] = []
    rerank_docs: List[str] = []
    rerank_index: List[str] = []
    rerank_scores: Dict[str, float] = defaultdict(float)
    colbert_scores: Dict[str, float] = {}
    for citation in rerank_shortlist:
        row = rows.setdefault(citation, initialize_statute_row(query_id, citation, gold_set, detected_codes, resources))
        dense_best = max(float(row[f"{view_name}_dense_score"]) for view_name in QUERY_VIEW_NAMES)
        sparse_best = max(float(row[f"{view_name}_sparse_score"]) for view_name in QUERY_VIEW_NAMES)
        colbert_scores[citation] = build_late_interaction_proxy(dense_best, sparse_best)
        law_text = f"{resources.laws_title.get(citation, '')}\n{resources.laws_text.get(citation, '')}"
        for view_name in QUERY_VIEW_NAMES:
            rerank_queries.append(encoded_views[view_name]["text"])
            rerank_docs.append(law_text)
            rerank_index.append(citation)
    statute_rerank_start = time.perf_counter()
    if rerank_shortlist:
        require_live_reranker(resources, "statute reranking")
    rerank_values = rerank_pairs(
        resources.reranker_tokenizer,
        resources.reranker_model,
        rerank_queries,
        rerank_docs,
        batch_size=resources.reranker_batch_size,
    )
    for citation, score in zip(rerank_index, rerank_values):
        rerank_scores[citation] = max(rerank_scores.get(citation, float("-inf")), float(score))
    rerank_scores = normalize_scores(rerank_scores)
    if rerank_shortlist and resources.require_reranker and (not rerank_scores or max(rerank_scores.values(), default=0.0) <= 0.0):
        raise RuntimeError("Statute reranker produced no usable scores in quality mode")
    record_runtime_metric(resources, "statute_rerank_sec", time.perf_counter() - statute_rerank_start)

    for citation in union_pool:
        row = rows[citation]
        row["colbert_score"] = float(colbert_scores.get(citation, 0.0))
        row["reranker_score"] = float(rerank_scores.get(citation, 0.0))
        row["heuristic_score"] = float(row["pre_rerank_fusion"]) + (
            0.55 * row["rrf_score"]
            + 0.8 * row["colbert_score"]
            + 1.0 * row["reranker_score"]
        )

    candidate_df = (
        pd.DataFrame([rows[citation] for citation in union_pool])
        if union_pool
        else pd.DataFrame(columns=["query_id", "citation", "label", "heuristic_score"])
    )
    if not candidate_df.empty:
        candidate_df = candidate_df.sort_values(["heuristic_score", "rrf_score", "reranker_score"], ascending=[False, False, False]).reset_index(drop=True)
        candidate_df.attrs["statute_candidate_stats"] = {
            "union_pool_size": int(len(union_pool)),
            "rerank_shortlist_size": int(len(rerank_shortlist)),
            "forced_regex_count": int(len(forced_regex_statutes)),
        }
    record_runtime_metric(resources, "statute_candidate_count", float(len(candidate_df)))
    if mode == "full_case":
        record_runtime_metric(resources, "case_candidate_count_full", float(len(case_df)))
    record_runtime_metric(resources, "build_total_sec", time.perf_counter() - total_start)
    return candidate_df, case_df, scored_profiles

def get_statute_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col not in STATUTE_NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])]


def validate_statute_feature_cols(feature_cols: Sequence[str]) -> None:
    forbidden = sorted(set(feature_cols) & STATUTE_NON_FEATURE_COLUMNS)
    if forbidden:
        raise RuntimeError(
            f"Stale statute feature schema includes metadata columns {forbidden}. "
            "Rerun `python lira_pipeline.py cv` to rebuild the V44 model bundle."
        )


def sanitize_statute_feature_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    validate_statute_feature_cols(feature_cols)
    aligned = df.copy()
    for column in feature_cols:
        if column not in aligned.columns:
            aligned[column] = 0.0
    aligned.loc[:, list(feature_cols)] = (
        aligned.loc[:, list(feature_cols)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
    )
    return aligned


def score_statute_frame(candidate_df: pd.DataFrame, model: Optional[lgb.LGBMRanker], calibrator: Optional[LogisticRegression], feature_cols: List[str]) -> pd.DataFrame:
    candidate_df = sanitize_statute_feature_frame(candidate_df, feature_cols)
    if candidate_df.empty:
        candidate_df["raw_score"] = np.zeros(0, dtype=np.float32)
        candidate_df["prob"] = np.zeros(0, dtype=np.float32)
        return candidate_df
    candidate_df["raw_score"] = predict_with_ranker(candidate_df, model, feature_cols)
    candidate_df["prob"] = apply_calibrator(candidate_df.assign(raw_score=candidate_df["raw_score"]), calibrator)
    return candidate_df


def build_statute_count_features(group_df: pd.DataFrame, query_text: str, config: V44Config) -> Dict[str, float]:
    group_df = group_df.sort_values("prob", ascending=False)
    probs = group_df["prob"].to_numpy(dtype=np.float32) if not group_df.empty else np.zeros(0, dtype=np.float32)
    feats = {
        "candidate_count": float(len(group_df)),
        "prob_at_1": float(probs[0]) if len(probs) >= 1 else 0.0,
        "prob_at_3": float(probs[2]) if len(probs) >= 3 else 0.0,
        "prob_at_5": float(probs[4]) if len(probs) >= 5 else 0.0,
        "prob_sum_top5": float(probs[:5].sum()) if len(probs) else 0.0,
        "prob_sum_top10": float(probs[:10].sum()) if len(probs) else 0.0,
        "regex_hits_top10": float(group_df.head(10)["regex_hit"].sum()) if "regex_hit" in group_df.columns else 0.0,
        "graph_support_top10": float(group_df.head(10)["graph_case_support"].sum()) if "graph_case_support" in group_df.columns else 0.0,
        "query_token_length": float(len(re.findall(r"\b\w+\b", str(query_text).lower()))),
        "query_domain_code_count": float(len(v43_detect_domain_codes(query_text))),
    }
    for threshold in config.query_prob_thresholds:
        feats[f"count_ge_{threshold_label(threshold)}"] = float((probs >= threshold).sum())
    return feats


def case_numeric_column(case_df: pd.DataFrame, column: str) -> np.ndarray:
    if case_df.empty:
        return np.zeros(0, dtype=np.float32)
    if column not in case_df.columns:
        return np.zeros(len(case_df), dtype=np.float32)
    return pd.to_numeric(case_df[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def train_statute_count_model(scored_oof: pd.DataFrame, train_df: pd.DataFrame, config: V44Config) -> Tuple[lgb.LGBMRegressor, List[str], pd.DataFrame]:
    rows = []
    query_lookup = train_df.set_index("query_id")
    for query_id, group in scored_oof.groupby("query_id", sort=False):
        golds = parse_citation_set(query_lookup.loc[query_id, "gold_citations"])
        statute_gold_count = sum(1 for citation in golds if get_citation_type(citation) == "statute")
        feats = build_statute_count_features(group, str(query_lookup.loc[query_id, "query"]), config)
        rows.append({"query_id": query_id, "gold_statute_count": statute_gold_count, **feats})
    query_df = pd.DataFrame(rows)
    feature_cols = [col for col in query_df.columns if col not in {"query_id", "gold_statute_count"}]
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
    model.fit(query_df[feature_cols], query_df["gold_statute_count"].astype(np.float32))
    query_df["pred_statute_count"] = np.rint(np.clip(model.predict(query_df[feature_cols]), config.statute_count_clip[0], config.statute_count_clip[1])).astype(int)
    return model, feature_cols, query_df


