"""LIRA cases stage."""

from .core import (
    CASE_CHANNELS,
    CitationMatch,
    Counter,
    Dict,
    List,
    Optional,
    QUERY_VIEW_NAMES,
    ScoredCaseProfile,
    Sequence,
    V44Resources,
    build_late_interaction_proxy,
    defaultdict,
    encode_query_views_for_query,
    extract_citations_regex,
    faiss_search,
    get_case_group_id,
    get_citation_idf,
    get_citation_type,
    get_official_case_statute_targets,
    get_top_co_citation_targets,
    get_valid_citation,
    math,
    normalize_case_citation,
    normalize_scores,
    np,
    pd,
    require_live_reranker,
    rerank_pairs,
    rrf_merge,
)

def retrieve_case_profile_candidates(
    resources: V44Resources,
    query_id: str,
    query: str,
    encoded_views: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[ScoredCaseProfile]:
    encoded_views = encoded_views or encode_query_views_for_query(resources, query_id, query)
    run_rank_maps: List[Dict[str, int]] = []
    score_maps: Dict[str, Dict[str, float]] = {channel: defaultdict(float) for channel in CASE_CHANNELS}
    exact_backproj_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

    for _, payload in encoded_views.items():
        q_dense = payload["dense"]
        q_sparse = payload["sparse"]
        dense_results = faiss_search(
            resources.case_profile_faiss,
            resources.case_profile_dense,
            q_dense,
            resources.config.case_profile_faiss_search_k,
            resources.config.case_dense_k,
        )
        dense_rank: Dict[str, int] = {}
        for rank, (idx, score) in enumerate(dense_results):
            if score < resources.config.dense_min_sim:
                break
            case_id = resources.case_profile_ids[idx]
            dense_rank[case_id] = rank
            score_maps["dense"][case_id] = max(score_maps["dense"].get(case_id, 0.0), float(score))
        run_rank_maps.append(dense_rank)

        sparse_results = resources.case_profile_sparse_index.search(q_sparse, resources.config.case_sparse_k)
        sparse_rank: Dict[str, int] = {}
        top_sparse = max((score for _, score in sparse_results), default=1.0)
        for rank, (doc_idx, score) in enumerate(sparse_results):
            case_id = resources.case_profile_sparse_index.doc_ids[doc_idx]
            sparse_rank[case_id] = rank
            score_maps["sparse"][case_id] = max(score_maps["sparse"].get(case_id, 0.0), float(score) / float(top_sparse or 1.0))
        run_rank_maps.append(sparse_rank)

        if resources.config.enable_case_bm25:
            bm25_results = resources.case_profile_bm25.search(payload["text"], top_k=resources.config.case_bm25_k)
            bm25_rank: Dict[str, int] = {}
            top_bm25 = max((score for _, score in bm25_results), default=1.0)
            for rank, (case_id, score) in enumerate(bm25_results):
                case_id = str(case_id)
                bm25_rank[case_id] = rank
                score_maps["bm25"][case_id] = max(score_maps["bm25"].get(case_id, 0.0), float(score) / float(top_bm25 or 1.0))
            run_rank_maps.append(bm25_rank)

        if resources.config.enable_case_backproj:
            backproj_dense = faiss_search(resources.court_faiss, resources.court_embeddings, q_dense, 4096, resources.config.case_backproj_k)
            backproj_rank: Dict[str, int] = {}
            for rank, (idx, score) in enumerate(backproj_dense):
                if score < resources.config.dense_min_sim:
                    break
                exact_citation = normalize_case_citation(resources.court_case_ids[idx])
                case_id = get_case_group_id(exact_citation)
                if case_id not in backproj_rank:
                    backproj_rank[case_id] = rank
                score_maps["backproj"][case_id] = max(score_maps["backproj"].get(case_id, 0.0), float(score))
                if get_citation_type(exact_citation) in {"bge", "docket"}:
                    exact_backproj_scores[case_id][exact_citation] = max(
                        exact_backproj_scores[case_id].get(exact_citation, 0.0), float(score)
                    )
            run_rank_maps.append(backproj_rank)

    fused_rrf = rrf_merge(run_rank_maps, resources.config.rrf_k)
    top_case_ids = [case_id for case_id, _ in sorted(fused_rrf.items(), key=lambda kv: kv[1], reverse=True)[: resources.config.case_profile_shortlist_k]]
    if not top_case_ids:
        return []

    candidate_profiles: List[ScoredCaseProfile] = []
    for case_id in top_case_ids:
        profile = resources.case_profiles[case_id]
        dense_best = float(score_maps["dense"].get(case_id, 0.0))
        sparse_best = float(score_maps["sparse"].get(case_id, 0.0))
        retrieved_exact = [
            citation for citation, _ in sorted(
                exact_backproj_scores.get(case_id, {}).items(),
                key=lambda item: (item[1], item[0]), reverse=True,
            )
        ]
        exact_case_ids = list(dict.fromkeys([*retrieved_exact, *profile.exact_case_ids]))[
            : resources.config.case_profile_exact_limit
        ]
        candidate_profiles.append(
            ScoredCaseProfile(
                base_case_id=case_id,
                rrf_score=float(fused_rrf.get(case_id, 0.0)),
                dense_score=float(score_maps["dense"].get(case_id, 0.0)),
                sparse_score=float(score_maps["sparse"].get(case_id, 0.0)),
                bm25_score=float(score_maps["bm25"].get(case_id, 0.0)),
                backproj_score=float(score_maps["backproj"].get(case_id, 0.0)),
                late_score=build_late_interaction_proxy(dense_best, sparse_best),
                reranker_score=0.0,
                profile_text=profile.profile_text,
                exact_case_ids=exact_case_ids,
                cited_cases=list(profile.cited_cases),
                cited_statutes=list(profile.cited_statutes),
            )
        )
    candidate_profiles.sort(key=lambda item: (item.rrf_score, item.late_score, item.dense_score), reverse=True)
    return candidate_profiles


def rerank_case_profiles(
    resources: V44Resources,
    encoded_views: Dict[str, Dict[str, object]],
    profile_candidates: Sequence[ScoredCaseProfile],
) -> List[ScoredCaseProfile]:
    if not profile_candidates:
        return []
    if not resources.config.enable_case_rerank or resources.config.case_profile_rerank_k <= 0:
        return list(profile_candidates[: max(1, min(len(profile_candidates), resources.config.case_profile_shortlist_k))])
    top_profiles = list(profile_candidates[: resources.config.case_profile_rerank_k])
    if resources.reranker_tokenizer is None or resources.reranker_model is None:
        require_live_reranker(resources, "case-profile reranking")
        return top_profiles
    query_strings: List[str] = []
    doc_strings: List[str] = []
    pair_index: List[str] = []
    for profile in top_profiles:
        for view_name in QUERY_VIEW_NAMES:
            query_strings.append(encoded_views[view_name]["text"])
            doc_strings.append(profile.profile_text)
            pair_index.append(profile.base_case_id)
    rerank_outputs = rerank_pairs(
        resources.reranker_tokenizer,
        resources.reranker_model,
        query_strings,
        doc_strings,
        batch_size=resources.reranker_batch_size,
    )
    rerank_map: Dict[str, float] = defaultdict(float)
    for case_id, score in zip(pair_index, rerank_outputs):
        rerank_map[case_id] = max(rerank_map.get(case_id, float("-inf")), float(score))
    if rerank_map:
        rerank_map = normalize_scores(rerank_map)
    reranked: List[ScoredCaseProfile] = []
    for profile in top_profiles:
        reranked.append(
            ScoredCaseProfile(
                base_case_id=profile.base_case_id,
                rrf_score=profile.rrf_score,
                dense_score=profile.dense_score,
                sparse_score=profile.sparse_score,
                bm25_score=profile.bm25_score,
                backproj_score=profile.backproj_score,
                late_score=profile.late_score,
                reranker_score=float(rerank_map.get(profile.base_case_id, 0.0)),
                profile_text=profile.profile_text,
                exact_case_ids=list(profile.exact_case_ids),
                cited_cases=list(profile.cited_cases),
                cited_statutes=list(profile.cited_statutes),
            )
        )
    reranked.sort(key=lambda item: (item.reranker_score, item.rrf_score, item.late_score), reverse=True)
    return reranked


def empty_case_frame(scored: bool = False) -> pd.DataFrame:
    columns = [
        "citation",
        "citation_type",
        "rrf_score",
        "max_profile_rrf",
        "reranker_score",
        "late_score",
        "support_profiles",
        "direct_profile_hits",
        "source_direct_exact",
        "source_indirect_cited",
        "regex_hit",
        "case_match",
        "best_profile_rank",
        "idf",
        "frequency_weight",
        "is_exact_case",
        "is_base_case",
        "max_profile_reranker",
        "prelim_score",
    ]
    if scored:
        columns.extend(["raw_score", "case_prob"])
    return pd.DataFrame(columns=columns)


def _new_case_candidate_row(valid: CitationMatch) -> Dict[str, object]:
    return {
        "citation": valid.citation,
        "citation_type": get_citation_type(valid.citation),
        "rrf_score": 0.0,
        "max_profile_rrf": 0.0,
        "reranker_score": 0.0,
        "late_score": 0.0,
        "max_profile_reranker": 0.0,
        "support_profiles": 0,
        "direct_profile_hits": 0,
        "source_direct_exact": 0,
        "source_indirect_cited": 0,
        "regex_hit": 0,
        "case_match": valid.case_match,
        "best_profile_rank": 10**6,
        "prelim_score": 0.0,
    }


def build_case_candidate_frame(resources: V44Resources, query: str, scored_profiles: Sequence[ScoredCaseProfile]) -> pd.DataFrame:
    regex_exact = set()
    for citation in extract_citations_regex(query):
        valid = get_valid_citation(citation, resources.citation_universe)
        if valid and get_citation_type(valid.citation) in {"bge", "docket"} and valid.case_match == "exact":
            regex_exact.add(valid.citation)

    rows: Dict[str, Dict[str, object]] = {}
    direct_profiles = list(scored_profiles[: resources.config.case_direct_profile_limit])
    indirect_profiles = list(scored_profiles[: resources.config.case_indirect_profile_limit])

    for profile_rank, profile in enumerate(direct_profiles, start=1):
        for citation in profile.exact_case_ids[: resources.config.case_profile_exact_limit]:
            valid = get_valid_citation(citation, resources.citation_universe)
            if not valid:
                continue
            row = rows.setdefault(valid.citation, _new_case_candidate_row(valid))
            row["rrf_score"] += profile.rrf_score
            row["max_profile_rrf"] = max(float(row["max_profile_rrf"]), profile.rrf_score)
            row["reranker_score"] = max(float(row["reranker_score"]), profile.reranker_score)
            row["late_score"] = max(float(row["late_score"]), profile.late_score)
            row["max_profile_reranker"] = max(float(row["max_profile_reranker"]), profile.reranker_score)
            row["support_profiles"] += 1
            row["direct_profile_hits"] += 1
            row["source_direct_exact"] += 1
            row["best_profile_rank"] = min(int(row["best_profile_rank"]), profile_rank)

    for profile_rank, profile in enumerate(indirect_profiles, start=1):
        for citation, ref_count in profile.cited_cases[: resources.config.case_indirect_citations_per_profile]:
            valid = get_valid_citation(citation, resources.citation_universe)
            if not valid:
                continue
            row = rows.setdefault(valid.citation, _new_case_candidate_row(valid))
            support_scale = min(int(ref_count), 3) / 3.0
            row["rrf_score"] += profile.rrf_score * support_scale
            row["max_profile_rrf"] = max(float(row["max_profile_rrf"]), profile.rrf_score)
            row["reranker_score"] = max(float(row["reranker_score"]), profile.reranker_score)
            row["late_score"] = max(float(row["late_score"]), profile.late_score)
            row["max_profile_reranker"] = max(float(row["max_profile_reranker"]), profile.reranker_score)
            row["support_profiles"] += 1
            row["source_indirect_cited"] += 1
            row["best_profile_rank"] = min(int(row["best_profile_rank"]), profile_rank)

    for citation in list(rows.keys()):
        if citation in regex_exact:
            rows[citation]["regex_hit"] = 1
        freq = max(1, resources.citation_doc_freq.get(citation, 1))
        idf = get_citation_idf(citation, resources.citation_doc_freq, resources.total_cases)
        freq_weight = max(0.5, min(1.5, 1.0 / math.sqrt(freq)))
        rows[citation]["idf"] = float(idf)
        rows[citation]["frequency_weight"] = float(freq_weight)
        rows[citation]["is_exact_case"] = int(rows[citation]["case_match"] == "exact")
        rows[citation]["is_base_case"] = int(rows[citation]["case_match"] == "base")

    filtered_rows: List[Dict[str, object]] = list(rows.values())
    if not filtered_rows:
        return empty_case_frame(scored=False)
    case_df = pd.DataFrame(filtered_rows)
    direct_scaled = np.clip(case_df["direct_profile_hits"].to_numpy(dtype=np.float32), 0, 3) / 3.0
    support_scaled = np.clip(case_df["support_profiles"].to_numpy(dtype=np.float32), 0, 5) / 5.0
    case_df["prelim_score"] = (
        0.45 * case_df["max_profile_reranker"].to_numpy(dtype=np.float32)
        + 0.35 * case_df["max_profile_rrf"].to_numpy(dtype=np.float32)
        + 0.15 * direct_scaled
        + 0.05 * support_scaled
    ).astype(np.float32)
    pre_prune_count = int(len(case_df))
    case_df = case_df.sort_values(
        ["prelim_score", "source_direct_exact", "max_profile_reranker", "best_profile_rank", "support_profiles", "rrf_score"],
        ascending=[False, False, False, True, False, False],
    ).reset_index(drop=True)
    selected_indices: List[int] = []
    selected_set = set()
    by_base: Counter = Counter()
    for index, citation in enumerate(case_df["citation"]):
        base = get_case_group_id(str(citation))
        if by_base[base] >= resources.config.case_diversity_per_base:
            continue
        selected_indices.append(index)
        selected_set.add(index)
        by_base[base] += 1
        if len(selected_indices) >= resources.config.case_candidate_cap:
            break
    if len(selected_indices) < resources.config.case_candidate_cap:
        for index in range(len(case_df)):
            if index in selected_set:
                continue
            selected_indices.append(index)
            if len(selected_indices) >= resources.config.case_candidate_cap:
                break
    case_df = case_df.iloc[selected_indices].reset_index(drop=True)
    case_df.attrs["candidate_stats"] = {
        "pre_prune_count": pre_prune_count,
        "post_prune_count": int(len(case_df)),
        "direct_exact_count": int((case_df["source_direct_exact"] > 0).sum()),
        "indirect_case_count": int((case_df["source_indirect_cited"] > 0).sum()),
    }
    return case_df


def normalize_feature_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns or df.empty:
        return np.zeros(len(df), dtype=np.float32)
    values = df[column].to_numpy(dtype=np.float32)
    vmax = float(values.max()) if len(values) else 0.0
    vmin = float(values.min()) if len(values) else 0.0
    if vmax <= vmin:
        return np.zeros(len(values), dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def apply_case_evidence_model(case_feature_df: pd.DataFrame, evidence_params: Dict[str, float]) -> pd.DataFrame:
    case_df = case_feature_df.copy()
    if case_df.empty:
        return empty_case_frame(scored=True)
    rrf_scaled = normalize_feature_column(case_df, "rrf_score")
    rerank_scaled = normalize_feature_column(case_df, "reranker_score")
    late_scaled = normalize_feature_column(case_df, "late_score")
    support_scaled = normalize_feature_column(case_df, "support_profiles")
    direct_scaled = normalize_feature_column(case_df, "direct_profile_hits")
    raw_scores = (
        float(evidence_params["rrf_weight"]) * rrf_scaled
        + float(evidence_params["reranker_weight"]) * rerank_scaled
        + float(evidence_params["late_weight"]) * late_scaled
        + float(evidence_params["support_weight"]) * support_scaled
        + float(evidence_params["direct_weight"]) * direct_scaled
        + float(evidence_params["regex_bonus"]) * case_df["regex_hit"].to_numpy(dtype=np.float32)
        + float(evidence_params["exact_case_bonus"]) * case_df["is_exact_case"].to_numpy(dtype=np.float32)
    )
    raw_scores *= case_df["idf"].to_numpy(dtype=np.float32)
    raw_scores *= case_df["frequency_weight"].to_numpy(dtype=np.float32)
    base_penalty = np.where(case_df["is_base_case"].to_numpy(dtype=np.float32) > 0, float(evidence_params["base_case_penalty"]), 1.0)
    raw_scores = raw_scores * base_penalty
    case_df["raw_score"] = raw_scores.astype(np.float32)
    raw_std = float(np.std(raw_scores)) if len(raw_scores) else 0.0
    if len(raw_scores) and raw_std >= 1e-6:
        temperature = max(1e-3, float(evidence_params.get("temperature", 1.0)))
        shifted = (raw_scores - float(np.max(raw_scores))) / temperature
        exp_scores = np.exp(np.clip(shifted, -50.0, 0.0))
        total = float(exp_scores.sum())
        if total > 0.0:
            case_df["case_prob"] = (exp_scores / total).astype(np.float32)
        else:
            raw_std = 0.0
    if not len(raw_scores) or raw_std < 1e-6:
        case_df = case_df.sort_values(
            ["source_direct_exact", "max_profile_reranker", "direct_profile_hits", "support_profiles", "rrf_score"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        weights = np.array([1.0 / (rank + 1.5) for rank in range(len(case_df))], dtype=np.float32)
        total = float(weights.sum())
        case_df["case_prob"] = (weights / total).astype(np.float32) if total > 0 else np.zeros(len(case_df), dtype=np.float32)
    case_df = case_df.sort_values(
        ["case_prob", "raw_score", "max_profile_reranker", "support_profiles"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return case_df


def score_case_candidates(
    resources: V44Resources,
    query: str,
    scored_profiles: Sequence[ScoredCaseProfile],
    evidence_params: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    case_feature_df = build_case_candidate_frame(resources, query, scored_profiles)
    if evidence_params is None:
        return case_feature_df
    return apply_case_evidence_model(case_feature_df, evidence_params)


def build_case_graph_support_from_profiles(resources: V44Resources, profiles: Sequence[ScoredCaseProfile]) -> Counter:
    support = Counter()
    bge_art = resources.co_citation_index.get("bge_art", {})
    if not profiles:
        return support
    signal_map = normalize_scores({profile.base_case_id: float(profile.reranker_score or profile.rrf_score) for profile in profiles[:30]})
    for profile in profiles[:30]:
        profile_signal = float(signal_map.get(profile.base_case_id, profile.rrf_score))
        for article, ref_count in profile.cited_statutes[:10]:
            valid = get_valid_citation(article, resources.citation_universe)
            if valid and get_citation_type(valid.citation) == "statute":
                support[valid.citation] += profile_signal * min(int(ref_count), 3) / 3.0
        case_refs = list(profile.exact_case_ids[:5]) + [citation for citation, _ in profile.cited_cases[:5]]
        for citation in case_refs:
            for article in get_top_co_citation_targets(bge_art.get(citation, []), limit=10):
                valid = get_valid_citation(article, resources.citation_universe)
                if valid and get_citation_type(valid.citation) == "statute":
                    support[valid.citation] += profile_signal * 0.75
        official_refs = [profile.base_case_id, *profile.exact_case_ids[:5], *[citation for citation, _ in profile.cited_cases[:5]]]
        for article in get_official_case_statute_targets(resources, official_refs, limit=10):
            valid = get_valid_citation(article, resources.citation_universe)
            if valid and get_citation_type(valid.citation) == "statute":
                support[valid.citation] += profile_signal * 0.85
    return support


