"""LIRA retrieval candidates stage."""

from .retrieval_core import (
    BOILERPLATE,
    Counter,
    Dict,
    List,
    Optional,
    Tuple,
    V43Resources,
    V43_CACHE_PATH,
    V43_PIPELINE_VERSION,
    detect_domain_codes,
    encode_sparse_query,
    extract_citations_regex,
    get_case_group_id,
    get_citation_code,
    get_citation_idf,
    get_citation_type,
    get_valid_citation,
    log,
    math,
    maybe_tqdm,
    np,
    parse_citation_set,
    pd,
)

class V43CandidateGenerator:
    def __init__(self, resources: V43Resources):
        self.r = resources
        self.config = resources.config

    def _new_row(self, query_id: str, citation: str, gold_set: set, detected_codes: set) -> Dict[str, object]:
        citation_type = get_citation_type(citation)
        code = self.r.laws_code.get(citation) or (get_citation_code(citation) or "")
        return {
            "query_id": query_id,
            "citation": citation,
            "citation_type": citation_type,
            "label": int(citation in gold_set),
            "heuristic_score": 0.0,
            "regex_hit": 0,
            "law_dense_hit": 0,
            "law_dense_score": 0.0,
            "law_dense_rank": 10**6,
            "law_sparse_hit": 0,
            "law_sparse_score": 0.0,
            "law_sparse_rank": 10**6,
            "court_dense_hit": 0,
            "court_dense_score": 0.0,
            "court_dense_rank": 10**6,
            "court_dense_consideration_count": 0,
            "court_dense_case_count": 0,
            "case_direct_dense_hit": 0,
            "court_caseexp_hit": 0,
            "court_caseexp_score": 0.0,
            "court_caseexp_rank": 10**6,
            "court_caseexp_case_count": 0,
            "court_sparse_hit": 0,
            "court_sparse_score": 0.0,
            "court_sparse_rank": 10**6,
            "court_sparse_case_count": 0,
            "case_direct_sparse_hit": 0,
            "bm25_hit": 0,
            "bm25_score": 0.0,
            "bm25_rank": 10**6,
            "bm25_case_count": 0,
            "case_direct_bm25_hit": 0,
            "knn_hit": 0,
            "knn_score": 0.0,
            "knn_rank": 10**6,
            "knn_vote_count": 0,
            "cooccur_hit": 0,
            "cooccur_score": 0.0,
            "cooccur_rank": 10**6,
            "case_exact_hit": int(citation_type != "statute" and " E. " in citation),
            "case_base_hit": int(citation_type != "statute" and " E. " not in citation),
            "domain_match": int(code in detected_codes) if code else 0,
            "boilerplate": int(citation in BOILERPLATE),
            "global_df": int(self.r.citation_doc_freq.get(citation, 0)),
            "idf": float(get_citation_idf(citation, self.r.citation_doc_freq, self.r.total_cases)),
            "citation_code": code,
            "is_statute": int(citation_type == "statute"),
            "is_bge": int(citation_type == "bge"),
            "is_docket": int(citation_type == "docket"),
            "_court_dense_cases": set(),
            "_court_caseexp_cases": set(),
            "_court_sparse_cases": set(),
            "_bm25_cases": set(),
        }

    def _ensure_row(
        self,
        rows: Dict[str, Dict[str, object]],
        query_id: str,
        citation: str,
        gold_set: set,
        detected_codes: set,
    ) -> Dict[str, object]:
        row = rows.get(citation)
        if row is None:
            row = self._new_row(query_id, citation, gold_set, detected_codes)
            rows[citation] = row
        return row

    def _add_signal(
        self,
        rows: Dict[str, Dict[str, object]],
        query_id: str,
        citation: str,
        gold_set: set,
        detected_codes: set,
        channel: str,
        score: float,
        rank: int,
        heuristic_boost: float,
        case_key: Optional[str] = None,
        consideration_count: int = 0,
        knn_vote_count: int = 0,
        case_match: str = "",
        is_case_direct: bool = False,
    ) -> None:
        row = self._ensure_row(rows, query_id, citation, gold_set, detected_codes)
        hit_key = f"{channel}_hit"
        score_key = f"{channel}_score"
        rank_key = f"{channel}_rank"
        if hit_key in row:
            row[hit_key] = 1
        if score_key in row:
            row[score_key] = max(float(row[score_key]), float(score))
        if rank_key in row:
            row[rank_key] = min(int(row[rank_key]), int(rank) + 1)
        row["heuristic_score"] = float(row["heuristic_score"]) + float(heuristic_boost)
        if consideration_count:
            row["court_dense_consideration_count"] = int(row["court_dense_consideration_count"]) + int(consideration_count)
        if knn_vote_count:
            row["knn_vote_count"] = int(row["knn_vote_count"]) + int(knn_vote_count)
        if case_match == "exact":
            row["case_exact_hit"] = 1
        elif case_match == "base":
            row["case_base_hit"] = 1
        if is_case_direct and channel == "court_dense":
            row["case_direct_dense_hit"] = 1
        if is_case_direct and channel == "court_sparse":
            row["case_direct_sparse_hit"] = 1
        if is_case_direct and channel == "bm25":
            row["case_direct_bm25_hit"] = 1
        if case_key is not None:
            hidden_key = {
                "court_dense": "_court_dense_cases",
                "court_caseexp": "_court_caseexp_cases",
                "court_sparse": "_court_sparse_cases",
                "bm25": "_bm25_cases",
            }.get(channel)
            if hidden_key:
                row[hidden_key].add(case_key)

    def _search_court_dense(self, query_embedding: np.ndarray) -> List[Tuple[int, float]]:
        scores, indices = self.r.court_faiss.search(query_embedding[np.newaxis, :], self.config.faiss_search_k)
        candidate_idx = indices[0]
        candidate_idx = candidate_idx[candidate_idx >= 0]
        if candidate_idx.size == 0:
            return []
        exact_vectors = np.asarray(self.r.court_embeddings[candidate_idx], dtype=np.float32)
        exact_scores = exact_vectors @ query_embedding
        order = np.argsort(exact_scores)[::-1][: self.config.faiss_final_k]
        return [(int(candidate_idx[i]), float(exact_scores[i])) for i in order]

    def generate_query_candidates(
        self,
        query_id: str,
        query: str,
        gold_set: Optional[set] = None,
        knn_pool_indices: Optional[np.ndarray] = None,
        preserve_gold_positives: bool = False,
    ) -> pd.DataFrame:
        gold_set = gold_set or set()
        detected_codes = detect_domain_codes(query)
        rows: Dict[str, Dict[str, object]] = {}
        query_embedding, query_sparse = encode_sparse_query(self.r.encoder, query)

        for citation in extract_citations_regex(query):
            valid = get_valid_citation(citation, self.r.citation_universe)
            if valid:
                self._add_signal(rows, query_id, valid.citation, gold_set, detected_codes, "regex", 1.0, 0, 25.0, case_match=valid.case_match)

        law_scores = query_embedding @ self.r.laws_embeddings.T
        top_law_idx = np.argsort(law_scores)[::-1][: self.config.laws_dense_k]
        for rank, idx in enumerate(top_law_idx):
            sim = float(law_scores[idx])
            if sim < self.config.laws_dense_min_sim:
                break
            citation = get_valid_citation(self.r.laws_citations[idx], self.r.citation_universe)
            if not citation:
                continue
            code = self.r.laws_code.get(citation.citation, "")
            boost = sim * (8.0 if rank < 10 else 4.0 if rank < 30 else 2.0)
            if code and code in detected_codes:
                boost *= 1.25
            self._add_signal(rows, query_id, citation.citation, gold_set, detected_codes, "law_dense", sim, rank, boost, case_match=citation.case_match)

        sparse_law_results = self.r.laws_sparse_index.search(query_sparse, self.config.laws_sparse_k)
        if sparse_law_results:
            top_sparse_law = max(score for _, score in sparse_law_results) or 1.0
            for rank, (doc_idx, score) in enumerate(sparse_law_results):
                citation = get_valid_citation(self.r.laws_sparse_index.doc_ids[doc_idx], self.r.citation_universe)
                if not citation:
                    continue
                norm_score = float(score) / float(top_sparse_law)
                boost = norm_score * (6.0 if rank < 10 else 3.0 if rank < 30 else 1.5)
                self._add_signal(rows, query_id, citation.citation, gold_set, detected_codes, "law_sparse", norm_score, rank, boost, case_match=citation.case_match)

        seen_cases: set = set()
        for rank, (idx, sim) in enumerate(self._search_court_dense(query_embedding)):
            if sim < self.config.court_dense_min_sim:
                break
            case_key = get_case_group_id(self.r.court_case_ids[idx])
            direct_case = get_valid_citation(self.r.court_case_ids[idx], self.r.citation_universe)
            if direct_case:
                direct_boost = sim * (7.0 if rank < 20 else 3.0)
                self._add_signal(
                    rows,
                    query_id,
                    direct_case.citation,
                    gold_set,
                    detected_codes,
                    "court_dense",
                    sim,
                    rank,
                    direct_boost,
                    case_key=case_key,
                    case_match=direct_case.case_match,
                    is_case_direct=True,
                )
            consideration_cits = self.r.court_case_citations[idx]
            for citation in consideration_cits:
                valid = get_valid_citation(citation, self.r.citation_universe)
                if not valid:
                    continue
                idf = get_citation_idf(valid.citation, self.r.citation_doc_freq, self.r.total_cases)
                boost = sim * idf * (10.0 if rank < 30 else 5.0 if rank < 100 else 2.5 if rank < 300 else 1.0)
                self._add_signal(
                    rows,
                    query_id,
                    valid.citation,
                    gold_set,
                    detected_codes,
                    "court_dense",
                    sim,
                    rank,
                    boost,
                    case_key=case_key,
                    consideration_count=1,
                    case_match=valid.case_match,
                )
            if case_key not in seen_cases and len(seen_cases) < 300:
                seen_cases.add(case_key)
                for citation, ref_count in self.r.case_extracted_cits.get(case_key, Counter()).most_common(20):
                    valid = get_valid_citation(citation, self.r.citation_universe)
                    if not valid:
                        continue
                    idf = get_citation_idf(valid.citation, self.r.citation_doc_freq, self.r.total_cases)
                    depth = len(seen_cases)
                    if depth <= 50:
                        boost = sim * 6.0 * idf * min(ref_count, 3) / 3.0
                    elif depth <= 150:
                        boost = sim * 2.0 * idf * min(ref_count, 3) / 3.0
                    else:
                        boost = sim * 0.75 * idf * min(ref_count, 3) / 3.0
                    self._add_signal(
                        rows,
                        query_id,
                        valid.citation,
                        gold_set,
                        detected_codes,
                        "court_caseexp",
                        sim,
                        rank,
                        boost,
                        case_key=case_key,
                        case_match=valid.case_match,
                    )
                docket_citation = get_valid_citation(case_key, self.r.citation_universe)
                if docket_citation:
                    docket_boost = sim * (5.0 if rank < 10 else 2.5)
                    self._add_signal(
                        rows,
                        query_id,
                        docket_citation.citation,
                        gold_set,
                        detected_codes,
                        "court_caseexp",
                        sim,
                        rank,
                        docket_boost,
                        case_key=case_key,
                        case_match=docket_citation.case_match,
                    )

        if self.r.court_sparse_index is not None:
            court_sparse_results = self.r.court_sparse_index.search(query_sparse, self.config.court_sparse_k)
            if court_sparse_results:
                top_sparse_case = max(score for _, score in court_sparse_results) or 1.0
                for rank, (doc_idx, score) in enumerate(court_sparse_results):
                    case_key = self.r.court_sparse_index.doc_ids[doc_idx]
                    direct_case = get_valid_citation(case_key, self.r.citation_universe)
                    norm_score = float(score) / float(top_sparse_case)
                    if direct_case:
                        self._add_signal(
                            rows,
                            query_id,
                            direct_case.citation,
                            gold_set,
                            detected_codes,
                            "court_sparse",
                            norm_score,
                            rank,
                            norm_score * (5.0 if rank < 20 else 2.0),
                            case_key=case_key,
                            case_match=direct_case.case_match,
                            is_case_direct=True,
                        )
                    for citation, ref_count in self.r.case_extracted_cits.get(case_key, Counter()).most_common(20):
                        valid = get_valid_citation(citation, self.r.citation_universe)
                        if not valid:
                            continue
                        idf = get_citation_idf(valid.citation, self.r.citation_doc_freq, self.r.total_cases)
                        boost = norm_score * 4.0 * idf * min(ref_count, 3) / 3.0
                        self._add_signal(
                            rows,
                            query_id,
                            valid.citation,
                            gold_set,
                            detected_codes,
                            "court_sparse",
                            norm_score,
                            rank,
                            boost,
                            case_key=case_key,
                            case_match=valid.case_match,
                        )

        bm25_results = self.r.court_bm25.search(query, top_k=max(self.config.bm25_k, 50))
        if bm25_results:
            top_bm25 = max(score for _, score in bm25_results) or 1.0
            for rank, (case_key, score) in enumerate(bm25_results[: self.config.bm25_k]):
                norm_score = float(score) / float(top_bm25)
                direct_case = get_valid_citation(case_key, self.r.citation_universe)
                if direct_case:
                    self._add_signal(
                        rows,
                        query_id,
                        direct_case.citation,
                        gold_set,
                        detected_codes,
                        "bm25",
                        norm_score,
                        rank,
                        norm_score * (5.0 if rank < 10 else 2.5),
                        case_key=case_key,
                        case_match=direct_case.case_match,
                        is_case_direct=True,
                    )
                for citation, ref_count in self.r.case_extracted_cits.get(case_key, Counter()).most_common(20):
                    valid = get_valid_citation(citation, self.r.citation_universe)
                    if not valid:
                        continue
                    idf = get_citation_idf(valid.citation, self.r.citation_doc_freq, self.r.total_cases)
                    boost = norm_score * 5.0 * idf * min(ref_count, 3) / 3.0
                    self._add_signal(
                        rows,
                        query_id,
                        valid.citation,
                        gold_set,
                        detected_codes,
                        "bm25",
                        norm_score,
                        rank,
                        boost,
                        case_key=case_key,
                        case_match=valid.case_match,
                    )

        if knn_pool_indices is not None and len(knn_pool_indices):
            pool_embs = self.r.train_query_embs[knn_pool_indices]
            sims = pool_embs @ query_embedding
            top_knn = np.argsort(sims)[::-1][: self.config.knn_k]
            for rank, local_idx in enumerate(top_knn):
                sim = float(sims[local_idx])
                if sim < 0.45:
                    break
                train_idx = int(knn_pool_indices[local_idx])
                weight = sim * (8.0 if rank < 3 else 4.0)
                for citation in self.r.train_gold_cits[train_idx]:
                    valid = get_valid_citation(citation, self.r.citation_universe)
                    if not valid:
                        continue
                    cit_weight = weight * (1.5 if get_citation_type(valid.citation) != "statute" else 1.0)
                    self._add_signal(
                        rows,
                        query_id,
                        valid.citation,
                        gold_set,
                        detected_codes,
                        "knn",
                        sim,
                        rank,
                        cit_weight,
                        knn_vote_count=1,
                        case_match=valid.case_match,
                    )

        for source_citation, source_row in sorted(rows.items(), key=lambda kv: float(kv[1]["heuristic_score"]), reverse=True)[:30]:
            for linked_citation, co_count in self.r.cooccurrence_graph.get(source_citation, [])[:5]:
                valid = get_valid_citation(linked_citation, self.r.citation_universe)
                if not valid or valid.citation == source_citation:
                    continue
                co_weight = min(3.0, math.log2(float(co_count) + 1.0) * 0.5)
                boost = float(source_row["heuristic_score"]) * co_weight * 0.15
                self._add_signal(
                    rows,
                    query_id,
                    valid.citation,
                    gold_set,
                    detected_codes,
                    "cooccur",
                    co_weight,
                    0,
                    boost,
                    case_match=valid.case_match,
                )

        finalized_rows: List[Dict[str, object]] = []
        for citation, row in rows.items():
            row["court_dense_case_count"] = len(row.pop("_court_dense_cases"))
            row["court_caseexp_case_count"] = len(row.pop("_court_caseexp_cases"))
            row["court_sparse_case_count"] = len(row.pop("_court_sparse_cases"))
            row["bm25_case_count"] = len(row.pop("_bm25_cases"))
            row["n_channels"] = int(
                row["regex_hit"]
                + row["law_dense_hit"]
                + row["law_sparse_hit"]
                + row["court_dense_hit"]
                + row["court_caseexp_hit"]
                + row["court_sparse_hit"]
                + row["bm25_hit"]
                + row["knn_hit"]
                + row["cooccur_hit"]
            )
            row["support_total"] = int(
                row["court_dense_consideration_count"]
                + row["court_dense_case_count"]
                + row["court_caseexp_case_count"]
                + row["court_sparse_case_count"]
                + row["bm25_case_count"]
                + row["knn_vote_count"]
            )
            finalized_rows.append(row)

        finalized_rows.sort(
            key=lambda item: (
                float(item["heuristic_score"]),
                float(item["law_dense_score"]),
                float(item["court_dense_score"]),
                float(item["bm25_score"]),
            ),
            reverse=True,
        )

        if len(finalized_rows) > self.config.candidate_cap_per_query:
            positives = [row for row in finalized_rows if row["label"] == 1]
            trimmed = finalized_rows[: self.config.candidate_cap_per_query]
            kept = {row["citation"] for row in trimmed}
            if preserve_gold_positives:
                for row in positives:
                    if row["citation"] not in kept:
                        trimmed.append(row)
            finalized_rows = trimmed

        return pd.DataFrame(finalized_rows)


NON_FEATURE_COLUMNS = {
    "query_id",
    "citation",
    "citation_type",
    "label",
    "fold",
    "citation_code",
    "raw_score",
    "prob",
}

COUNT_FEATURE_META_COLUMNS = {
    "query_id",
    "query_text",
    "gold_statute_count",
    "gold_case_count",
}


def get_feature_columns(candidate_df: pd.DataFrame) -> List[str]:
    feature_cols = [col for col in candidate_df.columns if col not in NON_FEATURE_COLUMNS]
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(candidate_df[col])]
    return sorted(numeric_cols)


def generate_fold_candidate_tables(resources: V43Resources) -> pd.DataFrame:
    V43_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    generator = V43CandidateGenerator(resources)
    all_folds: List[pd.DataFrame] = []
    train_indices = np.arange(len(resources.train_df))

    for fold in range(resources.config.n_splits):
        fold_path = V43_CACHE_PATH / f"fold_{fold}_{V43_PIPELINE_VERSION}.pkl"
        if fold_path.exists():
            fold_df = pd.read_pickle(fold_path)
            all_folds.append(fold_df)
            log(f"   Candidate table fold {fold}: cached")
            continue

        log(f"   Generating candidate table for fold {fold}...")
        heldout_mask = resources.train_df["query_id"].map(resources.folds).astype(int) == fold
        heldout_df = resources.train_df.loc[heldout_mask].reset_index(drop=False)
        knn_pool_indices = train_indices[~heldout_mask.to_numpy()]
        fold_rows: List[pd.DataFrame] = []

        for _, row in maybe_tqdm(heldout_df.iterrows(), total=len(heldout_df), desc=f"Fold {fold} queries"):
            gold_set = set(parse_citation_set(row["gold_citations"]))
            candidate_df = generator.generate_query_candidates(
                query_id=str(row["query_id"]),
                query=str(row["query"]),
                gold_set=gold_set,
                knn_pool_indices=knn_pool_indices,
                preserve_gold_positives=True,
            )
            if candidate_df.empty:
                candidate_df = pd.DataFrame([generator._new_row(str(row["query_id"]), "", gold_set, detect_domain_codes(str(row["query"])))])
            candidate_df["fold"] = fold
            fold_rows.append(candidate_df)

        fold_df = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
        fold_df.to_pickle(fold_path)
        all_folds.append(fold_df)

    return pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()


def generate_full_train_candidate_table(resources: V43Resources) -> pd.DataFrame:
    full_path = V43_CACHE_PATH / f"train_full_{V43_PIPELINE_VERSION}.pkl"
    if full_path.exists():
        return pd.read_pickle(full_path)
    generator = V43CandidateGenerator(resources)
    rows: List[pd.DataFrame] = []
    all_indices = np.arange(len(resources.train_df))
    for idx, row in maybe_tqdm(resources.train_df.reset_index(drop=False).iterrows(), total=len(resources.train_df), desc="Full-train candidates"):
        knn_pool = all_indices[all_indices != int(row["index"])]
        gold_set = set(parse_citation_set(row["gold_citations"]))
        candidate_df = generator.generate_query_candidates(
            query_id=str(row["query_id"]),
            query=str(row["query"]),
            gold_set=gold_set,
            knn_pool_indices=knn_pool,
            preserve_gold_positives=True,
        )
        if candidate_df.empty:
            candidate_df = pd.DataFrame([generator._new_row(str(row["query_id"]), "", gold_set, detect_domain_codes(str(row["query"])))])
        candidate_df["fold"] = -1
        rows.append(candidate_df)
    full_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    full_df.to_pickle(full_path)
    return full_df


