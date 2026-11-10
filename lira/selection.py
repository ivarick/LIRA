"""LIRA selection stage."""

from .core import (
    Dict,
    List,
    Sequence,
    V44Config,
    get_case_group_id,
    get_citation_type,
    np,
    pd,
)
from .statutes import (
    build_statute_count_features,
    case_numeric_column,
)

def get_case_selector_signals(case_df: pd.DataFrame, statute_df: pd.DataFrame, params: Dict[str, object]) -> Dict[str, float]:
    threshold = float(params["case_rerank_threshold"])
    direct_exact = case_numeric_column(case_df, "source_direct_exact")
    reranker_scores = case_numeric_column(case_df, "max_profile_reranker")
    regex_values = case_numeric_column(case_df, "regex_hit")
    direct_exact_hits_above = int(
        ((direct_exact > 0) & (reranker_scores >= threshold)).sum()
    ) if not case_df.empty else 0
    profiles_above_threshold = int((reranker_scores >= threshold).sum()) if not case_df.empty else 0
    regex_hits = int(regex_values.sum()) if not case_df.empty else 0
    top10_case_mass = float(case_df.head(10)["case_prob"].sum()) if not case_df.empty else 0.0
    top10_statute_mass = (
        float(statute_df.nlargest(10, "prob")["prob"].sum()) if not statute_df.empty else 0.0
    )
    return {
        "direct_exact_hits_above": float(direct_exact_hits_above),
        "profiles_above_threshold": float(profiles_above_threshold),
        "regex_hits": float(regex_hits),
        "top10_case_mass": float(top10_case_mass),
        "top10_statute_mass": float(top10_statute_mass),
    }


def classify_query_mode(statute_df: pd.DataFrame, case_df: pd.DataFrame, params: Dict[str, object]) -> str:
    signals = get_case_selector_signals(case_df, statute_df, params)
    if (
        signals["direct_exact_hits_above"] >= 3
        or signals["profiles_above_threshold"] >= 8
        or signals["top10_case_mass"] >= 1.25 * signals["top10_statute_mass"]
    ):
        return "case-heavy"
    if (
        signals["direct_exact_hits_above"] >= 2
        or signals["profiles_above_threshold"] >= 5
        or signals["top10_case_mass"] >= 0.75 * signals["top10_statute_mass"]
    ):
        return "mixed"
    return "statute-heavy"


def get_case_candidate_mask(case_df: pd.DataFrame, params: Dict[str, object]) -> np.ndarray:
    if case_df.empty:
        return np.zeros(0, dtype=bool)
    prob = case_numeric_column(case_df, "case_prob")
    rerank = case_numeric_column(case_df, "max_profile_reranker")
    direct = case_numeric_column(case_df, "source_direct_exact")
    support = case_numeric_column(case_df, "support_profiles")
    regex = case_numeric_column(case_df, "regex_hit")
    return (
        (prob >= float(params["case_floor"]))
        & (rerank >= float(params["case_rerank_threshold"]))
        & ((direct > 0) | (support >= 3.0) | ((regex > 0) & (support >= 2.0)))
    )


def select_case_predictions(case_df: pd.DataFrame, params: Dict[str, object], case_budget: int) -> List[str]:
    if case_df.empty or case_budget <= 0:
        return []
    candidate = case_df.copy()
    candidate["_eligible"] = get_case_candidate_mask(candidate, params)
    supported_exact_bases = {
        get_case_group_id(str(citation))
        for citation, eligible, is_exact in zip(
            candidate["citation"],
            candidate["_eligible"],
            case_numeric_column(candidate, "is_exact_case"),
        )
        if bool(eligible) and float(is_exact) > 0.0
    }
    preds: List[str] = []
    for _, row in candidate.iterrows():
        if not bool(row["_eligible"]):
            continue
        citation = str(row["citation"])
        if float(row.get("is_base_case", 0.0)) > 0.0:
            if float(row.get("support_profiles", 0.0)) < 3.0:
                continue
            if get_case_group_id(citation) in supported_exact_bases:
                continue
        preds.append(citation)
        if len(preds) >= case_budget:
            break
    return preds


def compute_case_budget(case_df: pd.DataFrame, statute_df: pd.DataFrame, params: Dict[str, object], config: V44Config, mode: str) -> int:
    if case_df.empty:
        return 0
    signals = get_case_selector_signals(case_df, statute_df, params)
    budget = (
        1.0 * signals["direct_exact_hits_above"]
        + 0.25 * signals["profiles_above_threshold"]
        + 4.0 * signals["top10_case_mass"]
        + float(params["case_regex_bonus"]) * signals["regex_hits"]
    )
    if mode == "case-heavy":
        budget *= 1.10
    elif mode == "statute-heavy":
        budget *= 0.35
    budget = int(np.clip(np.rint(budget), config.case_count_clip[0], config.case_count_clip[1]))
    if signals["top10_case_mass"] < float(config.case_low_mass_suppress_threshold) and signals["direct_exact_hits_above"] <= 0:
        budget = min(budget, int(config.case_low_mass_max_emit))
    return budget


def default_selector_params(config: V44Config) -> Dict[str, float]:
    return {
        "statute_multiplier": 2.25,
        "statute_floor": 0.01,
        "case_rerank_threshold": 0.15,
        "case_floor": 0.02,
        "case_regex_bonus": 0.5,
    }


def select_statute_predictions(statute_df: pd.DataFrame, floor: float, count: int) -> List[str]:
    if statute_df.empty or count <= 0:
        return []
    ranked = statute_df.sort_values("prob", ascending=False, kind="stable")
    return [
        str(citation)
        for citation, prob in zip(ranked["citation"], ranked["prob"])
        if float(prob) >= float(floor)
    ][:count]


def merge_predictions(statute_preds: Sequence[str], case_preds: Sequence[str]) -> List[str]:
    exact_by_base = {get_case_group_id(citation): citation for citation in case_preds if " E. " in citation}
    merged: List[str] = []
    seen = set()
    for citation in list(statute_preds) + list(case_preds):
        if citation in seen:
            continue
        if get_citation_type(citation) in {"bge", "docket"} and " E. " not in citation and get_case_group_id(citation) in exact_by_base:
            continue
        merged.append(citation)
        seen.add(citation)
    return merged


def select_predictions(statute_df: pd.DataFrame, case_df: pd.DataFrame, query_text: str, bundle: Dict[str, object]) -> List[str]:
    config = V44Config(case_count_clip=tuple(bundle.get("case_clip", V44Config().case_count_clip)))
    query_features = pd.DataFrame([build_statute_count_features(statute_df, query_text, config)])
    feature_cols = bundle["statute_count_feature_cols"]
    count_pred = float(bundle["statute_count_model"].predict(query_features[feature_cols])[0]) if feature_cols else 0.0
    statute_count = int(np.clip(np.rint(count_pred * float(bundle["selector_params"]["statute_multiplier"])), bundle["statute_clip"][0], bundle["statute_clip"][1]))
    mode = classify_query_mode(statute_df, case_df, bundle["selector_params"])
    case_budget = compute_case_budget(case_df, statute_df, bundle["selector_params"], config, mode)

    statute_preds = select_statute_predictions(
        statute_df, float(bundle["selector_params"]["statute_floor"]), statute_count
    )
    case_preds = select_case_predictions(case_df, bundle["selector_params"], case_budget)
    return merge_predictions(statute_preds, case_preds[:case_budget])


