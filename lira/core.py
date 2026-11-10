#!/usr/bin/env python3
import argparse  # noqa: F401 - re-exported to pipeline stages
import ctypes
import gc
import hashlib
import json
import math
import os
import pickle
import re
import time  # noqa: F401 - re-exported to pipeline stages
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import lightgbm as lgb  # noqa: F401 - re-exported to pipeline stages
import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel
from sklearn.linear_model import LogisticRegression  # noqa: F401 - re-exported to pipeline stages

try:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
except Exception:  
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

import lira_retrieval as v43

DATA_PATH = Path("./data")
CACHE_PATH = Path("./cache")
OUTPUT_PATH = Path("./output")


def resolve_model_snapshot(snapshots_dir: Path, override_env: str) -> Path:
    override = os.getenv(override_env)
    if override:
        return Path(override).expanduser()
    if snapshots_dir.exists():
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
        if snapshots:
            return snapshots[0]
    return snapshots_dir


V44_CACHE_PATH = CACHE_PATH / "v44_candidate_tables"
V44_MODEL_PATH = CACHE_PATH / "v44_models"
V44_PIPELINE_VERSION = "v448"
STATUTE_CANDIDATE_SCHEMA_VERSION = "statute_candidates_v1"
STATUTE_NON_FEATURE_COLUMNS = {"query_id", "citation", "label", "fold", "raw_score", "prob"}
STATUTE_REQUIRED_CANDIDATE_COLUMNS = {"query_id", "citation", "label", "heuristic_score", "fold", "reranker_score"}

V44_CASE_PROFILES_PATH = CACHE_PATH / "v44_case_profiles.pkl"
V44_CASE_DENSE_PATH = CACHE_PATH / "v44_case_dense.npy"
V44_CASE_DENSE_INDEX_PATH = CACHE_PATH / "v44_case_dense.index"
V44_CASE_DENSE_META_PATH = CACHE_PATH / "v44_case_dense.meta.json"
V44_CASE_SPARSE_PATH = CACHE_PATH / "v44_case_sparse.pkl"
V44_CASE_BM25_PATH = CACHE_PATH / "v44_case_bm25.pkl"
V44_LAWS_BM25_PATH = CACHE_PATH / "v44_laws_bm25.pkl"
V44_QUERY_VIEWS_PATH = CACHE_PATH / "v44_query_views.pkl"
V44_MODEL_BUNDLE_PATH = V44_MODEL_PATH / "model_bundle.pkl"
V44_OFFICIAL_ENRICHMENT_PATH = CACHE_PATH / "official_swiss_enrichment_v1.pkl"
V44_MANIFEST_PATH = CACHE_PATH / "v44_kaggle_manifest.json"
V44_OOF_STATUTE_PATH = V44_CACHE_PATH / f"oof_statute_{V44_PIPELINE_VERSION}.pkl"
V44_FULL_STATUTE_PATH = V44_CACHE_PATH / f"full_statute_{V44_PIPELINE_VERSION}.pkl"

COURT_FAISS_PATH = CACHE_PATH / "faiss_court_v43.index"
FOLDS_PATH = CACHE_PATH / "folds_v43.pkl"
LAWS_SPARSE_POSTINGS_PATH = CACHE_PATH / "v43_laws_sparse_postings.pkl"
QWEN_LOCAL_PATH = CACHE_PATH / "qwen_gptq_int4"
RERANKER_PATH = resolve_model_snapshot(
    CACHE_PATH / "hf_bypass" / "hub" / "models--BAAI--bge-reranker-v2-m3" / "snapshots",
    "LIRA_RERANKER_PATH",
)

QUERY_VIEW_NAMES = ("original", "issue_view", "precedent_view")
QUERY_VIEW_CHANNELS = ("dense", "sparse", "bm25")
CASE_CHANNELS = ("dense", "sparse", "bm25", "backproj")
STOPWORDS = {
    "der", "die", "das", "und", "oder", "ein", "eine", "einer", "eines", "dem", "den", "des",
    "von", "mit", "auf", "fur", "ist", "sind", "war", "wird", "werden", "bei", "im", "in", "am",
    "zu", "zur", "zum", "an", "nach", "vor", "gegen", "aus", "auch", "als", "dass", "ob", "wenn",
    "the", "and", "for", "with", "that", "from", "into", "were", "have", "has", "this", "their",
    "should", "could", "would", "under", "over", "within", "between", "while", "where", "which",
}

EfficientBM25 = v43.EfficientBM25
SparsePostingIndex = v43.SparsePostingIndex
CitationUniverse = v43.CitationUniverse
CitationMatch = v43.CitationMatch

log = v43.log
normalize_citation = v43.normalize_citation
parent_statute_citation = v43.parent_statute_citation
extract_citations_regex = v43.extract_citations_regex
get_case_group_id = v43.get_case_group_id
normalize_case_citation = v43.normalize_case_citation
get_citation_type = v43.get_citation_type
get_citation_code = v43.get_citation_code
parse_citation_set = v43.parse_citation_set
macro_f1_from_sets = v43.macro_f1_from_sets
choose_prediction_column = v43.choose_prediction_column
make_bucket_labels = v43.make_bucket_labels
save_pickle = v43.save_pickle
load_pickle = v43.load_pickle
prune_weights = v43.prune_weights
build_sparse_postings = v43.build_sparse_postings
build_citation_universe = v43.build_citation_universe
get_valid_citation = v43.get_valid_citation
get_citation_idf = v43.get_citation_idf
v43_detect_domain_codes = v43.detect_domain_codes
encode_sparse_query = v43.encode_sparse_query
build_folds = v43.build_folds
load_encoder = v43.load_encoder
train_ranker = v43.train_ranker
predict_with_ranker = v43.predict_with_ranker
fit_calibrator = v43.fit_calibrator
apply_calibrator = v43.apply_calibrator
maybe_tqdm = v43.maybe_tqdm
encode_with_optional_quiet = v43.encode_with_optional_quiet


@dataclass
class V44Config:
    encoder_path: Path = CACHE_PATH / "bge_m3_finetuned" / "model"
    train_query_embeddings_path: Path = CACHE_PATH / "train_query_embeddings.npy"
    court_dense_path: Path = CACHE_PATH / "bge_m3_dense.npy"
    court_meta_path: Path = CACHE_PATH / "court_meta.pkl"
    laws_dense_path: Path = CACHE_PATH / "bge_m3_laws_dense.npy"
    laws_sparse_path: Path = CACHE_PATH / "bge_m3_laws_sparse.pkl"
    court_bm25_path: Path = CACHE_PATH / "court_bm25.pkl"
    co_citation_path: Path = CACHE_PATH / "co_citation_index.pkl"
    official_enrichment_path: Path = V44_OFFICIAL_ENRICHMENT_PATH
    n_splits: int = 5
    random_seed: int = 42
    case_profile_chunk_rows: int = 50_000
    case_profile_snippet_count: int = 3
    case_profile_snippet_tokens: int = 256
    case_profile_exact_limit: int = 20
    case_profile_statute_limit: int = 10
    case_profile_case_limit: int = 10
    case_profile_batch_size: int = 8
    sparse_doc_top_tokens: int = 64
    case_dense_k: int = 300
    case_sparse_k: int = 300
    case_bm25_k: int = 300
    case_backproj_k: int = 300
    case_profile_shortlist_k: int = 300
    case_profile_rerank_k: int = 120
    enable_case_bm25: bool = True
    enable_case_backproj: bool = True
    enable_case_rerank: bool = True
    case_direct_profile_limit: int = 60
    case_indirect_profile_limit: int = 30
    case_indirect_citations_per_profile: int = 5
    case_candidate_cap: int = 120
    case_diversity_per_base: int = 3
    case_low_mass_suppress_threshold: float = 0.12
    case_low_mass_max_emit: int = 1
    laws_dense_k: int = 300
    laws_sparse_k: int = 300
    laws_bm25_k: int = 200
    laws_graph_k: int = 32
    laws_union_pool_cap: int = 1500
    laws_shortlist_k: int = 200
    laws_rerank_k: int = 200
    dense_min_sim: float = 0.18
    rrf_k: int = 60
    reranker_batch_size: int = 0
    query_prob_thresholds: Tuple[float, ...] = (0.003, 0.005, 0.01, 0.02, 0.03, 0.05)
    statute_floor_grid: Tuple[float, ...] = (0.01, 0.02, 0.03)
    case_floor_grid: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05)
    case_rerank_threshold_grid: Tuple[float, ...] = (0.10, 0.15, 0.20)
    statute_multiplier_grid: Tuple[float, ...] = (1.5, 2.0, 2.5)
    case_regex_bonus_grid: Tuple[float, ...] = (0.25, 0.5, 1.0)
    case_evidence_rrf_weight_grid: Tuple[float, ...] = (0.10, 0.20, 0.30)
    case_evidence_reranker_weight_grid: Tuple[float, ...] = (0.35, 0.50, 0.65)
    case_evidence_late_weight_grid: Tuple[float, ...] = (0.05, 0.10, 0.15)
    case_evidence_support_weight_grid: Tuple[float, ...] = (0.02, 0.05, 0.10)
    case_evidence_direct_weight_grid: Tuple[float, ...] = (0.15, 0.25, 0.35)
    case_evidence_regex_bonus_grid: Tuple[float, ...] = (0.05, 0.10, 0.20)
    case_evidence_exact_bonus_grid: Tuple[float, ...] = (0.10, 0.20, 0.30)
    case_evidence_base_penalty_grid: Tuple[float, ...] = (0.35, 0.50, 0.65)
    case_evidence_temperature_grid: Tuple[float, ...] = (0.5, 1.0, 1.5)
    case_evidence_floor: float = 0.01
    statute_count_clip: Tuple[int, int] = (0, 40)
    case_count_clip: Tuple[int, int] = (0, 25)
    case_profile_faiss_factory: str = "IVF2048,PQ64x8"
    case_profile_faiss_train_size: int = 100_000
    case_profile_faiss_add_batch: int = 20_000
    case_profile_faiss_search_k: int = 1024
    case_profile_faiss_nprobe: int = 32
    ranker_estimators: int = 350
    ranker_learning_rate: float = 0.05
    ranker_num_leaves: int = 63
    count_model_estimators: int = 250
    count_model_learning_rate: float = 0.05
    count_model_num_leaves: int = 31

    def __post_init__(self) -> None:
        if not has_high_memory_profile():
            return
        if self.case_profile_batch_size == 8:
            self.case_profile_batch_size = 12
        if self.case_profile_faiss_add_batch == 20_000:
            self.case_profile_faiss_add_batch = 40_000


@dataclass
class QueryViews:
    original: str
    issue_view: str
    precedent_view: str
    source: str = "fallback"

    def as_dict(self) -> Dict[str, str]:
        return {
            "original": self.original,
            "issue_view": self.issue_view,
            "precedent_view": self.precedent_view,
            "source": self.source,
        }


@dataclass
class CaseProfile:
    base_case_id: str
    exact_case_ids: List[str]
    cited_statutes: List[Tuple[str, int]]
    cited_cases: List[Tuple[str, int]]
    snippets: List[str]
    profile_text: str
    total_considerations: int


@dataclass
class ScoredCaseProfile:
    base_case_id: str
    rrf_score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    bm25_score: float = 0.0
    backproj_score: float = 0.0
    late_score: float = 0.0
    reranker_score: float = 0.0
    profile_text: str = ""
    exact_case_ids: List[str] = field(default_factory=list)
    cited_cases: List[Tuple[str, int]] = field(default_factory=list)
    cited_statutes: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class V44Resources:
    config: V44Config
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    laws_df: pd.DataFrame
    encoder: BGEM3FlagModel
    citation_universe: CitationUniverse
    laws_embeddings: np.ndarray
    laws_citations: List[str]
    laws_text: Dict[str, str]
    laws_title: Dict[str, str]
    laws_code: Dict[str, str]
    laws_sparse_index: SparsePostingIndex
    laws_sparse_meta: Dict[str, object]
    laws_bm25: EfficientBM25
    laws_citation_to_idx: Dict[str, int]
    case_profiles: Dict[str, CaseProfile]
    case_profile_ids: List[str]
    case_profile_dense: np.ndarray
    case_profile_faiss: faiss.Index
    case_profile_sparse_index: SparsePostingIndex
    case_profile_bm25: EfficientBM25
    court_embeddings: np.ndarray
    court_case_ids: List[str]
    court_case_citations: List[List[str]]
    court_faiss: faiss.Index
    case_extracted_cits: Dict[str, Counter]
    co_citation_index: Dict[str, Dict[str, List[str]]]
    official_enrichment: Dict[str, object]
    citation_doc_freq: Counter
    total_cases: int
    folds: Dict[str, int]
    query_views: Dict[str, QueryViews]
    reranker_tokenizer: Optional[object]
    reranker_model: Optional[object]
    reranker_device: str
    reranker_batch_size: int
    require_reranker: bool
    runtime_cache: Dict[str, Dict[str, object]] = field(default_factory=dict)
    runtime_diagnostics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


def truncate_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", str(text))
    return " ".join(words[:max_words]).strip()


def build_bm25_from_docs(doc_ids: Sequence[str], texts: Sequence[str]) -> EfficientBM25:
    bm25 = EfficientBM25()
    for doc_id, text in maybe_tqdm(list(zip(doc_ids, texts)), desc="Building BM25"):
        tokens = bm25._tok(text)
        if not tokens:
            continue
        doc_idx = len(bm25.doc_ids)
        bm25.doc_ids.append(str(doc_id))
        bm25.doc_lens.append(len(tokens))
        counts = Counter(tokens)
        for token, tf_val in counts.items():
            bm25.inverted[token].append((doc_idx, int(tf_val)))
            bm25.df[token] += 1
    bm25.N = len(bm25.doc_ids)
    bm25.avgdl = float(np.mean(bm25.doc_lens)) if bm25.doc_lens else 1.0
    return bm25


def load_cached_frame(
    path: Path,
    cache_kind: str,
    required_columns: Optional[set[str]] = None,
) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        payload = pd.read_pickle(path)
    except (pickle.UnpicklingError, EOFError, OSError, ValueError):
        log(f"   Removing corrupted cache: {path.name}")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    if not isinstance(payload, dict) or "frame" not in payload or "meta" not in payload:
        log(f"   Removing stale cache without metadata: {path.name}")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    frame = payload.get("frame")
    meta = payload.get("meta") or {}
    if not isinstance(frame, pd.DataFrame):
        log(f"   Removing invalid cache payload: {path.name}")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    if meta.get("pipeline_version") != V44_PIPELINE_VERSION or meta.get("cache_kind") != cache_kind or meta.get("schema_version") != STATUTE_CANDIDATE_SCHEMA_VERSION:
        log(f"   Removing stale cache version: {path.name}")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    required_columns = required_columns or set()
    if not required_columns.issubset(set(frame.columns)):
        log(f"   Removing schema-mismatched cache: {path.name}")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    return frame


def save_cached_frame(df: pd.DataFrame, path: Path, cache_kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    payload = {
        "meta": {
            "pipeline_version": V44_PIPELINE_VERSION,
            "cache_kind": cache_kind,
            "schema_version": STATUTE_CANDIDATE_SCHEMA_VERSION,
        },
        "frame": df,
    }
    pd.to_pickle(payload, temp_path)
    os.replace(temp_path, path)


def compose_case_profile_text(
    base_case_id: str,
    exact_case_ids: Sequence[str],
    cited_statutes: Sequence[Tuple[str, int]],
    cited_cases: Sequence[Tuple[str, int]],
    snippets: Sequence[str],
) -> str:
    return (
        f"Case: {base_case_id}\n"
        f"Exact citations: {'; '.join(exact_case_ids) or 'none'}\n"
        f"Top statutes: {'; '.join(c for c, _ in cited_statutes) or 'none'}\n"
        f"Top cited cases: {'; '.join(c for c, _ in cited_cases) or 'none'}\n"
        f"Snippets: {' '.join(snippets) or 'none'}"
    )


def build_case_extracted_cits_from_meta(
    court_case_ids: Sequence[object],
    court_case_citations: Sequence[Sequence[object]],
) -> Dict[str, Counter]:
    extracted: Dict[str, Counter] = {}
    for raw_case_id, citations in zip(court_case_ids, court_case_citations):
        case_id = get_case_group_id(str(raw_case_id))
        if not case_id:
            continue
        target = extracted.setdefault(case_id, Counter())
        for citation in citations or []:
            citation_text = WS_RE.sub(" ", str(citation).strip())
            if citation_text:
                target[citation_text] += 1
    return extracted


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def threshold_label(value: float) -> str:
    return str(value).replace(".", "p")


def normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    values = np.array(list(score_map.values()), dtype=np.float32)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return {key: 1.0 for key in score_map}
    return {key: float((value - vmin) / (vmax - vmin)) for key, value in score_map.items()}


def rrf_merge(rank_maps: Sequence[Dict[str, int]], k: int) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for rank_map in rank_maps:
        for doc_id, rank in rank_map.items():
            scores[str(doc_id)] += 1.0 / (k + int(rank) + 1)
    return dict(scores)


def get_top_terms(text: str, keep: int = 12) -> List[str]:
    tokens = [t for t in re.findall(r"\b\w+\b", str(text).lower()) if len(t) >= 4 and t not in STOPWORDS and not t.isdigit()]
    return [term for term, _ in Counter(tokens).most_common(keep)]


def to_v43_config(config: V44Config) -> v43.V43Config:
    return v43.V43Config(
        encoder_path=config.encoder_path,
        train_query_embeddings_path=config.train_query_embeddings_path,
        court_dense_path=config.court_dense_path,
        court_meta_path=config.court_meta_path,
        laws_dense_path=config.laws_dense_path,
        laws_sparse_path=config.laws_sparse_path,
        court_bm25_path=config.court_bm25_path,
        n_splits=config.n_splits,
        random_seed=config.random_seed,
        sparse_doc_top_tokens=config.sparse_doc_top_tokens,
        ranker_estimators=config.ranker_estimators,
        ranker_learning_rate=config.ranker_learning_rate,
        ranker_num_leaves=config.ranker_num_leaves,
        count_model_estimators=config.count_model_estimators,
        count_model_learning_rate=config.count_model_learning_rate,
        count_model_num_leaves=config.count_model_num_leaves,
    )


def ensure_required_artifact(path: Path, label: str, prepare_command: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing {label}: {path}. Prepare it with `{prepare_command}`.")


def empty_official_enrichment() -> Dict[str, object]:
    return {
        "version": "official_swiss_enrichment_v1",
        "statute_aliases": {},
        "code_aliases": {},
        "article_neighbors": {},
        "case_aliases": {},
        "case_statute_graph": {},
        "case_case_graph": {},
        "source_meta": {},
    }


def load_official_enrichment(config: V44Config) -> Dict[str, object]:
    if not config.official_enrichment_path.exists():
        return empty_official_enrichment()
    payload = load_pickle(config.official_enrichment_path)
    merged = empty_official_enrichment()
    if isinstance(payload, dict):
        merged.update(payload)
    return merged


def collect_v44_asset_paths(config: Optional[V44Config] = None, include_bundle: bool = True) -> Dict[str, Path]:
    cfg = config or V44Config()
    paths = {
        "folds": FOLDS_PATH,
        "laws_sparse_postings": LAWS_SPARSE_POSTINGS_PATH,
        "laws_bm25": V44_LAWS_BM25_PATH,
        "query_views": V44_QUERY_VIEWS_PATH,
        "case_profiles": V44_CASE_PROFILES_PATH,
        "case_dense": V44_CASE_DENSE_PATH,
        "case_dense_index": V44_CASE_DENSE_INDEX_PATH,
        "case_dense_meta": V44_CASE_DENSE_META_PATH,
        "case_sparse": V44_CASE_SPARSE_PATH,
        "case_bm25": V44_CASE_BM25_PATH,
        "court_faiss": COURT_FAISS_PATH,
        "court_faiss_meta": v43.COURT_FAISS_META_PATH,
    }
    if cfg.official_enrichment_path.exists():
        paths["official_enrichment"] = cfg.official_enrichment_path
    if include_bundle:
        paths["model_bundle"] = V44_MODEL_BUNDLE_PATH
    return paths


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_kaggle_manifest(config: Optional[V44Config] = None, include_bundle: bool = True, include_hashes: bool = False) -> Dict[str, object]:
    cfg = config or V44Config()
    entries = {}
    for name, path in collect_v44_asset_paths(cfg, include_bundle=include_bundle).items():
        if not path.exists():
            continue
        entry = {
            "path": str(path),
            "size": int(path.stat().st_size),
        }
        if include_hashes:
            entry["sha256"] = file_sha256(path)
        entries[name] = entry
    manifest = {
        "pipeline_version": V44_PIPELINE_VERSION,
        "bundle_included": bool(include_bundle and V44_MODEL_BUNDLE_PATH.exists()),
        "query_view_policy": "deterministic_default",
        "network_required": False,
        "entries": entries,
    }
    V44_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    V44_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def validate_kaggle_manifest(require_bundle: bool = False) -> Dict[str, object]:
    if not V44_MANIFEST_PATH.exists():
        raise RuntimeError(
            f"Missing Kaggle asset manifest: {V44_MANIFEST_PATH}. "
            "Run `python scripts/package_submission_assets.py --require-bundle` after `python lira_pipeline.py cv`."
        )
    manifest = json.loads(V44_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("pipeline_version") != V44_PIPELINE_VERSION:
        raise RuntimeError(
            f"Manifest version mismatch: {manifest.get('pipeline_version')} != {V44_PIPELINE_VERSION}. "
            "Regenerate the V44 Kaggle assets after rebuilding the bundle."
        )
    entries = manifest.get("entries", {})
    required = set(collect_v44_asset_paths(include_bundle=require_bundle).keys())
    missing = [name for name in required if name not in entries]
    if missing:
        raise RuntimeError(
            f"Manifest is missing required assets {sorted(missing)}. "
            "Regenerate the V44 Kaggle assets manifest before running predict."
        )
    for name in required:
        entry = entries[name]
        path = Path(entry["path"])
        if not path.exists():
            raise RuntimeError(f"Manifest-listed asset is missing on disk: {path}")
        expected_size = int(entry.get("size", -1))
        if expected_size >= 0 and path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Manifest-listed asset size mismatch for {path}. "
                "Regenerate the Kaggle assets manifest after rebuilding artifacts."
            )
    return manifest


def ensure_v43_base_artifacts(config: V44Config) -> None:
    v43_config = to_v43_config(config)
    if not config.court_dense_path.exists():
        raise RuntimeError(f"Missing court embedding matrix: {config.court_dense_path}. Prepare it with `python scripts/setup/encode_court_corpus.py`.")
    if not config.court_meta_path.exists():
        raise RuntimeError(f"Missing court metadata: {config.court_meta_path}. Prepare it with `python lira_retrieval.py prepare`.")
    if not FOLDS_PATH.exists():
        train_df = pd.read_csv(DATA_PATH / "train.csv").dropna(subset=["query", "gold_citations"]).reset_index(drop=True)
        save_pickle(FOLDS_PATH, build_folds(train_df, v43_config))
    else:
        log(f"   Folds: {FOLDS_PATH.name}")
    v43.ensure_laws_sparse_postings(v43_config)
    if not COURT_FAISS_PATH.exists() or not v43.COURT_FAISS_META_PATH.exists():
        log("   Building shared court FAISS dependency via V43...")
        v43.build_faiss_index(v43_config)
    ensure_required_artifact(COURT_FAISS_PATH, "shared court FAISS index", "python lira_retrieval.py prepare")
    ensure_required_artifact(v43.COURT_FAISS_META_PATH, "shared court FAISS metadata", "python lira_retrieval.py prepare")


def make_query_cache_key(query_id: str, query: str) -> str:
    return f"{query_id}|{hash(query)}"


def reset_runtime_diagnostics(resources: V44Resources) -> None:
    resources.runtime_diagnostics = defaultdict(list)


def record_runtime_metric(resources: V44Resources, name: str, value: float) -> None:
    resources.runtime_diagnostics[name].append(float(value))


def summarize_runtime_diagnostics(resources: V44Resources) -> Dict[str, float]:
    return {
        key: float(np.mean(values))
        for key, values in resources.runtime_diagnostics.items()
        if values
    }


def resolve_reranker_device() -> str:
    override = os.getenv("LIRA_V44_RERANKER_DEVICE", "").strip().lower()
    if override in {"cpu", "cuda"}:
        if override == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_system_memory_gb() -> float:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return float(stat.ullTotalPhys) / (1024 ** 3)
    if hasattr(os, "sysconf"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return float(page_size * page_count) / (1024 ** 3)
    return 0.0


def has_high_memory_profile() -> bool:
    override = os.getenv("LIRA_V44_HIGH_MEMORY", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    try:
        return get_system_memory_gb() >= 28.0
    except Exception:
        return False


def resolve_reranker_batch_size(device: str, config: V44Config) -> int:
    override = os.getenv("LIRA_V44_RERANKER_BATCH_SIZE", "").strip()
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass
    if int(config.reranker_batch_size) > 0:
        return int(config.reranker_batch_size)
    if device == "cuda":
        return 12 if has_high_memory_profile() else 8
    return 4


def default_case_evidence_params(config: V44Config) -> Dict[str, float]:
    return {
        "rrf_weight": 0.20,
        "reranker_weight": 0.50,
        "late_weight": 0.10,
        "support_weight": 0.05,
        "direct_weight": 0.25,
        "regex_bonus": 0.10,
        "exact_case_bonus": 0.20,
        "base_case_penalty": 0.50,
        "temperature": 1.0,
        "case_rerank_threshold": 0.10,
        "case_floor": config.case_evidence_floor,
    }


_LOCAL_QWEN = {"tokenizer": None, "model": None, "failed": False}


def maybe_load_local_qwen() -> Tuple[Optional[object], Optional[object]]:
    if _LOCAL_QWEN["failed"]:
        return None, None
    if _LOCAL_QWEN["tokenizer"] is not None and _LOCAL_QWEN["model"] is not None:
        return _LOCAL_QWEN["tokenizer"], _LOCAL_QWEN["model"]
    if os.environ.get("LIRA_V44_ENABLE_LOCAL_QWEN", "0") != "1":
        _LOCAL_QWEN["failed"] = True
        return None, None
    if AutoTokenizer is None or AutoModelForCausalLM is None or not QWEN_LOCAL_PATH.exists():
        _LOCAL_QWEN["failed"] = True
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(QWEN_LOCAL_PATH), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(QWEN_LOCAL_PATH), trust_remote_code=True)
        if torch.cuda.is_available():
            model = model.half().cuda()
        model.eval()
        _LOCAL_QWEN["tokenizer"] = tokenizer
        _LOCAL_QWEN["model"] = model
        return tokenizer, model
    except Exception:
        _LOCAL_QWEN["failed"] = True
        return None, None


def maybe_qwen_rewrite(query: str, mode: str) -> Optional[str]:
    tokenizer, model = maybe_load_local_qwen()
    if tokenizer is None or model is None:
        return None
    prompt = (
        "Rewrite the legal query for retrieval. Respond with one paragraph only. "
        f"Mode: {mode}. Keep all legal issues and important codes.\nQuery:\n{query}\nRewrite:"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False, temperature=0.0)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if "Rewrite:" in text:
            text = text.split("Rewrite:", 1)[-1].strip()
        text = WS_RE.sub(" ", text).strip()
        return text or None
    except Exception:
        return None


def deterministic_query_views(query: str) -> QueryViews:
    query = WS_RE.sub(" ", str(query).strip())
    codes = sorted(v43_detect_domain_codes(query))
    regex_hits = [WS_RE.sub(" ", hit.strip()) for hit in extract_citations_regex(query)[:8]]
    top_terms = get_top_terms(query, keep=12)
    issue_view = (
        f"Issues: {'; '.join(codes) if codes else 'general Swiss law'}. "
        f"Citations: {'; '.join(regex_hits) if regex_hits else 'none explicit'}. "
        f"Key terms: {' '.join(top_terms[:10])}. "
        f"Query: {truncate_words(query, 140)}"
    )
    precedent_view = (
        f"Precedent retrieval facts: {truncate_words(query, 180)}. "
        f"Fact anchors: {' '.join(top_terms)}. "
        f"Codes: {' '.join(codes) if codes else 'none'}"
    )
    q_issue = maybe_qwen_rewrite(query, "issue") or issue_view
    q_prec = maybe_qwen_rewrite(query, "precedent") or precedent_view
    source = "local_qwen" if q_issue != issue_view or q_prec != precedent_view else "fallback"
    return QueryViews(original=query, issue_view=q_issue, precedent_view=q_prec, source=source)


def build_query_views_cache(config: V44Config) -> Dict[str, QueryViews]:
    cached: Dict[str, QueryViews] = {}
    if V44_QUERY_VIEWS_PATH.exists():
        raw = load_pickle(V44_QUERY_VIEWS_PATH)
        for query_id, payload in raw.items():
            cached[str(query_id)] = QueryViews(**payload)
    merged: Dict[str, str] = {}
    for csv_name in ("train.csv", "val.csv", "test.csv"):
        df = pd.read_csv(DATA_PATH / csv_name)
        for _, row in df.iterrows():
            merged[str(row["query_id"])] = str(row["query"])
    updated = False
    for query_id, query in merged.items():
        current = cached.get(query_id)
        if current is None or current.original != query:
            cached[query_id] = deterministic_query_views(query)
            updated = True
    if updated or not V44_QUERY_VIEWS_PATH.exists():
        save_pickle(V44_QUERY_VIEWS_PATH, {query_id: view.as_dict() for query_id, view in cached.items()})
        log(f"   Query views: {V44_QUERY_VIEWS_PATH.name}")
    else:
        log(f"   Query views: {V44_QUERY_VIEWS_PATH.name}")
    return cached


WS_RE = v43.WS_RE


def iter_case_profile_feature_rows(
    config: V44Config,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, float]], Dict[str, CaseProfile]]:
    if not (DATA_PATH / "court_considerations.csv").exists():
        raise RuntimeError("Missing data/court_considerations.csv")
    court_meta = load_pickle(config.court_meta_path)
    case_ids = list(court_meta["case_ids"])
    case_citations = list(court_meta["case_citations"])

    entries: Dict[str, Dict[str, object]] = {}
    csv_path = DATA_PATH / "court_considerations.csv"
    row_ptr = 0
    for chunk in pd.read_csv(csv_path, chunksize=config.case_profile_chunk_rows, usecols=["citation", "text"]):
        mask = chunk["text"].notna() & (chunk["text"].astype(str).str.len() >= 50)
        filtered = chunk.loc[mask]
        if filtered.empty:
            continue
        for raw_case_id, text in zip(filtered["citation"].astype(str).tolist(), filtered["text"].astype(str).tolist()):
            if row_ptr >= len(case_ids):
                raise RuntimeError("court_considerations.csv has more filtered rows than court_meta.pkl")
            expected_case_id = WS_RE.sub(" ", str(case_ids[row_ptr]).strip())
            raw_case_id = WS_RE.sub(" ", raw_case_id.strip())
            if raw_case_id != expected_case_id:
                raise RuntimeError(f"Court profile alignment mismatch at row {row_ptr}: {raw_case_id!r} != {expected_case_id!r}")
            base_case_id = get_case_group_id(raw_case_id)
            target = entries.setdefault(
                base_case_id,
                {
                    "exact_case_ids": Counter(),
                    "cited_statutes": Counter(),
                    "cited_cases": Counter(),
                    "snippets": [],
                    "total_considerations": 0,
                },
            )
            target["exact_case_ids"][normalize_case_citation(raw_case_id)] += 1
            target["total_considerations"] += 1
            if len(target["snippets"]) < config.case_profile_snippet_count:
                target["snippets"].append(truncate_words(text, config.case_profile_snippet_tokens))
            for citation in case_citations[row_ptr]:
                normalized_case = normalize_case_citation(str(citation))
                ctype = get_citation_type(normalized_case)
                if ctype == "statute":
                    target["cited_statutes"][normalize_citation(str(citation))] += 1
                elif ctype in {"bge", "docket"}:
                    target["cited_cases"][normalized_case] += 1
            row_ptr += 1
    if row_ptr != len(case_ids):
        raise RuntimeError(f"Court profile builder consumed {row_ptr:,} rows, expected {len(case_ids):,}")

    profile_ids = sorted(entries.keys())
    profile_texts: List[str] = []
    sparse_weights_seed: Dict[str, Dict[str, float]] = {}
    profiles: Dict[str, CaseProfile] = {}
    for base_case_id in profile_ids:
        payload = entries[base_case_id]
        exact_case_ids = [citation for citation, _ in payload["exact_case_ids"].most_common(config.case_profile_exact_limit)]
        cited_statutes = [(citation, int(count)) for citation, count in payload["cited_statutes"].most_common(config.case_profile_statute_limit)]
        cited_cases = [(citation, int(count)) for citation, count in payload["cited_cases"].most_common(config.case_profile_case_limit)]
        snippets = list(payload["snippets"])
        profile_text = compose_case_profile_text(
            base_case_id=base_case_id,
            exact_case_ids=exact_case_ids[: config.case_profile_exact_limit],
            cited_statutes=cited_statutes,
            cited_cases=cited_cases,
            snippets=snippets,
        )
        profiles[base_case_id] = CaseProfile(
            base_case_id=base_case_id,
            exact_case_ids=exact_case_ids,
            cited_statutes=cited_statutes,
            cited_cases=cited_cases,
            snippets=snippets,
            profile_text=profile_text,
            total_considerations=int(payload["total_considerations"]),
        )
        profile_texts.append(profile_text)
        sparse_weights_seed[base_case_id] = {}
    return profile_ids, profile_texts, sparse_weights_seed, profiles


def build_case_profiles(config: V44Config) -> Dict[str, CaseProfile]:
    if V44_CASE_PROFILES_PATH.exists():
        raw = load_pickle(V44_CASE_PROFILES_PATH)
        profiles: Dict[str, CaseProfile] = {}
        for case_id, payload in raw.items():
            profile_payload = dict(payload)
            profile_payload.setdefault(
                "profile_text",
                compose_case_profile_text(
                    base_case_id=str(case_id),
                    exact_case_ids=profile_payload.get("exact_case_ids", []),
                    cited_statutes=profile_payload.get("cited_statutes", []),
                    cited_cases=profile_payload.get("cited_cases", []),
                    snippets=profile_payload.get("snippets", []),
                ),
            )
            profiles[str(case_id)] = CaseProfile(**profile_payload)
        return profiles
    log("   Building case profiles...")
    profile_ids, _, _, profiles = iter_case_profile_feature_rows(config)
    compact_profiles = {}
    for case_id in profile_ids:
        payload = asdict(profiles[case_id])
        payload.pop("profile_text", None)
        compact_profiles[case_id] = payload
    save_pickle(V44_CASE_PROFILES_PATH, compact_profiles)
    log(f"   Saved: {V44_CASE_PROFILES_PATH}")
    return profiles


def try_load_pickle(path: Path) -> Optional[object]:
    try:
        return load_pickle(path)
    except (EOFError, pickle.UnpicklingError, AttributeError, ValueError, OSError):
        return None


def build_case_sparse_postings(
    encoder: BGEM3FlagModel,
    case_ids: Sequence[str],
    texts: Sequence[str],
    config: V44Config,
) -> SparsePostingIndex:
    postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for start in maybe_tqdm(range(0, len(case_ids), config.case_profile_batch_size), desc="Case profile sparse encodings"):
        end = min(start + config.case_profile_batch_size, len(case_ids))
        batch_texts = texts[start:end]
        output = encode_with_optional_quiet(
            encoder,
            batch_texts,
            batch_size=min(config.case_profile_batch_size, len(batch_texts)),
            max_length=1024,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        for offset, weights in enumerate(output["lexical_weights"]):
            kept = prune_weights({str(k): float(v) for k, v in weights.items()}, 128)
            for token, weight in kept.items():
                postings[str(token)].append((start + offset, float(weight)))
        if start and start % 512 == 0:
            gc.collect()
    return SparsePostingIndex(postings=dict(postings), doc_ids=list(case_ids))

def build_case_profile_artifacts(config: V44Config, encoder: BGEM3FlagModel) -> None:
    profiles = build_case_profiles(config)
    case_ids = sorted(profiles.keys())
    texts = [profiles[case_id].profile_text for case_id in case_ids]

    sparse_payload = try_load_pickle(V44_CASE_SPARSE_PATH) if V44_CASE_SPARSE_PATH.exists() else None
    sparse_ready = isinstance(sparse_payload, SparsePostingIndex) or (
        isinstance(sparse_payload, dict) and "index" in sparse_payload
    )

    if not V44_CASE_DENSE_PATH.exists():
        log("   Encoding case profiles (dense)...")
        sample = encode_with_optional_quiet(
            encoder,
            [texts[0] if texts else "empty"],
            batch_size=1,
            max_length=1024,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dim = int(np.asarray(sample["dense_vecs"][0]).shape[-1])
        dense_mem = np.lib.format.open_memmap(str(V44_CASE_DENSE_PATH), mode="w+", dtype=np.float16, shape=(len(case_ids), dim))
        for start in maybe_tqdm(range(0, len(case_ids), config.case_profile_batch_size), desc="Case profile dense encodings"):
            end = min(start + config.case_profile_batch_size, len(case_ids))
            batch_texts = texts[start:end]
            output = encode_with_optional_quiet(
                encoder,
                batch_texts,
                batch_size=min(config.case_profile_batch_size, len(batch_texts)),
                max_length=1024,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense_mem[start:end] = np.asarray(output["dense_vecs"], dtype=np.float16)
            if start and start % 512 == 0:
                dense_mem.flush()
                gc.collect()
        dense_mem.flush()
        del dense_mem
        log(f"   Saved: {V44_CASE_DENSE_PATH.name}")
    else:
        log(f"   Case dense embeddings: {V44_CASE_DENSE_PATH.name}")

    if not sparse_ready:
        if V44_CASE_SPARSE_PATH.exists():
            log("   Rebuilding corrupted case sparse postings...")
            try:
                V44_CASE_SPARSE_PATH.unlink()
            except OSError:
                pass
        else:
            log("   Building case sparse postings...")
        sparse_index = build_case_sparse_postings(encoder, case_ids, texts, config)
        save_pickle(V44_CASE_SPARSE_PATH, {"index": sparse_index})
        log(f"   Saved: {V44_CASE_SPARSE_PATH.name}")
    else:
        log(f"   Case sparse postings: {V44_CASE_SPARSE_PATH.name}")

    if not V44_CASE_DENSE_INDEX_PATH.exists():
        log("   Building case-profile FAISS index...")
        dense = np.load(V44_CASE_DENSE_PATH, mmap_mode="r")
        total, dim = dense.shape
        rng = np.random.default_rng(config.random_seed)
        sample_size = min(config.case_profile_faiss_train_size, total)
        sample_idx = rng.choice(total, size=sample_size, replace=False)
        train_vecs = np.asarray(dense[sample_idx], dtype=np.float32)
        index = faiss.index_factory(dim, config.case_profile_faiss_factory, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            log(f"   Training case-profile FAISS on {sample_size:,} samples...")
            index.train(train_vecs)
        for start in maybe_tqdm(range(0, total, config.case_profile_faiss_add_batch), desc="Adding case profiles"):
            end = min(start + config.case_profile_faiss_add_batch, total)
            index.add(np.asarray(dense[start:end], dtype=np.float32))
        if hasattr(index, "nprobe"):
            index.nprobe = config.case_profile_faiss_nprobe
        faiss.write_index(index, str(V44_CASE_DENSE_INDEX_PATH))
        with V44_CASE_DENSE_META_PATH.open("w", encoding="utf-8") as handle:
            json.dump({"rows": int(total), "dim": int(dim), "factory": config.case_profile_faiss_factory}, handle, ensure_ascii=False, indent=2)
        log(f"   Saved: {V44_CASE_DENSE_INDEX_PATH}")
    else:
        log(f"   Case profile FAISS index: {V44_CASE_DENSE_INDEX_PATH.name}")

    if not V44_CASE_BM25_PATH.exists():
        log("   Building case-profile BM25...")
        bm25 = build_bm25_from_docs(case_ids, texts)
        save_pickle(V44_CASE_BM25_PATH, bm25)
        log(f"   Saved: {V44_CASE_BM25_PATH}")
    else:
        log(f"   Case profile BM25: {V44_CASE_BM25_PATH.name}")


def build_laws_bm25(config: V44Config) -> None:
    if V44_LAWS_BM25_PATH.exists():
        log(f"   Laws BM25: {V44_LAWS_BM25_PATH.name}")
        return
    laws_df = pd.read_csv(DATA_PATH / "laws_de.csv")
    doc_ids = laws_df["citation"].astype(str).tolist()
    texts = [f"{str(title)}\n{str(text)}" for title, text in zip(laws_df.get("title", "").astype(str), laws_df["text"].astype(str))]
    log("   Building laws BM25...")
    bm25 = build_bm25_from_docs(doc_ids, texts)
    save_pickle(V44_LAWS_BM25_PATH, bm25)
    log(f"   Saved: {V44_LAWS_BM25_PATH}")


def load_case_profile_bm25(
    case_profile_ids: Sequence[str],
    case_profiles: Dict[str, CaseProfile],
) -> EfficientBM25:
    try:
        return load_pickle(V44_CASE_BM25_PATH)
    except (MemoryError, EOFError, OSError, pickle.UnpicklingError):
        log("   Rebuilding case-profile BM25 in memory...")
        try:
            texts = [case_profiles[case_id].profile_text for case_id in case_profile_ids]
            return build_bm25_from_docs(case_profile_ids, texts)
        except MemoryError:
            log("   Disabling case-profile BM25 due to memory pressure.")
            return EfficientBM25()


def maybe_load_reranker(config: V44Config) -> Tuple[Optional[object], Optional[object], str, int]:
    device = resolve_reranker_device()
    batch_size = resolve_reranker_batch_size(device, config)
    if os.getenv("LIRA_V44_DISABLE_RERANKER", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return None, None, device, batch_size
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None, None, device, batch_size
    if not RERANKER_PATH.exists():
        return None, None, device, batch_size
    try:
        if device == "cpu":
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        else:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        tokenizer = AutoTokenizer.from_pretrained(str(RERANKER_PATH), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(RERANKER_PATH), local_files_only=True)
        model.eval()
        if device == "cuda":
            model = model.half().cuda()
        return tokenizer, model, device, batch_size
    except Exception:
        return None, None, device, batch_size


def rerank_pairs(tokenizer: Optional[object], model: Optional[object], queries: Sequence[str], passages: Sequence[str], batch_size: int = 16) -> List[float]:
    if tokenizer is None or model is None or not passages:
        return [0.0 for _ in passages]
    outputs: List[float] = []
    model_device = torch.device("cpu")
    if hasattr(model, "parameters"):
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
    is_cpu = not str(model_device).startswith("cuda")
    safe_batch_size = 1 if is_cpu and os.name == "nt" else batch_size
    max_length = 512 if is_cpu and os.name == "nt" else 1024
    for start in range(0, len(passages), safe_batch_size):
        end = min(start + safe_batch_size, len(passages))
        q_batch = list(queries[start:end])
        p_batch = list(passages[start:end])
        inputs = tokenizer(q_batch, p_batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        if str(model_device).startswith("cuda"):
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits.squeeze(-1)
        outputs.extend([float(x) for x in logits.detach().cpu().numpy().tolist()])
    return outputs


def require_live_reranker(resources: V44Resources, context: str) -> None:
    if resources.require_reranker and (resources.reranker_tokenizer is None or resources.reranker_model is None):
        raise RuntimeError(
            f"V44 reranker is required for {context}, but it is unavailable. "
            "Check the local reranker files and device configuration."
        )


def sparse_overlap_score(query_sparse: Dict[str, float], doc_sparse: Dict[str, float]) -> float:
    if not query_sparse or not doc_sparse:
        return 0.0
    if len(query_sparse) > len(doc_sparse):
        query_sparse, doc_sparse = doc_sparse, query_sparse
    score = 0.0
    q_norm = 0.0
    d_norm = 0.0
    for token, weight in query_sparse.items():
        weight = float(weight)
        q_norm += weight * weight
        score += weight * float(doc_sparse.get(token, 0.0))
    for value in doc_sparse.values():
        value = float(value)
        d_norm += value * value
    if q_norm <= 0 or d_norm <= 0:
        return 0.0
    return float(score / math.sqrt(q_norm * d_norm))


def build_late_interaction_proxy(best_dense: float, best_sparse: float) -> float:
    dense_term = max(0.0, float(best_dense))
    sparse_term = max(0.0, float(best_sparse))
    return float(0.6 * dense_term + 0.4 * sparse_term)


def prepare_artifacts(config: V44Config) -> None:
    V44_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    V44_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    ensure_v43_base_artifacts(config)
    build_laws_bm25(config)
    build_query_views_cache(config)
    encoder = load_encoder(config)
    build_case_profile_artifacts(config, encoder)
    write_kaggle_manifest(config, include_bundle=False, include_hashes=False)


def load_resources(config: V44Config, require_reranker: bool = False) -> V44Resources:
    log("Loading V44 resources...")
    ensure_required_artifact(COURT_FAISS_PATH, "shared court FAISS index", "python lira_pipeline.py prepare")
    ensure_required_artifact(v43.COURT_FAISS_META_PATH, "shared court FAISS metadata", "python lira_pipeline.py prepare")
    train_df = pd.read_csv(DATA_PATH / "train.csv").dropna(subset=["query", "gold_citations"]).reset_index(drop=True)
    val_df = pd.read_csv(DATA_PATH / "val.csv")
    test_df = pd.read_csv(DATA_PATH / "test.csv")
    laws_df = pd.read_csv(DATA_PATH / "laws_de.csv")
    encoder = load_encoder(config)

    laws_sparse_meta = load_pickle(config.laws_sparse_path)
    laws_sparse_index = load_pickle(LAWS_SPARSE_POSTINGS_PATH)
    laws_embeddings = np.load(config.laws_dense_path, mmap_mode="r")
    laws_citations = [str(c) for c in laws_sparse_index.doc_ids]
    laws_citation_to_idx = {citation: idx for idx, citation in enumerate(laws_citations)}
    laws_text: Dict[str, str] = {}
    laws_title: Dict[str, str] = {}
    for law in laws_df.itertuples(index=False):
        citation = WS_RE.sub(" ", str(law.citation).strip())
        laws_text[citation] = str(law.text) if pd.notna(law.text) else ""
        laws_title[citation] = str(law.title) if pd.notna(law.title) else ""
    for citation in laws_citations:
        parent = parent_statute_citation(citation)
        if parent and parent not in laws_text:
            normalized = WS_RE.sub(" ", citation.strip())
            laws_text[parent] = laws_text.get(normalized, "")
            laws_title[parent] = laws_title.get(normalized, "")
    laws_code = {citation: get_citation_code(citation) or "" for citation in laws_citations}
    laws_bm25 = load_pickle(V44_LAWS_BM25_PATH)

    court_meta = load_pickle(config.court_meta_path)
    court_case_ids = list(court_meta["case_ids"])
    court_case_citations = list(court_meta["case_citations"])
    court_embeddings = np.load(config.court_dense_path, mmap_mode="r")
    court_faiss = faiss.read_index(str(COURT_FAISS_PATH))
    if hasattr(court_faiss, "nprobe"):
        court_faiss.nprobe = 48

    profiles_raw = load_pickle(V44_CASE_PROFILES_PATH)
    case_profiles = {}
    for case_id, payload in profiles_raw.items():
        profile_payload = dict(payload)
        profile_payload.setdefault(
            "profile_text",
            compose_case_profile_text(
                base_case_id=str(case_id),
                exact_case_ids=profile_payload.get("exact_case_ids", []),
                cited_statutes=profile_payload.get("cited_statutes", []),
                cited_cases=profile_payload.get("cited_cases", []),
                snippets=profile_payload.get("snippets", []),
            ),
        )
        case_profiles[str(case_id)] = CaseProfile(**profile_payload)
    case_profile_ids = sorted(case_profiles.keys())
    case_profile_dense = np.load(V44_CASE_DENSE_PATH, mmap_mode="r")
    case_profile_faiss = faiss.read_index(str(V44_CASE_DENSE_INDEX_PATH))
    if hasattr(case_profile_faiss, "nprobe"):
        case_profile_faiss.nprobe = config.case_profile_faiss_nprobe
    case_sparse_payload = try_load_pickle(V44_CASE_SPARSE_PATH)
    if case_sparse_payload is None:
        raise RuntimeError(f"Corrupted case sparse artifact: {V44_CASE_SPARSE_PATH}")
    if isinstance(case_sparse_payload, dict):
        case_profile_sparse_index = case_sparse_payload["index"]
    else:
        case_profile_sparse_index = case_sparse_payload
    case_profile_bm25 = load_case_profile_bm25(case_profile_ids, case_profiles)

    case_extracted_cits = build_case_extracted_cits_from_meta(court_case_ids, court_case_citations)
    citation_universe = build_citation_universe(
        laws_df=laws_df,
        court_case_ids=court_case_ids,
        court_case_citations=court_case_citations,
        case_extracted_cits=case_extracted_cits,
    )
    citation_doc_freq: Counter = Counter()
    for case_id, citations in case_extracted_cits.items():
        base_case_id = get_case_group_id(str(case_id))
        citation_doc_freq[base_case_id] += 1
        for citation in citations:
            normalized_case = normalize_case_citation(str(citation))
            if get_citation_type(normalized_case) in {"bge", "docket"}:
                citation_doc_freq[normalized_case] += 1
                citation_doc_freq[get_case_group_id(normalized_case)] += 1
            else:
                citation_doc_freq[normalize_citation(str(citation))] += 1
    total_cases = len(case_extracted_cits)
    case_extracted_cits = {}
    folds = load_pickle(FOLDS_PATH)
    query_views_raw = load_pickle(V44_QUERY_VIEWS_PATH)
    query_views = {query_id: QueryViews(**payload) for query_id, payload in query_views_raw.items()}
    official_enrichment = load_official_enrichment(config)
    reranker_tokenizer, reranker_model, reranker_device, reranker_batch_size = maybe_load_reranker(config)
    if require_reranker and (reranker_tokenizer is None or reranker_model is None):
        disable_flag = os.getenv("LIRA_V44_DISABLE_RERANKER", "0").strip().lower() in {"1", "true", "yes", "on"}
        extra = " Unset `LIRA_V44_DISABLE_RERANKER` in this shell first." if disable_flag else ""
        raise RuntimeError(
            "V44 quality commands require the local reranker, but it could not be loaded. "
            f"Expected model files under {RERANKER_PATH}. "
            "Set `LIRA_V44_RERANKER_DEVICE=cuda` or `cpu` if you need to override device selection."
            f"{extra}"
        )
    if reranker_tokenizer is not None and reranker_model is not None:
        log(f"   Reranker: device={reranker_device} batch_size={reranker_batch_size}")
    elif not require_reranker:
        log("   Reranker: disabled for prepare mode")
    co_citation_index = load_pickle(config.co_citation_path) if config.co_citation_path.exists() else {"art_art": {}, "bge_art": {}}

    return V44Resources(
        config=config,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        laws_df=laws_df,
        encoder=encoder,
        citation_universe=citation_universe,
        laws_embeddings=laws_embeddings,
        laws_citations=laws_citations,
        laws_text=laws_text,
        laws_title=laws_title,
        laws_code=laws_code,
        laws_sparse_index=laws_sparse_index,
        laws_sparse_meta=laws_sparse_meta,
        laws_bm25=laws_bm25,
        laws_citation_to_idx=laws_citation_to_idx,
        case_profiles=case_profiles,
        case_profile_ids=case_profile_ids,
        case_profile_dense=case_profile_dense,
        case_profile_faiss=case_profile_faiss,
        case_profile_sparse_index=case_profile_sparse_index,
        case_profile_bm25=case_profile_bm25,
        court_embeddings=court_embeddings,
        court_case_ids=court_case_ids,
        court_case_citations=court_case_citations,
        court_faiss=court_faiss,
        case_extracted_cits=case_extracted_cits,
        co_citation_index=co_citation_index,
        official_enrichment=official_enrichment,
        citation_doc_freq=citation_doc_freq,
        total_cases=total_cases,
        folds=folds,
        query_views=query_views,
        reranker_tokenizer=reranker_tokenizer,
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        reranker_batch_size=reranker_batch_size,
        require_reranker=require_reranker,
    )


def get_query_views(resources: V44Resources, query_id: str, query: str) -> QueryViews:
    return resources.query_views.get(str(query_id), deterministic_query_views(query))


def encode_query_views_for_query(resources: V44Resources, query_id: str, query: str) -> Dict[str, Dict[str, object]]:
    query_views = get_query_views(resources, query_id, query)
    encoded: Dict[str, Dict[str, object]] = {}
    for view_name in QUERY_VIEW_NAMES:
        view_text = getattr(query_views, view_name)
        dense, sparse = encode_sparse_query(resources.encoder, view_text)
        encoded[view_name] = {"text": view_text, "dense": dense, "sparse": sparse}
    return encoded


def faiss_search(index: faiss.Index, embedding_matrix: np.ndarray, query_embedding: np.ndarray, search_k: int, final_k: int) -> List[Tuple[int, float]]:
    scores, indices = index.search(query_embedding[np.newaxis, :], search_k)
    candidate_idx = indices[0]
    candidate_idx = candidate_idx[candidate_idx >= 0]
    if candidate_idx.size == 0:
        return []
    exact_vectors = np.asarray(embedding_matrix[candidate_idx], dtype=np.float32)
    exact_scores = exact_vectors @ query_embedding
    order = np.argsort(exact_scores)[::-1][:final_k]
    return [(int(candidate_idx[i]), float(exact_scores[i])) for i in order]


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the highest scoring indices without sorting the full corpus."""
    if k <= 0 or scores.size == 0:
        return np.empty(0, dtype=np.int64)
    limit = min(int(k), int(scores.size))
    shortlist = np.argpartition(-scores, limit - 1)[:limit]
    return shortlist[np.argsort(-scores[shortlist], kind="stable")]


def parse_article_number(citation: str) -> Optional[int]:
    match = re.search(r"^Art\. (\d+)", str(citation))
    return int(match.group(1)) if match else None


def compute_neighbor_support(citations: Sequence[str]) -> Counter:
    by_code: Dict[str, set[int]] = defaultdict(set)
    for citation in citations:
        code = get_citation_code(str(citation))
        art_num = parse_article_number(str(citation))
        if code and art_num is not None:
            by_code[code].add(art_num)
    support = Counter()
    for citation in citations:
        code = get_citation_code(str(citation))
        art_num = parse_article_number(str(citation))
        if not code or art_num is None:
            continue
        hits = int((art_num - 1 in by_code[code]) or (art_num + 1 in by_code[code]))
        if hits:
            support[str(citation)] += hits
    return support


def compute_official_neighbor_support(citations: Sequence[str], official_enrichment: Dict[str, object]) -> Counter:
    support = Counter()
    article_neighbors = official_enrichment.get("article_neighbors", {}) if isinstance(official_enrichment, dict) else {}
    citation_set = {str(citation) for citation in citations}
    if not article_neighbors or not citation_set:
        return support
    for citation in citation_set:
        neighbors = article_neighbors.get(citation, [])
        if not isinstance(neighbors, (list, tuple, set)):
            continue
        hits = sum(1 for neighbor in neighbors if str(neighbor) in citation_set)
        if hits:
            support[citation] += float(hits)
    return support


def get_statute_view_weight(view_name: str) -> float:
    if view_name == "issue_view":
        return 1.30
    if view_name == "precedent_view":
        return 0.60
    return 1.00


def official_statute_alias_hit(query: str, citation: str, official_enrichment: Dict[str, object]) -> int:
    aliases = {}
    if isinstance(official_enrichment, dict):
        aliases = official_enrichment.get("statute_aliases", {}) or {}
    alias_values = aliases.get(str(citation), [])
    if not alias_values:
        return 0
    lowered_query = str(query).lower()
    for alias in alias_values:
        alias_text = str(alias).strip().lower()
        if alias_text and alias_text in lowered_query:
            return 1
    return 0


def get_official_case_statute_targets(resources: V44Resources, citations: Sequence[str], limit: int = 10) -> List[str]:
    graph = resources.official_enrichment.get("case_statute_graph", {}) if isinstance(resources.official_enrichment, dict) else {}
    targets: List[str] = []
    seen = set()
    for citation in citations:
        payload = graph.get(str(citation), {})
        for article in get_top_co_citation_targets(payload, limit=limit):
            if article not in seen:
                seen.add(article)
                targets.append(article)
    return targets[:limit]

def get_top_co_citation_targets(payload: object, limit: int = 10) -> List[str]:
    if isinstance(payload, dict):
        ranked = sorted(payload.items(), key=lambda kv: (float(kv[1]), str(kv[0])), reverse=True)
        return [str(citation) for citation, _ in ranked[:limit]]
    if isinstance(payload, (list, tuple)):
        return [str(citation) for citation in payload[:limit]]
    if isinstance(payload, set):
        return [str(citation) for citation in list(payload)[:limit]]
    if payload:
        return [str(payload)]
    return []


