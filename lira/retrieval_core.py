#!/usr/bin/env python3
import argparse  # noqa: F401 - re-exported to retrieval stages
import contextlib
import gc
import io
import json
import math
import os
import pickle
import random  # noqa: F401 - re-exported to retrieval stages
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import lightgbm as lgb  # noqa: F401 - re-exported to retrieval stages
import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel
from sklearn.linear_model import LogisticRegression  # noqa: F401 - re-exported to retrieval stages
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

DATA_PATH = Path("./data")
CACHE_PATH = Path("./cache")
OUTPUT_PATH = Path("./output")
V43_CACHE_PATH = CACHE_PATH / "v43_candidate_tables"
V43_MODEL_PATH = CACHE_PATH / "v43_models"
V43_PIPELINE_VERSION = "v432"

COURT_FAISS_PATH = CACHE_PATH / "faiss_court_v43.index"
COURT_FAISS_META_PATH = CACHE_PATH / "faiss_court_v43.meta.json"
FOLDS_PATH = CACHE_PATH / "folds_v43.pkl"
LAWS_SPARSE_POSTINGS_PATH = CACHE_PATH / "v43_laws_sparse_postings.pkl"
COURT_SPARSE_POSTINGS_PATH = CACHE_PATH / "v43_court_sparse_postings.pkl"
MODEL_BUNDLE_PATH = V43_MODEL_PATH / "model_bundle.pkl"
OOF_SCORED_PATH = V43_CACHE_PATH / f"oof_scored_{V43_PIPELINE_VERSION}.pkl"

DOCKET_PATTERN = re.compile(r"^(\d+[A-Z]_\d+/\d{4})")
BASE_CASE_SUFFIX_RE = re.compile(r"\s+E\.\s*.+$")
DOCKET_WITH_DATE_RE = re.compile(r"^(\d+[A-Z]_\d+/\d{4})\s+\d{2}\.\d{2}\.\d{4}(\s+E\.\s*.+)$")
WS_RE = re.compile(r"\s+")

CODE_CANON_MAP = {
    "CO": "OR",
    "CC": "ZGB",
    "CP": "StGB",
    "LP": "SchKG",
    "LTF": "BGG",
    "CPC": "ZPO",
    "CPP": "StPO",
    "Cst": "BV",
    "LAI": "IVG",
    "LAA": "UVG",
    "LAVS": "AHVG",
    "LAMal": "KVG",
    "LPP": "BVG",
    "LACI": "AVIG",
    "LEtr": "AIG",
    "OG": "BGG",
    "LOA": "OR",
    "CCS": "ZGB",
    "CPS": "StGB",
    "LDIP": "IPRG",
    "LCA": "VVG",
    "LPD": "DSG",
    "LEI": "AIG",
    "FITAF": "VGKE",
    "Cost": "BV",
    "DTF": "BGE",
    "LIFD": "DBG",
    "LIA": "VStG",
    "LDA": "URG",
    "LTVA": "MWSTG",
}

DOMAIN_KEYWORDS = {
    "StPO": ["detention", "criminal", "accused", "prosecution", "arrest", "trial", "prison", "remand", "offense", "sentence", "Untersuchungshaft", "Strafverfahren", "Beschuldigte"],
    "StGB": ["crime", "penalty", "offense", "conviction", "assault", "Strafe", "Straftat", "Verurteilung", "murder", "theft"],
    "ZGB": ["marriage", "divorce", "child", "custody", "inheritance", "estate", "property", "family", "heir", "guardian", "Ehe", "Scheidung", "Kind", "Erbe", "Eigentum", "possession", "movable", "immovable", "bona fide"],
    "OR": ["contract", "obligation", "damage", "liability", "compensation", "Vertrag", "Haftung", "Schadenersatz", "purchase", "lease"],
    "BGG": ["federal", "court", "appeal", "Bundesgericht", "Beschwerde"],
    "ATSG": ["insurance", "disability", "invalidity", "pension", "Versicherung", "Invalidität", "Rente"],
    "IVG": ["disability", "invalidity", "Invalidenversicherung", "IV-Stelle"],
    "SchKG": ["bankruptcy", "debt", "enforcement", "Betreibung", "Konkurs"],
    "ZPO": ["civil procedure", "Zivilprozess", "mediation", "Klage"],
    "IPRG": ["international", "private international", "foreign", "cross-border", "jurisdiction", "applicable law", "recognition", "enforcement abroad", "Lugano", "convention"],
    "UVG": ["accident", "occupational", "Unfallversicherung", "work injury"],
    "KVG": ["health insurance", "Krankenversicherung", "medical"],
    "DBG": ["tax", "income tax", "federal tax", "Steuer", "direkte Bundessteuer"],
    "VStG": ["withholding tax", "Verrechnungssteuer", "tax evasion"],
}

BOILERPLATE = {
    "Art. 42 Abs. 2 BGG",
    "Art. 66 Abs. 1 BGG",
    "Art. 68 Abs. 2 BGG",
    "Art. 90 BGG",
    "Art. 93 Abs. 1 BGG",
    "Art. 95 BGG",
    "Art. 97 Abs. 1 BGG",
    "Art. 100 Abs. 1 BGG",
    "Art. 105 Abs. 1 BGG",
    "Art. 105 Abs. 2 BGG",
    "Art. 106 Abs. 1 BGG",
    "Art. 106 Abs. 2 BGG",
    "Art. 64 Abs. 1 BGG",
    "Art. 9 BV",
    "Art. 29 Abs. 2 BV",
}

QUIET_EMBEDDINGS = os.getenv("LIRA_QUIET_EMBEDDINGS", "1") != "0"
SHOW_PROGRESS = os.getenv("LIRA_SHOW_PROGRESS", "0") == "1"


@dataclass
class V43Config:
    encoder_path: Path = CACHE_PATH / "bge_m3_finetuned" / "model"
    train_query_embeddings_path: Path = CACHE_PATH / "train_query_embeddings.npy"
    court_dense_path: Path = CACHE_PATH / "bge_m3_dense.npy"
    court_meta_path: Path = CACHE_PATH / "court_meta.pkl"
    laws_dense_path: Path = CACHE_PATH / "bge_m3_laws_dense.npy"
    laws_sparse_path: Path = CACHE_PATH / "bge_m3_laws_sparse.pkl"
    court_bm25_path: Path = CACHE_PATH / "court_bm25.pkl"
    cooccurrence_path: Path = CACHE_PATH / "cooccurrence_graph.pkl"
    n_splits: int = 5
    random_seed: int = 42
    faiss_factory: str = "IVF4096,PQ64x8"
    faiss_train_size: int = 200000
    faiss_add_batch: int = 50000
    faiss_search_k: int = 4096
    faiss_final_k: int = 1500
    faiss_nprobe: int = 48
    laws_dense_k: int = 300
    laws_sparse_k: int = 300
    court_sparse_k: int = 300
    bm25_k: int = 30
    knn_k: int = 20
    court_dense_min_sim: float = 0.18
    laws_dense_min_sim: float = 0.18
    candidate_cap_per_query: int = 600
    sparse_doc_top_tokens: int = 64
    sparse_case_top_tokens: int = 128
    court_sparse_chunk_rows: int = 2048
    ranker_estimators: int = 350
    ranker_learning_rate: float = 0.05
    ranker_num_leaves: int = 63
    count_model_estimators: int = 250
    count_model_learning_rate: float = 0.05
    count_model_num_leaves: int = 31
    statute_floor_grid: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05)
    case_floor_grid: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05)
    query_prob_thresholds: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
    statute_count_clip: Tuple[int, int] = (0, 25)
    case_count_clip: Tuple[int, int] = (0, 12)
    min_emit_if_high_prob: float = 0.30


@dataclass
class CitationUniverse:
    law_exact_universe: set[str]
    case_exact_universe: set[str]
    case_base_universe: set[str]

    def to_metadata(self) -> Dict[str, object]:
        return {
            "law_exact_count": len(self.law_exact_universe),
            "case_exact_count": len(self.case_exact_universe),
            "case_base_count": len(self.case_base_universe),
            "version": V43_PIPELINE_VERSION,
        }


@dataclass(frozen=True)
class CitationMatch:
    citation: str
    case_match: str = ""


@dataclass
class SparsePostingIndex:
    postings: Dict[str, List[Tuple[int, float]]]
    doc_ids: List[str]

    def search(self, query_weights: Dict[str, float], top_k: int) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = defaultdict(float)
        for token, q_weight in query_weights.items():
            docs = self.postings.get(str(token))
            if not docs:
                continue
            qv = float(q_weight)
            for doc_idx, doc_weight in docs:
                scores[doc_idx] += qv * float(doc_weight)
        if not scores:
            return []
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


class EfficientBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_lens: List[int] = []
        self.N = 0
        self.avgdl = 1.0
        self.inverted: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.df = Counter()

    def _tok(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", str(text).lower()) if text else []

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        tokens = set(self._tok(query))
        scores = Counter()
        for term in tokens:
            if term not in self.df:
                continue
            idf = math.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
            for doc_idx, tf_val in self.inverted[term]:
                dl = self.doc_lens[doc_idx]
                score = idf * (tf_val * (self.k1 + 1)) / (
                    tf_val + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                )
                scores[doc_idx] += score
        return [(self.doc_ids[idx], score) for idx, score in scores.most_common(top_k)]


if "__main__" in sys.modules:
    setattr(sys.modules["__main__"], "EfficientBM25", EfficientBM25)
    setattr(sys.modules["__main__"], "SparsePostingIndex", SparsePostingIndex)


def log(message: str) -> None:
    print(message, flush=True)


def maybe_tqdm(iterable, **kwargs):
    kwargs.setdefault("leave", False)
    kwargs.setdefault("disable", not SHOW_PROGRESS)
    return tqdm(iterable, **kwargs)


def run_quietly(func):
    if not QUIET_EMBEDDINGS:
        return func()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return func()


def encode_with_optional_quiet(encoder: BGEM3FlagModel, *args, **kwargs):
    return run_quietly(lambda: encoder.encode(*args, **kwargs))


def normalize_citation(cit: str) -> str:
    if not isinstance(cit, str):
        return ""
    cit = WS_RE.sub(" ", cit.strip())
    replacements = [
        (r"(?i)\bartikel\s*", "Art. "),
        (r"(?i)\bparagraph\s*", "Abs. "),
        (r"(?i)\bziffer\s*", "Ziff. "),
        (r"(?i)\bziff\.?\s*", "Ziff. "),
        (r"(?i)\balinéa\s*", "Abs. "),
        (r"(?i)\bal\.?\s+", "Abs. "),
        (r"(?i)\bart\.?\s*", "Art. "),
        (r"(?i)\babs\.?\s*", "Abs. "),
        (r"(?i)\blit\.?\s*", "lit. "),
        (r"(?i)\blettra\s*", "lit. "),
        (r"(?i)\blettre\s*", "lit. "),
        (r"(?i)\blet\.?\s+", "lit. "),
        (r"(?i)\bcpv\.?\s+", "Abs. "),
    ]
    for pattern, repl in replacements:
        cit = re.sub(pattern, repl, cit)
    for fr, de in CODE_CANON_MAP.items():
        cit = re.sub(rf"\b{re.escape(fr)}\b", de, cit)
    return cit.strip()


def parent_statute_citation(citation: str) -> Optional[str]:
    """Return the article-level citation for a paragraph or subdivision."""
    normalized = WS_RE.sub(" ", str(citation).strip())
    match = re.match(
        r"^Art\.\s+(\d+[a-zA-Z]*)\s+(?:Abs\.|lit\.|Ziff\.)\s+.+\s+(\S+)$",
        normalized,
    )
    if match is None:
        return None
    return f"Art. {match.group(1)} {match.group(2)}"


def extract_citations_regex(text: str) -> List[str]:
    if not text:
        return []
    citations: List[str] = []
    for match in re.finditer(
        r"(?:Art\.?|Artikel|article)\s*(\d+[a-z]?(?:bis|ter|quater)?)\s*"
        r"(?:(?:Abs|al|cpv)\.?\s*(\d+)\s*)?"
        r"(?:(?:lit|let|lett)\.?\s*([a-z])\s*)?"
        r"(?:(?:Ziff|ch|n)\.?\s*(\d+)\s*)?"
        r"([A-Z][A-Za-zÜüÖöÄä]{1,10})",
        text,
        re.IGNORECASE,
    ):
        code = CODE_CANON_MAP.get(match.group(5).strip(), match.group(5).strip())
        parts = [f"Art. {match.group(1)}"]
        if match.group(2):
            parts.append(f"Abs. {match.group(2)}")
        if match.group(3):
            parts.append(f"lit. {match.group(3)}")
        if match.group(4):
            parts.append(f"Ziff. {match.group(4)}")
        parts.append(code)
        citations.append(" ".join(parts))
    for match in re.finditer(
        r"(?:BGE|ATF|DTF)\s*(\d{2,3})\s+(I{1,3}V?|[IVX]+)\s+(\d+)(?:\s*(?:E\.|Erw\.|consid\.|c\.)\s*([\d\.]+))?",
        text,
    ):
        base = f"BGE {match.group(1)} {match.group(2)} {match.group(3)}"
        citations.append(base)
        if match.group(4):
            citations.append(f"{base} E. {match.group(4)}")
    for match in re.finditer(r"(\d+[A-Z]_\d+/\d{4})(?:\s*(?:E\.|consid\.|c\.)\s*([\d\.]+))?", text):
        citations.append(match.group(1))
        if match.group(2):
            citations.append(f"{match.group(1)} E. {match.group(2)}")
    return citations


def get_case_group_id(case_id: str) -> str:
    if not isinstance(case_id, str):
        return str(case_id)
    case_id = WS_RE.sub(" ", case_id.strip())
    case_id = BASE_CASE_SUFFIX_RE.sub("", case_id)
    docket = DOCKET_PATTERN.match(case_id)
    if docket:
        return docket.group(1)
    bge = re.match(r"((?:BGE|ATF|DTF)\s+\d+\s+[IVX]+\s+\d+)", case_id)
    if bge:
        return bge.group(1)
    return case_id


def normalize_case_citation(citation: str) -> str:
    if not isinstance(citation, str):
        return ""
    citation = WS_RE.sub(" ", citation.strip())
    if not citation:
        return ""
    citation = re.sub(r"^(ATF|DTF)\b", "BGE", citation)
    dated_match = DOCKET_WITH_DATE_RE.match(citation)
    if dated_match:
        citation = f"{dated_match.group(1)}{dated_match.group(2)}"
    return citation


def get_citation_type(citation: str) -> str:
    if citation.startswith("Art."):
        return "statute"
    if citation.startswith("BGE") or citation.startswith("ATF") or citation.startswith("DTF"):
        return "bge"
    if DOCKET_PATTERN.match(citation):
        return "docket"
    return "other"


def get_citation_code(citation: str) -> Optional[str]:
    if not citation:
        return None
    if citation.startswith("BGE") or citation.startswith("ATF") or citation.startswith("DTF"):
        return "BGE"
    if DOCKET_PATTERN.match(citation):
        return "Docket"
    match = re.search(r"([A-Z][A-Za-zÜüÖöÄä]{1,10})\s*$", citation)
    if not match:
        return None
    return CODE_CANON_MAP.get(match.group(1), match.group(1))


def parse_citation_set(value: object) -> List[str]:
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    return [WS_RE.sub(" ", part.strip()) for part in raw.split(";") if part and part.strip()]


def macro_f1_from_sets(predictions: Sequence[Sequence[str]], golds: Sequence[Sequence[str]]) -> float:
    if len(predictions) != len(golds):
        raise ValueError(
            f"Prediction and gold counts differ: {len(predictions)} != {len(golds)}"
        )
    scores: List[float] = []
    for pred, gold in zip(predictions, golds):
        pred_set = set(pred)
        gold_set = set(gold)
        if not pred_set and not gold_set:
            scores.append(1.0)
            continue
        if not pred_set or not gold_set:
            scores.append(0.0)
            continue
        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set)
        recall = tp / len(gold_set)
        scores.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def choose_prediction_column(df: pd.DataFrame) -> str:
    if "gold_citations" in df.columns:
        return "gold_citations"
    if "predicted_citations" in df.columns:
        return "predicted_citations"
    other_cols = [col for col in df.columns if col != "query_id"]
    if len(other_cols) != 1:
        raise RuntimeError(f"Could not infer prediction column from {list(df.columns)}")
    return other_cols[0]


def make_bucket_labels(train_df: pd.DataFrame) -> List[str]:
    gold_lists = [parse_citation_set(value) for value in train_df["gold_citations"].tolist()]
    count_buckets = pd.cut(
        pd.Series([len(g) for g in gold_lists]),
        bins=[-0.1, 2, 5, 10, 20, 1000],
        labels=["00_02", "03_05", "06_10", "11_20", "21_plus"],
    ).astype(str)
    has_case = [
        "case"
        if any(get_citation_type(cit) in {"bge", "docket", "other"} for cit in gold)
        else "statute"
        for gold in gold_lists
    ]
    combo_labels = [f"{bucket}|{case_flag}" for bucket, case_flag in zip(count_buckets, has_case)]
    combo_counts = Counter(combo_labels)
    bucket_to_combos: Dict[str, set] = defaultdict(set)
    for combo_label, bucket in zip(combo_labels, count_buckets):
        bucket_to_combos[bucket].add(combo_label)

    bucket_mode: Dict[str, str] = {}
    for bucket, combos in bucket_to_combos.items():
        if all(combo_counts[combo] >= 5 for combo in combos):
            bucket_mode[bucket] = "combo"
        else:
            bucket_mode[bucket] = "bucket"

    return [
        combo_label if bucket_mode[bucket] == "combo" else bucket
        for combo_label, bucket in zip(combo_labels, count_buckets)
    ]


def build_folds(train_df: pd.DataFrame, config: V43Config) -> Dict[str, int]:
    labels = make_bucket_labels(train_df)
    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )
    fold_map: Dict[str, int] = {}
    X = np.zeros(len(train_df))
    y = np.asarray(labels)
    for fold, (_, valid_idx) in enumerate(splitter.split(X, y)):
        for idx in valid_idx:
            fold_map[str(train_df.iloc[idx]["query_id"])] = int(fold)
    return fold_map


def save_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def prune_weights(weights: Dict[str, float], keep_top: int) -> Dict[str, float]:
    if len(weights) <= keep_top:
        return {str(k): float(v) for k, v in weights.items()}
    top_items = sorted(weights.items(), key=lambda kv: float(kv[1]), reverse=True)[:keep_top]
    return {str(k): float(v) for k, v in top_items}


def build_sparse_postings(doc_ids: Sequence[str], lexical_weights: Sequence[Dict[str, float]], keep_top: int) -> SparsePostingIndex:
    postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for doc_idx, weights in enumerate(maybe_tqdm(lexical_weights, desc="Building sparse postings")):
        kept = prune_weights(weights, keep_top)
        for token, weight in kept.items():
            postings[str(token)].append((doc_idx, float(weight)))
    return SparsePostingIndex(postings=dict(postings), doc_ids=list(doc_ids))


def ensure_laws_sparse_postings(config: V43Config) -> None:
    if LAWS_SPARSE_POSTINGS_PATH.exists():
        log(f"   Laws sparse postings: {LAWS_SPARSE_POSTINGS_PATH.name}")
        return
    log("   Building laws sparse postings...")
    laws_meta = load_pickle(config.laws_sparse_path)
    sparse_index = build_sparse_postings(
        doc_ids=[str(c) for c in laws_meta["citations"]],
        lexical_weights=laws_meta["lexical_weights"],
        keep_top=config.sparse_doc_top_tokens,
    )
    save_pickle(LAWS_SPARSE_POSTINGS_PATH, sparse_index)
    log(f"   Saved: {LAWS_SPARSE_POSTINGS_PATH}")


def encode_sparse_query(encoder: BGEM3FlagModel, query: str) -> Tuple[np.ndarray, Dict[str, float]]:
    encoded = encode_with_optional_quiet(
        encoder,
        [query],
        batch_size=1,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = encoded["dense_vecs"][0].astype(np.float32)
    sparse = {str(k): float(v) for k, v in encoded["lexical_weights"][0].items()}
    return dense, sparse


def load_encoder(config: V43Config) -> BGEM3FlagModel:
    use_cuda = torch.cuda.is_available()
    devices = ["cuda:0"] if use_cuda else ["cpu"]
    return BGEM3FlagModel(str(config.encoder_path), use_fp16=use_cuda, devices=devices)


def build_faiss_index(config: V43Config) -> None:
    if COURT_FAISS_PATH.exists() and COURT_FAISS_META_PATH.exists():
        log(f"   Court FAISS index: {COURT_FAISS_PATH.name}")
        return
    log("   Building FAISS court index...")
    court_embeddings = np.load(config.court_dense_path, mmap_mode="r")
    total, dim = court_embeddings.shape
    rng = np.random.default_rng(config.random_seed)
    sample_size = min(config.faiss_train_size, total)
    sample_idx = rng.choice(total, size=sample_size, replace=False)
    train_vecs = np.asarray(court_embeddings[sample_idx], dtype=np.float32)
    index = faiss.index_factory(dim, config.faiss_factory, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        log(f"   Training FAISS on {sample_size:,} samples...")
        index.train(train_vecs)
    for start in maybe_tqdm(range(0, total, config.faiss_add_batch), desc="Adding court vectors"):
        end = min(start + config.faiss_add_batch, total)
        batch = np.asarray(court_embeddings[start:end], dtype=np.float32)
        index.add(batch)
    if hasattr(index, "nprobe"):
        index.nprobe = config.faiss_nprobe
    faiss.write_index(index, str(COURT_FAISS_PATH))
    with COURT_FAISS_META_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "factory": config.faiss_factory,
                "nprobe": config.faiss_nprobe,
                "rows": int(total),
                "dim": int(dim),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    log(f"   Saved: {COURT_FAISS_PATH}")


def build_court_sparse_index(config: V43Config, encoder: BGEM3FlagModel) -> None:
    if COURT_SPARSE_POSTINGS_PATH.exists():
        log(f"   Court sparse postings: {COURT_SPARSE_POSTINGS_PATH.name}")
        return
    csv_path = DATA_PATH / "court_considerations.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Missing {csv_path}")
    log("   Building court sparse postings from court_considerations.csv...")
    case_weights: Dict[str, Dict[str, float]] = defaultdict(dict)
    for chunk in pd.read_csv(csv_path, chunksize=config.court_sparse_chunk_rows, usecols=["citation", "text"]):
        mask = chunk["text"].notna() & (chunk["text"].astype(str).str.len() >= 50)
        filtered = chunk.loc[mask]
        if filtered.empty:
            continue
        texts = filtered["text"].astype(str).tolist()
        case_ids = [get_case_group_id(cit) for cit in filtered["citation"].astype(str).tolist()]
        output = encode_with_optional_quiet(
            encoder,
            texts,
            batch_size=16,
            max_length=768,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        for case_id, weights in zip(case_ids, output["lexical_weights"]):
            target = case_weights[case_id]
            for token, weight in prune_weights(weights, 32).items():
                target[token] = float(target.get(token, 0.0)) + float(weight)
            if len(target) > config.sparse_case_top_tokens * 3:
                case_weights[case_id] = prune_weights(target, config.sparse_case_top_tokens)
        gc.collect()
    case_ids = sorted(case_weights.keys())
    lexical_weights = [prune_weights(case_weights[case_id], config.sparse_case_top_tokens) for case_id in case_ids]
    sparse_index = build_sparse_postings(case_ids, lexical_weights, config.sparse_case_top_tokens)
    save_pickle(COURT_SPARSE_POSTINGS_PATH, sparse_index)
    log(f"   Saved: {COURT_SPARSE_POSTINGS_PATH}")


@dataclass
class V43Resources:
    config: V43Config
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    laws_df: pd.DataFrame
    encoder: BGEM3FlagModel
    citation_universe: CitationUniverse
    laws_text: Dict[str, str]
    laws_code: Dict[str, str]
    laws_embeddings: np.ndarray
    laws_citations: List[str]
    laws_sparse_index: SparsePostingIndex
    court_embeddings: np.ndarray
    court_case_ids: List[str]
    court_case_citations: List[List[str]]
    court_faiss: faiss.Index
    court_sparse_index: Optional[SparsePostingIndex]
    court_bm25: EfficientBM25
    case_extracted_cits: Dict[str, Counter]
    citation_doc_freq: Counter
    total_cases: int
    cooccurrence_graph: Dict[str, List[Tuple[str, float]]]
    train_query_embs: np.ndarray
    train_gold_cits: List[List[str]]
    folds: Dict[str, int]
    query_lookup: Dict[str, str]


def build_citation_universe(
    laws_df: pd.DataFrame,
    court_case_ids: Sequence[str],
    court_case_citations: Sequence[Sequence[str]],
    case_extracted_cits: Dict[str, Counter],
) -> CitationUniverse:
    law_exact_universe = {
        WS_RE.sub(" ", str(citation).strip())
        for citation in laws_df["citation"].tolist()
        if isinstance(citation, str) and citation.strip()
    }
    law_exact_universe.update(
        parent
        for citation in tuple(law_exact_universe)
        if (parent := parent_statute_citation(citation)) is not None
    )
    case_exact_universe: set[str] = set()
    case_base_universe: set[str] = set()

    def add_case_citation(raw_citation: object) -> None:
        citation = normalize_case_citation(str(raw_citation))
        if not citation:
            return
        citation_type = get_citation_type(citation)
        if citation_type not in {"bge", "docket"}:
            return
        case_exact_universe.add(citation)
        case_base_universe.add(get_case_group_id(citation))

    for case_id in court_case_ids:
        add_case_citation(case_id)
        case_base_universe.add(get_case_group_id(str(case_id)))

    for citations in court_case_citations:
        for citation in citations:
            add_case_citation(citation)

    for case_id, citations in case_extracted_cits.items():
        add_case_citation(case_id)
        case_base_universe.add(get_case_group_id(str(case_id)))
        for citation in citations:
            add_case_citation(citation)

    return CitationUniverse(
        law_exact_universe=law_exact_universe,
        case_exact_universe=case_exact_universe,
        case_base_universe=case_base_universe,
    )


def get_valid_citation(raw: str, citation_universe: CitationUniverse) -> Optional[CitationMatch]:
    if not raw:
        return None
    raw = WS_RE.sub(" ", str(raw).strip())
    if not raw:
        return None

    case_raw = normalize_case_citation(raw)
    case_type = get_citation_type(case_raw)
    if case_type in {"bge", "docket"}:
        if case_raw in citation_universe.case_exact_universe:
            return CitationMatch(case_raw, case_match="exact")
        base = get_case_group_id(case_raw)
        if base in citation_universe.case_base_universe:
            return CitationMatch(base, case_match="base")
        return None

    if raw in citation_universe.law_exact_universe:
        return CitationMatch(raw)
    norm = normalize_citation(raw)
    if norm in citation_universe.law_exact_universe:
        return CitationMatch(norm)
    if " lit. " in norm:
        fallback = re.sub(r"\s+lit\. [a-z]\s+", " ", norm)
        fallback = WS_RE.sub(" ", fallback).strip()
        if fallback in citation_universe.law_exact_universe:
            return CitationMatch(fallback)
    if " Ziff. " in norm:
        fallback = re.sub(r"\s+Ziff\. \d+\s+", " ", norm)
        fallback = WS_RE.sub(" ", fallback).strip()
        if fallback in citation_universe.law_exact_universe:
            return CitationMatch(fallback)
    return None


def get_citation_idf(citation: str, citation_doc_freq: Counter, total_cases: int) -> float:
    df = citation_doc_freq.get(citation, 0)
    if df == 0:
        return 2.0
    idf = math.log(total_cases / (df + 1))
    return max(0.1, min(5.0, idf / 4.0))


def detect_domain_codes(query: str) -> set:
    detected_codes = set()
    lowered = query.lower()
    for code, keywords in DOMAIN_KEYWORDS.items():
        if sum(1 for kw in keywords if kw.lower() in lowered) >= 2:
            detected_codes.add(code)
    for code in ["StPO", "StGB", "ZGB", "OR", "BGG", "ZPO", "ATSG", "IVG", "SchKG", "BV", "EMRK", "IPRG", "VVG", "SVG", "AHVG", "UVG"]:
        if code in query:
            detected_codes.add(code)
    return detected_codes


def load_resources(config: V43Config, require_court_sparse: bool = False) -> V43Resources:
    log("Loading V43 resources...")
    train_df = pd.read_csv(DATA_PATH / "train.csv").dropna(subset=["query", "gold_citations"]).reset_index(drop=True)
    val_df = pd.read_csv(DATA_PATH / "val.csv")
    test_df = pd.read_csv(DATA_PATH / "test.csv")
    laws_df = pd.read_csv(DATA_PATH / "laws_de.csv")

    laws_text: Dict[str, str] = {}
    laws_code: Dict[str, str] = {}
    for _, row in laws_df.iterrows():
        citation = str(row.get("citation", "")).strip()
        text = str(row.get("text", "")).strip()
        if citation:
            laws_text[citation] = text
            laws_code[citation] = get_citation_code(citation) or ""
    train_gold_cits = [parse_citation_set(value) for value in train_df["gold_citations"].tolist()]

    if FOLDS_PATH.exists():
        folds = load_pickle(FOLDS_PATH)
    else:
        folds = build_folds(train_df, config)
        save_pickle(FOLDS_PATH, folds)

    encoder = load_encoder(config)
    laws_embeddings = np.load(config.laws_dense_path, mmap_mode="r")
    laws_sparse_index = load_pickle(LAWS_SPARSE_POSTINGS_PATH)
    laws_citations = list(laws_sparse_index.doc_ids)

    court_embeddings = np.load(config.court_dense_path, mmap_mode="r")
    court_meta = load_pickle(config.court_meta_path)
    court_case_ids = list(court_meta["case_ids"])
    court_case_citations = list(court_meta["case_citations"])
    court_faiss = faiss.read_index(str(COURT_FAISS_PATH))
    if hasattr(court_faiss, "nprobe"):
        court_faiss.nprobe = config.faiss_nprobe

    court_sparse_index = None
    if COURT_SPARSE_POSTINGS_PATH.exists():
        court_sparse_index = load_pickle(COURT_SPARSE_POSTINGS_PATH)
    elif require_court_sparse:
        raise RuntimeError(
            f"{COURT_SPARSE_POSTINGS_PATH} is missing. Run `python lira_retrieval.py prepare --build-court-sparse`."
        )

    bm25_cache = load_pickle(config.court_bm25_path)
    court_bm25 = bm25_cache["bm25"]
    case_extracted_cits = bm25_cache["extracted_cits"]
    citation_universe = build_citation_universe(
        laws_df=laws_df,
        court_case_ids=court_case_ids,
        court_case_citations=court_case_citations,
        case_extracted_cits=case_extracted_cits,
    )
    citation_doc_freq: Counter = Counter()
    for case_id, citations in case_extracted_cits.items():
        for citation in citations:
            citation_doc_freq[citation] += 1
    total_cases = len(case_extracted_cits)

    cooccurrence_graph = {}
    if config.cooccurrence_path.exists():
        cooccurrence_graph = load_pickle(config.cooccurrence_path)

    train_query_embs = np.load(config.train_query_embeddings_path)
    query_lookup = {}
    for df in (train_df, val_df, test_df):
        for _, row in df.iterrows():
            query_lookup[str(row["query_id"])] = str(row["query"])

    return V43Resources(
        config=config,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        laws_df=laws_df,
        encoder=encoder,
        citation_universe=citation_universe,
        laws_text=laws_text,
        laws_code=laws_code,
        laws_embeddings=laws_embeddings,
        laws_citations=laws_citations,
        laws_sparse_index=laws_sparse_index,
        court_embeddings=court_embeddings,
        court_case_ids=court_case_ids,
        court_case_citations=court_case_citations,
        court_faiss=court_faiss,
        court_sparse_index=court_sparse_index,
        court_bm25=court_bm25,
        case_extracted_cits=case_extracted_cits,
        citation_doc_freq=citation_doc_freq,
        total_cases=total_cases,
        cooccurrence_graph=cooccurrence_graph,
        train_query_embs=train_query_embs.astype(np.float32, copy=False),
        train_gold_cits=train_gold_cits,
        folds=folds,
        query_lookup=query_lookup,
    )


