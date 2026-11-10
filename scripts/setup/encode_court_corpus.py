#!/usr/bin/env python3
import gc
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

CACHE_PATH = Path("./cache")
DATA_PATH = Path("./data")

CSV_PATH = DATA_PATH / "court_considerations.csv"
COURT_META_PATH = CACHE_PATH / "court_meta.pkl"
MODEL_PATH = CACHE_PATH / "bge_m3_finetuned" / "model"
OUTPUT_PATH = CACHE_PATH / "bge_m3_dense.npy"

CSV_CHUNKSIZE = 50_000
ENCODE_GROUP_SIZE = 512
ENCODE_BATCH_SIZE = 32
FLUSH_EVERY_BATCHES = 20
EMBEDDING_DIM = 1024
MAX_LENGTH = 512


def build_keep_mask(chunk: pd.DataFrame) -> pd.Series:
    text_series = chunk["text"]
    return text_series.notna() & (text_series.str.len() >= 50)


def load_case_ids() -> list:
    if not COURT_META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {COURT_META_PATH}")
    with open(COURT_META_PATH, "rb") as f:
        court_meta = pickle.load(f)
    case_ids = court_meta.get("case_ids")
    if not isinstance(case_ids, list) or not case_ids:
        raise RuntimeError("court_meta.pkl does not contain a valid non-empty case_ids list")
    return case_ids


def validate_csv_columns() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV file: {CSV_PATH}")
    sample = pd.read_csv(CSV_PATH, nrows=0)
    columns = list(sample.columns)
    missing = [name for name in ["citation", "text"] if name not in sample.columns]
    if missing:
        raise RuntimeError(f"CSV missing required columns {missing}; found {columns}")
    print(f"   CSV columns verified: {columns}")


def validate_output_shape(expected_rows: int) -> tuple[int, int]:
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Missing output embedding file: {OUTPUT_PATH}")
    embeddings = np.load(OUTPUT_PATH, mmap_mode="r")
    shape = embeddings.shape
    dtype = embeddings.dtype
    del embeddings
    gc.collect()

    if shape != (expected_rows, EMBEDDING_DIM):
        raise RuntimeError(
            f"Existing embedding matrix has shape {shape}, expected {(expected_rows, EMBEDDING_DIM)}"
        )

    print(f"   Existing matrix verified: shape={shape}, dtype={dtype}")
    return shape


def count_filtered_rows(expected_rows: int) -> int:
    print("\nPhase 1: Counting filtered rows (no writing)...")
    kept = 0
    processed = 0
    for chunk in pd.read_csv(CSV_PATH, chunksize=CSV_CHUNKSIZE, usecols=["citation", "text"]):
        mask = build_keep_mask(chunk)
        kept += int(mask.sum())
        processed += len(chunk)
        if processed % (CSV_CHUNKSIZE * 10) == 0:
            print(f"   Processed: {processed:,} rows | Filtered: {kept:,}")

    print(f"   Filtered rows: {kept:,} | Expected: {expected_rows:,}")
    if kept != expected_rows:
        raise RuntimeError(f"ABORT: count mismatch ({kept} != {expected_rows}). Do not write.")

    print("   Count verified. Proceeding to encode.")
    return kept


def load_encoder() -> BGEM3FlagModel:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing fine-tuned model directory: {MODEL_PATH}")
    print("\nPhase 2: Loading fine-tuned BGE-M3...")
    encoder = BGEM3FlagModel(str(MODEL_PATH), use_fp16=True, devices=["cuda:0"])
    print(f"   Model loaded from {MODEL_PATH}")
    return encoder


def encode_buffer(encoder: BGEM3FlagModel, texts: list[str], target_dtype: np.dtype) -> np.ndarray:
    dense = encoder.encode(
        texts,
        return_dense=True,
        batch_size=ENCODE_BATCH_SIZE,
        max_length=MAX_LENGTH,
    )["dense_vecs"]
    return dense.astype(target_dtype, copy=False)


def write_embeddings(case_ids: list, encoder: BGEM3FlagModel) -> int:
    print("\nPhase 3: Encoding filtered rows and writing in place...")
    output = np.load(OUTPUT_PATH, mmap_mode="r+")
    target_dtype = output.dtype
    write_idx = 0
    batch_count = 0
    text_buffer = []

    progress = tqdm(total=len(case_ids), desc="   Re-encoding", unit="rows")

    def flush_buffer() -> None:
        nonlocal write_idx, batch_count, text_buffer
        if not text_buffer:
            return
        end_idx = write_idx + len(text_buffer)
        output[write_idx:end_idx] = encode_buffer(encoder, text_buffer, target_dtype)
        write_idx = end_idx
        batch_count += 1
        progress.update(len(text_buffer))
        text_buffer = []
        if batch_count % FLUSH_EVERY_BATCHES == 0:
            output.flush()
            gc.collect()

    for chunk in pd.read_csv(CSV_PATH, chunksize=CSV_CHUNKSIZE, usecols=["citation", "text"]):
        filtered = chunk.loc[build_keep_mask(chunk), ["citation", "text"]]
        for citation, text in filtered.itertuples(index=False, name=None):
            if write_idx + len(text_buffer) >= len(case_ids):
                raise RuntimeError("Encountered more filtered rows than expected before write completion")

            expected_citation = case_ids[write_idx + len(text_buffer)]
            if str(citation) != str(expected_citation):
                raise RuntimeError(
                    "Alignment mismatch at row "
                    f"{write_idx + len(text_buffer):,}: csv citation={citation!r} "
                    f"expected={expected_citation!r}"
                )

            text_buffer.append(str(text))
            if len(text_buffer) >= ENCODE_GROUP_SIZE:
                flush_buffer()

        flush_buffer()

    flush_buffer()
    progress.close()
    output.flush()
    gc.collect()

    return write_idx


def verify_final_shape(expected_rows: int) -> None:
    print("\nPhase 4: Verifying final matrix...")
    embeddings = np.load(OUTPUT_PATH, mmap_mode="r")
    shape = embeddings.shape
    dtype = embeddings.dtype
    del embeddings
    gc.collect()

    if shape != (expected_rows, EMBEDDING_DIM):
        raise RuntimeError(f"Post-write shape mismatch: found {shape}, expected {(expected_rows, EMBEDDING_DIM)}")

    print(f"   Final matrix verified: shape={shape}, dtype={dtype}")


def main() -> None:
    print("=" * 70)
    print("LIRA V42 COURT RE-ENCODER")
    print("=" * 70)

    print("\nPhase 0: Validating inputs...")
    validate_csv_columns()
    case_ids = load_case_ids()
    print(f"   Loaded case_ids: {len(case_ids):,}")
    validate_output_shape(len(case_ids))

    count_filtered_rows(len(case_ids))
    encoder = load_encoder()

    written_rows = write_embeddings(case_ids, encoder)
    if written_rows != len(case_ids):
        raise RuntimeError(
            f"Write count mismatch: wrote {written_rows:,} rows, expected {len(case_ids):,}"
        )

    verify_final_shape(len(case_ids))

    print("\n" + "=" * 70)
    print(f"Completed: {OUTPUT_PATH} rewritten in place")
    print(f"Rows written: {written_rows:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()


