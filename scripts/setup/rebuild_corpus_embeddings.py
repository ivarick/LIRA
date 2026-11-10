"""
LIRA — Corpus Re-Encoder
========================
Regenerates bge_m3_dense.npy and bge_m3_laws_dense.npy using the fine-tuned
BGE-M3 model. Must be run after finetune_bge_m3.py completes.

This is what makes the fine-tuning actually improve V41 — the embeddings stored
in cache/ must be regenerated with the new model. V41 loads them at startup via
np.load() and never re-encodes during prediction.

Runtime: ~30-60 minutes depending on corpus size.
The original .npy files are renamed with _original suffix as backup.
"""

import gc
import os
import sys
import numpy as np
import pickle
import torch
from pathlib import Path
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CACHE_PATH      = Path("./cache")
FINETUNED_PATH  = CACHE_PATH / "bge_m3_finetuned" / "model"
CHUNK_SIZE      = 512   # encode this many docs at a time
BATCH_SIZE      = 32    # batch size within each chunk

# ============================================================================
print("=" * 60)
print("LIRA — Corpus Re-Encoder with Fine-Tuned BGE-M3")
print("=" * 60)

if not FINETUNED_PATH.exists():
    print(f"\n❌ Fine-tuned model not found at {FINETUNED_PATH}")
    print("   Run finetune_bge_m3.py first.")
    sys.exit(1)

print(f"\nLoading fine-tuned BGE-M3 from {FINETUNED_PATH}...")
from FlagEmbedding import BGEM3FlagModel  # noqa: E402 - model is loaded after cache checks
encoder = BGEM3FlagModel(str(FINETUNED_PATH), use_fp16=True, devices=["cuda:0"])
print(f"   GPU after load: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# ============================================================================
# RE-ENCODE LAWS (smaller, do first)
# ============================================================================
print("\n1. Re-encoding laws corpus (laws_de.csv)...")
laws_emb_path = CACHE_PATH / "bge_m3_laws_dense.npy"
laws_meta_path = CACHE_PATH / "bge_m3_laws_sparse.pkl"

if not laws_emb_path.exists():
    print("   ❌ bge_m3_laws_dense.npy not found — skipping laws re-encoding")
else:
    # Load the metadata to get citation order
    with open(laws_meta_path, "rb") as f:
        laws_meta = pickle.load(f)
    laws_citations = laws_meta["citations"]
    print(f"   Laws corpus: {len(laws_citations):,} citations")
    
    # Load laws text
    import pandas as pd
    laws_df = pd.read_csv(Path("./data") / "laws_de.csv")
    cit_to_text = {}
    for _, row in laws_df.iterrows():
        cit  = str(row.get("citation", "")).strip()
        text = str(row.get("text", "")).strip()
        if cit and text:
            cit_to_text[cit] = text
    
    # Build document list in original citation order
    docs = [f"{cit}: {cit_to_text.get(cit, '')[:800]}" for cit in laws_citations]
    
    # Backup original
    backup_path = CACHE_PATH / "bge_m3_laws_dense_original.npy"
    if not backup_path.exists():
        import shutil
        shutil.copy(laws_emb_path, backup_path)
        print(f"   Backed up original: {backup_path.name}")
    
    # Re-encode
    print(f"   Encoding {len(docs):,} law articles...")
    new_embs = []
    for i in tqdm(range(0, len(docs), CHUNK_SIZE), desc="   Laws"):
        batch = docs[i:i+CHUNK_SIZE]
        emb   = encoder.encode(batch, return_dense=True,
                                batch_size=BATCH_SIZE, max_length=512)["dense_vecs"]
        new_embs.append(emb.astype(np.float32))
        gc.collect()
    
    new_embs = np.vstack(new_embs)
    np.save(laws_emb_path, new_embs)
    print(f"   Saved: {laws_emb_path} ({new_embs.shape}, {new_embs.nbytes/1024**2:.0f}MB)")
    del new_embs
    gc.collect()

# ============================================================================
# RE-ENCODE COURT CONSIDERATIONS (large — the main embedding matrix)
# ============================================================================
print("\n2. Re-encoding court considerations (bge_m3_dense.npy)...")
court_emb_path = CACHE_PATH / "bge_m3_dense.npy"
court_meta_path = CACHE_PATH / "court_meta.pkl"

if not court_emb_path.exists():
    print("   ❌ bge_m3_dense.npy not found — cannot proceed")
    sys.exit(1)

with open(court_meta_path, "rb") as f:
    court_meta = pickle.load(f)

court_case_ids   = court_meta["case_ids"]
# court_meta may have "texts" or they may be in a separate file
court_texts      = court_meta.get("texts", None)

if court_texts is None:
    print("   ⚠ court_meta.pkl does not contain texts field.")
    # Try court_texts.pkl
    court_texts_path = CACHE_PATH / "court_texts.pkl"
    if court_texts_path.exists():
        print("   Loading from court_texts.pkl...")
        with open(court_texts_path, "rb") as f:
            court_texts = pickle.load(f)
        print(f"   Loaded: {len(court_texts):,} texts")
    else:
        # Try court_considerations.csv (2.3GB in cache/)
        import pandas as pd
        csv_candidates = [
            CACHE_PATH / "court_considerations.csv",
            Path("./data") / "court_considerations.csv",
        ]
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        if csv_path:
            print(f"   Loading from {csv_path.name} ({csv_path.stat().st_size/1024**3:.1f}GB)...")
            court_df = pd.read_csv(csv_path, low_memory=False)
            # Find the text column
            text_col = next((c for c in ["consideration", "text", "considerations", "content"]
                             if c in court_df.columns), court_df.columns[0])
            print(f"   Using column: '{text_col}' ({len(court_df):,} rows)")
            court_texts = court_df[text_col].fillna("").tolist()
            del court_df
        else:
            print("   ❌ No court texts found anywhere. Skipping court re-encoding.")
            print("   Laws embeddings ARE updated. V41 will still benefit from fine-tuned laws retrieval.")
            sys.exit(0)

n_considerations = len(court_case_ids)
print(f"   Court corpus: {n_considerations:,} considerations")

# Check original embedding shape
original_embs = np.load(court_emb_path, mmap_mode="r")
orig_shape = original_embs.shape
del original_embs
print(f"   Original embedding shape: {orig_shape}")
print(f"   Estimated time: {n_considerations / (CHUNK_SIZE * 60):.0f}–{n_considerations / (CHUNK_SIZE * 30):.0f} minutes")

# Backup original — only if we have enough free disk space
backup_path = CACHE_PATH / "bge_m3_dense_original.npy"
if not backup_path.exists():
    import shutil
    orig_size_gb = court_emb_path.stat().st_size / 1024**3
    free_gb = shutil.disk_usage(str(CACHE_PATH)).free / 1024**3
    if free_gb > orig_size_gb + 1.0:   # need file size + 1GB headroom
        print(f"   Creating backup ({orig_size_gb:.1f}GB, {free_gb:.1f}GB free)...")
        shutil.copy(court_emb_path, backup_path)
        print(f"   Backed up: {backup_path.name}")
    else:
        print(f"   ⚠ Skipping backup — only {free_gb:.1f}GB free, need {orig_size_gb+1:.1f}GB")
        print("   (Original will be overwritten in-place at the end)")

# Re-encode in chunks, writing DIRECTLY IN-PLACE to the existing file.
# This avoids needing an extra 8.7GB — the file already exists on disk.
print(f"   Writing directly into existing {court_emb_path.name} (no extra disk space needed)...")
new_embs = np.load(str(court_emb_path), mmap_mode="r+")

for i in tqdm(range(0, n_considerations, CHUNK_SIZE), desc="   Court considerations"):
    end   = min(i + CHUNK_SIZE, n_considerations)
    batch = court_texts[i:end]
    # Handle both list and dict formats
    if isinstance(batch, dict):
        batch = [batch.get(j, "") for j in range(i, end)]
    batch = [str(t)[:1000] for t in batch]
    emb   = encoder.encode(batch, return_dense=True,
                            batch_size=BATCH_SIZE, max_length=512)["dense_vecs"]
    new_embs[i:end] = emb.astype(np.float32)
    if i % (CHUNK_SIZE * 20) == 0 and i > 0:
        new_embs.flush()
        gc.collect()

new_embs.flush()
del new_embs
print(f"   Saved: {court_emb_path} ({n_considerations:,} × {orig_shape[1]}d)")

# ============================================================================
print("\n" + "=" * 60)
print("  CORPUS RE-ENCODING COMPLETE")
print("=" * 60)
print("  bge_m3_dense.npy     — replaced with fine-tuned embeddings")
print("  bge_m3_laws_dense.npy — replaced with fine-tuned embeddings")
print("  *_original.npy        — original embeddings backed up")
print()
print("  V41 will now use fine-tuned BGE-M3 automatically.")
print("  Run: python lira_pipeline.py")
print("=" * 60)

