"""
LIRA — BGE-M3 Fine-Tuning for Swiss Legal Retrieval
=====================================================
Teaches BGE-M3 to map English legal queries → Swiss article embeddings.

This script trains an embedding model on labeled query/article pairs. It uses
query-disjoint validation and excludes paragraph candidates of gold articles
from hard negatives. Its effect on end-to-end citation F1 requires a full run.

Architecture:
  - CachedMultipleNegativesRankingLoss (CMNRL) — memory-efficient contrastive loss
  - In-batch negatives: each article in the batch is a negative for all other queries
  - Hard negatives: top-k retrieved articles NOT in gold set (hardest negatives)
  - SentenceTransformerTrainer (modern API, sentence-transformers ≥ 3.0)

Data construction:
  From 1,139 labelled (query, gold_citations) pairs:
    1. Match gold Art. citations to article text in laws_de.csv
    2. Each (query, article_text) pair = one positive training example
    3. Mine hard negatives using the CURRENT (untuned) BGE-M3 on the full laws corpus
    4. Format: {"query": str, "positive": str, "negative": str}
    Expected ~8,000-15,000 positive pairs (5-15 Art. citations per query avg)

Hardware: RTX 3060 12GB
  - BGE-M3 (567M params, fp16) = ~1.1GB VRAM
  - Training with batch_size=4, grad_accum=8 = effective batch 32
  - Gradient checkpointing enabled
  - Total VRAM estimate: ~6-8GB — fits on 12GB

Output: cache/bge_m3_finetuned/ (~2.2GB)
  After training, re-encode the law and court corpora with the saved model.
"""

import gc
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from lira_retrieval import parent_statute_citation
from training_utils import split_triplets_by_query, top_non_gold_indices

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH    = Path("./data")
CACHE_PATH   = Path("./cache")
OUTPUT_DIR   = CACHE_PATH / "bge_m3_finetuned"

BGE_M3_SNAPSHOTS = CACHE_PATH / "hf_bypass" / "hub" / "models--BAAI--bge-m3" / "snapshots"
cached_snapshots = sorted(
    (path for path in BGE_M3_SNAPSHOTS.glob("*") if path.is_dir()),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
BGE_M3_PATH = os.environ.get("LIRA_BGE_M3_PATH") or str(
    cached_snapshots[0] if cached_snapshots else BGE_M3_SNAPSHOTS
)

# Training hyperparameters — tuned for 12GB VRAM
BATCH_SIZE    = 4      # Per-device batch size
GRAD_ACCUM    = 8      # Effective batch = 32 (important for MNRL — more negatives)
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-5
MAX_QUERY_LEN = 512    # Legal queries can be long
MAX_DOC_LEN   = 512    # Article texts are paragraph-length
HARD_NEG_PER_QUERY = 5 # Hard negatives per query (stored in dataset)

# ============================================================================
# STEP 0: VERIFY
# ============================================================================
print("=" * 60)
print("LIRA BGE-M3 Fine-Tuning — Swiss Legal Retrieval")
print("=" * 60)

print("\n0. Checking environment...")
for mod in ['torch', 'sentence_transformers', 'datasets', 'transformers']:
    try:
        __import__(mod)
        print(f"   ✓ {mod}")
    except ImportError:
        print(f"   ✗ {mod} missing — run: pip install {mod}")
        sys.exit(1)

import psutil  # noqa: E402 - report environment after dependency checks
ram  = psutil.virtual_memory()
print(f"   RAM: {ram.total/1024**3:.0f}GB ({ram.available/1024**3:.1f}GB free)")
print(f"   GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")

if not Path(BGE_M3_PATH).exists():
    print(f"\n   ❌ BGE-M3 not found at:\n   {BGE_M3_PATH}")
    print("   Run scripts/setup/download_models.py or set LIRA_BGE_M3_PATH.")
    sys.exit(1)
print(f"   BGE-M3: found at {BGE_M3_PATH}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n1. Loading data...")
train_df = pd.read_csv(DATA_PATH / "train.csv").dropna(subset=["query", "gold_citations"])
laws_df  = pd.read_csv(DATA_PATH / "laws_de.csv")
print(f"   Train queries: {len(train_df)}")
print(f"   Laws articles: {len(laws_df)}")

# Build citation → text lookup (only Art. citations have text)
cit_to_text = {}
for _, row in laws_df.iterrows():
    cit  = str(row.get("citation", "")).strip()
    text = str(row.get("text", "")).strip()
    if cit and text and len(text) > 20:
        cit_to_text[cit] = text
print(f"   Citation→text lookup: {len(cit_to_text)} articles with text")
parent_to_text = {}
for citation, text in cit_to_text.items():
    parent = parent_statute_citation(citation)
    if parent and parent not in cit_to_text:
        parent_to_text.setdefault(parent, text)
positive_text_lookup = {**parent_to_text, **cit_to_text}
print(f"   Article-level text aliases: {len(parent_to_text)}")

# ============================================================================
# STEP 2: BUILD POSITIVE PAIRS
# ============================================================================
print("\n2. Building positive training pairs...")
# For each query, match gold citations to article text in laws_de.csv
# Each (query, article_text) pair = one positive training example
positives = []   # list of {"query": str, "positive": str, "gold_cits": [str]}
query_gold = []  # parallel list of gold citation sets (for hard neg mining)

for _, row in train_df.iterrows():
    q     = str(row["query"]).strip()
    golds = [c.strip() for c in str(row["gold_citations"]).split(";") if c.strip()]
    
    matched_texts = []
    for cit in golds:
        # Direct lookup
        if cit in positive_text_lookup:
            matched_texts.append((cit, positive_text_lookup[cit]))
            continue
        # Normalisation fallback: strip Abs./lit. suffixes
        base = cit
        for suffix_pattern in [r"\s+Abs\.\s+\d+\s+", r"\s+lit\.\s+[a-z]\s+", r"\s+Ziff\.\s+\d+\s+"]:
            import re
            base = re.sub(suffix_pattern, " ", base).strip()
        if base != cit and base in positive_text_lookup:
            matched_texts.append((base, positive_text_lookup[base]))
    
    # Create one positive pair per matched citation
    for cit, text in matched_texts:
        positives.append({
            "query":    q,
            "positive": f"{cit}: {text[:800]}",  # prepend citation for context
            "gold_cits": golds
        })
    
    # Store for hard negative mining
    query_gold.append({
        "query": q,
        "gold_set": set(golds),
        "gold_texts": {cit: text for cit, text in matched_texts}
    })

print(f"   Positive pairs: {len(positives)}")
print(f"   Queries with ≥1 match: {len([q for q in query_gold if q['gold_texts']])}")
print(f"   Avg pairs per query: {len(positives)/len(train_df):.1f}")

if len(positives) < 100:
    print("   ❌ Too few positive pairs — check citation format in laws_de.csv")
    sys.exit(1)

# ============================================================================
# STEP 3: MINE HARD NEGATIVES
# ============================================================================
print("\n3. Mining hard negatives using current BGE-M3...")
print("   (Negatives = top retrieved articles NOT in gold set)")
print("   Loading BGE-M3 for mining...")

from FlagEmbedding import BGEM3FlagModel  # noqa: E402 - delay model load until mining

miner_model = BGEM3FlagModel(BGE_M3_PATH, use_fp16=True, devices=["cuda:0"])

# Build all article texts for mining (citation + text)
all_cit_texts = [(cit, f"{cit}: {text[:800]}") for cit, text in cit_to_text.items()]
all_cit_ids   = [c[0] for c in all_cit_texts]
all_cit_docs  = [c[1] for c in all_cit_texts]

print(f"   Encoding {len(all_cit_docs):,} articles for hard negative mining...")
# Encode in chunks to avoid OOM
chunk = 512
all_doc_embs = []
for i in tqdm(range(0, len(all_cit_docs), chunk), desc="   Encoding articles"):
    batch = all_cit_docs[i:i+chunk]
    emb   = miner_model.encode(batch, return_dense=True, batch_size=32,
                                max_length=MAX_DOC_LEN)["dense_vecs"]
    all_doc_embs.append(emb)
    gc.collect()
all_doc_embs = np.vstack(all_doc_embs).astype(np.float32)
print(f"   Article embeddings: {all_doc_embs.shape}")

# Encode unique queries
unique_queries = sorted({q["query"] for q in query_gold})
print(f"   Encoding {len(unique_queries):,} unique queries...")
q_embs_list = []
for i in tqdm(range(0, len(unique_queries), chunk), desc="   Encoding queries"):
    batch = unique_queries[i:i+chunk]
    emb   = miner_model.encode(batch, return_dense=True, batch_size=16,
                                max_length=MAX_QUERY_LEN)["dense_vecs"]
    q_embs_list.append(emb)
q_embs = np.vstack(q_embs_list).astype(np.float32)
q_idx  = {q: i for i, q in enumerate(unique_queries)}

# Build training examples with hard negatives
print("   Building triplets (query, positive, hard_negative)...")
triplets = []
hard_negatives_by_query = {}

for pos in positives:
    query     = pos["query"]
    positive  = pos["positive"]
    gold_set  = set(pos["gold_cits"])
    
    if query not in hard_negatives_by_query:
        sims = q_embs[q_idx[query]] @ all_doc_embs.T
        top_idx = top_non_gold_indices(sims, all_cit_ids, gold_set, HARD_NEG_PER_QUERY)
        hard_negatives_by_query[query] = [all_cit_docs[index] for index in top_idx]
    hard_negs = hard_negatives_by_query[query]
    
    if not hard_negs:
        # Fallback: random negative if no hard negatives found
        raise RuntimeError(f"No valid hard negatives for query: {query[:80]}")
    
    # Add one triplet per hard negative (standard practice)
    for neg in hard_negs[:2]:  # limit to 2 per positive to avoid dataset imbalance
        triplets.append({
            "anchor":   query,
            "positive": positive,
            "negative": neg,
        })

# Free mining model to reclaim VRAM
del miner_model, all_doc_embs, q_embs
gc.collect()
torch.cuda.empty_cache()

train_triplets, val_triplets = split_triplets_by_query(triplets, validation_fraction=0.05, seed=42)
print(f"   Triplets: {len(train_triplets)} train, {len(val_triplets)} val")

# Save dataset to disk (in case of crash)
dataset_path = CACHE_PATH / "bge_m3_triplets.jsonl"
with open(dataset_path, "w", encoding="utf-8") as f:
    for t in triplets:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"   Dataset saved: {dataset_path}")

# ============================================================================
# STEP 4: FINE-TUNE BGE-M3
# ============================================================================
print("\n4. Fine-tuning BGE-M3...")
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer  # noqa: E402
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss  # noqa: E402
from sentence_transformers.training_args import (  # noqa: E402
    SentenceTransformerTrainingArguments, BatchSamplers
)
from sentence_transformers.evaluation import TripletEvaluator  # noqa: E402
from datasets import Dataset  # noqa: E402

# Load BGE-M3 via SentenceTransformer (uses the same weights, different interface)
print("   Loading BGE-M3 via SentenceTransformer...")
model = SentenceTransformer(BGE_M3_PATH, device="cuda")
model.max_seq_length = MAX_DOC_LEN

train_ds = Dataset.from_list([
    {"anchor": t["anchor"], "positive": t["positive"], "negative": t["negative"]}
    for t in train_triplets
])
val_ds = Dataset.from_list([
    {"anchor": t["anchor"], "positive": t["positive"], "negative": t["negative"]}
    for t in val_triplets
])
print(f"   Dataset: {len(train_ds)} train, {len(val_ds)} val examples")

# CachedMultipleNegativesRankingLoss:
#   - Standard MNRL but splits encoding and loss computation into two mini-batches
#   - Allows larger effective batch sizes on limited VRAM
#   - mini_batch_size controls the encoding pass (separate from training batch_size)
#   - The loss is computed over ALL examples in the batch as negatives for each other
loss = CachedMultipleNegativesRankingLoss(
    model,
    mini_batch_size=8,    # encoding mini-batch — fits in 12GB
)

# Evaluator: measures triplet accuracy (is positive closer to anchor than negative?)
evaluator = TripletEvaluator(
    anchors   = [t["anchor"]   for t in val_triplets[:200]],
    positives = [t["positive"] for t in val_triplets[:200]],
    negatives = [t["negative"] for t in val_triplets[:200]],
    name      = "swiss-legal-val",
)

n_steps = len(train_ds) * NUM_EPOCHS // (BATCH_SIZE * GRAD_ACCUM)
print(f"\n   Training: {NUM_EPOCHS} epochs, ~{n_steps} optimizer steps")
print(f"   Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"   GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

args = SentenceTransformerTrainingArguments(
    output_dir                  = str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = LEARNING_RATE,
    warmup_ratio                = 0.1,
    lr_scheduler_type           = "cosine",
    fp16                        = True,
    batch_sampler               = BatchSamplers.NO_DUPLICATES,
    eval_strategy               = "epoch",
    save_strategy               = "no",
    load_best_model_at_end      = False,
    metric_for_best_model       = "swiss-legal-val_cosine_accuracy",
    greater_is_better           = True,
    logging_steps               = 20,
    seed                        = 42,
    report_to                   = "none",
    dataloader_pin_memory       = False,
)

trainer = SentenceTransformerTrainer(
    model         = model,
    args          = args,
    train_dataset = train_ds,
    eval_dataset  = val_ds,
    loss          = loss,
    evaluator     = evaluator,
)

trainer.train()

# ============================================================================
# STEP 5: EVALUATE
# ============================================================================
print("\n5. Evaluating fine-tuned model...")
eval_result = evaluator(model)
acc = eval_result.get("swiss-legal-val_cosine_accuracy", eval_result.get("cosine_accuracy", list(eval_result.values())[0]))
print(f"   Triplet cosine accuracy: {acc:.4f}")
print("   Triplet accuracy is an internal diagnostic; verify retrieval and F1 separately.")

# ============================================================================
# STEP 6: SAVE
# ============================================================================
print("\n6. Saving fine-tuned model...")
save_path = str(OUTPUT_DIR / "model")
model.save(save_path)
sz = sum(f.stat().st_size for f in Path(save_path).rglob("*") if f.is_file())
print(f"   Saved: {save_path} ({sz/1024**2:.0f} MB)")

del model, trainer
gc.collect()
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("  BGE-M3 FINE-TUNING COMPLETE")
print("=" * 60)
print(f"  Model: {save_path}")
print(f"  Triplet accuracy: {acc:.4f}")
print()
print("  NEXT STEP: Re-encode the corpus with the fine-tuned model.")
print("  Run: python rebuild_embeddings.py")
print()
print("  Then V41 will automatically use the improved embeddings.")
print("=" * 60)

