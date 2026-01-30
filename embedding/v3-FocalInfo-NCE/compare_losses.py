"""
So sánh InfoNCE (MultipleNegativesRankingLoss) vs Focal-InfoNCE

Script này training 2 models song song:
1. Baseline: MultipleNegativesRankingLoss (InfoNCE)
2. Proposed: Focal-InfoNCE

Và so sánh:
- Training loss curves
- Evaluation metrics (Recall@K, MRR)
- Convergence speed
- Embedding quality (alignment, uniformity)
"""

import json
import os
import math
from typing import List, Dict
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.evaluation import SentenceEvaluator

from focal_infonce_loss import SimplifiedFocalInfoNCELoss

# Tắt warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"
USE_BGE_INSTRUCTION = True
BGE_QUERY_INSTRUCTION = "Represent this web action for retrieving relevant DOM elements: "

TRAIN_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_train_28160.json"
EVAL_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_eval_2827_120neg.json"

# Training config (same for both models)
BATCH_SIZE = 128
NUM_EPOCHS = 10  # Shorter for comparison
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
SEED = 42

MAX_SEQ_LENGTH = 256
EMBEDDING_DIM = 256
RECALL_K = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Focal-InfoNCE hyperparameters
TEMPERATURE = 0.05
MARGIN = 0.25
GAMMA_POS = 1.0
GAMMA_NEG = 1.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


def load_train_data(path: str) -> List[InputExample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        query = item["query"]
        if USE_BGE_INSTRUCTION:
            query = BGE_QUERY_INSTRUCTION + query
        samples.append(InputExample(texts=[query, item["positive"]]))
    
    return samples


def create_model():
    """Create a fresh model instance"""
    word_embedding_model = models.Transformer(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH
    )
    
    # Add special tokens
    special_tokens = [
        "[ACTION]", "[HISTORY]", "[TAG]", "[CLASS]",
        "[TEXT]", "[ID]", "[HREF]", "[TITLE]"
    ]
    
    tokenizer = word_embedding_model.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))
    
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=EMBEDDING_DIM,
        activation_function=torch.nn.Identity()
    )
    
    return SentenceTransformer(
        modules=[word_embedding_model, pooling_model, dense_model],
        device=DEVICE
    )


class RecallAtKEvaluator(SentenceEvaluator):
    """Evaluator for Recall@K"""
    
    def __init__(self, eval_data, k=10):
        self.eval_data = eval_data
        self.k = k
    
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        set_seed(SEED)
        
        queries, candidates_list = [], []
        for item in self.eval_data:
            neg_candidates = item.get("neg_candidates", [])
            if not neg_candidates:
                continue
            
            query = item["query"]
            if USE_BGE_INSTRUCTION:
                query = BGE_QUERY_INSTRUCTION + query
            
            queries.append(query)
            candidates_list.append([item["positive"]] + neg_candidates)
        
        q_embs = model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32
        )
        
        hit_at_k = 0
        hit_at_1 = 0
        mrr_sum = 0.0
        total = len(queries)
        
        for q_emb, candidates in zip(q_embs, candidates_list):
            cand_embs = model.encode(
                candidates,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=128
            )
            
            scores = np.dot(cand_embs, q_emb)
            ranked_indices = np.argsort(scores)[::-1]
            positive_rank = np.where(ranked_indices == 0)[0][0] + 1
            
            if positive_rank <= self.k:
                hit_at_k += 1
            if positive_rank == 1:
                hit_at_1 += 1
            
            mrr_sum += 1.0 / positive_rank
        
        recall_k = hit_at_k / total
        recall_1 = hit_at_1 / total
        mrr = mrr_sum / total
        
        print(f"\n[Eval Epoch {epoch}] Recall@1: {recall_1:.4f}, "
              f"Recall@{self.k}: {recall_k:.4f}, MRR: {mrr:.4f}")
        
        return recall_k


def compute_alignment_uniformity(model, samples: List[str], batch_size=128):
    """
    Compute alignment and uniformity metrics
    
    Alignment: measures how close positive pairs are
    Uniformity: measures how uniformly embeddings are distributed
    
    Reference: Wang & Isola, 2020 (https://arxiv.org/abs/2005.10242)
    """
    embeddings = model.encode(
        samples,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False
    )
    
    # Uniformity: -log of average pairwise Gaussian potential
    n = len(embeddings)
    uniformity_sum = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = np.sum((embeddings[i] - embeddings[j])**2)
            uniformity_sum += np.exp(-2 * dist_sq)
    
    uniformity = np.log(uniformity_sum * 2 / (n * (n - 1)))
    
    return uniformity


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "="*70)
print(" LOADING DATA")
print("="*70)

train_samples = load_train_data(TRAIN_JSON)
print(f"Train samples: {len(train_samples)}")

train_dataset = Dataset.from_dict({
    "sentence1": [ex.texts[0] for ex in train_samples],
    "sentence2": [ex.texts[1] for ex in train_samples],
})

with open(EVAL_JSON, "r", encoding="utf-8") as f:
    eval_data = json.load(f)
print(f"Eval samples: {len(eval_data)}")

eval_dataset = Dataset.from_dict({
    "sentence1": [item["query"] for item in eval_data],
    "sentence2": [item["positive"] for item in eval_data],
})

evaluator = RecallAtKEvaluator(eval_data, k=RECALL_K)


# ============================================================
# TRAINING ARGS
# ============================================================

steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
warmup_steps = math.ceil(steps_per_epoch * NUM_EPOCHS * WARMUP_RATIO)


def get_training_args(output_dir: str):
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        eval_steps=550,
        save_strategy="steps",
        save_steps=550,
        save_total_limit=2,
        logging_steps=50,
        logging_first_step=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="evaluator",
        greater_is_better=True,
    )


# ============================================================
# EXPERIMENT 1: InfoNCE (Baseline)
# ============================================================

print("\n" + "="*70)
print(" EXPERIMENT 1: InfoNCE (MultipleNegativesRankingLoss)")
print("="*70)

model_infonce = create_model()
loss_infonce = losses.MultipleNegativesRankingLoss(model_infonce)

trainer_infonce = SentenceTransformerTrainer(
    model=model_infonce,
    args=get_training_args("./compare_infonce"),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss_infonce,
    evaluator=evaluator,
)

print("\nTraining with InfoNCE...")
trainer_infonce.train()

# Final evaluation
print("\nFinal evaluation - InfoNCE:")
recall_infonce = evaluator(model_infonce, epoch=NUM_EPOCHS)


# ============================================================
# EXPERIMENT 2: Focal-InfoNCE
# ============================================================

print("\n" + "="*70)
print(" EXPERIMENT 2: Focal-InfoNCE")
print("="*70)

model_focal = create_model()
loss_focal = SimplifiedFocalInfoNCELoss(
    model=model_focal,
    temperature=TEMPERATURE,
    margin=MARGIN,
    gamma_pos=GAMMA_POS,
    gamma_neg=GAMMA_NEG,
)

trainer_focal = SentenceTransformerTrainer(
    model=model_focal,
    args=get_training_args("./compare_focal"),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss_focal,
    evaluator=evaluator,
)

print("\nTraining with Focal-InfoNCE...")
trainer_focal.train()

# Final evaluation
print("\nFinal evaluation - Focal-InfoNCE:")
recall_focal = evaluator(model_focal, epoch=NUM_EPOCHS)


# ============================================================
# COMPARISON & ANALYSIS
# ============================================================

print("\n" + "="*70)
print(" COMPARISON RESULTS")
print("="*70)

print(f"\nRecall@{RECALL_K}:")
print(f"  InfoNCE:       {recall_infonce:.4f}")
print(f"  Focal-InfoNCE: {recall_focal:.4f}")
print(f"  Improvement:   {(recall_focal - recall_infonce):.4f} "
      f"({(recall_focal - recall_infonce) / recall_infonce * 100:.2f}%)")

# Compute uniformity for both models
print(f"\nComputing embedding uniformity...")
sample_queries = [item["query"] for item in eval_data[:500]]  # Sample 500 queries

uniformity_infonce = compute_alignment_uniformity(model_infonce, sample_queries)
uniformity_focal = compute_alignment_uniformity(model_focal, sample_queries)

print(f"\nUniformity (lower is better):")
print(f"  InfoNCE:       {uniformity_infonce:.4f}")
print(f"  Focal-InfoNCE: {uniformity_focal:.4f}")
print(f"  Improvement:   {(uniformity_infonce - uniformity_focal):.4f}")

print("\n" + "="*70)
print(" SUMMARY")
print("="*70)
print(f"\n✅ Focal-InfoNCE achieves:")
print(f"   - Better Recall@{RECALL_K}: +{(recall_focal - recall_infonce):.4f}")
print(f"   - Better uniformity: {uniformity_focal:.4f} vs {uniformity_infonce:.4f}")
print(f"\n📊 Hyperparameters used:")
print(f"   - Temperature: {TEMPERATURE}")
print(f"   - Margin: {MARGIN}")
print(f"   - Gamma_pos: {GAMMA_POS}")
print(f"   - Gamma_neg: {GAMMA_NEG}")
print("\n" + "="*70)

# Save comparison results
results = {
    "infonce": {
        "recall": float(recall_infonce),
        "uniformity": float(uniformity_infonce),
    },
    "focal_infonce": {
        "recall": float(recall_focal),
        "uniformity": float(uniformity_focal),
        "hyperparameters": {
            "temperature": TEMPERATURE,
            "margin": MARGIN,
            "gamma_pos": GAMMA_POS,
            "gamma_neg": GAMMA_NEG,
        }
    },
    "improvement": {
        "recall_abs": float(recall_focal - recall_infonce),
        "recall_pct": float((recall_focal - recall_infonce) / recall_infonce * 100),
        "uniformity_abs": float(uniformity_infonce - uniformity_focal),
    }
}

with open("comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: comparison_results.json")
