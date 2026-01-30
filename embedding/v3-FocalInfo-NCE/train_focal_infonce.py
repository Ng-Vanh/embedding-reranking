"""
Training script with Focal-InfoNCE Loss for Stage-1 DOM Retrieval

Cải tiến so với train_eval_byBGE.py:
1. Sử dụng Focal-InfoNCE thay vì MultipleNegativesRankingLoss
2. Hard negative mining tự động trong loss function
3. Dropout noise aware cho positive pairs
4. Hyperparameter tuning cho margin và gamma

Tham khảo paper: https://openreview.net/pdf?id=j48JCRagwR
"""

import json
import math
import os
import random
from typing import List

import numpy as np
import torch
from datasets import Dataset
from transformers import TrainerCallback

# Tắt tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    models,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.evaluation import SentenceEvaluator

# Import Focal-InfoNCE loss
from focal_infonce_loss import FocalInfoNCELoss, SimplifiedFocalInfoNCELoss


# ============================================================
# 1. CONFIG
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE instruction prefix
USE_BGE_INSTRUCTION = True
BGE_QUERY_INSTRUCTION = "Represent this web action for retrieving relevant DOM elements: "

# Data paths
TRAIN_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_train_14_01_43562_with_hard_neg.json"
EVAL_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_eval_2827_120neg.json"

# Output directory
OUTPUT_DIR = "./stage1_focal_infonce_bge-small"

# Training hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 40
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
SEED = 42

# Model hyperparameters
MAX_SEQ_LENGTH = 256
EMBEDDING_DIM = 256  # Projection dimension

# Evaluation
RECALL_K = 50

# Focal-InfoNCE hyperparameters (theo paper)
TEMPERATURE = 0.05  # tau = 0.05 (tương đương scale = 20)
MARGIN = 0.25  # m = 0.25 (hard negative mining margin)
GAMMA_POS = 1.0  # gamma for positive pairs (dropout noise aware)
GAMMA_NEG = 1.0  # gamma for negative pairs (hard negative mining)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use simplified version for easier debugging
USE_SIMPLIFIED_LOSS = True


# ============================================================
# 2. SET SEED
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ============================================================
# 3. LOAD TRAIN DATA
# ============================================================

def load_train_data(path: str) -> List[InputExample]:
    """Load training data with query-positive pairs"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        query = item["query"]
        if USE_BGE_INSTRUCTION:
            query = BGE_QUERY_INSTRUCTION + query
        
        positive = item["positive"]
        
        # InputExample with 2 texts: [query, positive]
        samples.append(InputExample(texts=[query, positive]))
    
    return samples


print("\n" + "="*70)
print(" FOCAL-INFONCE TRAINING - STAGE 1 DOM RETRIEVAL")
print("="*70)

print("\nLoading train data...")
train_samples = load_train_data(TRAIN_JSON)
print(f"  Train samples: {len(train_samples)}")

train_dataset = Dataset.from_dict({
    "sentence1": [ex.texts[0] for ex in train_samples],
    "sentence2": [ex.texts[1] for ex in train_samples],
})


# ============================================================
# 4. LOAD EVAL DATA
# ============================================================

print("\nLoading eval data...")
with open(EVAL_JSON, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"  Eval samples: {len(eval_data)}")

eval_dataset = Dataset.from_dict({
    "sentence1": [item["query"] for item in eval_data],
    "sentence2": [item["positive"] for item in eval_data],
})


# ============================================================
# 5. MODEL SETUP
# ============================================================

print("\n" + "-"*70)
print(" MODEL SETUP")
print("-"*70)

# Transformer model
word_embedding_model = models.Transformer(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH
)

# Add special tokens for DOM elements
special_tokens = [
    "[ACTION]", "[HISTORY]", "[TAG]", "[CLASS]", 
    "[TEXT]", "[ID]", "[HREF]", "[TITLE]"
]

print(f"\nAdding special tokens: {special_tokens}")

tokenizer = word_embedding_model.tokenizer
num_added = tokenizer.add_special_tokens(
    {"additional_special_tokens": special_tokens}
)

print(f"  Added {num_added} special tokens")
print(f"  Vocab size: {len(tokenizer) - num_added} → {len(tokenizer)}")

# Resize embeddings
word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))

# Pooling layer (mean pooling)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

# Dense projection layer
dense_model = models.Dense(
    in_features=pooling_model.get_sentence_embedding_dimension(),
    out_features=EMBEDDING_DIM,
    activation_function=torch.nn.Identity()
)

# Build SentenceTransformer
model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model, dense_model],
    device=DEVICE
)

print(f"\nModel architecture:")
print(f"  - Base model: {MODEL_NAME}")
print(f"  - Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  - Embedding dimension: {EMBEDDING_DIM}")
print(f"  - Device: {DEVICE}")


# ============================================================
# 6. LOSS FUNCTION - FOCAL-INFONCE
# ============================================================

print("\n" + "-"*70)
print(" LOSS FUNCTION: Focal-InfoNCE")
print("-"*70)

if USE_SIMPLIFIED_LOSS:
    train_loss = SimplifiedFocalInfoNCELoss(
        model=model,
        temperature=TEMPERATURE,
        margin=MARGIN,
        gamma_pos=GAMMA_POS,
        gamma_neg=GAMMA_NEG,
    )
    loss_name = "SimplifiedFocalInfoNCELoss"
else:
    train_loss = FocalInfoNCELoss(
        model=model,
        scale=1.0/TEMPERATURE,
        margin=MARGIN,
        gamma_pos=GAMMA_POS,
        gamma_neg=GAMMA_NEG,
    )
    loss_name = "FocalInfoNCELoss"

print(f"\nLoss: {loss_name}")
print(f"  - Temperature (tau): {TEMPERATURE}")
print(f"  - Margin (m): {MARGIN}")
print(f"  - Gamma_pos: {GAMMA_POS}")
print(f"  - Gamma_neg: {GAMMA_NEG}")

print("\nFocal-InfoNCE features:")
print("  ✓ Hard negative mining: focus on high-similarity negatives")
print("  ✓ Dropout noise aware: reduce penalty for low-similarity positives")
print("  ✓ Improved alignment & uniformity")


# ============================================================
# 7. EVALUATOR - RECALL@K
# ============================================================

class RecallAtKEvaluator(SentenceEvaluator):
    """
    Retrieval evaluator with Recall@K, MRR, and multiple recall metrics
    """

    def __init__(self, eval_data, k=10, name="recall@k"):
        self.eval_data = eval_data
        self.k = k
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        set_seed(SEED)
        
        print(f"\n{'='*70}")
        print(f" EVALUATION - Epoch {epoch}, Steps {steps}")
        print(f"{'='*70}")
        
        total = 0
        hit_at_1, hit_at_3, hit_at_5, hit_at_k = 0, 0, 0, 0
        mrr_sum = 0.0
        
        # Prepare data
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
            total += 1
        
        print(f"\nEvaluating on {total} samples...")

        # Encode queries in batch
        q_embs = model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Evaluate each query
        for q_emb, candidates in zip(q_embs, candidates_list):
            # Encode candidates
            cand_embs = model.encode(
                candidates,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=128
            )

            # Compute scores and rank
            scores = np.dot(cand_embs, q_emb)
            ranked_indices = np.argsort(scores)[::-1]
            
            # Find positive rank (index 0)
            positive_rank = np.where(ranked_indices == 0)[0][0] + 1
            
            # Update metrics
            if positive_rank == 1:
                hit_at_1 += 1
            if positive_rank <= 3:
                hit_at_3 += 1
            if positive_rank <= 5:
                hit_at_5 += 1
            if positive_rank <= self.k:
                hit_at_k += 1
            
            mrr_sum += 1.0 / positive_rank

        # Compute final metrics
        recall_1 = hit_at_1 / total
        recall_3 = hit_at_3 / total
        recall_5 = hit_at_5 / total
        recall_k = hit_at_k / total
        mrr = mrr_sum / total

        print(f"\nResults:")
        print(f"  Recall@1:  {recall_1:.4f} ({hit_at_1}/{total})")
        print(f"  Recall@3:  {recall_3:.4f} ({hit_at_3}/{total})")
        print(f"  Recall@5:  {recall_5:.4f} ({hit_at_5}/{total})")
        print(f"  Recall@{self.k}: {recall_k:.4f} ({hit_at_k}/{total})")
        print(f"  MRR@{self.k}:   {mrr:.4f}")
        print(f"{'='*70}\n")

        return recall_k


evaluator = RecallAtKEvaluator(
    eval_data=eval_data,
    k=RECALL_K,
    name=f"recall@{RECALL_K}"
)


# ============================================================
# 8. CALLBACK - SAVE BEST MODEL
# ============================================================

class BestRecallCallback(TrainerCallback):
    """Save checkpoint when Recall@K improves"""
    
    def __init__(self, evaluator, output_dir):
        self.evaluator = evaluator
        self.output_dir = output_dir
        self.best_recall = 0.0
        self.best_checkpoint_path = None
    
    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        current_recall = metrics.get("eval_evaluator", 0.0)
        
        if current_recall > self.best_recall:
            self.best_recall = current_recall
            checkpoint_name = f"best_recall_{current_recall:.4f}_step_{state.global_step}"
            checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
            
            print(f"\n{' '*30}")
            print(f" NEW BEST RECALL: {current_recall:.4f}")
            print(f" Saving to: {checkpoint_name}")
            print(f"{' '*30}\n")
            
            model.save(checkpoint_path)
            self.best_checkpoint_path = checkpoint_path
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.best_checkpoint_path:
            print(f"\n{'='*70}")
            print(f" TRAINING COMPLETED")
            print(f"{'='*70}")
            print(f"  Best Recall@{RECALL_K}: {self.best_recall:.4f}")
            print(f"  Best checkpoint: {self.best_checkpoint_path}")
            print(f"{'='*70}\n")


best_recall_callback = BestRecallCallback(evaluator, OUTPUT_DIR)


# ============================================================
# 9. TRAINING ARGUMENTS
# ============================================================

steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = math.ceil(total_steps * WARMUP_RATIO)

print("\n" + "-"*70)
print(" TRAINING CONFIGURATION")
print("-"*70)
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Warmup ratio: {WARMUP_RATIO}")
print(f"  Total steps: {total_steps}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Steps per epoch: {steps_per_epoch}")
print("-"*70)

training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=warmup_steps,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=1710,
    
    # Checkpointing
    save_strategy="steps",
    save_steps=1710,
    save_total_limit=3,
    
    # Best model selection
    load_best_model_at_end=True,
    metric_for_best_model="evaluator",
    greater_is_better=True,
    
    # Logging
    logging_steps=50,
    logging_first_step=True,
    
    # Performance
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    
    # Reproducibility
    seed=SEED,
)


# ============================================================
# 10. TRAIN
# ============================================================

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=evaluator,
    callbacks=[best_recall_callback],
)

print("\n" + "="*70)
print(" STARTING TRAINING WITH FOCAL-INFONCE")
print("="*70 + "\n")

trainer.train()

print("\n" + "="*70)
print(" TRAINING COMPLETED")
print("="*70)
print(f"  Model saved to: {OUTPUT_DIR}")
print("="*70)

# Final evaluation
print("\n\nRunning final evaluation...")
final_score = evaluator(model, epoch=NUM_EPOCHS, steps=-1)
print(f"\nFinal Recall@{RECALL_K}: {final_score:.4f}")
print("\n" + "="*70)
