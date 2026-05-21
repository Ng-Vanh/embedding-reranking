"""
Paper: https://arxiv.org/pdf/2412.17364
"""

import json
import math
import os
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import TrainerCallback

# Tắt tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.evaluation import SentenceEvaluator


#═══════════════════════════════════════════════════════════════════════════
# 1. CONFIG
#═══════════════════════════════════════════════════════════════════════════

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE instruction prefix
USE_BGE_INSTRUCTION = True
BGE_QUERY_INSTRUCTION = "Represent this web action for retrieving relevant DOM elements: "

TRAIN_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_train_14_01_43562.json"
EVAL_JSON  = "/mnt/disk2/anhnv/rr/stage1/data/stage1_eval_2827_120neg.json"

OUTPUT_DIR = "./stage1_bge-small_ANCE_CLP_v2"

# Training hyperparameters
BATCH_SIZE = 80  
NUM_EPOCHS = 40
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
SEED = 42

# Model architecture
MAX_SEQ_LENGTH = 256
EMBEDDING_DIM = 256

# ANCE & CLP config
NUM_HARD_NEGATIVES = 12  # Số hard negatives từ neg_candidates
USE_CLP = True          # Bật Contrastive Learning Penalty
CLP_LAMBDA = 0.1        # Trọng số của CLP penalty (0.05-0.2 theo paper)

# Evaluation
RECALL_K = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#═══════════════════════════════════════════════════════════════════════════
# 2. SET SEED
#═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)


#═══════════════════════════════════════════════════════════════════════════
# 3. CONTRASTIVE LEARNING PENALTY LOSS (CLP)
#═══════════════════════════════════════════════════════════════════════════

class ContrastiveLearningPenaltyLoss(nn.Module):
    """
    Advanced loss function combining:
    1. Standard Contrastive Loss (MultipleNegativesRankingLoss)
    2. Contrastive Learning Penalty (CLP)
    
    CLP Rationale:
    --------------
    - Khi đẩy xa negative doc khỏi query hiện tại
    - Phải đảm bảo không đẩy xa negative khỏi positive queries của nó
    - Bảo toàn cấu trúc ngữ nghĩa toàn cục
    
    Loss Formula:
    -------------
    L_total = L_contrastive + λ * L_penalty
    
    L_contrastive = -log(exp(sim(q, p+)) / Σ exp(sim(q, p)))
    L_penalty = Σ max(0, sim(n, q_related) - sim(n, q_current))
    
    where:
    - q: current query
    - p+: positive document
    - n: negative document  
    - q_related: queries that are actually related to negative n
    """
    
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, 
                 penalty_lambda: float = 0.1):
        super().__init__()
        self.model = model
        self.scale = scale
        self.penalty_lambda = penalty_lambda
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, sentence_features: List[Dict[str, torch.Tensor]], labels: torch.Tensor):
        """
        Forward pass với CLP
        
        Args:
            sentence_features: List[Dict] với keys 'input_ids', 'attention_mask'
                - sentence_features[0]: queries (batch_size)
                - sentence_features[1]: positives (batch_size)
                - sentence_features[2:]: hard negatives (batch_size * num_neg)
            labels: Not used, placeholder for compatibility
        
        Returns:
            loss: Combined contrastive + penalty loss
        """
        # Encode queries and documents
        reps = [self.model(sentence_feature)['sentence_embedding'] 
                for sentence_feature in sentence_features]
        
        embeddings_query = reps[0]      # (batch_size, embed_dim)
        embeddings_positive = reps[1]    # (batch_size, embed_dim)
        
        # Hard negatives (if any)
        if len(reps) > 2:
            embeddings_negatives = torch.cat(reps[2:], dim=0)  # (batch_size * num_neg, embed_dim)
        else:
            embeddings_negatives = None
        
        # ============================================================
        # PART 1: Standard Contrastive Loss (In-Batch + Hard Negatives)
        # ============================================================
        
        batch_size = embeddings_query.size(0)
        
        # Normalize embeddings
        embeddings_query = F.normalize(embeddings_query, p=2, dim=1)
        embeddings_positive = F.normalize(embeddings_positive, p=2, dim=1)
        
        # Similarity: query × positive
        scores_pos = torch.sum(embeddings_query * embeddings_positive, dim=-1) * self.scale  # (batch_size,)
        
        # Similarity: query × all positives (in-batch negatives)
        scores_all = torch.mm(embeddings_query, embeddings_positive.t()) * self.scale  # (batch_size, batch_size)
        
        # Add hard negatives if available
        if embeddings_negatives is not None:
            embeddings_negatives = F.normalize(embeddings_negatives, p=2, dim=1)
            num_negatives = embeddings_negatives.size(0) // batch_size
            
            # Reshape negatives: (batch_size, num_negatives, embed_dim)
            embeddings_negatives_reshaped = embeddings_negatives.view(
                batch_size, num_negatives, -1
            )
            
            # Similarity: query × hard negatives
            # (batch_size, 1, embed_dim) × (batch_size, num_negatives, embed_dim)
            scores_hard_neg = torch.bmm(
                embeddings_query.unsqueeze(1),  # (batch_size, 1, embed_dim)
                embeddings_negatives_reshaped.transpose(1, 2)  # (batch_size, embed_dim, num_negatives)
            ).squeeze(1) * self.scale  # (batch_size, num_negatives)
            
            # Concatenate all scores: [positive | in-batch negatives | hard negatives]
            scores_all = torch.cat([scores_all, scores_hard_neg], dim=1)  # (batch_size, batch_size + num_neg)
        
        # Labels: positive is always at index i (diagonal)
        labels = torch.arange(batch_size, device=scores_all.device)
        
        # Cross-entropy loss
        loss_contrastive = self.cross_entropy_loss(scores_all, labels)
        
        # ============================================================
        # PART 2: Contrastive Learning Penalty (CLP)
        # ============================================================
        
        loss_penalty = torch.tensor(0.0, device=embeddings_query.device)
        
        if self.penalty_lambda > 0 and embeddings_negatives is not None:
            """
            CLP Idea:
            - Negative doc n được đẩy xa query q_current
            - Nhưng n có thể là positive của query khác q_related
            - Penalty nếu: sim(n, q_related) giảm so với baseline
            
            Implementation:
            - Trong batch, giả định: mỗi negative có thể liên quan đến 1 số queries khác
            - Penalty = max(0, margin - (sim(n, q_related) - sim(n, q_current)))
            - Chỉ áp dụng khi negative THỰC SỰ gần với query khác (top-K)
            """
            
            # Tính similarity: negatives × all queries
            # (batch_size * num_neg, embed_dim) × (embed_dim, batch_size)
            sim_neg_queries = torch.mm(
                embeddings_negatives,
                embeddings_query.t()
            )  # (batch_size * num_neg, batch_size)
            
            # Reshape: (batch_size, num_negatives, batch_size)
            sim_neg_queries = sim_neg_queries.view(batch_size, num_negatives, batch_size)
            
            # For each negative, find its max similarity with other queries (not current)
            for i in range(batch_size):
                # Lấy negatives của query i
                neg_sims = sim_neg_queries[i]  # (num_negatives, batch_size)
                
                # Similarity với query hiện tại i
                sim_with_current = neg_sims[:, i]  # (num_negatives,)
                
                # Similarity với các queries khác (mask out current query)
                mask = torch.ones(batch_size, dtype=torch.bool, device=neg_sims.device)
                mask[i] = False
                sim_with_others = neg_sims[:, mask]  # (num_negatives, batch_size-1)
                
                # Max similarity với query khác (potential related query)
                max_sim_others, _ = sim_with_others.max(dim=1)  # (num_negatives,)
                
                # Penalty: nếu negative gần query khác hơn query hiện tại
                # → Đang đẩy xa negative khỏi query mà nó có thể liên quan
                penalty = torch.relu(max_sim_others - sim_with_current)  # (num_negatives,)
                loss_penalty += penalty.mean()
            
            loss_penalty = loss_penalty / batch_size
        
        # ============================================================
        # TOTAL LOSS
        # ============================================================
        
        loss_total = loss_contrastive + self.penalty_lambda * loss_penalty
        
        # Logging
        if random.random() < 0.01:  # Log 1% samples để không spam
            print(f"[LOSS] Contrastive: {loss_contrastive.item():.4f} | "
                  f"Penalty: {loss_penalty.item():.4f} | "
                  f"Total: {loss_total.item():.4f}")
        
        return loss_total


#═══════════════════════════════════════════════════════════════════════════
# 4. LOAD TRAIN DATA WITH HARD NEGATIVES (ANCE)
#═══════════════════════════════════════════════════════════════════════════

def load_train_data_with_hard_negatives(
    path: str,
    num_hard_negatives: int = 5
) -> List[InputExample]:
    """
    Load training data với hard negatives từ neg_candidates
    
    ANCE Strategy:
    --------------
    - Mỗi sample có: query, positive, K hard negatives
    - Hard negatives = top-K documents gần query nhưng sai (từ dense retrieval)
    - Giàu thông tin, gradient mạnh, embedding phân biệt tốt hơn
    
    Data Format:
    ------------
    {
        "query": "...",
        "positive": "...",
        "neg_candidates": ["neg1", "neg2", ...],  # Hard negatives từ retrieval
    }
    
    Returns:
    --------
    List[InputExample]:
        texts = [query, positive, neg1, neg2, ..., negK]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    samples_with_hard_neg = 0
    samples_without_hard_neg = 0
    
    for item in data:
        # Add BGE instruction prefix
        query = item["query"]
        if USE_BGE_INSTRUCTION:
            query = BGE_QUERY_INSTRUCTION + query
        
        positive = item["positive"]
        
        # Get hard negatives from neg_candidates
        neg_candidates = item.get("neg_candidates", [])
        
        if neg_candidates:
            # Sample K hard negatives
            num_available = len(neg_candidates)
            num_sample = min(num_hard_negatives, num_available)
            
            # Random sample để tăng diversity
            hard_negatives = random.sample(neg_candidates, num_sample)
            
            # InputExample: [query, positive, neg1, neg2, ..., negK]
            texts = [query, positive] + hard_negatives
            samples.append(InputExample(texts=texts))
            samples_with_hard_neg += 1
        else:
            # Fallback: No hard negatives → use in-batch negatives only
            texts = [query, positive]
            samples.append(InputExample(texts=texts))
            samples_without_hard_neg += 1
    
    print(f"\n[DATA] Loaded {len(samples)} training samples:")
    print(f"  - With hard negatives: {samples_with_hard_neg}")
    print(f"  - Without hard negatives (in-batch only): {samples_without_hard_neg}")
    
    return samples


#═══════════════════════════════════════════════════════════════════════════
# 5. LOAD DATA
#═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print(" LOADING TRAINING DATA WITH ANCE (HARD NEGATIVES)")
print("="*80)

train_samples = load_train_data_with_hard_negatives(
    TRAIN_JSON,
    num_hard_negatives=NUM_HARD_NEGATIVES
)

print(f"\n[INFO] Total train samples: {len(train_samples)}")

# Convert to HuggingFace Dataset
# Note: Mỗi sample có thể có số lượng texts khác nhau (2 + num_hard_neg)
train_data_dict = {
    "query": [],
    "positive": [],
    "hard_negatives": []
}

for ex in train_samples:
    train_data_dict["query"].append(ex.texts[0])
    train_data_dict["positive"].append(ex.texts[1])
    
    # Hard negatives (if any)
    if len(ex.texts) > 2:
        train_data_dict["hard_negatives"].append(ex.texts[2:])
    else:
        train_data_dict["hard_negatives"].append([])

train_dataset = Dataset.from_dict(train_data_dict)

# Load eval data
print("\n" + "="*80)
print(" LOADING EVALUATION DATA")
print("="*80)

with open(EVAL_JSON, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"\n[INFO] Total eval samples: {len(eval_data)}")

eval_dataset = Dataset.from_dict({
    "sentence1": [item["query"] for item in eval_data],
    "sentence2": [item["positive"] for item in eval_data],
})


#═══════════════════════════════════════════════════════════════════════════
# 6. MODEL ARCHITECTURE
#═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print(" BUILDING MODEL")
print("="*80)

# Transformer encoder
word_embedding_model = models.Transformer(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH
)

# Add special tokens
special_tokens = [
    "[ACTION]", "[HISTORY]", "[TAG]", "[CLASS]", 
    "[TEXT]", "[ID]", "[HREF]", "[TITLE]"
]

print(f"\n[MODEL] Adding special tokens: {special_tokens}")
tokenizer = word_embedding_model.tokenizer
num_added = tokenizer.add_special_tokens(
    {"additional_special_tokens": special_tokens}
)

print(f"[MODEL] Added {num_added} tokens | Vocab size: {len(tokenizer)}")
word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))

# Pooling layer
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

# Dense projection: 384 → 256
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

print(f"\n[MODEL] Architecture:")
print(f"  - Base model: {MODEL_NAME}")
print(f"  - Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  - Embedding dimension: {EMBEDDING_DIM}")
print(f"  - Device: {DEVICE}")


#═══════════════════════════════════════════════════════════════════════════
# 7. LOSS FUNCTION
#═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print(" LOSS FUNCTION")
print("="*80)

if USE_CLP:
    train_loss = ContrastiveLearningPenaltyLoss(
        model=model,
        scale=20.0,
        penalty_lambda=CLP_LAMBDA
    )
    print(f"\n[LOSS] Using CLP (Contrastive Learning Penalty)")
    print(f"  - Penalty lambda: {CLP_LAMBDA}")
    print(f"  - Scale: 20.0")
else:
    # Fallback to standard Multiple Negatives Ranking Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    print(f"\n[LOSS] Using standard MultipleNegativesRankingLoss")

print(f"  - Hard negatives per sample: {NUM_HARD_NEGATIVES}")


#═══════════════════════════════════════════════════════════════════════════
# 8. RETRIEVAL EVALUATOR (RECALL@K)
#═══════════════════════════════════════════════════════════════════════════

class RecallAtKEvaluator(SentenceEvaluator):
    """
    Optimized Recall@K evaluator cho Stage-1 retrieval
    
    Metrics:
    --------
    - Recall@1, Recall@3, Recall@5, Recall@K
    - MRR (Mean Reciprocal Rank)
    """
    
    def __init__(self, eval_data, k=10, name="recall@k"):
        self.eval_data = eval_data
        self.k = k
        self.name = name
    
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        print(f"\n{'='*80}")
        print(f" EVALUATION: Epoch={epoch}, Steps={steps}")
        print("="*80)
        
        # Set seed
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        
        hit = 0
        total = 0
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_5 = 0
        mrr_sum = 0.0
        
        # Prepare batch data
        queries = []
        candidates_list = []
        
        for item in self.eval_data:
            neg_candidates = item.get("neg_candidates", [])
            if not neg_candidates:
                continue
            
            query = item["query"]
            if USE_BGE_INSTRUCTION:
                query = BGE_QUERY_INSTRUCTION + query
            
            queries.append(query)
            candidates = [item["positive"]] + neg_candidates
            candidates_list.append(candidates)
            total += 1
        
        print(f"[EVAL] Evaluating {total} samples...")
        
        # Encode all queries (batch)
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
            
            # Compute similarities
            scores = np.dot(cand_embs, q_emb)
            ranked_indices = np.argsort(scores)[::-1]
            
            # Find positive position (always at index 0)
            positive_rank = np.where(ranked_indices == 0)[0][0] + 1
            
            # Update metrics
            if positive_rank <= self.k:
                hit += 1
            if positive_rank == 1:
                hit_at_1 += 1
            if positive_rank <= 3:
                hit_at_3 += 1
            if positive_rank <= 5:
                hit_at_5 += 1
            
            mrr_sum += 1.0 / positive_rank
        
        # Calculate metrics
        recall_k = hit / total if total > 0 else 0.0
        recall_1 = hit_at_1 / total if total > 0 else 0.0
        recall_3 = hit_at_3 / total if total > 0 else 0.0
        recall_5 = hit_at_5 / total if total > 0 else 0.0
        mrr = mrr_sum / total if total > 0 else 0.0
        
        print(f"\n[RESULTS]")
        print(f"  Recall@1:  {recall_1:.4f} ({hit_at_1}/{total})")
        print(f"  Recall@3:  {recall_3:.4f} ({hit_at_3}/{total})")
        print(f"  Recall@5:  {recall_5:.4f} ({hit_at_5}/{total})")
        print(f"  Recall@{self.k}: {recall_k:.4f} ({hit}/{total})")
        print(f"  MRR@{self.k}:   {mrr:.4f}")
        print("="*80 + "\n")
        
        return recall_k


evaluator = RecallAtKEvaluator(
    eval_data=eval_data,
    k=RECALL_K,
    name=f"recall@{RECALL_K}"
)


#═══════════════════════════════════════════════════════════════════════════
# 9. CALLBACK: SAVE BEST MODEL
#═══════════════════════════════════════════════════════════════════════════

class BestRecallCallback(TrainerCallback):
    """Callback để lưu checkpoint với Recall@K cao nhất"""
    
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
            
            print(f"\n{'='*80}")
            print(f"  NEW BEST RECALL: {current_recall:.4f}")
            print(f"    Saving checkpoint: {checkpoint_name}")
            print(f"{'='*80}\n")
            
            model.save(checkpoint_path)
            self.best_checkpoint_path = checkpoint_path
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.best_checkpoint_path:
            print(f"\n{'='*80}")
            print(f"  TRAINING COMPLETED")
            print(f"    Best Recall@{RECALL_K}: {self.best_recall:.4f}")
            print(f"    Best checkpoint: {self.best_checkpoint_path}")
            print(f"{'='*80}\n")


best_recall_callback = BestRecallCallback(evaluator, OUTPUT_DIR)


#═══════════════════════════════════════════════════════════════════════════
# 10. TRAINING ARGUMENTS
#═══════════════════════════════════════════════════════════════════════════

steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = math.ceil(total_steps * WARMUP_RATIO)

print("\n" + "="*80)
print(" TRAINING CONFIGURATION")
print("="*80)
print(f"\n[CONFIG]")
print(f"  Model: {MODEL_NAME}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Method: ANCE + CLP (λ={CLP_LAMBDA})")
print(f"  Hard negatives per sample: {NUM_HARD_NEGATIVES}")
print(f"\n[TRAINING]")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Total steps: {total_steps}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Device: {DEVICE}")
print(f"  FP16: {torch.cuda.is_available()}")

training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=warmup_steps,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=2140,
    
    # Checkpoint
    save_strategy="steps",
    save_steps=2140,
    save_total_limit=3,
    
    # Best model
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


#═══════════════════════════════════════════════════════════════════════════
# 11. TRAINER
#═══════════════════════════════════════════════════════════════════════════

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=evaluator,
    callbacks=[best_recall_callback],
)


#═══════════════════════════════════════════════════════════════════════════
# 12. TRAIN
#═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print(" STARTING TRAINING")
print("="*80 + "\n")

trainer.train()

print("\n" + "="*80)
print(" TRAINING COMPLETED")
print("="*80)

# Final evaluation
print("\n[INFO] Running final evaluation...")
final_score = evaluator(model, epoch=NUM_EPOCHS, steps=-1)
print(f"\n[FINAL] Recall@{RECALL_K}: {final_score:.4f}")

print(f"\n[INFO] Model saved to: {OUTPUT_DIR}")
print("="*80 + "\n")
