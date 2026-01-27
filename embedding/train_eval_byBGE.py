import json
import math
import os
import random
from typing import List

import numpy as np
import torch
from datasets import Dataset
from transformers import TrainerCallback

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


 
# 1. CONFIG
 

MODEL_NAME = "BAAI/bge-small-en-v1.5"

USE_BGE_INSTRUCTION = True  
BGE_QUERY_INSTRUCTION = "Represent this web action for retrieving relevant DOM elements: "

TRAIN_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_train_28160.json"
EVAL_JSON  = "/mnt/disk2/anhnv/rr/stage1/data/stage1_eval_2827_120neg.json"

OUTPUT_DIR = "./stage1_bge-small_INSTRUCTION_1101_v3"  

BATCH_SIZE = 128
NUM_EPOCHS = 40             
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
SEED = 42

MAX_SEQ_LENGTH = 256
RECALL_K = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


 
# 2. SET SEED
 

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


 
# 3. LOAD TRAIN DATA (query, positive)
 

def load_train_data(path: str) -> List[InputExample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        # BGE: Thêm instruction prefix cho query nếu enabled
        query = item["query"]
        if USE_BGE_INSTRUCTION:
            query = BGE_QUERY_INSTRUCTION + query
        
        positive = item["positive"]
        
        texts = [query, positive]
        # if "neg_candidates" in item:
        #     texts.extend(item["neg_candidates"][:3])  # Thêm 3 hard negatives
        samples.append(InputExample(texts=texts))
    return samples


train_samples = load_train_data(TRAIN_JSON)
print(f"Train samples: {len(train_samples)}")

train_dataset = Dataset.from_dict({
    "sentence1": [ex.texts[0] for ex in train_samples],
    "sentence2": [ex.texts[1] for ex in train_samples],
})


 
# 4. LOAD EVAL DATA (query, positive, neg_candidates)
 

with open(EVAL_JSON, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"Eval samples: {len(eval_data)}")

# Tạo eval_dataset cho trainer
eval_dataset = Dataset.from_dict({
    "sentence1": [item["query"] for item in eval_data],
    "sentence2": [item["positive"] for item in eval_data],
})


 
# 5. MODEL (DUAL ENCODER) + SPECIAL TOKENS
 

# Khởi tạo transformer model
word_embedding_model = models.Transformer(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH
)

# Thêm special tokens cho DOM task
special_tokens = ["[ACTION]", "[HISTORY]", "[TAG]", "[CLASS]", "[TEXT]","[ID]","[HREF]","[TITLE]"]
print(f"\nAdding special tokens: {special_tokens}")

tokenizer = word_embedding_model.tokenizer
num_added = tokenizer.add_special_tokens(
    {"additional_special_tokens": special_tokens}
)


# Resize model embeddings để match vocab mới
word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))
print(f"Model embeddings resized to {len(tokenizer)}\n")

# Pooling layer
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

# Projection layer: 384 → 256
dense_model = models.Dense(
    in_features=pooling_model.get_sentence_embedding_dimension(),
    out_features=256,
    activation_function=torch.nn.Identity() 
)

# Sentence Transformer model
model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model, dense_model],
    device=DEVICE
)


 
# 6. LOSS (CONTRASTIVE – IN-BATCH NEGATIVES)
 

train_loss = losses.MultipleNegativesRankingLoss(model)


 
# 7. RETRIEVAL EVALUATOR (RECALL@K) - 
 

class RecallAtKEvaluator(SentenceEvaluator):
    """
    Stage-1 retrieval evaluator
    Metrics: Recall@K, MRR, Recall@1/3/5 for comprehensive evaluation
    Optimized: Batch encoding for faster evaluation
    """

    def __init__(self, eval_data, k=10, name="recall@k"):
        self.eval_data = eval_data
        self.k = k
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        # Set seed để đảm bảo reproducibility
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        
        print(f"\n[EVAL] Starting evaluation at epoch={epoch}, steps={steps}...")
        
        hit = 0
        total = 0
        
        # Additional metrics
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_5 = 0
        mrr_sum = 0.0
        
        # Batch process để nhanh hơn
        queries = []
        candidates_list = []
        valid_indices = []

        for idx, item in enumerate(self.eval_data):
            neg_candidates = item.get("neg_candidates", [])
            
            # Bỏ qua nếu không có negative candidates
            if not neg_candidates:
                continue
            
            # BGE: Thêm instruction prefix cho query nếu enabled
            query = item["query"]
            if USE_BGE_INSTRUCTION:
                query = BGE_QUERY_INSTRUCTION + query
            
            queries.append(query)
            candidates = [item["positive"]] + neg_candidates
            candidates_list.append(candidates)
            valid_indices.append(idx)
            total += 1
        
        print(f"[EVAL] Evaluating {total} samples...")

        # Encode tất cả queries một lần
        q_embs = model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Đánh giá từng query
        for i, (q_emb, candidates) in enumerate(zip(q_embs, candidates_list)):
            # Encode candidates
            cand_embs = model.encode(
                candidates,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=128
            )

            # Tính similarity scores
            scores = np.dot(cand_embs, q_emb)
            
            # Rank candidates by scores (descending)
            ranked_indices = np.argsort(scores)[::-1]  # High to low
            
            # Find position of positive (index 0)
            positive_rank = np.where(ranked_indices == 0)[0][0] + 1  # 1-indexed
            
            # Recall@K
            if positive_rank <= self.k:
                hit += 1
            
            # Additional metrics
            if positive_rank == 1:
                hit_at_1 += 1
            if positive_rank <= 3:
                hit_at_3 += 1
            if positive_rank <= 5:
                hit_at_5 += 1
            
            # MRR
            mrr_sum += 1.0 / positive_rank

        recall_k = hit / total if total > 0 else 0.0
        recall_1 = hit_at_1 / total if total > 0 else 0.0
        recall_3 = hit_at_3 / total if total > 0 else 0.0
        recall_5 = hit_at_5 / total if total > 0 else 0.0
        mrr = mrr_sum / total if total > 0 else 0.0

        print(
            f"[EVAL] Results (epoch={epoch}, steps={steps}):\n"
            f"  Recall@1:  {recall_1:.4f} ({hit_at_1}/{total})\n"
            f"  Recall@3:  {recall_3:.4f} ({hit_at_3}/{total})\n"
            f"  Recall@5:  {recall_5:.4f} ({hit_at_5}/{total})\n"
            f"  Recall@{self.k}: {recall_k:.4f} ({hit}/{total})\n"
            f"  MRR@{self.k}:   {mrr:.4f}"
        )
        print("[EVAL] Evaluation completed.\n")

        return recall_k


evaluator = RecallAtKEvaluator(
    eval_data=eval_data,
    k=RECALL_K,
    name=f"recall@{RECALL_K}"
)


 
# 7b. CALLBACK ĐỂ LƯU BEST MODEL THEO RECALL
 

class BestRecallCallback(TrainerCallback):
    """
    Callback để lưu checkpoint khi Recall@K cao nhất
    """
    def __init__(self, evaluator, output_dir):
        self.evaluator = evaluator
        self.output_dir = output_dir
        self.best_recall = 0.0
        self.best_checkpoint_path = None
    
    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        # Lấy recall score từ evaluator
        # Vì evaluator return scalar và được wrap vào 'eval_evaluator' key
        current_recall = metrics.get("eval_evaluator", 0.0)
        
        if current_recall > self.best_recall:
            self.best_recall = current_recall
            checkpoint_name = f"best_recall_{current_recall:.4f}_step_{state.global_step}"
            checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
            
            print(f"\n{'='*60}")
            print(f" NEW BEST RECALL: {current_recall:.4f}")
            print(f"   Saving checkpoint to: {checkpoint_name}")
            print(f"{'='*60}\n")
            
            # Lưu model
            model.save(checkpoint_path)
            self.best_checkpoint_path = checkpoint_path
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.best_checkpoint_path:
            print(f"\n{'='*60}")
            print(f" Training completed!")
            print(f"   Best Recall@{RECALL_K}: {self.best_recall:.4f}")
            print(f"   Best checkpoint: {self.best_checkpoint_path}")
            print(f"{'='*60}\n")


best_recall_callback = BestRecallCallback(evaluator, OUTPUT_DIR)


 
# 8. TRAINING ARGUMENTS
 

steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = math.ceil(total_steps * WARMUP_RATIO)

print(f"\nTraining info:")
print(f"  - Model: {MODEL_NAME}")
print(f"  - Starting Training with BGE-small {'ENABLED' if USE_BGE_INSTRUCTION else 'DISABLED'}")
print(f"  - Total steps: {total_steps}")
print(f"  - Steps per epoch: {steps_per_epoch}")
print(f"  - Warmup steps: {warmup_steps}")
print(f"  - Device: {DEVICE}\n")

training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=warmup_steps,
    
    # Evaluation strategy
    eval_strategy="steps",
    eval_steps=550,  
    
    # Save strategy
    save_strategy="steps",
    save_steps=550,
    save_total_limit=3,  
    
    # Load best model - dùng loss vì evaluator không return metric riêng
    load_best_model_at_end=True,
    metric_for_best_model="evaluator",
    greater_is_better=True,
    # metric_for_best_model="loss",  # Dùng loss thay vì evaluator metric
    # greater_is_better=False,  # Loss càng thấp càng tốt
    
    # Logging
    logging_steps=50,
    logging_first_step=True,
    
    # Performance
    fp16=torch.cuda.is_available(),  # Mixed precision nếu có GPU
    dataloader_num_workers=4,
    
    # Reproducibility
    seed=SEED,
)


 
# 9. TRAIN
 

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Thêm eval_dataset
    loss=train_loss,
    evaluator=evaluator,
    callbacks=[best_recall_callback], 
)



trainer.train()

print("\n" + "="*60)

print(f" Stage-1 retriever saved to: {OUTPUT_DIR}")
print("="*60)

# Final evaluation

final_score = evaluator(model, epoch=NUM_EPOCHS, steps=-1)
print(f" Final Recall@{RECALL_K}: {final_score:.4f}")
