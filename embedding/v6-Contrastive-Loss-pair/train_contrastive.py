"""
Training script với Contrastive Loss cho Stage-1 DOM Retrieval

Contrastive Loss: L = y·d² + (1-y)·max(0, m-d)²

Loss cổ điển từ Siamese Networks, khác với InfoNCE về cách hoạt động:
- InfoNCE: softmax over batch (relative comparison)
- Contrastive: pairwise distance (absolute comparison)
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    models,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.evaluation import SentenceEvaluator

from contrastive_loss import ContrastiveLoss, ContrastiveLossWithStats, BatchContrastiveLoss



# CONFIG


MODEL_NAME = "BAAI/bge-small-en-v1.5"

USE_BGE_INSTRUCTION = True
BGE_QUERY_INSTRUCTION = "Represent this web action for retrieving relevant DOM elements: "

TRAIN_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_train_14_01_43562_with_hard_neg.json"
EVAL_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_eval_2827_120neg.json"

OUTPUT_DIR = "./stage1_contrastive_bge-small"

BATCH_SIZE = 128
NUM_EPOCHS = 40
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
SEED = 42

MAX_SEQ_LENGTH = 256
EMBEDDING_DIM = 256

RECALL_K = 50

# Contrastive Loss hyperparameters
MARGIN = 1.0  # m = 1.0 (margin cho negative pairs)
DISTANCE_METRIC = 'euclidean'  # 'euclidean' hoặc 'cosine'

# Loss mode
USE_BATCH_MODE = True  # True: BatchContrastiveLoss (in-batch negatives), False: ContrastiveLoss (explicit pairs)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_STATS_VERSION = True



# DATA LOADING


def load_train_data(json_path: str, use_instruction: bool = False) -> List[InputExample]:
    """Load training data từ JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    instruction = BGE_QUERY_INSTRUCTION if use_instruction else ""
    
    for item in data:
        query = instruction + item['query']
        positive = item['positive']
        
        # Batch mode: chỉ cần positive pairs, negatives tự động tạo từ batch
        examples.append(InputExample(texts=[query, positive]))
    
    return examples


def load_eval_data(json_path: str, use_instruction: bool = False):
    """Load evaluation data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    instruction = BGE_QUERY_INSTRUCTION if use_instruction else ""
    
    # Return raw data for evaluator to process
    eval_data = []
    for item in data:
        eval_data.append({
            'query': instruction + item['query'],
            'positive': item['positive'],
            'neg_candidates': item.get('neg_candidates', item.get('negatives', []))
        })
    
    return eval_data



# EVALUATION


class RecallEvaluator(SentenceEvaluator):
    """Evaluator tính Recall@K"""
    
    def __init__(self, eval_data, recall_k=50, name="eval"):
        self.eval_data = eval_data
        self.recall_k = recall_k
        self.name = name
        
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        
        hit = 0
        total = 0
        
        # Prepare batch data
        queries = []
        candidates_list = []
        
        for item in self.eval_data:
            neg_candidates = item.get('neg_candidates', [])
            if not neg_candidates:
                continue
            
            queries.append(item['query'])
            candidates = [item['positive']] + neg_candidates
            candidates_list.append(candidates)
            total += 1
        
        if total == 0:
            print(f"\n{'='*60}")
            print(f"WARNING: No evaluation samples with negatives!")
            print(f"{'='*60}\n")
            return {"recall": 0.0}
        
        # Encode all queries (batch)
        q_embs = model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
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
            
            # Check if in top-K
            if positive_rank <= self.recall_k:
                hit += 1
        
        recall = hit / total if total > 0 else 0.0
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} - Step {steps}")
        print(f"Recall@{self.recall_k}: {recall*100:.2f}% ({hit}/{total})")
        print(f"{'='*60}\n")
        
        if output_path:
            with open(os.path.join(output_path, f"eval_results_epoch{epoch}.txt"), 'w') as f:
                f.write(f"Recall@{self.recall_k}: {recall*100:.2f}%\n")
                f.write(f"Correct: {hit}/{total}\n")
        
        return {"recall": recall}



# TRAINING CALLBACK


class LoggingCallback(TrainerCallback):
    """Callback để log statistics"""
    
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if hasattr(self.loss_fn, 'last_stats'):
            stats = self.loss_fn.last_stats
            print(f"\nStep {state.global_step}:")
            print(f"  Loss: {stats['loss']:.4f}")
            
            if 'num_positive' in stats:
                print(f"  Pos pairs: {stats['num_positive']}, Neg pairs: {stats['num_negative']}")
                print(f"  Pos distance: {stats['pos_distance_mean']:.4f} ± {stats['pos_distance_std']:.4f}")
                print(f"  Neg distance: {stats['neg_distance_mean']:.4f} ± {stats['neg_distance_std']:.4f}")
                print(f"  Neg violations: {stats['neg_violations_pct']:.1f}% (within margin)")



# MAIN TRAINING


def main():
    # Set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    print("="*60)
    print("Contrastive Loss Training")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Margin: {MARGIN}")
    print(f"Distance metric: {DISTANCE_METRIC}")
    print(f"Batch mode: {USE_BATCH_MODE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Load data
    print("\nLoading training data...")
    train_examples = load_train_data(TRAIN_JSON, USE_BGE_INSTRUCTION)
    print(f"Training examples: {len(train_examples)}")
    
    print("\nLoading evaluation data...")
    eval_data = load_eval_data(EVAL_JSON, USE_BGE_INSTRUCTION)
    print(f"Evaluation examples: {len(eval_data)}")
    
    # Debug: Check negatives
    non_empty_negs = sum(1 for item in eval_data if item.get('neg_candidates', []))
    print(f"Examples with negatives: {non_empty_negs}/{len(eval_data)}")
    if non_empty_negs > 0:
        first_item = next((item for item in eval_data if item.get('neg_candidates', [])), None)
        if first_item:
            print(f"Sample negatives count: {len(first_item['neg_candidates'])} for first example")
    
    # Create model
    print("\nInitializing model...")
    word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=EMBEDDING_DIM,
        activation_function=torch.nn.Tanh()
    )
    
    # Note: Không dùng Normalize layer vì Contrastive Loss dùng Euclidean distance
    if DISTANCE_METRIC == 'cosine':
        normalize_model = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model, normalize_model])
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    
    model.to(DEVICE)
    
    # Create loss
    if USE_BATCH_MODE:
        print("\nUsing BatchContrastiveLoss (in-batch negatives)")
        loss = BatchContrastiveLoss(model=model, margin=MARGIN, distance_metric=DISTANCE_METRIC)
    else:
        print("\nUsing ContrastiveLoss (explicit pairs)")
        if USE_STATS_VERSION:
            loss = ContrastiveLossWithStats(model=model, margin=MARGIN, distance_metric=DISTANCE_METRIC)
        else:
            loss = ContrastiveLoss(model=model, margin=MARGIN, distance_metric=DISTANCE_METRIC)
    
    # Create evaluator
    evaluator = RecallEvaluator(
        eval_data=eval_data,
        recall_k=RECALL_K,
        name="eval"
    )
    
    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=True if DEVICE == "cuda" else False,
        bf16=False,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_recall",
        greater_is_better=True,
        seed=SEED,
        dataloader_drop_last=True,
    )
    
    # Convert InputExample list to Dataset format
    train_dataset_dict = {
        "anchor": [ex.texts[0] for ex in train_examples],
        "positive": [ex.texts[1] for ex in train_examples],
    }
    train_dataset = Dataset.from_dict(train_dataset_dict)
    
    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    
    # Add callback
    if USE_STATS_VERSION and not USE_BATCH_MODE:
        trainer.add_callback(LoggingCallback(loss))
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    model.save(os.path.join(OUTPUT_DIR, "final_model"))
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_result = evaluator(model, output_path=OUTPUT_DIR)
    final_recall = final_result.get("recall", 0.0)
    print(f"\nFinal Recall@{RECALL_K}: {final_recall*100:.2f}%")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()
