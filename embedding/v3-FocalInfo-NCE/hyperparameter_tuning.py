"""
Hyperparameter Tuning for Focal-InfoNCE

Grid search over:
- Temperature (tau): [0.03, 0.05, 0.07]
- Margin (m): [0.1, 0.25, 0.5]
- Gamma_pos: [0.5, 1.0, 2.0]
- Gamma_neg: [0.5, 1.0, 2.0]

Để tìm optimal hyperparameters cho DOM retrieval task
"""

import json
import os
import math
from typing import List, Dict, Tuple
from itertools import product
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    models,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.evaluation import SentenceEvaluator

from focal_infonce_loss import SimplifiedFocalInfoNCELoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"
USE_BGE_INSTRUCTION = True
BGE_QUERY_INSTRUCTION = "Represent this web action for retrieving relevant DOM elements: "

TRAIN_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_train_28160.json"
EVAL_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_eval_2827_120neg.json"

# Training config - shorter for tuning
BATCH_SIZE = 128
NUM_EPOCHS = 5  # Short training for hyperparameter search
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
SEED = 42

MAX_SEQ_LENGTH = 256
EMBEDDING_DIM = 256
RECALL_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

# Grid search space
TEMPERATURE_VALUES = [0.03, 0.05, 0.07]
MARGIN_VALUES = [0.1, 0.25, 0.5]
GAMMA_POS_VALUES = [0.5, 1.0, 2.0]
GAMMA_NEG_VALUES = [0.5, 1.0, 2.0]

# Hoặc random search với số lượng configs cố định
USE_GRID_SEARCH = False  # False = random search
NUM_RANDOM_TRIALS = 20   # Số configs random để thử


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
        
        recall_k = hit_at_k / total
        return recall_k


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_with_config(
    config: Dict,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    evaluator: SentenceEvaluator,
    trial_id: int
) -> float:
    """
    Train a model with given hyperparameters and return validation Recall@K
    """
    print(f"\n{'='*70}")
    print(f" TRIAL {trial_id}")
    print(f"{'='*70}")
    print(f"  Config: {config}")
    
    # Create model
    model = create_model()
    
    # Create loss with config
    loss = SimplifiedFocalInfoNCELoss(
        model=model,
        temperature=config['temperature'],
        margin=config['margin'],
        gamma_pos=config['gamma_pos'],
        gamma_neg=config['gamma_neg'],
    )
    
    # Training args
    steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
    warmup_steps = math.ceil(steps_per_epoch * NUM_EPOCHS * WARMUP_RATIO)
    
    output_dir = f"./tuning_trial_{trial_id}"
    
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="no",  # Don't save checkpoints during tuning
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        seed=SEED,
    )
    
    # Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    
    trainer.train()
    
    # Final evaluation
    final_recall = evaluator(model, epoch=NUM_EPOCHS)
    
    print(f"\n  Final Recall@{RECALL_K}: {final_recall:.4f}")
    
    # Cleanup
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    return final_recall


# ============================================================
# MAIN HYPERPARAMETER SEARCH
# ============================================================

def main():
    # Load data
    print("\n" + "="*70)
    print(" HYPERPARAMETER TUNING - FOCAL-INFONCE")
    print("="*70)
    
    print("\nLoading data...")
    train_samples = load_train_data(TRAIN_JSON)
    train_dataset = Dataset.from_dict({
        "sentence1": [ex.texts[0] for ex in train_samples],
        "sentence2": [ex.texts[1] for ex in train_samples],
    })
    
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    
    eval_dataset = Dataset.from_dict({
        "sentence1": [item["query"] for item in eval_data],
        "sentence2": [item["positive"] for item in eval_data],
    })
    
    evaluator = RecallAtKEvaluator(eval_data, k=RECALL_K)
    
    # Generate configs
    if USE_GRID_SEARCH:
        # Grid search - all combinations
        configs = [
            {
                'temperature': t,
                'margin': m,
                'gamma_pos': gp,
                'gamma_neg': gn,
            }
            for t, m, gp, gn in product(
                TEMPERATURE_VALUES,
                MARGIN_VALUES,
                GAMMA_POS_VALUES,
                GAMMA_NEG_VALUES
            )
        ]
        print(f"\nGrid Search: {len(configs)} configurations")
    else:
        # Random search
        import random
        configs = []
        for _ in range(NUM_RANDOM_TRIALS):
            config = {
                'temperature': random.choice(TEMPERATURE_VALUES),
                'margin': random.choice(MARGIN_VALUES),
                'gamma_pos': random.choice(GAMMA_POS_VALUES),
                'gamma_neg': random.choice(GAMMA_NEG_VALUES),
            }
            configs.append(config)
        print(f"\nRandom Search: {NUM_RANDOM_TRIALS} configurations")
    
    # Run trials
    results = []
    
    for i, config in enumerate(configs, 1):
        recall = train_with_config(
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            evaluator=evaluator,
            trial_id=i
        )
        
        results.append({
            'trial_id': i,
            'config': config,
            'recall': recall
        })
        
        # Save intermediate results
        with open("tuning_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Find best config
    best_result = max(results, key=lambda x: x['recall'])
    
    print("\n" + "="*70)
    print(" HYPERPARAMETER TUNING RESULTS")
    print("="*70)
    
    # Sort by recall
    results_sorted = sorted(results, key=lambda x: x['recall'], reverse=True)
    
    print(f"\nTop 5 configurations:")
    for i, result in enumerate(results_sorted[:5], 1):
        print(f"\n{i}. Recall@{RECALL_K}: {result['recall']:.4f}")
        print(f"   Config: {result['config']}")
    
    print(f"\n{'='*70}")
    print(" BEST CONFIGURATION")
    print("="*70)
    print(f"\nRecall@{RECALL_K}: {best_result['recall']:.4f}")
    print(f"\nHyperparameters:")
    for key, value in best_result['config'].items():
        print(f"  {key}: {value}")
    
    # Save final results
    final_output = {
        'best_config': best_result,
        'all_results': results_sorted,
        'search_space': {
            'temperature': TEMPERATURE_VALUES,
            'margin': MARGIN_VALUES,
            'gamma_pos': GAMMA_POS_VALUES,
            'gamma_neg': GAMMA_NEG_VALUES,
        }
    }
    
    with open("best_hyperparameters.json", "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"   - tuning_results.json")
    print(f"   - best_hyperparameters.json")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
