"""
Test/Inference script for trained BGE retrieval model
- Load trained model
- Evaluate on test data
- Predict similarity scores between query and nodes
"""

import json
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm



# CONFIG


# Model path - thay đổi theo model bạn muốn test
MODEL_PATH = "/mnt/disk2/anhnv/rr/stage1/v6-Contrastive-Loss-pair/stage1_contrastive_bge-small/final_model"  
# MODEL_PATH = "BAAI/bge-small-en-v1.5"  

# amazon_queries arxiv_queries awward_queries bbc_news_queries behance_queries economist_queries 
# health_queries neflix_queries pinterest_queries sportify_queries stackover_queries twitch_queries
# Test data path
TEST_JSON = "/mnt/disk2/anhnv/rr/stage1/data/processed_test/twitch_queries.json"


# BGE instruction (phải giống lúc train)
USE_BGE_INSTRUCTION = True  # Set True nếu model được train với instruction
BGE_QUERY_INSTRUCTION = "Represent this web action for retrieving relevant DOM elements: "

# Evaluation settings
BATCH_SIZE = 1000
RECALL_K_VALUES = [1, 3, 5, 10, 20,30,40,50]



# 1. LOAD MODEL


def load_model(model_path: str) -> SentenceTransformer:
    if os.path.exists(model_path):
        print(" LOCAL model path")
        model = SentenceTransformer(model_path)
    else:
        print(" HuggingFace model name")
        model = SentenceTransformer(model_path)

    print(f" Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f" Max sequence length: {model.max_seq_length}")

    return model




# 2. PREDICT FUNCTIONS


def predict_similarity(
    model: SentenceTransformer,
    query: str,
    node: str,
    add_instruction: bool = False,
    instruction: str = BGE_QUERY_INSTRUCTION
) -> float:
    """
    Tính similarity score giữa 1 query và 1 node
    
    Args:
        model: Trained model
        query: Query string (action + history)
        node: Node representation (DOM element)
        add_instruction: Có thêm instruction prefix không
        instruction: Instruction string
    
    Returns:
        Similarity score (0-1, càng cao càng giống)
    """
    # Thêm instruction nếu cần
    if add_instruction:
        query = instruction + query
    
    # Encode
    query_emb = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    node_emb = model.encode(
        node,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    # Cosine similarity (với normalized embeddings = dot product)
    similarity = float(np.dot(query_emb, node_emb))
    
    return similarity


def predict_batch(
    model: SentenceTransformer,
    query: str,
    nodes: List[str],
    add_instruction: bool = False,
    instruction: str = BGE_QUERY_INSTRUCTION,
    return_scores: bool = True
) -> Dict:
    """
    Rank multiple nodes cho 1 query
    
    Args:
        model: Trained model
        query: Query string
        nodes: List of node representations
        add_instruction: Có thêm instruction prefix không
        instruction: Instruction string
        return_scores: Return similarity scores
    
    Returns:
        Dict với ranked results
    """
    # Thêm instruction nếu cần
    if add_instruction:
        query = instruction + query
    
    # Encode query
    query_emb = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    # Encode all nodes
    node_embs = model.encode(
        nodes,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=128,
        show_progress_bar=False
    )
    
    # Compute similarities
    similarities = np.dot(node_embs, query_emb)
    
    # Rank by similarity (descending)
    ranked_indices = np.argsort(similarities)[::-1]  # High to low
    
    results = {
        'query': query,
        'num_nodes': len(nodes),
        'ranked_indices': ranked_indices.tolist(),
        'top_node_idx': int(ranked_indices[0]),
        'top_node': nodes[ranked_indices[0]],
    }
    
    if return_scores:
        results['scores'] = similarities[ranked_indices].tolist()
        results['all_scores'] = similarities.tolist()
    
    return results


def predict_top_k(
    model: SentenceTransformer,
    query: str,
    nodes: List[str],
    k: int = 10,
    add_instruction: bool = False
) -> List[Tuple[int, str, float]]:
    """
    Get top-K nodes cho query
    
    Returns:
        List of (index, node, score) tuples
    """
    results = predict_batch(model, query, nodes, add_instruction)
    
    top_k_results = []
    for i in range(min(k, len(nodes))):
        idx = results['ranked_indices'][i]
        node = nodes[idx]
        score = results['scores'][i]
        top_k_results.append((idx, node, score))
    
    return top_k_results



# 3. EVALUATION


def evaluate_retrieval(
    model: SentenceTransformer,
    test_data: List[Dict],
    recall_k_values: List[int] = [1, 3, 5, 10, 20],
    add_instruction: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Evaluate model trên test data
    
    Args:
        model: Trained model
        test_data: List of dicts với keys: query, positive, neg_candidates
        recall_k_values: List of K values để tính Recall@K
        add_instruction: Có thêm instruction không
        verbose: Print progress
    
    Returns:
        Dict với evaluation metrics
    """
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
    
    # Khởi tạo counters
    total = 0
    hits = {k: 0 for k in recall_k_values}
    mrr_sum = 0.0
    rank_distribution = []
    
    # Filter data có neg_candidates
    valid_data = [
        item for item in test_data 
        if item.get("neg_candidates") and len(item["neg_candidates"]) > 0
    ]
    
    if verbose:
        print(f"Total test samples: {len(test_data)}")
        print(f"Valid samples (with negatives): {len(valid_data)}")
        print(f"Evaluating...")
    
    # Iterate through test data
    iterator = tqdm(valid_data) if verbose else valid_data
    
    for item in iterator:
        query = item["query"]
        positive = item["positive"]
        neg_candidates = item["neg_candidates"]
        
        # Tạo candidates list (positive ở index 0)
        candidates = [positive] + neg_candidates
        
        # Predict rankings
        result = predict_batch(
            model, 
            query, 
            candidates, 
            add_instruction=add_instruction,
            return_scores=False
        )
        
        # Find rank của positive (index 0)
        ranked_indices = result['ranked_indices']
        positive_rank = ranked_indices.index(0) + 1  # 1-indexed
        
        # Update metrics
        total += 1
        rank_distribution.append(positive_rank)
        
        # Recall@K
        for k in recall_k_values:
            if positive_rank <= k:
                hits[k] += 1
        
        # MRR
        mrr_sum += 1.0 / positive_rank
    
    # Compute final metrics
    metrics = {
        'total_samples': total,
        'num_candidates': len(candidates),
    }
    
    # Recall@K
    for k in recall_k_values:
        metrics[f'recall@{k}'] = hits[k] / total if total > 0 else 0.0
    
    # MRR
    metrics['mrr'] = mrr_sum / total if total > 0 else 0.0
    
    # Rank statistics
    rank_array = np.array(rank_distribution)
    metrics['mean_rank'] = float(np.mean(rank_array))
    metrics['median_rank'] = float(np.median(rank_array))
    metrics['rank_distribution'] = rank_distribution
    
    # Print results
    if verbose:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Total samples evaluated: {total}")
        print(f"Candidates per query: {metrics['num_candidates']}")
        print()
        
        for k in recall_k_values:
            recall = metrics[f'recall@{k}']
            count = hits[k]
            print(f"Recall@{k:2d}:  {recall:.4f}  ({count}/{total})")
        
        print(f"\nMRR:         {metrics['mrr']:.4f}")
        print(f"Mean Rank:   {metrics['mean_rank']:.2f}")
        print(f"Median Rank: {metrics['median_rank']:.1f}")
        print("="*60)
    
    return metrics


def analyze_errors(
    model: SentenceTransformer,
    test_data: List[Dict],
    top_k: int = 10,
    num_errors: int = 10,
    add_instruction: bool = False
) -> List[Dict]:
    """
    Phân tích các cases model predict sai
    
    Returns:
        List of error cases
    """
    print("\n" + "="*60)
    print(f"ANALYZING TOP {num_errors} ERRORS")
    print("="*60)
    
    errors = []
    
    for item in test_data:
        if not item.get("neg_candidates"):
            continue
        
        query = item["query"]
        positive = item["positive"]
        neg_candidates = item["neg_candidates"]
        candidates = [positive] + neg_candidates
        
        # Predict
        result = predict_batch(
            model, query, candidates, 
            add_instruction=add_instruction,
            return_scores=True
        )
        
        ranked_indices = result['ranked_indices']
        positive_rank = ranked_indices.index(0) + 1
        
        # Nếu positive không trong top-K = error
        if positive_rank > top_k:
            error_case = {
                'query': query,
                'positive': positive,
                'positive_rank': positive_rank,
                'top_predicted': candidates[ranked_indices[0]],
                'top_score': result['scores'][0],
                'positive_score': result['all_scores'][0],
            }
            errors.append(error_case)
    
    # Sort by worst errors (highest rank)
    errors = sorted(errors, key=lambda x: x['positive_rank'], reverse=True)
    
    # Print top errors
    for i, error in enumerate(errors[:num_errors], 1):
        print(f"\nError {i}:")
        print(f"  Query: {error['query'][:100]}...")
        print(f"  Positive (rank {error['positive_rank']}): {error['positive'][:80]}...")
        print(f"  Predicted Top-1: {error['top_predicted'][:80]}...")
        print(f"  Scores - Predicted: {error['top_score']:.3f}, Positive: {error['positive_score']:.3f}")
    
    print("="*60)
    
    return errors



# MAIN


def main():


    

    # model_path = "/mnt/disk2/anhnv/rr/stage1/stage1_bge-small_NO-INSTRUCTION/best_recall_0.9659_step_5400"
    model_path = MODEL_PATH

    
    # Test data path
    test_data_path = TEST_JSON

    

    use_instruction = True 
    
    # Options
    should_analyze_errors = False  
    num_errors_to_show = 5       
    
    

    
    
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_path}")
    print(f"Instruction: {'ENABLED' if use_instruction else 'DISABLED'}")
    
    # Load model
    model = load_model(model_path)
    
    # Load test data
    print(f"\nLoading test data from: {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    
    # Evaluate
    metrics = evaluate_retrieval(
        model,
        test_data,
        recall_k_values=RECALL_K_VALUES,
        add_instruction=use_instruction,
        verbose=True
    )
    
    # Analyze errors
    if should_analyze_errors:
        errors = analyze_errors(
            model,
            test_data,
            top_k=10,
            num_errors=num_errors_to_show,
            add_instruction=use_instruction
        )
        print(f"\nTotal errors (not in top-10): {len(errors)}/{metrics['total_samples']}")
    
    # Save results
    model_name = os.path.basename(model_path)
    file_name = os.path.basename(test_data_path).replace('.json', '')
    output_file = f"test_results_{model_name}_{file_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        metrics_to_save = {k: v for k, v in metrics.items() if k != 'rank_distribution'}
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    

if __name__ == "__main__":
    main()
