"""
Evaluate Stage 2 Reranker Model (DeBERTa)
- Load trained DeBERTa reranker model
- Evaluate on stage 2 test data (top 50 candidates)
- Compute metrics: Recall@K, MRR, MAP
"""

import json
import os
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset



# CONFIG


# Model path
MODEL_PATH = "/mnt/disk2/anhnv/rr/stage2/checkpoints/deberta_reranker/best_model"

# Test data files (các file stage2_*_queries_top50.json)
TEST_DATA_DIR = "/mnt/disk2/anhnv/rr/stage2/data/test"

# Evaluation settings
MAX_LENGTH = 512
BATCH_SIZE = 128
RECALL_K_VALUES = [1, 3, 5, 10, 20]

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# 1. DATASET


class RerankEvalDataset(Dataset):
    """

    """
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        

        self.samples = []
        self.query_groups = []  
        
        current_idx = 0
        for item in data:
            query = item['query']
            nodes = item['top_k_nodes']
            num_nodes = len(nodes)
            
            # Lưu group info
            self.query_groups.append({
                'start_idx': current_idx,
                'end_idx': current_idx + num_nodes,
                'gold_idx': item['gold_idx'],
                'query': query
            })
            
            # Flatten nodes
            for node in nodes:
                self.samples.append({
                    'query': query,
                    'node': node
                })
            
            current_idx += num_nodes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample['query'],
            sample['node'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }



# 2. LOAD MODEL


def load_model(model_path: str):
    """Load trained DeBERTa reranker"""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1
    )
    model.to(DEVICE)
    model.eval()
    
    print(f"  Device: {DEVICE}")
    
    return model, tokenizer



# 3. PREDICT & RANK


def predict_scores(model, dataloader) -> List[float]:
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get scores (apply sigmoid)
            logits = outputs.logits.squeeze(-1)
            scores = torch.sigmoid(logits)
            
            all_scores.extend(scores.cpu().numpy().tolist())
    
    return all_scores


def rerank_queries(scores: List[float], query_groups: List[Dict]) -> List[Dict]:
    """
    Args:
        scores: Flat list of all scores
        query_groups: List of query group info
    
    Returns:
        List of ranking results for each query
    """
    results = []
    
    for group in query_groups:
        # Extract scores for this query
        start_idx = group['start_idx']
        end_idx = group['end_idx']
        query_scores = scores[start_idx:end_idx]
        
        # Rank by score (descending)
        ranked_indices = np.argsort(query_scores)[::-1]  
        
        # Find gold node rank
        gold_idx = group['gold_idx']
        gold_rank = np.where(ranked_indices == gold_idx)[0][0] + 1 
        
        results.append({
            'query': group['query'],
            'gold_idx': gold_idx,
            'gold_rank': gold_rank,
            'ranked_indices': ranked_indices.tolist(),
            'scores': [query_scores[i] for i in ranked_indices]
        })
    
    return results



# 4. EVALUATION METRICS


def compute_metrics(ranking_results: List[Dict], recall_k_values: List[int]) -> Dict:
    """
    Metrics:
    - Recall@K: % of queries where gold is in top K
    - MRR: Mean Reciprocal Rank
    - MAP: Mean Average Precision
    """
    total = len(ranking_results)
    
    # Recall@K
    hits = {k: 0 for k in recall_k_values}
    
    # MRR
    mrr_sum = 0.0
    
    ap_sum = 0.0
    
    # Rank distribution
    ranks = []
    
    for result in ranking_results:
        gold_rank = result['gold_rank']
        ranks.append(gold_rank)
        
        # Recall@K
        for k in recall_k_values:
            if gold_rank <= k:
                hits[k] += 1
        
        # MRR
        mrr_sum += 1.0 / gold_rank
        
        # MAP 
        ap_sum += 1.0 / gold_rank
    
    # Compute final metrics
    metrics = {
        'total_queries': total,
    }
    
    # Recall@K
    for k in recall_k_values:
        metrics[f'recall@{k}'] = hits[k] / total
    
    # MRR
    metrics['mrr'] = mrr_sum / total
    
    # MAP
    metrics['map'] = ap_sum / total
    
    # Rank statistics
    ranks = np.array(ranks)
    metrics['mean_rank'] = float(np.mean(ranks))
    metrics['median_rank'] = float(np.median(ranks))
    metrics['std_rank'] = float(np.std(ranks))
    
    return metrics



# 5. MAIN EVALUATION


def evaluate_on_file(model, tokenizer, test_file: str, verbose: bool = True) -> Dict:
    """
    Evaluate model on one test file
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {os.path.basename(test_file)}")
        print(f"{'='*60}")
    
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if verbose:
        print(f"Loaded {len(test_data)} queries")
    
    # Create dataset
    dataset = RerankEvalDataset(test_data, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    if verbose:
        print(f"Total samples: {len(dataset)}")
    
    # Predict scores
    scores = predict_scores(model, dataloader)
    
    # Rerank
    ranking_results = rerank_queries(scores, dataset.query_groups)
    
    # Compute metrics
    metrics = compute_metrics(ranking_results, RECALL_K_VALUES)
    
    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Total queries: {metrics['total_queries']}")
        print()
        
        for k in RECALL_K_VALUES:
            recall = metrics[f'recall@{k}']
            count = int(recall * metrics['total_queries'])
            print(f"Recall@{k:2d}:  {recall:.4f}  ({count}/{metrics['total_queries']})")
        
        print(f"\nMRR:         {metrics['mrr']:.4f}")
        print(f"MAP:         {metrics['map']:.4f}")
        print(f"Mean Rank:   {metrics['mean_rank']:.2f}")
        print(f"Median Rank: {metrics['median_rank']:.1f}")
        print(f"Std Rank:    {metrics['std_rank']:.2f}")
        print(f"{'='*60}")
    
    return metrics, ranking_results


def evaluate_all_files(model, tokenizer, test_dir: str):
    # Find all test files
    import glob
    test_files = glob.glob(os.path.join(test_dir, "stage2_*_queries_top50.json"))
    test_files = sorted(test_files)
    
    if not test_files:
        print(f"Not found in {test_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"Found {len(test_files)} test files")
    
    # Evaluate each file
    all_metrics = []
    all_results = {}
    
    for test_file in test_files:
        metrics, results = evaluate_on_file(model, tokenizer, test_file, verbose=True)
        
        file_name = os.path.basename(test_file).replace('.json', '')
        all_metrics.append(metrics)
        all_results[file_name] = {
            'metrics': metrics,
            'num_queries': len(results)
        }
    
    # Aggregate metrics (weighted average by number of queries)
    total_queries = sum(m['total_queries'] for m in all_metrics)
    
    agg_metrics = {
        'total_queries': total_queries,
        'num_files': len(test_files)
    }
    
    # Weighted average for each metric
    for k in RECALL_K_VALUES:
        key = f'recall@{k}'
        weighted_sum = sum(m[key] * m['total_queries'] for m in all_metrics)
        agg_metrics[key] = weighted_sum / total_queries
    
    for metric_name in ['mrr', 'map', 'mean_rank', 'median_rank']:
        weighted_sum = sum(m[metric_name] * m['total_queries'] for m in all_metrics)
        agg_metrics[metric_name] = weighted_sum / total_queries
    
    # Print aggregated results

    print(f"Total queries: {agg_metrics['total_queries']}")
    print(f"Number of files: {agg_metrics['num_files']}")

    
    for k in RECALL_K_VALUES:
        recall = agg_metrics[f'recall@{k}']
        print(f"Recall@{k:2d}:  {recall:.4f}")
    
    print(f"\nMRR:         {agg_metrics['mrr']:.4f}")
    print(f"MAP:         {agg_metrics['map']:.4f}")
    print(f"Mean Rank:   {agg_metrics['mean_rank']:.2f}")
    print(f"Median Rank: {agg_metrics['median_rank']:.2f}")
    
    return agg_metrics, all_results


def main():

    print(f"Model: {MODEL_PATH}")
    print(f"Test data dir: {TEST_DATA_DIR}")
    print(f"Device: {DEVICE}")
    
    # Load model
    model, tokenizer = load_model(MODEL_PATH)
    
    # Evaluate on all files
    agg_metrics, all_results = evaluate_all_files(model, tokenizer, TEST_DATA_DIR)
    
    # Save results
    output_dir = os.path.dirname(MODEL_PATH)
    output_file = os.path.join(output_dir, "eval_results.json")
    
    
    final_results = {}
    
    # Thêm kết quả từng file
    for file_name, file_result in all_results.items():
        final_results[file_name] = file_result['metrics']
    
    # Thêm kết quả trung bình ở cuối
    final_results['AVERAGE_ALL_FILES'] = agg_metrics
    
    # Lưu vào 1 file JSON duy nhất
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f" Save to file: {output_file}")


if __name__ == "__main__":
    main()
