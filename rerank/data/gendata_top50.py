"""
Generate Stage 2 training data
- Load trained model from stage 1
- For each query, get top 50 nodes
- Only keep cases where gold node is in top 50
- Save to JSON file
"""

import json
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm



# CONFIG


MODEL_PATH = "/mnt/disk2/anhnv/rr/stage1/v5-margin-InfoNCE/stage1_margin_infonce_bge-small/final_model"  
# MODEL_PATH = "BAAI/bge-small-en-v1.5"  

# amazon_queries arxiv_queries awward_queries bbc_news_queries behance_queries economist_queries 
# health_queries neflix_queries pinterest_queries sportify_queries stackover_queries twitch_queries
# Input data path
INPUT_JSON = "/mnt/disk2/anhnv/rr/stage1/data/processed_test/stackover_queries.json"

# Output folder
OUTPUT_FOLDER = "/mnt/disk2/anhnv/rr/stage2/data"

# BGE instruction
USE_BGE_INSTRUCTION = True  
BGE_QUERY_INSTRUCTION = "Represent this web action for retrieving relevant DOM elements: "

TOP_K = 50  



# 1. LOAD MODEL


def load_model(model_path: str) -> SentenceTransformer:

    if os.path.exists(model_path):
        print(" Detected LOCAL model path")
        model = SentenceTransformer(model_path)
    else:
        model = SentenceTransformer(model_path)

    print("Model info")
    print(f"  - Embedding dim: {model.get_sentence_embedding_dimension()}")
    print(f"  - Max seq length: {model.max_seq_length}")

    return model




# 2. GENERATE STAGE 2 DATA


def generate_top50_data(
    model: SentenceTransformer,
    input_data: List[Dict],
    top_k: int = 50,
    add_instruction: bool = False,
    verbose: bool = True
) -> List[Dict]:
    valid_data = [
        item for item in input_data 
        if item.get("neg_candidates") and len(item["neg_candidates"]) > 0
    ]
    
    
    stage2_data = []
    skipped_count = 0
    
    iterator = tqdm(valid_data) if verbose else valid_data
    
    for item in iterator:
        query = item["query"]
        positive = item["positive"]
        neg_candidates = item["neg_candidates"]
        
        # Tạo candidates list
        candidates = [positive] + neg_candidates
        
        query_with_instruction = query
        if add_instruction:
            query_with_instruction = BGE_QUERY_INSTRUCTION + query
        
        # Encode query
        query_emb = model.encode(
            query_with_instruction,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Encode all candidates
        candidate_embs = model.encode(
            candidates,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=128,
            show_progress_bar=False
        )
        
        similarities = np.dot(candidate_embs, query_emb)
        
        # sort by similarity 
        ranked_indices = np.argsort(similarities)[::-1]  
        
        # Get top K indices
        top_k_indices = ranked_indices[:top_k].tolist()
        
        if 0 not in top_k_indices:
            skipped_count += 1
            continue
        
        # Get top K nodes
        top_k_nodes = [candidates[idx] for idx in top_k_indices]
        
        # Find gold node in top K
        gold_idx_in_topk = top_k_indices.index(0)
        
        # Create stage 2 sample
        stage2_sample = {
            "query": query,
            "gold_node": positive,
            "gold_idx": gold_idx_in_topk,  
            "top_k_nodes": top_k_nodes,
            "num_candidates": len(top_k_nodes)
        }
        
        if "metadata" in item:
            stage2_sample["metadata"] = item["metadata"]
        if "action" in item:
            stage2_sample["action"] = item["action"]
        if "history" in item:
            stage2_sample["history"] = item["history"]
        
        stage2_data.append(stage2_sample)
    
    print(f"Total valid samples: {len(valid_data)}")
    print(f"Samples with gold in top-{top_k}: {len(stage2_data)}")
    print(f"Skipped samples (gold not in top-{top_k}): {skipped_count}")
    print(f"Success rate: {len(stage2_data)/len(valid_data)*100:.2f}%")
    
    return stage2_data


def main():

    model_path = MODEL_PATH
    
    input_data_path = INPUT_JSON
    
    use_instruction = USE_BGE_INSTRUCTION
    
    # Top K
    top_k = TOP_K
    
    # Output folder
    output_folder = OUTPUT_FOLDER
    
    
    
    # RUN GENERATION
    
    print(f"Model: {model_path}")
    print(f"Input data: {input_data_path}")
    print(f"Instruction: {'ENABLED' if use_instruction else 'DISABLED'}")
    print(f"Top K: {top_k}")
    print(f"Output folder: {output_folder}")

    
    # Load model
    model = load_model(model_path)
    
    # Load input data
    print(f"\nLoading input data from: {input_data_path}")
    with open(input_data_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    print(f"Loaded {len(input_data)} input samples")
    
    stage2_data = generate_top50_data(
        model,
        input_data,
        top_k=top_k,
        add_instruction=use_instruction,
        verbose=True
    )
    
    # Create output folder 
    os.makedirs(output_folder, exist_ok=True)
    
    # Save results
    input_filename = os.path.basename(input_data_path).replace('.json', '')
    model_name = os.path.basename(model_path)
    output_filename = f"stage2_{input_filename}_top{top_k}.json"
    output_path = os.path.join(output_folder, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stage2_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Total samples: {len(stage2_data)}")
    


if __name__ == "__main__":
    main()
