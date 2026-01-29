"""
Generate Hard Negatives for ANCE Training
==========================================

Script này dùng để tạo hard negatives (neg_candidates) cho ANCE training.

PHƯƠNG PHÁP:
------------
1. Load baseline model (đã train hoặc pre-trained)
2. Encode tất cả documents trong corpus
3. Với mỗi query:
   - Tìm top-K documents gần nhất (cosine similarity)
   - Loại bỏ positive document
   - Lấy K hard negatives (gần query nhưng không phải positive)

OUTPUT:
-------
File JSON với format:
{
    "query": "...",
    "positive": "...",
    "neg_candidates": ["neg1", "neg2", ...]  // Hard negatives
}

USAGE:
------
Định nghĩa config trực tiếp trong file rồi chạy:
python generate_hard_negatives.py
"""

import json
from typing import List, Dict, Tuple
from tqdm import tqdm

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


#═══════════════════════════════════════════════════════════════════════════
# CONFIG
#═══════════════════════════════════════════════════════════════════════════

# Input/Output paths
INPUT_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_train_14_01_43562.json"
OUTPUT_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_train_14_01_43562_with_hard_neg.json"

# Model path (baseline model hoặc pretrained)
MODEL_PATH = "/mnt/disk2/anhnv/rr/stage1/stage1_bge-small_INSTRUCTION_1101_v3/best_recall_0.9597_step_3300"
# Hoặc dùng pretrained: MODEL_PATH = "BAAI/bge-small-en-v1.5"

# Hard negatives config
NUM_NEGATIVES = 30  # Số hard negatives cho mỗi query
BATCH_SIZE = 128    # Batch size khi encode

# BGE instruction
USE_INSTRUCTION = True
INSTRUCTION_PREFIX = "Represent this web action for retrieving relevant DOM elements: "


def load_data(path: str) -> List[Dict]:
    """Load training data"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} samples from {path}")
    return data


def build_document_corpus(data: List[Dict]) -> Tuple[List[str], Dict[str, int]]:
    """
    Build document corpus từ training data
    
    Returns:
    --------
    documents: List of unique documents
    doc_to_idx: Mapping document → index
    """
    print("\n[INFO] Building document corpus...")
    
    # Collect all unique documents
    doc_set = set()
    for item in data:
        doc_set.add(item["positive"])
    
    documents = list(doc_set)
    doc_to_idx = {doc: idx for idx, doc in enumerate(documents)}
    
    print(f"[INFO] Total unique documents: {len(documents)}")
    return documents, doc_to_idx


def encode_documents(
    model: SentenceTransformer,
    documents: List[str],
    batch_size: int = 128
) -> np.ndarray:
    """
    Encode all documents using the model
    
    Returns:
    --------
    doc_embeddings: numpy array of shape (num_docs, embedding_dim)
    """
    print("\n[INFO] Encoding documents...")
    
    doc_embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    print(f"[INFO] Encoded {len(documents)} documents")
    print(f"[INFO] Embedding shape: {doc_embeddings.shape}")
    
    return doc_embeddings


def find_hard_negatives(
    model: SentenceTransformer,
    query: str,
    positive_doc: str,
    documents: List[str],
    doc_embeddings: np.ndarray,
    doc_to_idx: Dict[str, int],
    num_negatives: int = 20,
    top_k: int = 100
) -> List[str]:
    """
    Find hard negatives for a query
    
    Strategy:
    ---------
    1. Encode query
    2. Compute similarity with all documents
    3. Retrieve top-K documents (excluding positive)
    4. Return top num_negatives as hard negatives
    
    Args:
    -----
    query: Query string
    positive_doc: Positive document
    documents: List of all documents
    doc_embeddings: Pre-computed document embeddings
    doc_to_idx: Document → index mapping
    num_negatives: Number of hard negatives to return
    top_k: Retrieve top-K before filtering
    
    Returns:
    --------
    hard_negatives: List of hard negative documents
    """
    # Encode query
    query_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )[0]  # Shape: (embedding_dim,)
    
    # Compute similarity: query × all documents
    similarities = np.dot(doc_embeddings, query_emb)  # Shape: (num_docs,)
    
    # Get top-K indices (most similar)
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Filter out positive document
    positive_idx = doc_to_idx.get(positive_doc, -1)
    
    hard_negatives = []
    for idx in top_k_indices:
        if idx != positive_idx:
            hard_negatives.append(documents[idx])
        
        if len(hard_negatives) >= num_negatives:
            break
    
    return hard_negatives


def generate_hard_negatives(
    model: SentenceTransformer,
    data: List[Dict],
    documents: List[str],
    doc_embeddings: np.ndarray,
    doc_to_idx: Dict[str, int],
    num_negatives: int = 20,
    use_instruction: bool = True,
    instruction_prefix: str = "Represent this web action for retrieving relevant DOM elements: "
) -> List[Dict]:
    """
    Generate hard negatives for all queries
    
    Returns:
    --------
    data_with_negatives: List of dicts with neg_candidates added
    """
    print(f"\n[INFO] Generating hard negatives...")
    print(f"[INFO] Num negatives per query: {num_negatives}")
    print(f"[INFO] Use instruction: {use_instruction}")
    
    data_with_negatives = []
    
    for item in tqdm(data, desc="Processing queries"):
        query = item["query"]
        positive_doc = item["positive"]
        
        # Add instruction prefix if enabled
        if use_instruction:
            query_with_instruction = instruction_prefix + query
        else:
            query_with_instruction = query
        
        # Find hard negatives
        hard_negatives = find_hard_negatives(
            model=model,
            query=query_with_instruction,
            positive_doc=positive_doc,
            documents=documents,
            doc_embeddings=doc_embeddings,
            doc_to_idx=doc_to_idx,
            num_negatives=num_negatives
        )
        
        # Add to result
        item_with_neg = {
            "query": query,  # Original query without instruction
            "positive": positive_doc,
            "neg_candidates": hard_negatives
        }
        data_with_negatives.append(item_with_neg)
    
    print(f"[INFO] Generated hard negatives for {len(data_with_negatives)} samples")
    
    return data_with_negatives


def save_data(data: List[Dict], output_path: str):
    """Save data to JSON file"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved to {output_path}")


def main():
    print("="*80)
    print(" GENERATE HARD NEGATIVES FOR ANCE")
    print("="*80)
    print(f"\n[CONFIG]")
    print(f"  Input: {INPUT_JSON}")
    print(f"  Output: {OUTPUT_JSON}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Num negatives: {NUM_NEGATIVES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Use instruction: {USE_INSTRUCTION}")
    
    # Step 1: Load model
    print("\n[STEP 1] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    model = SentenceTransformer(MODEL_PATH, device=device)
    print(f"[INFO] Model loaded: {MODEL_PATH}")
    
    # Step 2: Load data
    print("\n[STEP 2] Loading data...")
    data = load_data(INPUT_JSON)
    
    # Step 3: Build document corpus
    print("\n[STEP 3] Building document corpus...")
    documents, doc_to_idx = build_document_corpus(data)
    
    # Step 4: Encode documents
    print("\n[STEP 4] Encoding documents...")
    doc_embeddings = encode_documents(
        model=model,
        documents=documents,
        batch_size=BATCH_SIZE
    )
    
    # Step 5: Generate hard negatives
    print("\n[STEP 5] Generating hard negatives...")
    data_with_negatives = generate_hard_negatives(
        model=model,
        data=data,
        documents=documents,
        doc_embeddings=doc_embeddings,
        doc_to_idx=doc_to_idx,
        num_negatives=NUM_NEGATIVES,
        use_instruction=USE_INSTRUCTION,
        instruction_prefix=INSTRUCTION_PREFIX
    )
    
    # Step 6: Save output
    print("\n[STEP 6] Saving output...")
    save_data(data_with_negatives, OUTPUT_JSON)
    
    # Statistics
    print("\n" + "="*80)
    print(" STATISTICS")
    print("="*80)
    
    num_with_negatives = sum(1 for item in data_with_negatives if item["neg_candidates"])
    avg_negatives = np.mean([len(item["neg_candidates"]) for item in data_with_negatives])
    
    print(f"\n  Total samples: {len(data_with_negatives)}")
    print(f"  Samples with negatives: {num_with_negatives}")
    print(f"  Avg negatives per sample: {avg_negatives:.2f}")
    print(f"  Output saved to: {OUTPUT_JSON}")
    print("\n" + "="*80)
    print(" DONE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
