"""
Visualization & Analysis cho Focal-InfoNCE

Phân tích:
1. Embedding space visualization (t-SNE, UMAP)
2. Similarity distribution (positive vs negative)
3. Alignment & Uniformity metrics
4. Hard negative analysis
"""

import json
import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# Optional: UMAP for better visualization
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")

from sklearn.manifold import TSNE


# ============================================================
# CONFIG
# ============================================================

EVAL_JSON = "/mnt/disk2/anhnv/rr/stage1/data/stage1_eval_2827_120neg.json"
USE_BGE_INSTRUCTION = True
BGE_QUERY_INSTRUCTION = "Represent this web action for retrieving relevant DOM elements: "

# Models to compare
MODEL_INFONCE = "./compare_infonce"  # Path to InfoNCE model
MODEL_FOCAL = "./compare_focal"      # Path to Focal-InfoNCE model

# Visualization settings
NUM_SAMPLES = 500  # Number of samples to visualize
SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_eval_data(path: str, n_samples: int = None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if n_samples:
        np.random.seed(SEED)
        indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
        data = [data[i] for i in indices]
    
    return data


# ============================================================
# METRICS
# ============================================================

def compute_alignment_uniformity(embeddings: np.ndarray):
    """
    Compute alignment and uniformity metrics
    
    Alignment: measures how close positive pairs are (lower is better)
    Uniformity: measures how uniformly distributed embeddings are (lower is better)
    
    Reference: Wang & Isola, 2020
    """
    n = len(embeddings)
    
    # Uniformity: -log of average pairwise Gaussian potential
    uniformity_sum = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = np.sum((embeddings[i] - embeddings[j])**2)
            uniformity_sum += np.exp(-2 * dist_sq)
    
    uniformity = np.log(uniformity_sum * 2 / (n * (n - 1)))
    
    return uniformity


def compute_similarity_stats(model: SentenceTransformer, eval_data: List):
    """Compute similarity statistics for positive and negative pairs"""
    
    pos_similarities = []
    neg_similarities = []
    
    for item in eval_data[:200]:  # Sample 200 for speed
        query = item["query"]
        if USE_BGE_INSTRUCTION:
            query = BGE_QUERY_INSTRUCTION + query
        
        positive = item["positive"]
        negatives = item.get("neg_candidates", [])[:10]  # Top 10 negatives
        
        # Encode
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        pos_emb = model.encode([positive], normalize_embeddings=True)[0]
        
        # Positive similarity
        pos_sim = np.dot(q_emb, pos_emb)
        pos_similarities.append(pos_sim)
        
        # Negative similarities
        if negatives:
            neg_embs = model.encode(negatives, normalize_embeddings=True)
            neg_sims = np.dot(neg_embs, q_emb)
            neg_similarities.extend(neg_sims.tolist())
    
    return {
        'pos_mean': np.mean(pos_similarities),
        'pos_std': np.std(pos_similarities),
        'pos_min': np.min(pos_similarities),
        'pos_max': np.max(pos_similarities),
        'neg_mean': np.mean(neg_similarities),
        'neg_std': np.std(neg_similarities),
        'neg_min': np.min(neg_similarities),
        'neg_max': np.max(neg_similarities),
        'pos_similarities': pos_similarities,
        'neg_similarities': neg_similarities,
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_similarity_distributions(stats_infonce, stats_focal, save_path="similarity_dist.png"):
    """Plot similarity distributions for positive and negative pairs"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # InfoNCE
    axes[0].hist(stats_infonce['pos_similarities'], bins=30, alpha=0.6, label='Positive', color='green')
    axes[0].hist(stats_infonce['neg_similarities'], bins=30, alpha=0.6, label='Negative', color='red')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('InfoNCE - Similarity Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Focal-InfoNCE
    axes[1].hist(stats_focal['pos_similarities'], bins=30, alpha=0.6, label='Positive', color='green')
    axes[1].hist(stats_focal['neg_similarities'], bins=30, alpha=0.6, label='Negative', color='red')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Focal-InfoNCE - Similarity Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_embedding_space(embeddings_infonce, embeddings_focal, labels, save_path="embedding_space.png"):
    """Visualize embedding space using t-SNE"""
    
    print("\n  Computing t-SNE projections...")
    
    # t-SNE for InfoNCE
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    emb_2d_infonce = tsne.fit_transform(embeddings_infonce)
    
    # t-SNE for Focal-InfoNCE
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    emb_2d_focal = tsne.fit_transform(embeddings_focal)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # InfoNCE
    scatter1 = axes[0].scatter(
        emb_2d_infonce[:, 0],
        emb_2d_infonce[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=20
    )
    axes[0].set_title('InfoNCE - Embedding Space (t-SNE)')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # Focal-InfoNCE
    scatter2 = axes[1].scatter(
        emb_2d_focal[:, 0],
        emb_2d_focal[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=20
    )
    axes[1].set_title('Focal-InfoNCE - Embedding Space (t-SNE)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics, save_path="metrics_comparison.png"):
    """Compare metrics between InfoNCE and Focal-InfoNCE"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Similarity comparison
    models = ['InfoNCE', 'Focal-InfoNCE']
    pos_means = [metrics['infonce']['pos_mean'], metrics['focal']['pos_mean']]
    neg_means = [metrics['infonce']['neg_mean'], metrics['focal']['neg_mean']]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0].bar(x - width/2, pos_means, width, label='Positive', color='green', alpha=0.7)
    axes[0].bar(x + width/2, neg_means, width, label='Negative', color='red', alpha=0.7)
    axes[0].set_ylabel('Mean Cosine Similarity')
    axes[0].set_title('Mean Similarity Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Uniformity comparison
    uniformities = [metrics['infonce']['uniformity'], metrics['focal']['uniformity']]
    axes[1].bar(models, uniformities, color=['blue', 'orange'], alpha=0.7)
    axes[1].set_ylabel('Uniformity (lower is better)')
    axes[1].set_title('Uniformity Comparison')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print(" FOCAL-INFONCE VISUALIZATION & ANALYSIS")
    print("="*70)
    
    # Load data
    print("\nLoading evaluation data...")
    eval_data = load_eval_data(EVAL_JSON, n_samples=NUM_SAMPLES)
    print(f"  Loaded {len(eval_data)} samples")
    
    # Load models
    print("\nLoading models...")
    try:
        model_infonce = SentenceTransformer(MODEL_INFONCE)
        print(f"  ✓ InfoNCE model loaded from {MODEL_INFONCE}")
    except:
        print(f"  ✗ Failed to load InfoNCE model from {MODEL_INFONCE}")
        print(f"    Run compare_losses.py first!")
        return
    
    try:
        model_focal = SentenceTransformer(MODEL_FOCAL)
        print(f"  ✓ Focal-InfoNCE model loaded from {MODEL_FOCAL}")
    except:
        print(f"  ✗ Failed to load Focal-InfoNCE model from {MODEL_FOCAL}")
        print(f"    Run compare_losses.py first!")
        return
    
    # Extract queries for embedding analysis
    queries = []
    for item in eval_data:
        query = item["query"]
        if USE_BGE_INSTRUCTION:
            query = BGE_QUERY_INSTRUCTION + query
        queries.append(query)
    
    print("\n" + "-"*70)
    print(" 1. COMPUTING EMBEDDINGS")
    print("-"*70)
    
    print("\n  Encoding with InfoNCE...")
    embeddings_infonce = model_infonce.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print("\n  Encoding with Focal-InfoNCE...")
    embeddings_focal = model_focal.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Compute metrics
    print("\n" + "-"*70)
    print(" 2. COMPUTING METRICS")
    print("-"*70)
    
    print("\n  Computing similarity statistics...")
    stats_infonce = compute_similarity_stats(model_infonce, eval_data)
    stats_focal = compute_similarity_stats(model_focal, eval_data)
    
    print("\n  Computing uniformity...")
    uniformity_infonce = compute_alignment_uniformity(embeddings_infonce)
    uniformity_focal = compute_alignment_uniformity(embeddings_focal)
    
    # Collect all metrics
    metrics = {
        'infonce': {
            **stats_infonce,
            'uniformity': uniformity_infonce,
        },
        'focal': {
            **stats_focal,
            'uniformity': uniformity_focal,
        }
    }
    
    # Print metrics
    print("\n" + "-"*70)
    print(" 3. METRICS COMPARISON")
    print("-"*70)
    
    print("\n  InfoNCE:")
    print(f"    Positive similarity: {stats_infonce['pos_mean']:.4f} ± {stats_infonce['pos_std']:.4f}")
    print(f"    Negative similarity: {stats_infonce['neg_mean']:.4f} ± {stats_infonce['neg_std']:.4f}")
    print(f"    Uniformity: {uniformity_infonce:.4f}")
    
    print("\n  Focal-InfoNCE:")
    print(f"    Positive similarity: {stats_focal['pos_mean']:.4f} ± {stats_focal['pos_std']:.4f}")
    print(f"    Negative similarity: {stats_focal['neg_mean']:.4f} ± {stats_focal['neg_std']:.4f}")
    print(f"    Uniformity: {uniformity_focal:.4f}")
    
    print("\n  Improvement:")
    print(f"    Δ Positive sim: {stats_focal['pos_mean'] - stats_infonce['pos_mean']:+.4f}")
    print(f"    Δ Negative sim: {stats_focal['neg_mean'] - stats_infonce['neg_mean']:+.4f}")
    print(f"    Δ Uniformity: {uniformity_focal - uniformity_infonce:+.4f} (lower is better)")
    
    # Visualizations
    print("\n" + "-"*70)
    print(" 4. GENERATING VISUALIZATIONS")
    print("-"*70)
    
    print("\n  Creating similarity distribution plots...")
    plot_similarity_distributions(stats_infonce, stats_focal)
    
    print("\n  Creating metrics comparison plots...")
    plot_metrics_comparison(metrics)
    
    print("\n  Creating embedding space visualization...")
    # Create pseudo-labels for coloring (e.g., by query type)
    labels = np.arange(len(queries)) % 10  # Simple coloring
    plot_embedding_space(embeddings_infonce, embeddings_focal, labels)
    
    # Save metrics to JSON
    print("\n  Saving metrics to file...")
    
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {
        'infonce': {
            'pos_mean': float(stats_infonce['pos_mean']),
            'pos_std': float(stats_infonce['pos_std']),
            'neg_mean': float(stats_infonce['neg_mean']),
            'neg_std': float(stats_infonce['neg_std']),
            'uniformity': float(uniformity_infonce),
        },
        'focal': {
            'pos_mean': float(stats_focal['pos_mean']),
            'pos_std': float(stats_focal['pos_std']),
            'neg_mean': float(stats_focal['neg_mean']),
            'neg_std': float(stats_focal['neg_std']),
            'uniformity': float(uniformity_focal),
        }
    }
    
    with open("analysis_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Saved: analysis_metrics.json")
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - similarity_dist.png: Similarity distribution comparison")
    print("  - metrics_comparison.png: Metrics bar charts")
    print("  - embedding_space.png: t-SNE visualization")
    print("  - analysis_metrics.json: Detailed metrics")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
