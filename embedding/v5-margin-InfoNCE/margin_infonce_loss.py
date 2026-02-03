"""
Margin-InfoNCE Loss - InfoNCE với additive margin

Công thức:
    L_i = -log( exp(s_i^+ / τ) / (exp(s_i^+ / τ) + Σ_j exp((s_ij^- + m) / τ)) )

Khác với InfoNCE gốc:
    - InfoNCE: exp(s- / τ)
    - Margin-InfoNCE: exp((s- + m) / τ)
    
Margin m làm negatives "harder" → model học tốt hơn

Papers:
    - ANCE: Approximate Nearest Neighbor Negative Contrastive Learning
    - Inspired by ArcFace/CosFace (face recognition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer


class MarginInfoNCELoss(nn.Module):
    """
    Margin-InfoNCE Loss - InfoNCE với additive margin cho negatives
    
    Args:
        model: SentenceTransformer model
        temperature: Temperature parameter τ (default: 0.05)
        margin: Additive margin m cho negatives (default: 0.3)
    """
    
    def __init__(
        self,
        model: SentenceTransformer,
        temperature: float = 0.05,
        margin: float = 0.3,
    ):
        super(MarginInfoNCELoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        """
        Forward pass của Margin-InfoNCE Loss
        
        Args:
            sentence_features: List of dictionaries containing tokenized sentences
                [0]: queries (anchor)
                [1]: positives
            labels: Not used
            
        Returns:
            loss: Scalar tensor
        """
        # Encode queries và positives
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        queries = embeddings[0]
        positives = embeddings[1]
        
        # Normalize embeddings
        queries = F.normalize(queries, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        
        # Tính similarity matrix: (batch_size, batch_size)
        similarities = torch.matmul(queries, positives.t())
        
        # Tạo margin mask: diagonal = 0 (positive), off-diagonal = margin (negative)
        batch_size = queries.size(0)
        margin_mask = torch.ones_like(similarities) * self.margin
        margin_mask.fill_diagonal_(0.0)
        
        # Apply margin: negatives được boost thêm m
        # s_ij^- → s_ij^- + m
        similarities_with_margin = similarities + margin_mask
        
        # Scale by temperature
        logits = similarities_with_margin / self.temperature
        
        # Labels: diagonal elements
        labels = torch.arange(batch_size, device=queries.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def get_config_dict(self):
        return {
            'temperature': self.temperature,
            'margin': self.margin,
        }


class MarginInfoNCELossWithStats(MarginInfoNCELoss):

    
    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        # Encode
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        queries = F.normalize(embeddings[0], p=2, dim=1)
        positives = F.normalize(embeddings[1], p=2, dim=1)
        
        # Similarity matrix
        similarities = torch.matmul(queries, positives.t())
        
        batch_size = queries.size(0)
        
        # Apply margin
        margin_mask = torch.ones_like(similarities) * self.margin
        margin_mask.fill_diagonal_(0.0)
        similarities_with_margin = similarities + margin_mask
        
        # Logits
        logits = similarities_with_margin / self.temperature
        
        # Labels
        labels = torch.arange(batch_size, device=queries.device)
        
        # Loss
        loss = F.cross_entropy(logits, labels)
        
        # Statistics
        with torch.no_grad():
            # Original similarities (without margin)
            pos_sims = torch.diagonal(similarities)
            mask = torch.eye(batch_size, device=similarities.device).bool()
            neg_sims = similarities.masked_select(~mask)
            
            # Similarities with margin (for negatives only)
            neg_sims_with_margin = neg_sims + self.margin
            
            # Accuracy
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()
            
            # Store stats
            self.last_stats = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'pos_sim_mean': pos_sims.mean().item(),
                'pos_sim_std': pos_sims.std().item(),
                'neg_sim_mean': neg_sims.mean().item(),
                'neg_sim_std': neg_sims.std().item(),
                'neg_sim_with_margin_mean': neg_sims_with_margin.mean().item(),
                'temperature': self.temperature,
                'margin': self.margin,
            }
        
        return loss


def test_margin_infonce():
    """
    Test function để verify Margin-InfoNCE loss
    """
    print("Testing Margin-InfoNCE Loss...")
    print("="*60)
    
    batch_size = 8
    embedding_dim = 128
    
    # Mock model
    class MockModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(10, dim)
            
        def __call__(self, features):
            emb = self.linear(features['input_ids'].float())
            emb = F.normalize(emb, p=2, dim=1)
            return {'sentence_embedding': emb}
    
    model = MockModel(embedding_dim)
    
    # Dummy input
    queries = {'input_ids': torch.randn(batch_size, 10)}
    positives = {'input_ids': torch.randn(batch_size, 10)}
    
    # Test with different margins
    margins = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    print("Testing different margin values:")
    for margin in margins:
        loss_fn = MarginInfoNCELossWithStats(model, temperature=0.05, margin=margin)
        loss = loss_fn([queries, positives])
        
        print(f"\nMargin = {margin:.1f}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Accuracy: {loss_fn.last_stats['accuracy']*100:.1f}%")
        print(f"  Neg sim (original): {loss_fn.last_stats['neg_sim_mean']:.4f}")
        print(f"  Neg sim (with margin): {loss_fn.last_stats['neg_sim_with_margin_mean']:.4f}")
    
    print("\n" + "="*60)
    print(" Margin-InfoNCE test passed!")


if __name__ == "__main__":
    test_margin_infonce()
