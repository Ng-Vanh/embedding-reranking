"""
Contrastive Loss (Siamese Loss) - Loss function cổ điển cho Siamese Networks

Công thức:
    L = y·d² + (1-y)·max(0, m-d)²

Trong đó:
    - y = 1: positive pair (similar)
    - y = 0: negative pair (dissimilar)
    - d: Euclidean distance giữa 2 embeddings
    - m: margin (threshold cho negative pairs)

Ý nghĩa:
    - Positive pairs (y=1): L = d² → minimize distance (kéo gần lại)
    - Negative pairs (y=0): L = max(0, m-d)² → chỉ penalize nếu d < m (đẩy xa ra nếu quá gần)

Paper:
    - Dimensionality Reduction by Learning an Invariant Mapping (Hadsell et al., 2006)
    - http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Khác với InfoNCE:
    - InfoNCE: softmax over batch (relative comparison)
    - Contrastive: pairwise loss (absolute distance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (Siamese Loss)
    
    Args:
        model: SentenceTransformer model
        margin: Margin m cho negative pairs (default: 1.0)
        distance_metric: 'euclidean' hoặc 'cosine' (default: 'euclidean')
    """
    
    def __init__(
        self,
        model: SentenceTransformer,
        margin: float = 1.0,
        distance_metric: str = 'euclidean',
    ):
        super(ContrastiveLoss, self).__init__()
        self.model = model
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        """
        Forward pass của Contrastive Loss
        
        Args:
            sentence_features: List of dictionaries containing tokenized sentences
                [0]: queries (anchor)
                [1]: positives (hoặc có thể là negatives)
            labels: Tensor of shape (batch_size,)
                    1 = positive pair (similar)
                    0 = negative pair (dissimilar)
            
        Returns:
            loss: Scalar tensor
        """
        # Encode embeddings
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        emb1 = embeddings[0]  # (batch_size, embedding_dim)
        emb2 = embeddings[1]  # (batch_size, embedding_dim)
        
        # Compute distance
        if self.distance_metric == 'euclidean':
            # Euclidean distance: ||emb1 - emb2||_2
            distances = F.pairwise_distance(emb1, emb2, p=2)
        elif self.distance_metric == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            # Normalize embeddings
            emb1_norm = F.normalize(emb1, p=2, dim=1)
            emb2_norm = F.normalize(emb2, p=2, dim=1)
            cosine_sim = (emb1_norm * emb2_norm).sum(dim=1)
            distances = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Default labels: all positive pairs (y=1)
        if labels is None:
            labels = torch.ones(emb1.size(0), device=emb1.device)
        
        # Contrastive loss
        # L = y·d² + (1-y)·max(0, m-d)²
        positive_loss = labels * distances.pow(2)
        negative_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        loss = (positive_loss + negative_loss).mean()
        
        return loss
    
    def get_config_dict(self):
        return {
            'margin': self.margin,
            'distance_metric': self.distance_metric,
        }


class ContrastiveLossWithStats(ContrastiveLoss):
    """
    Contrastive Loss với logging để debug
    """
    
    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        # Encode
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        emb1 = embeddings[0]
        emb2 = embeddings[1]
        
        # Compute distance
        if self.distance_metric == 'euclidean':
            distances = F.pairwise_distance(emb1, emb2, p=2)
        elif self.distance_metric == 'cosine':
            emb1_norm = F.normalize(emb1, p=2, dim=1)
            emb2_norm = F.normalize(emb2, p=2, dim=1)
            cosine_sim = (emb1_norm * emb2_norm).sum(dim=1)
            distances = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Default labels
        if labels is None:
            labels = torch.ones(emb1.size(0), device=emb1.device)
        
        # Compute loss components
        positive_loss = labels * distances.pow(2)
        negative_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        loss = (positive_loss + negative_loss).mean()
        
        # Statistics
        with torch.no_grad():
            # Separate positive and negative pairs
            pos_mask = labels == 1
            neg_mask = labels == 0
            
            pos_distances = distances[pos_mask] if pos_mask.any() else torch.tensor([0.0])
            neg_distances = distances[neg_mask] if neg_mask.any() else torch.tensor([0.0])
            
            # Negative pairs violating margin
            neg_violations = (neg_distances < self.margin).sum().item() if neg_mask.any() else 0
            
            self.last_stats = {
                'loss': loss.item(),
                'pos_distance_mean': pos_distances.mean().item() if pos_mask.any() else 0.0,
                'pos_distance_std': pos_distances.std().item() if pos_mask.any() else 0.0,
                'neg_distance_mean': neg_distances.mean().item() if neg_mask.any() else 0.0,
                'neg_distance_std': neg_distances.std().item() if neg_mask.any() else 0.0,
                'neg_violations': neg_violations,
                'neg_violations_pct': neg_violations / neg_mask.sum().item() * 100 if neg_mask.any() else 0.0,
                'margin': self.margin,
                'num_positive': pos_mask.sum().item(),
                'num_negative': neg_mask.sum().item(),
            }
        
        return loss


class BatchContrastiveLoss(nn.Module):
    """
    Batch Contrastive Loss - Sử dụng in-batch negatives
    
    Tự động tạo positive và negative pairs từ batch:
    - Positive: (query_i, positive_i)
    - Negatives: (query_i, positive_j) với j != i
    
    Tương tự InfoNCE nhưng dùng Contrastive Loss thay vì cross-entropy
    """
    
    def __init__(
        self,
        model: SentenceTransformer,
        margin: float = 1.0,
        distance_metric: str = 'euclidean',
    ):
        super(BatchContrastiveLoss, self).__init__()
        self.model = model
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        """
        Forward pass với in-batch negatives
        
        Args:
            sentence_features: [queries, positives]
            labels: Not used (auto-generated from batch)
        """
        # Encode
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        queries = embeddings[0]  # (batch_size, dim)
        positives = embeddings[1]  # (batch_size, dim)
        
        batch_size = queries.size(0)
        
        # Tạo tất cả pairs trong batch
        # queries: (batch_size, 1, dim)
        # positives: (1, batch_size, dim)
        queries_expanded = queries.unsqueeze(1)
        positives_expanded = positives.unsqueeze(0)
        
        # Compute pairwise distances: (batch_size, batch_size)
        if self.distance_metric == 'euclidean':
            # ||q_i - p_j||_2
            distances = torch.norm(queries_expanded - positives_expanded, p=2, dim=2)
        elif self.distance_metric == 'cosine':
            queries_norm = F.normalize(queries, p=2, dim=1)
            positives_norm = F.normalize(positives, p=2, dim=1)
            cosine_sim = torch.matmul(queries_norm, positives_norm.t())
            distances = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Create labels: diagonal = 1 (positive), off-diagonal = 0 (negative)
        labels = torch.eye(batch_size, device=queries.device)
        
        # Contrastive loss
        positive_loss = labels * distances.pow(2)
        negative_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        loss = (positive_loss + negative_loss).sum() / batch_size
        
        return loss
    
    def get_config_dict(self):
        return {
            'margin': self.margin,
            'distance_metric': self.distance_metric,
            'batch_mode': True,
        }


def test_contrastive_loss():
    """
    Test function để verify Contrastive Loss
    """
    print("Testing Contrastive Loss...")
    print("="*60)
    
    batch_size = 4
    embedding_dim = 128
    
    # Mock model
    class MockModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(10, dim)
            
        def __call__(self, features):
            return {'sentence_embedding': self.linear(features['input_ids'].float())}
    
    model = MockModel(embedding_dim)
    
    # Test 1: All positive pairs
    print("\nTest 1: All positive pairs (y=1)")
    queries = {'input_ids': torch.randn(batch_size, 10)}
    positives = {'input_ids': torch.randn(batch_size, 10)}
    labels = torch.ones(batch_size)
    
    loss_fn = ContrastiveLossWithStats(model, margin=1.0)
    loss = loss_fn([queries, positives], labels)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {loss_fn.last_stats}")
    
    # Test 2: Mixed positive and negative
    print("\nTest 2: Mixed positive/negative pairs")
    labels = torch.tensor([1, 0, 1, 0]).float()
    
    loss = loss_fn([queries, positives], labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {loss_fn.last_stats}")
    
    # Test 3: Batch mode
    print("\nTest 3: Batch Contrastive Loss (in-batch negatives)")
    batch_loss_fn = BatchContrastiveLoss(model, margin=1.0)
    loss = batch_loss_fn([queries, positives])
    print(f"Loss: {loss.item():.4f}")
    
    print("\n✅ Contrastive Loss tests passed!")


if __name__ == "__main__":
    test_contrastive_loss()
