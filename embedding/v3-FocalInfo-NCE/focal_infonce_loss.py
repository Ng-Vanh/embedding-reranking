"""
Focal-InfoNCE Loss Implementation
Paper: "Improving Unsupervised Sentence Embeddings with Focal-InfoNCE"
URL: https://openreview.net/pdf?id=j48JCRagwR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss


class FocalInfoNCELoss(nn.Module):
    """
    Args:
        model: SentenceTransformer model
        scale: Temperature parameter (tau), default=20.0
        similarity_fct: Similarity function (cosine, dot product, etc.)
        margin: Margin parameter (m) for hard negative mining, default=0.25
        gamma_pos: Focal parameter for positive pairs, default=1.0
        gamma_neg: Focal parameter for negative pairs, default=1.0
    """
    
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct=None,
        margin: float = 0.25,
        gamma_pos: float = 1.0,
        gamma_neg: float = 1.0,
    ):
        super(FocalInfoNCELoss, self).__init__()
        self.model = model
        self.scale = scale
        self.tau = 1.0 / scale  # tau = 1/scale, thường tau = 0.05
        self.similarity_fct = similarity_fct if similarity_fct is not None else lambda x, y: F.cosine_similarity(x, y)
        self.margin = margin  # m trong paper
        self.gamma_pos = gamma_pos  # gamma cho positive pairs
        self.gamma_neg = gamma_neg  # gamma cho negative pairs
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        """
        Args:
            sentence_features: List of tokenized sentences [anchor, positive]
            labels: Not used, for compatibility with SentenceTransformer
        
        Returns:
            loss: Scalar loss value
        """
        # Move features to model's device
        device = next(self.model.parameters()).device
        sentence_features = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in feature.items()} for feature in sentence_features]
        
        # Encode sentences
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        # anchors: queries, positives: positive documents
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        positives = embeddings[1]  # (batch_size, embedding_dim)
        
        batch_size = anchors.size(0)
        
        # Normalize embeddings
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        
        # Compute similarity matrix: (batch_size, batch_size)
        # sim[i,j] = similarity between anchor_i and positive_j
        similarity_matrix = torch.mm(anchors, positives.t())
        
        # Positive pairs: diagonal elements
        positive_sim = torch.diagonal(similarity_matrix)  # (batch_size,)
        
        
        # FOCAL-INFONCE: Positive pairs re-weighting
        # Giảm trọng số cho positive có similarity thấp (do dropout noise)
        
        # w_pos = exp(-gamma_pos * (1 - s_pos))
        # Nếu s_pos thấp → w_pos nhỏ → giảm penalty
        positive_weights = torch.exp(-self.gamma_pos * (1 - positive_sim))
        
        # Positive logits with weighting
        positive_logits = positive_sim / self.tau
        weighted_positive_logits = positive_logits * positive_weights
        
        
        # FOCAL-INFONCE: Negative pairs re-weighting
        # Tăng trọng số cho negative có similarity cao (hard negatives)
        
        # Mask out diagonal (positive pairs)
        mask = torch.eye(batch_size, dtype=torch.bool, device=anchors.device)
        
        # Negative similarities: all off-diagonal elements
        negative_sim = similarity_matrix.masked_fill(mask, -1e9)  # Mask positives
        
        # w_neg = exp(s_neg * (s_neg + m) / tau)
        # s_neg càng cao → w_neg càng lớn → focus vào hard negatives
        negative_weights = torch.exp(
            negative_sim * (negative_sim + self.margin) / self.tau
        )
        
        # Mask positive positions
        negative_weights = negative_weights.masked_fill(mask, 0)
        
        # Negative logits with weighting
        negative_logits = negative_sim / self.tau
        weighted_negative_logits = negative_logits + torch.log(negative_weights + 1e-10)
        
        
        # Compute Focal-InfoNCE Loss
        
        # Logits matrix: [weighted_positive | weighted_negatives]
        # Shape: (batch_size, batch_size)
        logits = torch.zeros_like(similarity_matrix)
        
        # Fill diagonal with weighted positive logits
        logits.diagonal().copy_(weighted_positive_logits)
        
        # Fill off-diagonal with weighted negative logits
        logits = logits.masked_scatter(~mask, weighted_negative_logits[~mask])
        
        # Labels: positive pairs are on diagonal (index = i)
        labels = torch.arange(batch_size, dtype=torch.long, device=anchors.device)
        
        # Cross-entropy loss
        loss = self.cross_entropy_loss(logits, labels)
        
        return loss
    
    def get_config_dict(self) -> Dict[str, any]:
        """Return configuration for saving/loading"""
        return {
            'scale': self.scale,
            'margin': self.margin,
            'gamma_pos': self.gamma_pos,
            'gamma_neg': self.gamma_neg,
        }


class SimplifiedFocalInfoNCELoss(nn.Module):
    """
    Implementation đơn giản hóa với các bước rõ ràng:
    1. Tính similarity matrix
    2. Re-weight positive pairs (dropout noise aware)
    3. Re-weight negative pairs (hard negative mining)
    4. Compute cross-entropy loss
    """
    
    def __init__(
        self,
        model: SentenceTransformer,
        temperature: float = 0.05,
        margin: float = 0.25,
        gamma_pos: float = 1.0,
        gamma_neg: float = 1.0,
    ):
        super(SimplifiedFocalInfoNCELoss, self).__init__()
        self.model = model
        self.temperature = temperature  # tau
        self.margin = margin  # m
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
    
    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        # 1. Encode and normalize
        # Move features to model's device
        device = next(self.model.parameters()).device
        sentence_features = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in feature.items()} for feature in sentence_features]
        
        reps = [self.model(feature)['sentence_embedding'] for feature in sentence_features]
        anchor_embeddings = F.normalize(reps[0], p=2, dim=1)  # (B, D)
        positive_embeddings = F.normalize(reps[1], p=2, dim=1)  # (B, D)
        
        batch_size = anchor_embeddings.size(0)
        
        # 2. Compute similarity matrix
        sim_matrix = torch.mm(anchor_embeddings, positive_embeddings.t())  # (B, B)
        
        # 3. Extract positive and negative similarities
        positive_sim = sim_matrix.diagonal()  # (B,)
        
        # Create mask for negatives (all except diagonal)
        mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
        
        # 4. Positive re-weighting (dropout noise aware)
        # Giảm trọng số nếu positive similarity thấp
        pos_weight = torch.exp(-self.gamma_pos * (1 - positive_sim))
        pos_logits = (positive_sim / self.temperature) * pos_weight
        
        # 5. Negative re-weighting (hard negative mining)
        # Tăng trọng số nếu negative similarity cao (hard negatives)
        neg_logits = sim_matrix.masked_fill(mask, -float('inf')) / self.temperature
        
        # Apply focal weighting to negatives
        neg_sim_for_weight = sim_matrix.masked_fill(mask, 0)  # Zero out positives
        neg_weight = torch.exp(neg_sim_for_weight * (neg_sim_for_weight + self.margin) / self.temperature)
        neg_weight = neg_weight.masked_fill(mask, 0)
        
        # Weighted negative logits
        neg_logits_weighted = neg_logits + torch.log(neg_weight + 1e-10).masked_fill(mask, 0)
        
        # 6. Combine positive and negative logits
        # Construct full logit matrix
        logits = neg_logits_weighted.clone()
        logits.diagonal().copy_(pos_logits)
        
        # 7. Cross-entropy loss (target is diagonal)
        target = torch.arange(batch_size, device=logits.device)
        loss = F.cross_entropy(logits, target)
        
        return loss
    
    def get_config_dict(self) -> Dict[str, any]:
        return {
            'temperature': self.temperature,
            'margin': self.margin,
            'gamma_pos': self.gamma_pos,
            'gamma_neg': self.gamma_neg,
        }
