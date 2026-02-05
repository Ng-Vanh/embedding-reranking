

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer


class OriginalFocalInfoNCELoss(nn.Module):
    """
    Args:
        model: SentenceTransformer model
        temperature: tau parameter (default=0.05)
        margin: m parameter for hard negative mining (default=0.25)
    """
    
    def __init__(
        self,
        model: SentenceTransformer,
        temperature: float = 0.05,
        margin: float = 0.25,
    ):
        super(OriginalFocalInfoNCELoss, self).__init__()
        self.model = model
        self.temperature = temperature  # tau trong paper
        self.margin = margin  # m trong paper
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        """
        Args:
            sentence_features: List of tokenized sentences [anchor, positive]
            labels: Not used (for compatibility)
        
        Returns:
            loss: Scalar loss value
        """
        # Move features to model's device
        device = next(self.model.parameters()).device
        sentence_features = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in feature.items()} for feature in sentence_features]
        
        # 1. Encode sentences
        embeddings = [self.model(sentence_feature)['sentence_embedding'] 
                     for sentence_feature in sentence_features]
        
        anchors = embeddings[0]  # (batch_size, dim)
        positives = embeddings[1]  # (batch_size, dim)
        
        batch_size = anchors.size(0)
        
        # 2. Normalize embeddings
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        
        # 3. Compute similarity matrix: (batch_size, batch_size)
        # sim[i,j] = cosine similarity between anchor_i and positive_j
        sim_matrix = torch.mm(anchors, positives.t())
        
        # 4. Extract positive similarities (diagonal)
        positive_sim = sim_matrix.diagonal()  # (batch_size,)
        
        
        # POSITIVE TERM: (s^p)^2 / tau
        
        # Squaring để tăng gradient cho high-similarity pairs
        positive_logits = (positive_sim ** 2) / self.temperature
        
        
        # NEGATIVE TERM: s^n * (s^n + m) / tau
        
        # Mask diagonal (positive pairs)
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # Negative similarities: off-diagonal elements
        # Shape: (batch_size, batch_size)
        negative_sim = sim_matrix.clone()
        
        # Apply hard negative reweighting: s_n * (s_n + m) / tau
        negative_logits = (negative_sim * (negative_sim + self.margin)) / self.temperature
        
        # Mask out positive pairs (set to very small value)
        negative_logits = negative_logits.masked_fill(mask, -1e9)
        
        
        # CONSTRUCT LOGITS MATRIX
        
        # Logits shape: (batch_size, batch_size)
        # - Diagonal: positive logits (s^p)^2 / tau
        # - Off-diagonal: negative logits s^n(s^n+m) / tau
        
        logits = negative_logits.clone()
        logits.diagonal().copy_(positive_logits)
        
        
        # COMPUTE CROSS-ENTROPY LOSS
        
        # Target: diagonal elements (index = i for each sample i)
        target = torch.arange(batch_size, dtype=torch.long, device=device)
        
        loss = self.cross_entropy_loss(logits, target)
        
        return loss
    
    def get_config_dict(self) -> Dict[str, any]:
        """Return configuration for saving/loading"""
        return {
            'temperature': self.temperature,
            'margin': self.margin,
        }


class SimplifiedOriginalFocalInfoNCELoss(nn.Module):
    """
    Simplified Original Focal-InfoNCE Loss - Easier to understand
    
    Same formula as OriginalFocalInfoNCELoss but with clearer step-by-step computation
    """
    
    def __init__(
        self,
        model: SentenceTransformer,
        temperature: float = 0.05,
        margin: float = 0.25,
    ):
        super(SimplifiedOriginalFocalInfoNCELoss, self).__init__()
        self.model = model
        self.tau = temperature  # tau
        self.m = margin  # m
    
    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor = None):
        # Move to device
        device = next(self.model.parameters()).device
        sentence_features = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in feature.items()} for feature in sentence_features]
        
        # Step 1: Encode and normalize
        reps = [self.model(feature)['sentence_embedding'] for feature in sentence_features]
        anchor_emb = F.normalize(reps[0], p=2, dim=1)  # (B, D)
        positive_emb = F.normalize(reps[1], p=2, dim=1)  # (B, D)
        
        B = anchor_emb.size(0)
        
        # Step 2: Similarity matrix
        S = torch.mm(anchor_emb, positive_emb.t())  # (B, B)
        
        # Step 3: Positive logits - SQUARING
        s_pos = S.diagonal()  # (B,)
        logit_pos = (s_pos ** 2) / self.tau  # (s^p)^2 / tau
        
        # Step 4: Negative logits - HARD NEGATIVE REWEIGHTING
        # Create mask for negatives
        eye_mask = torch.eye(B, dtype=torch.bool, device=device)
        
        # s_neg * (s_neg + m) / tau for all pairs
        logit_neg = (S * (S + self.m)) / self.tau
        
        # Mask diagonal
        logit_neg = logit_neg.masked_fill(eye_mask, -1e9)
        
        # Step 5: Construct full logit matrix
        logits = logit_neg.clone()
        logits.diagonal().copy_(logit_pos)
        
        # Step 6: Cross-entropy (target = diagonal)
        labels = torch.arange(B, device=device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def get_config_dict(self) -> Dict[str, any]:
        return {
            'temperature': self.tau,
            'margin': self.m,
        }



# UTILITY FUNCTIONS


def compare_with_infonce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.05,
    margin: float = 0.25
) -> Dict[str, torch.Tensor]:
    """
    So sánh Focal-InfoNCE (paper) với InfoNCE baseline
    
    Returns:
        dict với các logits của từng method để visualize
    """
    anchors = F.normalize(anchors, p=2, dim=1)
    positives = F.normalize(positives, p=2, dim=1)
    
    B = anchors.size(0)
    S = torch.mm(anchors, positives.t())
    
    # InfoNCE logits
    infonce_logits = S / temperature
    
    # Focal-InfoNCE logits (paper)
    s_pos = S.diagonal()
    focal_logit_pos = (s_pos ** 2) / temperature
    focal_logit_neg = (S * (S + margin)) / temperature
    
    eye_mask = torch.eye(B, dtype=torch.bool, device=anchors.device)
    focal_logit_neg = focal_logit_neg.masked_fill(eye_mask, -1e9)
    
    focal_logits = focal_logit_neg.clone()
    focal_logits.diagonal().copy_(focal_logit_pos)
    
    return {
        'infonce_logits': infonce_logits,
        'focal_logits': focal_logits,
        'positive_sim': s_pos,
        'similarity_matrix': S,
    }


def analyze_loss_components(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.05,
    margin: float = 0.25
) -> Dict[str, any]:
    """
    Returns:
        dict với các metrics để debug/analyze
    """
    anchors = F.normalize(anchors, p=2, dim=1)
    positives = F.normalize(positives, p=2, dim=1)
    
    S = torch.mm(anchors, positives.t())
    s_pos = S.diagonal()
    
    # Positive contribution
    pos_logits = (s_pos ** 2) / temperature
    
    # Negative contributions
    eye_mask = torch.eye(S.size(0), dtype=torch.bool, device=anchors.device)
    s_neg = S.masked_fill(eye_mask, 0)
    neg_logits = (s_neg * (s_neg + margin)) / temperature
    neg_logits = neg_logits.masked_fill(eye_mask, 0)
    
    return {
        'positive_sim_mean': s_pos.mean().item(),
        'positive_sim_std': s_pos.std().item(),
        'positive_logit_mean': pos_logits.mean().item(),
        'negative_sim_mean': s_neg[~eye_mask].mean().item(),
        'negative_logit_mean': neg_logits[~eye_mask].mean().item(),
        'positive_sim_squared_mean': (s_pos ** 2).mean().item(),
        'margin_effect': (s_neg * margin / temperature)[~eye_mask].mean().item(),
    }
