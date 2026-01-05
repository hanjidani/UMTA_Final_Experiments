"""
Loss functions for UMTA training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss with multi-scale RBF kernel."""
    
    def __init__(self, bandwidths: List[float] = [0.5, 1.0, 2.0, 4.0]):
        super().__init__()
        self.bandwidths = bandwidths
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD between source and target embeddings.
        Args:
            source: [B, D] adversarial embeddings
            target: [B, D] target class embeddings
        """
        # Squared distances
        xx = (source * source).sum(-1, keepdim=True)
        yy = (target * target).sum(-1, keepdim=True)
        
        d_ss = xx + xx.t() - 2 * source @ source.t()
        d_tt = yy + yy.t() - 2 * target @ target.t()
        d_st = xx + yy.t() - 2 * source @ target.t()
        
        # Multi-scale RBF kernel
        k_ss = sum(torch.exp(-d_ss / (2 * s**2)) for s in self.bandwidths) / len(self.bandwidths)
        k_tt = sum(torch.exp(-d_tt / (2 * s**2)) for s in self.bandwidths) / len(self.bandwidths)
        k_st = sum(torch.exp(-d_st / (2 * s**2)) for s in self.bandwidths) / len(self.bandwidths)
        
        B = source.size(0)
        mmd = (k_ss.sum() - k_ss.trace()) / (B * (B - 1)) \
            + (k_tt.sum() - k_tt.trace()) / (B * (B - 1)) \
            - 2 * k_st.mean()
        
        return mmd


class CosineLoss(nn.Module):
    """Negative cosine similarity to target centroid."""
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_centroid = target.mean(dim=0, keepdim=True)
        return -F.cosine_similarity(source, target_centroid, dim=-1).mean()


class HybridLoss(nn.Module):
    """Combination of MMD and Cosine loss."""
    
    def __init__(self, alpha: float = 0.5, bandwidths: List[float] = [0.5, 1.0, 2.0]):
        super().__init__()
        self.alpha = alpha
        self.mmd = MMDLoss(bandwidths)
        self.cosine = CosineLoss()
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mmd(source, target) + (1 - self.alpha) * self.cosine(source, target)


LOSSES = {
    'mmd': MMDLoss,
    'cosine': CosineLoss,
    'hybrid': HybridLoss
}


def create_loss(loss_type: str, **kwargs) -> nn.Module:
    """Create loss function by name."""
    if loss_type not in LOSSES:
        raise ValueError(f"Unknown loss: {loss_type}. Available: {list(LOSSES.keys())}")
    return LOSSES[loss_type](**kwargs)



