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
    
    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel using torch.cdist for efficiency."""
        distances = torch.cdist(x, y) ** 2  # [B_x, B_y]
        kernel_val = torch.zeros_like(distances)
        for sigma in self.bandwidths:
            kernel_val += torch.exp(-distances / (2 * sigma**2))
        return kernel_val / len(self.bandwidths)
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD between source and target embeddings.
        Args:
            source: [B, D] adversarial embeddings
            target: [B, D] target class embeddings
        """
        B = source.size(0)
        xx = self.rbf_kernel(source, source)
        yy = self.rbf_kernel(target, target)
        xy = self.rbf_kernel(source, target)
        
        return (xx.sum() - xx.trace()) / (B * (B - 1)) \
            + (yy.sum() - yy.trace()) / (B * (B - 1)) \
            - 2 * xy.mean()


class CosineLoss(nn.Module):
    """Negative cosine similarity to target centroid."""
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_centroid = target.mean(dim=0, keepdim=True)
        # Return 1 - cosine_similarity for minimization (as per prompt specification)
        return 1 - F.cosine_similarity(source, target_centroid, dim=-1).mean()


class SinkhornLoss(nn.Module):
    """Entropy-regularized Optimal Transport (Sinkhorn) loss."""
    
    def __init__(self, epsilon: float = 0.1, num_iters: int = 20):
        """
        Args:
            epsilon: Regularization parameter (smaller = more accurate but less stable)
            num_iters: Number of Sinkhorn iterations
        """
        super().__init__()
        self.epsilon = epsilon
        self.num_iters = num_iters
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Sinkhorn divergence between source and target embeddings.
        Args:
            source: [B, D] adversarial embeddings (normalized)
            target: [B, D] or [N, D] target class embeddings (normalized)
        """
        # Cost matrix: 1 - Cosine Similarity (since embeddings are normalized)
        # source: [B, D], target: [N, D]
        cost = 1 - torch.matmul(source, target.t())  # [B, N]
        
        # Kernel matrix: K = exp(-C / epsilon)
        K = torch.exp(-cost / self.epsilon)
        
        # Initialize dual variables
        B = source.size(0)
        N = target.size(0)
        u = torch.ones(B, device=source.device) / B
        v = torch.ones(N, device=target.device) / N
        
        # Sinkhorn iterations
        for _ in range(self.num_iters):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.t() @ u + 1e-8)
        
        # Transport plan: P = diag(u) @ K @ diag(v)
        P = torch.diag(u) @ K @ torch.diag(v)
        
        # Sinkhorn divergence: <P, C>
        return torch.sum(P * cost)


class HybridLoss(nn.Module):
    """Combination of distribution loss (MMD or Sinkhorn) and Cosine loss."""
    
    def __init__(self, alpha: float = 0.5, dist_loss_type: str = "mmd", 
                 bandwidths: List[float] = [0.5, 1.0, 2.0], 
                 sinkhorn_epsilon: float = 0.1, sinkhorn_iters: int = 20):
        """
        Args:
            alpha: Weight for distribution loss (1-alpha for cosine)
            dist_loss_type: "mmd" or "sinkhorn"
            bandwidths: For MMD loss
            sinkhorn_epsilon: For Sinkhorn loss
            sinkhorn_iters: For Sinkhorn loss
        """
        super().__init__()
        self.alpha = alpha
        self.cosine = CosineLoss()
        
        if dist_loss_type == "mmd":
            self.dist_loss = MMDLoss(bandwidths)
        elif dist_loss_type == "sinkhorn":
            self.dist_loss = SinkhornLoss(sinkhorn_epsilon, sinkhorn_iters)
        else:
            raise ValueError(f"Unknown dist_loss_type: {dist_loss_type}. Use 'mmd' or 'sinkhorn'")
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.dist_loss(source, target) + (1 - self.alpha) * self.cosine(source, target)


LOSSES = {
    'mmd': MMDLoss,
    'sinkhorn': SinkhornLoss,
    'cosine': CosineLoss,
    'hybrid': HybridLoss
}


def create_loss(loss_type_or_config, **kwargs) -> nn.Module:
    """
    Create loss function by name or config dict.
    
    Args:
        loss_type_or_config: Either:
            - String: 'mmd', 'sinkhorn', 'cosine', 'hybrid'
            - Dict: {'type': 'mmd', 'bandwidths': [...]} (for backward compatibility)
        **kwargs: Loss-specific parameters (used if loss_type_or_config is a string):
            - MMDLoss: bandwidths (List[float])
            - SinkhornLoss: epsilon (float), num_iters (int)
            - CosineLoss: (no parameters)
            - HybridLoss: alpha (float), dist_loss_type (str), bandwidths, sinkhorn_epsilon, sinkhorn_iters
    """
    # Handle both string and dict config (for backward compatibility)
    if isinstance(loss_type_or_config, dict):
        config = loss_type_or_config
        loss_type = config.get('type', 'mmd')
        # Merge config kwargs with provided kwargs
        config_kwargs = {k: v for k, v in config.items() if k != 'type'}
        kwargs = {**config_kwargs, **kwargs}
    else:
        loss_type = loss_type_or_config
    
    if loss_type not in LOSSES:
        raise ValueError(f"Unknown loss: {loss_type}. Available: {list(LOSSES.keys())}")
    return LOSSES[loss_type](**kwargs)



