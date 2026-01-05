"""
Evaluation metrics for adversarial attacks.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
from dataclasses import dataclass, asdict


@dataclass
class AttackMetrics:
    asr_05: float
    asr_06: float
    asr_07: float
    cos_sim_mean: float
    cos_sim_std: float
    cos_sim_min: float
    cos_sim_max: float
    linf_mean: float
    l2_mean: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


def compute_attack_metrics(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    adv_embeddings: torch.Tensor,
    target_centroid: torch.Tensor
) -> AttackMetrics:
    """Compute comprehensive attack metrics."""
    if target_centroid.dim() == 1:
        target_centroid = target_centroid.unsqueeze(0)
    
    cos_sims = F.cosine_similarity(adv_embeddings, target_centroid, dim=-1)
    perturbation = adv_images - clean_images
    
    return AttackMetrics(
        asr_05=(cos_sims > 0.5).float().mean().item(),
        asr_06=(cos_sims > 0.6).float().mean().item(),
        asr_07=(cos_sims > 0.7).float().mean().item(),
        cos_sim_mean=cos_sims.mean().item(),
        cos_sim_std=cos_sims.std().item(),
        cos_sim_min=cos_sims.min().item(),
        cos_sim_max=cos_sims.max().item(),
        linf_mean=perturbation.abs().view(len(clean_images), -1).max(-1)[0].mean().item(),
        l2_mean=perturbation.view(len(clean_images), -1).norm(dim=-1).mean().item()
    )


def compute_target_centroid(clip_model, dataloader, device) -> torch.Tensor:
    """Compute centroid embedding for a class."""
    all_emb = []
    with torch.no_grad():
        for images, _ in dataloader:
            emb = clip_model.encode_image(images.to(device))
            emb = emb.float()  # Ensure float32
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_emb.append(emb)
    centroid = torch.cat(all_emb).mean(dim=0)
    return centroid / centroid.norm()

