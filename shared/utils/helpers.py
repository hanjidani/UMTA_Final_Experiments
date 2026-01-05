"""
Utility functions for device setup, seeding, memory management.
"""

import torch
import numpy as np
import random
import os
import gc


def get_device(device_id: int = 0):
    """
    Get the best available device.
    Priority: CUDA > MPS > CPU
    Optimized for GPU (Kaggle/Cloud).
    
    Args:
        device_id: GPU device ID (0, 1, 2, ...) for multi-GPU setups
    """
    if torch.cuda.is_available():
        if device_id >= torch.cuda.device_count():
            device_id = 0  # Fallback to GPU 0
        device = torch.device(f"cuda:{device_id}")
        print(f"Using CUDA:{device_id}: {torch.cuda.get_device_name(device_id)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.1f} GB")
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_num_gpus() -> int:
    """Get the number of available CUDA GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_memory(device):
    """Clear GPU/MPS memory."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path



