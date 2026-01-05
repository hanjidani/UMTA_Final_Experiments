"""
Dataset loading utilities.
"""

import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import torch.nn.functional as F


class CLIPDataset(Dataset):
    """Wrapper that applies CLIP preprocessing."""
    
    def __init__(self, base_dataset, clip_preprocess):
        self.base_dataset = base_dataset
        self.clip_preprocess = clip_preprocess
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        return self.clip_preprocess(image), label


CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


def get_cifar100_class_names() -> List[str]:
    return CIFAR100_CLASSES


def load_cifar100(clip_preprocess, data_dir: str = './data') -> Tuple[Dataset, Dataset]:
    """Load CIFAR-100 with CLIP preprocessing."""
    base_transform = transforms.Compose([transforms.Resize((224, 224))])
    
    train_base = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=base_transform)
    test_base = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=base_transform)
    
    return CLIPDataset(train_base, clip_preprocess), CLIPDataset(test_base, clip_preprocess)


def load_imagenet100(clip_preprocess, data_dir: str = './data') -> Tuple[Dataset, Dataset]:
    """
    Load ImageNet-100 with CLIP preprocessing.
    
    ImageNet-100 is a subset of ImageNet with 100 classes.
    Expected structure:
        data/imagenet-100/
            train/
                class1/
                class2/
                ...
            val/
                class1/
                class2/
                ...
    
    Robust to Kaggle path nesting - checks multiple possible locations.
    """
    import os
    from torchvision.datasets import ImageFolder
    
    # Prioritize the passed data_dir (for Kaggle unified dataset)
    # Try multiple possible paths (for Kaggle compatibility)
    possible_roots = [
        Path(data_dir),  # First: Check the exact path passed (unified dataset)
        Path(data_dir) / 'imagenet-100',  # Second: Check subdirectory
        Path(data_dir) / 'ImageNet100',  # Third: Alternative naming
        Path('/kaggle/input/imagenet-100') / 'imagenet-100',  # Fallback: Kaggle input
        Path('/kaggle/input/imagenet-100'),  # Fallback: Kaggle input root
    ]
    
    real_root = None
    for p in possible_roots:
        train_path = p / 'train'
        if train_path.exists():
            real_root = p
            break
    
    if real_root is None:
        # Final fallback: assume data_dir is the root with train/val subdirectories
        real_root = Path(data_dir)
        train_dir = real_root / 'train'
        val_dir = real_root / 'val'
        if not train_dir.exists():
            # Last resort: try imagenet-100 subdirectory
            real_root = Path(data_dir) / 'imagenet-100'
            train_dir = real_root / 'train'
            val_dir = real_root / 'val'
    else:
        train_dir = real_root / 'train'
        val_dir = real_root / 'val'
    
    print(f"Dataset found at: {real_root}")
    
    # Base transform: resize to 224x224 (ImageNet images are already larger)
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    
    train_base = ImageFolder(root=str(train_dir), transform=base_transform)
    test_base = ImageFolder(root=str(val_dir), transform=base_transform)
    
    return CLIPDataset(train_base, clip_preprocess), CLIPDataset(test_base, clip_preprocess)


def get_imagenet100_class_names(dataset) -> List[str]:
    """Get class names from ImageNet-100 dataset."""
    base_dataset = dataset.base_dataset if hasattr(dataset, 'base_dataset') else dataset
    
    if hasattr(base_dataset, 'classes'):
        return base_dataset.classes
    elif hasattr(base_dataset, 'class_to_idx'):
        # ImageFolder stores classes in class_to_idx
        return sorted(base_dataset.class_to_idx.keys(), key=lambda x: base_dataset.class_to_idx[x])
    else:
        # Fallback: generate from class indices
        if hasattr(base_dataset, 'targets'):
            num_classes = len(set(base_dataset.targets))
        else:
            num_classes = len(set([dataset[i][1] for i in range(min(100, len(dataset)))]))
        return [f"class_{i}" for i in range(num_classes)]


def get_class_indices(dataset, class_id: int) -> List[int]:
    """Get all indices for a specific class."""
    indices = []
    
    # Handle different dataset types
    base_dataset = dataset.base_dataset if hasattr(dataset, 'base_dataset') else dataset
    
    # Check if dataset has targets attribute (ImageFolder, CIFAR)
    if hasattr(base_dataset, 'targets'):
        targets = base_dataset.targets
        if isinstance(targets, list):
            indices = [i for i, t in enumerate(targets) if t == class_id]
        elif isinstance(targets, torch.Tensor):
            indices = (targets == class_id).nonzero(as_tuple=True)[0].tolist()
        else:
            # Fallback: iterate through dataset
            for idx in range(len(dataset)):
                _, label = dataset.base_dataset[idx]
                if label == class_id:
                    indices.append(idx)
    else:
        # Fallback: iterate through dataset
        for idx in range(len(dataset)):
            _, label = dataset.base_dataset[idx]
            if label == class_id:
                indices.append(idx)
    
    return indices


def get_class_loader(dataset, class_index: int, batch_size: int, max_samples: Optional[int] = 500) -> DataLoader:
    """
    Create DataLoader for a specific class (simplified interface for parallel execution).
    Alias/alternative to create_class_dataloader with simpler signature.
    """
    return create_class_dataloader(dataset, class_index, batch_size, max_samples=max_samples, shuffle=True)


def create_class_dataloader(
    dataset, class_id: int, batch_size: int, 
    max_samples: Optional[int] = None, shuffle: bool = True
) -> DataLoader:
    """Create DataLoader for a specific class."""
    indices = get_class_indices(dataset, class_id)
    if max_samples and len(indices) > max_samples:
        indices = np.random.choice(indices, max_samples, replace=False).tolist()
    # Optimize num_workers based on platform
    import platform
    import torch
    
    # Optimize for GPU (CUDA) - Kaggle/Cloud ready
    if torch.cuda.is_available():
        num_workers = 4  # Multiple workers for GPU
        pin_memory = True  # Faster GPU data transfer
    elif platform.system() == "Darwin":  # macOS (MPS)
        num_workers = 2  # Conservative for macOS
        pin_memory = False  # MPS doesn't support pin_memory
    else:  # CPU fallback
        num_workers = 2
        pin_memory = False
    
    return DataLoader(
        Subset(dataset, indices), 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )


def select_diverse_pairs(
    clip_model, dataset, num_pairs: int, num_classes: int, device, samples_per_class: int = 50
) -> List[Tuple[int, int]]:
    """Select source-target pairs stratified by semantic distance."""
    from tqdm import tqdm
    
    print(f"Computing class centroids for {num_classes} classes...")
    
    centroids = {}
    for class_id in tqdm(range(num_classes), desc="Class centroids", ncols=100):
        indices = get_class_indices(dataset, class_id)[:samples_per_class]
        if not indices:
            continue
        images = torch.stack([dataset[i][0] for i in indices]).to(device)
        with torch.no_grad():
            # Use clip_model.encode_image directly (full model, not just visual)
            emb = clip_model.encode_image(images)
            emb = emb.float()  # Ensure float32
            emb = emb / emb.norm(dim=-1, keepdim=True)
            centroids[class_id] = emb.mean(dim=0)
    
    # Compute pairwise distances
    all_pairs = []
    for src in centroids:
        for tgt in centroids:
            if src != tgt:
                dist = 1 - F.cosine_similarity(centroids[src].unsqueeze(0), centroids[tgt].unsqueeze(0)).item()
                all_pairs.append({'source': src, 'target': tgt, 'distance': dist})
    
    all_pairs.sort(key=lambda x: x['distance'])
    
    # Stratified sampling
    n_bins = min(num_pairs, 5)
    bin_size = len(all_pairs) // n_bins
    pairs_per_bin = max(1, num_pairs // n_bins)
    
    selected = []
    for bin_idx in range(n_bins):
        bin_pairs = all_pairs[bin_idx * bin_size : (bin_idx + 1) * bin_size]
        for idx in np.random.choice(len(bin_pairs), min(pairs_per_bin, len(bin_pairs)), replace=False):
            selected.append((bin_pairs[idx]['source'], bin_pairs[idx]['target']))
            if len(selected) >= num_pairs:
                break
        if len(selected) >= num_pairs:
            break
    
    # Get class names (works for both CIFAR-100 and ImageNet-100)
    try:
        # Try to get from dataset
        if hasattr(dataset.base_dataset, 'classes'):
            class_names = dataset.base_dataset.classes
        else:
            # Fallback to CIFAR-100 class names
            class_names = get_cifar100_class_names()
    except:
        class_names = get_cifar100_class_names()
    
    print(f"Selected {len(selected)} pairs:")
    for src, tgt in selected:
        if src < len(class_names) and tgt < len(class_names):
            print(f"  {class_names[src]} → {class_names[tgt]}")
        else:
            print(f"  Class {src} → Class {tgt}")
    
    return selected

