"""
Experiment 1: Architecture Search
Compares SimpleCNN, UNet, ResUNet, AttentionUNet
"""

import sys
from pathlib import Path
import json
import time
import os
import tempfile
from datetime import datetime
import argparse
import multiprocessing as mp
from functools import partial

import torch
import clip
import pandas as pd
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import get_device, get_num_gpus, set_seed, clear_memory, count_parameters, ensure_dir, load_config, save_config
from shared.models import create_mapper, create_loss
from shared.data import (
    load_cifar100, load_imagenet100,
    create_class_dataloader, select_diverse_pairs,
    get_cifar100_class_names, get_imagenet100_class_names
)
from shared.evaluation import compute_attack_metrics, compute_target_centroid


class Experiment1:
    def __init__(self, config_path: str, device_id: int = 0, pre_selected_pairs: list = None, use_multiprocessing: bool = False):
        self.config = load_config(config_path)
        self.device_id = device_id
        self.use_multiprocessing = use_multiprocessing  # Flag to disable DataLoader workers
        self.device = get_device(device_id)
        set_seed(self.config['experiment']['seed'])
        
        # Results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(__file__).parent / 'results' / timestamp
        ensure_dir(self.results_dir)
        save_config(self.config, str(self.results_dir / 'config.json'))
        
        # Load CLIP
        print("Loading CLIP...")
        self.clip_model, self.preprocess = clip.load(self.config['model']['clip_model'], device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        # Compile CLIP for faster inference on GPU (CUDA)
        # Skip compilation on older GPUs (P100, etc.) that don't support Triton
        try:
            if hasattr(torch, 'compile') and self.device.type == 'cuda':
                # Check CUDA capability (need >= 7.0 for Triton)
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability(self.device_id)
                    if capability[0] >= 7:  # Compute capability >= 7.0
                        print("    Compiling CLIP model for faster inference...")
                        self.clip_model.encode_image = torch.compile(self.clip_model.encode_image, mode='reduce-overhead')
                    else:
                        print(f"    Skipping CLIP compilation (GPU capability {capability[0]}.{capability[1]} < 7.0)")
        except Exception as e:
            print(f"    Note: Could not compile CLIP ({e}), using standard inference")
        
        # Load data based on config
        dataset_name = self.config['data']['dataset'].lower()
        print(f"Loading dataset: {dataset_name}...")
        
        # Check if config has explicit path (for Kaggle unified dataset)
        if 'path' in self.config.get('data', {}):
            data_dir = self.config['data']['path']
            print(f"Using dataset path from config: {data_dir}")
        else:
            data_dir = str(PROJECT_ROOT / 'data')
        
        if dataset_name == 'imagenet100':
            self.train_data, self.test_data = load_imagenet100(self.preprocess, data_dir)
            # Get class names from dataset
            self.class_names = get_imagenet100_class_names(self.train_data)
        elif dataset_name == 'cifar100':
            self.train_data, self.test_data = load_cifar100(self.preprocess, data_dir)
            self.class_names = get_cifar100_class_names()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'cifar100', 'imagenet100'")
        
        print(f"Loaded {len(self.train_data)} training samples, {len(self.test_data)} test samples")
        print(f"Number of classes: {len(self.class_names)}")
        
        # Select pairs (or use pre-selected pairs)
        if pre_selected_pairs is not None:
            self.pairs = pre_selected_pairs
            print(f"✅ Using pre-selected pairs: {len(self.pairs)} pair(s)")
            for idx, (src, tgt) in enumerate(self.pairs):
                print(f"   Pair {idx+1}: Class {src} → Class {tgt}")
        else:
            print("Selecting pairs from config...")
            self.pairs = select_diverse_pairs(
                self.clip_model, self.train_data,
                self.config['evaluation']['num_pairs'],
                self.config['data']['num_classes'],
                self.device
            )
        
        self.results = []
    
    def train_mapper(self, mapper, source_class, target_class):
        """Train mapper for one pair."""
        cfg = self.config
        
        # If using multiprocessing, disable DataLoader workers (daemonic processes can't spawn children)
        num_workers_override = 0 if self.use_multiprocessing else None
        
        src_loader = create_class_dataloader(
            self.train_data, source_class, cfg['training']['batch_size'],
            cfg['data']['train_samples_per_class'], num_workers=num_workers_override
        )
        tgt_loader = create_class_dataloader(
            self.train_data, target_class, cfg['training']['batch_size'],
            cfg['data']['train_samples_per_class'], num_workers=num_workers_override
        )
        
        # Pre-compute ALL target embeddings once (major speedup!)
        # Store as list of batches to match loader structure
        print("    Pre-computing target embeddings...")
        tgt_embeddings_list = []
        with torch.no_grad():
            for tgt_imgs, _ in tqdm(tgt_loader, desc="      Target embeddings", leave=False, ncols=100):
                tgt_imgs = tgt_imgs.to(self.device)
                tgt_emb = self.clip_model.encode_image(tgt_imgs)
                tgt_emb = tgt_emb / tgt_emb.norm(dim=-1, keepdim=True)
                # Ensure float32 and detach to ensure no gradient computation
                tgt_embeddings_list.append(tgt_emb.float().detach())
        
        loss_fn = create_loss(cfg['loss']['type'], bandwidths=cfg['loss']['bandwidths'])
        optimizer = torch.optim.Adam(mapper.parameters(), lr=cfg['training']['learning_rate'])
        epsilon = cfg['attack']['epsilon']
        
        losses = []
        mapper.train()
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(cfg['training']['epochs']), 
                         desc="    Training", 
                         leave=True,
                         ncols=100)
        
        for epoch in epoch_pbar:
            epoch_loss = 0
            n_batches = 0
            tgt_emb_iter = iter(tgt_embeddings_list)  # Cycle through pre-computed embeddings
            
            # Progress bar for batches
            batch_pbar = tqdm(src_loader, 
                             desc=f"      Epoch {epoch+1}", 
                             leave=False,
                             ncols=100)
            
            for src_imgs, _ in batch_pbar:
                src_imgs = src_imgs.to(self.device)
                
                # Get pre-computed target embeddings (no CLIP encoding needed!)
                try:
                    tgt_emb = next(tgt_emb_iter)
                except StopIteration:
                    tgt_emb_iter = iter(tgt_embeddings_list)
                    tgt_emb = next(tgt_emb_iter)
                
                # Generate adversarial
                pert = epsilon * torch.tanh(mapper(src_imgs))
                adv_imgs = torch.clamp(src_imgs + pert, 0, 1)
                
                # Compute adversarial embeddings through CLIP
                # Since CLIP has requires_grad=False, gradients won't flow through it
                # But we still need gradients through adv_imgs -> mapper
                # The key: adv_imgs has requires_grad=True (from mapper), so gradients will flow
                adv_emb = self.clip_model.encode_image(adv_imgs)
                adv_emb = adv_emb / adv_emb.norm(dim=-1, keepdim=True)
                # Ensure float32 for loss computation (loss functions handle this, but ensure here too)
                adv_emb = adv_emb.float()
                tgt_emb = tgt_emb.float()
                
                loss = loss_fn(adv_emb, tgt_emb)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Reduce synchronization: only call .item() once per batch
                loss_val = loss.item()
                epoch_loss += loss_val
                n_batches += 1
                
                # Update batch progress bar with current loss (less frequent updates)
                if n_batches % 1 == 0:  # Update every batch
                    batch_pbar.set_postfix({'loss': f'{loss_val:.4f}'})
            
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            
            # Update epoch progress bar with average loss
            epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}', 
                                   'best': f'{min(losses):.4f}'})
        
        return losses
    
    def evaluate_mapper(self, mapper, source_class, target_class):
        """Evaluate trained mapper."""
        cfg = self.config
        
        # If using multiprocessing, disable DataLoader workers
        num_workers_override = 0 if self.use_multiprocessing else None
        
        src_loader = create_class_dataloader(
            self.test_data, source_class, cfg['training']['batch_size'],
            cfg['data']['test_samples_per_class'], shuffle=False, num_workers=num_workers_override
        )
        tgt_loader = create_class_dataloader(
            self.test_data, target_class, cfg['training']['batch_size'],
            cfg['data']['test_samples_per_class'], shuffle=False, num_workers=num_workers_override
        )
        
        # Compute target centroid with progress
        print("    Computing target centroid...")
        target_centroid = compute_target_centroid(self.clip_model, tgt_loader, self.device)
        epsilon = cfg['attack']['epsilon']
        
        all_clean, all_adv, all_emb = [], [], []
        mapper.eval()
        
        # Progress bar for evaluation
        eval_pbar = tqdm(src_loader, desc="    Evaluating", leave=False, ncols=100)
        
        with torch.no_grad():
            for src_imgs, _ in eval_pbar:
                src_imgs = src_imgs.to(self.device)
                pert = epsilon * torch.tanh(mapper(src_imgs))
                adv_imgs = torch.clamp(src_imgs + pert, 0, 1)
                
                adv_emb = self.clip_model.encode_image(adv_imgs)
                adv_emb = adv_emb / adv_emb.norm(dim=-1, keepdim=True)
                
                all_clean.append(src_imgs.cpu())
                all_adv.append(adv_imgs.cpu())
                all_emb.append(adv_emb.cpu())
        
        return compute_attack_metrics(
            torch.cat(all_clean), torch.cat(all_adv),
            torch.cat(all_emb), target_centroid.cpu()
        )
    
    def run(self):
        print("\n" + "="*60)
        print("EXPERIMENT 1: ARCHITECTURE SEARCH")
        print("="*60)
        print(f"Total: {len(self.config['architectures'])} architectures × {len(self.pairs)} pairs = {len(self.config['architectures']) * len(self.pairs)} runs")
        print("="*60 + "\n")
        
        # Overall progress bar
        total_runs = len(self.config['architectures']) * len(self.pairs)
        overall_pbar = tqdm(total=total_runs, desc="Overall Progress", ncols=120, position=0)
        
        for arch_idx, arch_cfg in enumerate(self.config['architectures']):
            arch_name = arch_cfg['name']
            print(f"\n{'='*60}")
            print(f"Architecture {arch_idx+1}/{len(self.config['architectures'])}: {arch_name}")
            print(f"{'='*60}")
            
            for pair_idx, (src_cls, tgt_cls) in enumerate(self.pairs):
                print(f"\n  Pair {pair_idx+1}/{len(self.pairs)}: "
                      f"{self.class_names[src_cls]} → {self.class_names[tgt_cls]}")
                
                # Create mapper
                mapper = create_mapper(arch_name, **arch_cfg).to(self.device)
                n_params = count_parameters(mapper)
                print(f"    Parameters: {n_params:,}")
                
                # Train
                start = time.time()
                losses = self.train_mapper(mapper, src_cls, tgt_cls)
                train_time = time.time() - start
                
                # Evaluate
                metrics = self.evaluate_mapper(mapper, src_cls, tgt_cls)
                print(f"    ✅ Results: ASR@0.5={metrics.asr_05:.2%}, ASR@0.6={metrics.asr_06:.2%}, "
                      f"Cos-Sim={metrics.cos_sim_mean:.3f}, Time={train_time:.1f}s")
                
                self.results.append({
                    'architecture': arch_name,
                    'pair_idx': pair_idx,
                    'source_class': src_cls,
                    'target_class': tgt_cls,
                    'source_name': self.class_names[src_cls],
                    'target_name': self.class_names[tgt_cls],
                    'num_params': n_params,
                    'train_time': train_time,
                    'final_loss': losses[-1],
                    **metrics.to_dict()
                })
                
                del mapper
                clear_memory(self.device)
                
                # Update overall progress
                overall_pbar.update(1)
                overall_pbar.set_postfix({
                    'arch': arch_name[:10],
                    'ASR': f'{metrics.asr_05:.1%}'
                })
            
            self._save_intermediate()
            print(f"\n  💾 Intermediate results saved")
        
        overall_pbar.close()
        self._save_final()
    
    def _save_intermediate(self):
        pd.DataFrame(self.results).to_csv(self.results_dir / 'results.csv', index=False)
    
    def _save_final(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_dir / 'results.csv', index=False)
        
        # Summary
        summary = df.groupby('architecture').agg({
            'asr_05': ['mean', 'std'],
            'asr_06': ['mean', 'std'],
            'cos_sim_mean': ['mean', 'std'],
            'train_time': 'mean',
            'num_params': 'first'
        }).round(4)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(summary)
        
        mean_asr = df.groupby('architecture')['asr_05'].mean()
        best_arch = mean_asr.idxmax()
        best_config = next(a for a in self.config['architectures'] if a['name'] == best_arch)
        
        print(f"\n🏆 BEST: {best_arch} (ASR@0.5: {mean_asr[best_arch]:.2%})")
        
        # Save best architecture for next experiments
        with open(self.results_dir / 'best_architecture.json', 'w') as f:
            json.dump({
                'best_architecture': best_arch,
                'config': best_config,
                'asr_05': float(mean_asr[best_arch]),
                'all_results': mean_asr.to_dict()
            }, f, indent=2)
        
        summary.to_csv(self.results_dir / 'summary.csv')
        print(f"\nResults saved to: {self.results_dir}")


# Hardcoded pairs for parallel execution (10 notebooks, one per pair)
# Format: (Source Index, Target Index)
PARALLEL_PAIRS = [
    (1, 2),     # 0: Fish -> Shark (Easy)
    (12, 14),   # 1: Finch -> Bunting (Easy)
    (35, 31),   # 2: Plant -> Frog (Med-Easy)
    (56, 58),   # 3: Dog -> Cat (Med-Easy)
    (89, 81),   # 4: Truck -> Car (Medium)
    (44, 39),   # 5: Snake -> Lizard (Medium)
    (10, 95),   # 6: Bird -> Object (Med-Hard)
    (23, 67),   # 7: Bird -> Artifact (Med-Hard)
    (0, 99),    # 8: Fish -> Paper (Hard)
    (5, 92)     # 9: Fish -> Traffic Light (Hard)
]


def _run_single_architecture_wrapper(args_tuple):
    """
    Wrapper function for multiprocessing (must be at module level for pickling).
    Unpacks tuple arguments for starmap compatibility.
    """
    arch_idx, device_id, config_path, results_dir, selected_pairs = args_tuple
    print(f"[WRAPPER GPU {device_id}] DEBUG: Received selected_pairs = {selected_pairs}")
    # Convert Path to string for pickling safety
    if isinstance(results_dir, Path):
        results_dir = str(results_dir)
    return run_single_architecture(config_path, arch_idx, device_id, Path(results_dir), selected_pairs)


def run_single_architecture(config_path: str, arch_idx: int, device_id: int, results_dir: Path, selected_pairs: list = None):
    """
    Run a single architecture on a specific GPU (for multi-GPU parallel execution).
    
    Args:
        config_path: Path to config.yaml
        arch_idx: Index of architecture to run
        device_id: GPU device ID (0, 1, ...)
        results_dir: Shared results directory
        selected_pairs: Optional list of pairs to run. If None, uses pairs from config.
    """
    config = load_config(config_path)
    
    if arch_idx >= len(config['architectures']):
        return None
    
    arch_cfg = config['architectures'][arch_idx]
    arch_name = arch_cfg['name']
    
    print(f"\n[GPU {device_id}] Starting architecture: {arch_name}")
    
    # Keep original batch size since we're running only 1 architecture per GPU now
    # (No need to reduce batch size anymore)
    
    # Temporarily save modified config for Experiment1 to load
    temp_config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_config_path)
    temp_config_path.close()
    temp_config_path_str = temp_config_path.name
    
    # Create experiment instance on specific GPU
    # Pass pre_selected_pairs to avoid pair selection, and use_multiprocessing=True to disable DataLoader workers
    exp = Experiment1(
        temp_config_path_str, 
        device_id=device_id, 
        pre_selected_pairs=selected_pairs,
        use_multiprocessing=True  # Disable DataLoader workers in multiprocessing context
    )
    exp.results_dir = results_dir  # Use shared results directory
    
    # Clear memory after CLIP loading
    torch.cuda.empty_cache()
    
    # Run only this architecture
    arch_results = []
    for pair_idx, (src_cls, tgt_cls) in enumerate(exp.pairs):
        print(f"\n[GPU {device_id}] Pair {pair_idx+1}/{len(exp.pairs)}: "
              f"{exp.class_names[src_cls]} → {exp.class_names[tgt_cls]}")
        
        mapper = create_mapper(arch_name, **arch_cfg).to(exp.device)
        n_params = count_parameters(mapper)
        
        start = time.time()
        losses = exp.train_mapper(mapper, src_cls, tgt_cls)
        train_time = time.time() - start
        
        metrics = exp.evaluate_mapper(mapper, src_cls, tgt_cls)
        
        arch_results.append({
            'architecture': arch_name,
            'pair_idx': pair_idx,
            'source_class': src_cls,
            'target_class': tgt_cls,
            'source_name': exp.class_names[src_cls],
            'target_name': exp.class_names[tgt_cls],
            'num_params': n_params,
            'train_time': train_time,
            'final_loss': losses[-1],
            **metrics.to_dict()
        })
        
        del mapper
        clear_memory(exp.device)
        torch.cuda.empty_cache()  # Explicitly clear CUDA cache
    
    # Clean up temp config file
    import os
    try:
        os.unlink(temp_config_path_str)
    except:
        pass
    
    print(f"\n[GPU {device_id}] ✅ Completed architecture: {arch_name}")
    return arch_results


def run_parallel_pair(config_path: str, pair_index: int):
    """
    Run Experiment 1 for a single pair (for parallel execution across 10 notebooks).
    
    Args:
        config_path: Path to config.yaml
        pair_index: Index in PARALLEL_PAIRS (0-9)
    """
    if pair_index < 0 or pair_index >= len(PARALLEL_PAIRS):
        raise ValueError(f"Pair index {pair_index} out of range (0-{len(PARALLEL_PAIRS)-1})")
    
    src_cls, tgt_cls = PARALLEL_PAIRS[pair_index]
    print(f"\n{'='*60}")
    print(f"PARALLEL EXECUTION MODE: Pair Index {pair_index}")
    print(f"Source Class: {src_cls} -> Target Class: {tgt_cls}")
    print(f"{'='*60}\n")
    
    config = load_config(config_path)
    
    # Override pairs in config with single pair
    original_pairs = config.get('evaluation', {}).get('num_pairs', 3)
    config['evaluation'] = {'num_pairs': 1, 'pair_index': pair_index}
    
    # Create experiment instance
    exp = Experiment1(config_path)
    
    # Override pair selection with hardcoded pair
    exp.pairs = [(src_cls, tgt_cls)]
    
    # Run experiment
    exp.run()
    
    print(f"\n✅ Completed pair {pair_index}: {src_cls} -> {tgt_cls}")
    print(f"Results saved to: {exp.results_dir}")


def run_multi_gpu(config_path: str, pair_index: int = None):
    """
    Run architectures in parallel across multiple GPUs.
    Splits architectures across available GPUs.
    
    Args:
        config_path: Path to config.yaml
        pair_index: Optional pair index (0-9) to run specific pair. If None, runs all pairs from config.
    """
    num_gpus = get_num_gpus()
    if num_gpus < 2:
        print(f"⚠️  Only {num_gpus} GPU(s) available. Falling back to single-GPU mode.")
        if pair_index is not None:
            run_parallel_pair(config_path, pair_index)
        else:
            Experiment1(config_path).run()
        return
    
    config = load_config(config_path)
    num_architectures = len(config['architectures'])
    
    # Handle specific pair selection
    if pair_index is not None:
        if pair_index < 0 or pair_index >= len(PARALLEL_PAIRS):
            raise ValueError(f"Pair index {pair_index} out of range (0-{len(PARALLEL_PAIRS)-1})")
        src_cls, tgt_cls = PARALLEL_PAIRS[pair_index]
        selected_pairs = [(src_cls, tgt_cls)]
        print(f"\n{'='*60}")
        print(f"MULTI-GPU PARALLEL EXECUTION: Pair Index {pair_index}")
        print(f"Source Class: {src_cls} -> Target Class: {tgt_cls}")
    else:
        selected_pairs = None  # Will use pairs from config
        print(f"\n{'='*60}")
        print(f"MULTI-GPU PARALLEL EXECUTION: All Pairs")
    
    print(f"{'='*60}")
    print(f"GPUs Available: {num_gpus}")
    print(f"Architectures: {num_architectures}")
    num_rounds = (num_architectures + num_gpus - 1) // num_gpus
    print(f"Execution Strategy: {num_rounds} round(s), 1 architecture per GPU per round")
    print(f"{'='*60}\n")
    
    # Create shared results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / 'results' / f"multi_gpu_{timestamp}"
    ensure_dir(results_dir)
    save_config(config, str(results_dir / 'config.json'))
    
    # Distribute architectures: 1 architecture per GPU at a time (to avoid OOM)
    # If there are more architectures than GPUs, run them in rounds
    tasks = []
    for arch_idx in range(num_architectures):
        gpu_id = arch_idx % num_gpus  # Round-robin assignment
        tasks.append((arch_idx, gpu_id))
    
    # Run architectures in rounds: only num_gpus architectures at a time
    print(f"Running {num_architectures} architectures across {num_gpus} GPUs in rounds...\n")
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # CUDA cannot be re-initialized in forked subprocesses
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    # Process architectures in rounds to avoid OOM
    all_results = []
    num_rounds = (num_architectures + num_gpus - 1) // num_gpus  # Ceiling division
    
    for round_idx in range(num_rounds):
        round_start = round_idx * num_gpus
        round_end = min(round_start + num_gpus, num_architectures)
        round_tasks = tasks[round_start:round_end]
        
        print(f"\n{'='*60}")
        print(f"ROUND {round_idx + 1}/{num_rounds}: Running architectures {round_start} to {round_end - 1}")
        print(f"{'='*60}\n")
        
        # Prepare tasks with all required arguments
        # Each task is: (arch_idx, device_id, config_path, results_dir, selected_pairs)
        full_tasks = [
            (arch_idx, gpu_id, config_path, results_dir, selected_pairs)
            for arch_idx, gpu_id in round_tasks
        ]
        
        with mp.Pool(processes=len(round_tasks)) as pool:
            # Use map instead of starmap since wrapper unpacks the tuple
            round_results = pool.map(_run_single_architecture_wrapper, full_tasks)
        
        # Collect results from this round
        for arch_results in round_results:
            if arch_results:
                all_results.extend(arch_results)
        
        # Clear memory between rounds
        torch.cuda.empty_cache()
        print(f"\n✅ Round {round_idx + 1} completed. Memory cleared.\n")
    
    # Results already collected in rounds above
    
    # Save combined results
    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / 'results.csv', index=False)
    
    # Generate summary
    summary = df.groupby('architecture').agg({
        'asr_05': ['mean', 'std'],
        'asr_06': ['mean', 'std'],
        'cos_sim_mean': ['mean', 'std'],
        'train_time': 'mean',
        'num_params': 'first'
    }).round(4)
    
    print("\n" + "="*60)
    print("MULTI-GPU SUMMARY")
    print("="*60)
    print(summary)
    
    mean_asr = df.groupby('architecture')['asr_05'].mean()
    best_arch = mean_asr.idxmax()
    best_config = next(a for a in config['architectures'] if a['name'] == best_arch)
    
    print(f"\n🏆 BEST: {best_arch} (ASR@0.5: {mean_asr[best_arch]:.2%})")
    
    with open(results_dir / 'best_architecture.json', 'w') as f:
        json.dump({
            'best_architecture': best_arch,
            'config': best_config,
            'asr_05': float(mean_asr[best_arch]),
            'all_results': mean_asr.to_dict()
        }, f, indent=2)
    
    summary.to_csv(results_dir / 'summary.csv')
    print(f"\nResults saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description='UMTA Experiment 1: Architecture Search')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--pair_index', type=int, default=None, 
                       help='Pair index for parallel execution (0-9). If None, runs normal mode.')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Use multiple GPUs to train architectures in parallel')
    args = parser.parse_args()
    
    config_path = Path(__file__).parent / args.config
    
    if args.multi_gpu:
        # Multi-GPU parallel execution (architectures across GPUs)
        # Can optionally specify pair_index to run specific pair
        run_multi_gpu(str(config_path), pair_index=args.pair_index)
    elif args.pair_index is not None:
        # Parallel execution mode (single pair, single GPU)
        run_parallel_pair(str(config_path), args.pair_index)
    else:
        # Normal execution mode (all pairs from config, single GPU)
        Experiment1(str(config_path)).run()


if __name__ == '__main__':
    main()

