# UMTA: Universal Manifold Targeted Attack - Technical & Theoretical Explanation

## Executive Summary

This document provides a comprehensive explanation of the **UMTA (Universal Manifold Targeted Attack)** framework, covering both theoretical foundations and technical implementation details. UMTA is a novel approach to generating universal adversarial perturbations against Vision-Language Models (specifically CLIP) by treating the attack as a **distribution alignment problem** rather than point-wise optimization.

---

## 0. Project Structure

### 0.1 Directory Organization

The UMTA project follows a modular, experiment-based structure designed for scalability and reproducibility:

```
UMTA/
├── shared/                          # Shared code library (reusable across experiments)
│   ├── __init__.py                  # Package initialization
│   │
│   ├── models/                      # Neural network architectures and losses
│   │   ├── __init__.py
│   │   ├── mappers.py              # Generator architectures (SimpleCNN, UNet, ResUNet, AttentionUNet)
│   │   └── losses.py               # Loss functions (MMDLoss, CosineLoss, HybridLoss)
│   │
│   ├── data/                        # Dataset loading and preprocessing
│   │   ├── __init__.py
│   │   └── datasets.py             # ImageNet-100, CIFAR-100 loaders, pair selection
│   │
│   ├── evaluation/                 # Metrics and evaluation utilities
│   │   ├── __init__.py
│   │   └── metrics.py              # ASR computation, cosine similarity, perturbation norms
│   │
│   ├── utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── helpers.py              # Device detection, seeding, memory management
│   │   └── config.py               # Configuration loading/saving, YAML parsing
│   │
│   └── attacks/                    # Attack implementations (placeholder for future baselines)
│       └── __init__.py
│
├── exp1_architecture/              # Experiment 1: Architecture Search
│   ├── config.yaml                 # Experiment-specific configuration
│   ├── run.py                      # Main experiment runner
│   ├── analyze.py                  # Results analysis and visualization
│   ├── logs/                       # Training logs (gitignored)
│   └── results/                    # Experiment results (gitignored)
│       └── YYYYMMDD_HHMMSS/       # Timestamped result directories
│           ├── config.json         # Saved configuration
│           ├── results.csv         # Detailed results per architecture/pair
│           ├── summary.csv         # Aggregated statistics
│           └── best_architecture.json  # Best architecture selection
│
├── notebooks/                       # Jupyter notebooks for Kaggle/Colab execution
│   └── exp1_architecture.ipynb    # Kaggle-ready notebook for Experiment 1
│
├── data/                           # Dataset storage (gitignored)
│   └── imagenet-100/               # ImageNet-100 dataset
│       ├── train/                  # Training images (organized by class)
│       └── val/                    # Validation images (organized by class)
│
├── requirements.txt                # Python dependencies
├── .gitignore                     # Git ignore rules
├── readme.md                       # Project overview and checkpoint document
├── DATASET_SETUP.md               # Dataset download and setup guide
└── TECHNICAL_EXPLANATION.md        # This document
```

### 0.2 Module Breakdown

#### **shared/models/** - Generator Architectures & Loss Functions

**`mappers.py`:**
- **SimpleCNN**: Basic encoder-decoder (~500K params)
- **UNet**: Skip connections (~2M params)
- **ResUNet**: UNet + residual blocks (~3M params)
- **AttentionUNet**: UNet + attention gates (~3.5M params)
- **`create_mapper()`**: Factory function for architecture creation

**`losses.py`:**
- **MMDLoss**: Multi-scale RBF kernel MMD
- **CosineLoss**: Negative cosine similarity to centroid
- **HybridLoss**: Weighted combination of MMD and Cosine
- **`create_loss()`**: Factory function for loss creation

#### **shared/data/** - Dataset Management

**`datasets.py`:**
- **`load_imagenet100()`**: Load ImageNet-100 with CLIP preprocessing
- **`load_cifar100()`**: Load CIFAR-100 with CLIP preprocessing
- **`create_class_dataloader()`**: Create DataLoader for specific class
- **`get_class_indices()`**: Get all indices for a class
- **`select_diverse_pairs()`**: Stratified pair selection by semantic distance
- **`get_imagenet100_class_names()`**: Extract class names from dataset
- **`get_cifar100_class_names()`**: Return CIFAR-100 class names

**Key Features:**
- Automatic CLIP preprocessing wrapper (`CLIPDataset`)
- GPU-optimized DataLoader settings (`pin_memory`, `num_workers`)
- Support for both ImageNet-100 and CIFAR-100

#### **shared/evaluation/** - Metrics & Evaluation

**`metrics.py`:**
- **`AttackMetrics`**: Dataclass for storing evaluation results
  - `asr_05`, `asr_06`, `asr_07`: Attack Success Rates at different thresholds
  - `cos_sim_mean/std/min/max`: Cosine similarity statistics
  - `linf_mean`, `l2_mean`: Perturbation magnitude metrics
- **`compute_attack_metrics()`**: Compute all metrics for a batch
- **`compute_target_centroid()`**: Compute target class centroid embedding

#### **shared/utils/** - Utility Functions

**`helpers.py`:**
- **`get_device()`**: Auto-detect best device (CUDA > MPS > CPU)
- **`set_seed()`**: Reproducible random seeding
- **`clear_memory()`**: GPU/MPS memory cleanup
- **`count_parameters()`**: Count trainable parameters
- **`ensure_dir()`**: Create directory if missing

**`config.py`:**
- **`load_config()`**: Load YAML configuration file
- **`save_config()`**: Save configuration as JSON
- **`load_previous_experiment_results()`**: Load results from previous runs

#### **exp1_architecture/** - Experiment 1 Implementation

**`config.yaml`:**
```yaml
experiment:
  name: "exp1_architecture_search"
  seed: 42

data:
  dataset: "imagenet100"
  num_classes: 50
  train_samples_per_class: 500
  test_samples_per_class: 50

attack:
  epsilon: 0.0314  # 8/255

model:
  clip_model: "ViT-B/32"

architectures:
  - name: "SimpleCNN"
    base_channels: 32
  - name: "UNet"
    base_channels: 32
  # ... etc

loss:
  type: "mmd"
  bandwidths: [0.5, 1.0, 2.0]

training:
  epochs: 15
  batch_size: 32
  learning_rate: 0.0001

evaluation:
  num_pairs: 3
```

**`run.py`:**
- **`Experiment1` class**: Main experiment orchestrator
  - `__init__()`: Load config, CLIP, dataset, select pairs
  - `train_mapper()`: Train generator for one source-target pair
  - `evaluate_mapper()`: Evaluate trained generator on test set
  - `run()`: Execute full experiment (all architectures × all pairs)
  - `_save_intermediate()`: Save results after each architecture
  - `_save_final()`: Save final summary and best architecture
- **`main()`**: CLI entry point

**`analyze.py`:**
- Load results from CSV
- Generate visualizations (bar charts, comparisons)
- Statistical analysis

#### **notebooks/** - Kaggle/Colab Integration

**`exp1_architecture.ipynb`:**
- **Cell 1**: Clone/pull latest GitHub repo
- **Cell 2**: Install dependencies from `requirements.txt`
- **Cell 3**: Run Experiment 1 (`Experiment1.run()`)
- **Cell 4**: Inspect and visualize results

**Design Philosophy:**
- Always fetch latest code from GitHub (reproducibility)
- Self-contained (no manual setup needed)
- GPU-ready (automatic CUDA detection)

### 0.3 Code Flow & Execution

**Local Execution:**
```bash
cd exp1_architecture
python run.py
```

**Kaggle Execution:**
1. Upload `notebooks/exp1_architecture.ipynb` to Kaggle
2. Enable GPU accelerator
3. Run all cells sequentially
4. Results saved in `/kaggle/working/UMTA_Final_Experiments/exp1_architecture/results/`

**Execution Flow:**
```
1. Load Configuration (config.yaml)
   ↓
2. Initialize CLIP Model (frozen)
   ↓
3. Load Dataset (ImageNet-100)
   ↓
4. Select Diverse Pairs (stratified sampling)
   ↓
5. For each Architecture:
   ├─ For each Pair:
   │  ├─ Create Generator M_φ
   │  ├─ Train M_φ (MMD loss)
   │  ├─ Evaluate on Test Set
   │  └─ Save Metrics
   └─ Save Intermediate Results
   ↓
6. Compute Summary Statistics
   ↓
7. Identify Best Architecture
   ↓
8. Save Final Results
```

### 0.4 Design Principles

**Modularity:**
- Shared code in `shared/` (reusable across experiments)
- Experiment-specific code in `expN_*/` (isolated, independent)

**Reproducibility:**
- Configuration files (YAML) for all hyperparameters
- Random seed control
- Timestamped result directories
- Saved configurations with results

**Scalability:**
- Easy to add new experiments (`exp2_loss/`, `exp3_hyperparams/`, etc.)
- Shared utilities prevent code duplication
- Factory functions for flexible architecture/loss selection

**GPU Optimization:**
- Automatic device detection (CUDA > MPS > CPU)
- Optimized DataLoader settings
- Memory management utilities
- Pre-computed embeddings for efficiency

**Experiment Management:**
- Each experiment is self-contained
- Results stored with timestamps
- Intermediate saves prevent data loss
- Analysis scripts for post-processing

### 0.5 File Dependencies

**Import Graph:**
```
exp1_architecture/run.py
  ├─ shared.utils (helpers, config)
  ├─ shared.models (mappers, losses)
  ├─ shared.data (datasets)
  └─ shared.evaluation (metrics)

shared/models/mappers.py
  └─ torch, torch.nn

shared/models/losses.py
  └─ torch, torch.nn

shared/data/datasets.py
  ├─ torch, torchvision
  ├─ clip (for preprocessing)
  └─ tqdm (progress bars)

shared/evaluation/metrics.py
  └─ torch

shared/utils/helpers.py
  └─ torch, numpy, random

shared/utils/config.py
  └─ yaml, json
```

### 0.6 Configuration System

**Hierarchical Configuration:**
- **Base config**: `exp1_architecture/config.yaml`
- **Saved config**: `exp1_architecture/results/TIMESTAMP/config.json`
- **Best architecture**: `exp1_architecture/results/TIMESTAMP/best_architecture.json`

**Configuration Loading:**
```python
from shared.utils import load_config
config = load_config('exp1_architecture/config.yaml')

# Access nested values
dataset = config['data']['dataset']
batch_size = config['training']['batch_size']
```

**Configuration Validation:**
- Required fields checked at runtime
- Type checking for numeric values
- Default values for optional parameters

---

## 1. Theoretical Foundation

### 1.1 Problem Formulation

**Traditional Universal Adversarial Perturbations (UAPs):**
- Generate a **single fixed perturbation vector** δ that works across all source images
- Optimize: `δ* = argmin_δ Σ_i L(f(x_i + δ), y_target)`
- Limitation: Assumes a linear translation in embedding space

**UMTA's Distribution Alignment Approach:**
- Learn a **neural generator** M_φ that produces image-specific perturbations
- Optimize: `φ* = argmin_φ MMD(P(f(x_s + M_φ(x_s))), P(f(x_t)))`
- Key insight: Respects the **manifold structure** of both source and target classes

### 1.2 Mathematical Framework

Let:
- **f_θ**: CLIP image encoder (frozen), mapping images to normalized embeddings: `f_θ: ℝ^(H×W×3) → S^(d-1) ⊂ ℝ^d` (d=512 for ViT-B/32)
- **x_s**: Source class images (distribution P_s)
- **x_t**: Target class images (distribution P_t)
- **M_φ**: Neural generator (trainable), producing perturbations: `M_φ: ℝ^(H×W×3) → ℝ^(H×W×3)`
- **ε**: Perturbation budget (L∞ constraint)

**Adversarial Image Generation:**
```
δ = ε · tanh(M_φ(x_s))          [Bounded perturbation]
x_adv = clamp(x_s + δ, 0, 1)    [Valid image range]
```

**Distribution Alignment Objective:**
```
L = MMD(P(f_θ(x_adv)), P(f_θ(x_t)))
```

Where MMD (Maximum Mean Discrepancy) measures the distance between two distributions in a Reproducing Kernel Hilbert Space (RKHS).

### 1.3 Maximum Mean Discrepancy (MMD) Theory

**Definition:**
MMD measures the distance between two probability distributions P and Q using their mean embeddings in an RKHS:

```
MMD²(P, Q) = ||μ_P - μ_Q||²_H
```

Where μ_P = E_{x~P}[φ(x)] is the mean embedding, and φ is a feature map induced by kernel k.

**Empirical MMD (for batches):**
Given samples {x_i} ~ P and {y_j} ~ Q:

```
MMD² = (1/m²) Σ_i Σ_j k(x_i, x_j) 
     + (1/n²) Σ_i Σ_j k(y_i, y_j)
     - (2/mn) Σ_i Σ_j k(x_i, y_j)
```

**Multi-Scale RBF Kernel:**
We use a mixture of RBF kernels with different bandwidths σ:

```
k(x, y) = (1/L) Σ_l exp(-||x - y||² / (2σ_l²))
```

**Why MMD Works:**
1. **Captures Full Distribution:** Unlike point-wise losses (cosine similarity to centroid), MMD considers all pairwise relationships
2. **Manifold-Aware:** Respects the intrinsic geometry of the embedding space
3. **Variance-Aware:** Accounts for the spread/variance of the target class distribution
4. **Differentiable:** Enables end-to-end training via backpropagation

### 1.4 Why Distribution Alignment > Point-wise Alignment

**Point-wise (Cosine Loss):**
```
L_cosine = -cos(f_θ(x_adv), μ_t)
```
- Only aligns to the **centroid** of target distribution
- Ignores variance and shape of target distribution
- May collapse adversarial embeddings to a single point

**Distribution Alignment (MMD):**
```
L_MMD = MMD(P(f_θ(x_adv)), P(f_θ(x_t)))
```
- Aligns the **entire distribution** of adversarial embeddings
- Preserves diversity and natural spread
- Better generalization across source class distribution

**Geometric Intuition:**
- Target class forms a **manifold** in embedding space (not a single point)
- Source class also forms a manifold
- UMTA learns a **mapping between manifolds** rather than point-to-point

---

## 2. Technical Implementation

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    UMTA Pipeline                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: Source Image x_s (from class C_s)                │
│         Target Images {x_t} (from class C_t)             │
│                                                          │
│  ┌──────────────┐                                        │
│  │ Generator    │                                        │
│  │ M_φ(x_s)    │ ──→ δ = ε·tanh(M_φ(x_s))              │
│  └──────────────┘                                        │
│         │                                                 │
│         ↓                                                 │
│  x_adv = clamp(x_s + δ, 0, 1)                           │
│         │                                                 │
│         ├─────────────────┐                              │
│         ↓                 ↓                              │
│  ┌─────────────┐   ┌─────────────┐                      │
│  │ CLIP Encoder│   │ CLIP Encoder│                      │
│  │ f_θ(x_adv)  │   │ f_θ(x_t)    │                      │
│  └─────────────┘   └─────────────┘                      │
│         │                 │                              │
│         └────────┬────────┘                              │
│                  ↓                                       │
│         ┌────────────────┐                              │
│         │ MMD Loss       │                              │
│         │ L = MMD(P_adv, │                              │
│         │              P_t)                             │
│         └────────────────┘                              │
│                  │                                       │
│                  ↓                                       │
│         Backpropagation                                  │
│         Update φ                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Generator Architectures

We compare four generator architectures:

#### **SimpleCNN** (~500K parameters)
- Basic encoder-decoder without skip connections
- 4-level encoding: 32 → 64 → 128 → 256 channels
- Symmetric decoding with transposed convolutions
- Baseline architecture

#### **UNet** (~2M parameters)
- U-shaped architecture with skip connections
- Preserves fine-grained spatial details
- Standard in image-to-image tasks

#### **ResUNet** (~3M parameters)
- UNet + residual blocks in each encoder/decoder level
- Better gradient flow and deeper feature extraction
- 2 residual blocks per level

#### **AttentionUNet** (~3.5M parameters)
- UNet + attention gates on skip connections
- Focuses on relevant features for perturbation generation
- Most sophisticated architecture

**Architecture Selection Rationale:**
- **SimpleCNN**: Baseline, fastest training
- **UNet**: Standard for image-to-image tasks
- **ResUNet**: Better feature extraction
- **AttentionUNet**: Most expressive, can focus on discriminative regions

### 2.3 Loss Function Implementation

#### **MMD Loss (Primary)**

```python
class MMDLoss(nn.Module):
    def forward(self, source, target):
        # source: [B, D] adversarial embeddings
        # target: [B, D] target class embeddings
        
        # Compute squared pairwise distances
        d_ss = ||source_i - source_j||²  # Within adversarial
        d_tt = ||target_i - target_j||²  # Within target
        d_st = ||source_i - target_j||²  # Cross
        
        # Multi-scale RBF kernels
        k_ss = Σ_l exp(-d_ss / (2σ_l²)) / L
        k_tt = Σ_l exp(-d_tt / (2σ_l²)) / L
        k_st = Σ_l exp(-d_st / (2σ_l²)) / L
        
        # MMD² formula
        mmd = (k_ss.sum() - trace(k_ss)) / (B(B-1))
            + (k_tt.sum() - trace(k_tt)) / (B(B-1))
            - 2 * k_st.mean()
        
        return mmd
```

**Bandwidth Selection:**
- Default: `[0.5, 1.0, 2.0, 4.0]`
- Multi-scale captures both local and global structure
- Adaptive to different scales of embedding distances

#### **Cosine Loss (Baseline Comparison)**

```python
class CosineLoss(nn.Module):
    def forward(self, source, target):
        target_centroid = target.mean(dim=0)
        return -cosine_similarity(source, target_centroid).mean()
```

#### **Hybrid Loss (Ablation)**

```python
L_hybrid = α · L_MMD + (1 - α) · L_cosine
```

### 2.4 Training Procedure

**Algorithm: UMTA Training**

```
Input: Source class C_s, Target class C_t
       CLIP model f_θ (frozen)
       Generator M_φ (trainable)
       Hyperparameters: ε, learning_rate, epochs, batch_size

1. Pre-compute target embeddings:
   For each batch {x_t} in target dataloader:
       z_t = f_θ(x_t) / ||f_θ(x_t)||
       Store z_t (detached, no gradients)

2. For epoch = 1 to epochs:
   For each batch {x_s} in source dataloader:
       
       a. Generate perturbation:
          δ = ε · tanh(M_φ(x_s))
          x_adv = clamp(x_s + δ, 0, 1)
       
       b. Encode adversarial images:
          z_adv = f_θ(x_adv) / ||f_θ(x_adv)||
       
       c. Get pre-computed target embeddings z_t
       
       d. Compute loss:
          L = MMD(z_adv, z_t)
       
       e. Backpropagate:
          ∇_φ L → Update M_φ

3. Return trained M_φ
```

**Key Optimizations:**
1. **Target Embedding Pre-computation:** Compute all target embeddings once before training (major speedup)
2. **CLIP Freezing:** `requires_grad=False` on CLIP parameters (no gradient graph building)
3. **Batch Processing:** Process multiple images simultaneously
4. **GPU Optimization:** `pin_memory=True`, multiple data loader workers

### 2.5 Evaluation Metrics

**Primary Metrics:**

1. **Attack Success Rate (ASR@τ):**
   ```
   ASR@τ = (1/N) Σ_i 1[cos(f_θ(x_adv_i), μ_t) > τ]
   ```
   - τ ∈ {0.5, 0.6, 0.7} (different thresholds)
   - Percentage of adversarial images with cosine similarity > threshold

2. **Cosine Similarity Statistics:**
   - Mean, std, min, max cosine similarity to target centroid

**Secondary Metrics:**

3. **Perturbation Magnitude:**
   - L∞ norm: `||δ||_∞` (must be ≤ ε)
   - L2 norm: `||δ||_2` (average magnitude)

4. **Training Efficiency:**
   - Training time per pair
   - Inference time per image

**Evaluation Procedure:**
```
1. Train M_φ on training set (source + target classes)
2. Evaluate on test set:
   - Generate adversarial images for test source images
   - Compute embeddings: z_adv = f_θ(x_adv)
   - Compute target centroid: μ_t = mean(f_θ(x_t_test))
   - Calculate ASR@0.5, ASR@0.6, ASR@0.7
   - Calculate cosine similarity statistics
```

---

## 3. Experiment Setup: Architecture Search

### 3.1 Experiment 1: Architecture Comparison

**Objective:** Find the best generator architecture for UMTA

**Methodology:**
- **Dataset:** ImageNet-100 (100 classes, high-resolution images)
- **Architectures:** SimpleCNN, UNet, ResUNet, AttentionUNet
- **Loss:** MMD with bandwidths [0.5, 1.0, 2.0]
- **Pairs:** 3 diverse source-target pairs (stratified by semantic distance)
- **Training:** 15 epochs, batch_size=32, lr=1e-4
- **Evaluation:** ASR@0.5, ASR@0.6, ASR@0.7, cosine similarity

**Pair Selection Strategy:**
1. Compute class centroids using CLIP embeddings
2. Compute pairwise cosine distances between all classes
3. Stratify pairs into bins (close, medium, far)
4. Sample diverse pairs from each bin

**Why This Matters:**
- Different architectures have different capacities
- Skip connections (UNet) vs. no skip connections (SimpleCNN)
- Attention mechanisms may focus on discriminative regions
- Residual blocks improve gradient flow

### 3.2 Current Configuration

**Hardware:**
- **Development:** Mac Mini M4 (16GB unified memory) - MPS
- **Production:** Kaggle GPUs (CUDA) - T4 x2 or P100

**Dataset:**
- **ImageNet-100:** Subset of ImageNet with 100 classes
- **Structure:** `data/imagenet-100/{train,val}/{class_name}/`
- **Training:** 500 samples per class
- **Testing:** 50 samples per class
- **Classes Used:** First 50 classes for pair selection (configurable)

**Training Hyperparameters:**
```yaml
attack:
  epsilon: 0.0314  # 8/255 (L∞ constraint)

training:
  epochs: 15
  batch_size: 32
  learning_rate: 0.0001

loss:
  type: "mmd"
  bandwidths: [0.5, 1.0, 2.0]
```

**Model:**
- **CLIP:** ViT-B/32 (Vision Transformer, Base, 32×32 patches)
- **Embedding Dimension:** 512
- **Frozen:** Yes (no gradients through CLIP)

---

## 4. Implementation Details

### 4.1 Code Structure

```
UMTA/
├── shared/
│   ├── models/
│   │   ├── mappers.py      # Generator architectures
│   │   └── losses.py       # MMD, Cosine, Hybrid losses
│   ├── data/
│   │   └── datasets.py      # ImageNet-100, CIFAR-100 loaders
│   ├── evaluation/
│   │   └── metrics.py      # ASR, cosine similarity, etc.
│   └── utils/
│       ├── helpers.py       # Device detection, seeding
│       └── config.py        # Configuration loading
│
├── exp1_architecture/
│   ├── config.yaml         # Experiment configuration
│   ├── run.py              # Main experiment runner
│   └── analyze.py          # Results analysis
│
└── notebooks/
    └── exp1_architecture.ipynb  # Kaggle notebook
```

### 4.2 GPU Optimization

**CUDA Optimizations:**
1. **Device Priority:** CUDA > MPS > CPU
2. **cuDNN Benchmarking:** `torch.backends.cudnn.benchmark = True`
3. **Pin Memory:** `pin_memory=True` for faster GPU transfer
4. **Multiple Workers:** `num_workers=4` for parallel data loading
5. **Persistent Workers:** Keep workers alive between epochs
6. **CLIP Compilation:** `torch.compile()` for faster inference (CUDA only)

**Memory Management:**
- Pre-compute target embeddings (avoid redundant CLIP calls)
- Detach target embeddings (no gradient graph)
- Clear GPU cache between architecture runs
- Batch processing for efficiency

### 4.3 Data Loading

**ImageNet-100 Loader:**
```python
def load_imagenet100(clip_preprocess, data_dir):
    train_dir = Path(data_dir) / 'imagenet-100' / 'train'
    val_dir = Path(data_dir) / 'imagenet-100' / 'val'
    
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # CLIP input size
    ])
    
    train = ImageFolder(train_dir, transform=base_transform)
    val = ImageFolder(val_dir, transform=base_transform)
    
    # Wrap with CLIP preprocessing
    return CLIPDataset(train, clip_preprocess), CLIPDataset(val, clip_preprocess)
```

**Class-Specific DataLoader:**
- Filters dataset to specific class
- Supports max_samples limit
- Optimized num_workers based on device

---

## 5. Theoretical Contributions

### 5.1 Novel Formulation

**Key Innovation:** Treating universal adversarial attacks as **distribution alignment** rather than:
- Fixed-vector translation (traditional UAPs)
- Point-wise optimization (cosine similarity to centroid)
- Per-image optimization (instance attacks)

**Mathematical Rigor:**
- Uses MMD, a well-established statistical distance metric
- Provides theoretical guarantees (RKHS theory)
- Differentiable end-to-end training

### 5.2 Why This Matters

**For Vision-Language Models:**
- CLIP embeddings form **manifolds** in high-dimensional space
- Classes have intrinsic variance (not single points)
- Distribution alignment respects this structure

**For Adversarial Robustness:**
- Reveals vulnerabilities in distribution-level understanding
- Shows that point-wise defenses may be insufficient
- Demonstrates importance of manifold structure

**For Universal Attacks:**
- More effective than fixed-vector methods
- Better generalization across source distribution
- Maintains natural diversity in adversarial examples

---

## 6. Expected Results & Analysis

### 6.1 Success Criteria

**Primary:**
- ASR@0.5 > 60% (competitive with baselines)
- MMD loss < Cosine loss (validates distribution approach)
- ResUNet or AttentionUNet performs best (architecture matters)

**Secondary:**
- Training time < 10 minutes per pair (on GPU)
- L∞ perturbation ≤ ε = 8/255 (constraint satisfied)
- Cosine similarity mean > 0.5 (successful alignment)

### 6.2 Analysis Plan

**Architecture Comparison:**
- Compare ASR@0.5 across architectures
- Analyze parameter count vs. performance trade-off
- Identify best architecture for subsequent experiments

**Distribution vs. Point-wise:**
- Compare MMD loss vs. Cosine loss (if both implemented)
- Analyze embedding diversity (std of cosine similarities)
- Visualize embedding distributions (t-SNE)

**Pair Analysis:**
- Compare performance across different semantic distances
- Identify which pairs are easier/harder
- Understand failure cases

---

## 7. Future Work & Extensions

### 7.1 Planned Experiments

1. **Loss Function Ablation:** MMD vs. Cosine vs. Hybrid vs. Sinkhorn
2. **Hyperparameter Search:** Bandwidths, learning rate, batch size
3. **Baseline Comparison:** UAP-FGSM, U-MIM, GAP variants
4. **Dataset Scaling:** Full ImageNet-100, ImageNet-1K
5. **Model Transferability:** Test on other CLIP variants, other VLMs

### 7.2 Potential Improvements

**Theoretical:**
- Optimal Transport (Sinkhorn) as alternative to MMD
- Wasserstein distance for better geometric properties
- Theoretical analysis of manifold alignment

**Technical:**
- Adaptive bandwidth selection
- Curriculum learning (easy pairs → hard pairs)
- Multi-scale generators
- Attention visualization

---

## 8. Reproducibility

### 8.1 Code Availability

- **Repository:** https://github.com/hanjidani/UMTA_Final_Experiments
- **License:** (To be determined)
- **Dependencies:** See `requirements.txt`

### 8.2 Running Experiments

**Local (Mac):**
```bash
cd exp1_architecture
python run.py
```

**Kaggle (GPU):**
1. Upload `notebooks/exp1_architecture.ipynb`
2. Enable GPU (T4 x2 or P100)
3. Run all cells
4. Results saved in `/kaggle/working/UMTA_Final_Experiments/exp1_architecture/results/`

### 8.3 Dataset Setup

See `DATASET_SETUP.md` for ImageNet-100 download and organization instructions.

---

## 9. Conclusion

UMTA represents a **novel theoretical and practical approach** to universal adversarial attacks by:

1. **Theoretically:** Formulating the problem as distribution alignment using MMD
2. **Practically:** Implementing efficient training with GPU optimizations
3. **Empirically:** Comparing multiple architectures to find optimal design

The framework is **ready for experimentation** on ImageNet-100 with GPU acceleration, and results will inform subsequent experiments on loss functions, hyperparameters, and baseline comparisons.

---

## References & Key Concepts

**Maximum Mean Discrepancy:**
- Gretton et al., "A Kernel Two-Sample Test" (JMLR, 2012)
- Measures distance between distributions in RKHS

**Universal Adversarial Perturbations:**
- Moosavi-Dezfooli et al., "Universal Adversarial Perturbations" (CVPR, 2017)
- Single perturbation that fools multiple images

**CLIP:**
- Radford et al., "Learning Transferable Visual Models" (ICML, 2021)
- Vision-Language model with joint image-text embeddings

**Manifold Learning:**
- Understanding that high-dimensional data lies on low-dimensional manifolds
- Adversarial examples exploit this structure

---

*Document prepared for: [Professor Name]*  
*Date: [Current Date]*  
*Author: [Your Name]*

