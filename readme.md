# UMTA Project Checkpoint: Complete Context for AI Continuation

# BEGIN CHECKPOINT PROMPT

## Project Context: UMTA (Universal Manifold Targeted Attack)

I am working on a research project for a top-tier ML venue (ICLR/NeurIPS/ICML). The project is called **UMTA (Universal Manifold Targeted Attack)**. Here is everything you need to know to help me continue.

---

## 1. Project Overview

### 1.1 What is UMTA?

UMTA is a novel framework for generating **universal targeted adversarial perturbations** against Vision-Language Models (VLMs), specifically CLIP. 

**Key Innovation:** Instead of treating universal attacks as a fixed-vector translation (traditional UAPs) or point-wise optimization (cosine similarity to target centroid), UMTA treats the attack as a **distribution alignment problem**.

### 1.2 Core Hypothesis

> Distribution-level alignment (via MMD or Optimal Transport) produces more effective universal adversarial perturbations than point-wise alignment because it:
> 1. Captures the full geometry of the target manifold
> 2. Respects the intrinsic variance of target class
> 3. Generalizes better across the source class distribution

### 1.3 Technical Approach

```
Input: Source image x_s
       Target class distribution (images x_t)
       Frozen CLIP encoder f_θ

1. Train generator M_φ to produce perturbations:
   δ = ε · tanh(M_φ(x_s))
   
2. Create adversarial image:
   x_adv = clamp(x_s + δ, 0, 1)
   
3. Minimize distribution divergence:
   L = MMD(f_θ(x_adv), f_θ(x_t))
   or
   L = Sinkhorn(f_θ(x_adv), f_θ(x_t))

4. At inference: Apply trained M_φ to any source image
```

### 1.4 Problem Formulation

Let $f_\theta: \mathcal{X} \to \mathcal{Z}$ be CLIP's image encoder mapping to normalized embedding space $\mathcal{Z} \subset \mathbb{R}^{512}$.

**Objective:** Learn $G_\phi: \mathcal{X} \to \mathcal{X}$ such that:
1. **Manifold Alignment:** Distribution of $f_\theta(G_\phi(x_s))$ matches distribution of $f_\theta(x_t)$
2. **Imperceptibility:** $\|G_\phi(x) - x\|_\infty \leq \epsilon$

---

## 2. Current Paper Structure

### 2.1 Abstract (Draft)
```
This paper presents the Universal Manifold Targeted Attack (UMTA), a novel 
framework for generating universal targeted adversarial perturbations against 
Vision-Language Models (VLMs), specifically CLIP. Unlike traditional methods 
that optimize perturbations for individual images or minimize point-wise 
distances, UMTA trains a neural network "mapper" to align the distribution 
of perturbed source-class images with the distribution of target-class images 
in the victim model's embedding space. We achieve this by minimizing the 
Maximum Mean Discrepancy (MMD) or alternatively the Sinkhorn divergence 
(based on Optimal Transport). We evaluate UMTA on ImageNet-100 dataset 
against a frozen CLIP ViT-B-32 model. Our results demonstrate that UMTA 
achieves competitive Attack Success Rate (ASR), significantly outperforming 
traditional universal baselines.
```

### 2.2 Claimed Contributions
1. UMTA framework for distribution-aligned universal adversarial attacks
2. Demonstration that distributional approach outperforms point-wise methods
3. Analysis of geometric asymmetry in CLIP's embedding space

---

## 3. Methods to Compare

### 3.1 Baselines

| Method | Type | Description |
|--------|------|-------------|
| UAP-FGSM | Fixed-vector | Accumulated FGSM gradients |
| U-MIM | Fixed-vector | Momentum iterative method |
| U-PGD | Fixed-vector | PGD with random restarts |
| GAP-CE | Generator | Cross-entropy loss |
| GAP-Cosine | Generator | Cosine similarity to centroid |
| GAP-Margin | Generator | Triplet margin loss |
| Instance-PGD | Per-image | Upper bound reference |

### 3.2 Our Methods

| Method | Loss Function |
|--------|---------------|
| UMTA-MMD | Multi-kernel MMD with RBF |
| UMTA-Sinkhorn | Entropy-regularized optimal transport |
| UMTA-Hybrid | α·MMD + (1-α)·Cosine |

---

## 4. Key Technical Details

### 4.1 MMD Loss
```python
def compute_mmd(x, y, bandwidths=[0.5, 1.0, 2.0, 4.0]):
    """
    x: [B, D] adversarial embeddings
    y: [B, D] target embeddings
    """
    def rbf_kernel(a, b, sigmas):
        distances = torch.cdist(a, b) ** 2
        kernels = sum(torch.exp(-distances / (2 * s**2)) for s in sigmas)
        return kernels / len(sigmas)
    
    k_xx = rbf_kernel(x, x, bandwidths)
    k_yy = rbf_kernel(y, y, bandwidths)
    k_xy = rbf_kernel(x, y, bandwidths)
    
    B = x.size(0)
    mmd = (k_xx.sum() - k_xx.trace()) / (B * (B-1)) \
        + (k_yy.sum() - k_yy.trace()) / (B * (B-1)) \
        - 2 * k_xy.mean()
    return mmd
```

### 4.2 Sinkhorn Loss
```python
def compute_sinkhorn(x, y, epsilon=0.1, num_iters=50):
    """
    Entropy-regularized optimal transport
    """
    # Cost matrix (cosine distance)
    C = 1 - F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T
    
    # Sinkhorn iterations
    K = torch.exp(-C / epsilon)
    u = torch.ones(len(x), device=x.device)
    for _ in range(num_iters):
        v = 1.0 / (K.T @ u + 1e-8)
        u = 1.0 / (K @ v + 1e-8)
    
    P = torch.diag(u) @ K @ torch.diag(v)
    return (P * C).sum()
```

### 4.3 Generator Architecture Options
- **SimpleCNN**: Basic encoder-decoder (~500K params)
- **UNet**: Skip connections, 4 levels (~2M params)
- **ResUNet**: UNet + residual blocks (~3M params)
- **AttentionUNet**: UNet + attention gates (~3.5M params)

### 4.4 Perturbation Constraint
```python
def generate_adversarial(mapper, x, epsilon):
    raw_perturbation = mapper(x)
    perturbation = epsilon * torch.tanh(raw_perturbation)
    return torch.clamp(x + perturbation, 0, 1)
```

---

## 5. Experiment Plan

### 5.1 Complete Experiment List (22 experiments)

**Category A: Model Optimization (Experiments 1-4)**
1. Architecture Search - Find best generator architecture
2. Loss Function Comparison - Compare Cosine/MMD/Sinkhorn/Hybrid
3. Loss Hyperparameter Tuning - Optimize bandwidths, epsilon, alpha
4. Training Dynamics - LR, batch size, optimizer, scheduler

**Category B: Baseline Comparison (Experiments 5-7)**
5. Main Baseline Comparison - 100 pairs, 10 methods (CRITICAL)
6. Epsilon Sensitivity - ε ∈ {2,4,8,16,32}/255
7. Training Data Scaling - 50/100/200/500/1000 samples

**Category C: Ablation Studies (Experiments 8-12)**
8. Loss Component Ablation - Centroid vs MMD vs Hybrid
9. Architecture Component Ablation - Skip/ResBlocks/Attention
10. Kernel Ablation - RBF/IMQ/Polynomial/bandwidth selection
11. Batch Size Impact on MMD - 8/16/32/64/128
12. Perturbation Constraint Ablation - Tanh/Clamp/Sigmoid

**Category D: Analysis (Experiments 13-16)**
13. Embedding Space Visualization - t-SNE/UMAP plots
14. Failure Case Analysis - When/why UMTA fails
15. Geometric Asymmetry - A→B vs B→A difficulty
16. Per-Class Analysis - Easy/hard classes to attack

**Category E: Robustness (Experiments 17-19)**
17. Transferability - Transfer to other CLIP/VLM models
18. Transformation Robustness - JPEG/blur/noise/resize
19. Defense Evaluation - JPEG defense, bit reduction, etc.

**Category F: Extensions (Experiments 20-22)**
20. Multi-Target Attack - Conditional generator for any target
21. Text-Guided Attack - Use text prompts instead of images
22. Dataset Scaling - CIFAR-100 → ImageNet-100 → ImageNet-1K

### 5.2 Dual Configuration Strategy

Each experiment has two configurations:

**Mac Mini (M4, 16GB) - Development:**
- Smaller dataset (CIFAR-100, 20 classes)
- Fewer pairs (3-5)
- Fewer epochs (15-20)
- Smaller batch size (8)
- Purpose: Validate code, catch bugs

**GPU (RTX 3090) - Production:**
- Full dataset (ImageNet-100, 100 classes)
- Many pairs (20-100)
- Full epochs (30-50)
- Full batch size (32)
- Purpose: Publication-ready results

### 5.3 Execution Order
1. Mac: Exp 1-4 (find best config) → ~11h
2. Mac: Exp 5, 8, 13 (core validation) → ~6h
3. Mac: Exp 6, 10, 15, 17 (extended validation) → ~9h
4. GPU: Full versions of all experiments

---

## 6. Key Metrics

### 6.1 Primary Metrics
- **ASR@0.5**: % of adversarial images with cos_sim(z_adv, z_target) > 0.5
- **ASR@0.6**: Stricter threshold
- **ASR@0.7**: Very strict threshold
- **Cos-Sim Mean**: Average cosine similarity to target centroid

### 6.2 Secondary Metrics
- **LPIPS**: Perceptual similarity (lower = more imperceptible)
- **L∞**: Maximum perturbation (must be ≤ ε)
- **L2**: Average perturbation magnitude
- **Train Time**: Seconds per pair
- **Infer Time**: Milliseconds per image

### 6.3 Success Criteria for Publication
- UMTA beats best baseline by ≥5% ASR
- Statistical significance: p < 0.05 (paired t-test)
- Win rate ≥60% of pairs
- MMD/Sinkhorn beats Cosine by ≥3% (validates core claim)

---

## 7. Current Status

### 7.1 What Has Been Decided
- ✅ Core methodology (distribution alignment via MMD/Sinkhorn)
- ✅ Complete experiment plan (22 experiments)
- ✅ Dual Mac/GPU configuration strategy
- ✅ Baseline methods to compare against
- ✅ Metrics and success criteria
- ✅ Paper structure

### 7.2 What Needs Implementation
- [ ] Project code structure
- [ ] Dataset loading (CIFAR-100, ImageNet-100)
- [ ] Generator architectures (SimpleCNN, UNet, ResUNet, AttentionUNet)
- [ ] Loss functions (Cosine, MMD, Sinkhorn, Hybrid)
- [ ] Baseline implementations (UAP-FGSM, U-MIM, U-PGD, GAP variants)
- [ ] Training pipeline
- [ ] Evaluation metrics
- [ ] Experiment runner with checkpointing
- [ ] Visualization code (t-SNE, training curves)
- [ ] Analysis scripts

### 7.3 Hardware Available
- Mac Mini M4 (16GB unified memory) - Development
- Multiple RTX 3090 GPUs - Production (department provided)

---

## 8. Code Structure (Planned)

```
umta_project/
├── configs/
│   ├── base_config.yaml
│   ├── mac_config.yaml
│   └── gpu_config.yaml
├── src/
│   ├── models/
│   │   ├── mappers.py          # Generator architectures
│   │   └── losses.py           # MMD, Sinkhorn, Cosine, Hybrid
│   ├── attacks/
│   │   ├── umta.py             # Our method
│   │   └── baselines.py        # U-MIM, GAP, etc.
│   ├── data/
│   │   ├── datasets.py         # CIFAR-100, ImageNet loaders
│   │   └── pair_selection.py   # Strategic pair selection
│   ├── evaluation/
│   │   ├── metrics.py          # ASR, cos-sim, LPIPS, etc.
│   │   └── visualization.py    # t-SNE, training curves
│   └── utils/
│       ├── config.py           # Configuration classes
│       └── checkpoint.py       # Save/load utilities
├── scripts/
│   ├── run_optimization.py     # Experiments 1-4
│   ├── run_comparison.py       # Experiment 5
│   ├── run_ablations.py        # Experiments 8-12
│   └── run_analysis.py         # Experiments 13-16
├── experiments/
│   ├── exp01_architecture/
│   ├── exp02_loss/
│   └── ...
└── results/
    ├── optimization/
    ├── comparison/
    └── ...
```

---

## 9. Key Equations

### 9.1 Adversarial Example Generation
$$x_{adv} = \text{clamp}(x + \epsilon \cdot \tanh(M_\phi(x)), 0, 1)$$

### 9.2 MMD Loss
$$\mathcal{L}_{MMD} = \frac{1}{B^2}\sum_{i,j} k(z_s^i, z_s^j) - \frac{2}{B^2}\sum_{i,j} k(z_s^i, z_t^j) + \frac{1}{B^2}\sum_{i,j} k(z_t^i, z_t^j)$$

where $k(x,y) = \sum_\sigma \exp(-\|x-y\|^2 / 2\sigma^2)$

### 9.3 Sinkhorn Loss
$$\mathcal{L}_{Sink} = \min_{\pi \in \Pi} \langle C, \pi \rangle - \epsilon H(\pi)$$

where $C_{ij} = 1 - \cos(z_s^i, z_t^j)$

### 9.4 Hybrid Loss
$$\mathcal{L}_{Hybrid} = \alpha \cdot \mathcal{L}_{MMD} + (1-\alpha) \cdot \mathcal{L}_{Cosine}$$

---

## 10. Default Hyperparameters

| Category | Parameter | Value |
|----------|-----------|-------|
| Model | Victim | CLIP ViT-B-32 |
| Model | Generator | ResUNet, 32 base channels |
| Attack | ε | 8/255 = 0.0314 |
| Attack | Constraint | Tanh scaling |
| Loss | Type | MMD |
| Loss | Bandwidths | [0.5, 1.0, 2.0, 4.0] |
| Training | Optimizer | Adam |
| Training | LR | 1e-4 |
| Training | Batch (GPU) | 32 |
| Training | Batch (Mac) | 8 |
| Training | Epochs | 30-50 |
| Data | Train samples/class | 500 (GPU), 100 (Mac) |
| Data | Test samples/class | 100 (GPU), 20 (Mac) |

---

## 11. What I Need Help With

[FILL IN YOUR SPECIFIC REQUEST HERE]

Examples:
- "Help me implement the generator architectures in PyTorch"
- "Help me implement the MMD loss function"
- "Help me set up the experiment runner"
- "Help me implement the baseline methods"
- "Help me create the data loading pipeline"
- "Help me analyze the results from Experiment 5"
- "Help me create visualizations for the paper"

---

## 12. Important Context

1. **Fair Comparison**: Baselines must be implemented properly with same architecture/compute budget as UMTA
2. **Statistical Rigor**: Need enough pairs/seeds for significance testing
3. **Two-Phase Strategy**: Validate on Mac first, then scale to GPU
4. **Core Claim**: Must prove distribution alignment (MMD) > point-wise (Cosine)

---

## 13. Full Experiment Document

A complete 50+ page LaTeX document exists with detailed specifications for all 22 experiments, including:
- Exact hyperparameters for Mac and GPU versions
- Success criteria for each experiment
- Expected output tables and figures
- Execution timelines

If you need the full document, ask me to provide it.

---

# END CHECKPOINT PROMPT

---

## How to Continue This Project

### Option 1: Implementation Help
```
[Paste checkpoint above]

I need help implementing [specific component]. Here's what I have so far:
[paste any existing code]
```

### Option 2: Experiment Help
```
[Paste checkpoint above]

I'm running Experiment [X]. I'm getting [issue/result]. Help me:
- Debug this issue
- Analyze these results
- Decide next steps
```

### Option 3: Writing Help
```
[Paste checkpoint above]

I need help writing the [section] of the paper. Here are my results:
[paste results]
```

### Option 4: Analysis Help
```
[Paste checkpoint above]

I've completed experiments [X, Y, Z]. Here are the results:
[paste results]

Help me interpret these and decide if I'm ready for publication.
```

---

## Kaggle Setup

### Adding ImageNet-100 Dataset to Kaggle Notebooks

If you're running experiments on Kaggle and need to add the ImageNet-100 dataset:

**Option 1: Search in Kaggle**
- In your Kaggle notebook, click the "+ Add data" button
- Search for "ImageNet-100" or "imagenet100"
- Select the dataset and add it to your notebook

**Option 2: Use the Direct Link**
If search fails, you can add it via URL in your browser, then refresh the notebook page:

1. Open a new tab and go to: https://www.kaggle.com/datasets/ambityga/imagenet100
2. Click the **+ Add to Notebook** button (top right)
3. Select your notebook from the list
4. Refresh your notebook page to see the dataset added

After adding the dataset, update the dataset path in your configuration files to point to `/kaggle/input/imagenet100/` (or the appropriate path shown in Kaggle).

---

## Summary of What I Know

| Aspect | Status |
|--------|--------|
| **Core Method** | Fully designed - MMD/Sinkhorn distribution alignment |
| **Baselines** | 7 methods identified, specifications ready |
| **Experiments** | 22 experiments fully specified with Mac/GPU configs |
| **Metrics** | ASR@0.5/0.6/0.7, Cos-Sim, LPIPS, L∞, L2, timing |
| **Success Criteria** | ≥5% improvement, p<0.05, ≥60% win rate |
| **Timeline** | ~26h Mac development, ~45-65h GPU production |
| **Paper Structure** | Abstract drafted, sections planned |
| **Code Structure** | Directory layout designed, not implemented |
| **Hardware** | Mac Mini M4 + RTX 3090s available |

---