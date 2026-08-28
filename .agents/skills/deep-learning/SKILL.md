---
name: deep-learning
description: "Master foundational and modern deep learning paradigms: CNNs, Vision Transformers (ViT), Attention mechanisms, Loss functions (CrossEntropy, Focal, Triplet, Contrastive), Optimization (AdamW, Cosine Annealing), Regularization (DropPath, RMSNorm, LayerNorm), and self-supervised representation learning."
risk: unknown
source: community
date_added: '2026-02-28'
---

# Deep Learning Architecture & Mathematical Foundations

Comprehensive theoretical, architectural, and mathematical reference for deep learning paradigms, loss formulations, optimization dynamics, and modern neural network designs.

## When to Use This Skill

Use this skill when:
- Selecting or designing neural network architectures (MLP, CNN, ResNet, ConvNeXt, Vision Transformer / ViT, Swin Transformer, Autoencoders, Diffusion Models)
- Formulating specialized loss objectives (Cross-Entropy, Focal Loss, ArcFace, InfoNCE, Contrastive / Triplet Loss, Dice Loss, Smooth L1)
- Designing self-attention, multi-head attention (MHA), grouped-query attention (GQA), or flash attention layers
- Tuning optimization dynamics (AdamW, Lion, SGD with Nesterov Momentum, Cosine Annealing with Warmup, Weight Decay)
- Applying advanced regularization techniques (Stochastic Depth / DropPath, Mixup, CutMix, Label Smoothing, LayerNorm vs RMSNorm)
- Analyzing gradient dynamics, vanishing/exploding gradients, condition numbers, and neural network capacity

---

## Core Deep Learning Paradigms

### 1. Vision Transformer (ViT) & Multi-Head Self-Attention (MHSA)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

```python
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

---

### 2. Normalization & Activation Mechanisms

- **LayerNorm:** $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ (Independent of batch size, standard in NLP & Transformers).
- **RMSNorm (Root Mean Square Normalization):**
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
  (Reduces memory bandwidth and computation by omitting mean calculation).
- **SwiGLU Activation:**
  $$\text{SwiGLU}(x) = \text{Swish}(x W) \odot (x V)$$

---

### 3. Contrastive & Representation Learning (InfoNCE Loss)

$$\mathcal{L}_{\text{InfoNCE}} = - \log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\exp(\text{sim}(q, k_+) / \tau) + \sum_{j} \exp(\text{sim}(q, k_j^-) / \tau)}$$

Used in SimCLR, CLIP, MoCo, and modern multimodal embedding extractors.

---

## Optimization & Regularization Best Practices

| Technique | Mathematical Intuition | Typical Hyperparameters |
|---|---|---|
| **AdamW** | Decouples weight decay from gradient update steps | $\beta_1 = 0.9, \beta_2 = 0.999, \text{wd} = 0.05$ |
| **Cosine Annealing with Warmup** | Prevents early divergence; smoothly decays LR to $\eta_{\min}$ | $T_{\text{warmup}} = 5-10\text{ epochs}$ |
| **DropPath (Stochastic Depth)** | Randomly drops entire residual branches during training | $p_{\text{drop}} = 0.1 - 0.3$ |
| **Label Smoothing** | Replaces hard one-hot targets with $y_{\text{smooth}} = (1-\epsilon)y + \frac{\epsilon}{K}$ | $\epsilon = 0.1$ |
| **Mixup & CutMix** | Linearly interpolates pairs of inputs and labels to regularize decision boundaries | $\alpha = 0.8, \text{prob} = 0.5$ |
