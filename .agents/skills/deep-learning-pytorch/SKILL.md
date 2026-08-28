---
name: deep-learning-pytorch
description: "Master PyTorch 2.x for deep learning: custom nn.Module architectures, autograd, loss functions, optimizers, learning rate schedulers, AMP (Automatic Mixed Precision), torch.compile, custom Dataset & DataLoader pipelines, DDP (DistributedDataParallel), checkpointing, and production ONNX/TorchScript export."
risk: unknown
source: community
date_added: '2026-02-28'
---

# Deep Learning with PyTorch (PyTorch 2.x Masterclass)

Comprehensive architecture and production guide for building, training, optimizing, and deploying deep learning neural networks with PyTorch 2.x.

## When to Use This Skill

Use this skill when:
- Designing custom neural network architectures (`nn.Module`, sequential, residual, attention blocks)
- Implementing customized forward passes, backward autograd hooks, or custom loss functions (Focal Loss, Dice Loss, InfoNCE, Triplet Loss)
- Building robust training loops with Automatic Mixed Precision (`torch.amp.autocast`, `GradScaler`), Gradient Clipping, and Early Stopping
- Accelerating training and inference with `torch.compile(mode="reduce-overhead")`, JIT tracing, and TorchScript
- Designing high-performance `Dataset` and `DataLoader` pipelines (`pin_memory=True`, `num_workers`, prefetching, collate functions)
- Scaling training across multiple GPUs using Distributed Data Parallel (`torch.nn.parallel.DistributedDataParallel`) or FSDP
- Managing checkpoints, state dicts, learning rate warmup/cosine annealing schedules, and metric logging
- Exporting trained models to ONNX, TensorRT, or OpenVINO for high-speed edge and cloud inference

---

## Core Capabilities & PyTorch Patterns

### 1. Robust Modular Architecture (`nn.Module`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)
```

---

### 2. High-Performance Training Loop with AMP and Gradient Clipping

```python
import torch
from torch.amp import autocast, GradScaler

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    max_norm: float = 1.0
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Memory efficient zero_grad

        # Automatic Mixed Precision Forward
        with autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Scaled Backward Pass
        scaler.scale(loss).backward()

        # Unscale for Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # Step Optimizer and Update Scaler
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)
```

---

### 3. Custom Loss Functions & Autograd Mechanics

```python
class MultiClassFocalLoss(nn.Module):
    """Focal Loss for addressing severe class imbalance."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
```

---

### 4. PyTorch 2.x `torch.compile` & Inference Optimization

```python
# Compile for up to 30-100% speedup on modern GPUs (Ampere/Hopper/Ada)
compiled_model = torch.compile(model, mode="reduce-overhead")

# Production ONNX Export
dummy_input = torch.randn(1, 3, 224, 224, device=device)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17
)
```

---

## Best Practices & Engineering Rules

1. **`set_to_none=True` in `optimizer.zero_grad()`:** Always pass `set_to_none=True` to eliminate memory writes and boost speed.
2. **`non_blocking=True` Transfers:** Use `tensor.to(device, non_blocking=True)` together with `pin_memory=True` on the `DataLoader` for asynchronous GPU copies.
3. **Avoid Unnecessary Tensor Detach / Python Scalar Syncs:** Never call `.item()` or `.cpu()` inside tight inner loops unless logging periodically.
4. **Reproducibility:** Seed `torch.manual_seed()`, `torch.cuda.manual_seed_all()`, and set `torch.backends.cudnn.deterministic = True`.
