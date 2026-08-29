"""
Day 305: Cross-Coder Sparse Autoencoder (SAE) Engine
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration & Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CrossCoderConfig:
    """Hyperparameters for Cross-Layer Sparse Autoencoder."""
    num_layers: int = 3                # Number of adjacent model layers (e.g. L_0, L_1, L_2)
    d_model: int = 32                  # Dimension of residual stream activations
    dict_multiplier: int = 8           # Overcompleteness factor (M = dict_mult * d_model)
    top_k: int = 16                    # Top-K active features per forward pass
    l1_coeff: float = 0.002            # Group L1 sparsity penalty weight
    lr: float = 0.001                  # Adam optimizer learning rate
    batch_size: int = 64
    epochs: int = 40
    dead_feature_window: int = 200     # Steps without activation to declare a feature dead
    seed: int = 42

    @property
    def d_sae(self) -> int:
        return self.d_model * self.dict_multiplier


@dataclass
class CrossCoderResult:
    """Encapsulates training outcomes and mechanistic interpretability metrics."""
    fve_per_layer: List[float]         # Fraction of Variance Explained per layer (%)
    mean_fve: float                    # Overall average FVE (%)
    l0_sparsity: float                 # Average active latent features per sample
    dead_feature_pct: float            # Percentage of features that never fired
    cross_layer_sharing_idx: float     # Percentage of features active across >= 2 layers
    layer_norm_attributions: np.ndarray # [d_sae, num_layers] Decoder norm distribution
    history: Dict[str, List[float]]


# ---------------------------------------------------------------------------
# Synthetic Activation Generator (Simulating Polysemantic Superposition)
# ---------------------------------------------------------------------------

class SyntheticActivationGenerator:
    """
    Generates synthetic neural activations across K layers exhibiting:
    1. Ground-truth sparse latent features (superposition).
    2. Shared cross-layer circuit motifs (e.g. feature A persists L0->L2).
    3. Layer-specific transient features (e.g. feature B only active in L1).
    """
    def __init__(self, num_layers: int = 3, d_model: int = 32, num_true_concepts: int = 128, seed: int = 42):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_true_concepts = num_true_concepts
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Ground-truth dictionary vectors for each concept per layer: [K, d_model, num_true_concepts]
        self.true_directions = torch.randn(num_layers, d_model, num_true_concepts)
        # Normalize directions
        self.true_directions = F.normalize(self.true_directions, dim=1)
        
        # Cross-layer concept activity mask (some concepts active in all layers, some in single layers)
        self.concept_layer_mask = torch.ones(num_layers, num_true_concepts)
        if num_layers >= 3:
            for c in range(num_true_concepts):
                if c % 3 == 0:  # Layer 0 & 1 only
                    self.concept_layer_mask[2, c] = 0.0
                elif c % 3 == 1:  # Layer 1 & 2 only
                    self.concept_layer_mask[0, c] = 0.0

    def generate_batch(self, batch_size: int = 128, sparsity_p: float = 0.05) -> torch.Tensor:
        """
        Returns stacked layer activations [batch_size, num_layers, d_model].
        """
        # Bernoulli-Exponential sparse concept firing
        firing_mask = (torch.rand(batch_size, self.num_true_concepts) < sparsity_p).float()
        amplitudes = torch.distributions.Exponential(1.0).sample((batch_size, self.num_true_concepts))
        c_act = firing_mask * amplitudes  # [B, num_true_concepts]
        
        # Mix directions per layer
        activations = []
        for l in range(self.num_layers):
            layer_mask = self.concept_layer_mask[l]  # [num_true_concepts]
            effective_c = c_act * layer_mask.unsqueeze(0)  # [B, num_true_concepts]
            # Matrix multiplication: [B, C] @ [C, d_model] = [B, d_model]
            act_l = effective_c @ self.true_directions[l].T
            # Add small sensory noise
            act_l += 0.05 * torch.randn_like(act_l)
            activations.append(act_l)
            
        return torch.stack(activations, dim=1)  # [B, num_layers, d_model]


# ---------------------------------------------------------------------------
# Cross-Coder Sparse Autoencoder Module
# ---------------------------------------------------------------------------

class CrossCoderSAE(nn.Module):
    """
    Anthropic-style Cross-Coder Sparse Autoencoder:
    Jointly encodes multi-layer activations into a shared overcomplete dictionary,
    and decodes back into distinct layer reconstruction subspaces.
    """
    def __init__(self, config: CrossCoderConfig):
        super().__init__()
        self.config = config
        K = config.num_layers
        d_in = config.d_model
        d_sae = config.d_sae
        
        # Encoders per layer: [d_sae, d_in]
        self.W_enc = nn.Parameter(torch.randn(K, d_sae, d_in) * (1.0 / np.sqrt(d_in)))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        # Decoders per layer: [K, d_in, d_sae]
        self.W_dec = nn.Parameter(torch.randn(K, d_in, d_sae) * (1.0 / np.sqrt(d_sae)))
        self.b_dec = nn.Parameter(torch.zeros(K, d_in))
        
        # Normalize decoder weights initially
        self.normalize_decoder()
        
        # Tracking feature activation counts for dead-neuron detection
        self.register_buffer("activation_counts", torch.zeros(d_sae))

    def normalize_decoder(self):
        """Projects decoder column vectors to unit norm: ||W_dec^{(l)}_{:, j}||_2 = 1."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_layers, d_model]
        Returns latent activations h: [B, d_sae] using TopK activation.
        """
        # Centering: (x^{(l)} - b_dec^{(l)})
        x_centered = x - self.b_dec.unsqueeze(0)  # [B, K, d_in]
        
        # Summed linear projection: sum_l (x_centered_l @ W_enc_l^T)
        # Einsum: b=batch, k=layer, d=d_in, m=d_sae -> [B, d_sae]
        pre_acts = torch.einsum("bkd,kmd->bm", x_centered, self.W_enc) + self.b_enc
        
        # Top-K Sparsity mechanism (or ReLU + TopK)
        relu_acts = F.relu(pre_acts)
        if self.config.top_k > 0 and self.config.top_k < self.config.d_sae:
            topk_vals, topk_indices = torch.topk(relu_acts, k=self.config.top_k, dim=-1)
            h = torch.zeros_like(relu_acts)
            h.scatter_(-1, topk_indices, topk_vals)
        else:
            h = relu_acts
            
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, d_sae]
        Returns reconstructed layers x_hat: [B, num_layers, d_model]
        """
        # Einsum: b=batch, m=d_sae, k=layer, d=d_in -> [B, K, d_in]
        x_hat = torch.einsum("bm,kdm->bkd", h, self.W_dec) + self.b_dec.unsqueeze(0)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder forward pass returning (x_hat, h).
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes normalized MSE reconstruction loss + Group L1 sparsity penalty.
        """
        # 1. Reconstruction Loss per layer
        diff = x - x_hat  # [B, K, d_in]
        recon_loss = torch.mean(diff ** 2)
        
        # 2. Cross-layer Group L1 Sparsity Penalty
        # Decoder norm for feature j across all layers: sqrt(sum_l ||W_dec^{(l)}_{:, j}||^2)
        dec_norms = torch.norm(self.W_dec, dim=1)  # [K, d_sae]
        group_weights = torch.sqrt(torch.sum(dec_norms ** 2, dim=0))  # [d_sae]
        
        l1_penalty = torch.mean(torch.sum(h * group_weights.unsqueeze(0), dim=-1))
        total_loss = recon_loss + self.config.l1_coeff * l1_penalty
        
        # Update feature usage
        with torch.no_grad():
            self.activation_counts += (h > 0).float().sum(dim=0)
            
        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "l1_loss": l1_penalty.item(),
            "l0_sparsity": (h > 0).float().sum(dim=-1).mean().item()
        }
        return total_loss, metrics


# ---------------------------------------------------------------------------
# Cross-Coder Trainer & Evaluator
# ---------------------------------------------------------------------------

class CrossCoderTrainer:
    """
    Trains Cross-Coder SAE on multi-layer synthetic activations and
    evaluates mechanistically interpretable feature metrics.
    """
    def __init__(self, config: CrossCoderConfig):
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        self.model = CrossCoderSAE(config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=1e-5)
        self.history: Dict[str, List[float]] = {
            "total_loss": [],
            "recon_loss": [],
            "l1_loss": [],
            "l0_sparsity": []
        }

    def train_epoch(self, dataloader: List[torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        epoch_metrics: Dict[str, float] = {k: 0.0 for k in self.history.keys()}
        
        for batch_x in dataloader:
            self.optimizer.zero_grad()
            x_hat, h = self.model(batch_x)
            loss, metrics = self.model.compute_loss(batch_x, x_hat, h)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Constrain decoder norms
            self.model.normalize_decoder()
            
            for k, v in metrics.items():
                epoch_metrics[k] += v / len(dataloader)
                
        for k, v in epoch_metrics.items():
            self.history[k].append(v)
            
        return epoch_metrics

    def evaluate(self, test_x: torch.Tensor) -> CrossCoderResult:
        """
        Computes Fraction of Variance Explained (FVE), dead features, and cross-layer sharing.
        """
        self.model.eval()
        with torch.no_grad():
            x_hat, h = self.model(test_x)
            
            # 1. Fraction of Variance Explained (FVE = 1 - Var(x - x_hat)/Var(x))
            fve_list = []
            for l in range(self.config.num_layers):
                x_l = test_x[:, l, :]
                x_hat_l = x_hat[:, l, :]
                var_err = torch.var(x_l - x_hat_l).item()
                var_total = torch.var(x_l).item()
                fve = max(0.0, 1.0 - var_err / max(1e-6, var_total)) * 100.0
                fve_list.append(fve)
                
            mean_fve = float(np.mean(fve_list))
            l0 = float((h > 0).float().sum(dim=-1).mean().item())
            
            # 2. Dead features (% of latent neurons that never fired)
            total_active_per_feature = (h > 0).float().sum(dim=0)
            dead_count = (total_active_per_feature == 0).sum().item()
            dead_pct = (dead_count / self.config.d_sae) * 100.0
            
            # 3. Cross-layer sharing index (% of features with decoder norm > 0.3 in >= 2 layers)
            dec_norms = torch.norm(self.model.W_dec, dim=1).cpu().numpy()  # [K, d_sae]
            active_layers_per_feat = np.sum(dec_norms > 0.30, axis=0)  # [d_sae]
            shared_count = np.sum(active_layers_per_feat >= 2)
            sharing_idx = (shared_count / self.config.d_sae) * 100.0
            
            return CrossCoderResult(
                fve_per_layer=fve_list,
                mean_fve=mean_fve,
                l0_sparsity=l0,
                dead_feature_pct=dead_pct,
                cross_layer_sharing_idx=sharing_idx,
                layer_norm_attributions=dec_norms.T,  # [d_sae, K]
                history=self.history
            )
