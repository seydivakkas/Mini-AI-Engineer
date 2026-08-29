"""
Day 302: Recursive Meta-Architecture Search (Differentiable NAS & Bayesian Hypernet)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Candidate Operations for Differentiable Supernet
# ---------------------------------------------------------------------------

class IdentityOp(nn.Module):
    """Identity / Skip connection operation."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.flops_factor = 0.001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ZeroOp(nn.Module):
    """Zero / None operation."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.flops_factor = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class Conv3x3Op(nn.Module):
    """Separable 3x3 Depthwise-Pointwise Convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels)
        )
        self.flops_factor = 3.0 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Conv5x5Op(nn.Module):
    """Dilated 5x5 Separable Convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels)
        )
        self.flops_factor = 5.0 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class AvgPoolOp(nn.Module):
    """1D Average Pooling operation."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.op = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.flops_factor = 0.05 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class GELULinearOp(nn.Module):
    """GELU Activated Feed-Forward Projection."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        self.flops_factor = 2.0 * channels * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> transpose for linear -> [B, L, C]
        x_trans = x.transpose(1, 2)
        out = self.fc(x_trans)
        return out.transpose(1, 2)


OPERATIONS = {
    "identity": IdentityOp,
    "zero": ZeroOp,
    "conv3x3": Conv3x3Op,
    "conv5x5": Conv5x5Op,
    "avg_pool": AvgPoolOp,
    "gelu_linear": GELULinearOp
}
OP_NAMES = list(OPERATIONS.keys())


# ---------------------------------------------------------------------------
# Supernet Cell with Continuous Relaxation & Gumbel-Softmax
# ---------------------------------------------------------------------------

class MixedOp(nn.Module):
    """Mixed operation with continuous architecture weights alpha."""
    def __init__(self, channels: int):
        super().__init__()
        self._ops = nn.ModuleList([OPERATIONS[name](channels) for name in OP_NAMES])

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class SupernetCell(nn.Module):
    """
    DAG-based Supernet Cell with N intermediate nodes.
    Continuous architecture parameter alpha is shared or searched.
    """
    def __init__(self, num_nodes: int = 4, channels: int = 32):
        super().__init__()
        self.num_nodes = num_nodes
        self.channels = channels
        self.num_edges = sum(i for i in range(1, num_nodes + 1))
        
        # Build mixed operations for every directed edge (i -> j)
        self.dag_edges = nn.ModuleList()
        for i in range(1, num_nodes + 1):
            for j in range(i):
                self.dag_edges.append(MixedOp(channels))

    def forward(self, x: torch.Tensor, alphas: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Forward pass through DAG applying Gumbel-Softmax relaxed weights.
        """
        # Gumbel Softmax over candidate operations
        weights = F.gumbel_softmax(alphas, tau=tau, hard=hard, dim=-1)
        
        node_states = [x]
        edge_idx = 0
        
        for i in range(1, self.num_nodes + 1):
            curr_state = 0.0
            for j in range(i):
                edge_weights = weights[edge_idx]
                curr_state = curr_state + self.dag_edges[edge_idx](node_states[j], edge_weights)
                edge_idx += 1
            node_states.append(curr_state)
            
        # Cell output is concatenation or mean of intermediate nodes
        return node_states[-1]


# ---------------------------------------------------------------------------
# Bayesian Hypernetwork for Dynamic Weight Generation & Uncertainty
# ---------------------------------------------------------------------------

class BayesianHypernet(nn.Module):
    """
    Generates target model weights conditioned on architecture encoding alpha.
    Outputs mean and log-variance for variational Bayesian uncertainty.
    """
    def __init__(self, alpha_dim: int, target_param_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.alpha_dim = alpha_dim
        self.target_param_dim = target_param_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(alpha_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.mean_head = nn.Linear(hidden_dim, target_param_dim)
        self.logvar_head = nn.Linear(hidden_dim, target_param_dim)

    def forward(self, alpha_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sampled_w: Sampled parameter vector via reparameterization trick
            mu: Mean of parameter distribution
            logvar: Log-variance (epistemic uncertainty)
        """
        feat = self.encoder(alpha_vec)
        mu = self.mean_head(feat)
        logvar = torch.clamp(self.logvar_head(feat), min=-10.0, max=5.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sampled_w = mu + eps * std
        return sampled_w, mu, logvar


# ---------------------------------------------------------------------------
# Config and Candidate Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureCandidate:
    """Represents a discrete or continuous architecture candidate."""
    arch_id: int
    gene: List[str]
    accuracy: float
    flops_m: float
    latency_ms: float
    entropy: float
    pareto_rank: int = 0
    crowding_distance: float = 0.0


@dataclass
class NASSearchConfig:
    """Configuration parameters for Meta-NAS."""
    num_nodes: int = 3
    channels: int = 16
    in_features: int = 16
    out_classes: int = 5
    num_epochs: int = 30
    lr_w: float = 0.025
    lr_alpha: float = 0.003
    weight_decay: float = 3e-4
    tau_init: float = 2.0
    tau_min: float = 0.2
    entropy_coeff: float = 0.01
    latency_penalty_coeff: float = 0.002
    seed: int = 42


@dataclass
class NASSearchResult:
    """Encapsulates the complete results of a Meta-NAS search."""
    best_candidate: ArchitectureCandidate
    pareto_frontier: List[ArchitectureCandidate]
    all_candidates: List[ArchitectureCandidate]
    search_history: Dict[str, List[float]]
    supernet_alpha: np.ndarray
    final_tau: float
    search_time_sec: float


# ---------------------------------------------------------------------------
# Meta-NAS Search Engine
# ---------------------------------------------------------------------------

class MetaNASEngine:
    """
    Differentiable Neural Architecture Search Engine with Bi-Level Optimization,
    Bayesian Hypernet weight estimation, and Multi-Objective Pareto Sorting.
    """
    def __init__(self, config: NASSearchConfig):
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Instantiate Supernet
        self.supernet = SupernetCell(num_nodes=config.num_nodes, channels=config.channels)
        self.classifier = nn.Linear(config.channels, config.out_classes)
        
        # Architecture parameters (alphas)
        num_edges = self.supernet.num_edges
        num_ops = len(OP_NAMES)
        self.alphas = nn.Parameter(1e-3 * torch.randn(num_edges, num_ops))
        
        # Bayesian Hypernet
        alpha_flat_dim = num_edges * num_ops
        self.hypernet = BayesianHypernet(alpha_dim=alpha_flat_dim, target_param_dim=config.channels)
        
        # Optimizers
        self.optimizer_w = torch.optim.SGD(
            list(self.supernet.parameters()) + list(self.classifier.parameters()),
            lr=config.lr_w,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
        self.optimizer_alpha = torch.optim.Adam(
            [self.alphas] + list(self.hypernet.parameters()),
            lr=config.lr_alpha,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )
        
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "alpha_entropy": [],
            "temperature": [],
            "hypernet_kl": []
        }

    def compute_flops(self, weights: torch.Tensor) -> float:
        """Estimates total FLOPs (in MegaFLOPs) based on soft operation mixture."""
        total_flops = 0.0
        edge_idx = 0
        for i in range(1, self.config.num_nodes + 1):
            for j in range(i):
                edge_w = weights[edge_idx]
                for k, op_name in enumerate(OP_NAMES):
                    op_cls = OPERATIONS[op_name]
                    factor = op_cls(self.config.channels).flops_factor
                    total_flops += float(edge_w[k].item()) * factor
                edge_idx += 1
        return total_flops / 1e3  # normalize to MFLOPs

    def compute_latency(self, flops_m: float) -> float:
        """Surrogate hardware latency model (ms)."""
        base_overhead = 0.45
        return base_overhead + flops_m * 0.12 + np.random.uniform(0.01, 0.05)

    def forward_supernet(self, x: torch.Tensor, tau: float, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs input through Supernet cell followed by classification head."""
        cell_out = self.supernet(x, self.alphas, tau=tau, hard=hard)
        # Global pooling along sequence length
        pooled = torch.mean(cell_out, dim=2)
        logits = self.classifier(pooled)
        return logits, cell_out

    def run_search(self, train_loader: List[Tuple[torch.Tensor, torch.Tensor]], 
                   val_loader: List[Tuple[torch.Tensor, torch.Tensor]]) -> NASSearchResult:
        """
        Executes Bi-Level Meta-Architecture Search.
        """
        import time
        start_time = time.time()
        
        num_epochs = self.config.num_epochs
        
        for epoch in range(num_epochs):
            # Anneal temperature: exponential decay
            progress = epoch / max(1, num_epochs - 1)
            tau = self.config.tau_init * ((self.config.tau_min / self.config.tau_init) ** progress)
            
            # --- STEP 1: Outer Loop (Update Architecture Alphas on Validation Set) ---
            val_loss_epoch = 0.0
            val_correct = 0
            val_total = 0
            
            for x_val, y_val in val_loader:
                self.optimizer_alpha.zero_grad()
                
                # Architecture forward
                logits_val, _ = self.forward_supernet(x_val, tau=tau, hard=False)
                ce_loss = F.cross_entropy(logits_val, y_val)
                
                # Softmax probabilities for entropy and regularization
                probs = F.softmax(self.alphas, dim=-1)
                log_probs = F.log_softmax(self.alphas, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1).mean()
                
                # Estimated FLOPs penalty for hardware efficiency
                flops = self.compute_flops(probs)
                lat_penalty = self.config.latency_penalty_coeff * flops
                
                # Hypernet prior regularization (KL divergence against standard normal)
                alpha_flat = self.alphas.view(-1)
                _, mu, logvar = self.hypernet(alpha_flat)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 1e-4
                
                total_val_loss = ce_loss - self.config.entropy_coeff * entropy + lat_penalty + kl_div
                total_val_loss.backward()
                self.optimizer_alpha.step()
                
                val_loss_epoch += ce_loss.item()
                preds = torch.argmax(logits_val, dim=1)
                val_correct += (preds == y_val).sum().item()
                val_total += y_val.size(0)
            
            # --- STEP 2: Inner Loop (Update Supernet Weights on Training Set) ---
            train_loss_epoch = 0.0
            for x_train, y_train in train_loader:
                self.optimizer_w.zero_grad()
                logits_train, _ = self.forward_supernet(x_train, tau=tau, hard=False)
                loss_train = F.cross_entropy(logits_train, y_train)
                loss_train.backward()
                self.optimizer_w.step()
                train_loss_epoch += loss_train.item()
            
            # Logging
            val_acc = (val_correct / max(1, val_total)) * 100.0
            avg_train_loss = train_loss_epoch / max(1, len(train_loader))
            avg_val_loss = val_loss_epoch / max(1, len(val_loader))
            
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["alpha_entropy"].append(entropy.item())
            self.history["temperature"].append(tau)
            self.history["hypernet_kl"].append(kl_div.item())
        
        # --- STEP 3: Extract Candidates & Compute Pareto Frontier ---
        all_candidates = self._sample_and_evaluate_candidates(val_loader)
        pareto_frontier = self._compute_pareto_frontier(all_candidates)
        best_candidate = pareto_frontier[0] if pareto_frontier else all_candidates[0]
        
        search_duration = time.time() - start_time
        
        return NASSearchResult(
            best_candidate=best_candidate,
            pareto_frontier=pareto_frontier,
            all_candidates=all_candidates,
            search_history=self.history,
            supernet_alpha=self.alphas.detach().cpu().numpy(),
            final_tau=tau,
            search_time_sec=search_duration
        )

    def _sample_and_evaluate_candidates(self, val_loader: List[Tuple[torch.Tensor, torch.Tensor]], 
                                       num_samples: int = 16) -> List[ArchitectureCandidate]:
        """Derives discrete architecture candidates and benchmarks them."""
        candidates = []
        
        # Candidate 1: Argmax derived best architecture from final alpha
        argmax_ops = torch.argmax(self.alphas, dim=-1).cpu().numpy()
        gene = [OP_NAMES[idx] for idx in argmax_ops]
        cand1 = self._evaluate_discrete_gene(0, gene, val_loader)
        candidates.append(cand1)
        
        # Candidates 2..N: Stochastic sampling based on softmax probabilities
        probs = F.softmax(self.alphas, dim=-1).detach().cpu().numpy()
        for c_id in range(1, num_samples):
            sampled_gene = []
            for edge_p in probs:
                chosen_idx = np.random.choice(len(OP_NAMES), p=edge_p)
                sampled_gene.append(OP_NAMES[chosen_idx])
            cand = self._evaluate_discrete_gene(c_id, sampled_gene, val_loader)
            candidates.append(cand)
            
        return candidates

    def _evaluate_discrete_gene(self, arch_id: int, gene: List[str], 
                               val_loader: List[Tuple[torch.Tensor, torch.Tensor]]) -> ArchitectureCandidate:
        """Evaluates a single discrete architecture candidate."""
        # Convert gene to one-hot hard alpha
        num_edges = len(gene)
        num_ops = len(OP_NAMES)
        hard_alpha = torch.zeros(num_edges, num_ops)
        for i, op_name in enumerate(gene):
            op_idx = OP_NAMES.index(op_name)
            hard_alpha[i, op_idx] = 1.0
            
        # Measure accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                cell_out = self.supernet(x_val, hard_alpha, tau=0.1, hard=True)
                pooled = torch.mean(cell_out, dim=2)
                logits = self.classifier(pooled)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
                
        acc = (correct / max(1, total)) * 100.0
        flops = self.compute_flops(hard_alpha)
        lat = self.compute_latency(flops)
        
        # Entropy
        entropy = 0.0  # discrete gene has 0 entropy
        
        return ArchitectureCandidate(
            arch_id=arch_id,
            gene=gene,
            accuracy=acc,
            flops_m=flops,
            latency_ms=lat,
            entropy=entropy
        )

    def _compute_pareto_frontier(self, candidates: List[ArchitectureCandidate]) -> List[ArchitectureCandidate]:
        """
        Computes the Pareto Optimal Frontier maximizing Accuracy while minimizing FLOPs and Latency.
        """
        pareto = []
        for i, c1 in enumerate(candidates):
            is_dominated = False
            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                # c2 dominates c1 if: Acc(c2) >= Acc(c1) and FLOPs(c2) <= FLOPs(c1) and Lat(c2) <= Lat(c1)
                # with at least one strict inequality
                if (c2.accuracy >= c1.accuracy and 
                    c2.flops_m <= c1.flops_m and 
                    c2.latency_ms <= c1.latency_ms and 
                    (c2.accuracy > c1.accuracy or c2.flops_m < c1.flops_m or c2.latency_ms < c1.latency_ms)):
                    is_dominated = True
                    break
            if not is_dominated:
                c1.pareto_rank = 1
                pareto.append(c1)
                
        # Sort Pareto candidates by accuracy descending
        pareto.sort(key=lambda x: x.accuracy, reverse=True)
        return pareto
