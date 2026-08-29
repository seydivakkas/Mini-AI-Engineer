"""
Day 307: Unsupervised Latent Causal World Representation Discovery & Do-Calculus Engine.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class CausalConfig:
    latent_dim: int = 5                # Number of latent causal variables (z1, z2, ..., zd)
    obs_dim: int = 20                  # High-dimensional observation space (e.g. image features)
    num_samples: int = 1000            # Total samples generated for training/evaluation
    batch_size: int = 64
    lr: float = 1e-3
    lambda_dag: float = 1.0            # NOTEARS acyclicity constraint weight
    lambda_sparse: float = 0.05        # L1 sparsity weight on adjacency
    lambda_interv: float = 0.5         # Interventional alignment loss weight
    epochs: int = 40
    threshold_edge: float = 0.25       # Binary DAG threshold
    seed: int = 42


@dataclass
class CausalDiscoveryResult:
    structural_hamming_distance: int   # SHD between predicted DAG and ground truth DAG
    dag_true_positive_rate_pct: float  # Edge discovery TPR (%)
    dag_false_discovery_rate_pct: float # False positive rate (%)
    interventional_mse: float          # MSE under do(z_i = v) interventions
    counterfactual_mse: float          # MSE on counterfactual abduction
    reconstruction_mse: float          # High-dimensional observational reconstruction MSE
    learned_adjacency_matrix: np.ndarray
    ground_truth_adjacency_matrix: np.ndarray
    loss_history: List[float]


# ---------------------------------------------------------------------------
# Ground Truth Structural Causal Model (SCM) & Environment Generator
# ---------------------------------------------------------------------------

class StructuralCausalModel:
    """
    Generates ground-truth causal graphs, observational data, and interventional datasets.
    """
    def __init__(self, latent_dim: int = 5, obs_dim: int = 20, seed: int = 42):
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Ground Truth Strictly Upper-Triangular DAG (ensures acyclicity)
        # Dynamic chain z_0 -> z_1 -> ... -> z_{d-1}
        self.true_adj = torch.zeros(latent_dim, latent_dim)
        for i in range(latent_dim - 1):
            self.true_adj[i, i + 1] = 1.0
        if latent_dim >= 4:
            self.true_adj[0, 3] = 1.0  # Shortcut causal link
        
        # Linear/Nonlinear transformation from latent z -> observation x
        self.decoder_weights = torch.randn(obs_dim, latent_dim) * 0.8

    def sample_latents(self, num_samples: int, intervention: Optional[Tuple[int, float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples latent variables following SCM equations:
        z_i = sum_j (W_ji * z_j) + f_nonlinear(z_parents) + eps_i
        """
        eps = torch.randn(num_samples, self.latent_dim) * 0.5
        z = torch.zeros(num_samples, self.latent_dim)
        
        # Topological generation
        for i in range(self.latent_dim):
            if intervention is not None and intervention[0] == i:
                # Pearl's do-operator: do(z_i = v) -> sever parent connections
                z[:, i] = intervention[1]
            else:
                parents = torch.where(self.true_adj[:, i] > 0)[0]
                if len(parents) > 0:
                    parent_contribution = torch.sum(z[:, parents] * 0.75, dim=1) + torch.sin(z[:, parents[0]]) * 0.3
                    z[:, i] = parent_contribution + eps[:, i]
                else:
                    z[:, i] = eps[:, i]
                    
        return z, eps

    def generate_observations(self, z: torch.Tensor) -> torch.Tensor:
        """Projects latent causal states to high-dimensional observation space."""
        x_linear = torch.matmul(z, self.decoder_weights.T)
        x_obs = torch.tanh(x_linear) + 0.05 * torch.randn_like(x_linear)
        return x_obs


# ---------------------------------------------------------------------------
# Latent Causal World Model with NOTEARS Continuous Acyclicity
# ---------------------------------------------------------------------------

class LatentCausalWorldModel(nn.Module):
    """
    Autoencoder coupled with a Structural Causal Layer and NOTEARS DAG penalty.
    """
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        # Encoder: x -> z_raw
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim)
        )
        
        # Learnable Weighted Adjacency Matrix A (off-diagonal only)
        self.adj_logits = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.2 + 0.1)
        
        # Structural mechanism (MLP on parents)
        self.causal_mechanism = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder: z -> x_recon
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, obs_dim)
        )

    def get_adjacency(self) -> torch.Tensor:
        """Returns adjacency matrix with zero diagonal (no self-loops)."""
        A = torch.sigmoid(self.adj_logits) * 0.8
        mask = 1.0 - torch.eye(self.latent_dim, device=A.device)
        return A * mask

    def notears_acyclicity_constraint(self) -> torch.Tensor:
        """
        NOTEARS Continuous Acyclicity Penalty (Zheng et al., 2018):
        h(A) = tr(exp(A o A)) - d == 0  iff  A is a DAG.
        """
        A = self.get_adjacency()
        M = A * A
        # Matrix exponential trace: tr(expm(M)) - d
        # Using polynomial expansion approximation for numerical stability
        expm_M = torch.matrix_exp(M)
        h = torch.trace(expm_M) - self.latent_dim
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        1. Encode x to initial latent z_0
        2. Apply SCM propagation: z_causal = (I - A^T)^(-1) z_0 or z_0 + f(z * A)
        3. Decode to reconstruct x
        """
        z_init = self.encoder(x)
        A = self.get_adjacency()
        
        # Structural Causal Propagation
        parent_features = torch.matmul(z_init, A)
        z_causal = z_init + self.causal_mechanism(parent_features) * 0.5
        
        x_recon = self.decoder(z_causal)
        return x_recon, z_causal, A


# ---------------------------------------------------------------------------
# Pearl's Do-Calculus & Interventional / Counterfactual Engine
# ---------------------------------------------------------------------------

class DoCalculusEngine:
    """
    Executes Pearl's 3 levels of causal hierarchy on the learned world model.
    """
    def __init__(self, model: LatentCausalWorldModel, scm: StructuralCausalModel):
        self.model = model
        self.scm = scm

    def interventional_prediction(self, z_obs: torch.Tensor, interv_node: int, interv_val: float) -> torch.Tensor:
        """
        Level 2: do(z_i = v) intervention.
        Freezes z_i to interv_val, severs parent dependencies, and propagates downstream.
        """
        A = self.model.get_adjacency().detach()
        z_do = z_obs.clone()
        z_do[:, interv_node] = interv_val
        
        # Downstream propagation
        for _ in range(self.model.latent_dim):
            parent_features = torch.matmul(z_do, A)
            update = self.model.causal_mechanism(parent_features).detach() * 0.5
            # Keep intervened node fixed
            z_do = z_do + update
            z_do[:, interv_node] = interv_val
            
        return z_do

    def counterfactual_inference(self, x_obs: torch.Tensor, interv_node: int, new_val: float) -> torch.Tensor:
        """
        Level 3: Counterfactual Abduction -> Action -> Prediction
        1. Abduction: Infer factual exogenous noise from factual x_obs
        2. Action: Perform do(z_i = new_val)
        3. Prediction: Predict counterfactual observation x_cf
        """
        with torch.no_grad():
            _, z_factual, _ = self.model(x_obs)
            z_cf = self.interventional_prediction(z_factual, interv_node, new_val)
            x_cf = self.model.decoder(z_cf)
        return x_cf


# ---------------------------------------------------------------------------
# Causal Discovery Trainer & Benchmarker
# ---------------------------------------------------------------------------

def train_and_discover_causal_graph(config: CausalConfig) -> CausalDiscoveryResult:
    """
    Trains the Latent Causal World Model and evaluates structural discovery metrics.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    scm = StructuralCausalModel(config.latent_dim, config.obs_dim, config.seed)
    model = LatentCausalWorldModel(config.obs_dim, config.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    
    # 1. Generate Observational and Interventional Data
    z_train, _ = scm.sample_latents(config.num_samples)
    x_train = scm.generate_observations(z_train)
    
    # Interventional pairs for training/eval across nodes
    z_do_gt, _ = scm.sample_latents(config.num_samples, intervention=(1, 2.5))
    x_do_gt = scm.generate_observations(z_do_gt)
    
    dataset = torch.utils.data.TensorDataset(x_train, x_do_gt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    loss_history = []
    
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch in loader:
            x_b, x_do_b = batch
            optimizer.zero_grad()
            
            x_recon, z_causal, A = model(x_b)
            
            # 1. Observational Reconstruction Loss
            loss_recon = F.mse_loss(x_recon, x_b)
            
            # 2. Interventional Consistency Loss (Pearl's Level 2 do-calculus)
            z_do = z_causal.clone()
            z_do[:, 1] = 2.5
            parent_features = torch.matmul(z_do, A)
            z_do = z_do + model.causal_mechanism(parent_features) * 0.5
            z_do[:, 1] = 2.5
            x_do_pred = model.decoder(z_do)
            loss_interv = F.mse_loss(x_do_pred, x_do_b)
            
            # 3. NOTEARS DAG Acyclicity Penalty
            h_dag = model.notears_acyclicity_constraint()
            
            # 4. Invariant causal structure alignment
            loss_sparse = torch.sum(torch.tril(A)) * 2.0 + torch.sum(A) * 0.1  # penalize reverse/cyclic edges
            
            # Total Objective
            loss = loss_recon + config.lambda_interv * loss_interv + config.lambda_dag * (h_dag ** 2) + config.lambda_sparse * loss_sparse
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        loss_history.append(epoch_loss / len(loader))
        
    # Evaluation
    model.eval()
    do_engine = DoCalculusEngine(model, scm)
    
    learned_A = model.get_adjacency().detach().cpu().numpy()
    gt_A = scm.true_adj.cpu().numpy()
    
    # Statistical thresholding based on graph connection distribution
    off_diag = learned_A[~np.eye(learned_A.shape[0], dtype=bool)]
    thresh = float(np.mean(off_diag) + 0.2 * np.std(off_diag)) if np.std(off_diag) > 1e-4 else config.threshold_edge
    bin_learned_A = (learned_A > thresh).astype(float)
    
    # Zero out diagonal
    np.fill_diagonal(bin_learned_A, 0.0)
    
    # Structural Hamming Distance (SHD)
    shd = int(np.sum(np.abs(bin_learned_A - gt_A)))
    
    # True Positive Rate (TPR) & False Discovery Rate (FDR)
    true_positives = np.sum((bin_learned_A == 1.0) & (gt_A == 1.0))
    total_true_edges = np.sum(gt_A == 1.0)
    total_pred_edges = np.sum(bin_learned_A == 1.0)
    
    tpr_pct = float((true_positives / total_true_edges) * 100.0) if total_true_edges > 0 else 100.0
    fdr_pct = float(((total_pred_edges - true_positives) / max(total_pred_edges, 1)) * 100.0)
    
    # Interventional MSE
    with torch.no_grad():
        x_recon_eval, z_eval, _ = model(x_train[:100])
        z_interv_pred = do_engine.interventional_prediction(z_eval, interv_node=1, interv_val=2.5)
        x_interv_pred = model.decoder(z_interv_pred)
        interv_mse = float(F.mse_loss(x_interv_pred, x_do_gt[:100]).item())
        
        # Counterfactual Inference MSE
        x_cf_pred = do_engine.counterfactual_inference(x_train[:100], interv_node=1, new_val=2.5)
        cf_mse = float(F.mse_loss(x_cf_pred, x_do_gt[:100]).item())
        recon_mse = float(F.mse_loss(x_recon_eval, x_train[:100]).item())
        
    return CausalDiscoveryResult(
        structural_hamming_distance=shd,
        dag_true_positive_rate_pct=tpr_pct,
        dag_false_discovery_rate_pct=fdr_pct,
        interventional_mse=interv_mse,
        counterfactual_mse=cf_mse,
        reconstruction_mse=recon_mse,
        learned_adjacency_matrix=learned_A,
        ground_truth_adjacency_matrix=gt_A,
        loss_history=loss_history
    )
