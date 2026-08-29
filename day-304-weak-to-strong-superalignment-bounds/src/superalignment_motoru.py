"""
Day 304: Weak-to-Strong Superalignment with Confidence Bounds
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weak Supervisor & Strong Student Models
# ---------------------------------------------------------------------------

class WeakSupervisor(nn.Module):
    """
    Simulates a weak human/small model proxy with limited capacity and noise.
    """
    def __init__(self, in_features: int = 16, num_classes: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrongModel(nn.Module):
    """
    High-capacity foundation/ASI model capable of deep latent reasoning.
    """
    def __init__(self, in_features: int = 16, num_classes: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Temperature Calibrator & Conformal Predictor
# ---------------------------------------------------------------------------

class ConformalCalibrator(nn.Module):
    """
    Learns Platt temperature scaling and computes distribution-free
    conformal prediction confidence sets with 1-alpha statistical guarantees.
    """
    def __init__(self, alpha: float = 0.10):
        super().__init__()
        self.alpha = alpha  # Target error rate (e.g. 10% for 90% coverage)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.q_hat: float = 0.95

    def calibrate_temperature(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, epochs: int = 100):
        """Optimizes temperature to minimize negative log-likelihood on calibration set."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=epochs)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)

    def fit_conformal_threshold(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Computes conformal quantile threshold q_hat from non-conformity scores.
        """
        with torch.no_grad():
            scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Non-conformity score: s_i = 1 - P(true_label | x_i)
            batch_size = labels.size(0)
            true_probs = probs[torch.arange(batch_size), labels]
            scores = (1.0 - true_probs).cpu().numpy()
            
            # Conformal quantile calculation with finite-sample correction
            n = len(scores)
            q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
            q_level = float(np.clip(q_level, 0.0, 1.0))
            self.q_hat = float(np.quantile(scores, q_level))

    def predict_conformal_sets(self, logits: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]], List[float]]:
        """
        Returns calibrated probabilities, prediction sets, and set sizes.
        """
        with torch.no_grad():
            scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Prediction set: {y : 1 - P(y | x) <= q_hat}
            scores_matrix = 1.0 - probs.cpu().numpy()
            prediction_sets = []
            set_sizes = []
            
            for row in scores_matrix:
                valid_classes = [c for c, s in enumerate(row) if s <= self.q_hat]
                if not valid_classes:
                    valid_classes = [int(np.argmin(row))]
                prediction_sets.append(valid_classes)
                set_sizes.append(len(valid_classes))
                
            return probs, prediction_sets, set_sizes


# ---------------------------------------------------------------------------
# Dataclasses & Config
# ---------------------------------------------------------------------------

@dataclass
class SuperalignmentConfig:
    """Hyperparameters for Weak-to-Strong training and calibration."""
    in_features: int = 16
    num_classes: int = 4
    weak_epochs: int = 15
    strong_epochs: int = 30
    lr_weak: float = 0.01
    lr_strong: float = 0.003
    confidence_gate_tau: float = 0.40  # Filter out weak labels below confidence threshold
    lambda_consistency: float = 0.50   # Perturbation consistency regularization
    conformal_alpha: float = 0.10      # 90% confidence coverage guarantee
    seed: int = 42


@dataclass
class SuperalignmentResult:
    """Encapsulates the benchmarking and calibration outputs."""
    weak_acc: float
    strong_ceiling_acc: float
    weak_to_strong_acc: float
    pgr_score: float                   # Performance Gap Recovered (%)
    temperature: float
    ece_before: float
    ece_after: float
    conformal_coverage_pct: float
    avg_conformal_set_size: float
    history: Dict[str, List[float]]
    confidence_ablation: Dict[float, float]


# ---------------------------------------------------------------------------
# Weak-to-Strong Superalignment Trainer
# ---------------------------------------------------------------------------

class WeakToStrongTrainer:
    """
    Orchestrates the complete Weak-to-Strong superalignment pipeline:
    1. Trains Weak Supervisor on ground truth
    2. Generates weak pseudo-labels for unlabeled pool
    3. Trains Strong Model with Confidence-Gated loss & consistency regularization
    4. Trains Strong Ceiling (trained directly on ground truth for comparison)
    5. Calibrates temperature and computes Conformal Confidence Bounds
    """
    def __init__(self, config: SuperalignmentConfig):
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        self.weak_model = WeakSupervisor(config.in_features, config.num_classes)
        self.strong_model = StrongModel(config.in_features, config.num_classes)
        self.strong_ceiling_model = StrongModel(config.in_features, config.num_classes)
        self.calibrator = ConformalCalibrator(alpha=config.conformal_alpha)
        
        self.history: Dict[str, List[float]] = {
            "weak_loss": [],
            "strong_w2s_loss": [],
            "strong_w2s_acc": [],
            "strong_ceiling_acc": []
        }

    def train_weak_supervisor(self, train_loader: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Trains weak supervisor model on limited ground truth data."""
        optimizer = torch.optim.Adam(self.weak_model.parameters(), lr=self.config.lr_weak)
        
        for epoch in range(self.config.weak_epochs):
            total_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                logits = self.weak_model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.history["weak_loss"].append(total_loss / len(train_loader))

    def evaluate_model(self, model: nn.Module, loader: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Evaluates model accuracy and returns all logits and targets."""
        model.eval()
        correct = 0
        total = 0
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in loader:
                logits = model(x)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_logits.append(logits)
                all_targets.append(y)
                
        acc = (correct / max(1, total)) * 100.0
        return acc, torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)

    def train_strong_with_weak_supervision(self, unlabeled_loader: List[Tuple[torch.Tensor, torch.Tensor]], 
                                          val_loader: List[Tuple[torch.Tensor, torch.Tensor]],
                                          tau_gate: Optional[float] = None) -> float:
        """
        Trains Strong Model using weak supervisor predictions with confidence gating & consistency.
        """
        tau = tau_gate if tau_gate is not None else self.config.confidence_gate_tau
        optimizer = torch.optim.AdamW(self.strong_model.parameters(), lr=self.config.lr_strong, weight_decay=1e-4)
        
        self.weak_model.eval()
        self.strong_model.train()
        
        for epoch in range(self.config.strong_epochs):
            epoch_loss = 0.0
            for x, _ in unlabeled_loader:
                optimizer.zero_grad()
                
                # 1. Get weak supervisor soft labels & confidence
                with torch.no_grad():
                    weak_logits = self.weak_model(x)
                    weak_probs = F.softmax(weak_logits / 1.5, dim=-1)  # Soft distribution
                    max_conf, _ = torch.max(weak_probs, dim=-1)
                    mask = (max_conf >= tau).float().unsqueeze(-1)  # [B, 1]
                    
                # 2. Strong model forward
                strong_logits = self.strong_model(x)
                log_probs_strong = F.log_softmax(strong_logits, dim=-1)
                
                # 3. Soft distillation loss with confidence weighting
                sample_loss = -(weak_probs * log_probs_strong).sum(dim=-1, keepdim=True)
                masked_loss = (sample_loss * mask).sum() / max(1.0, mask.sum())
                
                # 4. Consistency regularization
                x_perturbed = x + 0.05 * torch.randn_like(x)
                strong_logits_pert = self.strong_model(x_perturbed)
                p_clean = F.softmax(strong_logits, dim=-1)
                log_p_pert = F.log_softmax(strong_logits_pert, dim=-1)
                kl_cons = F.kl_div(log_p_pert, p_clean, reduction="batchmean")
                
                total_loss = masked_loss + self.config.lambda_consistency * kl_cons
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                
            val_acc, _, _ = self.evaluate_model(self.strong_model, val_loader)
            self.history["strong_w2s_loss"].append(epoch_loss / len(unlabeled_loader))
            self.history["strong_w2s_acc"].append(val_acc)
            
        final_acc, _, _ = self.evaluate_model(self.strong_model, val_loader)
        return final_acc

    def train_strong_ceiling(self, train_loader: List[Tuple[torch.Tensor, torch.Tensor]], 
                             val_loader: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Trains strong model directly on true labels to establish theoretical ceiling."""
        optimizer = torch.optim.AdamW(self.strong_ceiling_model.parameters(), lr=self.config.lr_strong, weight_decay=1e-4)
        self.strong_ceiling_model.train()
        
        for epoch in range(self.config.strong_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                logits = self.strong_ceiling_model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                
        final_acc, _, _ = self.evaluate_model(self.strong_ceiling_model, val_loader)
        return final_acc

    def run_superalignment(self, train_loader: List[Tuple[torch.Tensor, torch.Tensor]],
                           unlabeled_loader: List[Tuple[torch.Tensor, torch.Tensor]],
                           calib_loader: List[Tuple[torch.Tensor, torch.Tensor]],
                           test_loader: List[Tuple[torch.Tensor, torch.Tensor]],
                           train_clean_loader: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> SuperalignmentResult:
        """
        Executes full weak-to-strong experimental flow and statistical validation.
        """
        # Step 1: Train Weak Model (trained on noisy/weak supervision)
        self.train_weak_supervisor(train_loader)
        weak_acc, _, _ = self.evaluate_model(self.weak_model, test_loader)
        
        # Step 2: Train Strong Ceiling (Upper Bound on clean ground truth)
        ceiling_loader = train_clean_loader if train_clean_loader is not None else train_loader
        strong_ceiling_acc = self.train_strong_ceiling(ceiling_loader, test_loader)
        
        # Step 3: Train Strong Model with Weak Supervision
        w2s_acc = self.train_strong_with_weak_supervision(unlabeled_loader, test_loader)
        
        # Step 4: Compute Performance Gap Recovered (PGR)
        # PGR = (Acc(Strong|Weak) - Acc(Weak)) / (Acc(Strong*) - Acc(Weak))
        denom = max(1e-4, strong_ceiling_acc - weak_acc)
        pgr = float(np.clip((w2s_acc - weak_acc) / denom, 0.0, 1.0)) * 100.0
        
        # Step 5: Conformal Calibration on Calibration Split
        _, calib_logits, calib_y = self.evaluate_model(self.strong_model, calib_loader)
        _, test_logits, test_y = self.evaluate_model(self.strong_model, test_loader)
        
        # Measure ECE before calibration
        ece_before = self._compute_ece(test_logits, test_y)
        
        # Calibrate
        self.calibrator.calibrate_temperature(calib_logits, calib_y)
        self.calibrator.fit_conformal_threshold(calib_logits, calib_y)
        
        # Measure ECE after calibration
        scaled_test_logits = test_logits / self.calibrator.temperature
        ece_after = self._compute_ece(scaled_test_logits, test_y)
        
        # Predict Conformal Sets & Measure Coverage on Test Split
        _, test_sets, set_sizes = self.calibrator.predict_conformal_sets(test_logits)
        coverage_count = sum(1 for i, s in enumerate(test_sets) if test_y[i].item() in s)
        conformal_coverage = (coverage_count / len(test_y)) * 100.0
        avg_set_size = float(np.mean(set_sizes))
        
        # Step 6: Confidence Threshold Gating Ablation
        ablation = {}
        for tau in [0.2, 0.4, 0.6, 0.8]:
            # Temporary fresh model
            temp_strong = StrongModel(self.config.in_features, self.config.num_classes)
            temp_trainer = WeakToStrongTrainer(self.config)
            temp_trainer.weak_model = self.weak_model
            temp_acc = temp_trainer.train_strong_with_weak_supervision(unlabeled_loader, test_loader, tau_gate=tau)
            ablation[tau] = temp_acc
            
        return SuperalignmentResult(
            weak_acc=weak_acc,
            strong_ceiling_acc=strong_ceiling_acc,
            weak_to_strong_acc=w2s_acc,
            pgr_score=pgr,
            temperature=float(self.calibrator.temperature.item()),
            ece_before=ece_before,
            ece_after=ece_after,
            conformal_coverage_pct=conformal_coverage,
            avg_conformal_set_size=avg_set_size,
            history=self.history,
            confidence_ablation=ablation
        )

    def _compute_ece(self, logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error across bins."""
        probs = F.softmax(logits, dim=-1)
        confidences, preds = torch.max(probs, dim=-1)
        accuracies = (preds == targets).float()
        
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()
            
            if prop_in_bin > 0:
                acc_in_bin = accuracies[in_bin].mean().item()
                conf_in_bin = confidences[in_bin].mean().item()
                ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin
                
        return float(ece * 100.0)
