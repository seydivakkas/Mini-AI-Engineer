"""
Day 316: Adversarial Byzantine Fault Tolerance Engine.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ByzantineSwarmConfig:
    num_nodes: int = 15               # Total swarm nodes (M)
    num_byzantine: int = 4            # Malicious Byzantine nodes (f < M/3)
    param_dim: int = 50               # Gradient parameter dimension
    attack_type: str = "sign_flipping" # 'sign_flipping', 'gaussian_drift', 'targeted_poison'
    iterations: int = 40
    learning_rate: float = 0.05
    seed: int = 42


@dataclass
class ByzantineBenchmarkResult:
    mean_cosine_fidelity: Dict[str, float]       # Cosine alignment with true gradient
    final_objective_loss: Dict[str, float]       # Final task loss per aggregator
    byzantine_detection_precision_pct: float     # Precision of identifying malicious nodes
    byzantine_detection_recall_pct: float        # Recall of identifying malicious nodes
    attack_mitigation_ratio_pct: float           # Relative improvement over naive mean (%)
    loss_trajectories: Dict[str, np.ndarray]
    cosine_trajectories: Dict[str, np.ndarray]
    attacker_indices: List[int]


# ---------------------------------------------------------------------------
# Byzantine Robust Aggregators
# ---------------------------------------------------------------------------

class ByzantineAggregatorBank:
    """
    Collection of classical and state-of-the-art Byzantine robust gradient aggregation rules.
    """
    
    @staticmethod
    def naive_mean(gradients: np.ndarray) -> np.ndarray:
        """Standard arithmetic mean (Non-robust)."""
        return np.mean(gradients, axis=0)

    @staticmethod
    def coordinate_median(gradients: np.ndarray) -> np.ndarray:
        """Coordinate-wise median."""
        return np.median(gradients, axis=0)

    @staticmethod
    def trimmed_mean(gradients: np.ndarray, num_byzantine: int) -> np.ndarray:
        """Coordinate-wise Trimmed Mean: Drops largest and smallest f values per dimension."""
        M, D = gradients.shape
        f = num_byzantine
        if 2 * f >= M:
            return np.median(gradients, axis=0)
            
        sorted_grads = np.sort(gradients, axis=0)
        trimmed = sorted_grads[f : M - f, :]
        return np.mean(trimmed, axis=0)

    @staticmethod
    def multi_krum(gradients: np.ndarray, num_byzantine: int, m: int = 3) -> Tuple[np.ndarray, List[int]]:
        """
        Multi-Krum Aggregator: Scores each node by Euclidean distance to (M - f - 2) nearest neighbors.
        Selects top m nodes with minimal score and averages them.
        """
        M = len(gradients)
        f = num_byzantine
        nb_neighbors = max(1, M - f - 2)
        
        # Compute pairwise distance matrix
        dist_matrix = np.zeros((M, M))
        for i in range(M):
            for j in range(i + 1, M):
                d = np.sum((gradients[i] - gradients[j]) ** 2)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                
        scores = np.zeros(M)
        for i in range(M):
            dists = np.sort(dist_matrix[i])
            scores[i] = np.sum(dists[1 : nb_neighbors + 1]) # Exclude self
            
        selected_indices = list(np.argsort(scores)[:m])
        aggregated_grad = np.mean(gradients[selected_indices], axis=0)
        return aggregated_grad, selected_indices

    @staticmethod
    def bulyan(gradients: np.ndarray, num_byzantine: int) -> np.ndarray:
        """
        Bulyan Aggregator: Combines Multi-Krum selection pool (size theta = M - 2f)
        with coordinate-wise trimmed mean.
        """
        M = len(gradients)
        f = num_byzantine
        theta = max(1, M - 2 * f)
        
        _, krum_candidates = ByzantineAggregatorBank.multi_krum(gradients, f, m=theta)
        candidate_grads = gradients[krum_candidates]
        
        # Apply trimmed mean on candidates with beta = f
        beta = min(f, (len(candidate_grads) - 1) // 2)
        return ByzantineAggregatorBank.trimmed_mean(candidate_grads, num_byzantine=beta)


# ---------------------------------------------------------------------------
# Swarm Byzantine Defense Engine
# ---------------------------------------------------------------------------

class ByzantineDefenseEngine:
    """
    Simulates adversarial distributed gradient optimization with poisoned nodes.
    """
    def __init__(self, config: ByzantineSwarmConfig):
        self.config = config
        np.random.seed(config.seed)
        
        M = config.num_nodes
        f = config.num_byzantine
        assert f < M / 2, "Byzantine tolerance requires f < M / 2"
        
        # Designate malicious nodes
        self.attacker_indices = list(np.random.choice(M, f, replace=False))
        self.honest_indices = [i for i in range(M) if i not in self.attacker_indices]

    def _inject_attack(self, honest_grad: np.ndarray) -> np.ndarray:
        """
        Generates adversarial poisoned gradients based on chosen attack type.
        """
        D = len(honest_grad)
        attack = self.config.attack_type
        
        if attack == "sign_flipping":
            # Inverts gradient direction with high amplification to dominate arithmetic mean
            return -4.0 * honest_grad
        elif attack == "gaussian_drift":
            # Injects high-variance Gaussian noise
            return honest_grad + np.random.normal(0.0, 5.0, size=D)
        elif attack == "targeted_poison":
            # Backdoor vector push
            target = np.ones(D) * 10.0
            return target
        else:
            return -honest_grad

    def run_defense_benchmark(self) -> ByzantineBenchmarkResult:
        """
        Runs comparative optimization for Mean, Median, Trimmed Mean, Multi-Krum, and Bulyan.
        """
        aggregators = ["Naive-Mean", "Coord-Median", "Trimmed-Mean", "Multi-Krum", "Bulyan"]
        M = self.config.num_nodes
        f = self.config.num_byzantine
        D = self.config.param_dim
        T = self.config.iterations
        lr = self.config.learning_rate
        
        loss_trajectories = {agg: np.zeros(T) for agg in aggregators}
        cosine_trajectories = {agg: np.zeros(T) for agg in aggregators}
        
        for agg_name in aggregators:
            np.random.seed(self.config.seed)
            # Initialize parameters theta
            theta = np.random.randn(D) * 2.0
            target_theta = np.zeros(D) # Target minimum
            
            for t in range(T):
                # 1. Compute true gradient for quadratic bowl f(theta) = 0.5 * ||theta||^2
                true_grad = theta - target_theta
                loss = 0.5 * np.sum((theta - target_theta) ** 2)
                loss_trajectories[agg_name][t] = loss
                
                # 2. Honest nodes compute noisy stochastic gradients
                grad_pool = np.zeros((M, D))
                for i in self.honest_indices:
                    grad_pool[i] = true_grad + np.random.normal(0.0, 0.2, size=D)
                    
                # 3. Byzantine nodes inject adversarial poison
                for i in self.attacker_indices:
                    grad_pool[i] = self._inject_attack(true_grad)
                    
                # 4. Aggregation
                if agg_name == "Naive-Mean":
                    g_agg = ByzantineAggregatorBank.naive_mean(grad_pool)
                elif agg_name == "Coord-Median":
                    g_agg = ByzantineAggregatorBank.coordinate_median(grad_pool)
                elif agg_name == "Trimmed-Mean":
                    g_agg = ByzantineAggregatorBank.trimmed_mean(grad_pool, num_byzantine=f)
                elif agg_name == "Multi-Krum":
                    g_agg, _ = ByzantineAggregatorBank.multi_krum(grad_pool, num_byzantine=f, m=3)
                elif agg_name == "Bulyan":
                    g_agg = ByzantineAggregatorBank.bulyan(grad_pool, num_byzantine=f)
                    
                # 5. Cosine alignment with true gradient
                cos_sim = np.dot(g_agg, true_grad) / (np.linalg.norm(g_agg) * np.linalg.norm(true_grad) + 1e-8)
                cosine_trajectories[agg_name][t] = max(-1.0, min(1.0, float(cos_sim)))
                
                # 6. Parameter update
                theta = theta - lr * g_agg
                
        # Calculate summary metrics
        final_losses = {agg: float(loss_trajectories[agg][-1]) for agg in aggregators}
        mean_cosines = {agg: float(np.mean(cosine_trajectories[agg])) for agg in aggregators}
        
        # Byzantine identification accuracy using Krum distance scoring
        _, krum_honest = ByzantineAggregatorBank.multi_krum(grad_pool, num_byzantine=f, m=M - f)
        detected_attackers = [i for i in range(M) if i not in krum_honest]
        
        true_pos = len(set(detected_attackers).intersection(set(self.attacker_indices)))
        precision = float(true_pos / max(len(detected_attackers), 1) * 100.0)
        recall = float(true_pos / len(self.attacker_indices) * 100.0)
        
        # Attack mitigation vs Naive Mean
        naive_loss = final_losses["Naive-Mean"]
        best_robust_loss = min(final_losses["Bulyan"], final_losses["Multi-Krum"], final_losses["Trimmed-Mean"])
        mitigation_ratio = float((naive_loss - best_robust_loss) / max(naive_loss, 1e-4) * 100.0)
        
        return ByzantineBenchmarkResult(
            mean_cosine_fidelity=mean_cosines,
            final_objective_loss=final_losses,
            byzantine_detection_precision_pct=precision,
            byzantine_detection_recall_pct=recall,
            attack_mitigation_ratio_pct=mitigation_ratio,
            loss_trajectories=loss_trajectories,
            cosine_trajectories=cosine_trajectories,
            attacker_indices=self.attacker_indices
        )
