"""
Day 311: Automated Scientific Theory & Paradigm Discovery Engine.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import scipy.integrate as integrate
from scipy.signal import savgol_filter


@dataclass
class ScientificDiscoveryConfig:
    poly_order: int = 3                # Maximum polynomial degree for candidate library
    include_trig: bool = True          # Include sin, cos terms
    sparsity_threshold: float = 0.08   # STLSQ cutoff threshold lambda
    ridge_alpha: float = 1e-4          # Ridge regularization factor
    noise_level: float = 0.01          # Measurement noise sigma
    time_steps: int = 500              # Observation timepoints
    dt: float = 0.02                   # Sampling interval
    seed: int = 42


@dataclass
class ScientificDiscoveryResult:
    equation_recovery_precision_pct: float # Percentage of exact active symbolic terms correctly identified (%)
    avg_parameter_relative_error_pct: float # Relative error on discovered physical constants (%)
    ood_extrapolation_r2: float            # R^2 coefficient of determination on unseen initial conditions
    parsimony_bic_score: float             # Bayesian Information Criterion
    discovered_equations: Dict[str, str]   # String representations of discovered differential equations
    ground_truth_equations: Dict[str, str] # True physical laws
    true_trajectories: np.ndarray          # Shape: [T, D]
    simulated_discovered_trajectories: np.ndarray # Shape: [T, D]
    time_axis: np.ndarray


# ---------------------------------------------------------------------------
# Candidate Symbolic Feature Library
# ---------------------------------------------------------------------------

class CandidateLibrary:
    """
    Constructs non-linear candidate function library Theta(X) for SINDy.
    """
    def __init__(self, poly_order: int = 2, include_trig: bool = True):
        self.poly_order = poly_order
        self.include_trig = include_trig
        self.feature_names = []

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Transforms state matrix X [N, D] into candidate matrix Theta(X) [N, P].
        """
        N, D = X.shape
        features = [np.ones((N, 1))]
        names = ["1"]
        
        # 1st order linear
        for i in range(D):
            features.append(X[:, i:i+1])
            names.append(f"x{i+1}")
            
        # 2nd order polynomial
        if self.poly_order >= 2:
            for i in range(D):
                for j in range(i, D):
                    features.append((X[:, i] * X[:, j]).reshape(-1, 1))
                    names.append(f"x{i+1}x{j+1}" if i != j else f"x{i+1}^2")
                    
        # 3rd order polynomial
        if self.poly_order >= 3:
            for i in range(D):
                features.append((X[:, i]**3).reshape(-1, 1))
                names.append(f"x{i+1}^3")
                
        # Trigonometric
        if self.include_trig:
            for i in range(D):
                features.append(np.sin(X[:, i:i+1]))
                names.append(f"sin(x{i+1})")
                features.append(np.cos(X[:, i:i+1]))
                names.append(f"cos(x{i+1})")
                
        Theta = np.hstack(features)
        self.feature_names = names
        return Theta, names


# ---------------------------------------------------------------------------
# SINDy Equation Discoverer (Sequentially Thresholded Least Squares)
# ---------------------------------------------------------------------------

class SINDyEquationDiscoverer:
    """
    Solves for sparse coefficient matrix Xi via STLSQ.
    """
    def __init__(self, threshold: float = 0.08, alpha: float = 1e-4, max_iter: int = 15):
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.Xi = None

    def fit(self, Theta: np.ndarray, dX: np.ndarray) -> np.ndarray:
        """
        dX: [N, D], Theta: [N, P] -> Xi: [P, D]
        """
        N, P = Theta.shape
        _, D = dX.shape
        
        # Initial Ridge Regression
        Xi = np.linalg.solve(Theta.T @ Theta + self.alpha * np.eye(P), Theta.T @ dX)
        
        # Sequentially threshold small coefficients to zero
        for _ in range(self.max_iter):
            small_indices = np.abs(Xi) < self.threshold
            Xi[small_indices] = 0.0
            
            for d in range(D):
                big_indices = ~small_indices[:, d]
                if np.sum(big_indices) > 0:
                    Xi[big_indices, d] = np.linalg.solve(
                        Theta[:, big_indices].T @ Theta[:, big_indices] + self.alpha * np.eye(np.sum(big_indices)),
                        Theta[:, big_indices].T @ dX[:, d]
                    )
        self.Xi = Xi
        return Xi

    def format_equations(self, feature_names: List[str]) -> Dict[str, str]:
        """
        Converts non-zero sparse coefficients into human-readable differential equations.
        """
        eqs = {}
        P, D = self.Xi.shape
        for d in range(D):
            terms = []
            for p in range(P):
                coef = self.Xi[p, d]
                if abs(coef) > 1e-3:
                    fname = feature_names[p]
                    terms.append(f"{coef:+.3f}*{fname}" if fname != "1" else f"{coef:+.3f}")
            eq_str = " + ".join(terms) if terms else "0.0"
            eq_str = eq_str.replace("+ -", "- ")
            eqs[f"dx{d+1}/dt"] = eq_str
        return eqs


# ---------------------------------------------------------------------------
# Automated Scientific Discovery Engine
# ---------------------------------------------------------------------------

class AutomatedScientificDiscoveryEngine:
    """
    Coordinates synthetic dynamic data generation, SINDy discovery, and physical law validation.
    """
    def __init__(self, config: ScientificDiscoveryConfig):
        self.config = config
        np.random.seed(config.seed)
        self.library = CandidateLibrary(poly_order=config.poly_order, include_trig=config.include_trig)
        self.discoverer = SINDyEquationDiscoverer(threshold=config.sparsity_threshold, alpha=config.ridge_alpha)

    def _lorenz_ode(self, state: np.ndarray, t: float, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0) -> np.ndarray:
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])

    def generate_data(self, system: str = "lorenz") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrates ground truth dynamical equations and adds observation noise.
        """
        t = np.linspace(0, self.config.time_steps * self.config.dt, self.config.time_steps)
        x0 = np.array([-8.0, 7.0, 27.0])
        
        sol = integrate.odeint(self._lorenz_ode, x0, t)
        
        # Add sensor measurement noise
        noisy_sol = sol + np.random.normal(0, self.config.noise_level, sol.shape)
        
        # Compute smooth numerical time derivatives using Savitzky-Golay filter
        dX = np.zeros_like(noisy_sol)
        for d in range(sol.shape[1]):
            dX[:, d] = savgol_filter(noisy_sol[:, d], window_length=9, polyorder=3, deriv=1, delta=self.config.dt)
            
        return t, noisy_sol, dX

    def discover_laws(self) -> ScientificDiscoveryResult:
        """
        Runs autonomous equation discovery on dynamical system data.
        """
        t, X, dX = self.generate_data("lorenz")
        
        # 1. Feature library generation
        Theta, feature_names = self.library.fit_transform(X)
        
        # 2. Sparse Equation Discovery
        Xi = self.discoverer.fit(Theta, dX)
        discovered_eqs = self.discoverer.format_equations(feature_names)
        
        ground_truth_eqs = {
            "dx1/dt": "-10.000*x1 + 10.000*x2",
            "dx2/dt": "+28.000*x1 - 1.000*x2 - 1.000*x1x3",
            "dx3/dt": "-2.667*x3 + 1.000*x1x2"
        }
        
        # 3. Ground truth parameter verification
        # True terms in Lorenz:
        # dx1/dt: x1 (-10), x2 (+10)
        # dx2/dt: x1 (+28), x2 (-1), x1x3 (-1)
        # dx3/dt: x3 (-2.667), x1x2 (+1)
        
        # Check active precision
        x1_idx = feature_names.index("x1")
        x2_idx = feature_names.index("x2")
        x3_idx = feature_names.index("x3")
        x1x2_idx = feature_names.index("x1x2")
        x1x3_idx = feature_names.index("x1x3")
        
        true_active = {
            0: {x1_idx: -10.0, x2_idx: 10.0},
            1: {x1_idx: 28.0, x2_idx: -1.0, x1x3_idx: -1.0},
            2: {x3_idx: -8.0/3.0, x1x2_idx: 1.0}
        }
        
        correct_terms = 0
        total_true_terms = 7
        errors = []
        
        for d, terms in true_active.items():
            for p_idx, true_val in terms.items():
                pred_val = Xi[p_idx, d]
                if abs(pred_val) > 0.05:
                    correct_terms += 1
                    rel_err = abs(pred_val - true_val) / abs(true_val) * 100.0
                    errors.append(rel_err)
                    
        recovery_precision = float(correct_terms / total_true_terms * 100.0)
        avg_rel_err = float(np.mean(errors)) if errors else 0.0
        
        # 4. Forward simulate discovered system
        def discovered_ode(state, t_val):
            s_vec = state.reshape(1, -1)
            th, _ = self.library.fit_transform(s_vec)
            return (th @ Xi).flatten()
            
        sim_discovered = integrate.odeint(discovered_ode, X[0], t)
        
        # 5. OOD extrapolation test on unseen initial condition
        x0_ood = np.array([5.0, 5.0, 20.0])
        sol_true_ood = integrate.odeint(self._lorenz_ode, x0_ood, t[:150])
        sol_disc_ood = integrate.odeint(discovered_ode, x0_ood, t[:150])
        
        ss_res = np.sum((sol_true_ood - sol_disc_ood)**2)
        ss_tot = np.sum((sol_true_ood - np.mean(sol_true_ood, axis=0))**2)
        ood_r2 = float(np.clip(1.0 - (ss_res / (ss_tot + 1e-6)), -1.0, 0.999))
        
        # 6. Bayesian Information Criterion (BIC)
        k_nonzero = np.sum(Xi != 0)
        n_samples = X.shape[0] * X.shape[1]
        residual_variance = np.mean((dX - Theta @ Xi)**2)
        bic = float(k_nonzero * np.log(n_samples) + n_samples * np.log(max(residual_variance, 1e-6)))
        
        return ScientificDiscoveryResult(
            equation_recovery_precision_pct=recovery_precision,
            avg_parameter_relative_error_pct=avg_rel_err,
            ood_extrapolation_r2=ood_r2,
            parsimony_bic_score=bic,
            discovered_equations=discovered_eqs,
            ground_truth_equations=ground_truth_eqs,
            true_trajectories=X,
            simulated_discovered_trajectories=sim_discovered,
            time_axis=t
        )
