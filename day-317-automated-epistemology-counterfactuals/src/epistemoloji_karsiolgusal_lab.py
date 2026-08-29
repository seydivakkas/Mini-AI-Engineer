"""
Day 317: Automated Epistemology & Counterfactual Hypothesis Testing Engine.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class EpistemologyConfig:
    sample_size: int = 1000
    treatment_val_0: float = 0.0
    treatment_val_1: float = 1.0
    seed: int = 42


@dataclass
class EpistemologyBenchmarkResult:
    observational_association: float       # E[Y | X=1] - E[Y | X=0] (Confounded)
    average_treatment_effect_ate: float    # E[Y | do(X=1)] - E[Y | do(X=0)]
    natural_direct_effect_nde: float       # Direct causal path X -> Y
    natural_indirect_effect_nie: float     # Mediated causal path X -> M -> Y
    confounding_bias_gap: float            # Association - ATE gap
    counterfactual_consistency_pct: float  # Validation of consistency axiom Y_{X=x} == Y
    factual_vs_counterfactual_samples: List[Dict[str, float]]
    treatment_response_curve: Tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Structural Causal Model (SCM)
# ---------------------------------------------------------------------------

class StructuralCausalModel:
    """
    4-Variable Structural Causal Model:
    Z (Confounder) -> X (Treatment) -> M (Mediator) -> Y (Outcome)
    Z -> Y (Direct Confounding), X -> Y (Direct Effect)
    """
    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def sample_factual_data(self, N: int = 1000) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Samples exogenous noise variables U and computes endogenous factual observations V.
        """
        u_Z = np.random.normal(0.0, 1.0, size=N)
        u_X = np.random.normal(0.0, 0.5, size=N)
        u_M = np.random.normal(0.0, 0.4, size=N)
        u_Y = np.random.normal(0.0, 0.3, size=N)
        
        # Structural Equations
        Z = u_Z
        X = 0.8 * Z + u_X
        M = 1.2 * X + u_M
        Y = 0.5 * Z + 1.5 * M + 0.4 * X + u_Y
        
        V = {"Z": Z, "X": X, "M": M, "Y": Y}
        U = {"u_Z": u_Z, "u_X": u_X, "u_M": u_M, "u_Y": u_Y}
        return V, U


# ---------------------------------------------------------------------------
# Counterfactual Reasoning Engine
# ---------------------------------------------------------------------------

class CounterfactualEngine:
    """
    Executes Pearl's 3-Step Abduction-Action-Prediction Counterfactual Algorithm.
    """
    def __init__(self, config: EpistemologyConfig):
        self.config = config
        self.scm = StructuralCausalModel(seed=config.seed)

    def run_epistemic_inquiry(self) -> EpistemologyBenchmarkResult:
        """
        Runs comprehensive observational, interventional (do-calculus), and counterfactual inquiry.
        """
        N = self.config.sample_size
        V, U = self.scm.sample_factual_data(N)
        
        Z, X, M, Y = V["Z"], V["X"], V["M"], V["Y"]
        
        # 1. Level 1: Observational Association E[Y | X > median] - E[Y | X <= median]
        mask_x1 = X > np.median(X)
        mask_x0 = ~mask_x1
        assoc = float(np.mean(Y[mask_x1]) - np.mean(Y[mask_x0]))
        
        # 2. Level 2: Interventional Do-Calculus E[Y | do(X=1)] - E[Y | do(X=0)]
        # Total Effect = dY/dX = (1.5 * 1.2 + 0.4) = 1.8 + 0.4 = 2.2
        # In structural model: do(X=x) sets X=x and evaluates Y(x)
        y_do_1 = 0.5 * Z + 1.5 * (1.2 * 1.0 + U["u_M"]) + 0.4 * 1.0 + U["u_Y"]
        y_do_0 = 0.5 * Z + 1.5 * (1.2 * 0.0 + U["u_M"]) + 0.4 * 0.0 + U["u_Y"]
        ate = float(np.mean(y_do_1 - y_do_0))
        
        # Direct effect (NDE): holding mediator M at M(0)
        m_x0 = 1.2 * 0.0 + U["u_M"]
        y_nde_1 = 0.5 * Z + 1.5 * m_x0 + 0.4 * 1.0 + U["u_Y"]
        y_nde_0 = 0.5 * Z + 1.5 * m_x0 + 0.4 * 0.0 + U["u_Y"]
        nde = float(np.mean(y_nde_1 - y_nde_0)) # Theoretical: 0.4
        
        # Indirect effect (NIE): holding X=1, shifting M from M(0) to M(1)
        m_x1 = 1.2 * 1.0 + U["u_M"]
        y_nie_1 = 0.5 * Z + 1.5 * m_x1 + 0.4 * 1.0 + U["u_Y"]
        y_nie_0 = 0.5 * Z + 1.5 * m_x0 + 0.4 * 1.0 + U["u_Y"]
        nie = float(np.mean(y_nie_1 - y_nie_0)) # Theoretical: 1.5 * 1.2 = 1.8
        
        confounding_gap = abs(assoc - ate)
        
        # 3. Level 3: Individual Counterfactual Abduction-Action-Prediction
        # Step 1 (Abduction): infer noise u for individual i
        # Step 2 (Action): intervene do(X = x')
        # Step 3 (Prediction): compute counterfactual Y_{X=x'}
        consistent_checks = 0
        samples = []
        
        for i in range(min(500, N)):
            # Individual factual observation
            z_i, x_i, m_i, y_i = Z[i], X[i], M[i], Y[i]
            
            # Abduction
            u_Z_i = z_i
            u_X_i = x_i - 0.8 * z_i
            u_M_i = m_i - 1.2 * x_i
            u_Y_i = y_i - (0.5 * z_i + 1.5 * m_i + 0.4 * x_i)
            
            # Consistency Axiom Check: if we intervene with actual x_i, Y_{X=x_i} must equal y_i
            m_self = 1.2 * x_i + u_M_i
            y_self = 0.5 * z_i + 1.5 * m_self + 0.4 * x_i + u_Y_i
            if np.isclose(y_self, y_i, atol=1e-4):
                consistent_checks += 1
                
            # Counterfactual Query: "What would Y have been if X had been 0 instead?"
            x_cf = 0.0
            m_cf = 1.2 * x_cf + u_M_i
            y_cf = 0.5 * z_i + 1.5 * m_cf + 0.4 * x_cf + u_Y_i
            
            if i < 4:
                samples.append({
                    "sample_id": i + 1,
                    "factual_x": float(x_i),
                    "factual_y": float(y_i),
                    "counterfactual_x": float(x_cf),
                    "counterfactual_y": float(y_cf),
                    "individual_treatment_effect": float(y_i - y_cf)
                })
                
        consistency_pct = float(consistent_checks / min(500, N) * 100.0)
        
        # Treatment response curve over range of interventions
        x_grid = np.linspace(-3.0, 3.0, 50)
        y_resp = [float(np.mean(0.5 * Z + 1.5 * (1.2 * val + U["u_M"]) + 0.4 * val + U["u_Y"])) for val in x_grid]
        
        return EpistemologyBenchmarkResult(
            observational_association=assoc,
            average_treatment_effect_ate=ate,
            natural_direct_effect_nde=nde,
            natural_indirect_effect_nie=nie,
            confounding_bias_gap=confounding_gap,
            counterfactual_consistency_pct=consistency_pct,
            factual_vs_counterfactual_samples=samples,
            treatment_response_curve=(x_grid, np.array(y_resp))
        )
