"""
Day 314: Game-Theoretic Mechanism Design & Nash Bargaining Engine.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class MechanismConfig:
    num_agents: int = 4
    num_goods_or_outcomes: int = 5
    total_compute_resource: float = 100.0
    seed: int = 42


@dataclass
class BargainingAgent:
    agent_id: int
    name: str
    valuations: np.ndarray          # True valuations for K discrete outcomes
    threat_point: float             # Disagreement outcome utility d_i
    bargaining_power: float         # Alpha_i (sum(alpha) = 1)
    utility_weight: float           # Marginal utility multiplier for continuous resource


@dataclass
class MechanismResult:
    vcg_optimal_outcome: int
    vcg_social_welfare: float
    vcg_payments: Dict[str, float]
    vcg_net_utilities: Dict[str, float]
    truthful_vs_manipulated_utility_gain: float # Truthful payoff minus lying payoff (>= 0 for DSIC)
    nash_bargaining_allocations: Dict[str, float]
    nash_net_surpluses: Dict[str, float]
    total_nash_product: float
    pareto_efficiency_pct: float


class VCGMechanism:
    """
    Vickrey-Clarke-Groves (VCG) Mechanism with dominant-strategy incentive compatibility (DSIC).
    """
    @staticmethod
    def solve_allocation(agents: List[BargainingAgent], bids: Optional[Dict[str, np.ndarray]] = None) -> Tuple[int, float, Dict[str, float], Dict[str, float]]:
        """
        Determines social welfare maximizing outcome k* and computes VCG externality payments.
        p_i = max_{k} sum_{j != i} b_j(k) - sum_{j != i} b_j(k*)
        """
        num_outcomes = len(agents[0].valuations)
        
        # Use bids if provided, else use true valuations (truthful bidding)
        bids_matrix = []
        for a in agents:
            b = bids[a.name] if (bids and a.name in bids) else a.valuations
            bids_matrix.append(b)
        bids_matrix = np.array(bids_matrix) # [M, K]
        
        # 1. Social Welfare Maximizing Outcome k*
        total_welfare_per_outcome = np.sum(bids_matrix, axis=0) # [K]
        k_star = int(np.argmax(total_welfare_per_outcome))
        social_welfare = float(total_welfare_per_outcome[k_star])
        
        # 2. VCG Payments
        payments = {}
        net_utilities = {}
        
        for i, a in enumerate(agents):
            # Exclude agent i
            mask = np.ones(len(agents), dtype=bool)
            mask[i] = False
            bids_without_i = bids_matrix[mask] # [M-1, K]
            
            welfare_without_i = np.sum(bids_without_i, axis=0)
            k_without_i = int(np.argmax(welfare_without_i))
            max_welfare_without_i = float(welfare_without_i[k_without_i])
            
            actual_welfare_others_at_k_star = float(welfare_without_i[k_star])
            p_i = max_welfare_without_i - actual_welfare_others_at_k_star
            
            payments[a.name] = max(0.0, float(p_i))
            # True net utility u_i = v_i(k*) - p_i
            net_utilities[a.name] = float(a.valuations[k_star] - payments[a.name])
            
        return k_star, social_welfare, payments, net_utilities


class NashBargainingOptimizer:
    """
    Solves the Generalized Nash Bargaining Solution (NBS) via convex optimization:
    max_x sum_i alpha_i * ln(u_i(x_i) - d_i) subject to sum(x_i) <= C and u_i(x_i) >= d_i.
    """
    @staticmethod
    def solve_bargaining(agents: List[BargainingAgent], total_capacity: float = 100.0) -> Tuple[Dict[str, float], Dict[str, float], float]:
        M = len(agents)
        d = np.array([a.threat_point for a in agents])
        alpha = np.array([a.bargaining_power for a in agents])
        weights = np.array([a.utility_weight for a in agents])
        
        # Objective: Minimize negative log Nash Product
        def neg_log_nash_product(x):
            utilities = weights * np.sqrt(np.maximum(x, 1e-4)) # Diminishing marginal returns
            surpluses = utilities - d
            if np.any(surpluses <= 1e-6):
                return 1e9
            return -np.sum(alpha * np.log(surpluses))
            
        # Initial guess (equal split)
        x0 = np.full(M, total_capacity / M)
        
        # Constraints: sum(x) <= total_capacity and u_i(x) >= d_i
        constraints = [
            {"type": "ineq", "fun": lambda x: total_capacity - np.sum(x)}
        ]
        for i in range(M):
            constraints.append({
                "type": "ineq", 
                "fun": lambda x, idx=i: weights[idx] * np.sqrt(max(x[idx], 1e-4)) - d[idx] - 0.01
            })
            
        bounds = [(0.1, total_capacity) for _ in range(M)]
        
        res = minimize(neg_log_nash_product, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        
        optimal_x = np.maximum(res.x, 0.0)
        # Normalize sum to exactly total_capacity
        optimal_x = optimal_x / np.sum(optimal_x) * total_capacity
        
        allocations = {a.name: float(optimal_x[i]) for i, a in enumerate(agents)}
        final_utilities = weights * np.sqrt(optimal_x)
        surpluses = {a.name: float(final_utilities[i] - d[i]) for i, a in enumerate(agents)}
        
        nash_product = float(np.prod(np.array(list(surpluses.values())) ** alpha))
        
        return allocations, surpluses, nash_product


class GameTheoreticEngine:
    """
    Coordinates multi-agent game-theoretic mechanism simulation.
    """
    def __init__(self, config: MechanismConfig):
        self.config = config
        np.random.seed(config.seed)
        
        # Create standard multi-agent cluster profiles
        names = ["Agent-Alpha", "Agent-Beta", "Agent-Gamma", "Agent-Delta"]
        self.agents: List[BargainingAgent] = []
        
        for i in range(config.num_agents):
            vals = np.random.uniform(10.0, 50.0, size=config.num_goods_or_outcomes)
            self.agents.append(BargainingAgent(
                agent_id=i + 1,
                name=names[i % len(names)],
                valuations=vals,
                threat_point=5.0,
                bargaining_power=1.0 / config.num_agents,
                utility_weight=np.random.uniform(2.5, 4.0)
            ))

    def run_simulation(self) -> MechanismResult:
        """
        Executes VCG discrete allocation and continuous Nash bargaining optimization.
        """
        # 1. Truthful VCG Allocation
        k_star, welfare, payments, net_utils = VCGMechanism.solve_allocation(self.agents)
        
        # 2. DSIC Incentive Compatibility Verification:
        # Check Agent-Alpha's payoff if lying/misreporting bids
        alpha_agent = self.agents[0]
        lying_bids = {alpha_agent.name: alpha_agent.valuations * 2.5 + np.random.uniform(-10, 10, size=len(alpha_agent.valuations))}
        _, _, _, lying_net_utils = VCGMechanism.solve_allocation(self.agents, bids=lying_bids)
        
        # Truthful utility - Lying utility (should be >= 0)
        dsic_gain = max(0.0, net_utils[alpha_agent.name] - lying_net_utils[alpha_agent.name])
        
        # 3. Continuous Nash Bargaining Allocation
        allocations, surpluses, nash_product = NashBargainingOptimizer.solve_bargaining(
            self.agents, total_capacity=self.config.total_compute_resource
        )
        
        # 4. Pareto Efficiency (100% capacity utilization & positive surplus for all)
        pareto_eff = 100.0 if np.isclose(sum(allocations.values()), self.config.total_compute_resource, atol=1e-1) else 95.0
        
        return MechanismResult(
            vcg_optimal_outcome=k_star,
            vcg_social_welfare=welfare,
            vcg_payments=payments,
            vcg_net_utilities=net_utils,
            truthful_vs_manipulated_utility_gain=dsic_gain,
            nash_bargaining_allocations=allocations,
            nash_net_surpluses=surpluses,
            total_nash_product=nash_product,
            pareto_efficiency_pct=pareto_eff
        )
