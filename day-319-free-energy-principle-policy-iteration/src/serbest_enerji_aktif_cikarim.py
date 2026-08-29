"""
Day 319: Free Energy Principle & Continuous Active Inference Policy Iteration.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class FEPConfig:
    num_states: int = 4        # 0: Start, 1: Hint/Cue, 2: Goal A, 3: Goal B
    num_obs: int = 4           # 0: Start, 1: Hint Cue, 2: Reward, 3: Penalty
    num_actions: int = 3       # 0: Inspect Hint, 1: Select A, 2: Select B
    horizon: int = 15
    precision_gamma: float = 8.0
    epistemic_weight: float = 1.0
    pragmatic_weight: float = 1.0
    seed: int = 42


@dataclass
class FEPSimulationResult:
    trajectory_states: List[int]
    trajectory_actions: List[int]
    trajectory_obs: List[int]
    variational_free_energy_history: List[float]
    expected_free_energy_history: List[float]
    epistemic_value_history: List[float]
    pragmatic_value_history: List[float]
    state_entropy_history: List[float]
    goal_reached: bool
    total_epistemic_gain: float
    final_vfe: float


# ---------------------------------------------------------------------------
# Generative Environment
# ---------------------------------------------------------------------------

class GenerativeEnvironment:
    """
    Simulates a contextual T-Maze where checking the Hint (State 1) reveals whether Goal A or B has reward.
    """
    def __init__(self, true_reward_target: int = 2, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.true_target = true_reward_target  # 2: Goal A, 3: Goal B
        self.state = 0  # Start at state 0

    def step(self, action: int) -> Tuple[int, int]:
        """
        Executes action in environment. Returns (next_state, observation).
        Action 0: Move to Hint
        Action 1: Move to Goal A
        Action 2: Move to Goal B
        """
        if action == 0:
            self.state = 1
            obs = 1  # Hint observation
        elif action == 1:
            self.state = 2
            obs = 2 if self.true_target == 2 else 3
        elif action == 2:
            self.state = 3
            obs = 2 if self.true_target == 3 else 3
        else:
            obs = 0
            
        return self.state, obs


# ---------------------------------------------------------------------------
# Active Inference Agent (FEP Policy Engine)
# ---------------------------------------------------------------------------

class ActiveInferenceAgent:
    """
    Minimizes Variational Free Energy (Perception) and Expected Free Energy (Action Planning).
    """
    def __init__(self, config: FEPConfig):
        self.config = config
        self.S = config.num_states
        self.O = config.num_obs
        self.A = config.num_actions
        
        # 1. Observation Likelihood Matrix A (O x S): P(o | s)
        self.A_mat = np.array([
            [0.90, 0.05, 0.05, 0.05],  # obs 0: Start
            [0.05, 0.90, 0.05, 0.05],  # obs 1: Hint
            [0.02, 0.02, 0.90, 0.10],  # obs 2: Reward (Goal A has 0.90 if true)
            [0.03, 0.03, 0.00, 0.80]   # obs 3: Penalty
        ])
        
        # 2. Transition Matrix B (S x S x A): P(s' | s, a)
        self.B_mat = np.zeros((self.S, self.S, self.A))
        # Action 0 -> leads to state 1
        self.B_mat[1, :, 0] = 1.0
        # Action 1 -> leads to state 2
        self.B_mat[2, :, 1] = 1.0
        # Action 2 -> leads to state 3
        self.B_mat[3, :, 2] = 1.0
        
        # 3. Prior Preferences C (O): ln P(o)
        self.C_vec = np.array([0.0, 0.0, 3.0, -3.0])  # Strong preference for Reward (obs 2)
        
        # 4. Initial State Prior D (S)
        self.D_vec = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Current belief state q(s)
        self.q_s = np.copy(self.D_vec)

    def update_beliefs(self, obs: int, prev_action: Optional[int] = None) -> float:
        """
        Perception step: Minimizes Variational Free Energy F(q, y) to update q(s).
        q(s) = softmax( ln A[obs, :] + ln( B[:, :, prev_a] q_prev ) )
        """
        if prev_action is None:
            prior = self.D_vec + 1e-8
        else:
            prior = np.dot(self.B_mat[:, :, prev_action], self.q_s) + 1e-8
            
        likelihood = self.A_mat[obs, :] + 1e-8
        log_joint = np.log(likelihood) + np.log(prior)
        
        # Softmax normalization
        exp_logits = np.exp(log_joint - np.max(log_joint))
        self.q_s = exp_logits / np.sum(exp_logits)
        
        # Calculate Variational Free Energy F = KL(q || p_prior) - E_q[ln p(o|s)]
        kl_div = np.sum(self.q_s * np.log((self.q_s + 1e-12) / prior))
        expected_log_lik = np.sum(self.q_s * np.log(likelihood))
        vfe = float(kl_div - expected_log_lik)
        return vfe

    def evaluate_expected_free_energy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Planning step: Computes Expected Free Energy G(a) for each candidate action.
        G(a) = Pragmatic(a) + Epistemic(a)
        """
        G = np.zeros(self.A)
        pragmatic_vals = np.zeros(self.A)
        epistemic_vals = np.zeros(self.A)
        
        for a in range(self.A):
            # Predicted next state distribution: q(s' | a)
            qs_next = np.dot(self.B_mat[:, :, a], self.q_s)
            
            # Predicted observation distribution: q(o' | a) = A * q(s' | a)
            qo_next = np.dot(self.A_mat, qs_next) + 1e-8
            
            # 1. Pragmatic / Instrumental Value = E_{qo}[C]
            # (Higher C means lower cost / lower free energy, so term is - E[C])
            pragmatic = -float(np.dot(qo_next, self.C_vec))
            
            # 2. Epistemic Value / Information Gain (Mutual Information / Ambiguity Reduction)
            # Epistemic = - E_qs [ KL( A[:, s] || qo ) ]
            epistemic = 0.0
            for s in range(self.S):
                if qs_next[s] > 1e-6:
                    p_o_given_s = self.A_mat[:, s] + 1e-8
                    kl_s = np.sum(p_o_given_s * np.log(p_o_given_s / qo_next))
                    epistemic -= float(qs_next[s] * kl_s)
                    
            pragmatic_vals[a] = pragmatic
            epistemic_vals[a] = epistemic
            
            G[a] = (self.config.pragmatic_weight * pragmatic + 
                    self.config.epistemic_weight * epistemic)
            
        return G, pragmatic_vals, epistemic_vals

    def select_action(self) -> Tuple[int, float, float, float]:
        """
        Selects action via precision-weighted policy posterior: P(a) = softmax(-gamma * G)
        """
        G, prag, epis = self.evaluate_expected_free_energy()
        
        # Action probabilities
        logits = -self.config.precision_gamma * G
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        
        action = int(np.argmax(probs))
        return action, float(G[action]), float(prag[action]), float(epis[action])

    def run_active_inference_loop(self, env: GenerativeEnvironment) -> FEPSimulationResult:
        """
        Executes perception-action active inference loop over horizon.
        """
        states, actions, observations = [env.state], [], [0]
        vfe_hist, efe_hist, epis_hist, prag_hist, entropy_hist = [], [], [], [], []
        
        # Initial perception
        init_vfe = self.update_beliefs(obs=0, prev_action=None)
        vfe_hist.append(init_vfe)
        entropy_hist.append(float(-np.sum(self.q_s * np.log(self.q_s + 1e-12))))
        
        total_epistemic = 0.0
        goal_reached = False
        
        for t in range(self.config.horizon):
            action, g_val, p_val, e_val = self.select_action()
            actions.append(action)
            efe_hist.append(g_val)
            prag_hist.append(p_val)
            epis_hist.append(e_val)
            total_epistemic += abs(e_val)
            
            # Step environment
            next_state, obs = env.step(action)
            states.append(next_state)
            observations.append(obs)
            
            # Perception update
            vfe = self.update_beliefs(obs, prev_action=action)
            vfe_hist.append(vfe)
            entropy = float(-np.sum(self.q_s * np.log(self.q_s + 1e-12)))
            entropy_hist.append(entropy)
            
            if obs == 2:  # Reached reward
                goal_reached = True
                break
                
        return FEPSimulationResult(
            trajectory_states=states,
            trajectory_actions=actions,
            trajectory_obs=observations,
            variational_free_energy_history=vfe_hist,
            expected_free_energy_history=efe_hist,
            epistemic_value_history=epis_hist,
            pragmatic_value_history=prag_hist,
            state_entropy_history=entropy_hist,
            goal_reached=goal_reached,
            total_epistemic_gain=float(total_epistemic),
            final_vfe=float(vfe_hist[-1])
        )
