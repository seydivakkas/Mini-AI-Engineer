"""
Day 318: Neuro-Symbolic Continuous Logic & Differentiable Fuzzy Theorem Prover.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class TNormType(Enum):
    PRODUCT = "product"
    LUKASIEWICZ = "lukasiewicz"
    GODEL = "godel"


@dataclass
class ContinuousLogicConfig:
    t_norm: TNormType = TNormType.LUKASIEWICZ
    embedding_dim: int = 16
    temperature: float = 0.5
    num_entities: int = 6
    num_steps: int = 50
    learning_rate: float = 0.05
    logic_loss_weight: float = 1.0
    seed: int = 42


@dataclass
class NeuroSymbolicResult:
    t_norm_name: str
    final_task_loss: float
    final_logical_violation_loss: float
    total_loss: float
    theorem_proof_accuracy_pct: float
    rule_satisfaction_rates: Dict[str, float]
    proven_queries: List[Dict[str, Any]]
    loss_history: List[float]
    rule_satisfaction_history: List[float]


# ---------------------------------------------------------------------------
# Continuous Logic Operator Bank
# ---------------------------------------------------------------------------

class ContinuousLogicEngine:
    """
    Implements continuous fuzzy logic operations (Conjunction T, Disjunction S, 
    Implication I, Negation N) for Product, Łukasiewicz, and Gödel T-Norms.
    """
    
    @staticmethod
    def conjunction(a: np.ndarray, b: np.ndarray, t_norm: TNormType) -> np.ndarray:
        """Fuzzy AND (T-Norm)"""
        a = np.clip(a, 0.0, 1.0)
        b = np.clip(b, 0.0, 1.0)
        if t_norm == TNormType.PRODUCT:
            return a * b
        elif t_norm == TNormType.LUKASIEWICZ:
            return np.maximum(0.0, a + b - 1.0)
        elif t_norm == TNormType.GODEL:
            return np.minimum(a, b)
        raise ValueError(f"Unknown T-Norm: {t_norm}")

    @staticmethod
    def disjunction(a: np.ndarray, b: np.ndarray, t_norm: TNormType) -> np.ndarray:
        """Fuzzy OR (T-Conorm / S-Norm)"""
        a = np.clip(a, 0.0, 1.0)
        b = np.clip(b, 0.0, 1.0)
        if t_norm == TNormType.PRODUCT:
            return a + b - a * b
        elif t_norm == TNormType.LUKASIEWICZ:
            return np.minimum(1.0, a + b)
        elif t_norm == TNormType.GODEL:
            return np.maximum(a, b)
        raise ValueError(f"Unknown T-Norm: {t_norm}")

    @staticmethod
    def implication(a: np.ndarray, b: np.ndarray, t_norm: TNormType) -> np.ndarray:
        """Fuzzy Implication (Residual Implication I(a, b): a -> b)"""
        a = np.clip(a, 0.0, 1.0)
        b = np.clip(b, 0.0, 1.0)
        if t_norm == TNormType.PRODUCT:
            return np.minimum(1.0, (b + 1e-8) / (a + 1e-8))
        elif t_norm == TNormType.LUKASIEWICZ:
            return np.minimum(1.0, 1.0 - a + b)
        elif t_norm == TNormType.GODEL:
            return np.where(a <= b, 1.0, b)
        raise ValueError(f"Unknown T-Norm: {t_norm}")

    @staticmethod
    def negation(a: np.ndarray) -> np.ndarray:
        """Standard Fuzzy NOT"""
        return 1.0 - np.clip(a, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Differentiable Soft Theorem Prover
# ---------------------------------------------------------------------------

class SoftTheoremProver:
    """
    Differentiable First-Order Logic (dFOL) Theorem Prover over Neural Knowledge Graphs.
    Proves Ancestor relations through recursive soft backward chaining:
      R1: Parent(X, Y) => Ancestor(X, Y)
      R2: Parent(X, Y) ^ Ancestor(Y, Z) => Ancestor(X, Z)
    """
    def __init__(self, config: ContinuousLogicConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        # Knowledge Base Entities: 0: Alice, 1: Bob, 2: Charlie, 3: Dave, 4: Eve, 5: Frank
        self.N = config.num_entities
        self.D = config.embedding_dim
        
        # Initialize Neural Embeddings
        self.entity_embeddings = self.rng.normal(0.0, 0.5, size=(self.N, self.D))
        self.relation_parent = self.rng.normal(0.0, 0.5, size=(self.D, self.D))
        self.relation_ancestor = self.rng.normal(0.0, 0.5, size=(self.D, self.D))
        
        # Ground Truth Facts: (Alice -> Bob), (Bob -> Charlie), (Charlie -> Dave)
        self.ground_parent_facts = [(0, 1), (1, 2), (2, 3), (3, 4)]
        
    def predict_predicate(self, head_idx: int, tail_idx: int, relation_matrix: np.ndarray) -> float:
        """
        Computes soft truth value in [0, 1] using bilinear scoring: sigmoid(e_h^T W e_t / temp)
        """
        e_h = self.entity_embeddings[head_idx]
        e_t = self.entity_embeddings[tail_idx]
        score = np.dot(e_h, np.dot(relation_matrix, e_t)) / self.config.temperature
        # Sigmoid
        return float(1.0 / (1.0 + np.exp(-np.clip(score, -15.0, 15.0))))

    def prove_ancestor_query(self, head_idx: int, tail_idx: int, max_depth: int = 3) -> Tuple[float, List[str]]:
        """
        Differentiable backward-chaining proof search.
        Ancestor(X, Z) = Base(Parent(X, Z)) v Max_{Y} [ Parent(X, Y) ^ Ancestor(Y, Z) ]
        """
        t_norm = self.config.t_norm
        
        # Direct Base Rule: R1
        r1_val = self.predict_predicate(head_idx, tail_idx, self.relation_parent)
        proof_steps = [f"Base Parent({head_idx}, {tail_idx}) = {r1_val:.3f}"]
        
        best_val = r1_val
        
        if max_depth > 1:
            for y in range(self.N):
                if y != head_idx and y != tail_idx:
                    p_xy = self.predict_predicate(head_idx, y, self.relation_parent)
                    # Recursive sub-proof with depth-1
                    sub_val, _ = self.prove_ancestor_query(y, tail_idx, max_depth=max_depth - 1)
                    
                    r2_conj = ContinuousLogicEngine.conjunction(np.array([p_xy]), np.array([sub_val]), t_norm)[0]
                    if r2_conj > best_val:
                        best_val = ContinuousLogicEngine.disjunction(np.array([best_val]), np.array([r2_conj]), t_norm)[0]
                        proof_steps.append(f"Recursive Parent({head_idx}, {y})={p_xy:.2f} ^ Anc({y}, {tail_idx})={sub_val:.2f} => Step={r2_conj:.3f}")
                        
        return float(best_val), proof_steps

    def evaluate_axioms(self) -> Dict[str, float]:
        """
        Evaluates logical satisfaction rate (truth value of implication) for all axioms:
        Axiom 1 (Base): Parent(X, Y) => Ancestor(X, Y)
        Axiom 2 (Transitivity): Parent(X, Y) ^ Ancestor(Y, Z) => Ancestor(X, Z)
        Axiom 3 (Asymmetry): Ancestor(X, Y) => NOT Ancestor(Y, X)
        """
        t_norm = self.config.t_norm
        
        # Axiom 1 Satisfaction
        ax1_vals = []
        for x in range(self.N):
            for y in range(self.N):
                p_xy = self.predict_predicate(x, y, self.relation_parent)
                a_xy = self.predict_predicate(x, y, self.relation_ancestor)
                imp = ContinuousLogicEngine.implication(np.array([p_xy]), np.array([a_xy]), t_norm)[0]
                ax1_vals.append(imp)
                
        # Axiom 2 Satisfaction
        ax2_vals = []
        for x in range(self.N):
            for y in range(self.N):
                for z in range(self.N):
                    p_xy = self.predict_predicate(x, y, self.relation_parent)
                    a_yz = self.predict_predicate(y, z, self.relation_ancestor)
                    a_xz = self.predict_predicate(x, z, self.relation_ancestor)
                    
                    premise = ContinuousLogicEngine.conjunction(np.array([p_xy]), np.array([a_yz]), t_norm)[0]
                    imp = ContinuousLogicEngine.implication(np.array([premise]), np.array([a_xz]), t_norm)[0]
                    ax2_vals.append(imp)
                    
        # Axiom 3 Satisfaction (Asymmetry)
        ax3_vals = []
        for x in range(self.N):
            for y in range(x + 1, self.N):
                a_xy = self.predict_predicate(x, y, self.relation_ancestor)
                a_yx = self.predict_predicate(y, x, self.relation_ancestor)
                not_yx = ContinuousLogicEngine.negation(np.array([a_yx]))[0]
                imp = ContinuousLogicEngine.implication(np.array([a_xy]), np.array([not_yx]), t_norm)[0]
                ax3_vals.append(imp)
                
        return {
            "Axiom_1_Base": float(np.mean(ax1_vals)),
            "Axiom_2_Transitivity": float(np.mean(ax2_vals)),
            "Axiom_3_Asymmetry": float(np.mean(ax3_vals))
        }

    def train_and_prove(self) -> NeuroSymbolicResult:
        """
        Executes neuro-symbolic optimization loop with combined supervision and logical axiom loss.
        """
        loss_history = []
        sat_history = []
        
        for step in range(self.config.num_steps):
            # 1. Supervised Task Loss (Binary Cross-Entropy on ground parent facts)
            task_loss = 0.0
            for h, t in self.ground_parent_facts:
                pred = self.predict_predicate(h, t, self.relation_parent)
                task_loss -= np.log(np.clip(pred, 1e-6, 1.0 - 1e-6))
                
                # Direct synthetic gradient step on relation matrix
                grad = (pred - 1.0) * np.outer(self.entity_embeddings[h], self.entity_embeddings[t])
                self.relation_parent -= self.config.learning_rate * grad
                
            task_loss /= len(self.ground_parent_facts)
            
            # 2. Logical Axiom Loss (1.0 - mean(Implications))
            axioms = self.evaluate_axioms()
            mean_axiom_truth = np.mean(list(axioms.values()))
            logic_loss = 1.0 - mean_axiom_truth
            
            # Align Ancestor matrix with Parent + Transitivity
            self.relation_ancestor += self.config.learning_rate * (self.relation_parent - self.relation_ancestor * 0.1)
            
            total_loss = task_loss + self.config.logic_loss_weight * logic_loss
            loss_history.append(float(total_loss))
            sat_history.append(float(mean_axiom_truth * 100.0))
            
        # Prove Test Queries: (0 -> 2, Alice -> Charlie), (0 -> 3, Alice -> Dave), (1 -> 3, Bob -> Dave)
        test_queries = [(0, 2), (0, 3), (1, 3), (0, 4)]
        proven = []
        correct = 0
        
        for h, t in test_queries:
            val, steps = self.prove_ancestor_query(h, t, max_depth=3)
            is_valid = val > 0.50
            if is_valid:
                correct += 1
            proven.append({
                "query": f"Ancestor({h}, {t})",
                "truth_value": round(val, 4),
                "is_proven": is_valid,
                "proof_trace": steps[-1] if steps else "Direct"
            })
            
        accuracy_pct = float(correct / len(test_queries) * 100.0)
        final_axioms = self.evaluate_axioms()
        
        return NeuroSymbolicResult(
            t_norm_name=self.config.t_norm.value,
            final_task_loss=float(loss_history[-1]),
            final_logical_violation_loss=float(1.0 - np.mean(list(final_axioms.values()))),
            total_loss=float(loss_history[-1]),
            theorem_proof_accuracy_pct=accuracy_pct,
            rule_satisfaction_rates=final_axioms,
            proven_queries=proven,
            loss_history=loss_history,
            rule_satisfaction_history=sat_history
        )
