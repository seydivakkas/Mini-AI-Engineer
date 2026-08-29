"""
Day 313: Contrastive Decoding Anti-Hallucination Engine.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContrastiveDecodingConfig:
    vocab_size: int = 256
    alpha: float = 1.2          # Amateur penalization weight
    beta: float = 0.1           # Adaptive plausibility threshold
    temperature: float = 0.8
    num_prompts: int = 50
    generation_length: int = 25
    seed: int = 42


@dataclass
class ContrastiveDecodingResult:
    standard_factuality_pct: float
    contrastive_factuality_pct: float
    hallucination_reduction_pct: float
    standard_ece: float
    contrastive_ece: float
    tokens_generated: int
    step_factuality_trajectory_std: np.ndarray
    step_factuality_trajectory_cd: np.ndarray
    sample_generations: List[Dict[str, Any]]


class ContrastiveDecoderEngine:
    """
    Simulates expert and amateur models to execute Adaptive Plausibility Contrastive Decoding.
    """
    def __init__(self, config: ContrastiveDecodingConfig):
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
    def generate_step_logits(self, prompt_id: int, step: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Generates simulated expert and amateur logits for a decoding step.
        Returns (expert_logits, amateur_logits, ground_truth_token).
        """
        V = self.config.vocab_size
        
        # Ground truth factual token
        true_token = (prompt_id * 7 + step * 3) % V
        hallucination_distractor = (true_token + 13) % V
        
        # Expert logits: Has knowledge of true token (3.5), but standard greedy gets pulled by distractor (3.8)
        expert_logits = torch.randn(V) * 0.5
        expert_logits[true_token] += 3.5
        expert_logits[hallucination_distractor] += 3.8
        
        # Amateur logits: Lacks deep grounding; strongly biased toward superficial hallucination distractor (5.0), zero/negative on hard true token
        amateur_logits = torch.randn(V) * 0.5
        amateur_logits[hallucination_distractor] += 5.0
        amateur_logits[true_token] -= 0.5
        
        return expert_logits, amateur_logits, true_token

    def contrastive_decode_step(self, expert_logits: torch.Tensor, amateur_logits: torch.Tensor) -> int:
        """
        Applies Adaptive Plausibility Head Truncation & Contrastive Logit Penalization.
        z_CD = z_expert - alpha * z_amateur
        """
        # 1. Expert probabilities
        p_exp = F.softmax(expert_logits / self.config.temperature, dim=-1)
        max_p = torch.max(p_exp)
        
        # 2. Plausibility Mask (V_head)
        plausibility_mask = p_exp >= (self.config.beta * max_p)
        
        # 3. Contrastive Logits
        z_cd = expert_logits - self.config.alpha * amateur_logits
        
        # 4. Truncate non-plausible tokens
        z_cd[~plausibility_mask] = -1e9
        
        # 5. Greedy / Temperature selection
        selected_token = int(torch.argmax(z_cd).item())
        return selected_token

    def run_benchmark(self) -> ContrastiveDecodingResult:
        """
        Executes full comparative decoding benchmark across prompt batch.
        """
        num_prompts = self.config.num_prompts
        gen_len = self.config.generation_length
        
        std_correct = 0
        cd_correct = 0
        total_tokens = num_prompts * gen_len
        
        std_confidences = []
        std_accuracies = []
        cd_confidences = []
        cd_accuracies = []
        
        std_step_accs = [0] * gen_len
        cd_step_accs = [0] * gen_len
        
        sample_gens = []
        
        for p_id in range(num_prompts):
            p_std_tokens = []
            p_cd_tokens = []
            p_true_tokens = []
            
            for step in range(gen_len):
                exp_logits, ama_logits, true_token = self.generate_step_logits(p_id, step)
                p_true_tokens.append(true_token)
                
                # Standard Decoding (Greedy on Expert)
                std_token = int(torch.argmax(exp_logits).item())
                p_std_tokens.append(std_token)
                is_std_ok = (std_token == true_token)
                if is_std_ok:
                    std_correct += 1
                    std_step_accs[step] += 1
                std_conf = float(F.softmax(exp_logits, dim=-1)[std_token].item())
                std_confidences.append(std_conf)
                std_accuracies.append(1.0 if is_std_ok else 0.0)
                
                # Contrastive Decoding
                cd_token = self.contrastive_decode_step(exp_logits, ama_logits)
                p_cd_tokens.append(cd_token)
                is_cd_ok = (cd_token == true_token)
                if is_cd_ok:
                    cd_correct += 1
                    cd_step_accs[step] += 1
                
                # Confidence from normalized contrastive distribution
                z_cd = exp_logits - self.config.alpha * ama_logits
                cd_conf = float(F.softmax(z_cd, dim=-1)[cd_token].item())
                cd_confidences.append(cd_conf)
                cd_accuracies.append(1.0 if is_cd_ok else 0.0)
                
            if p_id < 3:
                sample_gens.append({
                    "prompt_id": p_id + 1,
                    "std_accuracy": float(np.mean(np.array(p_std_tokens) == np.array(p_true_tokens)) * 100.0),
                    "cd_accuracy": float(np.mean(np.array(p_cd_tokens) == np.array(p_true_tokens)) * 100.0)
                })
                
        std_factuality = float(std_correct / total_tokens * 100.0)
        cd_factuality = float(cd_correct / total_tokens * 100.0)
        
        # Hallucination Reduction Rate (%)
        std_hallucinations = total_tokens - std_correct
        cd_hallucinations = total_tokens - cd_correct
        hallucination_reduction = float((std_hallucinations - cd_hallucinations) / max(std_hallucinations, 1) * 100.0)
        
        # Expected Calibration Error (ECE) calculation
        def calculate_ece(confs, accs, num_bins=10):
            confs = np.array(confs)
            accs = np.array(accs)
            bins = np.linspace(0, 1, num_bins + 1)
            ece = 0.0
            for i in range(num_bins):
                bin_mask = (confs >= bins[i]) & (confs < bins[i+1])
                if np.sum(bin_mask) > 0:
                    bin_acc = np.mean(accs[bin_mask])
                    bin_conf = np.mean(confs[bin_mask])
                    ece += np.sum(bin_mask) / len(confs) * np.abs(bin_acc - bin_conf)
            return float(ece)
            
        std_ece = calculate_ece(std_confidences, std_accuracies)
        cd_ece = calculate_ece(cd_confidences, cd_accuracies)
        
        step_traj_std = np.array(std_step_accs) / num_prompts * 100.0
        step_traj_cd = np.array(cd_step_accs) / num_prompts * 100.0
        
        return ContrastiveDecodingResult(
            standard_factuality_pct=std_factuality,
            contrastive_factuality_pct=cd_factuality,
            hallucination_reduction_pct=hallucination_reduction,
            standard_ece=std_ece,
            contrastive_ece=cd_ece,
            tokens_generated=total_tokens,
            step_factuality_trajectory_std=step_traj_std,
            step_factuality_trajectory_cd=step_traj_cd,
            sample_generations=sample_gens
        )
