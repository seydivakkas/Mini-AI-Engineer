"""
DPO (Direct Preference Optimization) Motoru (Day 204 - FAZ 11).
Ödül Modeli Olmadan Kapalı Formda İkili Tercih Kaybı Eğitimi (Rafailov et al., 2023).
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DPOModel(nn.Module):
    """DPO Politika Ağı (Policy Model)."""

    def __init__(self, vocab_size: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(self.embedding(x)))
        return self.head(h)


class DPOTrainer:
    """
    Direct Preference Optimization (DPO) Eğitmeni.
    Ayrı bir Ödül Modeli (Reward Model) veya RL Ajanı gerektirmeden
    tercih edilen (chosen - y_w) ve reddedilen (rejected - y_l) verileriyle eğitilir.
    """

    def __init__(
        self,
        policy: Optional[DPOModel] = None,
        beta: float = 0.1,
        lr: float = 5e-4,
    ):
        self.policy = policy if policy is not None else DPOModel()
        self.ref_policy = DPOModel()
        self.ref_policy.load_state_dict(self.policy.state_dict())
        self.ref_policy.eval()

        self.beta = beta
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _sekans_log_prob_hesapla(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len]
        Dönüş: [batch_size] sekans bazlı toplam log olasılığı
        """
        logits = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        # Her token için seçilen kelimenin log_prob değerini topla
        token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1)

    def dpo_kaybi_hesapla(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        L_DPO = -E [ log sigma( beta * log(pi(y_w)/pi_ref(y_w)) - beta * log(pi(y_l)/pi_ref(y_l)) ) ]
        """
        # 1. Politika Modeli Log Olasılıkları
        pi_chosen_logp = self._sekans_log_prob_hesapla(self.policy, chosen_ids)
        pi_rejected_logp = self._sekans_log_prob_hesapla(self.policy, rejected_ids)

        # 2. Dondurulmuş Referans Model Log Olasılıkları
        with torch.no_grad():
            ref_chosen_logp = self._sekans_log_prob_hesapla(self.ref_policy, chosen_ids)
            ref_rejected_logp = self._sekans_log_prob_hesapla(self.ref_policy, rejected_ids)

        # 3. Örtük (Implicit) Ödül Skorları
        chosen_rewards = self.beta * (pi_chosen_logp - ref_chosen_logp)
        rejected_rewards = self.beta * (pi_rejected_logp - ref_rejected_logp)

        # 4. Bradley-Terry Tercih Farkı (Logit Farkı)
        logits = chosen_rewards - rejected_rewards

        # 5. DPO Kaybı
        losses = -F.logsigmoid(logits)
        loss = losses.mean()

        # Metrikler
        chosen_rewards_mean = chosen_rewards.mean().item()
        rejected_rewards_mean = rejected_rewards.mean().item()
        reward_margin = chosen_rewards_mean - rejected_rewards_mean
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        metrikler = {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards_mean,
            "rejected_reward": rejected_rewards_mean,
            "reward_margin": reward_margin,
            "accuracy": accuracy,
        }

        return loss, metrikler

    def egitim_adimi(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """Tek bir DPO geri yayılım adımı."""
        self.optimizer.zero_grad()
        loss, metrikler = self.dpo_kaybi_hesapla(chosen_ids, rejected_ids)
        loss.backward()
        self.optimizer.step()
        return metrikler
