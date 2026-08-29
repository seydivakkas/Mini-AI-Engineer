"""
KTO (Kahneman-Tversky Optimization) Motoru (Day 205 - FAZ 11).
İkili (Binary Up/Down) Tercihlerle Eşleşmemiş (Unpaired) Asimetrik Kayıp Eğitimi (Ethayarajh et al., 2024).
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KTOModel(nn.Module):
    """KTO Politika Ağı (Policy Model)."""

    def __init__(self, vocab_size: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(self.embedding(x)))
        return self.head(h)


class KTOTrainer:
    """
    Kahneman-Tversky Optimization (KTO) Eğitmeni.
    Beklenti Teorisi (Prospect Theory) ve Kayıptan Kaçınma (Loss Aversion)
    prensibiyle eşleşmemiş (unpaired) tekil Up/Down verilerini optimize eder.
    """

    def __init__(
        self,
        policy: Optional[KTOModel] = None,
        beta: float = 0.1,
        lambda_d: float = 1.0,
        lambda_u: float = 1.33,
        lr: float = 5e-4,
    ):
        self.policy = policy if policy is not None else KTOModel()
        self.ref_policy = KTOModel()
        self.ref_policy.load_state_dict(self.policy.state_dict())
        self.ref_policy.eval()

        self.beta = beta
        self.lambda_d = lambda_d  # Tercih edilen (Upvote) ağırlığı
        self.lambda_u = lambda_u  # Reddedilen (Downvote) asimetrik kayıptan kaçınma ağırlığı
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _sekans_log_prob(self, model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        logits = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1)

    def kto_kaybi_hesapla(
        self,
        input_ids: torch.Tensor,
        is_desirable: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        input_ids: [batch_size, seq_len]
        is_desirable: [batch_size] (True / 1 = Upvote, False / 0 = Downvote)
        """
        # 1. Politika ve Dondurulmuş Referans Log Olasılıkları
        pi_logp = self._sekans_log_prob(self.policy, input_ids)
        with torch.no_grad():
            ref_logp = self._sekans_log_prob(self.ref_policy, input_ids)

        # 2. Örtük Ödül: r(x, y) = beta * (log pi - log pi_ref)
        implicit_rewards = self.beta * (pi_logp - ref_logp)

        # 3. Referans Noktası (Kahneman-Tversky Çapası z_ref)
        with torch.no_grad():
            z_ref = implicit_rewards.mean()

        # 4. Asimetrik Beklenti Değer Kaybı (Prospect Value Function)
        losses = []
        is_desirable_bool = is_desirable.bool()

        for i in range(len(input_ids)):
            r_i = implicit_rewards[i]
            if is_desirable_bool[i]:
                # Pozitif Tercih: 1 - sigma(r_i - z_ref)
                loss_i = self.lambda_d * (1.0 - torch.sigmoid(r_i - z_ref))
            else:
                # Negatif Tercih: 1 - sigma(z_ref - r_i) * lambda_u (Kayıptan Kaçınma)
                loss_i = self.lambda_u * (1.0 - torch.sigmoid(z_ref - r_i))
            losses.append(loss_i)

        total_loss = torch.stack(losses).mean()

        desirable_mask = is_desirable_bool
        undesirable_mask = ~is_desirable_bool

        desirable_r = implicit_rewards[desirable_mask].mean().item() if desirable_mask.any() else 0.0
        undesirable_r = implicit_rewards[undesirable_mask].mean().item() if undesirable_mask.any() else 0.0

        metrikler = {
            "loss": total_loss.item(),
            "desirable_reward": desirable_r,
            "undesirable_reward": undesirable_r,
            "reward_delta": desirable_r - undesirable_r,
            "z_ref": z_ref.item(),
        }

        return total_loss, metrikler

    def egitim_adimi(
        self,
        input_ids: torch.Tensor,
        is_desirable: torch.Tensor,
    ) -> Dict[str, float]:
        """Tek bir KTO asimetrik geri yayılım adımı."""
        self.optimizer.zero_grad()
        loss, metrikler = self.kto_kaybi_hesapla(input_ids, is_desirable)
        loss.backward()
        self.optimizer.step()
        return metrikler
