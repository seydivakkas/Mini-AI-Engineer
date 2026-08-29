"""
Simple Preference Optimization (SimPO) Kayıp Fonksiyonu Modülü (Day 113).
Referans modelsiz (pi_ref olmadan), doğrudan ortalama token log-olasılığı ve hedef marjin (gamma) ile hizalama.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimPOLoss(nn.Module):
    """
    SimPO Kayıp Fonksiyonu:
    L_SimPO = -log sigma( beta/|y_w| * log pi(y_w|x) - beta/|y_l| * log pi(y_l|x) - gamma )
            = -log sigma( beta * (avg_logp_w - avg_logp_l) - gamma )
    """

    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        chosen_logps_per_token: torch.Tensor,
        rejected_logps_per_token: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        chosen_logps_per_token: [B, S_c]
        rejected_logps_per_token: [B, S_r]
        chosen_mask: [B, S_c] (Yanıt kısmı 1, prompt/pad 0)
        rejected_mask: [B, S_r] (Yanıt kısmı 1, prompt/pad 0)
        """
        # 1. Uzunluk Normalizasyonu: Dizi uzunluğuna bölerek ortalama token log-olasılıklarını al
        chosen_token_count = chosen_mask.sum(dim=-1).clamp(min=1.0)
        rejected_token_count = rejected_mask.sum(dim=-1).clamp(min=1.0)

        avg_logp_chosen = (chosen_logps_per_token * chosen_mask).sum(dim=-1) / chosen_token_count
        avg_logp_rejected = (rejected_logps_per_token * rejected_mask).sum(dim=-1) / rejected_token_count

        # 2. Örtük Ödül (Implicit Reward) Tanımı: r(x, y) = beta * avg_logp(y)
        chosen_rewards = self.beta * avg_logp_chosen
        rejected_rewards = self.beta * avg_logp_rejected
        reward_farki = chosen_rewards - rejected_rewards  # [B]

        # 3. Hedef Marjinli SimPO Kaybı: -log sigmoid( reward_farki - gamma )
        logits = reward_farki - self.gamma
        kayip = -F.logsigmoid(logits).mean()

        # 4. Tanısal Metrikler
        dogruluk = (reward_farki > 0.0).float().mean()
        marjin_ihlali = (reward_farki < self.gamma).float().mean()

        metrikler = {
            "kayip": kayip.detach(),
            "chosen_odul": chosen_rewards.mean().detach(),
            "rejected_odul": rejected_rewards.mean().detach(),
            "odul_farki": reward_farki.mean().detach(),
            "marjin_ihlali": marjin_ihlali.detach(),
            "dogruluk": dogruluk.detach(),
        }

        return kayip, metrikler


def hesapla_token_bazli_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Logitlerden her bir token adımı için log-olasılık tensörünü [B, S] hesaplar."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    B = logits.shape[0]
    sifir_bas = torch.zeros((B, 1), device=logits.device, dtype=logits.dtype)
    return torch.cat([sifir_bas, per_token_logps], dim=1)
