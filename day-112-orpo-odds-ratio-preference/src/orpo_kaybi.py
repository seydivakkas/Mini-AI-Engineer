"""
Odds Ratio Preference Optimization (ORPO) Kayıp Fonksiyonu Modülü (Day 112).
Tek aşamalı SFT + Odds Ratio (Bahis Oranları) ceza mekanizması.
Referans model ve ayrı ödül modeli olmadan monolitik hizalama.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def sayisal_kararli_log_odds(avg_logp: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Ortalama token log-olasılığından sayısal kararlı log(odds) hesaplar.
    odds = p / (1 - p) -> log(odds) = log(p) - log(1 - p) = l - log1p(-exp(l))
    avg_logp: Negatif tensör (<= 0)
    """
    # p = exp(avg_logp) değerinin 1'e aşırı yaklaşmasını sınırla
    clamped_logp = torch.clamp(avg_logp, max=-eps)
    # log(1 - exp(clamped_logp)) = log(-expm1(clamped_logp))
    log_one_minus_p = torch.log(-torch.expm1(clamped_logp) + eps)
    return clamped_logp - log_one_minus_p


class ORPOLoss(nn.Module):
    """
    ORPO Monolitik Kayıp Fonksiyonu:
    L_ORPO = L_SFT(chosen) + lambda_or * L_OR(chosen, rejected)
    L_OR = -log sigma( log odds(chosen) - log odds(rejected) )
    """

    def __init__(self, lambda_or: float = 0.5):
        super().__init__()
        self.lambda_or = lambda_or

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
        # 1. SFT Kaybı (Yalnızca tercih edilen chosen dizisi üzerinde ortalama NLL)
        chosen_token_count = chosen_mask.sum(dim=-1).clamp(min=1.0)
        rejected_token_count = rejected_mask.sum(dim=-1).clamp(min=1.0)

        # Ortalama token log-olasılıkları: [B]
        avg_logp_chosen = (chosen_logps_per_token * chosen_mask).sum(dim=-1) / chosen_token_count
        avg_logp_rejected = (rejected_logps_per_token * rejected_mask).sum(dim=-1) / rejected_token_count

        kayip_sft = -avg_logp_chosen.mean()

        # 2. Odds Ratio (Bahis Oranları) Hesabı
        log_odds_chosen = sayisal_kararli_log_odds(avg_logp_chosen)
        log_odds_rejected = sayisal_kararli_log_odds(avg_logp_rejected)

        log_odds_ratio = log_odds_chosen - log_odds_rejected  # [B]

        # 3. Odds Ratio Kaybı: -log sigmoid(log_odds_ratio)
        kayip_or = -F.logsigmoid(log_odds_ratio).mean()

        # 4. Toplam Monolitik ORPO Kaybı
        toplam_kayip = kayip_sft + self.lambda_or * kayip_or

        # Doğruluk (% Accuracy: chosen odds > rejected odds)
        dogruluk = (log_odds_ratio > 0.0).float().mean()

        metrikler = {
            "toplam_kayip": toplam_kayip.detach(),
            "kayip_sft": kayip_sft.detach(),
            "kayip_or": kayip_or.detach(),
            "log_odds_ratio": log_odds_ratio.mean().detach(),
            "dogruluk": dogruluk.detach(),
        }

        return toplam_kayip, metrikler


def hesapla_token_bazli_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Logitlerden her bir token adımı için log-olasılık tensörünü [B, S] hesaplar.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # 1. token için sıfır padding ekle (orijinal boyuta eşitlemek için)
    B = logits.shape[0]
    sifir_bas = torch.zeros((B, 1), device=logits.device, dtype=logits.dtype)
    return torch.cat([sifir_bas, per_token_logps], dim=1)
