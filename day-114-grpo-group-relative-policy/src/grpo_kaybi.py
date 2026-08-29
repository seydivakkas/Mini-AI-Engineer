"""
Group Relative Policy Optimization (GRPO) Kayıp Fonksiyonu Modülü (Day 114).
DeepSeek-R1 / DeepSeekMath tarzı Critic-Free (Değer Ağsız) grup göreli avantaj normalizasyonu,
kırpılmış taşıyıcı hedef (clipped surrogate) ve token bazlı KL cezası.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def grup_goreli_avantaj_hesapla(
    oduller: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Grup içindeki ödüllerden (Z-Score) göreli avantajları [G] veya [B, G] hesaplar.
    A_i = (r_i - mean(r)) / (std(r) + eps)
    """
    if oduller.dim() == 1:
        # [G]
        ortalama = oduller.mean()
        std = oduller.std(unbiased=False)
        return (oduller - ortalama) / (std + eps)
    else:
        # [B, G]
        ortalama = oduller.mean(dim=-1, keepdim=True)
        std = oduller.std(dim=-1, keepdim=True, unbiased=False)
        return (oduller - ortalama) / (std + eps)


class GRPOLoss(nn.Module):
    """
    GRPO Kayıp Fonksiyonu (DeepSeek-R1):
    L_GRPO = 1/G * sum_{i=1}^G [ L_clip(pi_theta, pi_old, A_i) + beta * D_KL(pi_theta || pi_ref) ]
    """

    def __init__(self, clip_eps: float = 0.2, beta_kl: float = 0.04):
        super().__init__()
        self.clip_eps = clip_eps
        self.beta_kl = beta_kl

    def forward(
        self,
        logp_theta: torch.Tensor,       # [G, S] (veya [B*G, S])
        logp_old: torch.Tensor,         # [G, S]
        logp_ref: torch.Tensor,         # [G, S]
        oduller: torch.Tensor,          # [G] (veya [B, G])
        token_mask: torch.Tensor,       # [G, S] (Yanıt tokenları 1, prompt/pad 0)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. Grup İçi Göreli Avantaj Hesabı: [G]
        avantajlar = grup_goreli_avantaj_hesapla(oduller)  # [G]
        if avantajlar.dim() == 1:
            # Token boyutuna genişlet: [G, 1] -> broadcast to [G, S]
            avantajlar_genis = avantajlar.unsqueeze(-1)
        else:
            avantajlar_genis = avantajlar.view(-1, 1)

        # 2. Olasılık Oranı: r_t(theta) = exp(logp_theta - logp_old)
        oran = torch.exp(logp_theta - logp_old)  # [G, S]

        # 3. Kırpılmış Taşıyıcı Hedef (Clipped Surrogate)
        surr1 = oran * avantajlar_genis
        surr2 = torch.clamp(oran, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * avantajlar_genis
        politika_hedefi = torch.min(surr1, surr2)  # [G, S]

        token_sayilari = token_mask.sum(dim=-1).clamp(min=1.0)  # [G]
        politika_kaybi = -(politika_hedefi * token_mask).sum(dim=-1) / token_sayilari  # [G]

        # 4. Token Bazlı KL Sapma Cezası (Schulman Yansız Tahmincisi)
        # D_KL = exp(logp_ref - logp_theta) - (logp_ref - logp_theta) - 1
        log_oran_ref = logp_ref - logp_theta
        kl_per_token = torch.exp(log_oran_ref) - log_oran_ref - 1.0  # [G, S]
        kl_kaybi = (kl_per_token * token_mask).sum(dim=-1) / token_sayilari  # [G]

        # 5. Toplam GRPO Kaybı
        toplam_grup_kaybi = politika_kaybi + self.beta_kl * kl_kaybi  # [G]
        toplam_kayip = toplam_grup_kaybi.mean()

        # Kırpılma Oranı (Clip Fraction)
        kirpilma = ((oran < 1.0 - self.clip_eps) | (oran > 1.0 + self.clip_eps)).float()
        kirpilma_orani = (kirpilma * token_mask).sum() / token_mask.sum().clamp(min=1.0)

        metrikler = {
            "toplam_kayip": toplam_kayip.detach(),
            "politika_kaybi": politika_kaybi.mean().detach(),
            "kl_kaybi": kl_kaybi.mean().detach(),
            "ortalama_odul": oduller.mean().detach(),
            "std_odul": oduller.std(unbiased=False).detach(),
            "kirpilma_orani": kirpilma_orani.detach(),
        }

        return toplam_kayip, metrikler


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
