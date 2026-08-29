"""
PPO Matematik Çekirdeği: KL Cezası, GAE ve Kırpılmış Amaç Fonksiyonu Modülü (Day 109).
Token bazlı KL cezası, Generalized Advantage Estimation (GAE-lambda) ve PPO Clipped Loss.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def hesapla_kl_cezali_odul(
    logprobs_actor: torch.Tensor,
    logprobs_ref: torch.Tensor,
    skaler_odul_rm: torch.Tensor,
    kl_beta: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Token bazlı KL cezası ile Reward Model skaler ödülünü birleştirir.
    R_t = -beta * (log pi_theta - log pi_ref) (t < T)
    R_T = -beta * (log pi_theta - log pi_ref) + R_RM (t = T)
    """
    B, T = logprobs_actor.shape

    # Token bazlı KL sapması: log pi_theta - log pi_ref
    kl_divergence = logprobs_actor - logprobs_ref  # [B, T]
    kl_cezasi = -kl_beta * kl_divergence           # [B, T]

    birlesik_odul = kl_cezasi.clone()
    # Son token adımına Reward Model skaler ödülünü ekle
    birlesik_odul[:, -1] += skaler_odul_rm

    return birlesik_odul, kl_divergence.mean()


def hesapla_gae_avantaj(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (GAE-lambda) ve getiri (returns) hesaplar.
    delta_t = R_t + gamma * V_{t+1} - V_t
    A_t = delta_t + gamma * lambda * A_{t+1}
    """
    B, T = rewards.shape
    avantajlar = torch.zeros_like(rewards)
    son_gae_lam = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            sonraki_deger = 0.0
        else:
            sonraki_deger = values[:, t + 1]

        delta = rewards[:, t] + gamma * sonraki_deger - values[:, t]
        son_gae_lam = delta + gamma * lam * son_gae_lam
        avantajlar[:, t] = son_gae_lam

    getiriler = avantajlar + values
    # Avantaj normalizasyonu (Stabilite için)
    avantaj_norm = (avantajlar - avantajlar.mean()) / (avantajlar.std().clamp(min=1e-8))
    return avantaj_norm, getiriler


class PPOClippedLoss(nn.Module):
    """PPO Kırpılmış Politika ve Değer Kayıp Fonksiyonu."""

    def __init__(self, clip_eps: float = 0.2, vf_coef: float = 0.5):
        super().__init__()
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef

    def forward(
        self,
        logprobs_new: torch.Tensor,
        logprobs_old: torch.Tensor,
        avantajlar: torch.Tensor,
        values_pred: torch.Tensor,
        getiriler: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Kayıpları hesaplar.
        Çıktı: (toplam_kayip, politika_kaybi, deger_kaybi, kirpma_orani)
        """
        # Olasılık Oranı: r_t(theta) = exp(log pi_new - log pi_old)
        oran = torch.exp(logprobs_new - logprobs_old)

        # Kırpılmış Amaç (Clipped Surrogate)
        surr1 = oran * avantajlar
        surr2 = torch.clamp(oran, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * avantajlar
        politika_kaybi = -torch.min(surr1, surr2).mean()

        # Değer Ağı Kaybı (MSE)
        deger_kaybi = 0.5 * F.mse_loss(values_pred, getiriler)

        toplam_kayip = politika_kaybi + self.vf_coef * deger_kaybi

        # İzleme için kırpılma oranı (Clip Fraction)
        kirpildi = ((oran < (1.0 - self.clip_eps)) | (oran > (1.0 + self.clip_eps))).float()
        kirpma_orani = float(kirpildi.mean().item())

        return toplam_kayip, politika_kaybi, deger_kaybi, kirpma_orani
