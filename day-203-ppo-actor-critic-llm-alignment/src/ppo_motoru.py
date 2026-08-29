"""
PPO (Proximal Policy Optimization) Actor-Critic LLM Hizalama Motoru (Day 203 - FAZ 11).
RLHF için Actor, Critic, Ödül Modeli, Referans Model ve GAE (Generalized Advantage Estimation).
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    """LLM Politika Ağı (Actor Policy - pi_theta)."""

    def __init__(self, vocab_size: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(self.embedding(x)))
        return self.head(h)


class CriticNetwork(nn.Module):
    """LLM Değer Ağı (Critic / Value Model - V_phi)."""

    def __init__(self, vocab_size: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(self.embedding(x)))
        # [batch, seq_len, 1] -> [batch, seq_len]
        return self.value_head(h).squeeze(-1)


class GAECalculator:
    """
    Generalized Advantage Estimation (GAE - gamma, lambda) Hesaplayıcısı.
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t = sum_{l=0}^{inf} (gamma * lambda)^l delta_{t+l}
    """

    @staticmethod
    def hesapla_avantaj_ve_hedef(
        oduller: torch.Tensor,
        degerler: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        oduller: [batch_size, seq_len]
        degerler: [batch_size, seq_len]
        Dönüş: (avantajlar, hedef_degerler)
        """
        batch_size, seq_len = oduller.shape
        avantajlar = torch.zeros_like(oduller)
        son_gae_lam = 0.0

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                sonraki_deger = 0.0
            else:
                sonraki_deger = degerler[:, t + 1]

            delta = oduller[:, t] + gamma * sonraki_deger - degerler[:, t]
            son_gae_lam = delta + gamma * lam * son_gae_lam
            avantajlar[:, t] = son_gae_lam

        hedef_degerler = avantajlar + degerler
        # Avantaj standardizasyonu (Batch genelinde veya opsiyonel normalizasyon)
        adv_std = avantajlar.std()
        if adv_std > 1e-6 and batch_size > 1:
            avantajlar = (avantajlar - avantajlar.mean()) / (adv_std + 1e-8)

        return avantajlar, hedef_degerler


class PPOTrainer:
    """
    4 Modelli RLHF PPO Eğitmeni:
    1. Actor (pi_theta) - Eğitilen Politika
    2. Critic (V_phi) - Eğitilen Değer Tahmincisi
    3. Reference Policy (pi_ref) - Dondurulmuş Referans Modeli (KL Cezası)
    4. Reward Model (R_psi) - Dondurulmuş Ödül Modeli
    """

    def __init__(
        self,
        actor: Optional[ActorNetwork] = None,
        critic: Optional[CriticNetwork] = None,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.05,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
    ):
        self.actor = actor if actor is not None else ActorNetwork()
        self.critic = critic if critic is not None else CriticNetwork()

        # Dondurulmuş Referans Modeli
        self.ref_actor = ActorNetwork()
        self.ref_actor.load_state_dict(self.actor.state_dict())
        self.ref_actor.eval()

        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def ppo_egitim_adimi(
        self,
        input_ids: torch.Tensor,
        ham_odul_skorlari: torch.Tensor,
    ) -> Dict[str, float]:
        """
        input_ids: [batch_size, seq_len]
        ham_odul_skorlari: [batch_size]
        """
        batch_size, seq_len = input_ids.shape

        # 1. Mevcut ve Referans Logitler
        actor_logits = self.actor(input_ids)
        with torch.no_grad():
            ref_logits = self.ref_actor(input_ids)
            degerler = self.critic(input_ids)

        actor_log_p = F.log_softmax(actor_logits, dim=-1)
        ref_log_p = F.log_softmax(ref_logits, dim=-1)

        # 2. Token Başına KL Divergence Cezası Hesaplama
        # kl_div = log_p(actor) - log_p(ref)
        kl_per_token = actor_log_p.gather(2, input_ids.unsqueeze(-1)).squeeze(-1) - \
                       ref_log_p.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

        # 3. Token Ödül Matrisi: Son tokene ham ödül + her tokene negatif KL
        token_odulleri = -self.kl_coef * kl_per_token.detach()
        token_odulleri[:, -1] += ham_odul_skorlari

        # 4. GAE ile Avantaj ve Hedef Değerler
        avantajlar, hedef_degerler = GAECalculator.hesapla_avantaj_ve_hedef(
            token_odulleri, degerler.detach()
        )

        # 5. PPO Actor Kaybı (Clipped Surrogate Loss)
        ratio = torch.exp(kl_per_token)  # pi / pi_old oranı simülasyonu
        surr1 = ratio * avantajlar
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * avantajlar
        actor_loss = -torch.min(surr1, surr2).mean()

        # Entropi Bonusu (Keşif Teşviki)
        entropy = -(F.softmax(actor_logits, dim=-1) * actor_log_p).sum(dim=-1).mean()
        total_actor_loss = actor_loss - self.entropy_coef * entropy

        # Actor Güncellemesi
        self.opt_actor.zero_grad()
        total_actor_loss.backward()
        self.opt_actor.step()

        # 6. Critic Kaybı (Value MSE Loss)
        guncel_degerler = self.critic(input_ids)
        critic_loss = F.mse_loss(guncel_degerler, hedef_degerler)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "kl_divergence": float(kl_per_token.mean().item()),
            "entropy": float(entropy.item()),
            "ortalama_odul": float(ham_odul_skorlari.mean().item()),
            "ortalama_avantaj": float(avantajlar.mean().item()),
        }
