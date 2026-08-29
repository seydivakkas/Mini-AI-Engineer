"""
PPO LLM Hizalama Başarım ve Metrik Profilleyici Modülü (Day 203 - FAZ 11).
Actor/Critic Kayıpları, KL Divergence Bütçesi, GAE Stabilitesi ve İnsan Tercih Skoru.
"""

from typing import Dict, Any, List
import torch
import numpy as np
from .ppo_motoru import PPOTrainer


class PPOAkisProfilleyici:
    """PPO RLHF LLM Hizalama Profilleyicisi."""

    @classmethod
    def egitim_akisini_profili_cikar(cls, adim_sayisi: int = 10) -> Dict[str, Any]:
        """10 Adımlık PPO Actor-Critic LLM Hizalama Akışı."""
        trainer = PPOTrainer()

        adimlar = []
        actor_kayiplari = []
        critic_kayiplari = []
        odul_skorlari = []
        kl_degerleri = []

        batch_size = 4
        seq_len = 16

        for adim in range(1, adim_sayisi + 1):
            input_ids = torch.randint(0, 128, (batch_size, seq_len))
            # Eğitim ilerledikçe artan insan hizalama ödül skoru (-1.0 -> +2.8)
            temel_odul = -0.8 + (adim * 0.36) + np.random.uniform(-0.1, 0.1)
            ham_oduller = torch.full((batch_size,), temel_odul, dtype=torch.float32)

            sonuc = trainer.ppo_egitim_adimi(input_ids, ham_oduller)

            adimlar.append(adim)
            actor_kayiplari.append(sonuc["actor_loss"])
            critic_kayiplari.append(sonuc["critic_loss"])
            odul_skorlari.append(temel_odul)
            kl_degerleri.append(abs(sonuc["kl_divergence"]))

        return {
            "adimlar": adimlar,
            "actor_kayiplari": actor_kayiplari,
            "critic_kayiplari": critic_kayiplari,
            "odul_skorlari": odul_skorlari,
            "kl_degerleri": kl_degerleri,
            "son_odul": odul_skorlari[-1],
            "son_critic_loss": critic_kayiplari[-1],
            "model_kumesi": {
                "actor_model": "Aktif Eğitilen Politika (Policy)",
                "critic_model": "Aktif Eğitilen Değer Tahmincisi (Value)",
                "reference_model": "Dondurulmuş Temel SFT Modeli (KL Kıyası)",
                "reward_model": "Dondurulmuş İnsan Tercih Modeli (RM)",
            },
        }
