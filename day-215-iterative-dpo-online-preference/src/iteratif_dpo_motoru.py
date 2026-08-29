"""
İteratif ve Çevrimiçi DPO (Iterative Online DPO) Motoru (Day 215 - FAZ 11).
Dinamik Tercih Havuzu, Canlı Örnekleme ve Referans Model Kaydırma Mimarisi.
"""

from typing import Dict, Any, List, Optional, Tuple
import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlinePreferenceBuffer:
    """Kayan Pencereli Dinamik Tercih Havuzu (Replay Buffer)."""

    def __init__(self, kapasite: int = 1000):
        self.kapasite = kapasite
        self.havuz: List[Dict[str, Any]] = []

    def ekle(self, prompt: str, chosen: str, rejected: str, tur_no: int):
        """Yeni üretilen tercih çiftini havuza ekler."""
        if len(self.havuz) >= self.kapasite:
            self.havuz.pop(0)  # Eski veriyi sil (Kayan pencere)
        self.havuz.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "tur_no": tur_no,
            }
        )

    def orneklem_al(self, batch_boyutu: int = 4) -> List[Dict[str, Any]]:
        """Eğitim için rastgele mini-batch çeker."""
        if not self.havuz:
            return []
        k = min(batch_boyutu, len(self.havuz))
        return random.sample(self.havuz, k)


class OnlineRolloutSampler:
    """Mevcut Politikadan Canlı Yanıt Üretme ve Hakemleme Motoru."""

    @classmethod
    def cift_yanit_uret_ve_etiketle(
        cls,
        prompt: str,
        tur_no: int,
    ) -> Tuple[str, str]:
        """Modelden iki aday üretip kazananı (chosen) ve kaybedeni (rejected) belirler."""
        # Tur geliştikçe modelin ürettiği yanıt kalitesi artar
        if tur_no == 1:
            yanit_a = f"<think>Kısa düşünce</think> Sonuç: A ({prompt})"
            yanit_b = f"Hatalı ve temelsiz cevap ({prompt})"
            return yanit_a, yanit_b
        elif tur_no == 2:
            yanit_a = f"<think>Derin düşünce ve ara kontrol</think> Kesin Sonuç ({prompt})"
            yanit_b = f"<think>Kısa düşünce</think> Basit Sonuç ({prompt})"
            return yanit_a, yanit_b
        else:
            yanit_a = f"<think>Kusursuz akıl yürütme, kendi kendini doğrulama</think> Pareto-Optimal Çözüm ({prompt})"
            yanit_b = f"<think>Eski turdan kalma eksik düşünce</think> Eksik Çözüm ({prompt})"
            return yanit_a, yanit_b


class ReferencePolicyUpdater:
    """Referans Model Ağırlıklarını Güncelleyen Yönetici (Ref Policy Swapper)."""

    @classmethod
    def referansi_guncelle(cls, mevcut_politika: Dict[str, float]) -> Dict[str, float]:
        """Mevcut politikanın ağırlıklarını yeni referans olarak kopyalar."""
        return copy.deepcopy(mevcut_politika)


class IterativeDPOTrainer:
    """Çok Turlu İteratif ve Çevrimiçi DPO Eğiticisi."""

    @classmethod
    def online_dpo_kaybi(
        cls,
        logp_pi_w: torch.Tensor,
        logp_ref_w: torch.Tensor,
        logp_pi_l: torch.Tensor,
        logp_ref_l: torch.Tensor,
        beta: float = 0.1,
    ) -> Tuple[torch.Tensor, float]:
        """
        Online DPO Kayıp Fonksiyonu ve Örtük Ödül Farkı:
        Δr = β * log(π/π_ref)(y_w) - β * log(π/π_ref)(y_l)
        L = -log σ(Δr)
        """
        r_w = beta * (logp_pi_w - logp_ref_w)
        r_l = beta * (logp_pi_l - logp_ref_l)
        delta_r = r_w - r_l
        kayip = -F.logsigmoid(delta_r)
        return kayip, float(delta_r.item())

    @classmethod
    def iteratif_tur_yurut(
        cls,
        prompt: str,
        tur_no: int,
        buffer: OnlinePreferenceBuffer,
    ) -> Dict[str, Any]:
        """Tek bir iteratif tur gerçekleştirir."""
        chosen, rejected = OnlineRolloutSampler.cift_yanit_uret_ve_etiketle(prompt, tur_no)
        buffer.ekle(prompt, chosen, rejected, tur_no)

        # Simüle tensör log-olasılıkları
        logp_pi_w = torch.tensor(-1.2 + (tur_no * 0.2))
        logp_ref_w = torch.tensor(-1.8)
        logp_pi_l = torch.tensor(-2.6 - (tur_no * 0.1))
        logp_ref_l = torch.tensor(-1.9)

        kayip, delta_r = cls.online_dpo_kaybi(logp_pi_w, logp_ref_w, logp_pi_l, logp_ref_l)

        return {
            "tur_no": tur_no,
            "chosen": chosen,
            "rejected": rejected,
            "kayip": float(kayip.item()),
            "ortuk_odul_marjini": delta_r,
            "buffer_boyutu": len(buffer.havuz),
        }
