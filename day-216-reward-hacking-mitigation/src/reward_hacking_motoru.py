"""
Reward Hacking ve Goodhart Yasası Önleme Motoru (Day 216 - FAZ 11).
Adaptif KL Denetleyicisi, Tanh Ödül Kırpma (Squashing) ve Topluluk (Ensemble LCB) Mimarisi.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn


class AdaptiveKLController:
    """
    Dinamik Adaptif KL Denetleyicisi:
    KL sapması hedeften uzaklaştığında β katsayısını dinamik günceller.
    """

    def __init__(
        self,
        kl_hedef: float = 0.05,
        beta_baslangic: float = 0.10,
        beta_min: float = 0.01,
        beta_max: float = 1.00,
        degisim_orani: float = 0.10,
    ):
        self.kl_hedef = kl_hedef
        self.beta = beta_baslangic
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.degisim_orani = degisim_orani

    def guncelle(self, olculen_kl: float) -> float:
        """Ölçülen KL sapmasına göre yeni β katsayısını hesaplar."""
        if olculen_kl > self.kl_hedef * 1.5:
            # Model referanstan çok uzaklaştı -> KL cezasını artır
            self.beta = min(self.beta_max, self.beta * (1.0 + self.degisim_orani))
        elif olculen_kl < self.kl_hedef * 0.5:
            # Model referansa aşırı yapışık -> KL cezasını gevşet
            self.beta = max(self.beta_min, self.beta * (1.0 - self.degisim_orani))
        return float(self.beta)


class RewardSquasher:
    """Ödül Patlamalarını Engelleyen Kırpma (Squashing) Motoru."""

    @classmethod
    def tanh_kirp(cls, ham_odul: float, maks_odul: float = 5.0) -> float:
        """R_squashed = R_max * tanh(R_raw / R_max)"""
        return float(maks_odul * math.tanh(ham_odul / max(1e-5, maks_odul)))

    @classmethod
    def sert_kirp(cls, ham_odul: float, min_val: float = -5.0, maks_val: float = 5.0) -> float:
        """Aşırı değerleri [min_val, maks_val] aralığına kenetler."""
        return float(max(min_val, min(maks_val, ham_odul)))


class EnsembleRewardModel:
    """
    K'lı Ödül Modeli Topluluğu ve Muhafazakar Alt Güven Sınırı (LCB):
    R_LCB = μ_R - λ * σ_R
    """

    @classmethod
    def degerlendir(
        cls,
        ham_puanlar: List[float],
        lambda_lcb: float = 1.5,
    ) -> Dict[str, float]:
        """Topluluk modellerinin ortalama, standart sapma ve LCB ödülünü hesaplar."""
        arr = np.array(ham_puanlar, dtype=np.float32)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr)) if len(arr) > 1 else 0.0
        r_lcb = float(mu - lambda_lcb * sigma)

        return {
            "ortalama_odul": mu,
            "standart_sapma": sigma,
            "lcb_odul": r_lcb,
        }


class RewardHackingDetector:
    """Sahte Ödül Sıçraması (Reward Hacking) ve Dilsel Bozulma Tespiti."""

    DALKAVUKLUK_KALIPLARI = [
        "harika bir soru sordunuz efendim",
        "kesinlikle haklısınız",
        "siz mükemmel bir uzmansınız",
        "tamamen katılıyorum",
    ]

    @classmethod
    def denetle(
        cls,
        metin: str,
        ham_odul: float,
        perplexity: float,
    ) -> Dict[str, Any]:
        """Ödül istismarı ve çöküş belirtilerini analiz eder."""
        kucuk = metin.lower()
        dalkavukluk_var_mi = any(k in kucuk for k in cls.DALKAVUKLUK_KALIPLARI)

        # Anormal yüksek ödül + Yüksek Perplexity = Kesin Hacking
        hacking_suphesi = (ham_odul > 6.0 and perplexity > 50.0) or (dalkavukluk_var_mi and ham_odul > 4.0)

        return {
            "dalkavukluk_var_mi": dalkavukluk_var_mi,
            "perplexity": perplexity,
            "ham_odul": ham_odul,
            "hacking_suphesi": hacking_suphesi,
        }


class RobustRLTrainer:
    """İstismara Karşı Sağlamlaştırılmış RL Eğitim Motoru."""

    @classmethod
    def guvenli_odul_adimi(
        cls,
        model_yaniti: str,
        topluluk_puanlari: List[float],
        olculen_kl: float,
        kl_controller: AdaptiveKLController,
        perplexity: float = 12.5,
    ) -> Dict[str, Any]:
        """Ensemble LCB + Tanh Squashing + Adaptif KL ile güvenli ödül üretir."""
        # 1. Ensemble LCB
        ensemble = EnsembleRewardModel.degerlendir(topluluk_puanlari)

        # 2. Tanh Squashing
        kirpilmis_odul = RewardSquasher.tanh_kirp(ensemble["lcb_odul"])

        # 3. Adaptif KL Cezası
        beta = kl_controller.guncelle(olculen_kl)
        kl_cezasi = beta * olculen_kl
        nihai_odul = kirpilmis_odul - kl_cezasi

        # 4. Hacking Denetimi
        hacking_raporu = RewardHackingDetector.denetle(model_yaniti, ensemble["ortalama_odul"], perplexity)

        return {
            "ham_ortalama_odul": ensemble["ortalama_odul"],
            "lcb_odul": ensemble["lcb_odul"],
            "kirpilmis_odul": kirpilmis_odul,
            "beta_kl": beta,
            "kl_cezasi": kl_cezasi,
            "nihai_saglam_odul": float(nihai_odul),
            "hacking_raporu": hacking_raporu,
        }
