"""
DreamBooth Özne Öğretimi ve Sınıf Koruma Kaybı Modülü (Day 176 - FAZ 9).
Özel belirteç bağlama ('sks dog') ve dil kayması (Language Drift) önleyici Prior Preservation Loss.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DreamBoothEgitici:
    """DreamBooth ve LoRA ile Özel Nesne/Stil Öğretim Yöneticisi."""

    def __init__(self, prior_loss_weight: float = 1.0):
        self.prior_loss_weight = prior_loss_weight

    def toplam_kayip_hesapla(
        self,
        tahmin_ozne: torch.Tensor,
        hedef_ozne_gurultusu: torch.Tensor,
        tahmin_sinif: torch.Tensor,
        hedef_sinif_gurultusu: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        L_total = L_instance + lambda * L_prior
        L_instance : 'A photo of [sks] dog' için gürültü kestirim MSE kaybı
        L_prior    : 'A photo of a dog' genel sınıfı için koruma kaybı (Dil kaymasını engeller)
        """
        loss_instance = F.mse_loss(tahmin_ozne, hedef_ozne_gurultusu)
        loss_prior = F.mse_loss(tahmin_sinif, hedef_sinif_gurultusu)

        loss_total = loss_instance + self.prior_loss_weight * loss_prior
        return loss_total, loss_instance, loss_prior

    @classmethod
    def ornek_lora_raporu_getir(cls) -> Dict[str, Any]:
        """Farklı rank değerleri ve DreamBooth metrikleri."""
        return {
            "hedef_kavram": "Özel Karakter / Maskot ('sks robot')",
            "rank_deneyleri": [
                {"r": 4, "dosya_mb": 9.5, "param_yuzde": 0.05, "sadakat": 0.88, "durum": "Ultra Hafif (Hızlı İnce Ayar)"},
                {"r": 8, "dosya_mb": 18.2, "param_yuzde": 0.10, "sadakat": 0.94, "durum": "Dengeli (Endüstri Standardı)"},
                {"r": 16, "dosya_mb": 36.4, "param_yuzde": 0.20, "sadakat": 0.98, "durum": "Yüksek Detay & Sanat Stili"},
                {"r": 32, "dosya_mb": 72.8, "param_yuzde": 0.40, "sadakat": 0.99, "durum": "Karmaşık Nesne / Yüz"},
                {"r": 64, "dosya_mb": 145.6, "param_yuzde": 0.80, "sadakat": 0.99, "durum": "Aşırı Parametre (Aşırı Öğrenme Riski)"},
            ],
            "ortalama_kayip": 0.0215,
            "sinif_koruma_skoru": 0.97,
            "dosya_boyut_kazanci": "~100x (4.2 GB checkpoint -> 36 MB LoRA)",
        }
