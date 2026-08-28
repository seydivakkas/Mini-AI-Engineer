"""
Enerji Tabanlı Dağılım Dışı (OOD) Tespiti Modülü
-----------------------------------------------
Liu et al. (NeurIPS 2020) "Energy-based Out-of-distribution Detection" formülasyonu.
Softmax olasılıklarının normalizasyon bozulmalarından arındırılmış, serbest enerji (Free Energy)
skoru ile Dağılım İçi (ID) ve Dağılım Dışı (OOD) örnekleri ayrıştıran tespit motoru.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnerjiTabanliOODDedektoru:
    """
    Logit vektörleri üzerinden Enerji Skoru ve Softmax MSP hesaplayan OOD dedektörü.
    E(x; T) = -T * log( sum_k exp(z_k / T) )
    Skor: -E(x; T) = T * logsumexp(z / T)
    """
    def __init__(self, sicaklik: float = 1.0, esik_degeri: Optional[float] = None):
        assert sicaklik > 0.0, "Sıcaklık (T) pozitif olmalıdır!"
        self.sicaklik = sicaklik
        self.esik_degeri = esik_degeri

    @classmethod
    def enerji_skoru_hesapla(cls, logitler: torch.Tensor, sicaklik: float = 1.0) -> torch.Tensor:
        """
        Logit tensörü üzerinden serbest enerji skorunu (-E) hesaplar.
        Girdi: [N, K]
        Çıktı: [N] (Daha yüksek skor = Dağılım İçi / ID, Daha düşük skor = OOD)
        """
        # T * logsumexp(z / T)
        olcekli_logitler = logitler / sicaklik
        return sicaklik * torch.logsumexp(olcekli_logitler, dim=-1)

    @classmethod
    def msp_skoru_hesapla(cls, logitler: torch.Tensor) -> torch.Tensor:
        """
        Maksimum Softmax Olasılığı (MSP) temel yaklaşımı.
        Girdi: [N, K]
        Çıktı: [N]
        """
        olasiliklar = F.softmax(logitler, dim=-1)
        max_olasilik, _ = torch.max(olasiliklar, dim=-1)
        return max_olasilik

    def esik_belirle(self, id_logitler: torch.Tensor, hedef_tpr: float = 0.95) -> float:
        """
        Dağılım İçi (ID) doğrulama örneklerinin %95'ini (hedef_tpr) kabul edecek eşik değerini hesaplar.
        """
        skorlar = self.enerji_skoru_hesapla(id_logitler, self.sicaklik).detach().cpu().numpy()
        # %95 TPR için en düşük %5'lik dilim eşik seçilir
        yuzdelik = (1.0 - hedef_tpr) * 100.0
        self.esik_degeri = float(np.percentile(skorlar, yuzdelik))
        return self.esik_degeri

    def tahmin_et(self, logitler: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Logitleri ID (True) veya OOD (False) olarak sınıflandırır.
        Çıktı: (id_maskesi, enerji_skorlari)
        """
        assert self.esik_degeri is not None, "Önce esik_belirle() çağrılmalı veya eşik atanmalıdır!"
        skorlar = self.enerji_skoru_hesapla(logitler, self.sicaklik)
        id_maskesi = skorlar >= self.esik_degeri
        return id_maskesi, skorlar
