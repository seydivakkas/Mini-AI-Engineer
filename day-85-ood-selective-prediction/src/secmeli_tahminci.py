"""
Seçici Tahmin ve Çekimserlik (Selective Prediction & Abstention) Motoru
---------------------------------------------------------------------
Geifman & El-Yaniv (2017) "Selective Classification for Deep Neural Networks" yaklaşımı.
Modelin belirsiz olduğu veya OOD şüphesi taşıyan örneklerde tahmin yapmayı reddederek
insan uzmana devretmesini sağlayan, Kapsam (Coverage) vs Risk (Hata) optimizasyon motoru.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from .enerji_ood import EnerjiTabanliOODDedektoru


class SecmeliTahminci:
    """
    Güven/Enerji eşiğine göre tahminleri kabul eden veya çekimser kalan (Abstention) karar motoru.
    """
    def __init__(self, esik_degeri: float, skor_tipi: str = "enerji", sicaklik: float = 1.0):
        self.esik_degeri = esik_degeri
        self.skor_tipi = skor_tipi
        self.sicaklik = sicaklik

    def skor_hesapla(self, logitler: torch.Tensor) -> torch.Tensor:
        if self.skor_tipi == "enerji":
            return EnerjiTabanliOODDedektoru.enerji_skoru_hesapla(logitler, self.sicaklik)
        elif self.skor_tipi == "msp":
            return EnerjiTabanliOODDedektoru.msp_skoru_hesapla(logitler)
        else:
            raise ValueError(f"Bilinmeyen skor tipi: {self.skor_tipi}")

    def secmeli_tahmin_yap(self, logitler: torch.Tensor) -> Dict[str, Any]:
        """
        Logitleri işler; eşiği geçenleri tahmin eder, geçemeyenleri 'REDDEDILDI_UZMANA_DEVRET' olarak etiketler.
        """
        skorlar = self.skor_hesapla(logitler)
        kabul_maskesi = skorlar >= self.esik_degeri
        ham_tahminler = torch.argmax(logitler, dim=-1)

        kabul_edilen_indeksler = torch.where(kabul_maskesi)[0]
        reddedilen_indeksler = torch.where(~kabul_maskesi)[0]

        return {
            "kabul_maskesi": kabul_maskesi,
            "kabul_indeksleri": kabul_edilen_indeksler,
            "red_indeksleri": reddedilen_indeksler,
            "tahminler": ham_tahminler,
            "skorlar": skorlar,
            "kapsam_orani": (kabul_maskesi.sum().item() / max(1, logitler.size(0))) * 100.0
        }

    @classmethod
    def kapsam_risk_egrisi(
        cls,
        logitler: torch.Tensor,
        etiketler: torch.Tensor,
        skor_tipi: str = "enerji",
        sicaklik: float = 1.0,
        adim_sayisi: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Farklı eşik değerleri altında Kapsam (Coverage) ve Risk (Hata Oranı) eğrisini hesaplar.
        """
        if skor_tipi == "enerji":
            skorlar = EnerjiTabanliOODDedektoru.enerji_skoru_hesapla(logitler, sicaklik).cpu().numpy()
        else:
            skorlar = EnerjiTabanliOODDedektoru.msp_skoru_hesapla(logitler).cpu().numpy()

        tahminler = torch.argmax(logitler, dim=-1).cpu().numpy()
        y_true = etiketler.cpu().numpy()
        n_toplam = len(y_true)

        min_skor, max_skor = float(skorlar.min()), float(skorlar.max())
        esikler = np.linspace(min_skor, max_skor, adim_sayisi)

        kapsamlar = []
        riskler = []
        dogruluklar = []

        for esik in esikler:
            kabul_mask = skorlar >= esik
            n_kabul = int(kabul_mask.sum())

            if n_kabul > 0:
                hatalar = (tahminler[kabul_mask] != y_true[kabul_mask]).sum()
                risk = (hatalar / n_kabul) * 100.0
                dogruluk = 100.0 - risk
                kapsam = (n_kabul / n_toplam) * 100.0
            else:
                risk = 0.0
                dogruluk = 100.0
                kapsam = 0.0

            kapsamlar.append(kapsam)
            riskler.append(risk)
            dogruluklar.append(dogruluk)

        return {
            "esikler": esikler,
            "kapsam": np.array(kapsamlar),
            "risk": np.array(riskler),
            "dogruluk": np.array(dogruluklar)
        }
