"""
Day 93: Alt Grup Dilimleme (Data Slicing) ve Adillik / Yanlılık Denetleyicisi
----------------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class DilimDegerlendirmeSonucu:
    dilim_adi: str
    ornek_sayisi: int
    dogruluk: float
    f1_skoru: float
    pozitif_tahmin_orani: float


@dataclass
class AdillikRaporu:
    dilim_sonuclari: Dict[str, DilimDegerlendirmeSonucu]
    maks_dogruluk_farki: float
    demographic_parity_farki: float
    disparate_impact_orani: float
    adillik_esigi_gecti_mi: bool  # 80% Kuralı (Disparate Impact >= 0.80)
    tespit_edilen_uyarilar: List[str]


class YanlilikDenetleyicisi:
    """
    Modeli farklı alt grup dilimleri (slices: ışık, çözünürlük, demografik segment)
    üzerinde test edip Demographic Parity ve Disparate Impact adillik metriklerini hesaplar.
    """

    def __init__(self, adillik_esigi: float = 0.80, maks_fark_esigi: float = 0.15):
        self.adillik_esigi = adillik_esigi
        self.maks_fark_esigi = maks_fark_esigi

    def dilimleri_degerlendir(
        self,
        y_gercek: np.ndarray,
        y_tahmin: np.ndarray,
        dilim_maskeleri: Dict[str, np.ndarray],
    ) -> AdillikRaporu:
        dilim_sonuclari: Dict[str, DilimDegerlendirmeSonucu] = {}
        dogruluklar = []
        pozitif_oranlar = []

        for dilim_adi, maske in dilim_maskeleri.items():
            if np.sum(maske) == 0:
                continue

            dilim_gercek = y_gercek[maske]
            dilim_tahmin = y_tahmin[maske]

            acc = float(np.mean(dilim_gercek == dilim_tahmin))
            dogruluklar.append(acc)

            # Basit F1 (Binary veya Macro vekili)
            tp = np.sum((dilim_gercek == dilim_tahmin) & (dilim_tahmin > 0))
            fp = np.sum((dilim_gercek != dilim_tahmin) & (dilim_tahmin > 0))
            fn = np.sum((dilim_gercek != dilim_tahmin) & (dilim_tahmin == 0))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else acc

            pozitif_oran = float(np.mean(dilim_tahmin > 0))
            pozitif_oranlar.append(pozitif_oran)

            dilim_sonuclari[dilim_adi] = DilimDegerlendirmeSonucu(
                dilim_adi=dilim_adi,
                ornek_sayisi=int(np.sum(maske)),
                dogruluk=acc,
                f1_skoru=float(f1),
                pozitif_tahmin_orani=pozitif_oran,
            )

        # Adillik Metrikleri
        maks_acc_farki = float(max(dogruluklar) - min(dogruluklar)) if dogruluklar else 0.0
        dp_farki = float(max(pozitif_oranlar) - min(pozitif_oranlar)) if pozitif_oranlar else 0.0

        min_p = min(pozitif_oranlar) if pozitif_oranlar else 1.0
        maks_p = max(pozitif_oranlar) if pozitif_oranlar else 1.0
        dir_orani = float(min_p / maks_p) if maks_p > 0 else 1.0

        uyarilar = []
        if dir_orani < self.adillik_esigi:
            uyarilar.append(f"Disparate Impact Oranı (%{dir_orani * 100:.1f}) yasal '%80 kuralı' sınırının altında!")
        if maks_acc_farki > self.maks_fark_esigi:
            uyarilar.append(f"Dilimler arası performans uçurumu çok yüksek (%{maks_acc_farki * 100:.1f} fark)!")

        adil_mi = (dir_orani >= self.adillik_esigi) and (maks_acc_farki <= self.maks_fark_esigi)

        return AdillikRaporu(
            dilim_sonuclari=dilim_sonuclari,
            maks_dogruluk_farki=maks_acc_farki,
            demographic_parity_farki=dp_farki,
            disparate_impact_orani=dir_orani,
            adillik_esigi_gecti_mi=adil_mi,
            tespit_edilen_uyarilar=uyarilar,
        )
