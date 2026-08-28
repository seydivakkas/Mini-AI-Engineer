"""
İstatistiksel Dağılım ve Kayma Ölçüm Motoru (KS-Test, Wasserstein Distance & PSI).
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats


class KSVeWassersteinHesaplayici:
    """İki dağılım arasındaki istatistiksel sapmayı KS, Wasserstein ve PSI metrikleriyle ölçer."""

    @classmethod
    def ampirik_cdf_hesapla(
        cls,
        dizi: np.ndarray,
        izgara: np.ndarray
    ) -> np.ndarray:
        """Belirtilen ortak ızgara noktaları üzerinde ampirik CDF değerini üretir."""
        sirali = np.sort(dizi)
        return np.searchsorted(sirali, izgara, side="right") / float(len(dizi))

    @classmethod
    def psi_hesapla(
        cls,
        referans: np.ndarray,
        uretim: np.ndarray,
        kutucuk_sayisi: int = 10,
        eps: float = 1e-4
    ) -> float:
        """Nüfus Kararlılık İndeksini (Population Stability Index - PSI) hesaplar."""
        # Referans dağılım üzerinden yüzdelik dilim (quantile) sınırları belirle
        yuzdelikler = np.linspace(0, 100, kutucuk_sayisi + 1)
        sinirlar = np.percentile(referans, yuzdelikler)
        sinirlar[0] = -np.inf
        sinirlar[-1] = np.inf

        ref_frekans, _ = np.histogram(referans, bins=sinirlar)
        prod_frekans, _ = np.histogram(uretim, bins=sinirlar)

        ref_oran = ref_frekans / float(len(referans)) + eps
        prod_oran = prod_frekans / float(len(uretim)) + eps

        # Normalizasyon
        ref_oran = ref_oran / ref_oran.sum()
        prod_oran = prod_oran / prod_oran.sum()

        psi_degeri = np.sum((prod_oran - ref_oran) * np.log(prod_oran / ref_oran))
        return float(max(0.0, psi_degeri))

    @classmethod
    def olc(
        cls,
        referans: np.ndarray,
        uretim: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """İki dağılım arasındaki KS testi, Wasserstein mesafesi ve PSI skorlarını hesaplar."""
        ref_temiz = np.asarray(referans, dtype=np.float64).flatten()
        prod_temiz = np.asarray(uretim, dtype=np.float64).flatten()

        # 1. 2-Sample Kolmogorov-Smirnov Testi
        ks_res = stats.ks_2samp(ref_temiz, prod_temiz)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # 2. 1D Wasserstein Mesafesi (Earth Mover's Distance)
        w1_mesafe = float(stats.wasserstein_distance(ref_temiz, prod_temiz))

        # 3. Population Stability Index (PSI)
        psi_skor = cls.psi_hesapla(ref_temiz, prod_temiz)

        # 4. Görselleştirme İçin Ortak Izgara ve Ampirik CDF'ler
        min_x = min(np.min(ref_temiz), np.min(prod_temiz))
        max_x = max(np.max(ref_temiz), np.max(prod_temiz))
        izgara = np.linspace(min_x, max_x, 300)

        cdf_ref = cls.ampirik_cdf_hesapla(ref_temiz, izgara)
        cdf_prod = cls.ampirik_cdf_hesapla(prod_temiz, izgara)

        farklar = np.abs(cdf_ref - cdf_prod)
        maks_fark_idx = int(np.argmax(farklar))
        maks_fark_x = float(izgara[maks_fark_idx])

        # 5. Drift Kararı
        # p < alpha ve PSI >= 0.1 ise belirgin drift vardır
        if p_val < 0.01 or psi_skor >= 0.20:
            kayma_derecesi = "KRITIK_KAYMA_ALARM"
            drift_var = True
        elif p_val < alpha or psi_skor >= 0.10:
            kayma_derecesi = "ORTA_KAYMA_UYARI"
            drift_var = True
        else:
            kayma_derecesi = "KAYMA_YOK_STABIL"
            drift_var = False

        return {
            "drift_tespit_edildi": drift_var,
            "kayma_derecesi": kayma_derecesi,
            "ks_istatistigi": float(round(ks_stat, 4)),
            "p_degeri": float(round(p_val, 6)),
            "wasserstein_mesafesi": float(round(w1_mesafe, 4)),
            "psi_skoru": float(round(psi_skor, 4)),
            "alpha_esigi": alpha,
            "orneklem_boyutlari": {"referans": len(ref_temiz), "uretim": len(prod_temiz)},
            "grafik_verisi": {
                "izgara": izgara,
                "cdf_ref": cdf_ref,
                "cdf_prod": cdf_prod,
                "maks_fark_x": maks_fark_x,
                "ks_stat": ks_stat
            }
        }
