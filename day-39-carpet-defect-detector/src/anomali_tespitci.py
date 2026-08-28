"""
Halı Anomali ve Kalıntı Haritası Tespitçisi (Anomaly & Residual Map Detector).
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image


class AnomaliTespitci:
    """
    Halı yüzeyindeki dokuma gürültüsünü filtreleyip yerel ve referans bazlı
    kalıntı (residual) anomali haritası ve ikili (binary) maske üretir.
    """

    def __init__(self, sigma_filtre: float = 3.0, esik_carpani: float = 2.8):
        self.sigma = sigma_filtre
        self.esik_carpani = esik_carpani

    def anomali_haritasi_cikar(
        self,
        test_gorseli: Image.Image,
        referans_gorseli: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Giriş görseli üzerinden pikselsel anomali skoru ve ham ikili maske üretir.
        """
        img_arr = np.array(test_gorseli.convert("RGB"), dtype=np.float64)
        gri_test = np.array(test_gorseli.convert("L"), dtype=np.float64)

        if referans_gorseli is not None:
            gri_ref = np.array(referans_gorseli.convert("L"), dtype=np.float64)
            # Referans farkı
            kalinti = np.abs(gri_test - gri_ref)
        else:
            # Referans yoksa yerel arka plan tahmininden çıkar
            arka_plan = gaussian_filter(gri_test, sigma=self.sigma * 3.0)
            kalinti = np.abs(gri_test - arka_plan)

        # Kalıntı haritasını yumuşat (yüksek frekanslı dokuma gürültüsünü bastır)
        kalinti_duzgun = gaussian_filter(kalinti, sigma=self.sigma)

        # Normalize anomali skoru [0, 1]
        max_val = np.max(kalinti_duzgun) + 1e-12
        anomali_skoru = kalinti_duzgun / max_val

        # İstatistiksel Adaptif Eşikleme: mu + k * sigma
        mu = np.mean(kalinti_duzgun)
        std = np.std(kalinti_duzgun)
        esik_degeri = mu + (self.esik_carpani * std)

        ham_maske = (kalinti_duzgun > esik_degeri).astype(np.uint8)

        return {
            "gri_gorsel": gri_test,
            "kalinti_haritasi": kalinti_duzgun,
            "anomali_skor_haritasi": anomali_skoru,
            "esik_degeri": float(round(esik_degeri, 3)),
            "ham_maske": ham_maske
        }
