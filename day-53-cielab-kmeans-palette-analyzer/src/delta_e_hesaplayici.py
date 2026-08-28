"""
Delta-E (Delta-E 76 ve CIEDE2000) Algısal Renk Farkı ve Endüstriyel Tolerans Hesaplayıcısı.
"""

from typing import Dict, Any, Union
import numpy as np
from skimage import color


class DeltaEHesaplayici:
    """CIELAB uzayında iki renk arasındaki algısal farkı ve endüstriyel kalite toleransını belirler."""

    @classmethod
    def delta_e_76(
        cls,
        lab1: Union[np.ndarray, list],
        lab2: Union[np.ndarray, list]
    ) -> float:
        """CIE 1976 Öklid mesafesi formülü ile Delta-E 76 hesaplar."""
        v1 = np.asarray(lab1, dtype=np.float64)
        v2 = np.asarray(lab2, dtype=np.float64)
        fark = np.sqrt(np.sum((v1 - v2)**2, axis=-1))
        return float(round(float(np.mean(fark)), 3))

    @classmethod
    def delta_e_2000(
        cls,
        lab1: Union[np.ndarray, list],
        lab2: Union[np.ndarray, list],
        k_L: float = 1.0,
        k_C: float = 1.0,
        k_H: float = 1.0
    ) -> float:
        """ISO/CIE 116-2019 standardı olan CIEDE2000 algoritması ile hassas algısal renk farkını hesaplar."""
        v1 = np.asarray(lab1, dtype=np.float64)
        v2 = np.asarray(lab2, dtype=np.float64)

        if v1.ndim == 1:
            v1 = v1.reshape(1, 1, 3)
        elif v1.ndim == 2:
            v1 = v1.reshape(1, -1, 3)

        if v2.ndim == 1:
            v2 = v2.reshape(1, 1, 3)
        elif v2.ndim == 2:
            v2 = v2.reshape(1, -1, 3)

        de00_matris = color.deltaE_ciede2000(v1, v2, kL=k_L, kC=k_C, kH=k_H)
        return float(round(float(np.mean(de00_matris)), 3))

    @classmethod
    def tolerans_degerlendir(cls, delta_e_00: float) -> Dict[str, Any]:
        """CIEDE2000 değerini uluslararası endüstriyel kalite ve algı standartlarına göre sınıflandırır."""
        if delta_e_00 < 1.0:
            return {
                "kod": "MUKEMMEL_ESLESME",
                "seviye": "PASS",
                "deger": delta_e_00,
                "aciklama": "İnsan gözüyle ayırt edilemez fark (Imperceptible)",
                "renk": "#2ecc71"
            }
        elif delta_e_00 < 2.0:
            return {
                "kod": "TOLERANS_DAHILINDE",
                "seviye": "PASS",
                "deger": delta_e_00,
                "aciklama": "Yalnızca uzman gözle yakından bakıldığında fark edilir (Pass)",
                "renk": "#27ae60"
            }
        elif delta_e_00 < 5.0:
            return {
                "kod": "KABUL_SINIRINDA",
                "seviye": "WARNING",
                "deger": delta_e_00,
                "aciklama": "Standart gözlemci tarafından fark edilebilir renk sapması (Warning)",
                "renk": "#f39c12"
            }
        else:
            return {
                "kod": "KRITIK_RED",
                "seviye": "REJECT",
                "deger": delta_e_00,
                "aciklama": "Kabul edilemez belirgin renk uyumsuzluğu / boyama hatası (Reject)",
                "renk": "#e74c3c"
            }
