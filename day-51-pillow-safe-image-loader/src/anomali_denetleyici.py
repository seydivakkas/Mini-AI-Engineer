"""
Görsel Sağlık ve Kalite Denetleyicisi (Image Health & Sanity Inspector).
"""

from typing import Dict, Any
import numpy as np


class GorselSaglikDenetleyicisi:
    """Yüklenen görselin en-boy oranı, renk varyansı ve gürültü seviyesini teftiş eder."""

    @classmethod
    def denetle(cls, dizi_rgb: np.ndarray) -> Dict[str, Any]:
        H, W, C = dizi_rgb.shape
        en_boy_orani = float(round(W / max(H, 1), 3))

        # Renk ve piksel varyansı
        varyans = float(round(np.var(dizi_rgb), 2))
        tek_renk_mi = (varyans < 1.0)

        # Ortalama parlaklık
        ortalama_parlaklik = float(round(np.mean(dizi_rgb), 2))
        karanlik_mi = (ortalama_parlaklik < 10.0)
        asiri_parlak_mi = (ortalama_parlaklik > 245.0)

        # Gradyan varyansı ile basit netlik skoru
        gradyan_y = np.diff(dizi_rgb.astype(float), axis=0)
        gradyan_x = np.diff(dizi_rgb.astype(float), axis=1)
        netlik_skoru = float(round(np.mean(np.abs(gradyan_y)) + np.mean(np.abs(gradyan_x)), 2))

        risk_faktörleri = []
        if en_boy_orani > 6.0 or en_boy_orani < 0.16:
            risk_faktörleri.append("AŞIRI_ÇARPIK_EN_BOY_ORANI")
        if tek_renk_mi:
            risk_faktörleri.append("TEK_DÜZE_RENK_BOŞ_GÖRSEL")
        if karanlik_mi:
            risk_faktörleri.append("AŞIRI_KARANLIK_GÖRSEL")
        if asiri_parlak_mi:
            risk_faktörleri.append("AŞIRI_PARLAK_DOYMUŞ_GÖRSEL")

        return {
            "saglikli_mi": len(risk_faktörleri) == 0,
            "en_boy_orani": en_boy_orani,
            "ortalama_parlaklik": ortalama_parlaklik,
            "varyans": varyans,
            "netlik_skoru": netlik_skoru,
            "risk_faktörleri": risk_faktörleri
        }
