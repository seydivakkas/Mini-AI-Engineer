"""
Renk Zekası ve İplik Katalog Eşleme Motoru (CIELAB & Delta-E 2000).
"""

from typing import Dict, Any, List
import numpy as np
from PIL import Image


class RenkZekasiMotoru:
    """CIELAB uzayında K-Means ile iplik sarfiyatı çıkarır ve Delta-E 2000 ile eşler."""

    KATALOG = [
        {"kod": "YARN-101", "ad": "Kraliyet Bordosu", "rgb": [138, 28, 48]},
        {"kod": "YARN-102", "ad": "Derin Gece Mavisi", "rgb": [24, 43, 73]},
        {"kod": "YARN-103", "ad": "Klasik Krem Vizon", "rgb": [228, 217, 198]},
        {"kod": "YARN-104", "ad": "Anadolu Hardal Sarısı", "rgb": [204, 154, 45]},
        {"kod": "YARN-105", "ad": "Osmanlı Zümrüt Yeşili", "rgb": [32, 98, 65]},
        {"kod": "YARN-106", "ad": "Antrasit Kömür Grisi", "rgb": [48, 52, 58]}
    ]

    @classmethod
    def _rgb_to_lab(cls, rgb_arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(rgb_arr, dtype=np.float64) / 255.0
        # Gamma
        mask = arr > 0.04045
        lin = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
        # D65 XYZ
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        xyz = np.dot(lin.reshape(-1, 3), M.T)
        xyz[:, 0] /= 0.95047
        xyz[:, 2] /= 1.08883

        delta = 6.0 / 29.0
        f = np.where(xyz > (delta ** 3), xyz ** (1.0 / 3.0), (xyz / (3.0 * delta ** 2)) + (4.0 / 29.0))
        L = 116.0 * f[:, 1] - 16.0
        a = 500.0 * (f[:, 0] - f[:, 1])
        b = 200.0 * (f[:, 1] - f[:, 2])
        return np.stack([L, a, b], axis=-1)

    @classmethod
    def _delta_e_2000(cls, lab1: List[float], lab2: List[float]) -> float:
        L1, a1, b1 = lab1
        L2, a2, b2 = lab2
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_avg = (C1 + C2) / 2.0
        G = 0.5 * (1.0 - np.sqrt((C_avg**7) / (C_avg**7 + 25**7 + 1e-12)))

        a1_prime = (1.0 + G) * a1
        a2_prime = (1.0 + G) * a2
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)

        h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
        h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

        dL_prime = L2 - L1
        dC_prime = C2_prime - C1_prime

        if C1_prime * C2_prime == 0:
            dh_prime = 0.0
        elif abs(h1_prime - h2_prime) <= 180.0:
            dh_prime = h2_prime - h1_prime
        elif h2_prime <= h1_prime:
            dh_prime = h2_prime - h1_prime + 360.0
        else:
            dh_prime = h2_prime - h1_prime - 360.0

        dH_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2.0))

        L_avg_prime = (L1 + L2) / 2.0
        C_avg_prime = (C1_prime + C2_prime) / 2.0
        if C1_prime * C2_prime == 0:
            h_avg_prime = h1_prime + h2_prime
        elif abs(h1_prime - h2_prime) <= 180.0:
            h_avg_prime = (h1_prime + h2_prime) / 2.0
        elif (h1_prime + h2_prime) < 360.0:
            h_avg_prime = (h1_prime + h2_prime + 360.0) / 2.0
        else:
            h_avg_prime = (h1_prime + h2_prime - 360.0) / 2.0

        T = (1.0 - 0.17 * np.cos(np.radians(h_avg_prime - 30.0))
             + 0.24 * np.cos(np.radians(2.0 * h_avg_prime))
             + 0.32 * np.cos(np.radians(3.0 * h_avg_prime + 6.0))
             - 0.20 * np.cos(np.radians(4.0 * h_avg_prime - 63.0)))

        SL = 1.0 + (0.015 * (L_avg_prime - 50.0)**2) / np.sqrt(20.0 + (L_avg_prime - 50.0)**2)
        SC = 1.0 + 0.045 * C_avg_prime
        SH = 1.0 + 0.015 * C_avg_prime * T

        dTheta = 30.0 * np.exp(-(((h_avg_prime - 275.0) / 25.0)**2))
        RC = 2.0 * np.sqrt((C_avg_prime**7) / (C_avg_prime**7 + 25**7 + 1e-12))
        RT = -np.sin(np.radians(2.0 * dTheta)) * RC

        termL = dL_prime / SL
        termC = dC_prime / SC
        termH = dH_prime / SH
        dE = np.sqrt(termL**2 + termC**2 + termH**2 + RT * termC * termH)
        return float(dE)

    def analiz_et(self, gorsel: Image.Image, k_iplik: int = 5) -> Dict[str, Any]:
        """Halı görselindeki iplikleri kümeleyip katalogla eşler."""
        np_rgb = np.array(gorsel.convert("RGB"))
        H, W, _ = np_rgb.shape
        toplam_piksel = H * W

        lab_pikseller = self._rgb_to_lab(np_rgb)

        # Deterministik K-Means
        np.random.seed(42)
        secilen_idx = np.random.choice(len(lab_pikseller), k_iplik, replace=False)
        merkezler = lab_pikseller[secilen_idx].copy()

        for _ in range(15):
            mesafeler = np.stack([np.sum((lab_pikseller - m) ** 2, axis=1) for m in merkezler], axis=1)
            etiketler = np.argmin(mesafeler, axis=1)
            for j in range(k_iplik):
                mask = (etiketler == j)
                if np.any(mask):
                    merkezler[j] = lab_pikseller[mask].mean(axis=0)

        # Katalog Eşleme
        katalog_lab = [self._rgb_to_lab(np.array([k["rgb"]]))[0].tolist() for k in self.KATALOG]

        iplikler = []
        for i in range(k_iplik):
            yuzde = float(np.sum(etiketler == i) / toplam_piksel * 100.0)
            c_lab = merkezler[i].tolist()

            # En yakın katalog
            en_kucuk_de = float("inf")
            en_iyi_kat = None
            for idx, kat in enumerate(self.KATALOG):
                de = self._delta_e_2000(c_lab, katalog_lab[idx])
                if de < en_kucuk_de:
                    en_kucuk_de = de
                    en_iyi_kat = kat

            iplikler.append({
                "iplik_id": f"YARN-{i+1:02d}",
                "yuzde": float(round(yuzde, 2)),
                "lab": [float(round(c, 2)) for c in c_lab],
                "katalog_ad": en_iyi_kat["ad"],
                "katalog_kod": en_iyi_kat["kod"],
                "katalog_rgb": en_iyi_kat["rgb"],
                "delta_e_2000": float(round(en_kucuk_de, 2)),
                "uyum_durumu": "MUKEMMEL" if en_kucuk_de < 2.0 else "KABUL" if en_kucuk_de < 5.0 else "RED"
            })

        iplikler.sort(key=lambda x: x["yuzde"], reverse=True)
        parti_uyumu = all(i["delta_e_2000"] < 5.0 for i in iplikler)

        return {
            "iplik_sayisi": k_iplik,
            "iplikler": iplikler,
            "parti_renk_uyumu": parti_uyumu
        }
