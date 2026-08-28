"""
Halı Görselleri İçin Renk Özellik Çıkarıcı (Color Feature Extractor).
HSV Histogramı, Renk Momentleri ve Dominant Renk Özellikleri.
"""

from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image


class RenkOzellikCikarici:
    """Halı ve tekstil görsellerinden renk histogramı ve renk momentleri vektörü çıkarır."""

    def __init__(self, h_bins: int = 16, s_bins: int = 8, v_bins: int = 8):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins

    def _rgb_to_hsv(self, rgb_norm: np.ndarray) -> np.ndarray:
        """[0, 1] aralığındaki RGB dizisini HSV uzayına dönüştürür."""
        r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        deltac = maxc - minc

        s = np.zeros_like(v)
        nonzero = maxc != 0
        s[nonzero] = deltac[nonzero] / maxc[nonzero]

        h = np.zeros_like(v)
        # r is max
        rc = (maxc - r) / (deltac + 1e-12)
        gc = (maxc - g) / (deltac + 1e-12)
        bc = (maxc - b) / (deltac + 1e-12)

        mask_r = (r == maxc) & (deltac != 0)
        h[mask_r] = bc[mask_r] - gc[mask_r]

        mask_g = (g == maxc) & (deltac != 0)
        h[mask_g] = 2.0 + rc[mask_g] - bc[mask_g]

        mask_b = (b == maxc) & (deltac != 0)
        h[mask_b] = 4.0 + gc[mask_b] - rc[mask_b]

        h = (h / 6.0) % 1.0  # [0, 1]
        return np.stack([h, s, v], axis=-1)

    def _hesapla_renk_momentleri(self, rgb_arr: np.ndarray) -> np.ndarray:
        """Her renk kanalı için 1. Moment (Ortalama), 2. Moment (Std Sapma) ve 3. Moment (Çarpıklık)."""
        momentler = []
        for c in range(3):
            kanal = rgb_arr[..., c].flatten()
            mu = np.mean(kanal)
            std = np.std(kanal) + 1e-8
            skew = np.mean(((kanal - mu) / std) ** 3)
            momentler.extend([mu / 255.0, std / 255.0, np.clip(skew / 5.0, -1.0, 1.0)])
        return np.array(momentler, dtype=np.float64)

    def cikar(self, gorsel: Image.Image) -> Dict[str, Any]:
        """Görselden renk histogramı ve momentleri çıkarıp normalize edilmiş özellik vektörü üretir."""
        rgb_arr = np.array(gorsel.convert("RGB"), dtype=np.float64)
        rgb_norm = rgb_arr / 255.0

        hsv_arr = self._rgb_to_hsv(rgb_norm)

        # 1. 3 Boyutlu HSV Histogramı (H x S x V)
        hist, _ = np.histogramdd(
            hsv_arr.reshape(-1, 3),
            bins=(self.h_bins, self.s_bins, self.v_bins),
            range=((0, 1), (0, 1), (0, 1))
        )
        hist_flat = hist.flatten().astype(np.float64)
        hist_flat = hist_flat / (hist_flat.sum() + 1e-12)  # L1 normalize

        # 2. Renk Momentleri (9 boyut)
        momentler = self._hesapla_renk_momentleri(rgb_arr)

        # 3. Birleşik Renk Özellik Vektörü
        renk_vektoru = np.concatenate([hist_flat, momentler])
        l2_norm = np.linalg.norm(renk_vektoru) + 1e-12
        renk_vektoru_norm = renk_vektoru / l2_norm

        return {
            "renk_vektoru": renk_vektoru_norm,
            "hsv_histogram": hist_flat,
            "renk_momentleri": momentler,
            "vektor_boyutu": len(renk_vektoru_norm),
            "ortalama_rgb": [int(np.mean(rgb_arr[..., i])) for i in range(3)]
        }
