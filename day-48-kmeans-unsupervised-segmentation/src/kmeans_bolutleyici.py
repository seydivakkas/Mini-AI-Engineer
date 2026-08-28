"""
K-Means Görüntü ve Uzamsal Özellik Bölütleyici (Color Quantization & Spatial Segmentation).
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans


class KMeansGorselBolutleyici:
    """Görüntüleri renk ve uzamsal (RGB + XY) özellik vektörleriyle denetimsiz olarak bölütler."""

    def __init__(self, k_kume: int = 4, uzamsal_agirlik: float = 0.35, random_state: int = 42):
        self.k_kume = k_kume
        self.uzamsal_agirlik = uzamsal_agirlik
        self.random_state = random_state

    def renk_kuantalama_uygula(self, gorsel_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Yalnızca RGB renk uzayında K-Means ile renk kuantalama (Color Quantization) uygular."""
        H, W, C = gorsel_rgb.shape
        piksel_matrisi = (gorsel_rgb.reshape(-1, 3).astype(np.float32)) / 255.0

        km = KMeans(n_clusters=self.k_kume, random_state=self.random_state, n_init=5)
        etiketler = km.fit_predict(piksel_matrisi)
        merkezler = km.cluster_centers_

        kuantalanmis_duz = merkezler[etiketler]
        kuantalanmis_gorsel = (kuantalanmis_duz.reshape(H, W, 3) * 255.0).astype(np.uint8)
        maske = etiketler.reshape(H, W)

        return kuantalanmis_gorsel, maske, merkezler

    def uzamsal_bolutleme_uygula(self, gorsel_rgb: np.ndarray) -> Dict[str, Any]:
        """Renk (RGB) ve piksel koordinatlarını (XY) birleştirerek bitişik uzamsal bölütleme üretir."""
        H, W, C = gorsel_rgb.shape
        rgb_norm = (gorsel_rgb.reshape(-1, 3).astype(np.float32)) / 255.0

        # Normalize Koordinat Grid'i (0 ile 1 arası)
        y_grid, x_grid = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
        xy_ozellik = np.column_stack([x_grid.ravel(), y_grid.ravel()]) * self.uzamsal_agirlik

        # [R, G, B, alpha * X, alpha * Y] Özellik Füzyonu
        fuzyon_ozellik = np.hstack([rgb_norm, xy_ozellik])

        km = KMeans(n_clusters=self.k_kume, random_state=self.random_state, n_init=5)
        etiketler = km.fit_predict(fuzyon_ozellik)

        maske = etiketler.reshape(H, W)

        # Her kümenin ortalama RGB rengini hesapla
        kume_renkleri = np.zeros((self.k_kume, 3), dtype=np.float32)
        alan_yuzdeleri = {}
        toplam_piksel = H * W

        for k in range(self.k_kume):
            k_mask = (etiketler == k)
            alan_yuzdeleri[k] = float(round(np.sum(k_mask) / toplam_piksel * 100.0, 2))
            if np.sum(k_mask) > 0:
                kume_renkleri[k] = rgb_norm[k_mask].mean(axis=0)

        bolutlenmis_duz = kume_renkleri[etiketler]
        bolutlenmis_gorsel = (bolutlenmis_duz.reshape(H, W, 3) * 255.0).astype(np.uint8)

        return {
            "bolutlenmis_gorsel": bolutlenmis_gorsel,
            "maske": maske,
            "alan_yuzdeleri": alan_yuzdeleri,
            "kume_renkleri": kume_renkleri,
            "k_kume": self.k_kume,
            "uzamsal_agirlik": self.uzamsal_agirlik
        }
