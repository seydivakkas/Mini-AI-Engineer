"""Ara Katman Aktivasyon Çıkarıcı Modülü (Feature Map Extractor).

Eğitilmiş CNN modelinin Conv2D katmanlarındaki ara aktivasyon haritalarını
çıkararak görselleştirir (Model Explainability / XAI).
"""

from typing import List, Optional
import keras
from keras import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class AraKatmanAktivasyonCikarici:
    """CNN modelinin evrişim filtre çıktılarının aktivasyon haritasını çıkaran sınıf."""

    def __init__(self, model: keras.Model) -> None:
        """Aktivasyon çıkarıcıyı eğitilmiş model ile ilklendirir."""
        self.model = model

    def aktivasyon_haritasi_cikar(
        self, gorsel_rgb: np.ndarray, katman_adi: str = "conv2d_blok1"
    ) -> np.ndarray:
        """Belirtilen Conv2D katmanının aktivasyon haritasını döndürür.

        Args:
            gorsel_rgb: (H, W, 3) veya (1, H, W, 3) float32 [0.0, 1.0] görsel.
            katman_adi: Aktivasyonu çıkarılacak katman adı.

        Returns:
            np.ndarray: (1, H_out, W_out, C_filters) boyutunda aktivasyon tensörü.
        """
        if len(gorsel_rgb.shape) == 3:
            girdi = np.expand_dims(gorsel_rgb, axis=0)
        else:
            girdi = gorsel_rgb

        hedef_katman = self.model.get_layer(katman_adi)
        ara_model = models.Model(
            inputs=self.model.inputs,
            outputs=hedef_katman.output
        )

        aktivasyon = ara_model.predict(girdi, verbose=0)
        return aktivasyon

    def aktivasyon_grid_ciz(
        self,
        aktivasyon: np.ndarray,
        maks_filtre: int = 16,
        baslik: str = "Conv2D Aktivasyon Haritaları (Feature Maps)",
    ) -> plt.Figure:
        """Aktivasyon haritalarını ızgara (grid) şeklinde çizer."""
        n_filtre = min(maks_filtre, aktivasyon.shape[-1])
        satirlar = int(np.ceil(np.sqrt(n_filtre)))
        sutunlar = int(np.ceil(n_filtre / satirlar))

        fig, eksenler = plt.subplots(satirlar, sutunlar, figsize=(sutunlar * 2.5, satirlar * 2.5), dpi=120)
        fig.suptitle(baslik, fontsize=12, fontweight="bold")
        eksenler_duz = np.array(eksenler).reshape(-1)

        for i in range(len(eksenler_duz)):
            ax = eksenler_duz[i]
            if i < n_filtre:
                f_map = aktivasyon[0, :, :, i]
                ax.imshow(f_map, cmap="viridis")
                ax.set_title(f"Filtre #{i+1}", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        return fig
