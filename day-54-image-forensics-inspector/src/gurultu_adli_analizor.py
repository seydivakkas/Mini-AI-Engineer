"""
Sensör Gürültü Kalıntısı ve Lokal Varyans Tutarsızlığı Analizörü (Noise Residual & Inconsistency Analyzer).
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np


class GurultuAdliAnalizoru:
    """Kamera sensörü gürültü kalıntılarını (PRNU kalıntısı) ve ekleme/kopyalama alanlarındaki gürültü tutarsızlıklarını inceler."""

    @classmethod
    def gurultu_kalintisi_hesapla(
        cls,
        img_rgb: np.ndarray,
        filtre_ksize: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Medyan filtreleme ile görselin yüksek frekans sensör gürültü kalıntısını (Residual) ayıklar."""
        gri = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if len(img_rgb.shape) == 3 else img_rgb.copy()
        duzlestirilmis = cv2.medianBlur(gri, filtre_ksize)
        kalinti = cv2.absdiff(gri, duzlestirilmis)

        # Görselleştirme için normalize edilmiş gürültü haritası
        kalinti_norm = cv2.normalize(kalinti, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return kalinti, kalinti_norm

    @classmethod
    def lokal_gurultu_varyansi_haritasi(
        cls,
        kalinti: np.ndarray,
        blok_boyutu: int = 16
    ) -> Tuple[np.ndarray, float]:
        """Gürültü kalıntısı üzerinde blok tabanlı lokal varyans haritası ve tutarsızlık katsayısı (CV) çıkarır."""
        h, w = kalinti.shape
        pad_h = (blok_boyutu - (h % blok_boyutu)) % blok_boyutu
        pad_w = (blok_boyutu - (w % blok_boyutu)) % blok_boyutu

        kalinti_pad = np.pad(kalinti, ((0, pad_h), (0, pad_w)), mode="reflect").astype(float)
        h_pad, w_pad = kalinti_pad.shape

        varyans_haritasi = np.zeros((h_pad // blok_boyutu, w_pad // blok_boyutu), dtype=float)

        for i in range(0, h_pad, blok_boyutu):
            for j in range(0, w_pad, blok_boyutu):
                blok = kalinti_pad[i:i+blok_boyutu, j:j+blok_boyutu]
                varyans_haritasi[i // blok_boyutu, j // blok_boyutu] = float(np.var(blok))

        # Orijinal boyuta geri ölçekleme (inter-cubic)
        varyans_tam = cv2.resize(varyans_haritasi, (w, h), interpolation=cv2.INTER_CUBIC)

        # Değişim Katsayısı (Coefficient of Variation - CV = std / mean)
        mu_var = float(np.mean(varyans_haritasi)) + 1e-6
        std_var = float(np.std(varyans_haritasi))
        tutarsizlik_cv = float(round(std_var / mu_var, 3))

        return varyans_tam, tutarsizlik_cv
