"""
Laplacian Varyansı ve 2D FFT Frekans Spektrumu ile Bulanıklık Analizörü (Blur & Focus Analyzer).
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


class BulaniklikAnalizoru:
    """Laplacian türevi, 2D Hızlı Fourier Dönüşümü (FFT) ve Tenengrad gradyanları ile görüntü netliğini ölçer."""

    @classmethod
    def griye_cevir(cls, img: np.ndarray) -> np.ndarray:
        """Giriş görüntüsünü güvenle tek kanallı gri seviyeye (Grayscale) dönüştürür."""
        if len(img.shape) == 2:
            return img.copy()
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        raise ValueError(f"Desteklenmeyen görüntü boyutu: {img.shape}")

    @classmethod
    def laplacian_varyansi_hesapla(cls, gri_img: np.ndarray, ksize: int = 3) -> Tuple[float, np.ndarray]:
        """Laplacian ikinci dereceden türev operatörünü uygular ve varyansını (Focus Measure) döndürür."""
        laplacian = cv2.Laplacian(gri_img, cv2.CV_64F, ksize=ksize)
        varyans = float(round(float(laplacian.var()), 2))
        laplacian_norm = cv2.convertScaleAbs(laplacian)
        return varyans, laplacian_norm

    @classmethod
    def fft_frekans_spektrumu_hesapla(
        cls,
        gri_img: np.ndarray,
        merkez_cap: int = 30
    ) -> Tuple[float, np.ndarray]:
        """2D FFT ile frekans spektrumunu ve yüksek frekans enerji oranını (HFR) hesaplar."""
        h, w = gri_img.shape
        cy, cx = h // 2, w // 2

        # 2D FFT ve merkeze kaydırma
        f_transform = np.fft.fft2(gri_img.astype(float))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Logaritmik büyüklük spektrumu (görselleştirme için)
        spektrum_gorunur = 20 * np.log(magnitude + 1.0)
        spektrum_norm = cv2.normalize(spektrum_gorunur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Merkezdeki düşük frekansları maskeleme
        y, x = np.ogrid[:h, :w]
        maske_alcak_frekans = ((x - cx)**2 + (y - cy)**2) <= merkez_cap**2

        toplam_enerji = float(np.sum(magnitude**2)) + 1e-8
        alcak_enerji = float(np.sum((magnitude * maske_alcak_frekans)**2))
        yuksek_enerji = toplam_enerji - alcak_enerji

        yuksek_frekans_orani = float(round((yuksek_enerji / toplam_enerji) * 100.0, 3))
        return yuksek_frekans_orani, spektrum_norm

    @classmethod
    def tenengrad_netlik_skoru(cls, gri_img: np.ndarray) -> float:
        """Sobel gradyan kareleri toplamı ile Tenengrad odak ölçütünü hesaplar."""
        gx = cv2.Sobel(gri_img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gri_img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag_kare = gx**2 + gy**2
        return float(round(float(np.mean(grad_mag_kare)), 2))

    @classmethod
    def analiz_et(
        cls,
        img: np.ndarray,
        laplacian_esigi: float = 80.0,
        fft_esigi: float = 0.001
    ) -> Dict[str, Any]:
        """Tüm bulanıklık ve frekans metriklerini birleşik olarak hesaplar ve karar üretir."""
        gri = cls.griye_cevir(img)
        lap_var, lap_harita = cls.laplacian_varyansi_hesapla(gri)
        fft_hfr, fft_spektrum = cls.fft_frekans_spektrumu_hesapla(gri)
        tenengrad = cls.tenengrad_netlik_skoru(gri)

        net_mi = (lap_var >= laplacian_esigi)
        karar = "NET" if net_mi else "HAFİF_BULANIK" if (lap_var >= laplacian_esigi * 0.5) else "AŞIRI_BULANIK"

        return {
            "gri_goruntu": gri,
            "laplacian_varyansi": lap_var,
            "laplacian_haritasi": lap_harita,
            "fft_yuksek_frekans_orani": fft_hfr,
            "fft_spektrum": fft_spektrum,
            "tenengrad_skoru": tenengrad,
            "net_mi": net_mi,
            "karar": karar
        }
