"""
Error Level Analysis (ELA) Hata Düzeyi Analizörü (JPEG Recompression Error Analysis).
"""

from typing import Tuple, Dict, Any
import io
import numpy as np
from PIL import Image


class ErrorLevelAnalizoru:
    """JPEG DCT blok sıkıştırma dengesizliklerini ve manipüle edilmiş bölgelerdeki hata farklarını (ELA) tespit eder."""

    @classmethod
    def ela_hesapla(
        cls,
        img_rgb: np.ndarray,
        kalite: int = 90,
        olcek_carpani: float = 15.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Görseli bellekte belirli JPEG kalitesinde yeniden sıkıştırıp orijinaliyle farkını (ELA) çıkarır."""
        pil_orijinal = Image.fromarray(img_rgb)

        # Bellek içinde JPEG olarak yeniden sıkıştırma (in-memory buffer)
        tampon = io.BytesIO()
        pil_orijinal.save(tampon, format="JPEG", quality=kalite)
        tampon.seek(0)
        pil_sikistirilmis = Image.open(tampon)
        img_sikistirilmis = np.array(pil_sikistirilmis.convert("RGB"))

        # Mutlak piksel farkı (Absolute Difference)
        fark_kanallar = np.abs(img_rgb.astype(np.float32) - img_sikistirilmis.astype(np.float32))
        fark_gri = np.mean(fark_kanallar, axis=2)

        # Kontrast artırılmış ELA ısı haritası
        ela_rgb = np.clip(fark_kanallar * olcek_carpani, 0, 255).astype(np.uint8)

        # İstatistiki hata metrikleri
        istatistikler = {
            "ortalama_hata": float(round(float(np.mean(fark_gri)), 3)),
            "maks_hata": float(round(float(np.max(fark_gri)), 3)),
            "varyans_hata": float(round(float(np.var(fark_gri)), 3)),
            "standart_sapma": float(round(float(np.std(fark_gri)), 3))
        }

        return ela_rgb, fark_gri, istatistikler

    @classmethod
    def anomali_maskesi_uret(
        cls,
        fark_gri: np.ndarray,
        z_esik: float = 2.5
    ) -> np.ndarray:
        """ELA fark haritasında ortalamadan z_esik standart sapma kadar sapan pikselleri ikili maskeler."""
        mu = np.mean(fark_gri)
        sigma = np.std(fark_gri) + 1e-6
        z_haritasi = (fark_gri - mu) / sigma
        maske = (z_haritasi > z_esik).astype(np.uint8) * 255
        return maske
