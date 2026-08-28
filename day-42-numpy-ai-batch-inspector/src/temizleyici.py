"""
Tensör Otomatik Temizleyici ve Düzeltici (Batch Sanitizer & Rectifier).
"""

from typing import Tuple, Dict, Any
import numpy as np
from .sema import TensorSemasi


class BatchTemizleyici:
    """Uyarı alan girdi tensörlerini modelin beklediği formata güvenle uyarlar."""

    def __init__(self, sema: TensorSemasi):
        self.sema = sema

    def temizle_ve_uyarla(
        self,
        tensor: np.ndarray,
        denetim_raporu: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        1. NHWC -> NCHW transpose düzeltmesi.
        2. Değer aralığı kırpması (Clamping / Clipping).
        3. Veri tipi dönüştürme (float32).
        4. C-contiguous bellek hizalaması.
        """
        yapilan_islemler = []
        islenmis = tensor.copy()

        # 1. Kanal Düzeni Düzeltmesi (NHWC -> NCHW)
        if denetim_raporu.get("nhwc_tespit_edildi", False):
            # (B, H, W, C) -> (B, C, H, W)
            islenmis = np.transpose(islenmis, (0, 3, 1, 2))
            yapilan_islemler.append("KANAL_DUZENI_TRANSPOSE_NHWC_TO_NCHW")

        # 2. Değer Aralığı Kırpma (Clipping)
        alt, ust = self.sema.deger_araligi
        if denetim_raporu["istatistikler"].get("aralik_disi_piksel", 0) > 0:
            islenmis = np.clip(islenmis, alt, ust)
            yapilan_islemler.append(f"DEGERLER_KIRPILI_CLIP_[{alt},{ust}]")

        # 3. Dtype ve C-Contiguous Dönüşümü
        hedef_dtype = self.sema.gecerli_tipler[0]
        if islenmis.dtype != hedef_dtype or not islenmis.flags['C_CONTIGUOUS']:
            islenmis = np.ascontiguousarray(islenmis, dtype=hedef_dtype)
            yapilan_islemler.append(f"DTYPE_DONUSTURULDU_{hedef_dtype.__name__}_C_CONTIGUOUS")

        return islenmis, {
            "orijinal_sekil": denetim_raporu["sekil"],
            "yeni_sekil": list(islenmis.shape),
            "yeni_dtype": str(islenmis.dtype),
            "yeni_bellek_mb": float(round(islenmis.nbytes / (1024.0 * 1024.0), 3)),
            "c_contiguous": bool(islenmis.flags['C_CONTIGUOUS']),
            "yapilan_islemler": yapilan_islemler
        }
