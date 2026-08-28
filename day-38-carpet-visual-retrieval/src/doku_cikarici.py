"""
Halı Görselleri İçin Doku Özellik Çıkarıcı (Texture Feature Extractor).
GLCM (Haralick Doku İstatistikleri) ve LBP (Local Binary Pattern) Tanımlayıcıları.
"""

from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image


class DokuOzellikCikarici:
    """GLCM Haralick öznitelikleri ve LBP mikro-doku histogramını çıkarır."""

    def __init__(self, gri_seviye_sayisi: int = 16, lbp_bins: int = 16):
        self.seviye = gri_seviye_sayisi
        self.lbp_bins = lbp_bins

    def _griye_ve_kuantize_et(self, gorsel: Image.Image) -> np.ndarray:
        """Görseli gri tona çevirip belirlenen seviyede kuantize eder (0 .. seviye-1)."""
        gri = np.array(gorsel.convert("L"), dtype=np.float64)
        kuantize = (gri / 256.0 * self.seviye).astype(np.int32)
        return np.clip(kuantize, 0, self.seviye - 1)

    def _hesapla_glcm_matrisi(self, gri_img: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Belirtilen (dy, dx) yön vektöründe eş-oluşum matrisini hesaplar."""
        H, W = gri_img.shape
        glcm = np.zeros((self.seviye, self.seviye), dtype=np.float64)

        y_start = max(0, -dy)
        y_end = min(H, H - dy)
        x_start = max(0, -dx)
        x_end = min(W, W - dx)

        p1 = gri_img[y_start:y_end, x_start:x_end]
        p2 = gri_img[y_start + dy:y_end + dy, x_start + dx:x_end + dx]

        for i in range(p1.shape[0]):
            for j in range(p1.shape[1]):
                glcm[p1[i, j], p2[i, j]] += 1.0

        # Simetrik GLCM
        glcm_simetrik = glcm + glcm.T
        glcm_norm = glcm_simetrik / (glcm_simetrik.sum() + 1e-12)
        return glcm_norm

    def _haralick_istatistikleri(self, glcm: np.ndarray) -> Dict[str, float]:
        """Normalize GLCM matrisinden 5 temel Haralick istatistiğini çıkarır."""
        i_idx, j_idx = np.indices(glcm.shape)

        # Kontrast
        kontrast = np.sum(((i_idx - j_idx) ** 2) * glcm)

        # Homojenlik (Inverse Difference Moment)
        homojenlik = np.sum(glcm / (1.0 + np.abs(i_idx - j_idx)))

        # Enerji (Angular Second Moment - ASM)
        asm = np.sum(glcm ** 2)
        enerji = np.sqrt(asm)

        # Korelasyon
        mu_i = np.sum(i_idx * glcm)
        mu_j = np.sum(j_idx * glcm)
        sigma_i = np.sqrt(np.sum(((i_idx - mu_i) ** 2) * glcm)) + 1e-12
        sigma_j = np.sqrt(np.sum(((j_idx - mu_j) ** 2) * glcm)) + 1e-12
        korelasyon = np.sum((i_idx - mu_i) * (j_idx - mu_j) * glcm) / (sigma_i * sigma_j)

        # Entropi
        glcm_nonzero = glcm[glcm > 0]
        entropi = -np.sum(glcm_nonzero * np.log2(glcm_nonzero + 1e-12))

        return {
            "kontrast": float(kontrast),
            "homojenlik": float(homojenlik),
            "enerji": float(enerji),
            "korelasyon": float(korelasyon),
            "entropi": float(entropi)
        }

    def _hesapla_lbp(self, gri_img: np.ndarray) -> np.ndarray:
        """8-komşuluk Local Binary Pattern (LBP) mikro-doku haritası ve histogramı."""
        H, W = gri_img.shape
        lbp_harita = np.zeros((H - 2, W - 2), dtype=np.uint8)

        merkez = gri_img[1:-1, 1:-1]
        komsular = [
            gri_img[0:-2, 0:-2],  # Sol Üst
            gri_img[0:-2, 1:-1],  # Üst
            gri_img[0:-2, 2:],    # Sağ Üst
            gri_img[1:-1, 2:],    # Sağ
            gri_img[2:, 2:],      # Sağ Alt
            gri_img[2:, 1:-1],    # Alt
            gri_img[2:, 0:-2],    # Sol Alt
            gri_img[1:-1, 0:-2]   # Sol
        ]

        for bit, komsu in enumerate(komsular):
            lbp_harita += ((komsu >= merkez).astype(np.uint8) << bit)

        # LBP Histogramı
        hist, _ = np.histogram(lbp_harita.flatten(), bins=self.lbp_bins, range=(0, 256))
        hist_norm = hist.astype(np.float64) / (hist.sum() + 1e-12)
        return hist_norm

    def cikar(self, gorsel: Image.Image) -> Dict[str, Any]:
        """Görselden Haralick yönlü doku özellikleri ve LBP histogramı çıkarır."""
        gri_img = self._griye_ve_kuantize_et(gorsel)

        # 4 Yön: 0° (0,1), 45° (-1,1), 90° (-1,0), 135° (-1,-1)
        yonler = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
        haralick_ozellikleri = []
        yonlu_istatistikler = []

        for dy, dx in yonler:
            glcm = self._hesapla_glcm_matrisi(gri_img, dy, dx)
            ist = self._haralick_istatistikleri(glcm)
            yonlu_istatistikler.append(ist)
            haralick_ozellikleri.extend([
                ist["kontrast"] / (self.seviye ** 2),
                ist["homojenlik"],
                ist["enerji"],
                (ist["korelasyon"] + 1.0) / 2.0,  # [0, 1] aralığına normalize
                ist["entropi"] / 8.0
            ])

        # LBP Mikro-Doku
        lbp_hist = self._hesapla_lbp(np.array(gorsel.convert("L")))

        # Birleşik Doku Vektörü
        doku_vektoru = np.concatenate([np.array(haralick_ozellikleri), lbp_hist])
        l2_norm = np.linalg.norm(doku_vektoru) + 1e-12
        doku_vektoru_norm = doku_vektoru / l2_norm

        # Ortalama Haralick
        ortalama_haralick = {
            "kontrast": float(np.mean([x["kontrast"] for x in yonlu_istatistikler])),
            "homojenlik": float(np.mean([x["homojenlik"] for x in yonlu_istatistikler])),
            "enerji": float(np.mean([x["enerji"] for x in yonlu_istatistikler])),
            "korelasyon": float(np.mean([x["korelasyon"] for x in yonlu_istatistikler])),
            "entropi": float(np.mean([x["entropi"] for x in yonlu_istatistikler]))
        }

        return {
            "doku_vektoru": doku_vektoru_norm,
            "haralick_ortalama": ortalama_haralick,
            "lbp_histogram": lbp_hist,
            "vektor_boyutu": len(doku_vektoru_norm)
        }
