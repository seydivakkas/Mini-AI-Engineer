"""
Görsel Arama ve Desen Benzerlik Motoru (HSV & GLCM Haralick).
"""

from typing import List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw


class GorselAramaMotoru:
    """Halı görsellerini renk histogramı ve GLCM doku özellikleriyle indeksleyip arar."""

    KATALOG_TANIMLARI = [
        {"id": "CARPET-CLASSIC-01", "ad": "Hereke Klasik Madalyonlu Bordo", "kategori": "Klasik", "renkler": (138, 28, 48)},
        {"id": "CARPET-MODERN-02", "ad": "İskandinav Geometrik Triko", "kategori": "Modern", "renkler": (204, 154, 45)},
        {"id": "CARPET-SHAGGY-03", "ad": "Kabarık Yün Shaggy Antrasit", "kategori": "Shaggy", "renkler": (50, 52, 58)},
        {"id": "CARPET-VINTAGE-04", "ad": "Anadolu Eskitme Terracotta", "kategori": "Vintage", "renkler": (184, 85, 55)},
        {"id": "CARPET-SILK-05", "ad": "Osmanlı İpek Çiçekli Zümrüt", "kategori": "Saray İpeği", "renkler": (32, 98, 65)}
    ]

    def __init__(self):
        self.indeks: List[Dict[str, Any]] = []
        self._katalog_olustur_ve_indeksle()

    def _desen_uret(self, tanim: Dict[str, Any]) -> Image.Image:
        img = Image.new("RGB", (200, 200), color=(220, 210, 195))
        draw = ImageDraw.Draw(img)
        c = tanim["renkler"]
        draw.rectangle([10, 10, 190, 190], outline=c, width=12)
        draw.ellipse([60, 60, 140, 140], fill=c)
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 4, arr.shape).astype(np.int16)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    def _ozellik_cikar(self, gorsel: Image.Image) -> np.ndarray:
        rgb = np.array(gorsel.convert("RGB"), dtype=np.float64) / 255.0
        gri = np.array(gorsel.convert("L"), dtype=np.float64)

        # 1. Basitleştirilmiş Renk Özelliği (RGB Moments)
        r_oz = [np.mean(rgb[..., i]) for i in range(3)] + [np.std(rgb[..., i]) for i in range(3)]

        # 2. GLCM Haralick Basitleştirilmiş Kontrast & Homojenlik
        seviye = 8
        q_gri = (gri / 256.0 * seviye).astype(np.int32)
        glcm = np.zeros((seviye, seviye), dtype=np.float64)
        for y in range(q_gri.shape[0]):
            for x in range(q_gri.shape[1] - 1):
                glcm[q_gri[y, x], q_gri[y, x + 1]] += 1.0
        glcm = glcm / (glcm.sum() + 1e-12)

        i_idx, j_idx = np.indices(glcm.shape)
        kontrast = float(np.sum(((i_idx - j_idx) ** 2) * glcm))
        homojenlik = float(np.sum(glcm / (1.0 + np.abs(i_idx - j_idx))))

        vec = np.array(r_oz + [kontrast, homojenlik], dtype=np.float64)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def _katalog_olustur_ve_indeksle(self):
        self.indeks = []
        for tanim in self.KATALOG_TANIMLARI:
            g = self._desen_uret(tanim)
            vec = self._ozellik_cikar(g)
            self.indeks.append({
                "id": tanim["id"],
                "ad": tanim["ad"],
                "kategori": tanim["kategori"],
                "gorsel": g,
                "vektor": vec
            })

    def ara(self, sorgu_gorseli: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        """Sorgu görseline en benzer katalog halılarını döndürür."""
        q_vec = self._ozellik_cikar(sorgu_gorseli)

        eslesmeler = []
        for item in self.indeks:
            sim = float(np.clip((np.dot(q_vec, item["vektor"]) + 1.0) / 2.0, 0.0, 1.0))
            eslesmeler.append({
                "id": item["id"],
                "ad": item["ad"],
                "kategori": item["kategori"],
                "benzerlik_skoru": float(round(sim * 100.0, 2)),
                "gorsel": item["gorsel"]
            })

        eslesmeler.sort(key=lambda x: x["benzerlik_skoru"], reverse=True)
        return {
            "en_iyi_eslesme": eslesmeler[0] if eslesmeler else None,
            "top_sonuclar": eslesmeler[:top_k]
        }
