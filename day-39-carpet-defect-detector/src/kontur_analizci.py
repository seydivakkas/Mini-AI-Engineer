"""
Bağlantılı Bileşen ve Kontur Geometrisi Analizcisi (Connected Component & Shape Analyzer).
"""

from typing import List, Dict, Any
import numpy as np
from scipy.ndimage import label, find_objects, binary_erosion, generate_binary_structure


class KonturAnalizci:
    """Temizlenmiş ikili maskeden bağlantılı kusur bölgelerini bulur ve geometrik özelliklerini çıkarır."""

    def __init__(self, min_kusur_alani: int = 30):
        self.min_alan = min_kusur_alani

    def analiz_et(self, ikili_maske: np.ndarray) -> List[Dict[str, Any]]:
        """Maskedeki tüm bağımsız kusur bölgelerini tespit edip metriklerini döndürür."""
        yapi = generate_binary_structure(2, 2)
        etiketli_matris, nesne_sayisi = label(ikili_maske, structure=yapi)

        dilimler = find_objects(etiketli_matris)
        kusurlar = []

        for idx, dilim in enumerate(dilimler):
            if dilim is None:
                continue

            bolge_maske = (etiketli_matris[dilim] == (idx + 1))
            alan = int(np.sum(bolge_maske))

            if alan < self.min_alan:
                continue

            y_dilim, x_dilim = dilim
            y_min, y_max = y_dilim.start, y_dilim.stop
            x_min, x_max = x_dilim.start, x_dilim.stop

            genislik = x_max - x_min
            yukseklik = y_max - y_min

            # En-Boy Oranı (Uzunluk Faktörü)
            en_boy_orani = float(max(genislik, yukseklik) / (min(genislik, yukseklik) + 1e-6))

            # Çevre Hesabı (Erozyon farkı)
            erozyon = binary_erosion(bolge_maske, structure=yapi)
            cevre = int(np.sum(bolge_maske ^ erozyon)) + 1e-6

            # Dairesellik / Kompaktlık: 4 * pi * A / (P^2)
            dairesellik = float(np.clip((4.0 * np.pi * alan) / (cevre ** 2), 0.0, 1.0))

            # Doluluk Oranı (Extent / Density)
            doluluk = float(alan / (genislik * yukseklik + 1e-6))

            # Ağırlık Merkezi
            y_indis, x_indis = np.where(bolge_maske)
            merkez_x = float(round(x_min + np.mean(x_indis), 1))
            merkez_y = float(round(y_min + np.mean(y_indis), 1))

            kusurlar.append({
                "kusur_id": f"DEFECT-{len(kusurlar) + 1:02d}",
                "kutu": [x_min, y_min, genislik, yukseklik],
                "alan": alan,
                "genislik": genislik,
                "yukseklik": yukseklik,
                "en_boy_orani": float(round(en_boy_orani, 2)),
                "dairesellik": float(round(dairesellik, 2)),
                "doluluk": float(round(doluluk, 2)),
                "merkez": [merkez_x, merkez_y]
            })

        # Alana göre büyükten küçüğe sırala
        kusurlar.sort(key=lambda x: x["alan"], reverse=True)
        return kusurlar
