"""
Dokuma Hataları ve Kusur Tespit Motoru (Anomali, Morfoloji & Sınıflandırma).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter, label, find_objects, binary_opening, binary_closing, generate_binary_structure
from PIL import Image


class KusurTespitMotoru:
    """Halı görselinde yerel/referans anomali tespiti ve kontur analizi yapar."""

    def __init__(self, sigma: float = 2.5, esik_carpani: float = 2.8):
        self.sigma = sigma
        self.esik_carpani = esik_carpani

    def tespit_et(
        self,
        test_gorseli: Image.Image,
        referans_gorseli: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        gri_test = np.array(test_gorseli.convert("L"), dtype=np.float64)

        if referans_gorseli is not None:
            gri_ref = np.array(referans_gorseli.convert("L"), dtype=np.float64)
            kalinti = np.abs(gri_test - gri_ref)
        else:
            arka_plan = gaussian_filter(gri_test, sigma=self.sigma * 3.0)
            kalinti = np.abs(gri_test - arka_plan)

        kalinti_duzgun = gaussian_filter(kalinti, sigma=self.sigma)
        max_val = np.max(kalinti_duzgun) + 1e-12
        anomali_haritasi = kalinti_duzgun / max_val

        # Eşikleme
        mu = np.mean(kalinti_duzgun)
        std = np.std(kalinti_duzgun)
        esik = mu + (self.esik_carpani * std)
        ham_maske = (kalinti_duzgun > esik).astype(np.uint8)

        # Morfolojik Açma/Kapama
        yapi = generate_binary_structure(2, 2)
        maske_acik = binary_opening(ham_maske, structure=yapi, iterations=1)
        temiz_maske = binary_closing(maske_acik, structure=yapi, iterations=2).astype(np.uint8)

        # Kontur & Bağlantılı Bileşen Analizi
        etiketli, _ = label(temiz_maske, structure=yapi)
        dilimler = find_objects(etiketli)

        kusurlar = []
        for idx, dilim in enumerate(dilimler):
            if dilim is None:
                continue
            bolge = (etiketli[dilim] == (idx + 1))
            alan = int(np.sum(bolge))
            if alan < 30:
                continue

            y_dilim, x_dilim = dilim
            y_min, y_max = y_dilim.start, y_dilim.stop
            x_min, x_max = x_dilim.start, x_dilim.stop
            w, h = x_max - x_min, y_max - y_min

            ar = float(max(w, h) / (min(w, h) + 1e-6))
            cevre = max(2 * (w + h), 1)
            dairesellik = float(np.clip((4.0 * np.pi * alan) / (cevre ** 2), 0.0, 1.0))

            # Sınıflandırma
            if ar >= 3.2:
                tur = "IPLIK_KOPMASI"
                siddet = "KRITIK" if alan >= 500 else "ORTA_KUSUR"
            elif dairesellik >= 0.40 and ar < 2.0:
                tur = "YAG_BOYA_LEKESI"
                siddet = "KRITIK" if alan >= 500 else "ORTA_KUSUR"
            elif alan >= 400:
                tur = "DELIK_YIRTIK"
                siddet = "KRITIK"
            else:
                tur = "DUGUM_TOPAKLANMA"
                siddet = "KUCUK_KUSUR"

            kusurlar.append({
                "kusur_id": f"DEF-{len(kusurlar)+1:02d}",
                "kusur_turu": tur,
                "siddet": siddet,
                "kutu": [x_min, y_min, w, h],
                "alan": alan,
                "en_boy_orani": float(round(ar, 2)),
                "dairesellik": float(round(dairesellik, 2))
            })

        kusurlar.sort(key=lambda x: x["alan"], reverse=True)
        kritik_sayisi = sum(1 for k in kusurlar if k["siddet"] == "KRITIK")
        parti_onayi = (kritik_sayisi == 0)

        return {
            "kusur_sayisi": len(kusurlar),
            "kritik_kusur_sayisi": kritik_sayisi,
            "parti_onayi": parti_onayi,
            "kusurlar": kusurlar,
            "anomali_haritasi": anomali_haritasi,
            "temiz_maske": temiz_maske
        }
