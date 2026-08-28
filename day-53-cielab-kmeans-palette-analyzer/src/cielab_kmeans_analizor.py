"""
CIELAB Algısal Renk Uzayında K-Means Dominant Palet Çıkarıcı (CIELAB K-Means Palette Extractor).
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.cluster import KMeans
from .renk_uzayi_donusturucu import RenkUzayiDonusturucu


class CIELABKMeansPaletAnalizoru:
    """CIELAB (L*a*b*) uzayında K-Means kümeleme ile algısal olarak dengeli dominant renk paleti çıkarır."""

    def __init__(self, k_renk: int = 5, random_state: int = 42):
        self.k_renk = k_renk
        self.random_state = random_state

    def palet_cikar(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """Görseldeki tüm pikselleri CIELAB uzayına taşır, K-Means ile kümeleyip dominant palet üretir."""
        h, w = img_rgb.shape[:2]
        lab_img = RenkUzayiDonusturucu.rgb_to_cielab(img_rgb)
        lab_pikseller = lab_img.reshape(-1, 3)

        # K-Means Kümeleme
        kmeans = KMeans(
            n_clusters=self.k_renk,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=self.random_state
        )
        etiketler = kmeans.fit_predict(lab_pikseller)
        merkezler_lab = kmeans.cluster_centers_

        # Piksel Yoğunlukları ve Sıralama
        benzersiz, adetler = np.unique(etiketler, return_counts=True)
        toplam_piksel = float(len(etiketler))
        oranlar = (adetler / toplam_piksel) * 100.0

        # Azalan baskınlık sırasına göre indeksler
        sirali_indeksler = np.argsort(oranlar)[::-1]

        palet_listesi: List[Dict[str, Any]] = []
        for rank, idx in enumerate(sirali_indeksler):
            lab = merkezler_lab[idx]
            rgb = RenkUzayiDonusturucu.cielab_to_rgb(lab.reshape(1, 1, 3))[0, 0]
            hex_kod = RenkUzayiDonusturucu.rgb_to_hex(rgb)
            yuzde = float(round(oranlar[idx], 2))

            palet_listesi.append({
                "sira": rank + 1,
                "hex": hex_kod,
                "rgb": tuple(map(int, rgb)),
                "lab": tuple(round(float(v), 2) for v in lab),
                "yuzde": yuzde,
                "piksel_sayisi": int(adetler[idx])
            })

        # Kuantize Edilmiş Görsel Üretimi (Segmentasyon Haritası)
        kuantize_lab = merkezler_lab[etiketler].reshape(h, w, 3)
        kuantize_rgb = RenkUzayiDonusturucu.cielab_to_rgb(kuantize_lab)

        return {
            "k_renk": self.k_renk,
            "palet": palet_listesi,
            "kuantize_gorsel": kuantize_rgb,
            "lab_pikseller": lab_pikseller,
            "lab_merkezler": merkezler_lab[sirali_indeksler]
        }
