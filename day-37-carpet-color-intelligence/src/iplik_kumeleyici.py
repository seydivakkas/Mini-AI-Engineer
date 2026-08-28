"""
CIELAB Uzayında K-Means ile İplik Renk Kümeleme ve Yüzde Çıkarıcı (Yarn Color Clusterer).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
from .renk_donusturucu import rgb_to_lab, lab_to_rgb


class IplikRenkKumeleyici:
    """Halı piksel matrisini CIELAB uzayında kümeleyerek iplik renk oranlarını çıkarır."""

    def __init__(self, k_iplik: int = 5, max_iter: int = 25, rastgele_durum: int = 42):
        self.k_iplik = k_iplik
        self.max_iter = max_iter
        self.rastgele_durum = rastgele_durum

    def _kmeans_lab(self, lab_pikseller: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """CIELAB pikselleri üzerinde deterministik K-Means kümeleme."""
        np.random.seed(self.rastgele_durum)
        N = len(lab_pikseller)
        if N <= k:
            return lab_pikseller.copy(), np.arange(N)

        # K-Means++ başlangıç merkezleri
        merkezler = [lab_pikseller[np.random.choice(N)]]
        for _ in range(1, k):
            mesafeler = np.min([np.sum((lab_pikseller - m) ** 2, axis=1) for m in merkezler], axis=0)
            olasiliklar = mesafeler / (mesafeler.sum() + 1e-12)
            merkezler.append(lab_pikseller[np.random.choice(N, p=olasiliklar)])
        merkezler = np.array(merkezler, dtype=np.float64)

        etiketler = np.zeros(N, dtype=np.int32)
        for _ in range(self.max_iter):
            # En yakın merkeze ata
            mesafeler = np.stack([np.sum((lab_pikseller - m) ** 2, axis=1) for m in merkezler], axis=1)
            yeni_etiketler = np.argmin(mesafeler, axis=1)

            if np.array_equal(etiketler, yeni_etiketler):
                break
            etiketler = yeni_etiketler

            # Merkezleri güncelle
            for j in range(k):
                mask = (etiketler == j)
                if np.any(mask):
                    merkezler[j] = lab_pikseller[mask].mean(axis=0)

        return merkezler, etiketler

    def iplik_renklerini_ayristir(self, gorsel: Image.Image) -> Dict[str, Any]:
        """
        Görseldeki pikselleri CIELAB'e çevirir, K-Means ile kümeleyip iplik oranlarını hesaplar.
        """
        img_rgb = np.array(gorsel.convert("RGB"))
        H, W, _ = img_rgb.shape
        toplam_piksel = H * W

        # RGB -> LAB
        lab_img = rgb_to_lab(img_rgb)
        lab_pikseller = lab_img.reshape(-1, 3)

        merkezler_lab, etiketler = self._kmeans_lab(lab_pikseller, self.k_iplik)

        # LAB -> RGB
        merkezler_rgb = lab_to_rgb(merkezler_lab)

        # Sayım ve Yüzde Hesabı
        iplikler = []
        for i in range(self.k_iplik):
            piksel_sayisi = int(np.sum(etiketler == i))
            yuzde = (piksel_sayisi / float(toplam_piksel)) * 100.0

            lab_renk = [float(round(c, 2)) for c in merkezler_lab[i]]
            rgb_renk = [int(c) for c in merkezler_rgb[i]]
            hex_renk = f"#{rgb_renk[0]:02x}{rgb_renk[1]:02x}{rgb_renk[2]:02x}"

            iplikler.append({
                "iplik_id": f"IPLIK-{i+1:02d}",
                "yuzde": float(round(yuzde, 2)),
                "piksel_sayisi": piksel_sayisi,
                "lab": lab_renk,
                "rgb": rgb_renk,
                "hex": hex_renk
            })

        # Yüzdeye göre çoktan aza sırala
        iplikler.sort(key=lambda x: x["yuzde"], reverse=True)

        # Kuantize / Bölütlenmiş görsel
        kuantize_lab = merkezler_lab[etiketler].reshape(H, W, 3)
        kuantize_rgb = lab_to_rgb(kuantize_lab)

        return {
            "toplam_piksel": toplam_piksel,
            "genislik": W,
            "yukseklik": H,
            "iplik_sayisi": self.k_iplik,
            "iplikler": iplikler,
            "kuantize_gorsel_rgb": kuantize_rgb
        }
