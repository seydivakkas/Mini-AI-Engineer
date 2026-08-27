"""K-Means ile Baskın Renk Kümeleme ve Kuantizasyon Modülü.

Bu modül; dijital görüntülerden en baskın K adet renk paletini,
bunların yüzdesel ağırlıklarını, RGB ve HEX kodlarını K-Means kümeleme
algoritması ile çıkarır ve renk kuantizasyonu (Color Quantization) uygular.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class RenkBilgisi:
    """Tek bir baskın renge ait detaylı metrikler."""

    rgb: Tuple[int, int, int]
    hex_kodu: str
    yuzde: float
    piksel_adedi: int


class BaskinRenkCikarici:
    """Görüntülerden K-Means ile renk paleti çıkaran ve kuantize eden motor."""

    def __init__(
        self,
        k_kume_sayisi: int = 5,
        rastgele_durum: int = 42,
        orneklem_limiti: int = 20000
    ) -> None:
        """Kümeleme motorunu başlatır.

        Parametreler:
            k_kume_sayisi (int): Çıkarılacak baskın renk adedi (K).
            rastgele_durum (int): K-Means tekrarlanabilirlik tohumu.
            orneklem_limiti (int): Milyonlarca pikselde eğitimi hızlandırmak için alt örneklem sınırı.
        """
        if k_kume_sayisi < 1:
            raise ValueError("Küme sayısı (K) en az 1 olmalıdır.")

        self.k = int(k_kume_sayisi)
        self.rastgele_durum = int(rastgele_durum)
        self.orneklem_limiti = int(orneklem_limiti)

        self._kmeans = KMeans(
            n_clusters=self.k,
            random_state=self.rastgele_durum,
            n_init="auto"
        )
        self._egitildi_mi = False
        self._kumelenmis_merkezler_rgb: np.ndarray = None

    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int) -> str:
        """RGB tamsayı değerlerini standart HEX renk koduna çevirir."""
        return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

    def paleti_cikar(self, gorsel_bgr: np.ndarray) -> List[RenkBilgisi]:
        """Görüntünün en baskın renklerini ve yüzdesel ağırlıklarını hesaplar.

        Parametreler:
            gorsel_bgr (np.ndarray): H x W x 3 boyutlu BGR görüntü.

        Döndürür:
            List[RenkBilgisi]: Azalan yüzdeye göre sıralanmış renk listesi.
        """
        if gorsel_bgr.ndim != 3 or gorsel_bgr.shape[2] != 3:
            raise ValueError("Girdi 3 kanallı bir BGR görüntü olmalıdır.")

        # BGR -> RGB dönüşümü
        gorsel_rgb = cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2RGB)
        pikseller = gorsel_rgb.reshape(-1, 3).astype(np.float32)
        toplam_piksel = len(pikseller)

        # Çok büyük görüntülerde fit süresini saniyeler mertebesine indirmek için alt örnekleme
        if toplam_piksel > self.orneklem_limiti:
            np.random.seed(self.rastgele_durum)
            indeksler = np.random.choice(toplam_piksel, self.orneklem_limiti, replace=False)
            egitim_verisi = pikseller[indeksler]
        else:
            egitim_verisi = pikseller

        # K-Means eğitimi
        self._kmeans.fit(egitim_verisi)
        self._egitildi_mi = True
        self._kumelenmis_merkezler_rgb = np.clip(np.round(self._kmeans.cluster_centers_), 0, 255).astype(np.uint8)

        # Tüm piksellerin en yakın kümeye atanması
        etiketler = self._kmeans.predict(pikseller)
        etiket_sayilari = np.bincount(etiketler, minlength=self.k)

        # Yüzdelerin hesaplanması ve azalan sırada sıralanması
        sirali_indeksler = np.argsort(etiket_sayilari)[::-1]

        palet: List[RenkBilgisi] = []
        for idx in sirali_indeksler:
            merkez_rgb = self._kumelenmis_merkezler_rgb[idx]
            adet = int(etiket_sayilari[idx])
            yuzde = (adet / toplam_piksel) * 100.0
            hex_kod = self.rgb_to_hex(merkez_rgb[0], merkez_rgb[1], merkez_rgb[2])

            palet.append(
                RenkBilgisi(
                    rgb=(int(merkez_rgb[0]), int(merkez_rgb[1]), int(merkez_rgb[2])),
                    hex_kodu=hex_kod,
                    yuzde=round(yuzde, 2),
                    piksel_adedi=adet
                )
            )

        return palet

    def goruntuyu_quantize_et(self, gorsel_bgr: np.ndarray) -> np.ndarray:
        """Görüntüdeki her pikseli en yakın K-Means merkez rengiyle değiştirir (Kuantizasyon)."""
        if not self._egitildi_mi:
            self.paleti_cikar(gorsel_bgr)

        gorsel_rgb = cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2RGB)
        orijinal_sekil = gorsel_rgb.shape
        pikseller = gorsel_rgb.reshape(-1, 3).astype(np.float32)

        etiketler = self._kmeans.predict(pikseller)
        quantize_rgb = self._kumelenmis_merkezler_rgb[etiketler].reshape(orijinal_sekil)

        # RGB -> BGR dönüşümü ile geri ver
        return cv2.cvtColor(quantize_rgb, cv2.COLOR_RGB2BGR)
