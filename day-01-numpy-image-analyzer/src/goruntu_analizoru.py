"""NumPy Tabanlı Görüntü Analizörü ve Piksel Manipülasyon Çekirdeği.

Bu modül, ham görsel piksel matrislerini analiz etmek, renk kanallarını ayrıştırmak,
istatistiksel özetler çıkarmak ve derin öğrenme boru hatlarına uygun normalizasyon
işlemlerini gerçekleştirmek için üretim seviyesinde bir sınıf sağlar.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class KanalIstatistigi:
    """Tek bir renk kanalına veya gri seviyeye ait istatistiksel özet."""

    kanal_adi: str
    en_kucuk: float
    en_buyuk: float
    ortalama: float
    medyan: float
    varyans: float
    standart_sapma: float
    yuzdelik_25: float
    yuzdelik_75: float


@dataclass(frozen=True)
class GoruntuOzeti:
    """Görüntünün genel boyut, bellek ve kanal istatistiklerini içeren rapor yapısı."""

    yukseklik: int
    genislik: int
    kanal_sayisi: int
    veri_tipi: str
    toplam_piksel: int
    bellek_kullanimi_kb: float
    kanallar: Dict[str, KanalIstatistigi]
    genel_ortalama: float
    genel_varyans: float


class NumPyGoruntuAnalizoru:
    """NumPy kullanarak piksel düzeyinde görüntü işleme, istatistik ve normalizasyon motoru."""

    KANAL_ISIMLERI = ("Kirmizi", "Yesil", "Mavi", "Alfa")

    def __init__(self, piksel_matrisi: np.ndarray) -> None:
        """NumPyGoruntuAnalizoru sınıfını başlatır ve girdi matrisini doğrular.

        Parametreler:
            piksel_matrisi (np.ndarray): 2 boyutlu (H, W) veya 3 boyutlu (H, W, C) görüntü matrisi.

        Hatalar:
            TypeError: Girdi numpy.ndarray tipinde değilse.
            ValueError: Matris boşsa veya boyutları geçersizse.
        """
        self._girdiyi_dogrula(piksel_matrisi)
        self._matris = piksel_matrisi.copy()

    @staticmethod
    def _girdiyi_dogrula(matris: np.ndarray) -> None:
        """Girdi matrisinin geçerliliğini denetler."""
        if not isinstance(matris, np.ndarray):
            raise TypeError(
                f"Girdi bir NumPy ndarray olmalıdır. Alınan tip: {type(matris)}"
            )

        if matris.size == 0:
            raise ValueError("Girdi matrisi boş olamaz (boyut: 0).")

        if matris.ndim not in (2, 3):
            raise ValueError(
                f"Görüntü matrisi 2B (H, W) veya 3B (H, W, C) olmalıdır. Alınan boyut sayısı: {matris.ndim}"
            )

        if matris.ndim == 3 and matris.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Desteklenen kanal sayıları: 1 (Gri), 3 (RGB) veya 4 (RGBA). Alınan kanal sayısı: {matris.shape[2]}"
            )

    @property
    def matris(self) -> np.ndarray:
        """Orijinal piksel matrisinin kopyasını döndürür (kapsülleme koruması)."""
        return self._matris.copy()

    @property
    def boyutlar(self) -> Tuple[int, ...]:
        """Görüntünün matris boyutlarını (H, W[, C]) döndürür."""
        return self._matris.shape

    @property
    def yukseklik(self) -> int:
        """Piksel yüksekliği (Y ekseni)."""
        return self._matris.shape[0]

    @property
    def genislik(self) -> int:
        """Piksel genişliği (X ekseni)."""
        return self._matris.shape[1]

    @property
    def kanal_sayisi(self) -> int:
        """Renk kanalı sayısı. 2B matrislerde 1 kabul edilir."""
        return 1 if self._matris.ndim == 2 else self._matris.shape[2]

    def kanal_ayristir(self) -> Dict[str, np.ndarray]:
        """Görüntünün her bir renk kanalını bağımsız 2B matrisler olarak ayrıştırır.

        Döndürür:
            Dict[str, np.ndarray]: Kanal adı ve ilgili 2B piksel matrisi sözlüğü.
        """
        if self._matris.ndim == 2:
            return {"Gri": self._matris.copy()}

        kanal_adedi = self._matris.shape[2]
        ayrilmis_kanallar: Dict[str, np.ndarray] = {}

        for indeks in range(kanal_adedi):
            isim = (
                self.KANAL_ISIMLERI[indeks]
                if indeks < len(self.KANAL_ISIMLERI)
                else f"Kanal_{indeks}"
            )
            ayrilmis_kanallar[isim] = self._matris[:, :, indeks].copy()

        return ayrilmis_kanallar

    def kanal_izole_et(self, hedef_kanal: str) -> np.ndarray:
        """Seçilen kanal haricindeki diğer kanalları sıfırlayarak 3B izole görsel üretir.

        Parametreler:
            hedef_kanal (str): 'Kirmizi', 'Yesil' veya 'Mavi'.

        Döndürür:
            np.ndarray: Sadece seçilen kanalın aktif olduğu (H, W, 3) boyutunda matris.

        Hatalar:
            ValueError: 2B gri görüntülerde veya geçersiz kanal adında.
        """
        if self._matris.ndim == 2 or self._matris.shape[2] < 3:
            raise ValueError("Kanal izolasyonu için en az 3 kanallı (RGB) görüntü gereklidir.")

        kanallar = ["Kirmizi", "Yesil", "Mavi"]
        hedef_duzeltilmis = hedef_kanal.capitalize()
        if hedef_duzeltilmis not in kanallar:
            raise ValueError(f"Geçersiz kanal adı: {hedef_kanal}. Geçerli kanallar: {kanallar}")

        kanal_indeksi = kanallar.index(hedef_duzeltilmis)
        izole_matris = np.zeros_like(self._matris[:, :, :3])
        izole_matris[:, :, kanal_indeksi] = self._matris[:, :, kanal_indeksi]

        return izole_matris

    def gri_tona_donustur(
        self,
        agirliklar: Tuple[float, float, float] = (0.299, 0.587, 0.114)
    ) -> np.ndarray:
        """RGB görüntüyü ağırlıklı lüminans (ITU-R BT.601) formülü ile 2B gri tona dönüştürür.

        Formül:
            Y = 0.299 * R + 0.587 * G + 0.114 * B

        Parametreler:
            agirliklar (Tuple[float, float, float]): (Kırmızı, Yeşil, Mavi) kanal ağırlıkları.

        Döndürür:
            np.ndarray: 2B (H, W) gri seviye piksel matrisi.
        """
        if self._matris.ndim == 2:
            return self._matris.copy()

        if self._matris.shape[2] == 1:
            return self._matris[:, :, 0].copy()

        # Ağırlıkların toplamı 1.0 olacak şekilde normalleştirilir
        agirlik_toplami = sum(agirliklar)
        if not np.isclose(agirlik_toplami, 1.0):
            agirliklar = tuple(w / agirlik_toplami for w in agirliklar)  # type: ignore

        kirmizi = self._matris[:, :, 0].astype(np.float32)
        yesil = self._matris[:, :, 1].astype(np.float32)
        mavi = self._matris[:, :, 2].astype(np.float32)

        gri_matris = (
            agirliklar[0] * kirmizi +
            agirliklar[1] * yesil +
            agirliklar[2] * mavi
        )

        if np.issubdtype(self._matris.dtype, np.integer):
            return np.clip(gri_matris, 0, 255).astype(np.uint8)

        return gri_matris.astype(self._matris.dtype)

    def istatistikleri_hesapla(self) -> GoruntuOzeti:
        """Görüntünün her bir kanalı ve geneli için istatistiksel metrikleri çıkarır.

        Döndürür:
            GoruntuOzeti: Min, max, ortalama, medyan, varyans ve çeyreklikleri içeren özet nesnesi.
        """
        ayrik_kanallar = self.kanal_ayristir()
        kanal_istatistikleri: Dict[str, KanalIstatistigi] = {}

        for kanal_adi, kanal_verisi in ayrik_kanallar.items():
            duzlestirilmis = kanal_verisi.astype(np.float64).ravel()
            kanal_istatistikleri[kanal_adi] = KanalIstatistigi(
                kanal_adi=kanal_adi,
                en_kucuk=float(np.min(duzlestirilmis)),
                en_buyuk=float(np.max(duzlestirilmis)),
                ortalama=float(np.mean(duzlestirilmis)),
                medyan=float(np.median(duzlestirilmis)),
                varyans=float(np.var(duzlestirilmis)),
                standart_sapma=float(np.std(duzlestirilmis)),
                yuzdelik_25=float(np.percentile(duzlestirilmis, 25)),
                yuzdelik_75=float(np.percentile(duzlestirilmis, 75)),
            )

        genel_veri = self._matris.astype(np.float64).ravel()
        toplam_piksel = self.yukseklik * self.genislik
        bellek_kb = self._matris.nbytes / 1024.0

        return GoruntuOzeti(
            yukseklik=self.yukseklik,
            genislik=self.genislik,
            kanal_sayisi=self.kanal_sayisi,
            veri_tipi=str(self._matris.dtype),
            toplam_piksel=toplam_piksel,
            bellek_kullanimi_kb=round(bellek_kb, 3),
            kanallar=kanal_istatistikleri,
            genel_ortalama=float(np.mean(genel_veri)),
            genel_varyans=float(np.var(genel_veri)),
        )

    def min_max_normallestir(
        self,
        hedef_aralik: Tuple[float, float] = (0.0, 1.0),
        epsilon: float = 1e-8
    ) -> np.ndarray:
        """Piksel değerlerini belirtilen hedef aralığa doğrusal olarak ölçekler.

        Formül:
            X_norm = a + ((X - X_min) / (X_max - X_min + epsilon)) * (b - a)

        Parametreler:
            hedef_aralik (Tuple[float, float]): Hedef (en_kucuk, en_buyuk) aralığı. Varsayılan (0.0, 1.0).
            epsilon (float): Sıfıra bölme hatasını engelleyen küçük sayısal dengeleyici.

        Döndürür:
            np.ndarray: float32 tipinde ölçeklenmiş matris.
        """
        alt_sinir, ust_sinir = hedef_aralik
        if alt_sinir >= ust_sinir:
            raise ValueError(
                f"Hedef aralık alt sınırı üst sınırdan küçük olmalıdır: ({alt_sinir}, {ust_sinir})"
            )

        kayan_matris = self._matris.astype(np.float32)
        en_kucuk = float(np.min(kayan_matris))
        en_buyuk = float(np.max(kayan_matris))

        aralik_farki = en_buyuk - en_kucuk
        if np.isclose(aralik_farki, 0.0):
            # Tekdüze görüntü durumunda tüm pikseller alt sınıra eşitlenir
            return np.full_like(kayan_matris, alt_sinir, dtype=np.float32)

        oran = (kayan_matris - en_kucuk) / (aralik_farki + epsilon)
        olcekli = alt_sinir + oran * (ust_sinir - alt_sinir)
        return olcekli.astype(np.float32)

    def z_skoru_normallestir(
        self,
        kanal_bazli: bool = True,
        epsilon: float = 1e-7
    ) -> np.ndarray:
        """Pikselleri sıfır ortalama ve birim varyansa (Standartlaştırma) dönüştürür.

        Formül:
            Z = (X - mu) / (sigma + epsilon)

        Parametreler:
            kanal_bazli (bool): True ise her renk kanalı bağımsız ortalama ve sapmayla normalize edilir.
            epsilon (float): Sıfır varyans durumunda sıfıra bölmeyi önleyen payda koruyucu.

        Döndürür:
            np.ndarray: float32 tipinde standardize edilmiş tensör.
        """
        kayan_matris = self._matris.astype(np.float32)

        if not kanal_bazli or self._matris.ndim == 2:
            ortalama = float(np.mean(kayan_matris))
            sapma = float(np.std(kayan_matris))
            return ((kayan_matris - ortalama) / (sapma + epsilon)).astype(np.float32)

        ciktilar = []
        for k in range(self._matris.shape[2]):
            kanal = kayan_matris[:, :, k]
            mu = float(np.mean(kanal))
            sigma = float(np.std(kanal))
            normalize_kanal = (kanal - mu) / (sigma + epsilon)
            ciktilar.append(normalize_kanal)

        return np.stack(ciktilar, axis=-1).astype(np.float32)

    def parlaklik_ayarla(self, katsayi: float) -> np.ndarray:
        """Piksel değerlerini güvenli şekilde katsayı ile çarparak parlaklık değiştirir.

        Taşma (overflow/underflow) problemini önlemek için float dönüşümü yapılır
        ve [0, 255] aralığına sınırlandırılır (clipping).

        Parametreler:
            katsayi (float): Parlaklık çarpanı (ör. 1.2 parlaklaştırır, 0.8 koyulaştırır).

        Döndürür:
            np.ndarray: Orijinal veri tipinde sınırlanmış piksel matrisi.
        """
        if katsayi < 0:
            raise ValueError(f"Parlaklık katsayısı negatif olamaz: {katsayi}")

        kayan_matris = self._matris.astype(np.float32) * katsayi

        if np.issubdtype(self._matris.dtype, np.integer):
            return np.clip(kayan_matris, 0, 255).astype(np.uint8)

        return np.clip(kayan_matris, 0.0, 1.0).astype(self._matris.dtype)

    def kirp(
        self,
        y_baslangic: int,
        y_bitis: int,
        x_baslangic: int,
        x_bitis: int
    ) -> np.ndarray:
        """Verilen koordinat sınırlarına göre görüntüyü güvenli bir şekilde dilimler (crop).

        Parametreler:
            y_baslangic (int): Y ekseni başlangıç pikseli.
            y_bitis (int): Y ekseni bitiş pikseli.
            x_baslangic (int): X ekseni başlangıç pikseli.
            x_bitis (int): X ekseni bitiş pikseli.

        Döndürür:
            np.ndarray: Kırpılmış alt matris.
        """
        # Sınır denetimleri ve sınırlama (clamping)
        y_bas = max(0, min(y_baslangic, self.yukseklik))
        y_son = max(y_bas, min(y_bitis, self.yukseklik))
        x_bas = max(0, min(x_baslangic, self.genislik))
        x_son = max(x_bas, min(x_bitis, self.genislik))

        if y_bas == y_son or x_bas == x_son:
            raise ValueError(
                f"Kırpma alanı sıfır piksel üretiyor: Y[{y_bas}:{y_son}], X[{x_bas}:{x_son}]"
            )

        if self._matris.ndim == 2:
            return self._matris[y_bas:y_son, x_bas:x_son].copy()

        return self._matris[y_bas:y_son, x_bas:x_son, :].copy()
