"""Vektörel ve Piksel Düzeyinde Mesafe ve Benzerlik Metrikleri Modülü.

Bu modül; Öklid (L2), Manhattan (L1), Kosinüs, Chebyshev (L-sonsuz) ve Minkowski
metriklerini hem tekil vektörler hem de toplu (batch) tensörler ve piksel haritaları
için yüksek performanslı, tamamen vektörize edilmiş NumPy operasyonlarıyla hesaplar.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class MesafeSonucu:
    """Tekil bir mesafe veya benzerlik metriği ölçüm sonucu."""

    metrik_adi: str
    deger: float
    olcek_tipi: str  # 'mesafe' (düşük iyi) veya 'benzerlik' (yüksek iyi)


class MesafeOlcer:
    """Vektörler ve görüntüler arasında matematiksel mesafe ve benzerlik hesaplama motoru."""

    @staticmethod
    def _vektorleri_dogrula(
        vektor_a: np.ndarray,
        vektor_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Girdi dizilerini doğrular, float64 tipine çevirir ve 1B vektör haline getirir."""
        if not isinstance(vektor_a, np.ndarray) or not isinstance(vektor_b, np.ndarray):
            raise TypeError("Girdiler NumPy ndarray formatında olmalıdır.")

        if vektor_a.size == 0 or vektor_b.size == 0:
            raise ValueError("Vektörler boş olamaz.")

        a_duz = vektor_a.astype(np.float64).ravel()
        b_duz = vektor_b.astype(np.float64).ravel()

        if a_duz.shape != b_duz.shape:
            raise ValueError(
                f"Vektör boyutları uyuşmuyor: {a_duz.shape} != {b_duz.shape}"
            )

        return a_duz, b_duz

    @classmethod
    def oklid_mesafesi(
        cls,
        vektor_a: np.ndarray,
        vektor_b: np.ndarray
    ) -> float:
        """İki vektör arasındaki Öklid (L2 Normu) mesafesini hesaplar.

        Formül:
            d_2(u, v) = sqrt(sum((u_i - v_i)^2))

        Parametreler:
            vektor_a (np.ndarray): Birinci vektör veya matris.
            vektor_b (np.ndarray): İkinci vektör veya matris.

        Döndürür:
            float: Öklid geometrik mesafesi (>= 0.0).
        """
        a, b = cls._vektorleri_dogrula(vektor_a, vektor_b)
        fark = a - b
        return float(np.sqrt(np.dot(fark, fark)))

    @classmethod
    def manhattan_mesafesi(
        cls,
        vektor_a: np.ndarray,
        vektor_b: np.ndarray
    ) -> float:
        """İki vektör arasındaki Manhattan (Şehir Bloku / L1 Normu) mesafesini hesaplar.

        Formül:
            d_1(u, v) = sum(|u_i - v_i|)

        Parametreler:
            vektor_a (np.ndarray): Birinci vektör.
            vektor_b (np.ndarray): İkinci vektör.

        Döndürür:
            float: Mutlak koordinat farkları toplamı (>= 0.0).
        """
        a, b = cls._vektorleri_dogrula(vektor_a, vektor_b)
        return float(np.sum(np.abs(a - b)))

    @classmethod
    def chebyshev_mesafesi(
        cls,
        vektor_a: np.ndarray,
        vektor_b: np.ndarray
    ) -> float:
        """İki vektör arasındaki Chebyshev (Satranç Tahtası / L-sonsuz Normu) mesafesini hesaplar.

        Formül:
            d_inf(u, v) = max_i(|u_i - v_i|)

        Parametreler:
            vektor_a (np.ndarray): Birinci vektör.
            vektor_b (np.ndarray): İkinci vektör.

        Döndürür:
            float: En büyük tekil boyut farkı (>= 0.0).
        """
        a, b = cls._vektorleri_dogrula(vektor_a, vektor_b)
        return float(np.max(np.abs(a - b)))

    @classmethod
    def kosinus_benzerligi(
        cls,
        vektor_a: np.ndarray,
        vektor_b: np.ndarray,
        epsilon: float = 1e-9
    ) -> float:
        """İki vektör arasındaki açının kosinüsünü (Açısal Benzerlik) hesaplar.

        Büyüklükten (magnitude) bağımsız, sadece vektörlerin yönelimini ölçer.

        Formül:
            S_cos(u, v) = (u . v) / (||u||_2 * ||v||_2 + epsilon)

        Parametreler:
            vektor_a (np.ndarray): Birinci vektör.
            vektor_b (np.ndarray): İkinci vektör.
            epsilon (float): Sıfır norm durumunda payda koruyucu.

        Döndürür:
            float: [-1.0, 1.0] aralığında benzerlik katsayısı (1.0 = özdeş yön).
        """
        a, b = cls._vektorleri_dogrula(vektor_a, vektor_b)
        nokta_carpim = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        payda = (norm_a * norm_b) + epsilon
        benzerlik = nokta_carpim / payda
        # Kayan nokta yuvarlama hatalarını [-1.0, 1.0] sınırına kilitler
        return float(np.clip(benzerlik, -1.0, 1.0))

    @classmethod
    def kosinus_mesafesi(
        cls,
        vektor_a: np.ndarray,
        vektor_b: np.ndarray,
        epsilon: float = 1e-9
    ) -> float:
        """Kosinüs benzerliğini mesafeye dönüştürür.

        Formül:
            D_cos(u, v) = 1.0 - S_cos(u, v)

        Döndürür:
            float: [0.0, 2.0] aralığında mesafe skoru (0.0 = özdeş).
        """
        return 1.0 - cls.kosinus_benzerligi(vektor_a, vektor_b, epsilon)

    @classmethod
    def minkowski_mesafesi(
        cls,
        vektor_a: np.ndarray,
        vektor_b: np.ndarray,
        p_derecesi: float = 3.0
    ) -> float:
        """Genelleştirilmiş L_p normu olan Minkowski mesafesini hesaplar.

        Formül:
            d_p(u, v) = (sum(|u_i - v_i|^p))^(1/p)

        Notlar:
            p=1 -> Manhattan
            p=2 -> Öklid
            p->sonsuz -> Chebyshev

        Parametreler:
            vektor_a (np.ndarray): Birinci vektör.
            vektor_b (np.ndarray): İkinci vektör.
            p_derecesi (float): Norm derecesi (>= 1.0).

        Döndürür:
            float: Minkowski mesafesi.
        """
        if p_derecesi < 1.0:
            raise ValueError(f"Minkowski norm derecesi p >= 1 olmalıdır. Girilen: {p_derecesi}")

        a, b = cls._vektorleri_dogrula(vektor_a, vektor_b)
        mutlak_fark = np.abs(a - b)
        return float(np.sum(mutlak_fark ** p_derecesi) ** (1.0 / p_derecesi))

    @classmethod
    def tum_metrikleri_hesapla(
        cls,
        vektor_a: np.ndarray,
        vektor_b: np.ndarray
    ) -> Dict[str, MesafeSonucu]:
        """Tüm temel mesafe ve benzerlik metriklerini tek seferde hesaplar."""
        return {
            "Oklid_L2": MesafeSonucu("Öklid (L2)", cls.oklid_mesafesi(vektor_a, vektor_b), "mesafe"),
            "Manhattan_L1": MesafeSonucu("Manhattan (L1)", cls.manhattan_mesafesi(vektor_a, vektor_b), "mesafe"),
            "Chebyshev_L_Sonsuz": MesafeSonucu("Chebyshev (L-Sonsuz)", cls.chebyshev_mesafesi(vektor_a, vektor_b), "mesafe"),
            "Minkowski_P3": MesafeSonucu("Minkowski (p=3)", cls.minkowski_mesafesi(vektor_a, vektor_b, p_derecesi=3.0), "mesafe"),
            "Kosinus_Benzerligi": MesafeSonucu("Kosinüs Benzerliği", cls.kosinus_benzerligi(vektor_a, vektor_b), "benzerlik"),
            "Kosinus_Mesafesi": MesafeSonucu("Kosinüs Mesafesi", cls.kosinus_mesafesi(vektor_a, vektor_b), "mesafe"),
        }

    @classmethod
    def piksel_fark_haritasi(
        cls,
        gorsel_a: np.ndarray,
        gorsel_b: np.ndarray,
        metrik: str = "oklid"
    ) -> np.ndarray:
        """İki görsel arasında piksel düzeyinde uzamsal fark haritası (Spatial Heatmap) üretir.

        Parametreler:
            gorsel_a (np.ndarray): (H, W) veya (H, W, C) birinci görsel.
            gorsel_b (np.ndarray): (H, W) veya (H, W, C) ikinci görsel.
            metrik (str): 'oklid', 'manhattan' veya 'chebyshev'.

        Döndürür:
            np.ndarray: (H, W) boyutunda piksel fark şiddeti haritası (float32).
        """
        if gorsel_a.shape != gorsel_b.shape:
            raise ValueError(
                f"Görsel boyutları eşleşmelidir: {gorsel_a.shape} != {gorsel_b.shape}"
            )

        a = gorsel_a.astype(np.float32)
        b = gorsel_b.astype(np.float32)

        if gorsel_a.ndim == 2:
            return np.abs(a - b)

        # 3B Renkli görsellerde kanal ekseninde (axis=-1) indirgeme yapılır
        if metrik == "oklid":
            return np.sqrt(np.sum((a - b) ** 2, axis=-1)).astype(np.float32)
        elif metrik == "manhattan":
            return np.sum(np.abs(a - b), axis=-1).astype(np.float32)
        elif metrik == "chebyshev":
            return np.max(np.abs(a - b), axis=-1).astype(np.float32)
        else:
            raise ValueError(f"Desteklenmeyen piksel metriği: {metrik}")

    @classmethod
    def toplu_oklid_mesafesi(
        cls,
        sorgu_vektoru: np.ndarray,
        veri_kumesi_matrisi: np.ndarray
    ) -> np.ndarray:
        """Tek bir sorgu vektörü ile N adet vektör arasındaki Öklid mesafelerini döngüsüz hesaplar.

        Broadcasting formülü:
            d_i = sqrt(sum((Q - D_i)^2))

        Parametreler:
            sorgu_vektoru (np.ndarray): (D,) boyutunda sorgu özniteliği.
            veri_kumesi_matrisi (np.ndarray): (N, D) boyutunda veri tabanı matrisi.

        Döndürür:
            np.ndarray: (N,) boyutunda Öklid mesafeleri dizisi (float64).
        """
        sorgu = sorgu_vektoru.astype(np.float64).ravel()
        if veri_kumesi_matrisi.ndim != 2 or veri_kumesi_matrisi.shape[1] != sorgu.shape[0]:
            raise ValueError(
                f"Boyut uyuşmazlığı: Sorgu={sorgu.shape}, Veri Kümesi={veri_kumesi_matrisi.shape}"
            )

        farklar = veri_kumesi_matrisi.astype(np.float64) - sorgu
        return np.sqrt(np.sum(farklar ** 2, axis=1))

    @classmethod
    def toplu_kosinus_benzerligi(
        cls,
        sorgu_vektoru: np.ndarray,
        veri_kumesi_matrisi: np.ndarray,
        epsilon: float = 1e-9
    ) -> np.ndarray:
        """Tek bir sorgu vektörü ile N adet veri kümesi vektörünün kosinüs benzerliğini döngüsüz hesaplar.

        Parametreler:
            sorgu_vektoru (np.ndarray): (D,) boyutunda sorgu vektörü.
            veri_kumesi_matrisi (np.ndarray): (N, D) boyutunda veri tabanı matrisi.

        Döndürür:
            np.ndarray: (N,) boyutunda kosinüs benzerlikleri dizisi (float64).
        """
        sorgu = sorgu_vektoru.astype(np.float64).ravel()
        matris = veri_kumesi_matrisi.astype(np.float64)

        nokta_carpimlar = np.dot(matris, sorgu)  # (N,)
        sorgu_normu = np.linalg.norm(sorgu)
        matris_normlari = np.linalg.norm(matris, axis=1)  # (N,)

        payda = (matris_normlari * sorgu_normu) + epsilon
        return np.clip(nokta_carpimlar / payda, -1.0, 1.0)
