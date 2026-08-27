"""Mahalanobis Tabanlı Çok Değişkenli Anomali ve Aykırı Değer Dedektörü.

Bu modül; çok değişkenli normal dağılım varsayımı altında Mahalanobis mesafesinin
karesinin (D_M^2) Ki-Kare (Chi-Square, chi2) dağılımına uyması ilkesini kullanarak
istatistiksel anomali ve kusur tespiti yapar.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats
from src.kovaryans_ve_mesafe import MahalanobisMesafeOlcer


@dataclass(frozen=True)
class AnomaliTahmini:
    """Tek bir örneğe ait anomali tespit kararı ve istatistiksel olasılık."""

    ornek_indeksi: int
    mahalanobis_mesafesi: float
    esik_degeri: float
    p_degeri: float
    anomali_mi: bool


class MahalanobisAnomaliDedektoru:
    """Çok değişkenli endüstriyel ve görsel veriler için istatistiksel anomali tespit edici."""

    def __init__(
        self,
        anlamlilik_duzeyi: float = 0.01,
        duzenleme_katsayisi: float = 1e-6
    ) -> None:
        """Dedektörü başlatır.

        Parametreler:
            anlamlilik_duzeyi (float): Yanlış alarm (Type I hata) toleransı alpha (örn: 0.01 = %99 güven).
            duzenleme_katsayisi (float): Ters kovaryans Tikhonov regülarizasyon katsayısı.
        """
        if not (0.0 < anlamlilik_duzeyi < 1.0):
            raise ValueError(f"Anlamlılık düzeyi (0, 1) aralığında olmalıdır. Girilen: {anlamlilik_duzeyi}")

        self.alpha = anlamlilik_duzeyi
        self.duzenleme = duzenleme_katsayisi
        self._olcer: Optional[MahalanobisMesafeOlcer] = None
        self.esik_mesafe: float = 0.0
        self.serbestlik_derecesi: int = 0

    def egit(self, normal_veri_matrisi: np.ndarray) -> None:
        """Yalnızca 'normal' (kusursuz) örnekler üzerinden dağılım parametrelerini öğrenir."""
        if not isinstance(normal_veri_matrisi, np.ndarray) or normal_veri_matrisi.ndim != 2:
            raise ValueError("Eğitim verisi (N, D) boyutunda 2B ndarray olmalıdır.")

        self._olcer = MahalanobisMesafeOlcer(
            referans_verisi=normal_veri_matrisi,
            duzenleme_katsayisi=self.duzenleme
        )
        self.serbestlik_derecesi = normal_veri_matrisi.shape[1]

        # Mahalanobis mesafesinin karesi, D serbestlik dereceli Ki-Kare dağılımına uyar
        # Kritik Ki-kare değeri: P(Chi2 > esik_kare) = alpha
        esik_kare = float(stats.chi2.ppf(1.0 - self.alpha, df=self.serbestlik_derecesi))
        self.esik_mesafe = float(np.sqrt(esik_kare))

    def tahmin_et(self, test_verisi: np.ndarray) -> List[AnomaliTahmini]:
        """Verilen test örneklerini anomali olup olmadıklarına göre sınıflandırır."""
        if self._olcer is None:
            raise RuntimeError("Model henüz eğitilmedi. Önce egit() fonksiyonunu çağırınız.")

        matris = test_verisi.astype(np.float64)
        if matris.ndim == 1:
            matris = np.expand_dims(matris, axis=0)

        mesafeler = self._olcer.toplu_mahalanobis_mesafesi(matris)
        kare_mesafeler = mesafeler ** 2

        # p-değeri (survival function: 1 - CDF)
        p_degerleri = stats.chi2.sf(kare_mesafeler, df=self.serbestlik_derecesi)

        tahminler: List[AnomaliTahmini] = []
        for i, (mesafe, p_val) in enumerate(zip(mesafeler, p_degerleri)):
            anomali = bool(mesafe > self.esik_mesafe)
            tahminler.append(
                AnomaliTahmini(
                    ornek_indeksi=i,
                    mahalanobis_mesafesi=round(float(mesafe), 4),
                    esik_degeri=round(self.esik_mesafe, 4),
                    p_degeri=round(float(p_val), 6),
                    anomali_mi=anomali
                )
            )

        return tahminler
