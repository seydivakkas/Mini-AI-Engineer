"""
Sıfırdan Matematiksel TPE (Tree-structured Parzen Estimator) ve Budama Motoru
-----------------------------------------------------------------------------
Bergstra et al. (NeurIPS 2011) TPE formülasyonu:
l(x) / g(x) olasılık yoğunluk oranı maksimizasyonu ve Medyan Budama (Median Pruning).

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.stats import gaussian_kde


class MatematikselTPESampler:
    """
    Sıfırdan Gauss Çekirdek Yoğunluk Tahmini (KDE) tabanlı TPE Örnekleyicisi.
    """
    def __init__(self, gama: float = 0.25, aday_sayisi: int = 64, tohum: int = 42):
        self.gama = gama
        self.aday_sayisi = aday_sayisi
        self.rng = np.random.RandomState(tohum)

    def ornekle(
        self,
        gecmis_x: List[float],
        gecmis_y: List[float],
        alt_sinir: float,
        ust_sinir: float,
        log_olcek: bool = False
    ) -> float:
        """
        Geçmiş deneme sonuçlarına göre l(x)/g(x) oranını maksimize eden yeni hiperparametre değeri önerir.
        """
        # Eğer yeterli gözlem yoksa (ör. < 5) düzgün rastgele örnekle
        if len(gecmis_x) < 5:
            if log_olcek:
                return float(np.exp(self.rng.uniform(np.log(alt_sinir), np.log(ust_sinir))))
            return float(self.rng.uniform(alt_sinir, ust_sinir))

        x_arr = np.array(gecmis_x)
        y_arr = np.array(gecmis_y)

        if log_olcek:
            x_arr = np.log(x_arr)
            alt = np.log(alt_sinir)
            ust = np.log(ust_sinir)
        else:
            alt, ust = alt_sinir, ust_sinir

        # Quantile eşik değeri (y*): Minimizasyon için en iyi gama yüzdeliği
        esik_y = np.percentile(y_arr, self.gama * 100.0)

        iyi_maske = y_arr <= esik_y
        kotu_maske = y_arr > esik_y

        # En az 2 nokta gereklidir (KDE kovaryansı için)
        if np.sum(iyi_maske) < 2 or np.sum(kotu_maske) < 2:
            rastgele_val = self.rng.uniform(alt, ust)
            return float(np.exp(rastgele_val)) if log_olcek else float(rastgele_val)

        x_iyi = x_arr[iyi_maske]
        x_kotu = x_arr[kotu_maske]

        # KDE (Parzen Window Density Estimator)
        try:
            kde_iyi = gaussian_kde(x_iyi, bw_method="scott")
            kde_kotu = gaussian_kde(x_kotu, bw_method="scott")
        except Exception:
            rastgele_val = self.rng.uniform(alt, ust)
            return float(np.exp(rastgele_val)) if log_olcek else float(rastgele_val)

        # l(x) dağılımından adaylar üret ve sınırla
        adaylar = kde_iyi.resample(self.aday_sayisi, seed=self.rng.randint(0, 100000))[0]
        adaylar = np.clip(adaylar, alt, ust)

        # l(x) ve g(x) yoğunluklarını hesapla
        l_x = kde_iyi.evaluate(adaylar) + 1e-9
        g_x = kde_kotu.evaluate(adaylar) + 1e-9

        # Beklenen İyileşme (Expected Improvement) Oranı: l(x) / g(x)
        oranlar = l_x / g_x
        en_iyi_aday = adaylar[np.argmax(oranlar)]

        if log_olcek:
            return float(np.exp(en_iyi_aday))
        return float(en_iyi_aday)


class MedyanBudayici:
    """
    Önceki tamamlanmış denemelerin medyan başarımının altında kalan koşuları erken durduran motor.
    """
    def __init__(self, baslangic_adimi: int = 2):
        self.baslangic_adimi = baslangic_adimi
        self.adim_gecmisi: Dict[int, List[float]] = {}

    def adim_raporla(self, step: int, deger: float) -> None:
        if step not in self.adim_gecmisi:
            self.adim_gecmisi[step] = []
        self.adim_gecmisi[step].append(float(deger))

    def budanmali_mi(self, step: int, guncel_deger: float, mod: str = "min") -> bool:
        """
        Minimizasyonda medyanın üstündeyse veya Maksimizasyonda medyanın altındaysa buda (Prune).
        """
        if step < self.baslangic_adimi:
            return False

        if step not in self.adim_gecmisi or len(self.adim_gecmisi[step]) < 2:
            return False

        medyan = float(np.median(self.adim_gecmisi[step]))

        if mod == "min":
            # Kayıp minimizasyonu: Güncel kayıp medyandan yüksekse buda
            return guncel_deger > medyan
        else:
            # Doğruluk maksimizasyonu: Güncel doğruluk medyandan düşükse buda
            return guncel_deger < medyan
