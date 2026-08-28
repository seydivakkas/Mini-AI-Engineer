"""Görsel Kümeleme Motoru Modülü.

Etiketsiz görsel embedding vektörleri üzerinde K-Means, DBSCAN ve
Agglomerative (Hiyerarşik) kümeleme algoritmalarını koşturur ve
Silhouette, Davies-Bouldin ve Calinski-Harabasz metrikleriyle değerlendirir.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


@dataclass
class KumelemeSonucu:
    """Bir kümeleme algoritmasının çıktı ve değerlendirme metrikleri."""

    algoritma: str
    etiketler: np.ndarray
    kume_sayisi: int
    gurultu_sayisi: int
    silhouette: Optional[float] = None
    davies_bouldin: Optional[float] = None
    calinski_harabasz: Optional[float] = None

    def ozet(self) -> str:
        """Sonucun tek satırlık anlaşılır özetini döndürür."""
        sil_str = f"{self.silhouette:.4f}" if self.silhouette is not None else "N/A"
        db_str = f"{self.davies_bouldin:.4f}" if self.davies_bouldin is not None else "N/A"
        ch_str = f"{self.calinski_harabasz:.2f}" if self.calinski_harabasz is not None else "N/A"
        return (
            f"[{self.algoritma}] Kümeler: {self.kume_sayisi} | "
            f"Gürültü: {self.gurultu_sayisi} | "
            f"Silhouette: {sil_str} | Davies-Bouldin: {db_str} | Calinski-Harabasz: {ch_str}"
        )


class GorselKumelemeMotoru:
    """Görsel öznitelik vektörlerini denetimsiz olarak kümeleyen ve değerlendiren motor."""

    def __init__(self, random_state: int = 42) -> None:
        """Motoru ilklendirir."""
        self.random_state = random_state

    @staticmethod
    def _metrikleri_hesapla(
        X: np.ndarray, etiketler: np.ndarray
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Kümeleme kalitesi metriklerini hesaplar.
        
        Silhouette skoru için en az 2 farklı küme ve gürültü dışı en az 2 örnek gerekir.
        """
        # DBSCAN'da -1 gürültü etiketidir. Metrikleri hesaplarken gürültü dışı noktaları filtrele
        gecerli_maske = etiketler != -1
        gecerli_etiketler = etiketler[gecerli_maske]
        gecerli_X = X[gecerli_maske]

        benzersiz_kumeler = set(gecerli_etiketler)
        if len(benzersiz_kumeler) < 2 or len(gecerli_X) <= len(benzersiz_kumeler):
            return None, None, None

        sil = float(silhouette_score(gecerli_X, gecerli_etiketler, metric="cosine"))
        db = float(davies_bouldin_score(gecerli_X, gecerli_etiketler))
        ch = float(calinski_harabasz_score(gecerli_X, gecerli_etiketler))
        return sil, db, ch

    def k_means_kumele(
        self, X: np.ndarray, n_clusters: int = 4, n_init: int = 10
    ) -> KumelemeSonucu:
        """K-Means algoritması ile kümeleme yapar."""
        if len(X) < n_clusters:
            raise ValueError(
                f"Örnek sayısı ({len(X)}) küme sayısından ({n_clusters}) küçük olamaz!"
            )

        model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            random_state=self.random_state,
        )
        etiketler = model.fit_predict(X)
        sil, db, ch = self._metrikleri_hesapla(X, etiketler)

        return KumelemeSonucu(
            algoritma=f"K-Means (k={n_clusters})",
            etiketler=etiketler,
            kume_sayisi=n_clusters,
            gurultu_sayisi=0,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
        )

    def dbscan_kumele(
        self,
        X: np.ndarray,
        eps: float = 0.35,
        min_samples: int = 2,
        metric: str = "cosine",
    ) -> KumelemeSonucu:
        """DBSCAN yoğunluk tabanlı kümeleme algoritması uygular.
        
        Görsel embedding'lerinde kosinüs mesafesi birim kürede 0 ile 2 arasındadır.
        """
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        etiketler = model.fit_predict(X)

        benzersiz_etiketler = set(etiketler)
        gurultu_sayisi = int(np.sum(etiketler == -1))
        kume_sayisi = len(benzersiz_etiketler - {-1})

        sil, db, ch = self._metrikleri_hesapla(X, etiketler)

        return KumelemeSonucu(
            algoritma=f"DBSCAN (eps={eps}, min_pts={min_samples})",
            etiketler=etiketler,
            kume_sayisi=kume_sayisi,
            gurultu_sayisi=gurultu_sayisi,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
        )

    def agglomerative_kumele(
        self,
        X: np.ndarray,
        n_clusters: int = 4,
        metric: str = "cosine",
        linkage: str = "average",
    ) -> KumelemeSonucu:
        """Hiyerarşik (Agglomerative) kümeleme uygular."""
        if len(X) < n_clusters:
            raise ValueError(
                f"Örnek sayısı ({len(X)}) küme sayısından ({n_clusters}) küçük olamaz!"
            )

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            linkage=linkage,
        )
        etiketler = model.fit_predict(X)
        sil, db, ch = self._metrikleri_hesapla(X, etiketler)

        return KumelemeSonucu(
            algoritma=f"Agglomerative (k={n_clusters}, {linkage})",
            etiketler=etiketler,
            kume_sayisi=n_clusters,
            gurultu_sayisi=0,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
        )

    def en_iyi_k_bul_kmeans(
        self, X: np.ndarray, k_araligi: range = range(2, 7)
    ) -> Tuple[int, Dict[int, float], KumelemeSonucu]:
        """Farklı K değerleri için Silhouette skorunu hesaplar ve en yüksek skora sahip K'yı seçer."""
        skorlar: Dict[int, float] = {}
        en_iyi_k = list(k_araligi)[0]
        en_yuksek_skor = -1.0
        en_iyi_sonuc: Optional[KumelemeSonucu] = None

        for k in k_araligi:
            if k >= len(X):
                continue
            sonuc = self.k_means_kumele(X, n_clusters=k)
            skor = sonuc.silhouette if sonuc.silhouette is not None else -1.0
            skorlar[k] = skor
            if skor > en_yuksek_skor:
                en_yuksek_skor = skor
                en_iyi_k = k
                en_iyi_sonuc = sonuc

        if en_iyi_sonuc is None:
            raise RuntimeError("Geçerli bir K değeri bulunamadı!")

        return en_iyi_k, skorlar, en_iyi_sonuc
