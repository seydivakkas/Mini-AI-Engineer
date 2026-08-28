"""
Küme Sayısı Optimizatörü (Elbow Yöntemi ve Silhouette Analizi).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KumeOptimizatoru:
    """Optimal küme sayısı K'yı Elbow (WCSS) ve Silhouette metrikleriyle otomatik belirler."""

    @classmethod
    def en_iyi_k_bul(
        cls,
        ozellik_matrisi: np.ndarray,
        k_araligi: Tuple[int, int] = (2, 8),
        orneklem_boyutu: int = 1500,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """K aralığındaki WCSS ve Silhouette skorlarını hesaplayıp en iyi K'yı seçer."""
        X = np.asarray(ozellik_matrisi, dtype=np.float32)
        n_samples = len(X)

        if n_samples > orneklem_boyutu:
            np.random.seed(random_state)
            idx = np.random.choice(n_samples, orneklem_boyutu, replace=False)
            X_eval = X[idx]
        else:
            X_eval = X

        k_degerleri = list(range(k_araligi[0], k_araligi[1] + 1))
        wcss_list = []
        silhouette_list = []

        for k in k_degerleri:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=5, max_iter=200)
            labels = km.fit_predict(X_eval)

            wcss_list.append(float(km.inertia_))
            sil = float(silhouette_score(X_eval, labels))
            silhouette_list.append(round(sil, 4))

        # En yüksek silhouette skoruna sahip K seçilir
        en_iyi_k_idx = int(np.argmax(silhouette_list))
        en_iyi_k = k_degerleri[en_iyi_k_idx]

        return {
            "k_degerleri": k_degerleri,
            "wcss_degerleri": [round(w, 2) for w in wcss_list],
            "silhouette_degerleri": silhouette_list,
            "en_iyi_k": en_iyi_k,
            "en_iyi_silhouette": silhouette_list[en_iyi_k_idx]
        }
