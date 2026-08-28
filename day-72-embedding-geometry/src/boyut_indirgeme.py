"""
Boyut İndirgeme Motoru (PCA, t-SNE, UMAP)
----------------------------------------
Yüksek boyutlu temsil uzaylarını 2D/3D manifoldlara izdüşüren, yerel ve küresel
geometrik ilişkileri koruyan modüler boyut indirgeme motoru.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class BoyutIndirgemeMotoru:
    """
    PCA, t-SNE ve UMAP algoritmalarını tek bir endüstriyel arayüzde birleştiren motor.
    """
    def __init__(self, rastgele_tohum: int = 42):
        self.rastgele_tohum = rastgele_tohum

    def uygula_pca(
        self,
        X: np.ndarray,
        bilesen_sayisi: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lineer Boyut İndirgeme (Principal Component Analysis).
        Maksimum varyans yönlerini bulur; küresel mesafeleri korur ancak yerel manifoldları bükebilir.
        
        Döndürür: (X_2d, aciklanan_varyans_oranlari)
        """
        pca = PCA(n_components=bilesen_sayisi, random_state=self.rastgele_tohum)
        X_pca = pca.fit_transform(X)
        aciklanan_varyans = pca.explained_variance_ratio_
        return X_pca, aciklanan_varyans

    def uygula_tsne(
        self,
        X: np.ndarray,
        bilesen_sayisi: int = 2,
        perplexity: float = 30.0,
        ogrenme_orani: Any = "auto",
        iterasyon_sayisi: int = 1000
    ) -> Tuple[np.ndarray, float]:
        """
        Non-Lineer Manifold İzdüşümü (t-Distributed Stochastic Neighbor Embedding).
        Student-t dağılımı ve KL-Diverjansı ile yerel komşulukları mükemmel kümelendirir.
        
        Döndürür: (X_tsne, kl_diverjans_kaybi)
        """
        tsne = TSNE(
            n_components=bilesen_sayisi,
            perplexity=perplexity,
            learning_rate=ogrenme_orani,
            max_iter=iterasyon_sayisi,
            random_state=self.rastgele_tohum,
            init="pca"
        )
        X_tsne = tsne.fit_transform(X)
        kl_kayip = float(tsne.kl_divergence_)
        return X_tsne, kl_kayip

    def uygula_umap(
        self,
        X: np.ndarray,
        bilesen_sayisi: int = 2,
        komsu_sayisi: int = 15,
        min_mesafe: float = 0.1,
        metrik: str = "cosine"
    ) -> np.ndarray:
        """
        Uniform Manifold Approximation and Projection (UMAP).
        Bulanık Basit Kümeler (Fuzzy Simplicial Sets) ve Riemann Geometrisi ile
        hem YEREL kümeleri hem de KÜRESEL topolojiyi t-SNE'den çok daha hızlı korur.
        
        Döndürür: X_umap (2D izdüşüm matrisi)
        """
        indirgeyici = umap.UMAP(
            n_components=bilesen_sayisi,
            n_neighbors=komsu_sayisi,
            min_dist=min_mesafe,
            metric=metrik,
            random_state=self.rastgele_tohum
        )
        X_umap = indirgeyici.fit_transform(X)
        return X_umap

    def karsilastirmali_indirgeme(
        self,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """Tüm algoritmaları çalıştırıp toplu sonuç sözlüğü üretir."""
        X_pca, varyans = self.uygula_pca(X, bilesen_sayisi=2)
        X_tsne, kl_kayip = self.uygula_tsne(X, bilesen_sayisi=2)
        X_umap = self.uygula_umap(X, bilesen_sayisi=2)
        
        return {
            "PCA": {
                "izdusum": X_pca,
                "toplam_varyans": float(np.sum(varyans)),
                "bilesen_varyanslari": varyans.tolist()
            },
            "t-SNE": {
                "izdusum": X_tsne,
                "kl_diverjans": kl_kayip
            },
            "UMAP": {
                "izdusum": X_umap
            }
        }
