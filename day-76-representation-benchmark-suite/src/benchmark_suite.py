"""
Temsil Kalitesi Değerlendirme Paketi (Representation Benchmark Suite)
---------------------------------------------------------------------
Linear Probing, k-NN Protokolü, Few-Shot Veri Verimliliği ve Geometrik
Manifold Kalite metriklerini tek bir kurumsal raporda birleştiren süit.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score

from .linear_probe import LinearProbeProtokolu
from .knn_degerlendirici import KNNDegerlendirici


class TemsilDegerlendirmePaketi:
    """
    Temsil kalitesini tüm endüstri standartlarında değerlendiren ana orkestratör.
    """
    def __init__(
        self,
        temsil_boyutu: int = 64,
        sinif_sayisi: int = 6,
        knn_sicaklik: float = 0.07,
        cihaz: str = "cpu"
    ):
        self.temsil_boyutu = temsil_boyutu
        self.sinif_sayisi = sinif_sayisi
        self.cihaz = cihaz
        
        self.linear_probe = LinearProbeProtokolu(
            temsil_boyutu=temsil_boyutu,
            sinif_sayisi=sinif_sayisi,
            ogrenme_orani=1e-2,
            cihaz=cihaz
        )
        self.knn_eval = KNNDegerlendirici(sicaklik=knn_sicaklik)

    def calistir_kapsamli_benchmark(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Tüm benchmark protokollerini (Linear Probe %100, %10, %1, k-NN, Geometrik Metrikler) yürütür.
        """
        sonuclar = {}

        # 1. Full Linear Probing (%100 Etiket)
        lp_100 = self.linear_probe.egit_ve_degerlendir(
            x_train, y_train, x_val, y_val, etiket_orani=1.0, epoch_sayisi=15
        )
        sonuclar["linear_probe_100"] = lp_100["dogruluk_yuzdesi"]

        # 2. Few-Shot Linear Probing (%10 ve %1 Etiket)
        lp_10 = self.linear_probe.egit_ve_degerlendir(
            x_train, y_train, x_val, y_val, etiket_orani=0.10, epoch_sayisi=15
        )
        sonuclar["linear_probe_10"] = lp_10["dogruluk_yuzdesi"]

        lp_1 = self.linear_probe.egit_ve_degerlendir(
            x_train, y_train, x_val, y_val, etiket_orani=0.02, epoch_sayisi=15
        )
        sonuclar["linear_probe_fewshot"] = lp_1["dogruluk_yuzdesi"]

        # 3. Non-Parametrik k-NN Değerlendirmesi
        knn_sonuclari = self.knn_eval.degerlendir(
            x_train, y_train, x_val, y_val, k_degerleri=[1, 5, 10, 20], sinif_sayisi=self.sinif_sayisi
        )
        sonuclar.update(knn_sonuclari)

        # 4. Geometrik Manifold Metrikleri
        geometri = self.hesapla_geometrik_metrikler(x_val, y_val)
        sonuclar.update(geometri)

        return sonuclar

    def hesapla_geometrik_metrikler(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Silhouette Skoru, İzotropi İndeksi, Efektif Boyut (SVD Entropisi) ve Kosinüs Marjini hesaplar.
        """
        x_norm = F.normalize(x, p=2, dim=1).numpy()
        y_np = y.numpy()

        # Silhouette Skoru
        try:
            sil = float(silhouette_score(x_norm, y_np))
        except Exception:
            sil = 0.0

        # SVD ve İzotropi
        U, S, Vh = np.linalg.svd(x_norm - x_norm.mean(axis=0), full_matrices=False)
        varyans_oranlari = (S ** 2) / ((S ** 2).sum() + 1e-12)
        
        # Shannon Entropisi -> Efektif Boyut
        shannon_entropi = -np.sum(varyans_oranlari * np.log(varyans_oranlari + 1e-12))
        efektif_boyut = float(np.exp(shannon_entropi))

        # İzotropi: min_singular / max_singular
        izotropi = float(S[-1] / (S[0] + 1e-12))

        # Sınıf içi vs sınıflar arası kosinüs marjini
        sim_matrisi = np.dot(x_norm, x_norm.T)
        N = x_norm.shape[0]
        y_kolon = y_np.reshape(-1, 1)
        ayni_sinif = (y_kolon == y_kolon.T) & (~np.eye(N, dtype=bool))
        farkli_sinif = (y_kolon != y_kolon.T)

        sinif_ici_ort = float(sim_matrisi[ayni_sinif].mean()) if ayni_sinif.any() else 0.0
        siniflar_arasi_ort = float(sim_matrisi[farkli_sinif].mean()) if farkli_sinif.any() else 0.0

        return {
            "silhouette_skoru": sil,
            "efektif_boyut": efektif_boyut,
            "izotropi_indeksi": izotropi,
            "sinif_ici_kosinus": sinif_ici_ort,
            "siniflar_arasi_kosinus": siniflar_arasi_ort,
            "ayrisma_marjini": sinif_ici_ort - siniflar_arasi_ort
        }
