"""
Temsil Uzayı Geometrisi ve İzotropi Analizörü
--------------------------------------------
Bu modül; gömme (embedding) uzaylarının izotropi derecesini, boyutsal çöküş
(dimensional collapse) riskini, kosinüs benzerlik dağılımlarını ve SVD spektrumunu
ölçen matematiksel analiz motorunu içerir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class TemsilGeometrisiAnalizoru:
    """
    Yüksek boyutlu temsil uzaylarının geometrik kalitesini ve izotropisini denetleyen sınıf.
    """

    @staticmethod
    def hesapla_izotropi(X: np.ndarray) -> Dict[str, Any]:
        """
        Gömme vektörlerinin uzaydaki izotropi derecesini SVD ve Entropi ile hesaplar.
        
        Matematiksel Formül:
        ---------------------
        1. X_ort = X - mean(X) (Merkezlenmiş Temsiller)
        2. U, S, Vt = SVD(X_ort)
        3. p_i = S_i^2 / sum(S_j^2) (Varyans Enerji Olasılığı)
        4. H = -sum(p_i * log(p_i)) (Tekil Değer Entropisi)
        5. Izotropi = exp(H) / d (0 ile 1 arasında normalize indeks)
        """
        N, d = X.shape
        X_merkezli = X - np.mean(X, axis=0, keepdims=True)
        
        # Tekil Değer Ayrışımı (SVD)
        _, S, _ = np.linalg.svd(X_merkezli, full_matrices=False)
        
        # Varyans enerjisi ve olasılık dağılımı
        enerjiler = S ** 2
        toplam_enerji = np.sum(enerjiler) + 1e-12
        p = enerjiler / toplam_enerji
        
        # Entropi hesabı (Sıfır olasılıklar için güvenli)
        p_guvenli = p[p > 1e-12]
        entropi = -np.sum(p_guvenli * np.log(p_guvenli))
        
        # Normalize İzotropi Skoru: exp(H) / d
        izotropi_skoru = float(np.exp(entropi) / d)
        
        # Min/Max Tekil Değer Oranı (Koşul sayısı tersi)
        min_max_orani = float(S[-1] / (S[0] + 1e-12))
        
        # Kümülatif varyans
        kumulatif_varyans = np.cumsum(enerjiler) / toplam_enerji
        
        return {
            "izotropi_skoru": izotropi_skoru,
            "min_max_tekil_orani": min_max_orani,
            "tekil_deger_entropisi": float(entropi),
            "tekil_degerler": S.tolist(),
            "kumulatif_varyans": kumulatif_varyans.tolist(),
            "efektif_boyut": float(np.exp(entropi)),
            "toplam_boyut": d
        }

    @staticmethod
    def hesapla_kosinus_geometrisi(
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Sınıf içi ve sınıflar arası kosinüs benzerlik dağılımlarını hesaplar.
        """
        # L2 Normalizasyon (Birim norm)
        normlar = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X_norm = X / normlar
        
        # Çiftler arası kosinüs benzerlik matrisi: S_ij = e_i . e_j
        benzerlik_matrisi = np.dot(X_norm, X_norm.T)
        N = X.shape[0]
        
        # Köşegen hariç maskeler
        ayni_sinif_maskesi = (y[:, None] == y[None, :]) & (~np.eye(N, dtype=bool))
        farkli_sinif_maskesi = (y[:, None] != y[None, :])
        
        sinif_ici_benzerlikler = benzerlik_matrisi[ayni_sinif_maskesi]
        siniflar_arasi_benzerlikler = benzerlik_matrisi[farkli_sinif_maskesi]
        
        sinif_ici_ortalama = float(np.mean(sinif_ici_benzerlikler)) if len(sinif_ici_benzerlikler) > 0 else 0.0
        siniflar_arasi_ortalama = float(np.mean(siniflar_arasi_benzerlikler)) if len(siniflar_arasi_benzerlikler) > 0 else 0.0
        
        # Ayrışma Marjini (Separation Margin)
        ayrisma_marjini = sinif_ici_ortalama - siniflar_arasi_ortalama
        
        return {
            "sinif_ici_ortalama_kosinus": sinif_ici_ortalama,
            "siniflar_arasi_ortalama_kosinus": siniflar_arasi_ortalama,
            "ayrisma_marjini": ayrisma_marjini,
            "sinif_ici_ornekler": sinif_ici_benzerlikler[:200].tolist(),
            "siniflar_arasi_ornekler": siniflar_arasi_benzerlikler[:200].tolist()
        }

    @staticmethod
    def teshis_boyutsal_cokus(
        X: np.ndarray,
        esik_varyans: float = 0.90,
        ilk_k: int = 3
    ) -> Dict[str, Any]:
        """
        Temsil uzayında Boyutsal Çöküş (Dimensional Collapse / Anisotropy Cone) olup olmadığını denetler.
        Eğer ilk k adet tekil değer varyansın %90'ından fazlasını açıklıyorsa çöküş teşhisi konur.
        """
        N, d = X.shape
        X_merkezli = X - np.mean(X, axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(X_merkezli, full_matrices=False)
        
        enerjiler = S ** 2
        toplam_enerji = np.sum(enerjiler) + 1e-12
        kumulatif = np.cumsum(enerjiler) / toplam_enerji
        
        ilk_k_aciklanan_varyans = float(kumulatif[min(ilk_k - 1, len(kumulatif) - 1)])
        cokus_var = ilk_k_aciklanan_varyans >= esik_varyans and d > ilk_k
        
        durum_aciklamasi = (
            f"TEHLİKE: Boyutsal Çöküş Tespit Edildi! İlk {ilk_k} eksen varyansın %{ilk_k_aciklanan_varyans*100:.1f}'ini yutuyor."
            if cokus_var else
            f"SAĞLIKLI: Temsil uzayı boyutlara dengeli yayılmış (İlk {ilk_k} eksen: %{ilk_k_aciklanan_varyans*100:.1f})."
        )
        
        return {
            "cokus_tespit_edildi": bool(cokus_var),
            "ilk_k_aciklanan_varyans": ilk_k_aciklanan_varyans,
            "esik": esik_varyans,
            "teshis_mesaji": durum_aciklamasi
        }
