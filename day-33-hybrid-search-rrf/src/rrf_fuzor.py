"""
Reciprocal Rank Fusion (RRF) ve Skor Normalizasyon Füzyon Motoru.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict


class RRFFuzor:
    """
    Reciprocal Rank Fusion (RRF - Cormack et al., SIGIR 2009) Algoritması.
    Skor büyüklüklerinden bağımsız olarak dokümanların sıralama (rank) derecelerini birleştirir:

    RRF_Score(d) = sum_{m in Modeller} [ w_m / (k + rank_m(d)) ]
    """

    def __init__(self, k: int = 60):
        self.k = int(k)

    def birlestir(
        self,
        sonuc_listeleri: Dict[str, List[Dict[str, Any]]],
        agirliklar: Optional[Dict[str, float]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Farklı arama motorlarından (ör. 'bm25', 'dense') gelen sıralı sonuç listelerini RRF ile birleştirir.
        """
        if agirliklar is None:
            # Eşit ağırlık
            n_mod = max(len(sonuc_listeleri), 1)
            agirliklar = {ad: 1.0 / n_mod for ad in sonuc_listeleri}

        rrf_skorlari: Dict[str, float] = defaultdict(float)
        dokuman_detaylari: Dict[str, Dict[str, Any]] = {}
        siralama_gecmisi: Dict[str, Dict[str, int]] = defaultdict(dict)
        orijinal_skorlar: Dict[str, Dict[str, float]] = defaultdict(dict)

        for model_adi, sonuclar in sonuc_listeleri.items():
            w = agirliklar.get(model_adi, 1.0)
            for sira_0, doc in enumerate(sonuclar):
                doc_id = doc["doc_id"]
                rank = sira_0 + 1  # 1-indexed rank

                # RRF Katkısı: w / (k + rank)
                katki = w / float(self.k + rank)
                rrf_skorlari[doc_id] += katki

                siralama_gecmisi[doc_id][model_adi] = rank
                orijinal_skorlar[doc_id][model_adi] = doc.get("skor", 0.0)

                if doc_id not in dokuman_detaylari:
                    dokuman_detaylari[doc_id] = {
                        "baslik": doc.get("baslik", ""),
                        "icerik": doc.get("icerik", ""),
                        "metaveri": doc.get("metaveri", {})
                    }

        birlesik_sonuclar = []
        for doc_id, final_skor in rrf_skorlari.items():
            birlesik_sonuclar.append({
                "doc_id": doc_id,
                "skor": float(final_skor),
                "baslik": dokuman_detaylari[doc_id]["baslik"],
                "icerik": dokuman_detaylari[doc_id]["icerik"],
                "metaveri": dokuman_detaylari[doc_id]["metaveri"],
                "siralama_gecmisi": siralama_gecmisi[doc_id],
                "orijinal_skorlar": orijinal_skorlar[doc_id]
            })

        birlesik_sonuclar.sort(key=lambda x: x["skor"], reverse=True)
        return birlesik_sonuclar[:top_k]


class PuanNormalizasyonFuzor:
    """
    Min-Max Skor Normalizasyonu ile Ağırlıklı Doğrusal Skor Füzyonu (Weighted Linear Score Fusion):
    Score_norm(d) = alpha * norm_bm25(d) + (1 - alpha) * norm_dense(d)
    """

    @classmethod
    def normalize_ve_birlestir(
        cls,
        sonuc_listeleri: Dict[str, List[Dict[str, Any]]],
        agirliklar: Optional[Dict[str, float]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if agirliklar is None:
            n_mod = max(len(sonuc_listeleri), 1)
            agirliklar = {ad: 1.0 / n_mod for ad in sonuc_listeleri}

        # Model bazında min-max skorları bul
        norm_katkilar: Dict[str, float] = defaultdict(float)
        dokuman_detaylari: Dict[str, Dict[str, Any]] = {}
        model_norm_skorlari: Dict[str, Dict[str, float]] = defaultdict(dict)

        for model_adi, sonuclar in sonuc_listeleri.items():
            if not sonuclar:
                continue
            skorlar = [d["skor"] for d in sonuclar]
            min_s, max_s = min(skorlar), max(skorlar)
            fark = max_s - min_s if max_s > min_s else 1.0
            w = agirliklar.get(model_adi, 1.0)

            for d in sonuclar:
                doc_id = d["doc_id"]
                norm_skor = (d["skor"] - min_s) / fark
                norm_katkilar[doc_id] += w * norm_skor
                model_norm_skorlari[doc_id][model_adi] = float(norm_skor)

                if doc_id not in dokuman_detaylari:
                    dokuman_detaylari[doc_id] = {
                        "baslik": d.get("baslik", ""),
                        "icerik": d.get("icerik", ""),
                        "metaveri": d.get("metaveri", {})
                    }

        sonuclar = []
        for doc_id, skor in norm_katkilar.items():
            sonuclar.append({
                "doc_id": doc_id,
                "skor": float(skor),
                "baslik": dokuman_detaylari[doc_id]["baslik"],
                "icerik": dokuman_detaylari[doc_id]["icerik"],
                "metaveri": dokuman_detaylari[doc_id]["metaveri"],
                "norm_skorlar": model_norm_skorlari[doc_id]
            })

        sonuclar.sort(key=lambda x: x["skor"], reverse=True)
        return sonuclar[:top_k]
