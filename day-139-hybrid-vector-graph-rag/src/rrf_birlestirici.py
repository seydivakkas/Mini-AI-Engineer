"""
Reciprocal Rank Fusion (RRF) Birleştirici Modülü (Day 139 - Faz 7).
Vektör ve graf sıralama listelerini RRF formülüyle harmanlayan hibrit füzyon motoru.
"""

from typing import List, Dict, Any


class RRFBirlestirici:
    """Vektör ve Graf getirme sıralarını Reciprocal Rank Fusion ile birleştiren modül."""

    @classmethod
    def birlestir(
        cls,
        vektor_sonuclari: List[Dict[str, Any]],
        graf_sonuclari: List[Dict[str, Any]],
        w_vec: float = 0.5,
        w_graph: float = 0.5,
        k_rrf: int = 60,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        RRF Formülü: RRF(d) = w_v / (k + rank_v) + w_g / (k + rank_g)
        """
        belge_havuzu: Dict[str, Dict[str, Any]] = {}
        rrf_skorlari: Dict[str, float] = {}

        # 1. Vektör Sıralarını Ekle
        for item in vektor_sonuclari:
            doc_id = item["id"]
            rank_v = item["vektor_sirasi"]
            rrf_skorlari[doc_id] = rrf_skorlari.get(doc_id, 0.0) + (w_vec / (k_rrf + rank_v))
            if doc_id not in belge_havuzu:
                belge_havuzu[doc_id] = item.copy()
            belge_havuzu[doc_id]["vektor_sirasi"] = rank_v
            belge_havuzu[doc_id]["vektor_skoru"] = item.get("vektor_skoru", 0.0)

        # 2. Graf Sıralarını Ekle
        for item in graf_sonuclari:
            doc_id = item["id"]
            rank_g = item["graf_sirasi"]
            rrf_skorlari[doc_id] = rrf_skorlari.get(doc_id, 0.0) + (w_graph / (k_rrf + rank_g))
            if doc_id not in belge_havuzu:
                belge_havuzu[doc_id] = item.copy()
            belge_havuzu[doc_id]["graf_sirasi"] = rank_g
            belge_havuzu[doc_id]["graf_skoru"] = item.get("graf_skoru", 0.0)

        # 3. Sırala ve Sıralama Kaymasını (Rank Shift) Hesapla
        sirali_id_ler = sorted(rrf_skorlari.items(), key=lambda x: x[1], reverse=True)

        nihai_sonuclar = []
        for nihai_rank, (doc_id, rrf_skor) in enumerate(sirali_id_ler[:top_k], start=1):
            doc = belge_havuzu[doc_id].copy()
            doc["rrf_skoru"] = round(rrf_skor, 5)
            doc["nihai_sira"] = nihai_rank

            # Sıralama Kayması (İlk Vektör sırasından nihai sıraya değişim)
            v_rank = doc.get("vektor_sirasi", 99)
            doc["siralama_kaymasi"] = v_rank - nihai_rank
            nihai_sonuclar.append(doc)

        return nihai_sonuclar
