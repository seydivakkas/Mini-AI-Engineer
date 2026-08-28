"""
Farklı Arama ve Bilgi Erişimi Stratejilerini (Dense, BM25, Hibrit RRF, IVF) Kıyaslayan Değerlendirme Motoru.
"""

from typing import Dict, List, Any, Set, Tuple
import numpy as np
from .metrik_motoru import RetrievalMetrikMotoru


class AramaDegerlendirici:
    """Çoklu semantik ve leksikal arama boru hatlarını NDCG, MRR, MAP ve Gecikme açısından değerlendirir."""

    @staticmethod
    def sentetik_arama_senaryosu_uret(
        num_queries: int = 500,
        katalog_boyutu: int = 10_000
    ) -> List[Dict[str, Any]]:
        """Gerçekçi sorgular, ilgili belgeler ve dereceli (graded: 0-3) ilgi skorları üretir."""
        np.random.seed(42)
        senaryolar = []

        for q_id in range(num_queries):
            num_relevant = np.random.randint(5, 15)
            relevant_docs = np.random.choice(katalog_boyutu, size=num_relevant, replace=False).tolist()

            # Dereceli ilgililik puanları: 3: Çok İlgili, 2: İlgili, 1: Kısmen İlgili
            graded_relevance: Dict[int, float] = {}
            for i, doc in enumerate(relevant_docs):
                if i < 2:
                    graded_relevance[doc] = 3.0
                elif i < 5:
                    graded_relevance[doc] = 2.0
                else:
                    graded_relevance[doc] = 1.0

            senaryolar.append({
                "query_id": q_id,
                "relevant_set": set(relevant_docs),
                "graded_relevance": graded_relevance
            })

        return senaryolar

    @staticmethod
    def _strateji_sonucu_simule_et(
        senaryo: Dict[str, Any],
        strateji_adi: str,
        top_k: int = 20
    ) -> Tuple[List[int], float]:
        """Belirtilen arama stratejisinin sıralama kalitesini ve gecikmesini simüle eder."""
        relevant_list = list(senaryo["relevant_set"])
        all_graded = senaryo["graded_relevance"]

        # İlgili belgeleri ve rastgele ilgisiz belgeleri harmanla
        if strateji_adi == "Hybrid RRF (Dense + BM25)":
            # Yüksek kaliteli sıralama, ilgililer en başta
            sirali_relevant = sorted(relevant_list, key=lambda d: all_graded.get(d, 0.0), reverse=True)
            gurultu = np.random.choice(10000, size=top_k, replace=False).tolist()
            retrieved = (sirali_relevant[:int(top_k * 0.7)] + gurultu)[:top_k]
            gecikme = np.random.lognormal(mean=1.5, sigma=0.2)  # ~4.5 - 6 ms
        elif strateji_adi == "Dense Vector (HNSW)":
            # Semantik olarak güçlü, nadiren ıskalar
            np.random.shuffle(relevant_list)
            gurultu = np.random.choice(10000, size=top_k, replace=False).tolist()
            retrieved = (relevant_list[:int(top_k * 0.5)] + gurultu)[:top_k]
            gecikme = np.random.lognormal(mean=0.7, sigma=0.25)  # ~1.8 - 2.5 ms
        elif strateji_adi == "Lexical BM25 (Keyword)":
            # Anahtar kelime eşleşmesi, semantikte zayıf kalabilir
            gurultu = np.random.choice(10000, size=top_k, replace=False).tolist()
            retrieved = (relevant_list[:int(top_k * 0.35)] + gurultu)[:top_k]
            gecikme = np.random.lognormal(mean=1.1, sigma=0.3)  # ~3.0 - 4.0 ms
        elif strateji_adi == "Approx IVF-Flat (Fast Baseline)":
            # Hızlı fakat Voronoi sınır kayıplı
            gurultu = np.random.choice(10000, size=top_k, replace=False).tolist()
            retrieved = (relevant_list[:int(top_k * 0.2)] + gurultu)[:top_k]
            gecikme = np.random.lognormal(mean=-0.2, sigma=0.2)  # ~0.7 - 1.0 ms
        else:
            raise ValueError(f"Bilinmeyen strateji: {strateji_adi}")

        return retrieved, gecikme

    @classmethod
    def calistir_karsilastirma(
        cls,
        senaryolar: List[Dict[str, Any]],
        k_list: List[int] = [5, 10]
    ) -> Dict[str, Dict[str, Any]]:
        stratejiler = [
            "Hybrid RRF (Dense + BM25)",
            "Dense Vector (HNSW)",
            "Lexical BM25 (Keyword)",
            "Approx IVF-Flat (Fast Baseline)"
        ]

        sonuclar: Dict[str, Dict[str, Any]] = {}

        for strat in stratejiler:
            ndcg_5_list, ndcg_10_list = [], []
            mrr_list = []
            map_10_list = []
            prec_5_list, prec_10_list = [], []
            recall_10_list = []
            gecikmeler = []

            for s in senaryolar:
                retrieved, lat = cls._strateji_sonucu_simule_et(s, strat, top_k=20)
                gecikmeler.append(lat)

                rel_set = s["relevant_set"]
                graded_dict = s["graded_relevance"]

                # Getirilen belgelerin dereceli puanları
                retrieved_relevances = [graded_dict.get(doc, 0.0) for doc in retrieved]
                all_true_relevances = list(graded_dict.values())

                # Metrik hesaplamaları
                ndcg_5_list.append(RetrievalMetrikMotoru.ndcg_at_k(retrieved_relevances, all_true_relevances, k=5))
                ndcg_10_list.append(RetrievalMetrikMotoru.ndcg_at_k(retrieved_relevances, all_true_relevances, k=10))
                mrr_list.append(RetrievalMetrikMotoru.reciprocal_rank(retrieved, rel_set))
                map_10_list.append(RetrievalMetrikMotoru.average_precision_at_k(retrieved, rel_set, k=10))
                prec_5_list.append(RetrievalMetrikMotoru.precision_at_k(retrieved, rel_set, k=5))
                prec_10_list.append(RetrievalMetrikMotoru.precision_at_k(retrieved, rel_set, k=10))
                recall_10_list.append(RetrievalMetrikMotoru.recall_at_k(retrieved, rel_set, k=10))

            gecikme_profil = RetrievalMetrikMotoru.gecikme_profili_cikar(gecikmeler)

            sonuclar[strat] = {
                "ndcg@5": float(np.mean(ndcg_5_list)),
                "ndcg@10": float(np.mean(ndcg_10_list)),
                "mrr": float(np.mean(mrr_list)),
                "map@10": float(np.mean(map_10_list)),
                "precision@5": float(np.mean(prec_5_list)),
                "precision@10": float(np.mean(prec_10_list)),
                "recall@10": float(np.mean(recall_10_list)),
                "gecikme_istatistikleri": gecikme_profil,
                "ham_gecikmeler": gecikmeler
            }

        return sonuclar
