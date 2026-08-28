"""
Bilgi Erişimi ve Vektör Arama Değerlendirme Metrikleri Motoru (NDCG, MRR, MAP, Precision, Latency).
"""

from typing import List, Set, Dict, Any, Union
import numpy as np


class RetrievalMetrikMotoru:
    """Endüstri standardı Information Retrieval (IR) ve Semantik Arama metrik hesaplayıcısı."""

    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: Set[int], k: int = 10) -> float:
        """Precision@k = (Getirilen ilk k belgedeki ilgili belge sayısı) / k"""
        if k <= 0:
            return 0.0
        retrieved_k = retrieved[:k]
        ilgili_sayisi = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return float(ilgili_sayisi / k)

    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: Set[int], k: int = 10) -> float:
        """Recall@k = (Getirilen ilk k belgedeki ilgili belge sayısı) / (Toplam ilgili belge sayısı)"""
        if len(relevant) == 0:
            return 0.0
        retrieved_k = retrieved[:k]
        ilgili_sayisi = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return float(ilgili_sayisi / len(relevant))

    @staticmethod
    def reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:
        """Reciprocal Rank (RR) = 1 / (İlk ilgili belgenin 1-tabanlı sıralama indeksi)"""
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return float(1.0 / rank)
        return 0.0

    @staticmethod
    def average_precision_at_k(retrieved: List[int], relevant: Set[int], k: int = 10) -> float:
        """Average Precision@k (AP@k) = İlgili her pozisyondaki Precision@i ortalaması."""
        if len(relevant) == 0:
            return 0.0

        toplam_precision = 0.0
        ilgili_sayisi = 0
        retrieved_k = retrieved[:k]

        for i, doc_id in enumerate(retrieved_k, start=1):
            if doc_id in relevant:
                ilgili_sayisi += 1
                toplam_precision += ilgili_sayisi / i

        return float(toplam_precision / min(len(relevant), k))

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int = 10) -> float:
        """Discounted Cumulative Gain@k (DCG@k) = sum((2^rel - 1) / log2(i + 1))"""
        relevances_k = relevances[:k]
        dcg = 0.0
        for i, rel in enumerate(relevances_k, start=1):
            dcg += (np.power(2.0, rel) - 1.0) / np.log2(i + 1.0)
        return float(dcg)

    @classmethod
    def ndcg_at_k(
        cls,
        retrieved_relevances: List[float],
        all_true_relevances: List[float],
        k: int = 10
    ) -> float:
        """Normalized Discounted Cumulative Gain@k (NDCG@k) = DCG@k / IDCG@k"""
        dcg = cls.dcg_at_k(retrieved_relevances, k=k)
        ideal_relevances = sorted(all_true_relevances, reverse=True)
        idcg = cls.dcg_at_k(ideal_relevances, k=k)

        if idcg <= 0.0:
            return 0.0
        return float(dcg / idcg)

    @staticmethod
    def gecikme_profili_cikar(gecikmeler_ms: List[float]) -> Dict[str, float]:
        """Gecikme (latency) serisi için p50, p90, p95, p99, ortalama ve QPS üretir."""
        if not gecikmeler_ms:
            return {
                "ortalama_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0,
                "p95_ms": 0.0, "p99_ms": 0.0, "qps": 0.0
            }

        dizi = np.array(gecikmeler_ms, dtype=np.float64)
        ortalama = float(np.mean(dizi))
        qps = float(1000.0 / ortalama) if ortalama > 0 else 0.0

        return {
            "ortalama_ms": ortalama,
            "std_ms": float(np.std(dizi)),
            "p50_ms": float(np.percentile(dizi, 50)),
            "p90_ms": float(np.percentile(dizi, 90)),
            "p95_ms": float(np.percentile(dizi, 95)),
            "p99_ms": float(np.percentile(dizi, 99)),
            "min_ms": float(np.min(dizi)),
            "max_ms": float(np.max(dizi)),
            "qps": qps
        }
