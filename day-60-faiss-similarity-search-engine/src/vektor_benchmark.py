"""
FAISS Vektör İndeksleri Benchmark ve Performans Kıyaslama Motoru (Recall, QPS, Latency).
"""

from typing import Dict, Any, List, Tuple
import time
import numpy as np
from .indeks_motoru import FAISSIndeksMotoru, IndeksTuru


class VektorBenchmarkRunner:
    """IndexFlatIP, IndexIVFFlat ve IndexHNSWFlat indekslerini Recall@k, QPS ve Gecikme açısından kıyaslar."""

    @staticmethod
    def sentetik_vektor_olustur(
        num_vectors: int = 100_000,
        num_queries: int = 1_000,
        dim: int = 128,
        num_clusters: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kümelenmiş ve L2-normalize edilmiş gerçekçi vektör veritabanı ve sorgu seti üretir."""
        np.random.seed(42)

        # Küme merkezleri
        merkezler = np.random.randn(num_clusters, dim).astype(np.float32)
        merkezler = merkezler / np.linalg.norm(merkezler, axis=1, keepdims=True)

        atamalar = np.random.randint(0, num_clusters, size=num_vectors)
        gurultu = np.random.randn(num_vectors, dim).astype(np.float32) * 0.25
        vektorler = merkezler[atamalar] + gurultu
        vektorler = vektorler / np.linalg.norm(vektorler, axis=1, keepdims=True)

        q_atamalar = np.random.randint(0, num_clusters, size=num_queries)
        q_gurultu = np.random.randn(num_queries, dim).astype(np.float32) * 0.25
        sorgular = merkezler[q_atamalar] + q_gurultu
        sorgular = sorgular / np.linalg.norm(sorgular, axis=1, keepdims=True)

        return np.ascontiguousarray(vektorler, dtype=np.float32), np.ascontiguousarray(sorgular, dtype=np.float32)

    @staticmethod
    def recall_hesapla(
        ann_indeksler: np.ndarray,
        ground_truth_indeksler: np.ndarray,
        k: int = 10
    ) -> float:
        """Yaklaşık Arama (ANN) ile Tam Arama (Exact) arasındaki Recall@k kesişim oranını hesaplar."""
        toplam_kesisim = 0
        num_queries = len(ground_truth_indeksler)

        for i in range(num_queries):
            gt_set = set(ground_truth_indeksler[i, :k])
            ann_set = set(ann_indeksler[i, :k])
            toplam_kesisim += len(gt_set.intersection(ann_set))

        return float((toplam_kesisim / (num_queries * k)) * 100.0)

    @classmethod
    def calistir_karsilastirma(
        cls,
        vektorler: np.ndarray,
        sorgular: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        dim = vektorler.shape[1]
        sonuclar: Dict[str, Dict[str, Any]] = {}

        # 1. Ground Truth — IndexFlatIP (Brute Force Exact)
        flat_motor = FAISSIndeksMotoru(dim=dim, indeks_turu=IndeksTuru.FLAT_IP)
        flat_build_s = flat_motor.egit_ve_ekle(vektorler)
        _, gt_indeksler, flat_ms = flat_motor.ara(sorgular, top_k=top_k)
        flat_qps = float(len(sorgular) / max(flat_ms / 1000.0, 1e-6))

        sonuclar["IndexFlatIP (Exact)"] = {
            "indeks_turu": "IndexFlatIP",
            "parametre": "Tam Arama (Exact)",
            "build_suresi_s": flat_build_s,
            "toplam_arama_ms": flat_ms,
            "tekil_sorgu_ms": float(flat_ms / len(sorgular)),
            "qps": flat_qps,
            "recall": 100.0,
            "bellek_tahmini_mb": float((vektorler.nbytes) / (1024 * 1024))
        }

        # 2. IndexIVFFlat (Varying nprobe: 1, 8, 32)
        nlist = int(np.sqrt(len(vektorler)) * 2)  # Örn: 100K için ~632 nlist
        ivf_motor = FAISSIndeksMotoru(dim=dim, indeks_turu=IndeksTuru.IVF_FLAT, nlist=nlist)
        ivf_build_s = ivf_motor.egit_ve_ekle(vektorler)

        for nprobe in [1, 8, 32]:
            _, ivf_indeksler, ivf_ms = ivf_motor.ara(sorgular, top_k=top_k, nprobe=nprobe)
            recall = cls.recall_hesapla(ivf_indeksler, gt_indeksler, k=top_k)
            qps = float(len(sorgular) / max(ivf_ms / 1000.0, 1e-6))

            sonuclar[f"IndexIVFFlat (nprobe={nprobe})"] = {
                "indeks_turu": "IndexIVFFlat",
                "parametre": f"nlist={nlist}, nprobe={nprobe}",
                "build_suresi_s": ivf_build_s,
                "toplam_arama_ms": ivf_ms,
                "tekil_sorgu_ms": float(ivf_ms / len(sorgular)),
                "qps": qps,
                "recall": recall,
                "bellek_tahmini_mb": float((vektorler.nbytes + nlist * dim * 4) / (1024 * 1024))
            }

        # 3. IndexHNSWFlat (Varying efSearch: 16, 32, 64)
        hnsw_motor = FAISSIndeksMotoru(dim=dim, indeks_turu=IndeksTuru.HNSW_FLAT, M=32, efConstruction=64)
        hnsw_build_s = hnsw_motor.egit_ve_ekle(vektorler)

        for ef_search in [16, 32, 64]:
            _, hnsw_indeksler, hnsw_ms = hnsw_motor.ara(sorgular, top_k=top_k, ef_search=ef_search)
            recall = cls.recall_hesapla(hnsw_indeksler, gt_indeksler, k=top_k)
            qps = float(len(sorgular) / max(hnsw_ms / 1000.0, 1e-6))

            sonuclar[f"IndexHNSWFlat (ef={ef_search})"] = {
                "indeks_turu": "IndexHNSWFlat",
                "parametre": f"M=32, efSearch={ef_search}",
                "build_suresi_s": hnsw_build_s,
                "toplam_arama_ms": hnsw_ms,
                "tekil_sorgu_ms": float(hnsw_ms / len(sorgular)),
                "qps": qps,
                "recall": recall,
                "bellek_tahmini_mb": float((vektorler.nbytes * 1.6) / (1024 * 1024))
            }

        return sonuclar
