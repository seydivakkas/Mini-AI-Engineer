"""
Ray Cluster ve Ray Serve Performans ve Yük Profilleyici Modülü (Day 196 - FAZ 10).
Trafik Patlaması, Otomatik Ölçekleme (Autoscaling) ve P50/P95/P99 Gecikme Analitiği.
"""

from typing import Dict, Any, List
import numpy as np
from .ray_serve_motoru import (
    RayClusterNode,
    RayServeDeploymentManager,
    RayServeRouter,
)


class RayClusterYukProfilleyici:
    """
    Ray Kümesi Çok Düğümlü Yük ve Ölçekleme Profilleyicisi.
    """

    @classmethod
    def kume_yuk_simulasyonu_calistir(cls) -> Dict[str, Any]:
        """Çok düğümlü kümede 3 farklı trafik fazında (Düşük, Orta, Zirve) yük testi simüle eder."""
        # 1. Küme Topolojisi: 1 Head Node (4 GPU) + 2 Worker Node (4'er GPU) = Toplam 12 GPU
        manager = RayServeDeploymentManager(min_replicas=2, max_replicas=8, target_ongoing_requests=5)
        manager.dugum_ekle(RayClusterNode("head_node", "10.0.0.1", gpu_count=4))
        manager.dugum_ekle(RayClusterNode("worker_node_1", "10.0.0.2", gpu_count=4))
        manager.dugum_ekle(RayClusterNode("worker_node_2", "10.0.0.3", gpu_count=4))

        manager.baslat_varsayilan_replikalar()

        fazlar = [
            {"faz_adi": "1. Düşük Trafik (Sakin)", "anlik_istek_sayisi": 8},
            {"faz_adi": "2. Orta Trafik (Normal)", "anlik_istek_sayisi": 22},
            {"faz_adi": "3. Zirve Trafik Patlaması", "anlik_istek_sayisi": 40},
        ]

        faz_sonuclari = []
        for f in fazlar:
            istek_sayisi = f["anlik_istek_sayisi"]
            aktif_replika_sayisi = manager.autoscale(istek_sayisi)

            latencies = []
            for _ in range(istek_sayisi):
                secilen_rep = RayServeRouter.en_uygun_replika_sec(manager.replicas)
                if secilen_rep is not None:
                    res = secilen_rep.process_request(prompt_len=128, gen_tokens=32)
                    latencies.append(res["latency_ms"])

            lat_arr = np.array(latencies) if latencies else np.array([20.0])
            p50 = float(np.percentile(lat_arr, 50))
            p95 = float(np.percentile(lat_arr, 95))
            p99 = float(np.percentile(lat_arr, 99))

            faz_sonuclari.append({
                "faz_adi": f["faz_adi"],
                "istek_sayisi": istek_sayisi,
                "aktif_replika_sayisi": aktif_replika_sayisi,
                "p50_gecikme_ms": p50,
                "p95_gecikme_ms": p95,
                "p99_gecikme_ms": p99,
                "kume_gpu_kullanimi_yuzde": (aktif_replika_sayisi / 12.0) * 100.0,
            })

        return {
            "toplam_kume_gpu": 12,
            "faz_raporlari": faz_sonuclari,
        }
