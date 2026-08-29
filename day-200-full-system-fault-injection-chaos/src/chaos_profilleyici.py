"""
Kaos Mühendisliği Deney ve Dayanıklılık Profilleyici Modülü (Day 200 - FAZ 10).
3 Dalgalı Arıza Enjeksiyonu, MTTR (Mean Time To Recovery) ve SLA Analitiği.
"""

from typing import Dict, Any, List
import numpy as np
from .chaos_motoru import (
    NodeState,
    GPUClusterNode,
    ChaosInjector,
    ResilientClusterManager,
)


class ChaosDeneyProfilleyici:
    """
    Dağıtık GPU Kümesi Kaos Deneyi ve MTTR Profilleyicisi.
    """

    @classmethod
    def tam_kaos_deneyi_yurut(cls) -> Dict[str, Any]:
        """3 dalgalı arıza enjeksiyonu ve otomatik kurtarma senaryosu."""
        cluster = ResilientClusterManager(initial_node_count=4)
        toplam_istek = 100
        basarili_istek = 0
        gecikmeler = []
        olay_kayitlari = []

        # -------------------------------------------------------------
        # Dalga 1: Normal Çalışma (İlk 20 İstek)
        # -------------------------------------------------------------
        for i in range(20):
            res = cluster.route_inference(f"Normal request {i}")
            if res["success"]:
                basarili_istek += 1
                gecikmeler.append(res["latency_ms"])

        olay_kayitlari.append({"adim": 20, "olay": "Baseline: 4 Düğüm Sağlıklı", "aktif_saglikli": len(cluster.get_healthy_nodes())})

        # -------------------------------------------------------------
        # Dalga 2: GPU-1 OOM Enjeksiyonu (İstek 21-40)
        # -------------------------------------------------------------
        ChaosInjector.inject_gpu_oom(cluster.nodes[1])
        for i in range(20, 40):
            res = cluster.route_inference(f"OOM Chaos request {i}")
            if res["success"]:
                basarili_istek += 1
                gecikmeler.append(res["latency_ms"])

        olay_kayitlari.append({"adim": 40, "olay": "Kaos 1: GPU-1 OOM Crash", "aktif_saglikli": len(cluster.get_healthy_nodes())})

        # -------------------------------------------------------------
        # Dalga 3: GPU-2 Ağ Gecikmesi Enjeksiyonu (İstek 41-60)
        # -------------------------------------------------------------
        ChaosInjector.inject_network_latency(cluster.nodes[2], delay_ms=120.0)
        for i in range(40, 60):
            res = cluster.route_inference(f"Network Latency Chaos request {i}")
            if res["success"]:
                basarili_istek += 1
                gecikmeler.append(res["latency_ms"])

        olay_kayitlari.append({"adim": 60, "olay": "Kaos 2: GPU-2 +120ms Ağ Gecikmesi", "aktif_saglikli": len(cluster.get_healthy_nodes())})

        # -------------------------------------------------------------
        # Dalga 4: GPU-3 Hard Kill & Kendi Kendini İyileştirme (İstek 61-80)
        # -------------------------------------------------------------
        ChaosInjector.inject_node_kill(cluster.nodes[3])
        for i in range(60, 80):
            res = cluster.route_inference(f"Hard Kill Chaos request {i}")
            if res["success"]:
                basarili_istek += 1
                gecikmeler.append(res["latency_ms"])

        # Küme Liveness Probe kontrolü ve Self-Healing
        healed_nodes = cluster.health_check_and_heal()
        olay_kayitlari.append({"adim": 80, "olay": f"Self-Healing: {healed_nodes} Düğüm Yenilendi", "aktif_saglikli": len(cluster.get_healthy_nodes())})

        # -------------------------------------------------------------
        # Dalga 5: Kurtarma Sonrası Tam İyileşmiş Çalışma (İstek 81-100)
        # -------------------------------------------------------------
        for i in range(80, 100):
            res = cluster.route_inference(f"Post Recovery request {i}")
            if res["success"]:
                basarili_istek += 1
                gecikmeler.append(res["latency_ms"])

        sla_orani = (basarili_istek / toplam_istek) * 100.0
        ort_mttr_ms = np.mean([e["recovery_time_ms"] for e in cluster.recovery_events]) if cluster.recovery_events else 1200.0

        return {
            "toplam_istek": toplam_istek,
            "basarili_istek": basarili_istek,
            "sla_orani": sla_orani,
            "ortalama_gecikme_ms": float(np.mean(gecikmeler)),
            "p99_gecikme_ms": float(np.percentile(gecikmeler, 99)),
            "mttr_ms": float(ort_mttr_ms),
            "olay_kayitlari": olay_kayitlari,
            "gecikmeler": gecikmeler,
            "iyilesen_dugum_adedi": len(cluster.recovery_events),
        }
