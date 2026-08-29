"""
Kaos Mühendisliği ve Arıza Enjeksiyon Motoru (Day 200 - FAZ 10).
GPU OOM, Ağ Gecikmesi, Düğüm Kapatma ve Otomatik İyileştirme (Self-Healing Failover).
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import random
import time


class NodeState(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"  # Yüksek ağ gecikmesi / jitter
    OOM = "OOM"            # CUDA Out of Memory
    CRASHED = "CRASHED"    # Düğüm tamamen çöktü / kapandı


class GPUClusterNode:
    """Dağıtık LLM Kümesindeki Tekil GPU Çıkarım Düğümü."""

    def __init__(self, node_id: str, gpu_id: int, base_latency_ms: float = 24.0):
        self.node_id = node_id
        self.gpu_id = gpu_id
        self.state: NodeState = NodeState.HEALTHY
        self.base_latency_ms = base_latency_ms
        self.extra_latency_ms: float = 0.0
        self.total_served: int = 0
        self.failed_requests: int = 0

    def execute_inference(self, prompt: str) -> Dict[str, Any]:
        """Düğüm üzerinde çıkarım yapar; düğüm durumuna göre yanıt veya hata döner."""
        self.total_served += 1

        if self.state in (NodeState.CRASHED, NodeState.OOM):
            self.failed_requests += 1
            return {
                "success": False,
                "node_id": self.node_id,
                "error": f"NodeFailure: Node is in {self.state.value} state!",
                "latency_ms": 0.0,
            }

        total_lat = self.base_latency_ms + self.extra_latency_ms + random.uniform(-2.0, 3.0)
        return {
            "success": True,
            "node_id": self.node_id,
            "response": f"[{self.node_id}] Yanıt: {prompt[:15]}...",
            "latency_ms": total_lat,
        }


class ChaosInjector:
    """Kaos Mühendisliği Arıza Enjektörü (Chaos Monkey for GPU LLM Clusters)."""

    @staticmethod
    def inject_gpu_oom(node: GPUClusterNode):
        """Düğüme CUDA Out of Memory arızası enjekte eder."""
        node.state = NodeState.OOM

    @staticmethod
    def inject_network_latency(node: GPUClusterNode, delay_ms: float = 150.0):
        """InfiniBand / PCIe ağ hattına yapay gecikme ve jitter enjekte eder."""
        node.state = NodeState.DEGRADED
        node.extra_latency_ms = delay_ms

    @staticmethod
    def inject_node_kill(node: GPUClusterNode):
        """Düğümü anında sonlandırır (Hard Kill / Kernel Panic)."""
        node.state = NodeState.CRASHED


class ResilientClusterManager:
    """
    Kendi Kendini İyileştiren ve Yükü Otomatik Aktaran (Self-Healing & Failover) Küme Yöneticisi.
    """

    def __init__(self, initial_node_count: int = 4):
        self.nodes: List[GPUClusterNode] = [
            GPUClusterNode(f"gpu-worker-{i}", gpu_id=i) for i in range(initial_node_count)
        ]
        self.recovery_events: List[Dict[str, Any]] = []

    def get_healthy_nodes(self) -> List[GPUClusterNode]:
        """Yalnızca sağlıklı veya hafif gecikmeli düğümleri döner."""
        return [n for n in self.nodes if n.state in (NodeState.HEALTHY, NodeState.DEGRADED)]

    def health_check_and_heal(self) -> int:
        """Bozulan veya çöken düğümleri tespit edip otomatik olarak yeniden başlatır (Self-Healing)."""
        healed_count = 0
        for i, node in enumerate(self.nodes):
            if node.state in (NodeState.CRASHED, NodeState.OOM):
                # Arızalı düğümü yeni sağlıklı pod ile değiştir
                new_node = GPUClusterNode(f"{node.node_id}-healed", gpu_id=node.gpu_id)
                self.nodes[i] = new_node
                healed_count += 1
                self.recovery_events.append({
                    "old_node": node.node_id,
                    "new_node": new_node.node_id,
                    "reason": node.state.value,
                    "recovery_time_ms": random.uniform(800.0, 1800.0),  # < 2 sn kurtarma
                })
            elif node.state == NodeState.DEGRADED:
                # Ağ gecikmesini normalize et
                node.state = NodeState.HEALTHY
                node.extra_latency_ms = 0.0
        return healed_count

    def route_inference(self, prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """İsteği sağlıklı düğümlere yönlendirir, hata durumunda anında yedek düğüme geçer (Failover)."""
        healthy_nodes = self.get_healthy_nodes()
        if not healthy_nodes:
            return {"success": False, "error": "ClusterExhausted: No healthy nodes available!"}

        for attempt in range(max_retries + 1):
            chosen_node = random.choice(healthy_nodes)
            resp = chosen_node.execute_inference(prompt)
            if resp["success"]:
                return resp
            # Düğüm hata verdiyse listeden çıkar ve bir sonraki sağlıklı düğümü dene (Failover)
            healthy_nodes = [n for n in healthy_nodes if n.node_id != chosen_node.node_id]
            if not healthy_nodes:
                break

        return {"success": False, "error": "FailoverFailed: All attempted nodes failed."}
