"""
Ray Core & Ray Serve Dağıtık Model Dağıtım ve Yük Dengeleme Motoru (Day 196 - FAZ 10).
Çok Düğümlü Küme Yönetimi, Ray Actor Replikaları ve Akıllı Yönlendirme (Router) Mimarisi.
"""

from typing import Dict, Any, List, Optional
import time
import random
import torch


class RayClusterNode:
    """
    Ray Kümesi Fiziksel / Sanal Hesaplama Düğümü (Head veya Worker Node).
    """

    def __init__(self, node_id: str, node_ip: str, gpu_count: int = 4, cpu_count: int = 32):
        self.node_id = node_id
        self.node_ip = node_ip
        self.gpu_count = gpu_count
        self.cpu_count = cpu_count
        self.allocated_gpus: int = 0

    def gpu_ayir(self) -> Optional[int]:
        """Düğümden 1 adet boş GPU tahsis eder."""
        if self.allocated_gpus < self.gpu_count:
            gpu_id = self.allocated_gpus
            self.allocated_gpus += 1
            return gpu_id
        return None

    def gpu_serbest_birak(self):
        """Kullanılan 1 adet GPU'yu boş havuza iade eder."""
        if self.allocated_gpus > 0:
            self.allocated_gpus -= 1


class RayServeModelReplica:
    """
    Ray Actor Replikası (@serve.deployment).
    Tek bir GPU üzerinde izole çalışan model servis örneği.
    """

    def __init__(self, replica_id: str, node_id: str, gpu_id: int):
        self.replica_id = replica_id
        self.node_id = node_id
        self.gpu_id = gpu_id
        self.current_queue_depth: int = 0
        self.processed_requests: int = 0
        self.is_active: bool = True

    def process_request(self, prompt_len: int = 128, gen_tokens: int = 32) -> Dict[str, Any]:
        """Gelen çıkarım isteğini işler ve gecikme metriği üretir."""
        self.current_queue_depth += 1
        # Simüle edilmiş GPU işlem süresi (ms)
        baz_gecikme = 15.0 + (gen_tokens * 1.2) + (self.current_queue_depth * 2.5)
        gecikme_ms = max(5.0, baz_gecikme + random.uniform(-2.0, 3.0))

        self.processed_requests += 1
        self.current_queue_depth = max(0, self.current_queue_depth - 1)

        return {
            "replica_id": self.replica_id,
            "node_id": self.node_id,
            "gpu_id": self.gpu_id,
            "prompt_len": prompt_len,
            "gen_tokens": gen_tokens,
            "latency_ms": gecikme_ms,
        }


class RayServeRouter:
    """
    Ray Serve Akıllı İstek Yönlendiricisi (Power-of-Two-Choices & Least-Connections).
    """

    @classmethod
    def en_uygun_replika_sec(cls, replicas: List[RayServeModelReplica]) -> Optional[RayServeModelReplica]:
        """Aktif replikalar arasından en az kuyruk yüküne sahip olanı seçer."""
        aktif_replikalar = [r for r in replicas if r.is_active]
        if not aktif_replikalar:
            return None

        # Power-of-Two-Choices: Rastgele 2 replika seç ve daha az yoğun olanı ata
        if len(aktif_replikalar) >= 2:
            r1, r2 = random.sample(aktif_replikalar, 2)
            return r1 if r1.current_queue_depth <= r2.current_queue_depth else r2
        return aktif_replikalar[0]


class RayServeDeploymentManager:
    """
    Ray Serve Dağıtık Dağıtım ve Otomatik Ölçekleme (Autoscaler) Yöneticisi.
    """

    def __init__(self, min_replicas: int = 2, max_replicas: int = 8, target_ongoing_requests: int = 5):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_ongoing_requests = target_ongoing_requests

        self.nodes: List[RayClusterNode] = []
        self.replicas: List[RayServeModelReplica] = []

    def dugum_ekle(self, node: RayClusterNode):
        """Kümeye yeni bir fiziksel/sanal hesaplama düğümü ekler."""
        self.nodes.append(node)

    def baslat_varsayilan_replikalar(self):
        """Minimum replika sayısınca başlangıç aktörlerini küme düğümlerine dağıtır."""
        for i in range(self.min_replicas):
            self._yeni_replika_olustur(f"replica_{i+1}")

    def _yeni_replika_olustur(self, replica_id: str) -> Optional[RayServeModelReplica]:
        """Boş GPU'su olan ilk düğümde yeni bir Ray Serve replikası başlatır."""
        for node in self.nodes:
            gpu_id = node.gpu_ayir()
            if gpu_id is not None:
                rep = RayServeModelReplica(replica_id=replica_id, node_id=node.node_id, gpu_id=gpu_id)
                self.replicas.append(rep)
                return rep
        return None

    def autoscale(self, total_incoming_requests: int) -> int:
        """Gelen anlık istek sayısına göre replika sayısını dinamik olarak ölçekler."""
        ideal_replika = max(
            self.min_replicas,
            min(self.max_replicas, (total_incoming_requests + self.target_ongoing_requests - 1) // self.target_ongoing_requests)
        )

        mevcut_sayi = len([r for r in self.replicas if r.is_active])

        # Ölçekleme Artırımı (Scale Up)
        if ideal_replika > mevcut_sayi:
            for i in range(mevcut_sayi, ideal_replika):
                self._yeni_replika_olustur(f"replica_{len(self.replicas)+1}")
        # Ölçekleme Azaltımı (Scale Down)
        elif ideal_replika < mevcut_sayi:
            aktifler = [r for r in self.replicas if r.is_active]
            kaldilacak_adet = mevcut_sayi - ideal_replika
            for r in aktifler[-kaldilacak_adet:]:
                r.is_active = False

        return len([r for r in self.replicas if r.is_active])
