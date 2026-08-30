"""
Day 377: Wafer-Scale Engine (WSE) 2D-Torus Network-on-Chip (NoC) & Fault Tolerance
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Wafer-Scale Engine (WSE) devasa çip-ölçeğinde 2D-Torus NoC yönlendiricisini,
kusurlu çekirdek (silicon defect) dinamik baypas yönlendirmesini ve flit-seviyesi
AI tensör veri akışı simülatörünü içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from collections import deque
from enum import Enum


class NodeStatus(Enum):
    HEALTHY = "HEALTHY"
    DEFECTIVE = "DEFECTIVE"
    SPARE = "SPARE"


class FlitType(Enum):
    HEADER = "HEADER"
    BODY = "BODY"
    TAIL = "TAIL"
    SINGLE = "SINGLE"


class FlitPacket:
    """
    Wafer NoC Flit Paketi.
    """
    def __init__(
        self,
        packet_id: int,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        flit_type: FlitType = FlitType.SINGLE,
        payload_bytes: int = 64,
        birth_cycle: int = 0
    ):
        self.packet_id = packet_id
        self.src = src
        self.dst = dst
        self.flit_type = flit_type
        self.payload_bytes = payload_bytes
        self.birth_cycle = birth_cycle
        self.curr_pos = src
        self.hop_count = 0
        self.arrival_cycle: Optional[int] = None
        self.route_path: List[Tuple[int, int]] = [src]

    @property
    def is_delivered(self) -> bool:
        return self.curr_pos == self.dst


class WaferGridNode:
    """
    Wafer-Scale 2D Grid Üzerindeki Tekil AI Çekirdeği (Processing Element - PE).
    """
    def __init__(self, x: int, y: int, status: NodeStatus = NodeStatus.HEALTHY, buffer_depth: int = 16):
        self.x = x
        self.y = y
        self.status = status
        self.buffer_depth = buffer_depth
        self.rx_queue: List[FlitPacket] = []
        self.tx_queue: List[FlitPacket] = []
        self.delivered_packets: List[FlitPacket] = []


class Torus2DRouter:
    """
    2D-Torus Boyut-Sıralı (XY DOR) ve Hata Baypas Yönlendiricisi.
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def torus_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """İki nokta arasındaki toroidal Manhattan mesafesini hesaplar."""
        dx = min((p1[0] - p2[0]) % self.width, (p2[0] - p1[0]) % self.width)
        dy = min((p1[1] - p2[1]) % self.height, (p2[1] - p1[1]) % self.height)
        return dx + dy

    def next_hop_torus(self, curr: Tuple[int, int], dst: Tuple[int, int]) -> Tuple[int, int]:
        """Klasik 2D-Torus üzerinde en kısa toroidal adımı belirler."""
        cx, cy = curr
        dx, dy = dst

        if cx != dx:
            diff_x = (dx - cx) % self.width
            if diff_x <= self.width // 2:
                nx = (cx + 1) % self.width
            else:
                nx = (cx - 1) % self.width
            return (nx, cy)

        if cy != dy:
            diff_y = (dy - cy) % self.height
            if diff_y <= self.height // 2:
                ny = (cy + 1) % self.height
            else:
                ny = (cy - 1) % self.height
            return (cx, ny)

        return curr

    def route_step_fault_tolerant(
        self,
        curr: Tuple[int, int],
        dst: Tuple[int, int],
        defect_map: np.ndarray,
        recent_path: List[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """
        Kusurlu silikon çekirdeklerini dinamik baypas eden, kilitlenmesiz en kısa yol yönlendirmesi.
        """
        cx, cy = curr
        if curr == dst:
            return curr

        # Kusur yoksa hızlı XY DOR kullan
        if not np.any(defect_map):
            return self.next_hop_torus(curr, dst)

        # Kusurlu ızgara üzerinde BFS ile en kısa sağlıklı adımı bul
        queue = deque([(curr, [curr])])
        visited = {curr}

        while queue:
            node, path = queue.popleft()
            if node == dst:
                return path[1] if len(path) > 1 else curr

            nx, ny = node
            neighbors = [
                ((nx + 1) % self.width, ny),
                ((nx - 1) % self.width, ny),
                (nx, (ny + 1) % self.height),
                (nx, (ny - 1) % self.height)
            ]

            for neigh in neighbors:
                if neigh not in visited and not defect_map[neigh[0], neigh[1]]:
                    visited.add(neigh)
                    queue.append((neigh, path + [neigh]))

        return curr


class WaferScaleEngineFabric:
    """
    Wafer-Scale Engine 2D-Torus Kumaş Simülatörü.
    """
    def __init__(self, width: int = 16, height: int = 16, link_bw_gbps: float = 100.0):
        self.width = width
        self.height = height
        self.link_bw_gbps = link_bw_gbps  # Hat başına 100 GB/s
        self.router = Torus2DRouter(width, height)
        self.nodes: Dict[Tuple[int, int], WaferGridNode] = {}
        self.defect_map = np.zeros((width, height), dtype=bool)
        self._init_grid()

    def _init_grid(self):
        for x in range(self.width):
            for y in range(self.height):
                self.nodes[(x, y)] = WaferGridNode(x, y)

    def inject_silicon_defects(self, defect_rate: float = 0.05, seed: int = 42):
        """Wafer yüzeyine rastgele silikon üretim kusurları enjekte eder."""
        np.random.seed(seed)
        num_defects = int(self.width * self.height * defect_rate)
        defect_indices = np.random.choice(self.width * self.height, size=num_defects, replace=False)
        
        self.defect_map.fill(False)
        for idx in defect_indices:
            x = idx // self.height
            y = idx % self.height
            self.defect_map[x, y] = True
            self.nodes[(x, y)].status = NodeStatus.DEFECTIVE

    def bisection_bandwidth_pbps(self) -> float:
        """Kumaşın toplam Bisection Bant Genişliğini (Petabytes/sec cinsinden) hesaplar."""
        total_cut_links = 4 * min(self.width, self.height)
        bw_tbps = total_cut_links * self.link_bw_gbps / 8.0  # GB/s
        bw_pbps = bw_tbps / 1000.0
        return bw_pbps

    def simulate_traffic(self, packets: List[FlitPacket], max_cycles: int = 150) -> Dict[str, Any]:
        """Enjekte edilen paketlerin wafer üzerinde yönlendirilmesini simüle eder."""
        active_packets = [p for p in packets]
        delivered_packets: List[FlitPacket] = []

        for cycle in range(max_cycles):
            if not active_packets:
                break

            remaining_packets = []
            for pkt in active_packets:
                if pkt.is_delivered:
                    pkt.arrival_cycle = cycle
                    delivered_packets.append(pkt)
                    continue

                # Bir adım yönlendir
                next_pos = self.router.route_step_fault_tolerant(
                    pkt.curr_pos, pkt.dst, self.defect_map, recent_path=pkt.route_path
                )
                pkt.curr_pos = next_pos
                pkt.hop_count += 1
                pkt.route_path.append(next_pos)

                if pkt.is_delivered:
                    pkt.arrival_cycle = cycle + 1
                    delivered_packets.append(pkt)
                else:
                    remaining_packets.append(pkt)

            active_packets = remaining_packets

        delivery_rate = (len(delivered_packets) / max(1, len(packets))) * 100.0
        avg_hops = float(np.mean([p.hop_count for p in delivered_packets])) if delivered_packets else 0.0
        avg_latency = float(np.mean([p.arrival_cycle - p.birth_cycle for p in delivered_packets])) if delivered_packets else 0.0

        return {
            "total_packets": len(packets),
            "delivered_count": len(delivered_packets),
            "delivery_rate": delivery_rate,
            "avg_hops": avg_hops,
            "avg_latency_cycles": avg_latency,
            "delivered_packets": delivered_packets,
            "bisection_bw_pbps": self.bisection_bandwidth_pbps()
        }


class WSEBenchmark:
    """
    Wafer-Scale 2D-Torus Kumaş Kıyaslama ve Kusur Dayanıklılığı Test Motoru.
    """
    def __init__(self, width: int = 16, height: int = 16):
        self.width = width
        self.height = height

    def generate_synthetic_tensor_traffic(self, num_packets: int = 300, defect_map: np.ndarray = None) -> List[FlitPacket]:
        """Sağlıklı düğümler arasında AI tensör all-to-all veri paketleri oluşturur."""
        packets = []
        healthy_nodes = []
        for x in range(self.width):
            for y in range(self.height):
                if defect_map is None or not defect_map[x, y]:
                    healthy_nodes.append((x, y))

        np.random.seed(42)
        for i in range(num_packets):
            src_idx = np.random.randint(0, len(healthy_nodes))
            dst_idx = np.random.randint(0, len(healthy_nodes))
            while dst_idx == src_idx:
                dst_idx = np.random.randint(0, len(healthy_nodes))

            src = healthy_nodes[src_idx]
            dst = healthy_nodes[dst_idx]
            pkt = FlitPacket(packet_id=i, src=src, dst=dst, birth_cycle=0)
            packets.append(pkt)

        return packets

    def run_benchmark(self) -> Dict[str, Any]:
        """Kusursuz vs %5 Kusurlu Wafer Başarımını Kıyaslar."""
        # 1. Kusursuz Wafer Koşumu
        fabric_healthy = WaferScaleEngineFabric(self.width, self.height)
        pkts_healthy = self.generate_synthetic_tensor_traffic(num_packets=300)
        res_healthy = fabric_healthy.simulate_traffic(pkts_healthy, max_cycles=150)

        # 2. %5 Kusurlu Wafer Koşumu (Fault-Tolerant Bypass)
        fabric_faulty = WaferScaleEngineFabric(self.width, self.height)
        fabric_faulty.inject_silicon_defects(defect_rate=0.05, seed=42)
        pkts_faulty = self.generate_synthetic_tensor_traffic(num_packets=300, defect_map=fabric_faulty.defect_map)
        res_faulty = fabric_faulty.simulate_traffic(pkts_faulty, max_cycles=150)

        return {
            "width": self.width,
            "height": self.height,
            "healthy": res_healthy,
            "faulty": res_faulty,
            "defect_map": fabric_faulty.defect_map,
            "bisection_bw_pbps": fabric_healthy.bisection_bandwidth_pbps()
        }

    def kos(self) -> Dict[str, Any]:
        return self.run_benchmark()
