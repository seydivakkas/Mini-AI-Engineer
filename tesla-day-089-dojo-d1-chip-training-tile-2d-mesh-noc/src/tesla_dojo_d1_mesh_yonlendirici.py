r"""
Tesla Dojo D1 Çipi ve 2D Mesh NoC Yönlendirme Çekirdeği
========================================================
Bu modül; Tesla Dojo Süperbilgisayarının temel yapı taşı olan D1 özel silikonunu,
25 D1 çipinden oluşan 9 PFLOPS Training Tile matrisini ($5 \times 5$),
Dimension-Ordered Routing (XY Yönlendirme) algoritmasını ve 36 TB/s biseksiyon
bant genişliği gecikme modellemesini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaDojoMeshRouter:
    """
    Tesla Dojo 2D Mesh Network-on-Chip (NoC) Yönlendiricisi.
    """
    def __init__(
        self,
        grid_width: int = 5,
        grid_height: int = 5,
        hop_latency_ns: float = 2.5,
        link_bandwidth_tb_s: float = 2.0,  # 2 TB/s çip kenar bant genişliği
        tflops_per_d1: float = 362.0       # 362 TFLOPS BF16/CFP8
    ):
        self.w = grid_width
        self.h = grid_height
        self.num_chips = self.w * self.h
        self.hop_lat_ns = hop_latency_ns
        self.link_bw_tb = link_bandwidth_tb_s
        self.tflops_per_chip = tflops_per_d1
        self.tile_pflops = (self.num_chips * self.tflops_per_chip) / 1000.0  # 9.05 PFLOPS

    def compute_manhattan_distance(self, src: Tuple[int, int], dst: Tuple[int, int]) -> int:
        """İki D1 çipi arasındaki 2D Mesh Manhattan mesafe (atlama) sayısı."""
        return abs(src[0] - dst[0]) + abs(src[1] - dst[1])

    def route_xy_dimension_ordered(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Dimension-Ordered (XY) Yönlendirme Algoritması (Deadlock-Free).
        Önce X ekseninde, ardından Y ekseninde ilerler.
        """
        path = [src]
        curr_x, curr_y = src
        dst_x, dst_y = dst

        # 1. X Ekseninde Yönlendirme
        step_x = 1 if dst_x > curr_x else -1
        while curr_x != dst_x:
            curr_x += step_x
            path.append((curr_x, curr_y))

        # 2. Y Ekseninde Yönlendirme
        step_y = 1 if dst_y > curr_y else -1
        while curr_y != dst_y:
            curr_y += step_y
            path.append((curr_x, curr_y))

        return path

    def calculate_packet_transfer_latency(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        payload_bytes: int = 1024 * 1024  # 1 MB Tensor parçası
    ) -> Dict[str, Any]:
        """
        D1 NoC ağı üzerinden veri transfer gecikmesini (ns) modeller.
        """
        hops = self.compute_manhattan_distance(src, dst)
        path = self.route_xy_dimension_ordered(src, dst)

        # Atlama gecikmesi (Hop Latency)
        t_hop_total_ns = hops * self.hop_lat_ns

        # İletim gecikmesi (Serialization Latency)
        # link_bw: 2.0 TB/s = 2.0 * 10^12 Bayt/s = 2000 Bayt/ns
        bytes_per_ns = (self.link_bw_tb * 1e12) / 1e9
        t_serial_ns = payload_bytes / bytes_per_ns

        total_latency_ns = t_hop_total_ns + t_serial_ns

        return {
            "src": src,
            "dst": dst,
            "hops": hops,
            "path": path,
            "payload_bytes": payload_bytes,
            "t_hop_ns": float(np.round(t_hop_total_ns, 2)),
            "t_serial_ns": float(np.round(t_serial_ns, 2)),
            "total_latency_ns": float(np.round(total_latency_ns, 2)),
            "effective_bw_gb_s": float(np.round((payload_bytes / (total_latency_ns * 1e-9)) / 1e9, 2))
        }
