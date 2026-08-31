"""
Tesla Dojo D1 Mesh Birim Testleri (PyTest)
==========================================
Bu test paketi; 2D Mesh Manhattan mesafe hesabını, Dimension-Ordered (XY)
yönlendirmeyi ve paket transfer gecikmesini test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_dojo_d1_mesh_yonlendirici import TeslaDojoMeshRouter


def test_manhattan_mesafe_hesabi():
    """İki D1 çipi arasındaki Manhattan atlama sayısının doğru hesaplandığı test edilir."""
    router = TeslaDojoMeshRouter()

    hops_corner = router.compute_manhattan_distance((0, 0), (4, 4))
    hops_asym = router.compute_manhattan_distance((0, 0), (4, 5))

    assert hops_corner == 8
    assert hops_asym == 9


def test_xy_yonlendirme_yolu():
    """XY Dimension-Ordered yönlendirmenin önce X sonra Y koordinatlarını geçtiği test edilir."""
    router = TeslaDojoMeshRouter()
    path = router.route_xy_dimension_ordered((0, 0), (2, 2))

    assert path[0] == (0, 0)
    assert path[1] == (1, 0)
    assert path[2] == (2, 0)
    assert path[3] == (2, 1)
    assert path[4] == (2, 2)
    assert len(path) == 5


def test_dojo_paket_transfer_gecikmesi():
    """1 MB veri transferinin nanosaniye gecikme ve yüksek bant genişliği sağladığı test edilir."""
    router = TeslaDojoMeshRouter()
    res = router.calculate_packet_transfer_latency(src=(0, 0), dst=(4, 4), payload_bytes=1024 * 1024)

    assert res["hops"] == 8
    assert res["t_hop_ns"] == 20.0  # 8 * 2.5 ns
    assert res["total_latency_ns"] > 0.0
    assert res["effective_bw_gb_s"] > 1000.0
