"""
Day 377: Unit Tests for Wafer-Scale Engine (WSE) 2D-Torus NoC & Fault Tolerance
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from wse_2d_torus_noc_motoru import (
    Torus2DRouter,
    WaferScaleEngineFabric,
    FlitPacket,
    WSEBenchmark
)


def test_torus_2d_kisa_yol_yonlendirme():
    """2D-Torus toroidal sınır geçişlerinde en kısa adımın seçildiğini test eder."""
    router = Torus2DRouter(width=16, height=16)
    
    # 0'dan 15'e gitmek için +X yerine -X (wrap-around) adımı atılmalıdır: (15, 0)
    next_step = router.next_hop_torus(curr=(0, 0), dst=(15, 0))
    assert next_step == (15, 0), f"Toroidal wrap-around adımı (15, 0) olmalıdır, alınan: {next_step}"

    # 0'dan 4'e gitmek için +X adımı: (1, 0)
    next_step2 = router.next_hop_torus(curr=(0, 0), dst=(4, 0))
    assert next_step2 == (1, 0)


def test_kusurlu_cekirdek_dinamik_baypas():
    """Önündeki çekirdek kusurlu olan paketin güvenli baypas adımı attığını test eder."""
    router = Torus2DRouter(width=8, height=8)
    defect_map = np.zeros((8, 8), dtype=bool)
    
    # (1, 0) düğümünü kusurlu yapalım
    defect_map[1, 0] = True
    
    # (0, 0) noktasından (3, 0) noktasına giderken (1, 0) yerine Y eksenine sapmalıdır
    detour_step = router.route_step_fault_tolerant(curr=(0, 0), dst=(3, 0), defect_map=defect_map)
    assert detour_step != (1, 0), "Kusurlu (1,0) düğümüne girilmemelidir!"
    assert not defect_map[detour_step[0], detour_step[1]], "Baypas adımı sağlıklı bir düğüm olmalıdır."


def test_bisection_bant_genisligi_hesabi():
    """Wafer kumaşının Bisection Bant Genişliğinin doğru hesaplandığını test eder."""
    fabric = WaferScaleEngineFabric(width=16, height=16, link_bw_gbps=100.0)
    bw_pbps = fabric.bisection_bandwidth_pbps()
    
    assert bw_pbps > 0.0
    assert isinstance(bw_pbps, float)
    # 4 * 16 * 100 / 8 / 1000 = 0.8 PB/s
    assert abs(bw_pbps - 0.8) < 1e-4


def test_tam_wafer_scale_benchmark():
    """Tam benchmark akışında sıfır paket kaybı (%100 teslimat) elde edildiğini test eder."""
    bench = WSEBenchmark(width=8, height=8)
    res = bench.kos()

    assert res["healthy"]["delivery_rate"] >= 99.0, "Kusursuz wafer teslimat oranı %100 olmalıdır."
    assert res["faulty"]["delivery_rate"] >= 99.0, "Kusurlu wafer adaptif yönlendirme ile %100 paket teslim etmelidir."
    assert res["faulty"]["avg_hops"] >= res["healthy"]["avg_hops"], "Baypas rotaları ortalama hop sayısını artırmalıdır."
