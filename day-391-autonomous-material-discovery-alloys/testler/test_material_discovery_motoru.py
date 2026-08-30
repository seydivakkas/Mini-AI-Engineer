"""
Day 391: Unit Tests for Autonomous Materials Discovery: HEAs & Superconductors
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from material_discovery_motoru import (
    HEAThermodynamics,
    CGCNNSuperconductorPredictor,
    MaterialDiscoveryBenchmark
)


def test_hea_configurational_entropy_equiatomic():
    """5-elementli eş-atomik alaşımda S_config = R*ln(5) ~ 13.38 J/mol.K olduğunu test eder."""
    thermo = HEAThermodynamics()
    fractions_5 = {"Fe": 0.2, "Co": 0.2, "Ni": 0.2, "Cr": 0.2, "Mn": 0.2}
    s_config = thermo.compute_configurational_entropy(fractions_5)

    expected = 8.314 * np.log(5.0)
    assert abs(s_config - expected) < 0.05
    assert s_config > 1.5 * 8.314, "Yüksek Entropi Sınırını (1.5 R) aşmalıdır."


def test_atomic_size_difference_delta():
    """Atomik boyut uyumsuzluğu delta hesabını test eder."""
    thermo = HEAThermodynamics()
    # Benzer boyutlu metaller (Fe, Co, Ni, Cr ~ 124-128 pm)
    fractions_similar = {"Fe": 0.25, "Co": 0.25, "Ni": 0.25, "Cr": 0.25}
    delta_similar = thermo.compute_atomic_size_difference(fractions_similar)

    # Çok farklı boyutlu metaller (Ba ~ 222 pm ile Fe ~ 126 pm)
    fractions_disparate = {"Fe": 0.50, "Ba": 0.50}
    delta_disparate = thermo.compute_atomic_size_difference(fractions_disparate)

    assert delta_similar < 6.6, "Kantor alaşımı delta < 6.6% sağlamalıdır."
    assert delta_disparate > delta_similar


def test_cgcnn_superconductor_prediction():
    """CGCNN modelinin Lantan bazlı kuprat/hidritlerde yüksek Tc tahmin ettiğini test eder."""
    cgcnn = CGCNNSuperconductorPredictor()
    fractions_la = {"La": 0.2, "Cu": 0.4, "Ba": 0.2, "Y": 0.2}
    res = cgcnn.predict_critical_temperature("La20Cu40Ba20Y20", fractions_la)

    assert res["tc_kelvin"] > 0.0
    assert res["electron_phonon_coupling_lambda"] > 1.0


def test_tam_material_discovery_benchmark():
    """Tam malzeme keşif ve tarama benchmarkını test eder."""
    bench = MaterialDiscoveryBenchmark(num_candidates=200)
    res = bench.kos()

    assert res["total_candidates_screened"] == 200
    assert res["stable_hea_alloys_found"] > 0
    assert res["hea_solid_solution_yield_pct"] > 5.0
    assert res["max_predicted_tc_kelvin"] > 0.0
