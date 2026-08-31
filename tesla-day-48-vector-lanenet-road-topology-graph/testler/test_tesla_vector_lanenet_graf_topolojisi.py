r"""
Tesla VectorLaneNet Birim Testleri (PyTest)
===========================================
Bu test paketi; 3. Derece şerit polinomunu, analitik eğrilik ($\kappa$) hesabını,
kavşak yönlendirilmiş grafını (DAG) ve şerit geçiş mantığını test eder.

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

from src.tesla_vector_lanenet_graf_topolojisi import TeslaVectorLaneNet


def test_serit_polinom_hesabi():
    """Polinomun x = 0 ve x = 10 noktalarında doğru Y değerlerini ürettiği test edilir."""
    net = TeslaVectorLaneNet()
    poly = np.array([1.75, 0.0, 0.001, 0.0])  # y = 1.75 + 0.001*x^2

    y_0 = net.evaluate_lane_polynomial(poly, np.array([0.0]))[0]
    y_10 = net.evaluate_lane_polynomial(poly, np.array([10.0]))[0]

    assert np.isclose(y_0, 1.75)
    assert np.isclose(y_10, 1.75 + 0.001 * 100.0)


def test_analitik_egrilik_hesabi():
    """Düz şeritte eğriliğin 0, parabolik virajda pozitif olduğu test edilir."""
    net = TeslaVectorLaneNet()
    poly_straight = np.array([0.0, 0.0, 0.0, 0.0])
    poly_curve = np.array([0.0, 0.0, 0.01, 0.0])  # y'' = 0.02

    kappa_straight = net.compute_lane_curvature(poly_straight, x_val=5.0)
    kappa_curve = net.compute_lane_curvature(poly_curve, x_val=0.0)

    assert np.isclose(kappa_straight, 0.0)
    assert np.isclose(kappa_curve, 0.02)


def test_kavsak_graf_baglantilari():
    """Sol yaklaşım şeridinden (0) Sola Dönüş (2) ve Düz (3) seçeneklerinin döndüğü test edilir."""
    net = TeslaVectorLaneNet()
    net.construct_synthetic_intersection_graph()

    next_lanes = net.get_legal_next_lanes(current_lane_id=0)

    assert 2 in next_lanes  # Sola Dönüş
    assert 3 in next_lanes  # Düz
    assert 4 not in next_lanes  # Sol şeritten direkt sağa dönülemez
