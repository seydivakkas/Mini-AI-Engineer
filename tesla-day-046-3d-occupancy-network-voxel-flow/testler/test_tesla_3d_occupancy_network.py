"""
Tesla 3D Occupancy Network Birim Testleri (PyTest)
==================================================
Bu test paketi; 3D Voksel doluluk matrisini, Sigmoid olasılık eşiklemesini,
3D Voxel Flow hız sorgularını ve düzensiz engel tespitini test eder.

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

from src.tesla_3d_occupancy_network import Tesla3DOccupancyNetwork


def test_3d_voksel_boyutlari_ve_olasiliklar():
    """Voksel matrisinin (50, 50, 16) boyutunda ve olasılıkların [0, 1] aralığında olduğu test edilir."""
    occ = Tesla3DOccupancyNetwork(grid_dim_x=50, grid_dim_y=50, grid_dim_z=16)
    occ.insert_synthetic_scene()

    probs, binary_mask = occ.compute_occupancy_probabilities(threshold=0.5)

    assert probs.shape == (50, 50, 16)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.sum(binary_mask) > 0


def test_3d_voxel_flow_hiz_kestirimi():
    """Öncü aracın 15 m/s boyuna hızının, yayanın -1.2 m/s yanal hızının doğru sorgulandığı test edilir."""
    occ = Tesla3DOccupancyNetwork()
    occ.insert_synthetic_scene()

    # Öncü Araç (X = +15m)
    prob_car, flow_car = occ.query_point_velocity(x_m=15.0, y_m=0.0, z_m=1.0)
    assert prob_car > 0.8
    assert np.isclose(flow_car[0], 15.0, atol=0.5)

    # Yürüyen Yaya (Y = +6m, X = +5m)
    prob_ped, flow_ped = occ.query_point_velocity(x_m=5.0, y_m=6.0, z_m=1.0)
    assert prob_ped > 0.8
    assert np.isclose(flow_ped[1], -1.2, atol=0.2)


def test_duzensiz_engel_yakalama():
    """Geleneksel 3D kutulara sığmayan devrilmiş ağacın dolu voksel olarak yakalandığı test edilir."""
    occ = Tesla3DOccupancyNetwork()
    occ.insert_synthetic_scene()

    prob_tree, _ = occ.query_point_velocity(x_m=20.0, y_m=0.0, z_m=1.0)
    assert prob_tree > 0.8
