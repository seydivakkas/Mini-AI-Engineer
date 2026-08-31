"""
Tesla Vision Park Asistanı Birim Testleri (PyTest)
==================================================
Bu test paketi; 3D Voxel doluluk ızgarasını, 360° ışın atma (Ray-Casting)
mesafe konturunu ve kademeli park ikaz durumlarını test eder.

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

from src.tesla_vision_park_asistani import TeslaVisionParkAssist


def test_doluluk_izgarasi_ve_engel_ekleme():
    """Nokta bulutunun doluluk ızgarasında doğru hücrelere yazıldığı test edilir."""
    park = TeslaVisionParkAssist(grid_resolution_m=0.1, grid_size_m=10.0)
    pts = np.array([[1.0, 2.0], [0.0, -1.0]])

    park.update_occupancy_and_memory(pts, ego_delta_x=0.0, ego_delta_y=0.0)
    # Merkez 50,50 -> (1.0/0.1) = +10 -> gx=60, (2.0/0.1) = +20 -> gy=70
    assert park.occupancy_grid[70, 60] > 0.5


def test_360_derece_isin_atma_mesafesi():
    """Yakındaki bir engelin 360 konturda tespit edildiği test edilir."""
    park = TeslaVisionParkAssist(grid_resolution_m=0.05, grid_size_m=10.0)
    # Ön tarafa (X = 2.0m, Y = 0.0m) engel koy
    pts = np.array([[2.0, 0.0]])
    park.update_occupancy_and_memory(pts, ego_delta_x=0.0, ego_delta_y=0.0)

    distances = park.compute_360_distance_contour(num_angles=36)
    valid_dists = distances[distances < 500.0]
    assert len(valid_dists) > 0


def test_kademeli_park_ikaz_durumlari():
    """Mesafeye göre STOP ve Uyarı durumlarının doğru belirlendiği test edilir."""
    park = TeslaVisionParkAssist()

    durum_stop, renk_stop = park.evaluate_park_warnings(25.0)
    assert durum_stop == "STOP"
    assert renk_stop == "#E82127"

    durum_uyari, _ = park.evaluate_park_warnings(45.0)
    assert "KRİTİK" in durum_uyari

    durum_guvenli, _ = park.evaluate_park_warnings(120.0)
    assert "GÜVENLİ" in durum_guvenli
