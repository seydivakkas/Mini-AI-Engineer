"""
Tesla Faz 4 Capstone Birim Testleri (PyTest)
=============================================
Bu test paketi; 8 Kamera, Radar, IMU, Odometri ve BEV Transformer'dan
oluşan entegre Faz 4 Capstone füzyon hattını uçtan uca test eder.

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

from src.tesla_faz4_capstone_bev_fuzyon_hatti import TeslaPhase4CapstonePipeline


def test_capstone_pipeline_adim_calismasi():
    """Tüm 8 kamera ve sensör verisiyle tek bir FSD adımının başarıyla çalıştığı test edilir."""
    pipeline = TeslaPhase4CapstonePipeline(bev_grid_size=60, bev_resolution_m=0.5)

    cam_projections = {cam: np.zeros((60, 60), dtype=np.float32) for cam in pipeline.camera_names}
    z_radar = np.array([25.0, 0.0, 15.0])
    imu = (0.0, 0.0)
    wheels = (15.0, 15.0)

    res = pipeline.process_fsd_step(cam_projections, z_radar, imu, wheels, dt_s=0.0277)

    assert "bev_occupancy_grid" in res
    assert res["bev_occupancy_grid"].shape == (60, 60)
    assert np.all((res["bev_occupancy_grid"] >= 0.0) & (res["bev_occupancy_grid"] <= 1.0))


def test_capstone_radar_ekf_takip_yakinsemasi():
    """Radar ölçümleri geldiğinde EKF'in öndeki aracın mesafesini ve hızını koruduğu test edilir."""
    pipeline = TeslaPhase4CapstonePipeline()
    cam_projections = {cam: np.zeros((60, 60), dtype=np.float32) for cam in pipeline.camera_names}

    for _ in range(20):
        z_radar = np.array([25.0, 0.0, 15.0])
        res = pipeline.process_fsd_step(cam_projections, z_radar, (0.0, 0.0), (15.0, 15.0))

    assert np.isclose(res["lead_distance_m"], 25.0, atol=1.0)
    assert np.isclose(res["lead_speed_mps"], 15.0, atol=1.0)


def test_capstone_dead_reckoning_ilerleme():
    """15 m/s hızla 1 saniyede yaklaşık 15 metre kat edildiği test edilir."""
    pipeline = TeslaPhase4CapstonePipeline()
    cam_projections = {cam: np.zeros((60, 60), dtype=np.float32) for cam in pipeline.camera_names}

    for _ in range(36):  # ~1 saniye (36 Hz)
        res = pipeline.process_fsd_step(cam_projections, None, (0.0, 0.0), (15.0, 15.0), dt_s=1.0/36.0)

    assert np.isclose(res["dead_reckoning_pose"][0], 15.0, atol=1.0)
