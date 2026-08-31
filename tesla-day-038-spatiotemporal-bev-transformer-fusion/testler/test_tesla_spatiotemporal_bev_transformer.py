"""
Tesla Spatiotemporal BEV Transformer Birim Testleri (PyTest)
============================================================
Bu test paketi; 3D BEV sorgu ızgarasını, Ego-Motion warp fonksiyonunu,
Spatial Cross-Attention ve Zamansal Bellek (Temporal Self-Attention) mekanizmasını test eder.

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

from src.tesla_spatiotemporal_bev_transformer import TeslaSpatiotemporalBEVTransformer


def test_bev_sorgu_izgarasi_boyutlari():
    """BEV tensör boyutlarının (50, 50, 64) doğru oluşturulduğu test edilir."""
    transformer = TeslaSpatiotemporalBEVTransformer(bev_h=50, bev_w=50, feature_dim=64)
    assert transformer.bev_queries.shape == (50, 50, 64)


def test_ego_motion_warp_kaydirma():
    """Araç 2 metre ileri gittiğinde harita tensörünün 2 birim geriye kaydığı test edilir."""
    transformer = TeslaSpatiotemporalBEVTransformer(bev_h=50, bev_w=50, feature_dim=8)
    bev_grid = np.zeros((50, 50, 8))
    bev_grid[20, 25, :] = 5.0  # (X=20, Y=25) noktasında engel

    # dx = 2.0 metre ileri hareket
    warped = transformer.ego_motion_warp(bev_grid, dx_m=2.0, dy_m=0.0, dyaw_rad=0.0, grid_res_m=1.0)

    # 2 metre ileri gidince engel aracın gerisine (20 - 2 = 18) kaymalıdır
    assert np.allclose(warped[18, 25, :], 5.0)
    assert np.allclose(warped[20, 25, :], 0.0)


def test_zamansal_bellek_ve_okluzyon_korumasi():
    """Görsel öznitelik sıfırlansa bile zamansal belleğin engeli hafızasında tuttuğu test edilir."""
    transformer = TeslaSpatiotemporalBEVTransformer(bev_h=50, bev_w=50, feature_dim=16)

    cam_feats = {"Front_Main": np.zeros((50, 50, 16))}
    cam_feats["Front_Main"][25, 25, :] = 4.0  # Güçlü engel tespiti

    # 1. Adım: Engel algılandı
    out1 = transformer.step(cam_feats, dx_m=0.0, dy_m=0.0, dyaw_rad=0.0)
    prob_step1 = out1["occupancy_prob"][25, 25]

    # 2. Adım: Oklüzyon (Kamera özniteliği sıfırlandı)
    cam_feats_zero = {"Front_Main": np.zeros((50, 50, 16))}
    out2 = transformer.step(cam_feats_zero, dx_m=0.0, dy_m=0.0, dyaw_rad=0.0)
    prob_step2 = out2["occupancy_prob"][25, 25]

    # Zamansal bellek sayesinde olasılık bir anda sıfıra düşmemeli, yüksek kalmalıdır
    assert prob_step1 > 0.60
    assert prob_step2 > 0.50
