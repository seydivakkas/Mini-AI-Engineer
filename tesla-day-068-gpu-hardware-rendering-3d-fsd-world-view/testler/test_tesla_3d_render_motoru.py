"""
Tesla 3D Render Motoru Birim Testleri (PyTest)
==============================================
Bu test paketi; Model-View-Projection (MVP) matris hesaplarını,
kırpma ve ekran izdüşüm koordinatlarını test eder.

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

from src.tesla_3d_render_motoru import Tesla3DWorldRenderer


def test_projeksiyon_matrisi_boyut_ve_degerleri():
    """Perspektif projeksiyon matrisinin 4x4 ve geçerli değerlerde olduğu test edilir."""
    renderer = Tesla3DWorldRenderer(screen_width=1920, screen_height=1200, fov_deg=60.0)
    P = renderer.compute_projection_matrix()

    assert P.shape == (4, 4)
    assert P[3, 2] == -1.0
    assert P[0, 0] > 0.0
    assert P[1, 1] > 0.0


def test_3d_ekran_izdusum_donusumu():
    """3D noktaların ekran koordinatlarına (u, v) doğru dönüştüğü test edilir."""
    renderer = Tesla3DWorldRenderer(screen_width=1920, screen_height=1200)
    pts_3d = np.array([[0.0, 10.0, 0.0]], dtype=np.float32)

    cam_pos = np.array([0.0, -8.0, 3.2], dtype=np.float32)
    target_pos = np.array([0.0, 15.0, 0.5], dtype=np.float32)
    V = renderer.compute_view_matrix(cam_pos, target_pos)
    P = renderer.compute_projection_matrix()
    M = renderer.compute_model_matrix(np.zeros(3, dtype=np.float32))
    MVP = P @ V @ M

    screen_coords, depth = renderer.project_3d_points_to_screen(pts_3d, MVP)

    assert screen_coords.shape == (1, 2)
    # Nokta ekranın içinde olmalı
    assert 0.0 <= screen_coords[0, 0] <= 1920.0
    assert 0.0 <= screen_coords[0, 1] <= 1200.0


def test_fsd_sahne_render_zinciri():
    """Tam FSD sahnesinin başarıyla render edildiği ve tepe noktalarının üretildiği test edilir."""
    renderer = Tesla3DWorldRenderer()
    scene = renderer.render_fsd_scene()

    assert scene["num_rendered_vertices"] > 50
    assert len(scene["ego_screen_pts"]) == 8
    assert len(scene["left_lane_screen"]) == 25
    assert len(scene["path_screen"]) == 25
