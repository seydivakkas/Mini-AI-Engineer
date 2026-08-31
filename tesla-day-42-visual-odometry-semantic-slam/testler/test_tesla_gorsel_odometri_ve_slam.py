"""
Tesla Görsel Odometri ve SLAM Birim Testleri (PyTest)
=====================================================
Bu test paketi; 3D-to-2D projeksiyonunu, PnP RANSAC poz kestirimini,
semantik dinamik filtrelemeyi ve döngü kapatma (Loop Closure) mantığını test eder.

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

from src.tesla_gorsel_odometri_ve_slam import TeslaVisualOdometrySLAM


def test_3d_to_2d_projeksiyon():
    """Kamera merkezinin tam karşısındaki [0, 0, 10] noktasının ana noktaya (cx, cy) düştüğü test edilir."""
    slam = TeslaVisualOdometrySLAM(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
    P_3d = np.array([0.0, 0.0, 10.0])
    R = np.eye(3)
    t = np.zeros((3, 1))

    uv = slam.project_3d_to_2d(P_3d, R, t)
    assert np.isclose(uv[0], 640.0)
    assert np.isclose(uv[1], 360.0)


def test_semantik_dinamik_filtreleme():
    """Dinamik olarak etiketlenen noktaların PnP RANSAC tarafından elendiği test edilir."""
    slam = TeslaVisualOdometrySLAM()
    pts_3d = np.random.uniform(-10, 10, (20, 3))
    pts_3d[:, 2] = np.random.uniform(5, 20, 20)
    pts_2d = np.zeros((20, 2))
    for i in range(20):
        pts_2d[i] = slam.project_3d_to_2d(pts_3d[i], np.eye(3), np.zeros((3, 1)))

    # Tüm noktalar dinamik (etiket 1) işaretlenirse inlier 0 dönmelidir
    all_dynamic = np.ones(20, dtype=int)
    _, _, inliers, _ = slam.estimate_pose_pnp_ransac(pts_3d, pts_2d, semantic_labels=all_dynamic)
    assert inliers == 0


def test_keyframe_ve_loop_closure():
    """1.5 metreden fazla ilerlendiğinde keyframe eklendiği ve başlangıca dönüldüğünde döngü kapandığı test edilir."""
    slam = TeslaVisualOdometrySLAM()

    # 1. Başlangıç keyframe'i
    is_kf, is_l = slam.check_keyframe_and_loop_closure(np.array([[0.0], [0.0], [0.0]]))
    assert is_kf is True
    assert is_l is False

    # 2. 20 metre ileri git (Birçok keyframe ekle)
    for z in range(2, 25, 2):
        slam.check_keyframe_and_loop_closure(np.array([[0.0], [0.0], [float(z)]]))

    # 3. Başlangıç noktasına geri dön (Loop Closure tetiklenmeli)
    _, is_loop = slam.check_keyframe_and_loop_closure(np.array([[0.1], [0.0], [0.1]]))
    assert is_loop is True
