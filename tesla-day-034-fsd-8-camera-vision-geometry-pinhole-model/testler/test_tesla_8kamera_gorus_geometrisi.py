"""
Tesla 8-Kamera Görüş Geometrisi Birim Testleri (PyTest)
======================================================
Bu test paketi; 8 kameranın içsel ve dışsal matrislerini, Brown-Conrady
distorsiyon projeksiyonunu ve 360° görüş alanı kapsamasını test eder.

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

from src.tesla_8kamera_gorus_geometrisi import Tesla8CameraVisionRig, CameraModel


def test_8_kamera_kurulumu():
    """Tesla donanımında tam 8 adet kameranın tanımlı olduğu test edilir."""
    rig = Tesla8CameraVisionRig()
    assert len(rig.cameras) == 8
    cam_names = [c.name for c in rig.cameras]
    assert "Front_Main" in cam_names
    assert "Rear_View" in cam_names
    assert "Left_Pillar" in cam_names
    assert "Right_Repeater" in cam_names


def test_on_kamera_merkez_izdusumu():
    """Doğrudan aracın önündeki bir noktanın ön kameranın merkezine yakın izdüştüğü test edilir."""
    rig = Tesla8CameraVisionRig()
    front_main = [c for c in rig.cameras if c.name == "Front_Main"][0]

    # Tam ön 15 metrede nokta (X=15, Y=0, Z=1.35)
    pt_front = np.array([15.0, 0.0, 1.35])
    uv, z_cam = front_main.project_point_3d(pt_front)

    assert uv is not None
    u, v = uv
    # 1280x960 çözünürlükte merkez (640, 480) civarında olmalıdır
    assert 550.0 < u < 730.0
    assert 400.0 < v < 600.0
    assert z_cam > 10.0


def test_arka_kamera_gorunurluk():
    """Aracın arkasındaki bir noktanın ön kamerada görünmeyip arka kamerada göründüğü test edilir."""
    rig = Tesla8CameraVisionRig()
    front_main = [c for c in rig.cameras if c.name == "Front_Main"][0]
    rear_view = [c for c in rig.cameras if c.name == "Rear_View"][0]

    # Aracın arkasında 15 metrede nokta (X=-15, Y=0, Z=1.0)
    pt_rear = np.array([-15.0, 0.0, 1.0])

    uv_front, _ = front_main.project_point_3d(pt_rear)
    uv_rear, _ = rear_view.project_point_3d(pt_rear)

    assert uv_front is None  # Ön kamerada görünmemeli
    assert uv_rear is not None  # Arka kamerada görünmeli
