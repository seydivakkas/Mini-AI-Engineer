"""
Tesla NeRF ve Otomatik Etiketleme Birim Testleri (PyTest)
=========================================================
Bu test paketi; Işın örneklemesini, Hacimsel Işın İzleme (Volume Rendering)
derinlik integralini ve 3D Bounding Box otomatik etiketlemesini test eder.

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

from src.tesla_nerf_ve_otomatik_etiketleme import TeslaNeRFAutoLabeler


def test_isin_ornekleme_geometrisi():
    """Işın boyunca örneklenen noktaların r(t) = o + t*d denklemine uyduğu test edilir."""
    labeler = TeslaNeRFAutoLabeler(num_samples_per_ray=10, near_m=1.0, far_m=10.0)
    ray_o = np.array([0.0, 0.0, 0.0])
    ray_d = np.array([0.0, 1.0, 0.0])

    pts, t_vals, delta = labeler.sample_ray_points(ray_o, ray_d)

    assert pts.shape == (10, 3)
    assert np.isclose(pts[0, 1], 1.0)
    assert np.isclose(pts[-1, 1], 10.0)


def test_hacimsel_isin_derinlik_integrali():
    """15 metredeki hedefe gönderilen ışının hacimsel derinliğinin ~15 metre çıktığı test edilir."""
    labeler = TeslaNeRFAutoLabeler(num_samples_per_ray=32, near_m=1.0, far_m=35.0)
    ray_o = np.array([0.0, 0.0, 0.0])
    ray_d = np.array([0.0, 1.0, 0.0])

    rgb, depth, opacity = labeler.render_volume_ray(ray_o, ray_d, object_center_3d=np.array([0.0, 15.0, 0.0]))

    assert 12.0 < depth < 17.0
    assert opacity > 0.5


def test_otomatik_3d_bbox_etiketleme():
    """NeRF rekonstrüksiyonundan 3D kutunun ve yüksek PSNR değerinin elde edildiği test edilir."""
    labeler = TeslaNeRFAutoLabeler()
    ray_origins = np.zeros((10, 3))
    ray_dirs = np.tile(np.array([0.0, 1.0, 0.0]), (10, 1))
    depths = np.full(10, 15.0)

    bbox_res = labeler.auto_label_3d_bounding_box(depths, ray_origins, ray_dirs)

    assert bbox_res["detected"] is True
    assert np.isclose(bbox_res["bbox_center"][1], 15.0)
    assert bbox_res["psnr_db"] > 30.0
