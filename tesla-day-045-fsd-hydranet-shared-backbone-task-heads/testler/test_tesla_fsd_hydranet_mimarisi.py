"""
Tesla FSD HydraNet Birim Testleri (PyTest)
==========================================
Bu test paketi; Paylaşılan Omurga öznitelik çıkarımını, 4 Görev Kafasının
çıktılarını ve Homoscedastic çoklu görev kayıp fonksiyonunu test eder.

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

from src.tesla_fsd_hydranet_mimarisi import TeslaFSDHydraNet


def test_hydranet_omurga_ve_ciktilar():
    """HydraNet forward adımının tüm görev kafaları çıktılarını eksiksiz ürettiği test edilir."""
    net = TeslaFSDHydraNet(feature_dim=64)
    frame = np.random.uniform(0, 255, (128, 128, 3)).astype(np.float32)

    res = net.forward_hydranet(frame)

    assert "features" in res and len(res["features"]) == 64
    assert "objects" in res and len(res["objects"]["bbox_3d"]) == 7
    assert "lanes" in res and len(res["lanes"]["left_lane"]) == 4
    assert "traffic_light" in res and res["traffic_light"]["state"] in ["GREEN", "YELLOW", "RED"]
    assert "drivable_mask" in res and res["drivable_mask"].shape == (32, 32)


def test_coklu_gorev_kayip_fonksiyonu():
    """Homoscedastic belirsizlik ağırlıklı kaybın pozitif ve sonlu olduğu test edilir."""
    net = TeslaFSDHydraNet()
    task_losses = {
        "object": 0.5,
        "lane": 0.3,
        "traffic_light": 0.1,
        "drivable": 0.2
    }

    total_loss = net.compute_multi_task_loss(task_losses)
    assert total_loss > 0.0
    assert np.isfinite(total_loss)


def test_serit_polinom_geometrisi():
    """3. derece şerit polinomunun sol şerit için negatif, sağ şerit için pozitif Y ürettiği test edilir."""
    net = TeslaFSDHydraNet()
    features = np.zeros(64)
    lanes = net.lane_prediction_head(features)

    # x = 0 noktasında şeritler -1.85m ve +1.85m olmalıdır
    y_left_0 = lanes["left_lane"][0]
    y_right_0 = lanes["right_lane"][0]

    assert np.isclose(y_left_0, -1.85)
    assert np.isclose(y_right_0, 1.85)
