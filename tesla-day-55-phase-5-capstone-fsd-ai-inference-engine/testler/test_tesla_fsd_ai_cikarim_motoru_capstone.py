"""
Tesla Faz 5 Büyük Capstone Birim Testleri (PyTest)
===================================================
Bu test paketi; FSD AI Çıkarım Motorunun 10 alt bileşeninin
bütünleşik çalışmasını ve telemetri doğruluğunu test eder.

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

from src.tesla_fsd_ai_cikarim_motoru_capstone import TeslaFSDAIInferenceEngineCapstone


def test_fsd_ai_cikarim_motoru_tam_adim():
    """Tüm alt modellerin tek bir adımda hatasız telemetri ürettiği test edilir."""
    engine = TeslaFSDAIInferenceEngineCapstone()
    frame = np.ones((64, 64), dtype=np.float32)

    res = engine.step_fsd_ai_engine(frame, ego_speed_mps=20.0, human_steering_deg=0.0, human_accel_mps2=-1.2)

    assert "occupied_voxels" in res
    assert res["occupied_voxels"] > 0
    assert res["traffic_light"] == "RED"
    assert res["tl_countdown_sec"] > 0.0
    assert res["traffic_sign"] == "SPEED_70"
    assert len(res["legal_dag_lanes"]) > 0
    assert "KEEP" in res["trajectories"]


def test_fsd_ai_int8_ve_damitma_metrikleri():
    """INT8 bellek tasarrufu (%75) ve damıtma doğruluk korumasının (%99.2) sağlandığı test edilir."""
    engine = TeslaFSDAIInferenceEngineCapstone()
    frame = np.ones((64, 64), dtype=np.float32)

    res = engine.step_fsd_ai_engine(frame)

    assert res["int8_memory_saving_pct"] == 75.0
    assert res["distillation_accuracy_retention"] >= 99.0
    assert res["autolabel_3d_iou"] > 0.95
