"""
Tesla V12 UI Model Birim Testleri (PyTest)
==========================================
Bu test paketi; C++ Q_PROPERTY model veri akışını,
sinyal tetikleme mekanizmasını ve 60 FPS yayın döngüsünü test eder.

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

from src.tesla_v12_ui_model import TeslaV12VehicleModel


def test_hiz_ozelligi_ve_sinyal_yayilimi():
    """Hız güncellendiğinde sinyalin listeye eklendiği ve değerin sınırlandığı test edilir."""
    model = TeslaV12VehicleModel()
    degisti = model.set_speed(124.5)

    assert degisti is True
    assert np.isclose(model.speed_kmh, 124.5)
    assert any("speedChanged(124.5 km/h)" in s for s in model._signals_emitted)


def test_vites_ve_fsd_baglama():
    """Vites ve FSD durumlarının başarıyla bağlandığı test edilir."""
    model = TeslaV12VehicleModel()
    model.set_gear("D")
    model.set_fsd_active(True)

    assert model.gear == "D"
    assert model.fsd_active is True


def test_60fps_ekran_yayini():
    """60 karelik QML akışının başarıyla simüle edildiği test edilir."""
    model = TeslaV12VehicleModel()
    res = model.simulate_ui_stream(frames=60)

    assert res["frames"] == 60
    assert res["is_60fps_ready"] is True
    assert len(res["speeds_stream"]) == 60
    assert res["final_speed_kmh"] > 90.0
