"""
Tesla Batarya Dijital İkiz Birim Testleri (PyTest)
==================================================
Bu test paketi; 96S paket gerilim toplamını, termal gradyan dağılımını ve
tekil hücre termal anomali erken tespit mekanizmasını test eder.

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

from src.tesla_dijital_ikiz_simulasyonu import TeslaBatteryPackDigitalTwin


def test_96s_paket_baslatma_ve_gerilim():
    """96S paketin toplam geriliminin ~380-400V arasında başladığı test edilir."""
    twin = TeslaBatteryPackDigitalTwin(num_series_cells=96, seed=42)
    assert len(twin.cells) == 96

    out = twin.step(pack_current_a=0.0, dt_s=0.1)
    assert 370.0 < out["v_pack"] < 410.0
    assert out["anomaly_flag"] is False


def test_termal_gradyan_dagilimi():
    """Soğutma sıvısı çıkışındaki son hücrelerin ilk hücrelerden daha sıcak olduğu test edilir."""
    twin = TeslaBatteryPackDigitalTwin(num_series_cells=96, seed=42)
    out = twin.step(pack_current_a=50.0, dt_s=1.0)

    # Son hücre sıcaklığı ilk hücreden yüksek olmalıdır
    assert out["cell_temperatures"][-1] > out["cell_temperatures"][0]
    assert out["t_gradient_c"] > 0.0


def test_termal_kacak_anomali_tespiti():
    """Hücre #30'a anomali enjekte edildiğinde sistemin doğru hücreyi tespit ettiği test edilir."""
    twin = TeslaBatteryPackDigitalTwin(num_series_cells=96, seed=42)
    twin.inject_thermal_anomaly(cell_id=30)

    # 50 adım 100A yüksek akım deşarjı
    for _ in range(50):
        out = twin.step(pack_current_a=100.0, dt_s=0.1)

    assert out["anomaly_flag"] is True
    assert out["faulty_cell_id"] == 30
