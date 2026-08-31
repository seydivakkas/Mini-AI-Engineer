"""
Tesla Faz 3 Büyük Capstone Birim Testleri (PyTest)
==================================================
Bu test paketi; 96S ECM, FOC Motor, Rejeneratif Frenleme ve HVIL Acil Güvenlik
fonksiyonlarının uçtan uca entegrasyonunu test eder.

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

from src.tesla_faz3_capstone_bms_ve_cekis_cekirdegi import CapstonePowertrainCore


def test_tam_gaz_ivmelenme_ve_voltaj_cokmesi():
    """Tam gaz ivmelenmede pozitif tork üretildiği, hızın arttığı ve bataryada voltaj çökmesi olduğu test edilir."""
    pt = CapstonePowertrainCore()
    init_voltage = pt.pack_voltage_v

    # 100 adım %100 gaz
    for _ in range(100):
        out = pt.step_powertrain_cycle(accel_pedal_pct=100.0, brake_pedal_pct=0.0, target_speed_kmh=100.0, dt_s=0.01)

    assert out["safe"] is True
    assert out["speed_kmh"] > 0.0
    assert out["torque_nm"] > 100.0
    assert out["power_kw"] > 0.0
    assert out["pack_voltage_v"] < init_voltage  # Voltaj çöker


def test_rejeneratif_yavaslama_ve_sarj():
    """Yüksek hızda gaz bırakıldığında negatif rejen torku üretildiği test edilir."""
    pt = CapstonePowertrainCore()
    pt.vehicle_speed_kmh = 80.0

    out = pt.step_powertrain_cycle(accel_pedal_pct=0.0, brake_pedal_pct=0.0, target_speed_kmh=0.0, dt_s=0.01)

    assert out["safe"] is True
    assert out["torque_nm"] < 0.0  # Negatif rejen torku
    assert out["power_kw"] < 0.0   # Bataryaya şarj girer


def test_hvil_kesilmesinde_acil_tork_kesme():
    """HVIL hattı açıldığında torkun derhal sıfırlandığı ve kontaktörlerin açıldığı test edilir."""
    pt = CapstonePowertrainCore()
    pt.vehicle_speed_kmh = 50.0
    pt.hvil_closed = False  # Arıza enjeksiyonu

    out = pt.step_powertrain_cycle(accel_pedal_pct=50.0, brake_pedal_pct=0.0, target_speed_kmh=100.0, dt_s=0.01)

    assert out["safe"] is False
    assert out["torque_nm"] == 0.0
    assert pt.main_contactors_closed is False
