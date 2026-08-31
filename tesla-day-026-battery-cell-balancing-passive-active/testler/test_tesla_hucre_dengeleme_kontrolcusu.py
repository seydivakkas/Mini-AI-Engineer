"""
Tesla Hücre Dengeleme Kontrolcüsü Birim Testleri (PyTest)
=========================================================
Bu test paketi; Pasif Direnç Dengeleme, Aktif Endüktif Dengeleme ve
Aşırı Sıcaklık Termal Koruma mantıklarını test eder.

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

from src.tesla_hucre_dengeleme_kontrolcusu import (
    BatteryCell,
    TeslaBalancingController,
    BalancingStrategy
)


def test_pasif_dengeleme_gerilim_azaltma():
    """Pasif dengelemenin yüksek voltajlı hücreyi hedeflenen eşiğe indirdiği test edilir."""
    c1 = BatteryCell(cell_id=1, voltage_v=4.00, capacity_ah=5.0, soc=0.80)
    c2 = BatteryCell(cell_id=2, voltage_v=4.10, capacity_ah=5.0, soc=0.90)  # Yüksek
    cells = [c1, c2]

    ctrl = TeslaBalancingController(
        strategy=BalancingStrategy.PASSIVE_BLEEDING,
        voltage_threshold_mv=10.0,
        bleed_resistor_ohm=33.0
    )

    # 500 saniye dengeleme
    for _ in range(500):
        res = ctrl.step_balancing(cells, dt_s=1.0)

    assert c2.voltage_v < 4.10
    assert c2.soc < 0.90
    assert res["heat_w"] >= 0.0


def test_aktif_dengeleme_enerji_aktarimi():
    """Aktif dengelemede düşük hücrenin şarjının arttığı test edilir."""
    c1 = BatteryCell(cell_id=1, voltage_v=3.80, capacity_ah=5.0, soc=0.60)  # Düşük
    c2 = BatteryCell(cell_id=2, voltage_v=4.10, capacity_ah=5.0, soc=0.90)  # Yüksek
    cells = [c1, c2]

    ctrl = TeslaBalancingController(
        strategy=BalancingStrategy.ACTIVE_INDUCTIVE,
        voltage_threshold_mv=10.0,
        active_transfer_current_a=2.0,
        active_efficiency=0.88
    )

    init_soc_c1 = c1.soc
    init_soc_c2 = c2.soc

    for _ in range(100):
        ctrl.step_balancing(cells, dt_s=1.0)

    assert c1.soc > init_soc_c1  # Düşük hücre şarj aldı
    assert c2.soc < init_soc_c2  # Yüksek hücre deşarj oldu


def test_termal_koruma_kesme():
    """Hücre sıcaklığı max_temp_c üzerine çıktığında pasif anahtarın kapandığı test edilir."""
    c1 = BatteryCell(cell_id=1, voltage_v=3.80, capacity_ah=5.0, soc=0.60, temperature_c=25.0)
    c2 = BatteryCell(cell_id=2, voltage_v=4.10, capacity_ah=5.0, soc=0.90, temperature_c=65.0)  # Aşırı sıcak!
    cells = [c1, c2]

    ctrl = TeslaBalancingController(
        strategy=BalancingStrategy.PASSIVE_BLEEDING,
        voltage_threshold_mv=10.0,
        max_bleed_temp_c=55.0
    )

    res = ctrl.step_balancing(cells, dt_s=1.0)
    assert c2.bleed_switch_active is False
    assert res["heat_w"] == 0.0
