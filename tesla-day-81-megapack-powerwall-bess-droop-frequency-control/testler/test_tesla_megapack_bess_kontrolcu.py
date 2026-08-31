"""
Tesla Megapack BESS Birim Testleri (PyTest)
===========================================
Bu test paketi; şebeke frekans sapmasında P-f Droop güç tepkisini,
batarya SoC sınırlarını ve şebeke stabilizasyonunu test eder.

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

from src.tesla_megapack_bess_kontrolcu import TeslaMegapackBESSController


def test_dusuk_frekansta_guc_enjeksiyonu():
    """Şebeke frekansı 49.8 Hz'e düştüğünde Megapack'in pozitif güç (deşarj) verdiği test edilir."""
    bess = TeslaMegapackBESSController(nominal_freq_hz=50.0, droop_gain_kw_per_hz=10000.0)
    p, action = bess.compute_active_droop_power(grid_freq_hz=49.8)

    # delta_f = 0.2 Hz -> P = 0.2 * 10000 = 2000 kW -> Clamped to 1950 kW
    assert p == 1950.0
    assert "ŞEBEKEYE GÜÇ ENJEKSİYONU" in action


def test_yuksek_frekansta_fazla_gucu_emme():
    """Şebeke frekansı 50.1 Hz'e çıktığında Megapack'in negatif güç (şarj) çektiği test edilir."""
    bess = TeslaMegapackBESSController(nominal_freq_hz=50.0, droop_gain_kw_per_hz=10000.0)
    p, action = bess.compute_active_droop_power(grid_freq_hz=50.1)

    # delta_f = -0.1 Hz -> P = -1000 kW
    assert np.isclose(p, -1000.0, atol=1e-2)
    assert "ŞEBEKEDEN FAZLA GÜCÜ EMME" in action


def test_bess_adim_ve_soc_guncellemesi():
    """Simülasyon adımında aktif/reaktif güç üretildiği ve SoC'nin güncellendiği test edilir."""
    bess = TeslaMegapackBESSController(initial_soc_pct=80.0)
    res = bess.step_bess_simulation(grid_freq_hz=49.9, grid_voltage_v=390.0, dt_s=1.0)

    assert np.isclose(res["active_power_kw"], 1000.0, atol=1e-2)  # 0.1 * 10000 = 1000 kW
    assert res["reactive_power_kvar"] > 0.0
    assert res["soc_pct"] < 80.0  # Deşarj edildiği için SoC azaldı
