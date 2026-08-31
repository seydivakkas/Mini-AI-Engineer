"""
Tesla LLC Dönüştürücü Birim Testleri (PyTest)
==============================================
Bu test paketi; LLC rezonans frekansını, SiC MOSFET sıcaklık direncini,
ZVS yumuşak anahtarlama avantajını ve %98.5+ verimliliği test eder.

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

from src.tesla_llc_donusturucu import TeslaLLCResonantConverter


def test_rezonans_frekansi_ve_sicaklik_direnci():
    """Rezonans frekansının yaklaşık 265 kHz olduğu ve Rdson'un sıcaklıkla arttığı test edilir."""
    conv = TeslaLLCResonantConverter(l_r_henry=15.0e-6, c_r_farad=24.0e-9)

    assert np.isclose(conv.f_r_hz / 1000.0, 265.26, atol=1.0)

    r_25 = conv.calculate_r_dson(25.0)
    r_75 = conv.calculate_r_dson(75.0)

    assert r_25 == 0.015
    assert r_75 > r_25
    assert np.isclose(r_75, 0.01875, atol=1e-4)


def test_zvs_ve_donusturucu_verimliligi():
    """ZVS aktifken dönüştürücü veriminin %98.5 üzerine çıktığı test edilir."""
    conv = TeslaLLCResonantConverter()
    res = conv.calculate_losses(i_rms_a=40.0, junction_temp_c=75.0, enable_zvs=True)

    assert res["efficiency_pct"] >= 98.5
    assert res["total_loss_w"] > 0.0
    assert res["p_switching_w"] < res["p_conduction_w"]


def test_zvs_kayip_tasarrufu():
    """ZVS'nin anahtarlama kaybını sert anahtarlamaya göre bariz azalttığı test edilir."""
    conv = TeslaLLCResonantConverter()
    res_zvs = conv.calculate_losses(i_rms_a=40.0, enable_zvs=True)
    res_hard = conv.calculate_losses(i_rms_a=40.0, enable_zvs=False)

    assert res_zvs["p_switching_w"] < res_hard["p_switching_w"]
    assert res_zvs["efficiency_pct"] > res_hard["efficiency_pct"]
