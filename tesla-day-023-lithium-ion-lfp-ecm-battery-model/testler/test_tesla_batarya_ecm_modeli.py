"""
Tesla Batarya ECM Modeli Birim Testleri (PyTest)
================================================
Bu test paketi; LFP ve NMC OCV modellerini, 2-RC dinamik voltaj yanıtını,
Arrhenius sıcaklık denklemini ve şarj/deşarj sınırlarını test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import sys
import os
import numpy as np

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_batarya_ecm_modeli import (
    TeslaBatteryECM,
    BatteryCellParameters,
    BatteryChemistry
)


def test_ocv_egrileri_lfp_ve_nmc():
    """LFP'nin düz plato, NMC'nin eğimli OCV karakteristiğine sahip olduğu test edilir."""
    params_lfp = BatteryCellParameters(chemistry=BatteryChemistry.LFP)
    params_nmc = BatteryCellParameters(chemistry=BatteryChemistry.NMC)

    ecm_lfp = TeslaBatteryECM(params_lfp)
    ecm_nmc = TeslaBatteryECM(params_nmc)

    # %50 SoC OCV değerleri
    ocv_lfp_50 = ecm_lfp.compute_ocv(0.50)
    ocv_nmc_50 = ecm_nmc.compute_ocv(0.50)

    assert 3.20 <= ocv_lfp_50 <= 3.35
    assert 3.50 <= ocv_nmc_50 <= 3.90

    # LFP %30 ile %70 arasında çok az değişmeli
    assert abs(ecm_lfp.compute_ocv(0.70) - ecm_lfp.compute_ocv(0.30)) < 0.15


def test_desarj_ve_voltaj_cokmesi():
    """Pozitif akımla deşarj olurken terminal voltajının OCV'nin altına düştüğü test edilir."""
    params = BatteryCellParameters(chemistry=BatteryChemistry.NMC)
    ecm = TeslaBatteryECM(params, initial_soc=0.90, initial_temp_c=25.0)

    # 100A Deşarj
    res = ecm.step(current_a=100.0, dt_s=1.0)

    assert res["soc"] < 0.90
    assert res["v_terminal"] < res["ocv_v"]
    assert res["v_rc1"] > 0.0
    assert res["v_rc2"] > 0.0


def test_rejenerasyon_sarj_dinamikleri():
    """Negatif akımla şarj olurken terminal voltajının OCV'nin üzerine çıktığı test edilir."""
    params = BatteryCellParameters(chemistry=BatteryChemistry.NMC)
    ecm = TeslaBatteryECM(params, initial_soc=0.50, initial_temp_c=25.0)

    # -50A Şarj / Fren Rejenerasyonu
    res = ecm.step(current_a=-50.0, dt_s=1.0)

    assert res["soc"] > 0.50
    assert res["v_terminal"] > res["ocv_v"]


def test_arrhenius_sicaklik_etkisi():
    """Soğuk havalarda (-10°C) iç direncin normal sıcaklığa (25°C) göre belirgin arttığı test edilir."""
    params = BatteryCellParameters(chemistry=BatteryChemistry.NMC)
    ecm_warm = TeslaBatteryECM(params, initial_temp_c=25.0)
    ecm_cold = TeslaBatteryECM(params, initial_temp_c=-10.0)

    r0_warm = ecm_warm.get_temperature_adjusted_r0()
    r0_cold = ecm_cold.get_temperature_adjusted_r0()

    assert r0_cold > r0_warm * 2.0  # -10°C'de iç direnç 2 kattan fazla artmalıdır
