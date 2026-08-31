"""
Tesla MPC Kinematik Kontrolcü Birim Testleri (PyTest)
======================================================
Bu test paketi; MPC durum geri besleme hesaplamasını, aktüatör doyum
sınırlarını ve kapalı çevrim takip hatası yakınsamasını test eder.

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

from src.tesla_mpc_kinematik_kontrolcu import TeslaKinematicMPCController


def test_mpc_aktuator_doyum_sinirlari():
    """Çok büyük hata durumunda bile komutların doyum sınırlarını aşmadığı test edilir."""
    controller = TeslaKinematicMPCController()
    acc, steer = controller.compute_optimal_control(
        lateral_error_m=10.0,
        heading_error_rad=1.0,
        speed_error_mps=20.0,
        current_speed_mps=15.0
    )

    assert -4.0 <= acc <= 2.5
    assert -0.55 <= steer <= 0.55


def test_mpc_yanal_hata_duzeltme_yonu():
    """Araç şeridin solundayken (e_y > 0) düzeltici sağ direksiyon komutu (steer < 0) üretildiği test edilir."""
    controller = TeslaKinematicMPCController()
    acc, steer = controller.compute_optimal_control(
        lateral_error_m=1.5,
        heading_error_rad=0.0,
        speed_error_mps=0.0,
        current_speed_mps=15.0
    )

    assert steer < 0.0  # Şerit merkezine doğru düzeltme


def test_mpc_kapali_cevrim_yakinsamasi():
    """40 adım sonunda yanal takip hatasının 0.10 metre (10 cm) altına indiği test edilir."""
    controller = TeslaKinematicMPCController()
    res = controller.simulate_closed_loop_tracking(init_lateral_err_m=1.2, init_heading_err_rad=0.15)

    assert res["is_converged"] is True
    assert res["final_lat_err_m"] < 0.10
    assert res["final_yaw_err_deg"] < 2.0
