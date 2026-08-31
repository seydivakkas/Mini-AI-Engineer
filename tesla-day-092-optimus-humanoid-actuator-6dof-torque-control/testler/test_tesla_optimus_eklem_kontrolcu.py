"""
Tesla Optimus Eklem Kontrol Birim Testleri (PyTest)
===================================================
Bu test paketi; yerçekimi kompanzasyonunu, ters dinamik tork hesabını
ve empedans kontrolcüsünün yörünge yakınsamasını test eder.

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

from src.tesla_optimus_eklem_kontrolcu import TeslaOptimusJointController


def test_yercekimi_kompanzasyonu():
    """Yatay pozisyonda (q=0 rad) yerçekimi torkunun maksimum olduğu test edilir."""
    ctrl = TeslaOptimusJointController()
    q_zero = np.zeros(6)

    g_vec = ctrl.compute_gravity_vector(q_zero)

    assert len(g_vec) == 6
    assert np.all(g_vec > 0.0)
    assert np.isclose(g_vec[0], 4.5 * 9.81 * 0.35, atol=1e-2)


def test_ters_dinamik_tork_hesabi():
    """Atalet, Coriolis ve yerçekimi bileşenlerinin toplam torku oluşturduğu test edilir."""
    ctrl = TeslaOptimusJointController()
    q = np.zeros(6)
    q_dot = np.ones(6) * 0.5
    q_ddot = np.ones(6) * 2.0

    tau = ctrl.compute_inverse_dynamics_torque(q, q_dot, q_ddot)

    assert len(tau) == 6
    # İlk eklem: 12.5 * 2.0 + 2.5 * 0.5 + 15.45 = 25.0 + 1.25 + 15.45 = 41.7 Nm
    assert np.isclose(tau[0], 41.7, atol=0.5)


def test_empedans_kontrol_ve_hata_azalmasi():
    """Empedans kontrolcüsü ile konum hatasının azaldığı ve tork sınırının (150 Nm) aşılmadığı test edilir."""
    ctrl = TeslaOptimusJointController()
    q_curr = np.array([0.1, 0.2, -0.3, 0.4, -0.1, 0.05])
    q_des = np.array([0.5, 0.6, 0.0, 0.8, 0.2, 0.1])
    q_dot = np.zeros(6)

    res1 = ctrl.simulate_joint_step(q_curr, q_dot, q_des, dt_s=0.01)
    q_next = np.array(res1["q_next_rad"])
    q_dot_next = np.array(res1["q_dot_next_rad_s"])

    res2 = ctrl.simulate_joint_step(q_next, q_dot_next, q_des, dt_s=0.01)

    assert res1["max_joint_torque_nm"] <= 150.0
    assert res2["pos_error_norm_rad"] < np.linalg.norm(q_des - q_curr)
