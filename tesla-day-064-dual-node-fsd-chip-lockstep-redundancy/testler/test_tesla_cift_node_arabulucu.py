"""
Tesla Çift Düğüm Arabulucu Birim Testleri (PyTest)
==================================================
Bu test paketi; FSD Node A / Node B tam uzlaşı mekanizmasını,
failover düğüm devrini ve karar ayrışması güvenli duruşunu test eder.

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

from src.tesla_cift_node_arabulucu import FSDHardwareArbiter, ArbiterMode


def test_tam_uzlasi_oylamasi():
    """Her iki düğümün benzer çıkarımlar ürettiğinde ortalama komut uygulandığı test edilir."""
    arbiter = FSDHardwareArbiter()
    res = arbiter.arbitrate_decision(
        node_a_steer_rad=0.12,
        node_b_steer_rad=0.14,
        node_a_acc_mps2=1.0,
        node_b_acc_mps2=1.1,
        node_a_healthy=True,
        node_b_healthy=True
    )

    assert res["arbiter_mode"] == ArbiterMode.FULL_CONSENSUS.value
    assert np.isclose(res["applied_steering_rad"], 0.13)
    assert np.isclose(res["applied_acc_mps2"], 1.05)
    assert res["is_nominal"] is True


def test_failover_dugum_devri():
    """Node B arızalandığında Node A'nın komutlarının doğrudan uygulandığı test edilir."""
    arbiter = FSDHardwareArbiter()
    res = arbiter.arbitrate_decision(
        node_a_steer_rad=0.20,
        node_b_steer_rad=0.00,
        node_a_acc_mps2=0.5,
        node_b_acc_mps2=0.0,
        node_a_healthy=True,
        node_b_healthy=False
    )

    assert res["arbiter_mode"] == ArbiterMode.FAILOVER_NODE_A.value
    assert np.isclose(res["applied_steering_rad"], 0.20)
    assert np.isclose(res["applied_acc_mps2"], 0.5)


def test_karar_ayrismasi_guvenli_durus():
    """İki sağlıklı düğüm arasında büyük uyuşmazlık olduğunda DISCREPANCY_SAFE_STOP tetiklendiği test edilir."""
    arbiter = FSDHardwareArbiter()
    # Fark = 0.20 rad > 0.05 rad
    res = arbiter.arbitrate_decision(
        node_a_steer_rad=0.10,
        node_b_steer_rad=0.30,
        node_a_acc_mps2=1.0,
        node_b_acc_mps2=1.0,
        node_a_healthy=True,
        node_b_healthy=True
    )

    assert res["arbiter_mode"] == ArbiterMode.DISCREPANCY_SAFE_STOP.value
    assert res["applied_steering_rad"] == 0.0  # Düz hat
    assert res["applied_acc_mps2"] == -1.5     # Güvenli yavaşlama
    assert res["is_emergency_stop"] is True
