"""
Day 352: UAV Anti-Spoofing & Tamper-Proof Cryptographic Telemetry
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.anti_spoofing_crypto_motoru import (
    GNSSVIOKinematicResidualValidator,
    SecureTelemetryPacketAuth,
    TamperProofZeroizeEngine,
)
from src.crypto_profilleyici import CryptoProfilleyici


def test_gnss_vio_kinematic_validator_normal_vs_spoofed():
    """
    GNSS İnovasyon Kapısı ve Spoofing Ayrıştırma Testi.
    """
    validator = GNSSVIOKinematicResidualValidator(chi2_threshold=9.21)
    vio_pos = np.array([10.0, 10.0, 50.0])
    
    # 1. Normal GNSS (1 metre sapma)
    normal_gnss = np.array([10.5, 10.2, 50.1])
    res_norm = validator.validate_gnss_fix(normal_gnss, vio_pos)
    assert res_norm["is_spoofed"] is False
    assert res_norm["gnss_trusted"] is True
    
    # 2. Aldatılmış GNSS (50 metre sapma)
    spoofed_gnss = np.array([50.0, 60.0, 50.0])
    res_spoof = validator.validate_gnss_fix(spoofed_gnss, vio_pos)
    assert res_spoof["is_spoofed"] is True
    assert res_spoof["gnss_trusted"] is False


def test_secure_telemetry_packet_auth():
    """
    HMAC-SHA256 Telemetri Doğrulama ve Replay Koruması Testi.
    """
    key = b"TEST_KEY_1234567890"
    auth = SecureTelemetryPacketAuth(key)
    
    # 1. Geçerli paket
    pkt1 = auth.sign_telemetry(b"NAV_DATA_1", nonce=1)
    assert auth.verify_and_accept_packet(pkt1) is True
    
    # 2. Tekrar Saldırısı (Replay Attack - Eski Nonce 1)
    assert auth.verify_and_accept_packet(pkt1) is False
    
    # 3. Sahte İmza
    pkt2 = {"payload": b"NAV_DATA_2", "nonce": 2, "signature": "BAD_SIG"}
    assert auth.verify_and_accept_packet(pkt2) is False


def test_tamper_proof_zeroize_engine():
    """
    Donanımsal Zeroize Bellek İmha Testi.
    """
    engine = TamperProofZeroizeEngine("a1b2c3d4e5f6")
    assert engine.is_zeroized is False
    
    engine.check_tamper_sensors(chassis_open_sensor=True, impact_accel_g=10.0)
    assert engine.is_zeroized is True
    assert all(b == 0 for b in engine.volatile_key_register)


def test_crypto_profiler_metrics():
    """
    Kripto Profilleyici Metrik Testi.
    """
    metrics = CryptoProfilleyici.profille(
        spoofing_rejected=True,
        valid_packets=85,
        dropped_packets=15,
        zeroized=True
    )
    
    assert metrics["anti_spoofing_score"] == 100.0
    assert metrics["cyber_resilience_score"] == 100.0
