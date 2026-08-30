"""
Day 339: Biocompatible BCI Implant Communication Protocol & Cryptographic Telemetry
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

from src.crypto_telemetry_motoru import (
    NeuralSpikeFrame,
    LightweightAEADCrypto,
    BiocompatibleTelemetryLink,
    SecurityError,
)
from src.telemetry_profilleyici import TelemetryProfilleyici


def test_neural_spike_frame_encoding_decoding():
    """
    Nöronal Spike Paket Kodlama ve Çözme Testi.
    """
    encoder = NeuralSpikeFrame(implant_id=0x00A1)
    spikes = (np.random.rand(1024) > 0.8).astype(np.uint8)
    
    frame_bytes = encoder.encode_frame(sequence_no=50, timestamp_ms=1000, spike_mask=spikes)
    decoded = encoder.decode_frame(frame_bytes)
    
    assert decoded["implant_id"] == 0x00A1
    assert decoded["sequence_no"] == 50
    assert decoded["crc_valid"] == True


def test_lightweight_aead_crypto_encryption():
    """
    AEAD Şifreleme ve Çözme Doğrulaması.
    """
    crypto = LightweightAEADCrypto(secret_key=b"TEST_KEY_1234567")
    plaintext = b"SECRET_NEURAL_TELEMETRY_DATA_1234"
    nonce = os.urandom(12)
    
    ciphertext, auth_tag = crypto.encrypt_payload(plaintext, nonce)
    decrypted = crypto.decrypt_payload(ciphertext, nonce, auth_tag)
    
    assert decrypted == plaintext
    assert len(auth_tag) == 16


def test_crypto_tamper_attack_rejection():
    """
    Tahrifat Saldırısı Engelleme Testi.
    """
    crypto = LightweightAEADCrypto(secret_key=b"TEST_KEY_1234567")
    plaintext = b"SECRET_DATA"
    nonce = os.urandom(12)
    
    ciphertext, auth_tag = crypto.encrypt_payload(plaintext, nonce)
    tampered_ciphertext = bytes([ciphertext[0] ^ 0xFF]) + ciphertext[1:]
    
    with pytest.raises(SecurityError):
        crypto.decrypt_payload(tampered_ciphertext, nonce, auth_tag)


def test_biocompatible_telemetry_power_safety():
    """
    Termal Güç Güvenlik Sınırı Testi (< 15 mW).
    """
    link = BiocompatibleTelemetryLink(voltage_v=1.8, current_ma=2.0)
    assert link.power_mw == 3.6
    assert link.is_thermally_safe() == True
