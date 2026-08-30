"""
Day 379: Unit Tests for Co-Packaged Optics (CPO) High-Speed Transceiver Modeling
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cpo_transceiver_motoru import (
    PAM4Encoder,
    MZMModulator,
    OpticalFiberChannel,
    PhotodiodeTIA,
    CPOTransceiverLink,
    CPOBenchmark
)


def test_pam4_gray_kodlama_ve_dekodlama():
    """PAM4 Gray kodlayıcı ve dekodlayıcısının seviye eşleşmelerini test eder."""
    encoder = PAM4Encoder(baud_rate_gbaud=56.0)
    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
    
    symbols = encoder.encode(bits)
    assert len(symbols) == 4, "8 bit 4 PAM4 sembolü üretmelidir."
    np.testing.assert_array_equal(symbols, [-3.0, -1.0, 1.0, 3.0])

    decoded_bits = encoder.decode(symbols)
    np.testing.assert_array_equal(decoded_bits, bits)


def test_mzm_elektro_optik_modulasyon():
    """Mach-Zehnder Modülatörünün Cos^2(V) optik güç çıkışını test eder."""
    mzm = MZMModulator(v_pi_v=1.5, laser_power_mw=10.0, extinction_ratio_db=6.5)
    
    # 0V sürüşte maksimum güç çıkmalı: P_out(0) = 10 mW
    v_zero = np.array([0.0])
    p_zero = mzm.modulate(v_zero)
    assert abs(p_zero[0] - 10.0) < 1e-4

    # V_pi (1.5V) sürüşte minimum sönümleme gücü çıkmalı
    v_pi = np.array([1.5])
    p_pi = mzm.modulate(v_pi)
    assert p_pi[0] < p_zero[0]
    assert p_pi[0] > 0.0, "Extinction ratio sebebiyle sıfırın üstünde kalmalıdır."


def test_fotodiyot_tia_ve_gurultu_alicisi():
    """Fotodiyot akım üretimi ve TIA voltaj dönüşümünü test eder."""
    receiver = PhotodiodeTIA(responsivity_a_w=0.8, tia_gain_ohm=1000.0, noise_std_mv=0.0)
    p_in = np.array([5.0])  # 5 mW
    
    v_out = receiver.receive(p_in, seed=42)
    # I_ph = 5 mW * 0.8 A/W = 4.0 mA -> V_out = 4.0 mA * 1.0 kOhm = 4000 mV
    assert abs(v_out[0] - 4000.0) < 1e-2


def test_tam_cpo_benchmark_ve_enerji_tasarrufu():
    """Tam 800G CPO bağlantı simülasyonunu ve 4.7x enerji tasarrufunu test eder."""
    bench = CPOBenchmark()
    res = bench.kos(num_symbols=2000)

    assert res["aggregate_data_rate_gbps"] >= 800.0, "Toplam bant genişliği en az 800 Gbps olmalıdır."
    assert res["cpo_energy_pj_bit"] <= 4.0, "CPO enerji tüketimi <= 4.0 pJ/bit olmalıdır."
    assert res["energy_savings_x"] >= 4.5, "Klasik optiğe göre en az 4.5x enerji tasarrufu sağlanmalıdır."
    assert res["ber"] < 0.05, "Ham BER KP4 FEC eşiği toleransında olmalıdır."
