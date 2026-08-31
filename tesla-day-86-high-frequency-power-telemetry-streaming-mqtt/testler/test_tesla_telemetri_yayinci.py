"""
Tesla Telemetri Birim Testleri (PyTest)
=======================================
Bu test paketi; 32 baytlık binary paket serileştirmeyi, kayan pencere
istatistiklerini ve halka arabellek taşma korumasını test eder.

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

from src.tesla_telemetri_yayinci import TeslaPowerTelemetryStreamer


def test_binary_serilestirme_ve_paket_boyutu():
    """Telemetrinin tam 32 bayt olarak paketlendiği ve kayıpsız açıldığı test edilir."""
    streamer = TeslaPowerTelemetryStreamer()

    raw = streamer.pack_telemetry(
        timestamp_ns=1700000000000000000,
        voltage_v=400.5,
        current_a=250.0,
        active_power_kw=100.125,
        reactive_power_kvar=12.5,
        freq_hz=50.02,
        temp_c=42.5
    )

    assert len(raw) == 32
    assert len(raw) == streamer.PACKET_SIZE_BYTES

    unpacked = streamer.unpack_telemetry(raw)
    assert unpacked["timestamp_ns"] == 1700000000000000000
    assert np.isclose(unpacked["voltage_v"], 400.5, atol=1e-2)
    assert np.isclose(unpacked["active_power_kw"], 100.125, atol=1e-2)


def test_kayan_pencere_istatistikleri():
    """100 örnekli kayan pencerenin ortalama ve limitleri doğru hesapladığı test edilir."""
    streamer = TeslaPowerTelemetryStreamer()

    for i in range(100):
        p = 100.0 + i  # 100 to 199
        streamer.push_sample(
            timestamp_ns=i,
            voltage_v=400.0,
            current_a=250.0,
            active_power_kw=p,
            reactive_power_kvar=0.0,
            freq_hz=50.0,
            temp_c=40.0
        )

    stats = streamer.get_sliding_window_stats()
    assert stats["count"] == 100
    assert stats["min_kw"] == 100.0
    assert stats["max_kw"] == 199.0
    assert np.isclose(stats["mean_kw"], 149.5, atol=0.1)


def test_halka_arabellek_kapasite_siniri():
    """Halka arabelleğin belirlenen kapasiteyi (50) aşmadığı ve eski veriyi döngüsel sildiği test edilir."""
    streamer = TeslaPowerTelemetryStreamer(buffer_capacity=50)

    for i in range(120):
        streamer.push_sample(
            timestamp_ns=i,
            voltage_v=400.0,
            current_a=200.0,
            active_power_kw=80.0,
            reactive_power_kvar=0.0,
            freq_hz=50.0,
            temp_c=40.0
        )

    assert len(streamer.ring_buffer) == 50
