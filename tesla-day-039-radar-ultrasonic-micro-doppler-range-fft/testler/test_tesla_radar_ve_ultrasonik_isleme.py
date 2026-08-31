"""
Tesla Radar ve Ultrasonik Birim Testleri (PyTest)
=================================================
Bu test paketi; 2D Range-Doppler FFT matrisini, CA-CFAR dinamik eşikleme
algoritmasını ve Ultrasonik ToF sıcaklık kompanzasyonunu test eder.

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

from src.tesla_radar_ve_ultrasonik_isleme import TeslaRadarAndUltrasonicProcessor


def test_radar_range_doppler_fft_boyutlari():
    """2D FFT çıktısının (64, 256) boyutunda ve değerlerin reel (dB) olduğu test edilir."""
    processor = TeslaRadarAndUltrasonicProcessor()
    raw = processor.generate_synthetic_radar_frame(target_range_m=20.0, target_speed_mps=-5.0)

    rd_map = processor.compute_range_doppler_fft(raw)
    assert rd_map.shape == (64, 256)
    assert np.all(np.isfinite(rd_map))


def test_ca_cfar_hedef_yakalama():
    """Arka plan gürültüsü içinde belirgin bir pik olduğunda CA-CFAR'ın hedefi bulduğu test edilir."""
    processor = TeslaRadarAndUltrasonicProcessor()
    signal = np.ones(100) * 10.0  # 10 dB gürültü tabanı
    signal[50] = 30.0  # 50. hücrede 30 dB hedef piki

    detections = processor.ca_cfar_1d(signal, num_train=16, num_guard=4, threshold_offset_db=8.0)
    assert detections[50] == True
    assert detections[10] == False


def test_ultrasonik_tof_sicaklik_kompanzasyonu():
    """Sıcak havadaki (40°C) ses hızının soğuk havadan (-10°C) daha hızlı olduğu ve mesafeyi etkilediği test edilir."""
    processor = TeslaRadarAndUltrasonicProcessor()
    t_echo = 0.010  # 10 ms yankı

    d_hot = processor.compute_ultrasonic_distance(t_echo, ambient_temp_c=40.0)
    d_cold = processor.compute_ultrasonic_distance(t_echo, ambient_temp_c=-10.0)

    assert d_hot > d_cold
    assert 1.5 < d_hot < 2.0
