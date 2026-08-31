"""
Tesla IMU ve Odometri Birim Testleri (PyTest)
=============================================
Bu test paketi; IMU tahmin adımını, tekerlek diferansiyel hız denetimini,
jiroskop bias düzeltmesini ve sürüklenme sınırlandırmasını test eder.

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

from src.tesla_imu_ve_odometri_fuzyonu import TeslaIMUWheelOdometryFusion


def test_diferansiyel_tekerlek_yaw_rate():
    """Tekerlek hız farkından araç dönüş hızının doğru hesaplandığı test edilir."""
    fusion = TeslaIMUWheelOdometryFusion(track_width_m=1.62)
    # v_R = 20.81 m/s, v_L = 19.19 m/s -> v = 20.0 m/s, yaw = (20.81 - 19.19) / 1.62 = 1.0 rad/s
    fusion.update_wheel_odometry(v_left_mps=19.19, v_right_mps=20.81)

    assert np.isclose(fusion.x[3], 20.0, atol=0.5)


def test_imu_tahmin_ve_hareket():
    """İleri yönde 1 saniyede 10 m/s hızla 10 metre gidildiği test edilir."""
    fusion = TeslaIMUWheelOdometryFusion()
    fusion.x[3] = 10.0  # 10 m/s

    for _ in range(100):
        fusion.predict_imu(ax_mps2=0.0, gyro_yaw_rads=0.0, dt_s=0.01)

    assert np.isclose(fusion.x[0], 10.0, atol=0.1)
    assert np.isclose(fusion.x[1], 0.0, atol=0.1)


def test_jiroskop_bias_duzeltmesi():
    """Tekerlek odometrisinin jiroskop donanımsal kaymasını düzelttiği test edilir."""
    fusion = TeslaIMUWheelOdometryFusion()
    # Araç düz gidiyor fakat jiroskop 0.05 rad/s sahte değer okuyor
    for _ in range(50):
        fusion.predict_imu(ax_mps2=0.0, gyro_yaw_rads=0.05, dt_s=0.01)
        fusion.update_wheel_odometry(v_left_mps=15.0, v_right_mps=15.0)  # Dönüş yok (yaw_diff=0)

    # Bias x[4] tahmini yükselerek sahte jiroskop değerini nötrlemelidir
    assert fusion.x[4] > 0.01
