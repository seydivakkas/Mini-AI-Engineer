"""
Tesla Sensör Füzyonu Birim Testleri (PyTest)
============================================
Bu test paketi; 6-durumlu EKF durum tahminini, kamera ve radar ölçüm
güncellemelerini ve Mahalanobis kapılama (Gating) mekanizmasını test eder.

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

from src.tesla_sensor_fuzyonu_ekf_ukf import TeslaSensorFusionEKF


def test_ekf_tahmin_adimi():
    """10 m/s hızla 0.1 saniyede konumun 1 metre ilerlediği test edilir."""
    fusion = TeslaSensorFusionEKF(init_x=0.0, init_y=0.0, init_vx=10.0, init_vy=0.0)
    fusion.predict(dt_s=0.1)

    assert np.isclose(fusion.x[0], 1.0)
    assert np.isclose(fusion.x[2], 10.0)


def test_kamera_olcumu_ve_kovaryans_dususu():
    """Kamera ölçümü yapıldığında konum belirsizliğinin (P matrisi) azaldığı test edilir."""
    fusion = TeslaSensorFusionEKF(init_x=10.0, init_y=0.0)
    init_p_trace = np.trace(fusion.P)

    accepted = fusion.update_camera(np.array([10.2, 0.1]))
    assert accepted is True
    assert np.trace(fusion.P) < init_p_trace


def test_radar_nonlineer_jacobian_guncelleme():
    """Radar polar ölçümünün [r, theta, r_dot] doğru işlendiği test edilir."""
    fusion = TeslaSensorFusionEKF(init_x=20.0, init_y=0.0, init_vx=5.0, init_vy=0.0)

    # 20.1 metre, 0 radyan, 5.0 m/s radar ölçümü
    z_radar = np.array([20.1, 0.0, 5.0])
    accepted = fusion.update_radar(z_radar)

    assert accepted is True
    assert 19.5 < fusion.x[0] < 20.5


def test_mahalanobis_outlier_reddi():
    """Fiziksel olarak imkânsız sahte bir ölçümün Mahalanobis kapısı tarafından reddedildiği test edilir."""
    fusion = TeslaSensorFusionEKF(init_x=10.0, init_y=0.0)

    # Aniden 100 metre uzakta sahte kamera tespiti
    z_fake = np.array([100.0, 50.0])
    accepted = fusion.update_camera(z_fake)

    assert accepted is False  # Outlier reddedilmeli
    assert fusion.x[0] < 20.0  # Durum bozulmamalı
