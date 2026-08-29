"""
PyTest Birim Testleri - Day 254: Kapalı Çevrim Dokunsal Geri Bildirim Kontrolü.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tactile_feedback_motoru import (
    TactileSlipDetector,
    AdaptiveStiffnessEstimator,
    ClosedLoopTactileController,
)
from src.tactile_feedback_profilleyici import TactileFeedbackProfilleyici
from src.gorsellestirici import TactileFeedbackGorsellestirici


def test_tactile_slip_detector_imminent_slip():
    """1. detect_slip sürtünme konisi sınırında kayma uyarısı vermelidir."""
    res = TactileSlipDetector.detect_slip(f_normal=1.0, f_tangential=0.50, mu_s=0.55)
    assert res["kayma_tehlikesi_var_mi"] is True
    assert res["surtunme_orani_eta"] == 0.50


def test_tactile_slip_detector_vibration():
    """2. detect_slip yüksek mikro titreşim enerjisinde kayma uyarısı vermelidir."""
    vib = np.ones(50) * 0.4
    res = TactileSlipDetector.detect_slip(f_normal=2.0, f_tangential=0.1, vib_signal=vib, mu_s=0.55)
    assert res["kayma_tehlikesi_var_mi"] is True


def test_adaptive_stiffness_estimator_fragile():
    """3. estimate_stiffness yumuşak nesne için düşük emniyet tavanı belirlemelidir."""
    res = AdaptiveStiffnessEstimator.estimate_stiffness(delta_f_N=0.5, delta_x_mm=2.0)
    assert res["sertlik_k_N_mm"] == 0.25
    assert "KIRILGAN" in res["nesne_sinifi"]
    assert res["maksimum_guvenli_kuvvet_N"] == 3.5


def test_adaptive_stiffness_estimator_rigid():
    """4. estimate_stiffness rijit nesne için yüksek emniyet tavanı belirlemelidir."""
    res = AdaptiveStiffnessEstimator.estimate_stiffness(delta_f_N=10.0, delta_x_mm=1.0)
    assert res["sertlik_k_N_mm"] == 10.0
    assert "RİJİT" in res["nesne_sinifi"]
    assert res["maksimum_guvenli_kuvvet_N"] == 25.0


def test_closed_loop_tactile_controller_init():
    """5. ClosedLoopTactileController başlangıç değerlerini doğru kurmalıdır."""
    ctrl = ClosedLoopTactileController(mu_s=0.55)
    assert ctrl.mu_s == 0.55
    assert ctrl.f_normal > 0.0


def test_closed_loop_tactile_controller_slip_compensation():
    """6. step_control kayma anında normal kuvveti emniyet tavanını aşmadan artırmalıdır."""
    ctrl = ClosedLoopTactileController(mu_s=0.55)
    ctrl.f_normal = 0.8
    # Yumurta için kayma anı (ft=0.7)
    res = ctrl.step_control(f_tangential=0.7, delta_x_gripper_mm=2.0, delta_f_sensor_N=0.4)
    assert res["durum"] == "KAYMA_KOMPANZASYONU"
    assert res["uygulanan_f_normal_N"] > 0.8
    assert res["uygulanan_f_normal_N"] <= res["emniyet_tavani_N"]
    assert res["ezilme_riski_var_mi"] is False


def test_tactile_feedback_profiler_output():
    """7. TactileFeedbackProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = TactileFeedbackProfilleyici.basarim_profili_cikar()
    assert "Closed_Loop_Impedance" in profil["karsilastirma"]["kirilgan_nesne_ezilme_yuzdesi"]
    assert profil["karsilastirma"]["kirilgan_nesne_ezilme_yuzdesi"]["Closed_Loop_Impedance"] == 0.4


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. TactileFeedbackGorsellestirici 6 panelli teşhis panosunu üretmelidir."""
    cikti = str(tmp_path / "test_tactile_feedback_paneli.png")
    profil = TactileFeedbackProfilleyici.basarim_profili_cikar()

    TactileFeedbackGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
