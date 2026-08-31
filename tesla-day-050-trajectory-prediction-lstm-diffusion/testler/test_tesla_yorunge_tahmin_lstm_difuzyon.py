"""
Tesla Yörünge Tahmini Birim Testleri (PyTest)
=============================================
Bu test paketi; Çoklu modalite yörünge üretimini, 50 adımlık zaman ufkunu
ve TTC çarpışma süresi analizini test eder.

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

from src.tesla_yorunge_tahmin_lstm_difuzyon import TeslaTrajectoryPredictor


def test_yorunge_boyutlari_ve_zaman_ufku():
    """50 zaman adımlı ve (50, 2) koordinat matrisli 3 yörüngenin üretildiği test edilir."""
    pred = TeslaTrajectoryPredictor(horizon_steps=50, dt_s=0.1)
    res = pred.predict_multi_modal_trajectories()

    trajs = res["trajectories"]
    assert "LANE_KEEP" in trajs
    assert "LANE_CHANGE_LEFT" in trajs
    assert "HARD_BRAKE" in trajs

    assert trajs["LANE_KEEP"].shape == (50, 2)
    assert trajs["LANE_CHANGE_LEFT"].shape == (50, 2)
    assert trajs["HARD_BRAKE"].shape == (50, 2)


def test_olasilik_dagilimi_tutarliligi():
    """Mod olasılıklarının toplamının 1.0 olduğu test edilir."""
    pred = TeslaTrajectoryPredictor()
    res = pred.predict_multi_modal_trajectories()

    probs = res["probabilities"]
    assert np.isclose(np.sum(probs), 1.0)
    assert np.all(probs >= 0.0)


def test_ttc_carpislama_suresi():
    """20 metre önde ve 5 m/s yaklaşma hızında TTC'nin ~4.0 saniye çıktığı test edilir."""
    pred = TeslaTrajectoryPredictor()
    res = pred.predict_multi_modal_trajectories(current_pos=np.array([0.0, 20.0]), current_vel=np.array([0.0, 15.0]))

    ttc = res["ttc_seconds"]
    assert 3.5 < ttc < 4.5
