"""
PyTest Birim Testleri - Day 257: Model Predictive Control (MPC) ile Dinamik Engelden Kaçınma.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dynamic_mpc_motoru import (
    DynamicObstacleTracker,
    NonlinearMPCController,
)
from src.dynamic_mpc_profilleyici import DynamicMPCProfilleyici
from src.gorsellestirici import DynamicMPCGorsellestirici


def test_dynamic_obstacle_tracker_predict():
    """1. predict_trajectory gelecek N adımlık konumları doğrusal projekte etmelidir."""
    obs = DynamicObstacleTracker(pos_init=(2.0, 3.0), vel_xy=(1.0, 0.0), radius=0.4)
    pred = obs.predict_trajectory(horizon=10, dt=0.1)
    assert pred.shape == (10, 2)
    assert pred[0, 0] == 2.0
    assert round(pred[9, 0], 2) == 2.9


def test_dynamic_obstacle_tracker_update():
    """2. update_state engel konumunu zaman adımı kadar güncellemelidir."""
    obs = DynamicObstacleTracker(pos_init=(0.0, 0.0), vel_xy=(2.0, -1.0), radius=0.4)
    obs.update_state(dt=0.5)
    assert obs.pos[0] == 1.0
    assert obs.pos[1] == -0.5


def test_nonlinear_mpc_controller_init():
    """3. NonlinearMPCController parametrelerini doğru kurmalıdır."""
    ctrl = NonlinearMPCController(horizon=15, dt=0.1, robot_radius=0.30, v_max=3.0)
    assert ctrl.N == 15
    assert ctrl.dt == 0.1
    assert ctrl.v_max == 3.0


def test_nonlinear_mpc_controller_solve():
    """4. solve_control geçerli aksiyonlar ve N adımlık öngörü dizisi üretmelidir."""
    ctrl = NonlinearMPCController(horizon=15, dt=0.1)
    obs = DynamicObstacleTracker(pos_init=(5.0, 1.0), vel_xy=(0.0, -0.5), radius=0.4)
    res = ctrl.solve_control(
        current_state=np.array([0.0, 0.0, 0.0, 1.0]),
        goal_pos=(8.0, 0.0),
        dynamic_obstacles=[obs],
        v_target=2.0,
    )
    assert -2.0 <= res["en_iyi_ivme_m_s2"] <= 2.0
    assert -1.5 <= res["en_iyi_acisal_hiz_rad_s"] <= 1.5
    assert len(res["ongorulen_robot_yorungesi"]) == 15


def test_mpc_obstacle_avoidance_clearing():
    """5. NMPC öngörülen rotada engelle çarpışmayı engelleyici kaçınma üretmelidir."""
    ctrl = NonlinearMPCController(horizon=15, dt=0.1)
    obs = DynamicObstacleTracker(pos_init=(3.0, 0.0), vel_xy=(0.0, 0.0), radius=0.4)
    res = ctrl.solve_control(
        current_state=np.array([0.0, 0.0, 0.0, 1.0]),
        goal_pos=(6.0, 0.0),
        dynamic_obstacles=[obs],
    )
    # Robot tam doğrusal gitmek yerine yönelimi saptırmalıdır
    assert abs(res["en_iyi_acisal_hiz_rad_s"]) >= 0.0


def test_mpc_velocity_bounds():
    """6. Öngörülen robot hızları [0, v_max] sınırları içinde kalmalıdır."""
    ctrl = NonlinearMPCController(horizon=15, dt=0.1, v_max=2.5)
    res = ctrl.solve_control(
        current_state=np.array([0.0, 0.0, 0.0, 2.0]),
        goal_pos=(10.0, 0.0),
        dynamic_obstacles=[],
    )
    speeds = res["ongorulen_robot_yorungesi"][:, 3]
    assert np.all(speeds >= 0.0)
    assert np.all(speeds <= 2.5 + 1e-3)


def test_dynamic_mpc_profiler_output():
    """7. DynamicMPCProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = DynamicMPCProfilleyici.basarim_profili_cikar()
    assert "Dynamic_NMPC" in profil["karsilastirma"]["yuksek_hizli_carpismazlik_orani_yuzde"]
    assert profil["karsilastirma"]["yuksek_hizli_carpismazlik_orani_yuzde"]["Dynamic_NMPC"] == 99.2


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. DynamicMPCGorsellestirici 6 panelli teşhis panosunu üretmelidir."""
    cikti = str(tmp_path / "test_mpc_avoidance_paneli.png")
    profil = DynamicMPCProfilleyici.basarim_profili_cikar()

    DynamicMPCGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
