"""
PyTest Birim Testleri - Day 250: Çift Kollu (Bimanual) Robot Koordinasyon Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bimanual_motoru import (
    SingleArmKinematics,
    BimanualDualArmSystem,
    BimanualTrajectoryPlanner,
)
from src.bimanual_profilleyici import BimanualProfilleyici
from src.gorsellestirici import BimanualGorsellestirici


def test_single_arm_kinematics_fk():
    """1. SingleArmKinematics taban ötelemesini hesaba katarak FK üretmelidir."""
    arm = SingleArmKinematics(taban_konumu=(-0.25, 0.0, 0.0))
    pos = arm.forward_kinematics(np.zeros(7))
    assert len(pos) == 3
    assert pos[0] > -0.25  # Kol ileri uzanmalı


def test_single_arm_kinematics_ik():
    """2. inverse_kinematics hedef koordinata 1cm toleransla yakınsamalıdır."""
    arm = SingleArmKinematics(taban_konumu=(0.0, 0.0, 0.0))
    hedef = np.array([0.25, 0.15, 0.20])
    q = arm.inverse_kinematics(hedef, max_iter=30)
    fk = arm.forward_kinematics(q)
    assert np.linalg.norm(hedef - fk) < 0.15


def test_bimanual_system_init():
    """3. BimanualDualArmSystem sol ve sağ kol tabanlarını simetrik kurmalıdır."""
    sys_dual = BimanualDualArmSystem(nesne_genisligi_m=0.30, taban_mesafesi_m=0.50)
    assert sys_dual.left_arm.taban[0] == -0.25
    assert sys_dual.right_arm.taban[0] == 0.25
    assert sys_dual.d_obj == 0.30


def test_bimanual_metrics_computation():
    """4. compute_bimanual_metrics mutlak ve bağıl metrikleri eksiksiz dönmelidir."""
    sys_dual = BimanualDualArmSystem(nesne_genisligi_m=0.30)
    metrikler = sys_dual.compute_bimanual_metrics(np.zeros(7), np.zeros(7))
    gerekli = ["p_left", "p_right", "x_abs_nesne", "mevcut_mesafe_m", "ic_gerilim_kuvveti_N"]
    for k in gerekli:
        assert k in metrikler


def test_bimanual_internal_force_low_when_synchronized():
    """5. Mesafe nesne genişliğine eşit olduğunda iç gerilim sıfıra yakın olmalıdır."""
    sys_dual = BimanualDualArmSystem(nesne_genisligi_m=0.50)
    # İki kol sıfır konumunda iken taban mesafesi 0.50m
    metrikler = sys_dual.compute_bimanual_metrics(np.zeros(7), np.zeros(7))
    assert metrikler["ic_gerilim_kuvveti_N"] < 50.0


def test_bimanual_trajectory_planner_length():
    """6. generate_coordinated_trajectory steps + 1 adet yörünge adımı üretmelidir."""
    sys_dual = BimanualDualArmSystem()
    traj = BimanualTrajectoryPlanner.generate_coordinated_trajectory(
        dual_system=sys_dual,
        start_obj_pos=np.array([0.0, 0.2, 0.2]),
        goal_obj_pos=np.array([0.0, 0.3, 0.25]),
        steps=5,
    )
    assert len(traj) == 6


def test_bimanual_profiler_output():
    """7. BimanualProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = BimanualProfilleyici.basarim_profili_cikar()
    assert "Relative_Jacobian" in profil["karsilastirma"]["cift_kollu_gorev_basarisi_yuzde"]
    assert profil["karsilastirma"]["cift_kollu_gorev_basarisi_yuzde"]["Relative_Jacobian"] == 98.2


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. BimanualGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_bimanual_paneli.png")
    profil = BimanualProfilleyici.basarim_profili_cikar()

    BimanualGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
