"""
PyTest Birim Testleri - Day 259: Robotik Başarım Paketi (Embodied AI Benchmarking Suite).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embodied_benchmark_motoru import (
    RoboticsMetricHarvester,
    FailureRootCauseAnalyzer,
    EmbodiedBenchmarkSuite,
)
from src.embodied_benchmark_profilleyici import EmbodiedBenchmarkProfilleyici
from src.gorsellestirici import EmbodiedBenchmarkGorsellestirici


def test_robotics_metric_harvester_gsr():
    """1. compute_gsr Grasp Success Rate oranını doğru hesaplamalıdır."""
    gsr = RoboticsMetricHarvester.compute_gsr(successes=95, total_trials=100)
    assert gsr == 0.95


def test_robotics_metric_harvester_path_efficiency():
    """2. compute_path_efficiency [0, 1] aralığında geodezik verimlilik dönmelidir."""
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([5.0, 0.0, 0.0])
    # Doğrusal rota (1.0 verimlilik)
    traj = np.linspace(start, goal, 10)
    eff = RoboticsMetricHarvester.compute_path_efficiency(traj, start, goal)
    assert round(eff, 2) == 1.0


def test_robotics_metric_harvester_curvature():
    """3. compute_curvature_smoothness yörünge ivme maliyetini hesaplamalıdır."""
    traj = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    smoothness = RoboticsMetricHarvester.compute_curvature_smoothness(traj)
    assert smoothness == 0.0  # Sabit hızda doğrusal yol sıfır ivme maliyetidir


def test_robotics_metric_harvester_collision_risk():
    """4. compute_collision_risk uzak engellerde sıfıra yakın risk skoru üretmelidir."""
    dists = np.array([2.0, 3.0, 4.0])
    hazard = RoboticsMetricHarvester.compute_collision_risk(dists, sigma=0.25)
    assert hazard < 0.01


def test_failure_root_cause_analyzer():
    """5. classify_failure telemetriden arıza kök nedenini teşhis etmelidir."""
    fail_col = FailureRootCauseAnalyzer.classify_failure(min_dist=0.01, is_singular=False, is_slip=False, is_timeout=False)
    assert fail_col == "DINAMIK_CARPISMA"

    fail_sing = FailureRootCauseAnalyzer.classify_failure(min_dist=0.5, is_singular=True, is_slip=False, is_timeout=False)
    assert fail_sing == "KINEMATIK_TEKILLIK"


def test_wilson_score_interval_bounds():
    """6. compute_wilson_score_interval geçerli [alt, üst] güven aralığı hesaplamalıdır."""
    lower, upper = EmbodiedBenchmarkSuite.compute_wilson_score_interval(successes=493, total=500)
    assert 0.0 <= lower <= upper <= 1.0
    assert lower > 0.95


def test_embodied_benchmark_profiler_output():
    """7. EmbodiedBenchmarkProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = EmbodiedBenchmarkProfilleyici.basarim_profili_cikar()
    assert "Calibrated_Embodied_AI" in profil["karsilastirma"]["global_gorev_basarisi_yuzde"]
    assert profil["karsilastirma"]["global_gorev_basarisi_yuzde"]["Calibrated_Embodied_AI"] == 98.6


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. EmbodiedBenchmarkGorsellestirici 6 panelli teşhis panosunu üretmelidir."""
    cikti = str(tmp_path / "test_embodied_benchmark_paneli.png")
    profil = EmbodiedBenchmarkProfilleyici.basarim_profili_cikar()

    EmbodiedBenchmarkGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
