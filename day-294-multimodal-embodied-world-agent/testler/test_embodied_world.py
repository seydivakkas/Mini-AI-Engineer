"""
PyTest Birim Testleri - Day 294 (FAZ 15): Çok Modlu Bedenlenmiş Dünya Ajanı (Spatial VLM).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embodied_world_motoru import (
    Spatial3DObject,
    MultimodalEmbodiedAgent,
    TrajectoryPlanner,
)
from src.embodied_world_profilleyici import EmbodiedWorldProfilleyici
from src.gorsellestirici import EmbodiedWorldGorsellestirici


def test_spatial_3d_object_initialization():
    """1. Mekansal 3D nesne konum ve affordance vektörleriyle başlatılmalıdır."""
    obj = Spatial3DObject("Test Obje", (0.1, 0.2, 0.3), (0.05, 0.05, 0.1), (0.1, 0.2, 0.35))
    assert obj.name == "Test Obje"
    assert obj.position.shape == (3,)
    assert obj.affordance_point.shape == (3,)


def test_multimodal_embodied_agent_grounding():
    """2. Bedenlenmiş ajan doğal dil komutundan hedef nesneyi doğru seçmelidir."""
    agent = MultimodalEmbodiedAgent()
    o1 = Spatial3DObject("Engel", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    o2 = Spatial3DObject("Tıbbi Numune Şişesi", (0.5, 0.5, 0.5), (0.1, 0.1, 0.1), (0.5, 0.5, 0.6))
    target = agent.parse_instruction_and_ground("Numune şişesini al", [o1, o2])
    assert target.name == "Tıbbi Numune Şişesi"


def test_trajectory_planner_waypoints_shape():
    """3. Yörünge planlayıcı belirtilen sayıda (N, 3) koordinat noktası üretmelidir."""
    start = np.array([0.0, 0.0, 0.0])
    target = np.array([1.0, 1.0, 1.0])
    waypoints = TrajectoryPlanner.plan_trajectory(start, target, num_waypoints=12)
    assert waypoints.shape == (12, 3)


def test_trajectory_planner_start_and_end():
    """4. Yörüngenin ilk noktası başlangıç, son noktası hedef koordinat olmalıdır."""
    start = np.array([0.1, 0.2, 0.3])
    target = np.array([0.8, 0.7, 0.9])
    waypoints = TrajectoryPlanner.plan_trajectory(start, target, num_waypoints=10)
    assert np.allclose(waypoints[0], start, atol=1e-3)
    assert np.allclose(waypoints[-1], target, atol=1e-3)


def test_profiler_grasping_superiority():
    """5. 3D Mekansal ajan kavrama başarı oranı %95'in üzerinde olmalıdır."""
    profil = EmbodiedWorldProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["tutma_basarisi_yuzde"]["3. Spatial World Agent"] > 95.0


def test_profiler_spatial_precision_speedup():
    """6. Mekansal konum hassasiyet artışı en az 10 kat olmalıdır."""
    profil = EmbodiedWorldProfilleyici.basarim_profili_cikar()
    assert profil["hassasiyet_artisi"] >= 10.0


def test_profiler_collision_free_safety():
    """7. Çarpışmasız hareket oranı %98'in üzerinde olmalıdır."""
    profil = EmbodiedWorldProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["carpismazlik_orani_yuzde"]["3. Spatial World Agent"] > 98.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. EmbodiedWorldGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_embodied_paneli.png")
    profil = EmbodiedWorldProfilleyici.basarim_profili_cikar()

    EmbodiedWorldGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
