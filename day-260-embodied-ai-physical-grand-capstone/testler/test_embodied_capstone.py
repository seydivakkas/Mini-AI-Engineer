"""
PyTest Birim Testleri - Day 260 (FAZ 13 BÜYÜK FİNALİ): Embodied AI Fiziksel Robotik Süiti.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embodied_capstone_motoru import (
    OpenVLAEmbeddingGenerator,
    DiffusionPolicyActionGenerator,
    ROS2MiddlewareBridge,
    UnifiedEmbodiedAIEngine,
)
from src.embodied_capstone_profilleyici import EmbodiedCapstoneProfilleyici
from src.gorsellestirici import EmbodiedCapstoneGorsellestirici


def test_openvla_embedding_generator_shape():
    """1. OpenVLAEmbeddingGenerator istenen boyutta şartlandırma vektörü üretmelidir."""
    encoder = OpenVLAEmbeddingGenerator(embed_dim=64)
    emb = encoder.encode("Test prompt", image_features=np.array([1.0, 2.0]))
    assert emb.shape == (64,)


def test_openvla_embedding_generator_normalization():
    """2. OpenVLAEmbeddingGenerator birim normlu vektör üretmelidir."""
    encoder = OpenVLAEmbeddingGenerator(embed_dim=32)
    emb = encoder.encode("Birim norm testi")
    assert round(float(np.linalg.norm(emb)), 2) == 1.0


def test_diffusion_policy_action_generator_shape():
    """3. DiffusionPolicyActionGenerator (16, 7) eylem yığını üretmelidir."""
    diff = DiffusionPolicyActionGenerator(chunk_size=16, action_dim=7)
    cond = np.random.randn(64)
    actions = diff.generate_action_chunk(cond)
    assert actions.shape == (16, 7)


def test_diffusion_policy_action_generator_smoothness():
    """4. Diffusion Policy yörüngesi başlangıçtan hedefe pürüzsüz ilerlemelidir."""
    diff = DiffusionPolicyActionGenerator(chunk_size=16, action_dim=7)
    cond = np.zeros(64)
    actions = diff.generate_action_chunk(cond)
    # X konumu 0.2'den 0.7'ye artmalı
    assert actions[0, 0] < actions[-1, 0]


def test_ros2_middleware_bridge_normal_command():
    """5. ROS2MiddlewareBridge normal şartlarda komutu başarıyla yayınlamalıdır."""
    bridge = ROS2MiddlewareBridge()
    res = bridge.publish_command(np.zeros(7), min_obstacle_dist=0.5, contact_force=10.0)
    assert res["durum"] == "KOMUT_YAYINLANDI"
    assert res["guvenlik_ihlal"] is False


def test_ros2_middleware_bridge_estop_trigger():
    """6. ROS2MiddlewareBridge aşırı yakınlıkta veya kuvvette E-Stop tetiklemelidir."""
    bridge = ROS2MiddlewareBridge()
    res_dist = bridge.publish_command(np.zeros(7), min_obstacle_dist=0.02, contact_force=10.0)
    assert res_dist["durum"] == "ESTOP_AKTIF"
    assert res_dist["guvenlik_ihlal"] is True

    res_force = bridge.publish_command(np.zeros(7), min_obstacle_dist=0.5, contact_force=40.0)
    assert res_force["durum"] == "ESTOP_AKTIF"
    assert res_force["guvenlik_ihlal"] is True


def test_unified_embodied_ai_engine_mission():
    """7. UnifiedEmbodiedAIEngine görevi uçtan uca hatasız icra etmelidir."""
    engine = UnifiedEmbodiedAIEngine()
    res = engine.execute_mission("Kırılgan bardağı taşı")
    assert res["gorev_durumu"] == "BASARIYLA_ICRA_EDILDI"
    assert res["condition_vector_dim"] == 64


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. EmbodiedCapstoneGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_embodied_capstone_paneli.png")
    profil = EmbodiedCapstoneProfilleyici.basarim_profili_cikar()

    EmbodiedCapstoneGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
