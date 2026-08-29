"""
PyTest Birim Testleri - Day 248: VLM Destekli Semantik SLAM Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.semantic_slam_motoru import (
    OccupancyGridMap,
    VLMSemanticAnchor,
    AStarPathPlanner,
    SemanticSLAMSystem,
)
from src.slam_profilleyici import SemanticSLAMProfilleyici
from src.gorsellestirici import SemanticSLAMGorsellestirici


def test_occupancy_grid_init():
    """1. OccupancyGridMap doğru boyutlarda sıfır matrisi başlatmalıdır."""
    grid = OccupancyGridMap(genislik=30, yukseklik=40, cozunurluk_m=0.05)
    assert grid.W == 30
    assert grid.H == 40
    assert grid.izgara.shape == (40, 30)
    assert np.all(grid.izgara == 0)


def test_occupancy_grid_add_obstacle():
    """2. add_obstacle belirtilen koordinatı 1.0 olarak işaretlemelidir."""
    grid = OccupancyGridMap(genislik=20, yukseklik=20)
    grid.add_obstacle(10, 10, yaricap=0)
    assert grid.izgara[10, 10] == 1.0
    assert grid.izgara[0, 0] == 0.0


def test_occupancy_grid_inflation():
    """3. compute_inflation_costmap engelin etrafında pozitif maliyet üretmelidir."""
    grid = OccupancyGridMap(genislik=20, yukseklik=20)
    grid.add_obstacle(10, 10, yaricap=0)
    costmap = grid.compute_inflation_costmap(guvenlik_yaricapi=2)
    assert costmap[10, 11] > 0.0
    assert costmap[10, 10] == 1.0


def test_vlm_semantic_anchor_grounding():
    """4. ground_language_query kahve kupası sorgusunu doğru nesneye eşlemelidir."""
    vlm = VLMSemanticAnchor()
    res = vlm.ground_language_query("kırmızı kahve kupası")
    assert res["eslesen_nesne"]["id"] == "cup_red"
    assert res["hedef_koordinat"] == (12, 38)
    assert res["guven_skoru"] >= 0.8


def test_vlm_semantic_anchor_custom_query():
    """5. ground_language_query şarj istasyonu sorgusunu charging_dock'a eşlemelidir."""
    vlm = VLMSemanticAnchor()
    res = vlm.ground_language_query("şarj istasyonuna git")
    assert res["eslesen_nesne"]["id"] == "charging_dock"
    assert res["hedef_koordinat"] == (42, 42)


def test_astar_path_planner_straight():
    """6. AStarPathPlanner engelsiz boş haritada başlangıçtan hedefe yol bulmalıdır."""
    costmap = np.zeros((20, 20), dtype=np.float32)
    yol = AStarPathPlanner.plan_path(costmap, (2, 2), (8, 8))
    assert len(yol) > 1
    assert yol[0] == (2, 2)
    assert yol[-1] == (8, 8)


def test_semantic_slam_system_navigate():
    """7. SemanticSLAMSystem doğal dil komutuyla tam yörünge üretmelidir."""
    slam = SemanticSLAMSystem(W=50, H=50)
    res = slam.navigate_with_language("mavi su şişesine git")
    assert res["basarili"] is True
    assert res["hedef_nesne"] == "mavi su şişesi"
    assert res["yol_nokta_sayisi"] > 5


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. SemanticSLAMGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_semantic_slam_paneli.png")
    profil = SemanticSLAMProfilleyici.basarim_profili_cikar()

    SemanticSLAMGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
