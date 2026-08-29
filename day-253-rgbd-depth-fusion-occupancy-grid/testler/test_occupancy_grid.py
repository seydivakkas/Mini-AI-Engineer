"""
PyTest Birim Testleri - Day 253: RGB-D Derinlik Füzyonu ve 3D Doluluk Izgarası.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.occupancy_grid_motoru import (
    RGBDProjector,
    VoxelOccupancyGrid,
    DynamicObstacleAvoidance,
)
from src.occupancy_grid_profilleyici import OccupancyGridProfilleyici
from src.gorsellestirici import OccupancyGridGorsellestirici


def test_rgbd_projector_point_cloud_generation():
    """1. RGBDProjector derinlik haritasını (N, 3) 3D nokta bulutuna çevirmelidir."""
    projector = RGBDProjector()
    depth_img = np.ones((50, 50), dtype=np.float32) * 2.0
    pcd = projector.depth_to_point_cloud(depth_img)
    assert pcd.ndim == 2
    assert pcd.shape[1] == 3
    assert len(pcd) == 2500


def test_voxel_occupancy_grid_init():
    """2. VoxelOccupancyGrid boyutları doğru çözünürlükle oluşturmalıdır."""
    grid = VoxelOccupancyGrid(min_bound=(-1.0, -1.0, 0.0), max_bound=(1.0, 1.0, 1.0), resolution_m=0.1)
    assert grid.grid_dims[0] == 20
    assert grid.grid_dims[1] == 20
    assert grid.grid_dims[2] == 10


def test_voxel_occupancy_grid_coord_to_index():
    """3. coord_to_index geçerli 3D ızgara indeksleri üretmelidir."""
    grid = VoxelOccupancyGrid()
    pts = np.array([[0.0, 0.0, 0.5], [1.0, 1.0, 1.0]])
    idx = grid.coord_to_index(pts)
    assert len(idx) == 2
    assert idx.shape[1] == 3


def test_voxel_occupancy_grid_update():
    """4. update_with_points hit voksellerin log-odds değerini artırmalıdır."""
    grid = VoxelOccupancyGrid()
    pts = np.array([[0.0, 0.0, 0.5]])
    grid.update_with_points(pts)
    assert np.max(grid.log_odds) > 0.0


def test_voxel_occupancy_grid_count():
    """5. get_occupied_voxel_count pozitif tamsayı dönmelidir."""
    grid = VoxelOccupancyGrid()
    pts = np.array([[0.0, 0.0, 0.5]] * 5)
    grid.update_with_points(pts)
    count = grid.get_occupied_voxel_count(threshold_prob=0.50)
    assert count >= 1
    assert isinstance(count, int)


def test_dynamic_obstacle_avoidance_path():
    """6. plan_avoidance_path engele çarpmayan güvenli rota üretmelidir."""
    start = np.array([0.0, 0.0, 0.5])
    goal = np.array([0.0, 2.0, 0.5])
    obs = [np.array([0.0, 1.0, 0.5])]
    res = DynamicObstacleAvoidance.plan_avoidance_path(start, goal, obs)
    assert res["carpisma_var_mi"] is False
    assert res["min_clearance_m"] >= 0.25


def test_occupancy_grid_profiler_output():
    """7. OccupancyGridProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = OccupancyGridProfilleyici.basarim_profili_cikar()
    assert "3D_Voxel_LogOdds_Fusion" in profil["karsilastirma"]["dinamik_engel_kacinma_yuzde"]
    assert profil["karsilastirma"]["dinamik_engel_kacinma_yuzde"]["3D_Voxel_LogOdds_Fusion"] == 99.4


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. OccupancyGridGorsellestirici 6 panelli teşhis panosunu üretmelidir."""
    cikti = str(tmp_path / "test_occupancy_grid_paneli.png")
    profil = OccupancyGridProfilleyici.basarim_profili_cikar()

    OccupancyGridGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
