"""
Day 253: RGB-D Derinlik Füzyonu ve 3D Doluluk Izgarası (Occupancy Grid) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.occupancy_grid_motoru import (
    RGBDProjector,
    VoxelOccupancyGrid,
    DynamicObstacleAvoidance,
)
from src.occupancy_grid_profilleyici import OccupancyGridProfilleyici
from src.gorsellestirici import OccupancyGridGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 253 (FAZ 13): RGB-D DERİNLİK FÜZYONU VE 3D DOLULUK IZGARASI (OCCUPANCY GRID & ENGEL KAÇINMA)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: RGB-D Projektör ve 3D Voxel Izgarasının Kurulması
    # -------------------------------------------------------------
    print("\n[1/4] RGB-D Kamera Projektörü ve 3D Voxel Izgarası Başlatılıyor...")
    projector = RGBDProjector(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
    grid = VoxelOccupancyGrid(min_bound=(-2.0, -2.0, 0.0), max_bound=(2.0, 2.0, 1.5), resolution_m=0.05)

    print(f"  • Voxel Izgara Boyutları : {grid.grid_dims.tolist()} voksel")
    print(f"  • Voxel Çözünürlüğü      : {grid.res * 100:.1f} cm")
    print(f"  • Log-Odds Parametreleri : l_occ=+{grid.l_occ}, l_free={grid.l_free}")

    # -------------------------------------------------------------
    # ADIM 2: Derinlik Haritasından 3D Nokta Bulutu ve Bayesyen Füzyon
    # -------------------------------------------------------------
    print("\n[2/4] Derinlik Haritasından 3D Nokta Bulutu İzdüşümü ve Log-Odds Füzyonu...")
    depth_img = np.full((120, 160), 3.0, dtype=np.float32)
    depth_img[40:80, 60:100] = 1.5  # 1.5m uzaklıkta engel

    pcd = projector.depth_to_point_cloud(depth_img)
    grid.update_with_points(pcd)
    dolu_sayisi = grid.get_occupied_voxel_count(threshold_prob=0.70)

    print(f"  • Üretilen 3D Nokta Sayısı : {len(pcd)} adet")
    print(f"  • Dolu Voxel Sayısı (P>=0.7): {dolu_sayisi} adet")

    # -------------------------------------------------------------
    # ADIM 3: Dinamik Engel Kaçınma ve Güvenli Rota Planlama
    # -------------------------------------------------------------
    print("\n[3/4] 3D Enflasyonlu Güvenli Kaçış Rotası Hesaplanıyor...")
    start = np.array([0.0, 0.0, 0.5])
    goal = np.array([0.0, 3.0, 0.5])
    obs_center = [np.array([0.0, 1.5, 0.5])]
    rota_res = DynamicObstacleAvoidance.plan_avoidance_path(start, goal, obs_center)

    print(f"  • Rota Adım Sayısı         : {len(rota_res['path'])}")
    print(f"  • En Yakın Engel Mesafesi  : {rota_res['min_clearance_m']} m (Güvenli: {not rota_res['carpisma_var_mi']})")
    print(f"  • Başlangıç / Bitiş        : {start.tolist()} -> {goal.tolist()}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli 3D Occupancy Grid Teşhis Panosu Oluşturuluyor...")
    profil_raporu = OccupancyGridProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "occupancy_grid_paneli.png")

    OccupancyGridGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Occupancy Grid Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 253 (FAZ 13): RGB-D DERİNLİK FÜZYONU VE 3D DOLULUK IZGARASI MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
