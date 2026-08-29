"""
Day 243: 3D Nokta Bulutu ve Mekansal Akıl Yürütme (PointNet++) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from src.point_cloud_motoru import (
    PointNetPlusPlusModel,
    farthest_point_sampling,
    ball_query,
    ornek_3d_fincan_bulutu_olustur,
)
from src.point_cloud_profilleyici import PointCloudProfilleyici
from src.gorsellestirici import PointCloudGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 243 (FAZ 13): 3D NOKTA BULUTU VE MEKANSAL AKIL YÜRÜTME (SPATIAL AI - POINTNET++)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Sentetik 3D Fincan Nokta Bulutunun Oluşturulması
    # -------------------------------------------------------------
    print("\n[1/4] 3D Fincan Nokta Bulutu Üretiliyor (N=512 Nokta)...")
    bulut = ornek_3d_fincan_bulutu_olustur(nokta_sayisi=512)
    print(f"  ✓ Nokta Bulutu Boyutu: {bulut.shape} [X, Y, Z Koordinatları]")
    print(f"  • Min Sınırlar: {bulut.min(axis=0).round(3).tolist()}")
    print(f"  • Max Sınırlar: {bulut.max(axis=0).round(3).tolist()}")

    # -------------------------------------------------------------
    # ADIM 2: En Uzak Nokta Örneklemesi (FPS) ve Ball Query
    # -------------------------------------------------------------
    print("\n[2/4] Farthest Point Sampling (FPS) ve Ball Query İşletiliyor...")
    xyz_tensor = torch.from_numpy(bulut).unsqueeze(0)  # [1, 512, 3]
    fps_indisler = farthest_point_sampling(xyz_tensor, npoint=128)
    merkez_noktalar = torch.gather(xyz_tensor, 1, fps_indisler.unsqueeze(-1).repeat(1, 1, 3))
    komsuluklar = ball_query(radius=0.2, nsample=16, xyz=xyz_tensor, new_xyz=merkez_noktalar)

    print(f"  ✓ FPS Örneklemesi: 512 noktadan {fps_indisler.shape[1]} homojen merkez seçildi.")
    print(f"  ✓ Ball Query: Her merkez için r=0.2 yarıçapında {komsuluklar.shape[2]} yerel komşu toplandı.")

    # -------------------------------------------------------------
    # ADIM 3: PointNet++ Hiyerarşik Çıkarım ve Tutma Afordansı
    # -------------------------------------------------------------
    print("\n[3/4] PointNet++ Hiyerarşik Modeli ve Tutma Afordansı Çıkarımı...")
    model = PointNetPlusPlusModel(num_classes=1)
    toplam_parametre = sum(p.numel() for p in model.parameters())
    print(f"  ✓ PointNet++ Modeli Hazır. Toplam Parametre: {toplam_parametre:,}")

    model.eval()
    with torch.no_grad():
        skor = model(xyz_tensor).item()

    print(f"  🎯 3D Geometrik Tutma (Grasp) Afordans Skoru: %{skor * 100:.2f}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli PointNet++ Teşhis Panosu Oluşturuluyor...")
    profil_raporu = PointCloudProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "point_cloud_paneli.png")

    PointCloudGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ PointNet++ Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 243 (FAZ 13): 3D NOKTA BULUTU VE MEKANSAL AKIL YÜRÜTME MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
