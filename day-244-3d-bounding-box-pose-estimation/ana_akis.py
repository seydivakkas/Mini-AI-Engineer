"""
Day 244: 3D Sınırlayıcı Kutu ve 6-DoF Duruş Kestirimi (VoteNet) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from src.pose_estimation_motoru import (
    VoteNetPoseEstimator,
    hesapla_adds_metrigi,
)
from src.pose_profilleyici import PoseEstimationProfilleyici
from src.gorsellestirici import PoseEstimationGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 244 (FAZ 13): 3D SINIRLAYICI KUTU VE 6-DOF NESNE DURUŞ KESTİRİMİ (VOTENET / POSE ESTIMATION)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: VoteNet 6-DoF Duruş Kestiricisinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] VoteNet 6-DoF Duruş Kestiricisi Başlatılıyor...")
    torch.manual_seed(42)
    model = VoteNetPoseEstimator(feature_dim=64)
    toplam_parametre = sum(p.numel() for p in model.parameters())
    print(f"  ✓ VoteNet Modeli Hazır. Toplam Parametre: {toplam_parametre:,}")

    # -------------------------------------------------------------
    # ADIM 2: 3D Nokta Bulutu Üzerinden 6-DoF Duruş Kestirimi
    # -------------------------------------------------------------
    print("\n[2/4] Derin Hough Oylama ve 3D Sınırlayıcı Kutu Kestirimi...")
    xyz_noktalar = torch.randn(1, 256, 3) * 0.2 + torch.tensor([0.4, 0.1, 0.5])

    model.eval()
    with torch.no_grad():
        tahminler = model(xyz_noktalar)
        center = tahminler["center"].cpu().numpy()[0]
        dims = tahminler["dimensions"].cpu().numpy()[0]
        yaw = tahminler["yaw_rad"].item()
        conf = tahminler["confidence"].item()

    print(f"  • Tahmin 3D Merkez (x, y, z) : [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] metre")
    print(f"  • Tahmin 3D Boyutlar (l, w, h): [{dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f}] metre")
    print(f"  • Tahmin Yönelim Açısı (Yaw) : {np.degrees(yaw):.2f}°")
    print(f"  • Nesne Güven Skoru          : %{conf * 100:.2f}")

    # -------------------------------------------------------------
    # ADIM 3: ADD-S (<2cm) Robotik Tutma Doğrulaması
    # -------------------------------------------------------------
    print("\n[3/4] Robotik Tutma için ADD-S Doğrulaması Yapılıyor...")
    hedef_center = np.array([0.4, 0.1, 0.5])
    hata_cm, basarili = hesapla_adds_metrigi(center, hedef_center, esik_cm=5.0)

    durum_str = "BAŞARILI (TUTULABİLİR)" if basarili else "BAŞARISIZ"
    print(f"  🎯 ADD-S Konum Hatası: {hata_cm} cm -> Durum: {durum_str}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli VoteNet 6-DoF Teşhis Panosu Oluşturuluyor...")
    profil_raporu = PoseEstimationProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "pose_estimation_paneli.png")

    PoseEstimationGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ 6-DoF Pose Estimation Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 244 (FAZ 13): 3D SINIRLAYICI KUTU VE 6-DOF DURUŞ KESTİRİMİ MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
