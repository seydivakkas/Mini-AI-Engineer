"""
Day 258: Sıfır Örnekli (Zero-Shot) Görülmemiş Nesneleri Kavrama ve Ayırma Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.zero_shot_grasping_motoru import (
    PointCloudPreprocessor,
    AntipodalGraspGenerator,
    ZeroShotBinSortingPipeline,
)
from src.zero_shot_grasping_profilleyici import ZeroShotGraspingProfilleyici
from src.gorsellestirici import ZeroShotGraspingGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 258 (FAZ 13): SIFIR ÖRNEKLİ (ZERO-SHOT) GÖRÜLMEMİŞ NESNELERİ KAVRAMA VE AYIRMA (ANYGRASP)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: 3D Nokta Bulutu ve Yüzey Normalleri
    # -------------------------------------------------------------
    print("\n[1/4] Görülmemiş Nesne 3D Nokta Bulutu Yükleniyor ve Normaller Hesaplanıyor...")
    np.random.seed(42)
    n_pts = 50
    theta = np.linspace(0, 2 * np.pi, n_pts)
    z = np.linspace(0.03, 0.10, n_pts)
    r = 0.03
    object_pts = np.vstack([r * np.cos(theta) + 0.50, r * np.sin(theta) + 0.10, z]).T

    normals = PointCloudPreprocessor.estimate_normals(object_pts)
    print(f"  • Yüklenen Nokta Sayısı       : {len(object_pts)} Nokta")
    print(f"  • Hesaplanan Normal Boyutu    : {normals.shape}")

    # -------------------------------------------------------------
    # ADIM 2: 6-DoF Antipodal Kavrama Pozlarının Üretilmesi
    # -------------------------------------------------------------
    print("\n[2/4] 6-DoF Antipodal Kavrama Adayları ve Kalite Puanlaması Yapılıyor...")
    grasps = AntipodalGraspGenerator.generate_grasps(object_pts, normals, max_grasps=5)
    print(f"  • Bulunan Uygun Grasp Sayısı  : {len(grasps)} Adet")
    if grasps:
        en_iyi = grasps[0]
        print(f"  • En İyi 6-DoF Grasp Merkezi  : {en_iyi['merkez_3d']} m")
        print(f"  • Kavrama Genişliği           : {en_iyi['kavrama_genisligi_m']} m")
        print(f"  • Antipodal Kalite Skoru      : {en_iyi['kalite_skoru']}")

    # -------------------------------------------------------------
    # ADIM 3: Sıfır Örnekli Kutuya Ayırma Hattı
    # -------------------------------------------------------------
    print("\n[3/4] Görülmemiş Nesne Semantik Kategoriye Göre Hedef Kutuya Ayrıştırılıyor...")
    kategoriler = ["PLASTİK", "ORGANİK", "METAL"]
    for kat in kategoriler:
        res = ZeroShotBinSortingPipeline.sort_unseen_object(object_pts, semantic_category=kat)
        print(f"  • Kategori: {kat.ljust(8)} -> Hedef Kutu: {res['hedef_kutu_koordinati']} | Durum: {res['durum']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Sıfır Örnekli Kavrama Teşhis Panosu Oluşturuluyor...")
    profil_raporu = ZeroShotGraspingProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "zero_shot_grasping_paneli.png")

    ZeroShotGraspingGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Sıfır Örnekli Kavrama Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 258 (FAZ 13): SIFIR ÖRNEKLİ KAVRAMA VE AYIRMA MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
