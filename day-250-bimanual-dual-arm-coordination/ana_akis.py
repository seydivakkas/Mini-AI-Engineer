"""
Day 250: Çift Kollu (Bimanual) Robot Koordinasyonu Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.bimanual_motoru import (
    SingleArmKinematics,
    BimanualDualArmSystem,
    BimanualTrajectoryPlanner,
)
from src.bimanual_profilleyici import BimanualProfilleyici
from src.gorsellestirici import BimanualGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 250 (FAZ 13): ÇİFT KOLLU (BIMANUAL) ROBOT KOORDİNASYONU VE SENKRONİZE GÖREV PAYLAŞIMI")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Bimanual Çift Kol Sisteminin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Bimanual Çift Kol Sistemi ve Taban Konfigürasyonu Kuruluyor...")
    dual_sys = BimanualDualArmSystem(nesne_genisligi_m=0.30, taban_mesafesi_m=0.50)
    print(f"  • Sol Kol Tabanı (Arm_L) : {dual_sys.left_arm.taban.tolist()} m")
    print(f"  • Sağ Kol Tabanı (Arm_R) : {dual_sys.right_arm.taban.tolist()} m")
    print(f"  • Ortak Nesne Genişliği  : {dual_sys.d_obj} m (Sabit Kapalı Zincir Kısıtı)")

    # -------------------------------------------------------------
    # ADIM 2: Mutlak ve Bağıl Kinematik Ölçümü
    # -------------------------------------------------------------
    print("\n[2/4] Mutlak (x_abs) ve Bağıl (x_rel) Kinematik Ayrışımı Hesaplanıyor...")
    q_sol = np.array([0.4, -0.3, 0.2, 0.5, 0.0, 0.1, 0.0])
    q_sag = np.array([-0.4, -0.3, -0.2, 0.5, 0.0, 0.1, 0.0])
    metrikler = dual_sys.compute_bimanual_metrics(q_sol, q_sag)

    print(f"  • Sol Uç Nokta (p_L)      : {metrikler['p_left']} m")
    print(f"  • Sağ Uç Nokta (p_R)      : {metrikler['p_right']} m")
    print(f"  • Mutlak Nesne Konumu     : {metrikler['x_abs_nesne']} m")
    print(f"  • Kollar Arası Mesafe     : {metrikler['mevcut_mesafe_m']} m (Sapma: {metrikler['mesafe_sapmasi_mm']} mm)")
    print(f"  • İç Gerilim / Ezme Yükü  : {metrikler['ic_gerilim_kuvveti_N']} Newton")

    # -------------------------------------------------------------
    # ADIM 3: Eşzamanlı Senkronize Yörünge İcrası
    # -------------------------------------------------------------
    print("\n[3/4] Senkronize Bimanual Nesne Taşıma Yörüngesi Çalıştırılıyor...")
    baslangic = np.array([0.0, 0.25, 0.20])
    hedef = np.array([0.05, 0.35, 0.30])
    traj = BimanualTrajectoryPlanner.generate_coordinated_trajectory(
        dual_system=dual_sys,
        start_obj_pos=baslangic,
        goal_obj_pos=hedef,
        steps=6,
    )

    for step in traj:
        adim_no = step["adim"]
        pos = step["hedef_obj_pos"]
        f_int = step["metrikler"]["ic_gerilim_kuvveti_N"]
        sapma = step["metrikler"]["mesafe_sapmasi_mm"]
        print(f"  [Adım {adim_no}] Nesne Hedef: {pos} | Mesafe Sapması: {sapma:.2f}mm | İç Kuvvet: {f_int:.2f}N")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Bimanual Teşhis Panosu Oluşturuluyor...")
    profil_raporu = BimanualProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "bimanual_paneli.png")

    BimanualGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Bimanual Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 250 (FAZ 13): ÇİFT KOLLU (BIMANUAL) ROBOT KOORDİNASYON MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
