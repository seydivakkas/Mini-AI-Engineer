"""
Day 251: İnsansı (Humanoid) Robotik Bütünsel Hareket Kontrolü (Whole-Body Control & ZMP) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.humanoid_wbc_motoru import (
    LIPMDynamics,
    SupportPolygon,
    HierarchicalQPController,
)
from src.humanoid_wbc_profilleyici import HumanoidWBCProfilleyici
from src.gorsellestirici import HumanoidWBCGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 251 (FAZ 13): İNSANSI ROBOTİK BÜTÜNSEL HAREKET KONTROLÜ (WHOLE-BODY CONTROL & ZMP DENGESİ)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: 3D LIPM ve Destek Poligonunun Kurulması
    # -------------------------------------------------------------
    print("\n[1/4] 3D LIPM ve Çift Ayak Destek Poligonu Modelleniyor...")
    lipm = LIPMDynamics(com_yukseklik_m=0.85)
    polygon = SupportPolygon(ayak_uzunlugu_m=0.22, ayak_genisligi_m=0.12, ayak_arasi_mesafe_m=0.20)

    print(f"  • Kütle Merkezi Yüksekliği (z_c): {lipm.z_c} m")
    print(f"  • Doğal Sarkaç Frekansı (omega) : {lipm.omega:.2f} rad/s")
    print(f"  • Destek Poligonu Sınırları (X) : [{polygon.x_min:.2f} m, {polygon.x_max:.2f} m]")
    print(f"  • Destek Poligonu Sınırları (Y) : [{polygon.y_min:.2f} m, {polygon.y_max:.2f} m]")

    # -------------------------------------------------------------
    # ADIM 2: Hiyerarşik QP Kontrolcüsünün Yapılandırılması
    # -------------------------------------------------------------
    print("\n[2/4] Hiyerarşik QP Whole-Body Kontrolcüsü Başlatılıyor...")
    wbc = HierarchicalQPController(lipm=lipm, polygon=polygon)

    # -------------------------------------------------------------
    # ADIM 3: 80N Dış İtme Karşısında Denge Optimizasyonu
    # -------------------------------------------------------------
    print("\n[3/4] 80N Dış İtme (Push Disturbance) ve Dengeleme Simülasyonu...")
    com_pos = np.array([0.02, 0.01])
    com_vel = np.array([0.15, 0.05])
    hedef_com = np.array([0.0, 0.0])
    dis_kuvvet = np.array([80.0, 20.0])  # 80N X yönünde, 20N Y yönünde ani itme

    sonuc = wbc.optimize_wbc_step(
        com_pos=com_pos,
        com_vel=com_vel,
        hedef_com=hedef_com,
        dis_kuvvet_N=dis_kuvvet,
        robot_kutlesi_kg=55.0,
    )

    print(f"  • Optimize CoM İvmesi     : {sonuc['opt_com_acc']} m/s²")
    print(f"  • Hesaplanan ZMP Konumu   : {sonuc['opt_zmp']} m")
    print(f"  • ZMP Denge Kararlılığı   : {'STABİL VE GÜVENLİ' if sonuc['stabil_mi'] else 'DÜŞÜYOR'}")
    print(f"  • Denge Güvenlik Marjini  : {sonuc['guvenlik_marjini_cm']} cm")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Humanoid WBC Teşhis Panosu Oluşturuluyor...")
    profil_raporu = HumanoidWBCProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "humanoid_wbc_paneli.png")

    HumanoidWBCGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Humanoid WBC Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 251 (FAZ 13): İNSANSI ROBOTİK BÜTÜNSEL HAREKET KONTROLÜ (WBC) MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
