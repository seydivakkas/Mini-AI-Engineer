"""
Day 246: Isaac Sim & PyBullet ile Dijital İkiz ve Sentetik Veri Üretimi Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.digital_twin_motoru import (
    RobotKinematics,
    DigitalTwinSimulator,
    SyntheticDataFactory,
)
from src.digital_twin_profilleyici import DigitalTwinProfilleyici
from src.gorsellestirici import DigitalTwinGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 246 (FAZ 13): SİMÜLASYONDA ROBOTİK — ISAAC SIM & PYBULLET İLE DİJİTAL İKİZ VE SENTETİK VERİ ÜRETİMİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Robot Kinematik Modeli ve IK Çözümü
    # -------------------------------------------------------------
    print("\n[1/4] 7-DoF Robot Kinematik Modeli ve Ters Kinematik (IK) Çözülüyor...")
    kinematics = RobotKinematics()
    hedef_pos = np.array([0.35, 0.15, 0.25])
    eklem_cozumu = kinematics.inverse_kinematics(hedef_pos, np.zeros(7))
    fk_dogrulama = kinematics.forward_kinematics(eklem_cozumu)

    print(f"  • Hedef 3D Konum (x, y, z)    : {hedef_pos.tolist()} metre")
    print(f"  • Hesaplanan Eklem Açıları (q): {eklem_cozumu.round(3).tolist()} rad")
    print(f"  • İleri Kinematik Doğrulama   : {fk_dogrulama.tolist()} metre")

    # -------------------------------------------------------------
    # ADIM 2: Dijital İkiz Fizik Simülasyon Adımları
    # -------------------------------------------------------------
    print("\n[2/4] Dijital İkiz Fiziksel Simülasyonu Çalıştırılıyor (100Hz)...")
    sim = DigitalTwinSimulator(dof=7, dt=0.01)
    for adim in range(1, 6):
        durum = sim.step_simulation(eklem_cozumu)
        eef = durum["eef_3d_konum"]
        print(f"  [Sim Adımı {adim}] Zaman: {durum['zaman_sn']}s | EEF: [x={eef[0]:.3f}, y={eef[1]:.3f}, z={eef[2]:.3f}]")

    # -------------------------------------------------------------
    # ADIM 3: Çok Modlu Sentetik Veri Üretimi
    # -------------------------------------------------------------
    print("\n[3/4] Sentetik RGB-D ve Semantik Segmentasyon Sahnesi Render Ediliyor...")
    sentetik = SyntheticDataFactory.render_synthetic_scene(
        eef_pos=fk_dogrulama,
        object_pos=hedef_pos,
        res=64,
    )
    print(f"  ✓ Sentetik RGB Boyutu        : {sentetik['rgb'].shape}")
    print(f"  ✓ Sentetik Derinlik Boyutu   : {sentetik['depth'].shape}")
    print(f"  ✓ Semantik Maske Sınıfları   : {np.unique(sentetik['seg_mask']).tolist()} [0: Zemin, 1: Robot, 2: Nesne]")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Dijital İkiz Teşhis Panosu Oluşturuluyor...")
    profil_raporu = DigitalTwinProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "digital_twin_paneli.png")

    DigitalTwinGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Dijital İkiz Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 246 (FAZ 13): ROBOTİK DİJİTAL İKİZ VE SENTETİK VERİ MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
