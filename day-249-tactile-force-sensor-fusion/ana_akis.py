"""
Day 249: Dokunsal ve Kuvvet Sensörü Füzyonu ile Hassas Nesne Tutma Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.tactile_fusion_motoru import (
    GelSightTactileSensor,
    WristForceTorqueSensor,
    SlipDetectorAndGraspController,
    TactileGraspPipeline,
)
from src.tactile_profilleyici import TactileProfilleyici
from src.gorsellestirici import TactileGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 249 (FAZ 13): DOKUNSAL (TACTILE) VE KUVVET SENSÖRÜ FÜZYONU İLE HASSAS NESNE TUTMA")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: GelSight Optik Dokunsal Basınç Ölçümü
    # -------------------------------------------------------------
    print("\n[1/4] GelSight Optik Dokunsal Basınç Alanı Hesaplanıyor...")
    tactile = GelSightTactileSensor(res=32)
    patch = tactile.get_contact_patch(normal_kuvvet=4.5)
    print(f"  • Jel Basınç Haritası Boyutu: {patch['basinc_haritasi'].shape}")
    print(f"  • Temas Alanı (Piksel Sayısı): {patch['temas_alani_piksel']} piksel")
    print(f"  • Toplam Birlikte Baskı Kuvveti: {patch['toplam_baski_kuvveti']} N")

    # -------------------------------------------------------------
    # ADIM 2: 6-Eksenli Bilek Kuvvet ve Tork (F/T) Ölçümü
    # -------------------------------------------------------------
    print("\n[2/4] 6-Eksenli Bilek Kuvvet/Tork Sensörü (F/T Wrench) Okunuyor...")
    wrench = WristForceTorqueSensor.read_wrench(nesne_kutlesi_kg=0.25, ivme_m_s2=0.5)
    print(f"  • Kuvvet Vektörü (Fx, Fy, Fz)  : {wrench[:3].tolist()} N")
    print(f"  • Moment Vektörü (Tx, Ty, Tz)  : {wrench[3:].tolist()} Nm")

    # -------------------------------------------------------------
    # ADIM 3: 1000Hz Adaptif Kuvvet ve Kayma Önleme Simülasyonu
    # -------------------------------------------------------------
    print("\n[3/4] Kırılgan Nesne İçin 1000Hz Adaptif Kapalı Döngü Çalıştırılıyor...")
    pipeline = TactileGraspPipeline()
    sim_sonuc = pipeline.simulate_fragile_grasp(adim_sayisi=6)
    for adim, (fn, ft) in enumerate(zip(sim_sonuc["gecmis_Fn"], sim_sonuc["gecmis_Ft"]), 1):
        print(f"  [Zaman {adim}] Fn: {fn:.2f}N | Ft: {ft:.2f}N | Kırılma Riski: {'GÜVENLİ' if fn <= 12.0 else 'TEHLİKE'}")

    print(f"  ✓ Simülasyon Sonucu: Ezilme Yok={not sim_sonuc['kirilma_oldu_mu']} | Düşürme Yok={not sim_sonuc['dusurme_oldu_mu']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Dokunsal Füzyon Teşhis Panosu Oluşturuluyor...")
    profil_raporu = TactileProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "tactile_fusion_paneli.png")

    TactileGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Dokunsal Füzyon Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 249 (FAZ 13): DOKUNSAL VE KUVVET SENSÖRÜ FÜZYONU MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
