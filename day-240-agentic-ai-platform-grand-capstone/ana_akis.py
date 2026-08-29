"""
Day 240: Otonom Ajan Süiti (Agentic AI OS) - FAZ 12 BİTİRME PROJESİ & FİNALİ Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.platform_ajani_motoru import AgenticAIPlatform
from src.platform_profilleyici import PlatformProfilleyici
from src.gorsellestirici import PlatformGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 240 (FAZ 12 BİTİRME PROJESİ): OTONOM AJAN SÜİTİ VE İŞLETİM SİSTEMİ (AGENTIC AI OS CAPSTONE)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Platform Alt Sistemlerinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Agentic AI OS Platform Alt Sistemleri Kontrol Ediliyor...")
    platform = AgenticAIPlatform()
    for sistem, durum in platform.sistem_durumu.items():
        print(f"  • Alt Sistem: [{sistem.upper()}] -> Durum: {durum}")

    # -------------------------------------------------------------
    # ADIM 2: Uçtan Uca Kurumsal Görev İcrası
    # -------------------------------------------------------------
    gorev = "Kurumsal Finansal Raporlama, SQL Sorgulama ve Güvenli Canlıya Alma"
    print(f"\n[2/4] Kapsamlı Görev Boru Hattı Başlatılıyor: '{gorev}'")

    rapor = platform.tam_is_akisi_yurut(gorev, kritik_eylem_var_mi=True, insan_onayi=True)

    # -------------------------------------------------------------
    # ADIM 3: İcra Günlüğü ve Sistem Karnesi
    # -------------------------------------------------------------
    print("\n[3/4] Platform İcra Günlüğü:")
    for log in rapor["gunluk"]:
        print("  " + log)

    print(f"\n  🎯 Nihai Görev Durumu : {rapor['durum']}")
    print(f"  ⭐ Kalite ve Onay Skoru: {rapor['kalite_skoru']}/100 (Kusursuz)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli FAZ 12 Büyük Bitirme Teşhis Panosu Oluşturuluyor...")
    profil_raporu = PlatformProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "capstone_ajani_paneli.png")

    PlatformGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Agentic AI OS Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("🎉 TEBRİKLER: FAZ 12 (GÜN 221 - 240) %100 EKSİKSİZ VE BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
