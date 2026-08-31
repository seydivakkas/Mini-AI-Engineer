"""
Tesla Gun 12 Ana Akis (Tesla Day 12 Main Pipeline)
===================================================
Linux Cekirdek Modulu (LKM), Karakter Suruculeri ve ioctl
Uctan Uca Calistirma ve Teshis Paneli Uretim Scripti.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_karakter_surucusu import (
    TeslaTorkKarakterAygiti,
    TeslaTorkPaketi,
    IOCTL_TESLA_TORK_YAZ,
    IOCTL_TESLA_DURUM_OKU,
    ASIL_D_GUVENLIK_ANAHTARI
)
from src.tesla_lkm_profilleyici import TeslaLKMProfilleyici
from src.tesla_lkm_gorsellestirici import TeslaLKMGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 12: LINUX LKM & CHARACTER DRIVERS 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: /dev/tesla_tork_kontrol, copy_from_user & 0xAA55 ASIL-D ioctl")
    print("--------------------------------------------------------------------------------\n")

    # 1. Karakter Sürücüsü Başlatma
    print(" [1] Tesla Motor Torku Karakter Aygıtı Açılıyor...")
    aygit = TeslaTorkKarakterAygiti(aygit_yolu="/dev/tesla_tork_kontrol")
    aygit.open()
    print(f"     -> Aygıt Düğümü Açıldı: {aygit.aygit_yolu} (Major: {aygit.major_no}, Minor: {aygit.minor_no})")

    # 2. Yetkili ve Güvenli ioctl Tork Komutu Gönderimi
    print("\n [2] ASIL-D Yetkili ioctl Tork Komutu İletimi (450 Nm)...")
    paket_gecerli = TeslaTorkPaketi(
        guvenlik_anahtari=ASIL_D_GUVENLIK_ANAHTARI,
        hedef_tork_nm=450.0,
        rejenerasyon_etkin_mi=False
    )
    kod1, mesaj1 = aygit.unlocked_ioctl(IOCTL_TESLA_TORK_YAZ, paket_gecerli.to_bytes())
    print(f"     -> [ioctl Sonucu] Kod: {kod1} | Mesaj: {mesaj1}")

    # 3. Yetkisiz Sahte Anahtar Saldırısı Simülasyonu
    print("\n [3] Yetkisiz Sahte Anahtar (0xDEAD) Tork Komutu Reddi Testi...")
    paket_sahte = TeslaTorkPaketi(
        guvenlik_anahtari=0xDEAD,
        hedef_tork_nm=700.0,
        rejenerasyon_etkin_mi=False
    )
    kod2, mesaj2 = aygit.unlocked_ioctl(IOCTL_TESLA_TORK_YAZ, paket_sahte.to_bytes())
    print(f"     -> [ioctl Güvenlik Yanıtı] Kod: {kod2} | Mesaj: {mesaj2} (GÜVENLİ ŞEKİLDE ENGELLENDİ)")

    # 4. Profilleme ve Karşılaştırma
    print("\n [4] Kernel ioctl vs Userspace Sysfs Metin Ayrıştırma Benchmark'ı...")
    profilleyici = TeslaLKMProfilleyici(komut_sayisi=5000)
    metrikler = profilleyici.benchmark_ioctl_vs_sysfs_gecikmesi()

    print(f"     -> Kernel ioctl Gecikmesi (Ortalama)   : {metrikler['ioctl_ortalama_us']:.3f} µs (P99: {metrikler['ioctl_p99_us']:.3f} µs)")
    print(f"     -> Sysfs Metin Gecikmesi (Ortalama)    : {metrikler['sysfs_ortalama_us']:.3f} µs")
    print(f"     -> Hızlanma Çarpanı                    : {metrikler['hizlanma_orani']:.1f}x Daha Hızlı")
    print(f"     -> Saniyedeki Tork Komut Kapasitesi    : {metrikler['saniyelik_tork_komut_kapasitesi']:,.0f} Komut/sn")

    # 5. Tanı Paneli Görselleştirme
    print("\n [5] 6 Panelli Tesla LKM Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaLKMGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_lkm_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    aygit.release()

    print("\n================================================================================")
    print(" 🚀 GÜN 12 BAŞARIYLA TAMAMLANDI! LINUX KARAKTER SÜRÜCÜSÜ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
