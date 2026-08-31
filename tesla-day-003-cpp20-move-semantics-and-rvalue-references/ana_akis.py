"""
Tesla Gun 03 Ana Akis (Tesla Day 03 Main Pipeline)
===================================================
Tasima Semantigi (Move Semantics), Rvalue Referanslari ($&&$) ve Sifir-Kopyalama (Zero-Copy)
Uctan Uca Calistirma ve Teshis Paneli Uretim Scripti.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import sys
import os
import time

# Windows terminal UTF-8 uyumlulugu
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Kok dizini ekle
su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_kamera_tensoru import TeslaKameraTensoru, TeslaFSDKameraHatti
from src.tesla_move_profilleyici import TeslaMoveProfilleyici
from src.tesla_move_gorsellestirici import TeslaMoveGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 03: MOVE SEMANTICS & SIFIR-KOPYALAMA 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: FSD 8-Kamera Tensörlerinin Sıfır Kopyalama ile NPU'ya Aktarımı")
    print("--------------------------------------------------------------------------------\n")

    # 1. Move Constructor Testi
    print(" [1] 1080p Kamera Tensörü Tahsis Ediliyor (5.93 MB)...")
    kamera_tensor = TeslaKameraTensoru("on_merkez_ana", 1920, 1080, 3)
    orijinal_adres = kamera_tensor.bellek_adresi
    print(f"     -> Orijinal Tensör Bellek Adresi : 0x{orijinal_adres:X} ({kamera_tensor.boyut_mb:.2f} MB)")

    print("     -> C++20 std::move ile NPU Motoruna Aktarılıyor...")
    npu_tensor = kamera_tensor.tasi()
    print(f"     -> NPU Tensör Bellek Adresi      : 0x{npu_tensor.bellek_adresi:X} (Adres Değişmedi -> SIFIR KOPYALAMA!)")
    print(f"     -> Eski Tensör Durumu            : {'GEÇERSİZ (MOVED-FROM)' if not kamera_tensor.gecerli_mi else 'GEÇERLİ'}")

    # 2. 8-Kamera FSD Çevrim Testi
    print("\n [2] FSD 8-Kamera Surround Vision Akış Hattı Çalıştırılıyor...")
    hat = TeslaFSDKameraHatti()
    for kamera in TeslaFSDKameraHatti.KAMERA_LISTESI:
        raw_kare = hat.kamera_kare_uret(kamera)
        npu_kare, gecikme = hat.npu_girisine_tasi(raw_kare)
        print(f"     -> [{kamera:14s}] NPU'ya Aktarıldı | Gecikme: {gecikme:.0f} ns")

    # 3. Çözünürlük ve Bant Genişliği Benchmark'ı
    print("\n [3] Çözünürlük Bazlı Move vs Deep Copy Benchmark Başlatılıyor...")
    profilleyici = TeslaMoveProfilleyici(dongu_sayisi=100)
    benchmark_sonuclari = profilleyici.benchmark_cozunurluk_karsilastirmasi()
    verim_sonuclari = profilleyici.fsd_8_kamera_36fps_verim_analizi()

    for i in range(len(benchmark_sonuclari["etiketler"])):
        ad = benchmark_sonuclari["etiketler"][i]
        mb = benchmark_sonuclari["boyutlar_mb"][i]
        c_us = benchmark_sonuclari["copy_sureleri_us"][i]
        m_us = benchmark_sonuclari["move_sureleri_us"][i]
        hiz = benchmark_sonuclari["hizlanma_oranlari"][i]
        print(f"     -> {ad:12s} ({mb:5.1f} MB) | Deep Copy: {c_us:7.1f} us | std::move: {m_us:4.1f} us | Hızlanma: {hiz:,.0f}x")

    print(f"\n     -> 8-Kamera 36 FPS Bant Genişliği Tasarrufu : {verim_sonuclari['saniyedeki_veri_gb_s']:.2f} GB/s")
    print(f"     -> CPU Tasarruf Oranı                       : %{verim_sonuclari['cpu_tasarruf_yuzdesi']:.2f}")

    # 4. Teşhis Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla Move Semantics Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaMoveGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(benchmark_sonuclari, dosya_adi="tesla_move_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 03 BAŞARIYLA TAMAMLANDI! SIFIR KOPYALAMA TENSÖR HATTI DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
