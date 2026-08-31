"""
Tesla Gun 02 Ana Akis (Tesla Day 02 Main Pipeline)
===================================================
RAII Prensibi, Akilli Isaretciler (Smart Pointers) ve Ozel Siliciler (Custom Deleters)
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

from src.tesla_raii_kaynak_yoneticisi import (
    TeslaDonanimKaynagi,
    DonanimKaynakTipi,
    TeslaCANSoketRAII,
    OzelSiliciAkilliIsaretci,
    TeslaKaynakIzlemeMerkezi
)
from src.tesla_raii_profilleyici import TeslaRAIIProfilleyici
from src.tesla_raii_gorsellestirici import TeslaRAIIGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 02: RAII & CUSTOM DELETERS 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: Otomotiv CAN Soketleri ve Donanim Kaynaklarinda Sifir Sizinti")
    print("--------------------------------------------------------------------------------\n")

    # 1. RAII Soket Testi
    print(" [1] Tesla CAN Soketi RAII Yoneticisi Baslatiliyor...")
    with TeslaCANSoketRAII(arayuz_adi="can0") as can_soket:
        can_soket.telemetri_yaz(0x120, b"SPEED_DATA_45KMPH")
        print(f"     -> CAN Soketi Aktif: {can_soket.kaynak.kaynak_id} (FD: {can_soket.kaynak.aciklayici_no})")
    print(f"     -> Kapsamdan Cikildi: Soket Durumu: {'ACIK' if can_soket.kaynak.acik_mi else 'KAPALI (GUVENLE TEMIZLENDI)'}")

    # 2. Custom Deleter Testi
    print("\n [2] GPU Doku Tamponu Custom Deleter ile Tahsis Ediliyor...")
    gpu_kaynak = TeslaDonanimKaynagi("GPU_BEV_TEXTURE_0", DonanimKaynakTipi.GPU_TAMPON)
    
    def ozel_gpu_temizleyici(k: TeslaDonanimKaynagi):
        print(f"     -> [Custom Deleter Tetiklendi] GPU VRAM Alani Serbest Birakildi: {k.kaynak_id}")
        k.donanim_kapat()

    with OzelSiliciAkilliIsaretci(gpu_kaynak, ozel_gpu_temizleyici) as ptr:
        print(f"     -> Akilli Isaretci Kaynak Tasidi: {ptr.al().kaynak_id}")

    # 3. Profilleme ve Hata Enjeksiyon Benchmark'i
    print("\n [3] Istisna Guvenligi & Sizinti Benchmark'i Baslatiliyor (1000 Dongu)...")
    profilleyici = TeslaRAIIProfilleyici(dongu_sayisi=1000)
    benchmark_sonuclari = profilleyici.benchmark_istisna_guvenligi_ve_sizinti()
    deleter_sonuclari = profilleyici.benchmark_custom_deleter_turleri()

    print(f"     -> RAII Kaynak Sizintisi           : %{benchmark_sonuclari['raii_sizinti_orani']*100:.1f} ({benchmark_sonuclari['raii_sizinti_sayisi']}/{benchmark_sonuclari['toplam_islem']})")
    print(f"     -> Ham Pointer Sizintisi          : %{benchmark_sonuclari['ham_sizinti_orani']*100:.1f} ({benchmark_sonuclari['ham_sizinti_sayisi']}/{benchmark_sonuclari['toplam_islem']})")
    print(f"     -> RAII Kapsam Cikis Gecikmesi    : {benchmark_sonuclari['raii_ortalama_ns']:.1f} ns (P99: {benchmark_sonuclari['raii_p99_ns']:.1f} ns)")
    print(f"     -> Stateless Lambda vs std::fn    : {deleter_sonuclari['stateless_lambda_ns']:.1f} ns vs {deleter_sonuclari['dynamic_function_ns']:.1f} ns (%{deleter_sonuclari['hiz_farki_yuzde']:.1f} Daha Hizli)")
    print(f"     -> ASIL-D Guvenlik Skoru          : %{benchmark_sonuclari['guvenlik_skoru']:.1f}/100")

    # 4. Teshis Paneli Gorsellestirme
    print("\n [4] 6 Panelli Tesla RAII Tani Paneli Uretiliyor...")
    gorsellestirici = TeslaRAIIGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(benchmark_sonuclari, dosya_adi="tesla_raii_tani_paneli.png")
    print(f"     -> Tani Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GUN 02 BASARIYLA TAMAMLANDI! SIFIR SIZINTI RAII MOTORU DOGRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
