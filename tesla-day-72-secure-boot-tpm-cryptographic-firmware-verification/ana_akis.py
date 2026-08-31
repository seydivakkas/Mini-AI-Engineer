"""
Tesla Gün 72 Ana Akış (Tesla Day 72 Main Pipeline)
===================================================
Güvenli Önyükleme (Secure Boot), TPM 2.0 ve Kriptografik Doğrulama
Uçtan Uca Çalıştırma ve Teşhis Paneli Üretim Scripti.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
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

from src.tesla_secure_boot_dogrulayici import TeslaSecureBootValidator
from src.tesla_tpm_profilleyici import TeslaTPMProfilleyici
from src.tesla_tpm_gorsellestirici import TeslaTPMGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 72: GÜVENLİ ÖNYÜKLEME (SECURE BOOT) VE TPM 2.0 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Root of Trust, SHA-256 İmzalar, dm-verity & Tahrifat Kalkanı")
    print("--------------------------------------------------------------------------------\n")

    # 1. Secure Boot Benchmark'ı
    print(" [1] 4 Aşamalı Donanımsal Güven Zinciri (Chain of Trust) Doğrulanıyor...")
    profilleyici = TeslaTPMProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_secure_boot()

    print(f"     -> Güven Zinciri Durumu     : {'%100 ONAYLANDI (Root of Trust)' if metrikler['chain_verified'] else 'REDDEDİLDİ'}")
    print(f"     -> Stage 1 (ROM RoT)        : {'GEÇTİ' if metrikler['stages'][0] else 'KALDI'}")
    print(f"     -> Stage 2 (U-Boot RSA-4096): {'GEÇTİ' if metrikler['stages'][1] else 'KALDI'}")
    print(f"     -> Stage 3 (Kernel ECDSA)   : {'GEÇTİ' if metrikler['stages'][2] else 'KALDI'}")
    print(f"     -> Stage 4 (dm-verity Root) : {'GEÇTİ' if metrikler['stages'][3] else 'KALDI'}")

    # 2. Doğrulama Hızı
    print("\n [2] Kriptografik İmza ve Hash RTOS Performansı...")
    print(f"     -> Ortalama Doğrulama Süresi: {metrikler['validation_step_ortalama_us']:.3f} µs (P99: {metrikler['validation_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Doğrulama Hacmi: {metrikler['saniyelik_dogrulama_kapasitesi']:,} Blok/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Secure Boot ve TPM Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaTPMGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_secure_boot_tpm_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 72 BAŞARIYLA TAMAMLANDI! SECURE BOOT TPM 2.0 DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
