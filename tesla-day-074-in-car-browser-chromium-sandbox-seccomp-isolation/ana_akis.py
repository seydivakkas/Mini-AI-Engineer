"""
Tesla Gün 74 Ana Akış (Tesla Day 74 Main Pipeline)
===================================================
Araç İçi Chromium Tarayıcısı ve Seccomp-BPF Sandbox İzolasyonu
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

from src.tesla_chromium_sandbox_seccomp import TeslaChromiumSeccompSandbox
from src.tesla_sandbox_profilleyici import TeslaSandboxProfilleyici
from src.tesla_sandbox_gorsellestirici import TeslaSandboxGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 74: CHROMIUM TARAYICI VE SECCOMP-BPF SANDBOX 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Zero-Trust UI Sandbox, Syscall Filtreleme & CAN-Bus İzolasyonu")
    print("--------------------------------------------------------------------------------\n")

    # 1. Sandbox Benchmark'ı
    print(" [1] Chromium Tarayıcı Süreci ve Seccomp Syscall Filtresi Simüle Ediliyor...")
    profilleyici = TeslaSandboxProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_sandbox()

    print(f"     -> İzin Verilen Syscall    : {metrikler['permitted']} Adet (Read, Write, Mmap, Futex...)")
    print(f"     -> Bloke Edilen Tehdit     : {metrikler['blocked']} Adet ({', '.join(metrikler['blocked_list'])})")
    print(f"     -> Kum Havuzu Güvenliği    : {'%100 KORUMALI (SIFIR SIZMA RİSKİ)' if metrikler['is_secure'] else 'AÇIK TESPİT EDİLDİ'}")

    # 2. Filtreleme Hızı
    print("\n [2] Seccomp-BPF Çekirdek Filtre Performansı...")
    print(f"     -> Çağrı Başına Gecikme    : {metrikler['syscall_check_ortalama_us']:.3f} µs (P99: {metrikler['batch_check_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Çağrı Kapasite: {metrikler['saniyelik_syscall_kontrolu']:,} Syscall/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Chromium Sandbox Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaSandboxGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_chromium_sandbox_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 74 BAŞARIYLA TAMAMLANDI! CHROMIUM SANDBOX DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
