"""
Tesla Gün 80 Ana Akış (Tesla Day 80 Main Pipeline)
===================================================
Dağıtık Güç Dağıtımı ve Dinamik Şebeke Yük Dengeleme Algoritmaları
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

from src.tesla_dinamik_yuk_dengeleyici import TeslaDynamicLoadBalancer
from src.tesla_yuk_profilleyici import TeslaYukProfilleyici
from src.tesla_yuk_gorsellestirici import TeslaYukGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 80: SUPERCHARGER DİNAMİK YÜK DENGELEME 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 1.0 MW Trafo Limiti, 8-Stall SoC Paylaşımı & Aşırı Yük Kalkanı")
    print("--------------------------------------------------------------------------------\n")

    # 1. Yük Dengeleme Benchmark'ı
    print(" [1] 8 Araçlık Dinamik Güç Paylaşımı ve Trafo Optimizasyonu Başlatılıyor...")
    profilleyici = TeslaYukProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_load_balancing()

    print(f"     -> Trafo Kapasitesi        : 1000.0 kW (1.0 MW)")
    print(f"     -> Toplam Dağıtılan Güç    : {metrikler['total_allocated']:.1f} kW")
    print(f"     -> Kalan Şebeke Rezervi    : {metrikler['grid_headroom']:.1f} kW")
    print(f"     -> Trafo Aşırı Yük Koruması: {'%100 SAĞLANDI (SIFIR AŞIM)' if metrikler['overload_prevented'] else 'AŞIM TESPİT EDİLDİ'}")

    # 2. Optimizasyon Hızı
    print("\n [2] Yük Dengeleme Optimizasyonu RTOS Performansı...")
    print(f"     -> Ortalama Adım Süresi    : {metrikler['balance_step_ortalama_us']:.3f} µs (P99: {metrikler['balance_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Dengeleme Hızı: {metrikler['saniyelik_dengeleme_kapasitesi']:,} Optimizasyon/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Dinamik Yük Dengeleme Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaYukGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_dinamik_yuk_dengeleme_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 80 BAŞARIYLA TAMAMLANDI! DİNAMİK YÜK DENGELEME DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
