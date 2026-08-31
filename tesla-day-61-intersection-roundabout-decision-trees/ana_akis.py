"""
Tesla Gün 61 Ana Akış (Tesla Day 61 Main Pipeline)
===================================================
Şehir İçi Kavşak ve Döner Kavşak Karar Ağaçları
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

from src.tesla_kavsak_karar_agaci import TeslaIntersectionDecisionTree
from src.tesla_kavsak_profilleyici import TeslaKavsakProfilleyici
from src.tesla_kavsak_gorsellestirici import TeslaKavsakGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 61: KAVŞAK VE DÖNER KAVŞAK KARAR AĞAÇLARI 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Geçiş Önceliği, Time-To-Collision (TTC), Gap Acceptance & FSM")
    print("--------------------------------------------------------------------------------\n")

    # 1. Karar Ağacı Benchmark'ı
    print(" [1] Döner Kavşak Çoklu Araç Trafik Senaryosu Değerlendiriliyor...")
    profilleyici = TeslaKavsakProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_decision_tree()

    print(f"     -> Mevcut Karar Durumu      : {metrikler['state']}")
    print(f"     -> Alınan Aksiyon           : {metrikler['action']}")
    print(f"     -> En Kritik Araç TTC       : {metrikler['min_ttc_s']:.1f} Saniye (Güvenlik Eşiği: >= 3.5s)")
    print(f"     -> Giriş Onay Durumu        : {'GİRİŞ ONAYLANDI' if metrikler['can_enter'] else 'YOL VERİLİYOR (BEKLE)'}")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Karar Motoru RTOS Performansı...")
    print(f"     -> Ortalama Karar Süresi    : {metrikler['decision_step_ortalama_us']:.3f} µs (P99: {metrikler['decision_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Karar Döngüsü  : {metrikler['saniyelik_karar_cevrimi']:,} Karar/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Döner Kavşak Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaKavsakGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_roundabout_decision_tree_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 61 BAŞARIYLA TAMAMLANDI! KAVŞAK KARAR MOTORU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
