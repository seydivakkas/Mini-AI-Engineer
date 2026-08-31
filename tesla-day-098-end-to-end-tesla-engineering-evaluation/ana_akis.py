"""
Tesla Gün 98 Ana Akış (Tesla Day 98 Main Pipeline)
===================================================
Uçtan Uca Tesla Yazılım Mühendisliği Şampiyonluk Değerlendirmesi
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

from src.tesla_e2e_degerlendirici import TeslaE2EEngineeringEvaluator
from src.tesla_e2e_profilleyici import TeslaE2EProfilleyici
from src.tesla_e2e_gorsellestirici import TeslaE2EGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 98: UÇTAN UCA ŞAMPİYONLUK DEĞERLENDİRMESİ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 8 Temel Sütunda Bütünsel Sistem İncelemesi & Grandmaster Onayı")
    print("--------------------------------------------------------------------------------\n")

    # 1. 8 Sütun Değerlendirmesi
    print(" [1] Tesla 8 Temel Mühendislik Sütunu Metrikleri Denetleniyor...")
    evaluator = TeslaE2EEngineeringEvaluator()
    pillars = evaluator.evaluate_all_pillars()

    for p in pillars:
        print(f"     [{p.pillar_id}/8] {p.name:<40} | Hedef: {p.target_metric:<22} | Başarı: {p.achieved_value:<16} | Skor: %{p.score:.1f} [{p.status}]")

    final_res = evaluator.calculate_championship_score(pillars)
    print("\n [2] Şampiyonluk Skoru ve Sertifikasyon...")
    print(f"     -> Toplam Şampiyonluk Skoru: %{final_res['total_championship_score']:.1f} / 100.0 (KUSURSUZ)")
    print(f"     -> Kazanılan Unvan          : {final_res['title_awarded']}")
    print(f"     -> Sertifikasyon Durumu     : {final_res['certification_status']}")

    # 2. Profilleme
    profilleyici = TeslaE2EProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_evaluation_engine()
    print(f"\n [3] Değerlendirici RTOS Döngü Hızı: {metrikler['step_ortalama_us']:.2f} µs (P99: {metrikler['step_p99_us']:.2f} µs)")

    # 3. Tanı Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla Şampiyonluk Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaE2EGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_e2e_sampiyonluk_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🏆 GÜN 98 BAŞARIYLA TAMAMLANDI! TESLA GRANDMASTER MİMARİSİ ONAYLANDI! 🏆")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
