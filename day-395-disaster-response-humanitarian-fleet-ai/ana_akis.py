"""
Day 395: Autonomous Disaster Response & Humanitarian Logistics Fleet AI
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 72 Saatlik Büyük Deprem Afet Müdahale ve İnsani Yardım Filosu Simülasyonu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from disaster_response_motoru import DisasterResponseBenchmark
from disaster_profilleyici import DisasterProfilleyici
from disaster_gorsellestirici import DisasterGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 395: AFET MUDAHALE & INSANI YARDIM FILOSU OTONOM TRIYAJ VE DAGITIM AI")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = DisasterResponseBenchmark(num_zones=20)
    print("\n[1/4] 20 Sektorluk Afet Bolgesi Simule Ediliyor ve Filo Gorevlendiriliyor...")
    bench_res = bench.kos()

    print(f"  -> Toplam Kazazede Sayisi : {bench_res['total_victims']}")
    print(f"  -> Kirmizi Kritik Vaka    : {bench_res['red_critical_count']}")
    print(f"  -> Ortalama Mudahale Sure : {bench_res['avg_response_time_min']:.1f} dk")
    print(f"  -> Hayatta Kalma Orani    : %{bench_res['overall_survival_rate_pct']:.1f}")
    print(f"  -> Asilan Yol Blokaji     : {bench_res['roadblocks_bypassed_count']} Sektor")

    # 2. Profilleme
    print("\n[2/4] Insani Yardim Lojistigi ve Kriz Otonomi Profillemesi...")
    profilleyici = DisasterProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Afet Mudahale Teshis Paneli Ciziliyor...")
    gorsellestirici = DisasterGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 395: AFET MUDAHALE VE INSANI YARDIM BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
