"""
Day 396: Autonomous Cyber Defense: Real-Time Zero-Day Vaccine Synthesis
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 500 Zero-Day Siber Saldırısı Analiz ve Anlık İkili Aşı Sentezi.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from cyber_defense_motoru import AutonomousImmunizationBenchmark
from cyber_profilleyici import CyberProfilleyici
from cyber_gorsellestirici import CyberGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 396: OTONOM SIBER SAVUNMA: GERCEK ZAMANLI ZERO-DAY ASI SENTEZI")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = AutonomousImmunizationBenchmark(num_exploits=500)
    print("\n[1/4] 500 Zero-Day Istismar Girdisi Sembolik Analiz Ediliyor ve Asi Sentezleniyor...")
    bench_res = bench.kos()

    print(f"  -> Test Edilen Saldiri  : {bench_res['total_exploits_tested']}")
    print(f"  -> Neutralized Sayisi   : {bench_res['neutralized_count']}")
    print(f"  -> Notralizasyon Orani  : %{bench_res['neutralization_rate_pct']:.1f}")
    print(f"  -> Ortalama Asi Suresi  : {bench_res['avg_synthesis_time_ms']:.1f} ms")
    print(f"  -> Maksimum Asi Suresi  : {bench_res['max_synthesis_time_ms']:.1f} ms")

    # 2. Profilleme
    print("\n[2/4] Siber Guvenlik ve Canli Yama Guvenlik Profillemesi...")
    profilleyici = CyberProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Siber Bagisiklik Teshis Paneli Ciziliyor...")
    gorsellestirici = CyberGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 396: OTONOM SIBER SAVUNMA BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
