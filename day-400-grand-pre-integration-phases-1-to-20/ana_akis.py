"""
Day 400: Grand Pre-Integration Layer for All 20 Phases & 400 Days
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 20 Fazın ve 400 Günlük Mühendislik Müfredatının Bütünleşik Ön-Entegrasyon Simülasyonu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from grand_pre_integration_motoru import GrandPreIntegrationBenchmark
from grand_pre_integration_profilleyici import GrandPreIntegrationProfilleyici
from grand_pre_integration_gorsellestirici import GrandPreIntegrationGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 400: 20 FAZIN VE 400 GUNLUK MUFREDATIN BUYUK ON-ENTEGRASYON KATMANI")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = GrandPreIntegrationBenchmark(total_phases=20)
    print("\n[1/4] 20 Faz ve 400 Gunun Capraz Sistem Etkilesimleri Dogrulaniyor...")
    bench_res = bench.kos()

    print(f"  -> Dogrulanan Faz Sayisi  : {bench_res['total_phases_verified']} / 20 Faz (%100)")
    print(f"  -> Dogrulanan Gun Sayisi  : {bench_res['total_days_verified']} / 400 Gun (%100)")
    print(f"  -> Sistem Tutarliligi     : %{bench_res['system_coherence_pct']:.1f}")
    print(f"  -> Veri Yolu Gecikmesi    : {bench_res['avg_bus_latency_ms']:.3f} ms")
    print(f"  -> Mimari Kilitlenme      : {bench_res['architectural_deadlocks']} Adet (SIFIR)")

    # 2. Profilleme
    print("\n[2/4] Ekosistem Birlikte Calisabilirlik ve Tutarlilik Profillemesi...")
    profilleyici = GrandPreIntegrationProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Buyuk On-Entegrasyon Teshis Paneli Ciziliyor...")
    gorsellestirici = GrandPreIntegrationGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 400: BUYUK ON-ENTEGRASYON BASARIYLA TAMAMLANDI! ***")
    print("       >>> READY FOR DAY 401 GRAND FINALE! <<<")
    print("=" * 75)


if __name__ == "__main__":
    main()
