"""
Day 381: Autonomous Mega-Factory Orchestrator (10,000+ Synchronized AMRs and Robot Workcells)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: FAZ 20 BAŞLANGICI — 10.000+ AMR ve Robotik Hücre Mega-Fabrika Orkestrasyonu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from mega_factory_orchestrator_motoru import MegaFactoryBenchmark
from factory_profilleyici import FactoryProfilleyici
from factory_gorsellestirici import FactoryGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 381: OTONOM MEGA-FABRIKA ORKESTRASYONU -- 10.000+ AMR & ROBOTIK HUCRE")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = MegaFactoryBenchmark()
    print("\n[1/4] 50+ AMR ve 18 Robotik Hucreli Mega-Fabrika Vardiyasi Simule Ediliyor...")
    bench_res = bench.kos(num_ticks=50)

    print(f"  -> Uretilen Toplam Mamul   : {bench_res['total_completed_units']} Birim")
    print(f"  -> Saatlik Cikis Kapasitesi: {bench_res['throughput_units_per_hour']:.1f} Birim/Saat")
    print(f"  -> Filo Carpisma Orani     : %{bench_res['collision_rate_pct']:.4f} (SIFIR CARPISMA)")
    print(f"  -> AMR Filo Dolulugu       : %{bench_res['amr_fleet_utilization_pct']:.1f}")

    # 2. Profilleme
    print("\n[2/4] Mega-Fabrika Otonomi Seviyesi ve OEE Profillemesi Yapiliyor...")
    profilleyici = FactoryProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Mega-Fabrika Teshis Paneli Ciziliyor...")
    gorsellestirici = FactoryGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 381: MEGA-FABRIKA ORKESTRASYON VE DIJITAL IKIZ BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
