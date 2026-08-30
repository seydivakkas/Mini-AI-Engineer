"""
Day 388: Autonomous Legal Arbitration & Multi-Jurisdictional Compliance Sandbox
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 100 Sınır Ötesi Ticari Uyuşmazlık Dosyası Otonom Tahkim ve Tazminat Koşumu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from legal_arbitration_motoru import LegalArbitrationBenchmark
from legal_profilleyici import LegalProfilleyici
from legal_gorsellestirici import LegalGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 388: OTONOM HUKUKI TAHKIM & COKLU YARGI ALANI UYUMLULUK SANDBOX'I")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = LegalArbitrationBenchmark(num_cases=100)
    print("\n[1/4] 100 Sinir Otesi Ticari Uyusmazlik Dosyasi Tahkime Aliniyor...")
    bench_res = bench.kos()

    print(f"  -> Islenen Dava Sayisi     : {bench_res['total_cases_processed']}")
    print(f"  -> Kabul Edilen Ihaller    : {bench_res['liable_cases_count']}")
    print(f"  -> Reddedilen Dosyalar     : {bench_res['dismissed_cases_count']}")
    print(f"  -> Toplam Tazminat Tutari  : €{bench_res['total_damages_awarded_eur']:,.2f}")
    print(f"  -> Ortalama Cozum Suresi   : {bench_res['avg_arbitration_latency_ms']:.2f} ms")
    print(f"  -> Karar Dogruluk Orani    : %{bench_res['decision_accuracy_pct']:.1f}")

    # 2. Profilleme
    print("\n[2/4] Hukuki Tahkim ve Normatif Otonomi Seviyesi Profillemesi...")
    profilleyici = LegalProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Hukuki Tahkim Teshis Paneli Ciziliyor...")
    gorsellestirici = LegalGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 388: OTONOM HUKUKI TAHKIM VE UYUMLULUK BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
