"""
Day 399: Universal Polymath Autonomous Scientific Researcher & Patent Drafter
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 50 Disiplinlerarası Bilimsel Keşif Hipotezi ve Resmi USPTO Patent Başvuru Taslağı.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from polymath_patent_motoru import UniversalPolymathBenchmark
from polymath_profilleyici import PolymathProfilleyici
from polymath_gorsellestirici import PolymathGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 399: EVRENSEL BILIMSEL ARASTIRMACI: HIPOTEZDEN PATENT BASVURUSUNA UCTAN UCA AJAN")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = UniversalPolymathBenchmark(num_hypotheses=50)
    print("\n[1/4] 50 Disiplinlerarasi Hipotez Sentezleniyor ve USPTO Istemleri Taslaklaniyor...")
    bench_res = bench.kos()

    print(f"  -> Uretilen Hipotez     : {bench_res['num_hypotheses']} Adet")
    print(f"  -> Ortalama Yenilik     : %{bench_res['avg_novelty_pct']:.1f}")
    print(f"  -> Fiziksel Gerceklik   : %{bench_res['avg_plausibility_pct']:.1f}")
    print(f"  -> In-Silico Dogrulama  : %{bench_res['in_silico_validated_pct']:.1f}")
    print(f"  -> Secilen Bulus Basligi: {bench_res['best_hypothesis'].title}")
    print(f"  -> Hazirlanan Istem     : {bench_res['drafted_claims_count']} Istem (1 Bagimsiz, 9 Bagimli)")

    # 2. Profilleme
    print("\n[2/4] Polimat Bilimsel Arastirma ve Patentlenebilirlik Profillemesi...")
    profilleyici = PolymathProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Polimat Patent Teshis Paneli Ciziliyor...")
    gorsellestirici = PolymathGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 399: EVRENSEL BILIMSEL ARASTIRMACI BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
