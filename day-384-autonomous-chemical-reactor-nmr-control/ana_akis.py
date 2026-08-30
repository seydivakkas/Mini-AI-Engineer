"""
Day 384: Autonomous Chemical Reactor Control with Real-Time NMR Spectroscopy Feedback
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: CSTR Kimyasal Reaktör, Çevrimiçi 1H-NMR ve Termal Kaçak Önleme Koşumu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from chemical_reactor_nmr_motoru import ChemicalReactorBenchmark
from reactor_profilleyici import ReactorProfilleyici
from reactor_gorsellestirici import ReactorGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 384: OTONOM KIMYASAL REAKTOR & CEVRIMICI 1H-NMR GERI BILDIRIM KONTROLU")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = ChemicalReactorBenchmark()
    print("\n[1/4] 50 Adimli CSTR Kimyasal Sentez ve NMR Tabanli Kontrol Simule Ediliyor...")
    bench_res = bench.kos(num_steps=50)

    print(f"  -> Hedef Urun C Sentez Verimi: %{bench_res['final_yield_pct']:.2f}")
    print(f"  -> Maksimum Reaktor Sicakligi: {bench_res['max_reactor_temp_k']:.1f} K")
    print(f"  -> Termal Kacak Guvenligi    : {bench_res['thermal_runaway_safe']}")
    print(f"  -> NMR Pik Kestirim Hatasi   : %{bench_res['avg_nmr_estimation_error_pct']:.2f}")

    # 2. Profilleme
    print("\n[2/4] Kimyasal Reaktor Otonomi Seviyesi Profillemesi Yapiliyor...")
    profilleyici = ReactorProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Kimyasal Reaktor Teshis Paneli Ciziliyor...")
    gorsellestirici = ReactorGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 384: OTONOM KIMYASAL REAKTOR VE NMR KONTROLU BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
