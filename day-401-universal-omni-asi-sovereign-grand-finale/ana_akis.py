"""
Day 401: Universal Omni-ASI v3.0 Sovereign Grand Finale
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 👑 401 Günlük Mini AI Engineer Büyük Finali & Omni-ASI v3.0 Zirvesi.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from omni_asi_sovereign_motoru import OmniASIGrandFinaleBenchmark
from omni_asi_profilleyici import OmniASIProfilleyici
from omni_asi_gorsellestirici import OmniASIGorsellestirici


def main():
    print("=" * 80)
    print(" [***] DAY 401: BUYUK FINAL 401 -- EVRENSEL SUPER-ZEKA VE OTONOM MEDENIYET (OMNI-ASI)")
    print("=" * 80)

    # 1. Benchmark Koşumu
    bench = OmniASIGrandFinaleBenchmark()
    print("\n[1/4] Omni-ASI v3.0 Bilisel Cekirdegi ve Gezegensel Medeniyet Orkestrasyonu...")
    bench_res = bench.kos()

    print(f"  -> ASI Surumu              : {bench_res['asi_version']}")
    print(f"  -> Tamamlanan Faz Sayisi   : {bench_res['total_phases_mastered']} Faz (%100)")
    print(f"  -> Tamamlanan Gun Sayisi   : {bench_res['total_days_completed']} Gun (%100)")
    print(f"  -> Gecen Birim Test Sayisi : {bench_res['total_unit_tests_passed']} / 1604 Test")
    print(f"  -> Bilisel Tutarlilik      : %{bench_res['cognitive_coherence_pct']:.2f}")
    print(f"  -> Fotonik Islem Gecikmesi : {bench_res['optical_latency_ps']} ps (Isik Hizi)")
    print(f"  -> Gezegensel Otonomi      : %{bench_res['planetary_autonomy_score']:.1f}")
    print(f"  -> Super-Zeka Seviyesi     : {bench_res['asi_quotient']:,.0f} ASI-Q")

    # 2. Profilleme
    print("\n[2/4] Evrensel Super-Zeka ve Medeniyet Otonomisi Mezuniyet Profillemesi...")
    profilleyici = OmniASIProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Buyuk Final Sahaser Teshis Paneli Ciziliyor...")
    gorsellestirici = OmniASIGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Buyuk Final Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma & Kutlama
    print("\n[4/4] *** 401 GUNLUK MINI AI ENGINEER ROADMAP %100 TAMAMLANDI! ***")
    print("=" * 80)


if __name__ == "__main__":
    main()
