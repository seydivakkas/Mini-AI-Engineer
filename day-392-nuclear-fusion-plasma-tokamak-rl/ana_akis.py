"""
Day 392: Nuclear Fusion Plasma Control: Tokamak Magnetic Field Deep RL
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 1000 Adımlık Tokamak Plazma Kararlılığı ve Deep RL Kapalı Çevrim Kontrol Simülasyonu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from fusion_tokamak_rl_motoru import FusionTokamakBenchmark
from fusion_profilleyici import FusionProfilleyici
from fusion_gorsellestirici import FusionGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 392: NUKLEER FUZYON PLAZMA KARARLILIGI: TOKAMAK DEEP RL KONTROLCUSU")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = FusionTokamakBenchmark(steps=1000)
    print("\n[1/4] 1000 Adimlik (10 kHz) Tokamak Manyetik Atim Simule Ediliyor...")
    bench_res = bench.kos()

    print(f"  -> Simule Edilen Sure   : {bench_res['simulated_duration_ms']:.1f} ms")
    print(f"  -> VDE Onleme Basarisi  : %{bench_res['vde_avoidance_success_pct']:.1f}")
    print(f"  -> Maksimum Dikey Sapma : {bench_res['max_vertical_drift_mm']:.2f} mm")
    print(f"  -> RMS Dikey Hata       : {bench_res['rms_vertical_error_mm']:.2f} mm")
    print(f"  -> Maksimum Bobin Voltaji: {bench_res['max_coil_voltage_kv']:.2f} kV")

    # 2. Profilleme
    print("\n[2/4] Fuzyon Plazma Kararlilik ve Manyetik Otonomi Profillemesi...")
    profilleyici = FusionProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Fuzyon Plazma Teshis Paneli Ciziliyor...")
    gorsellestirici = FusionGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 392: NUKLEER FUZYON PLAZMA KONTROLU BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
