"""
Day 385: Sub-Millimeter Precision Microsurgery Robot (Vascular Anastomosis & Tremor Cancellation)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 0.8 mm Vasküler Anastomoz, Cerrah El Titremesi Sönümleme ve Doku Koruyucu Robot Koşumu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from microsurgery_robot_motoru import MicrosurgeryBenchmark
from microsurgery_profilleyici import MicrosurgeryProfilleyici
from microsurgery_gorsellestirici import MicrosurgeryGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 385: MILIMETRE-ALTI HASSAS MIKRO-CERRAHI ROBOTU & TITREME SONUMLEME")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = MicrosurgeryBenchmark()
    print("\n[1/4] 100 Adimli Vaskuler Anastomoz Dikis ve Titreme Bastirma Simulasyonu...")
    bench_res = bench.kos(num_steps=100)

    print(f"  -> Titreme Sonumleme Orani: %{bench_res['tremor_attenuation_pct']:.2f}")
    print(f"  -> Ortalama Konum Hatasi  : {bench_res['avg_positioning_error_um']:.2f} mikrometre (um)")
    print(f"  -> Ham Cerrah Titremesi   : {bench_res['raw_hand_error_um']:.1f} mikrometre (um)")
    print(f"  -> Maksimum Temas Kuvveti : {bench_res['max_contact_force_n']:.4f} N")
    print(f"  -> Doku Butunlugu         : {bench_res['tissue_integrity_safe']}")

    # 2. Profilleme
    print("\n[2/4] Mikro-Cerrahi Robotik Performans ve Guvenlik Profillemesi...")
    profilleyici = MicrosurgeryProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Mikro-Cerrahi Teshis Paneli Ciziliyor...")
    gorsellestirici = MicrosurgeryGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 385: MILIMETRE-ALTI MIKRO-CERRAHI ROBOTU BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
