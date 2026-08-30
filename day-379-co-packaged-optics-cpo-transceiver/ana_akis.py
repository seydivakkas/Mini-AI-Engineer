"""
Day 379: Co-Packaged Optics (CPO) High-Speed Optical Transceiver Modeling
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 800G CPO Optik Bağlantı Simülasyonu, Göz Diyagramı ve Raporlama.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from cpo_transceiver_motoru import CPOBenchmark
from cpo_profilleyici import CPOProfilleyici
from cpo_gorsellestirici import CPOGorsellestirici


def main():
    print("=" * 70)
    print(" DAY 379: CO-PACKAGED OPTICS (CPO) 800G/1.6T HIGH-SPEED OPTICAL TRANSCEIVER")
    print("=" * 70)

    # 1. Benchmark Koşumu
    bench = CPOBenchmark()
    print("\n[1/4] 8-Şeritli 112 Gbps PAM4 CPO Bağlantı Simülasyonu Koşturuluyor...")
    bench_res = bench.kos(num_symbols=5000)

    print(f"  -> Toplam İletim Hızı          : {bench_res['aggregate_data_rate_gbps']:.1f} Gbps")
    print(f"  -> CPO Enerji Tüketimi         : {bench_res['cpo_energy_pj_bit']:.1f} pJ/bit")
    print(f"  -> Takılabilir Modül Tüketimi  : {bench_res['pluggable_energy_pj_bit']:.1f} pJ/bit ({bench_res['energy_savings_x']:.1f}x Tasarruf)")
    print(f"  -> Ham Bit Hata Oranı (BER)    : {bench_res['ber']:.6f}")

    # 2. Profilleme
    print("\n[2/4] Optik Sinyal Bütünlüğü ve Enerji Tasarrufu Profillemesi Yapılıyor...")
    profilleyici = CPOProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yüksek Çözünürlüklü CPO Teşhis Paneli Çiziliyor...")
    gorsellestirici = CPOGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teşhis Paneli Kaydedildi: {panel_yolu}")

    # 4. Özet Çıktı
    print("\n[4/4] Co-Packaged Optics (CPO) Simülasyonu Başarıyla Tamamlandı!")
    print("=" * 70)


if __name__ == "__main__":
    main()
