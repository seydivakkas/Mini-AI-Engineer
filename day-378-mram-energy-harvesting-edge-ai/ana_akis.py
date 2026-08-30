"""
Day 378: Energy-Harvesting STT-MRAM Ultra-Low-Power Edge AI Accelerator
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: Ortam Enerjisi Hasadı, STT-MRAM Kesintili Çıkarım ve Raporlama.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from mram_edge_ai_motoru import MRAMEdgeAIBenchmark
from mram_edge_profilleyici import MRAMEdgeProfilleyici
from mram_edge_gorsellestirici import MRAMEdgeGorsellestirici


def main():
    print("=" * 70)
    print(" DAY 378: ENERGY-HARVESTING STT-MRAM ULTRA-LOW-POWER EDGE AI ACCELERATOR")
    print("=" * 70)

    # 1. Benchmark Koşumu
    bench = MRAMEdgeAIBenchmark()
    print("\n[1/4] Dalgalı Ortam Enerjisi ve Kesintili STT-MRAM Simülasyonu Başlatılıyor...")
    bench_res = bench.kos(num_steps=300)

    print(f"  -> Tamamlanan Çıkarım Sayısı    : {bench_res['completed_inferences']} Adet")
    print(f"  -> STT-MRAM TMR Oranı           : %{bench_res['tmr_ratio']:.1f}")
    print(f"  -> SRAM Sızıntı Kaybı           : {bench_res['sram_leakage_uj']:.4f} uJ")
    print(f"  -> MRAM Sızıntı Kaybı           : {bench_res['mram_leakage_uj']:.4f} uJ (%100 Tasarruf)")

    # 2. Profilleme
    print("\n[2/4] Enerji Hasadı ve Kesintili Hesaplama Profillemesi Yapılıyor...")
    profilleyici = MRAMEdgeProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yüksek Çözünürlüklü STT-MRAM Teşhis Paneli Çiziliyor...")
    gorsellestirici = MRAMEdgeGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teşhis Paneli Kaydedildi: {panel_yolu}")

    # 4. Özet Çıktı
    print("\n[4/4] Energy-Harvesting STT-MRAM Edge AI Akışı Başarıyla Tamamlandı!")
    print("=" * 70)


if __name__ == "__main__":
    main()
