"""
Day 376: LLM-Assisted RTL Verification and SystemVerilog Assertions (SVA) Generation
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: LLM ile RTL Şartnamesinden SVA Üretimi, Formel Doğrulama ve Raporlama.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from llm_rtl_sva_motoru import LLMRTLSVABenchmark
from rtl_sva_profilleyici import RTLSVAProfilleyici
from rtl_sva_gorsellestirici import RTLSVAGorsellestirici


def main():
    print("=" * 70)
    print(" DAY 376: LLM DESTEKLİ RTL DOĞRULAMA VE SYSTEMVERILOG ASSERTIONS (SVA)")
    print("=" * 70)

    # 1. Benchmark Koşumu
    bench = LLMRTLSVABenchmark()
    print("\n[1/4] LLM Destekli SVA Sentezi ve Formel RTL Simülasyonu Başlatılıyor...")
    bench_res = bench.kos(num_cycles=500)

    print(f"  -> Sentezlenen SVA İddia Sayısı : {len(bench_res['assertions'])}")
    print(f"  -> Enjekte Edilen Hatalar       : {bench_res['bugs_injected']}")
    print(f"  -> Yakalanan Hata Sayısı        : {bench_res['bugs_detected']} (%{bench_res['detection_rate']:.1f})")
    print(f"  -> Doğrulama Hızlanması         : {bench_res['speedup_x']}x")

    # 2. Profilleme
    print("\n[2/4] Doğrulama Metrikleri Profillemesi Yapılıyor...")
    profilleyici = RTLSVAProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yüksek Çözünürlüklü Teşhis Paneli Çiziliyor...")
    gorsellestirici = RTLSVAGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teşhis Paneli Kaydedildi: {panel_yolu}")

    # 4. Özet Çıktı
    print("\n[4/4] LLM-RTL SVA Doğrulama Akışı Başarıyla Tamamlandı! Tape-out İçin Hazır.")
    print("=" * 70)


if __name__ == "__main__":
    main()
