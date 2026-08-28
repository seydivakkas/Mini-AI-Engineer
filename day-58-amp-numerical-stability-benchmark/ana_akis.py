"""
Day 58: Otomatik Karma Hassasiyet (AMP), FP16 vs BF16, GradScaler ve Sayısal Kararlılık Ana Yürütme Betiği.
"""

import os
import sys
import torch

MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.sayisal_kararlilik import SayisalKararlilikAnalizoru
from src.amp_benchmark_motoru import AMPBenchmarkMotoru
from src.gorsellestirici import AMPGorsellestirici


def main():
    print("=" * 85, flush=True)
    print(">>> DAY 58: OTOMATİK KARMA HASSASİYET (AMP), FP16 VS BF16 & GRADSCALER", flush=True)
    print("=" * 85, flush=True)

    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Kullanılan Cihaz: {cihaz.upper()}", flush=True)
    if cihaz == "cuda":
        print(f"    - GPU Modeli : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"    - BF16 Desteği: {torch.cuda.is_bf16_supported()}", flush=True)

    # 1. Format Özellikleri ve Sayısal Kararlılık Analizi
    print("\n[+] 1. Adım: Kayan Nokta Formatlarının Sayısal Kararlılık Analizi...", flush=True)
    formatlar = SayisalKararlilikAnalizoru.format_ozelliklerini_getir()
    for f_ad, f_bilgi in formatlar.items():
        print(f"    - {f_ad:18s} | Üs: {f_bilgi['us_biti (exponent)']} bit | Mantis: {f_bilgi['mantis_biti (fraction)']} bit | Min Normal: {f_bilgi['min_pozitif_normal']} | Aralık: {f_bilgi['dinamik_aralik_onluk']}")

    simulasyon = SayisalKararlilikAnalizoru.gradyan_kaybi_simulasyonu(ornek_sayisi=100_000)
    print("\n[+] Gradyan Underflow / Overflow Simülasyonu (100.000 Gradyan Örneği):", flush=True)
    for mod_ad, metrikler in simulasyon.items():
        print(f"    - {mod_ad:22s} | Underflow: %{metrikler['underflow_orani']:6.2f} | Overflow: %{metrikler['overflow_orani']:4.2f} | Relatif Hata: {metrikler['ort_relatif_hata']:.2e}")

    # 2. AMP Benchmark Çalıştırma
    print("\n[+] 2. Adım: FP32 vs AMP-FP16 vs BF16 Performans ve Bellek Kıyaslaması Başlatılıyor...", flush=True)
    benchmark_motoru = AMPBenchmarkMotoru(device=cihaz)
    loader = benchmark_motoru.sentetik_veri_olustur(num_samples=1000, img_size=32, batch_size=64)

    sonuclar = benchmark_motoru.calistir_kiyaslama(loader=loader, epochs=4, isinma_adimlari=2)

    print("\n" + "=" * 85, flush=True)
    print(">>> 3. BENCHMARK VE BELLEK PERFORMANS TABLOSU", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Eğitim Modu':<24} | {'Throughput (img/s)':<18} | {'Ort. Batch (ms)':<15} | {'Zirve VRAM (MB)':<15} | {'Nihai Loss':<10}")
    print("-" * 90)

    fp32_thru = sonuclar.get("FP32 (Standart)", {}).get("throughput_ornek_s", 1.0)

    for mod, veri in sonuclar.items():
        hiz_katsayi = veri['throughput_ornek_s'] / max(fp32_thru, 1e-5)
        print(
            f"{mod:<24} | "
            f"{veri['throughput_ornek_s']:>14.1f} img/s | "
            f"{veri['ort_batch_ms']:>12.2f} ms | "
            f"{veri['peak_vram_mb']:>13.1f} MB | "
            f"{veri['nihai_loss']:>9.4f} ({hiz_katsayi:.2f}x)"
        )

    # 3. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 85, flush=True)
    print(">>> 4. 6 PANELLİ TEŞHİS PANOSUNUN ÜRETİLMESİ", flush=True)
    print("=" * 85, flush=True)

    hedef_pano = os.path.join(MEVCUT_DIZIN, "ciktilar", "amp_benchmark_paneli.png")
    cikis_yolu = AMPGorsellestirici.panel_ciz(
        benchmark_sonuclari=sonuclar,
        kararlilik_sonuclari=simulasyon,
        hedef_path=hedef_pano
    )
    print(f"[+] 6 Panelli AMP Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}", flush=True)
    print("=" * 85, flush=True)
    print("DAY 58: OTOMATİK KARMA HASSASİYET BENCHMARK BAŞARIYLA TAMAMLANDI!", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
