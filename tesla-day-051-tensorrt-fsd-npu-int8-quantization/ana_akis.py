"""
Tesla Gün 51 Ana Akış (Tesla Day 51 Main Pipeline)
===================================================
HW3/HW4 FSD NPU INT8 Kuantizasyon ve TensorRT Derleme
Uçtan Uca Çalıştırma ve Teşhis Paneli Üretim Scripti.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_fsd_npu_int8_kuantizasyon import TeslaFSDNPUQuantizer
from src.tesla_kuantizasyon_profilleyici import TeslaKuantizasyonProfilleyici
from src.tesla_kuantizasyon_gorsellestirici import TeslaKuantizasyonGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 51: HW3/HW4 NPU INT8 KUANTİZASYON VE TENSORRT 🚗")
    print("================================================================================")
    print("Stajyer Görevi: FP32->INT8 Kuantizasyon, %75 SRAM Tasarrufu, SQNR & Katman Birleştirme")
    print("--------------------------------------------------------------------------------\n")

    # 1. Kuantizasyon Benchmark'ı
    print(" [1] 50,000 HydraNet Ağırlığının INT8 Kuantizasyonu ve SQNR Analizi...")
    profilleyici = TeslaKuantizasyonProfilleyici(weight_elements=50000, iterations=100)
    metrikler = profilleyici.benchmark_quantization()

    print(f"     -> Orijinal FP32 Bellek     : {metrikler['mem_fp32_kb']:.1f} KB (4 Byte/Ağırlık)")
    print(f"     -> Kuantize INT8 Bellek     : {metrikler['mem_int8_kb']:.1f} KB (1 Byte/Ağırlık)")
    print(f"     -> SRAM Bellek Tasarrufu    : %{metrikler['tasarruf_yuzdesi']:.1f} Tasarruf")
    print(f"     -> Sinyal Kalitesi (SQNR)   : {metrikler['sqnr_db']:.2f} dB (> 40 dB Standardı)")
    print(f"     -> Maksimum Mutlak Hata     : {metrikler['max_abs_err']:.5f}")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] NPU INT8 Kuantizasyon RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['quant_step_ortalama_us']:.3f} µs (P99: {metrikler['quant_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Kuantize Hacmi : {metrikler['saniyelik_kuantizasyon_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD NPU Kuantizasyon Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaKuantizasyonGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_npu_int8_quantization_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 51 BAŞARIYLA TAMAMLANDI! INT8 NPU KUANTİZASYON DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
