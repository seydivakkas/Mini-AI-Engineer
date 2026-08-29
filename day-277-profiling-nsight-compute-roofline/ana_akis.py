"""
Day 277 (FAZ 14): NVIDIA Nsight Compute & Roofline Modeli Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.roofline_profilleyici_motoru import NsightRooflineEngine
from src.roofline_raporlayici import RooflineRaporlayici
from src.gorsellestirici import RooflineGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 277 (FAZ 14): NVIDIA NSIGHT COMPUTE & HİYERARŞİK ROOFLINE MODELİ — DONANIM DARBOĞAZI ANALİZİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: NVIDIA H100 Donanım ve Roofline Sınırları Başlatılıyor
    # -------------------------------------------------------------
    print("\n[1/4] NVIDIA H100 SXM5 Donanım Özellikleri ve Roofline Sınırları Başlatılıyor...")
    print(f"  • Tepe Hesaplama Gücü (FP16 Tensor) : {NsightRooflineEngine.PEAK_TFLOPS_FP16:.1f} TFLOPS")
    print(f"  • HBM3 Bellek Bant Genişliği        : {NsightRooflineEngine.HBM3_BANDWIDTH_TB_S:.2f} TB/s (3350 GB/s)")
    print(f"  • L2 Cache Bant Genişliği           : {NsightRooflineEngine.L2_BANDWIDTH_TB_S:.1f} TB/s")
    print(f"  • Paylaşımlı Bellek (SRAM) Genişliği: {NsightRooflineEngine.SRAM_BANDWIDTH_TB_S:.1f} TB/s")
    print(f"  • Kritik Ridge Point Eşiği          : {NsightRooflineEngine.RIDGE_POINT:.2f} FLOP / Byte")

    # -------------------------------------------------------------
    # ADIM 2: LLM Çekirdeklerinin Nsight Aritmetik Yoğunluk Analizi
    # -------------------------------------------------------------
    print("\n[2/4] LLM Çekirdeklerinin Aritmetik Yoğunluk (FLOP/Byte) ve Darboğaz Analizi Yapılıyor...")
    rapor = RooflineRaporlayici.basarim_profili_cikar()

    for k in rapor["kernel_analizleri"]:
        print(f"  • [{k['kernel_name']}]")
        print(f"    - Aritmetik Yoğunluk : {k['arithmetic_intensity_flop_per_byte']:.1f} FLOP/Byte")
        print(f"    - Ulaşılan Başarım   : {k['achieved_tflops']:.1f} TFLOPS (%{k['hardware_efficiency_pct']:.1f} Verim)")
        print(f"    - Donanım Sınıfı     : {k['bottleneck_type']}")
        print(f"    - Önerilen İyileştirme: {k['primary_remedy']}")

    # -------------------------------------------------------------
    # ADIM 3: Nsight Warp Scheduler Stall ve Bellek Tıkanma Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Nsight Warp Scheduler Stall Dağılımı ve Memory-Bound Tıkanma Analizi...")
    stalls = rapor["warp_stalls"]
    for sebep, oran in zip(stalls["sebepler"], stalls["oranlar_yuzde"]):
        clean_sebep = sebep.replace("\n", " ")
        print(f"  • {clean_sebep:50s} : %{oran:.1f}")

    print(f"  • Kernel Füzyonu (SRAM Tiling) ile Hızlanma Kazancı : {rapor['fuzed_speedup']:.1f}x Kat Daha Hızlı")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Nsight Roofline Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "nsight_roofline_paneli.png")

    RooflineGorsellestirici.teshis_paneli_olustur(
        profil_raporu=rapor,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Nsight Roofline Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 277 (FAZ 14): NSIGHT COMPUTE & ROOFLINE MODELİ MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
