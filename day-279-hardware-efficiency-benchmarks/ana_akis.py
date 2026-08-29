"""
Day 279 (FAZ 14): Donanım Verimliliği Başarım Paketi Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.mfu_benchmark_motoru import MFUBenchmarkEngine
from src.mfu_profilleyici import MFUProfilleyici
from src.gorsellestirici import MFUGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 279 (FAZ 14): DONANIM VERİMLİLİĞİ BAŞARIM PAKETİ — MFU, HFUS VE MBU KIYASLAMA SÜİTİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: PaLM Standartlarında MFU Ölçüm Motoru Başlatılıyor
    # -------------------------------------------------------------
    print("\n[1/4] PaLM / Megatron-LM Uyumlu Donanım Verimliliği (MFU/HFUS/MBU) Motoru Başlatılıyor...")
    print(f"  • Metrik Standartları                : Chowdhery et al. (PaLM) & Kaplan et al.")
    print(f"  • Model FLOPs Utilization (MFU)      : Teorik Minimum İşlemin Donanım Tavanına Oranı")
    print(f"  • Hardware FLOPs Utilization (HFUS)  : Gerçekte Koşturulan İşlemlerin Donanım Tavanına Oranı")
    print(f"  • Memory Bandwidth Util (MBU)        : Erişilen HBM Bellek Veriyolu Doyumu")

    # -------------------------------------------------------------
    # ADIM 2: LLaMA-70B Teorik FLOP ve Sistem Kıyaslaması
    # -------------------------------------------------------------
    print("\n[2/4] LLaMA-70B Mimarisi İçin Teorik FLOPs/Token ve Sistem Kıyaslaması Hesaplanıyor...")
    bench_res = MFUBenchmarkEngine.run_llama_70b_benchmark_comparison()

    print(f"  • LLaMA-70B Teorik FLOP / Token      : {bench_res['flops_per_token'] / 1e9:.2f} GFLOPs / token")
    for isim, veri in bench_res["sistem_sonuclari"].items():
        m = veri["metrikler"]
        print(f"  • [{isim}]")
        print(f"    - Throughput : {veri['throughput_tok_s']:.1f} tok/s")
        print(f"    - MFU        : %{m['mfu_yuzde']:.1f}")
        print(f"    - HFUS       : %{m['hfus_yuzde']:.1f}")
        print(f"    - MBU        : %{m['mbu_yuzde']:.1f}")

    # -------------------------------------------------------------
    # ADIM 3: Donanım MFU, MBU ve Hızlanma Kazancı Raporu
    # -------------------------------------------------------------
    print("\n[3/4] FAZ-14 Custom Fused Suite vs Standart PyTorch Başarım Raporu...")
    profil = MFUProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • Model FLOPs Util (MFU)             : %{kars['mfu_yuzde']['Naive_PyTorch_Baseline']:.1f} -> %{kars['mfu_yuzde']['FAZ14_Fused_Custom_Suite']:.1f} (%67.8 SOTA)")
    print(f"  • HBM3 Bellek Doyumu (MBU)           : %{kars['mbu_yuzde']['Naive_PyTorch_Baseline']:.1f} -> %{kars['mbu_yuzde']['FAZ14_Fused_Custom_Suite']:.1f}")
    print(f"  • LLaMA-70B Üretim Hızı              : {kars['llama_70b_throughput_tok_s']['Naive_PyTorch_Baseline']:.1f} tok/s -> {kars['llama_70b_throughput_tok_s']['FAZ14_Fused_Custom_Suite']:.1f} tok/s ({profil['hizlanma_orani']:.1f}x Hızlanma)")
    print(f"  • 405B Model Skalalaması             : %{profil['skala']['faz14_custom_mfu'][-1]:.1f} MFU (Maksimum Donanım Doyumu)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Donanım Verimliliği Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "donanim_verimliligi_mfu_paneli.png")

    MFUGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Donanım Verimliliği Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 279 (FAZ 14): DONANIM VERİMLİLİĞİ BAŞARIM PAKETİ (MFU/HFUS) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
