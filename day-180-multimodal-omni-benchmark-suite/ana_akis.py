"""
Day 180 (FAZ 9 BÜYÜK FİNALİ): Multimodal Omni Benchmark Suite Ana Akış Motoru.
MME, MMBench (CircularEval), MathVista ve POPE ile 360° Çok Modlu Model Doğrulama ve Liderlik Tablosu.
"""

import sys
import os

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Modül yolunu ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mme_degerlendirici import MMEDegerlendirici
from src.mmbench_degerlendirici import MMBenchDegerlendirici
from src.mathvista_degerlendirici import MathVistaDegerlendirici
from src.omni_karsilastirici import OmniBenchmarkMerkezi
from src.gorsellestirici import MultimodalBenchmarkGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 180 (FAZ 9 BÜYÜK FİNALİ): MULTIMODAL OMNI BENCHMARK SUITE (MME + MMBench + MathVista)")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # 1. FAZ 9 Capstone Modeli (Mini-Omni-v1) Detaylı Benchmark Değerlendirmesi
    capstone_model = "Mini-Omni-v1 (FAZ 9 Capstone 2026)"
    print(f"\n[1/4] '{capstone_model}' İçin 4 Ana Multimodal Benchmark Çalıştırılıyor...")

    mme_res = MMEDegerlendirici.ornek_model_mme_raporu(capstone_model)
    print(f"  • [MME] Toplam Skor : {mme_res['toplam_mme_skoru']} / 2800 ({mme_res['genel_basari_yuzdesi']}%)")
    print(f"    - Perception (Algı) : {mme_res['perception_skoru']} / 2000 ({mme_res['perception_basari_yuzdesi']}%)")
    print(f"    - Cognition (Biliş) : {mme_res['cognition_skoru']} / 800 ({mme_res['cognition_basari_yuzdesi']}%)")

    mmbench_res = MMBenchDegerlendirici.ornek_model_mmbench_raporu(capstone_model)
    print(f"  • [MMBench] CircularEval : %{mmbench_res['genel_circular_acc']:.2f} (Vanilla: %{mmbench_res['genel_vanilla_acc']:.2f})")
    print(f"    - Tutarlılık Oranı    : %{mmbench_res['genel_tutarlilik_orani']:.2f}")
    print(f"    - Önyargı Farkı       : %{mmbench_res['pozisyon_onyargi_farki']:.2f}")

    math_res = MathVistaDegerlendirici.ornek_model_mathvista_raporu(capstone_model)
    print(f"  • [MathVista] Doğruluk  : %{math_res['genel_mathvista_skoru']:.2f} ({math_res['toplam_dogru']}/{math_res['toplam_soru']} soru)")

    # 2. Çok Modlu Model Karşılaştırma Merkezi & Liderlik Tablosu
    print(f"\n[2/4] SOTA Çok Modlu Modeller İçin Bütünleşik Liderlik Tablosu Derleniyor...")
    benchmark_merkezi = OmniBenchmarkMerkezi()
    liderlik_raporu = benchmark_merkezi.tum_modelleri_karsilastir()

    print("\n" + "-" * 105)
    print(f"{'Sıra':<5} | {'Model Adı':<34} | {'Omni-Score':<11} | {'MME (/2800)':<12} | {'MMBench (%)':<12} | {'MathVista (%)':<13} | {'POPE (F1)'}")
    print("-" * 105)
    for item in liderlik_raporu["liderlik_tablosu"]:
        print(
            f"#{item['siralama']:<4} | {item['model_adi']:<34} | {item['omni_score']:>9.2f}% | "
            f"{item['mme']['toplam_puan']:>10.1f} | {item['mmbench']['circular_acc']:>10.2f}% | "
            f"{item['mathvista']['dogruluk']:>11.2f}% | {item['pope']['f1_skoru']:>7.1f}%"
        )
    print("-" * 105)

    # 3. 6 Panelli Görselleştirme Panosu Üretimi
    print(f"\n[3/4] 6 Panelli FAZ 9 BÜYÜK FİNALİ Teşhis Panosu Oluşturuluyor...")
    gorsellestirici = MultimodalBenchmarkGorsellestirici(dpi=300)
    cikti_resmi = os.path.join(cikis_dizini, "multimodal_omni_benchmark_paneli.png")
    gorsellestirici.pano_olustur(liderlik_raporu, kayit_yolu=cikti_resmi)

    # 4. FAZ 9 Tamamlanma Sertifikasyon Özeti
    print(f"\n[4/4] FAZ 9 (Çok Modlu Temel Modeller) Başarıyla Mühürlendi:")
    print("  ✓ Gün 161 - Gün 180 arasındaki 20 çok modlu modülün tamamı %100 eksiksiz tamamlandı.")
    print("  ✓ VLM, OCR-Free, GUI Ajanları, Video LLMs, Whisper, S2S, Difüzyon, DiT, NeRF, 3DGS ve Omni-Eval.")
    print("  ✓ Sıradaki Büyük Faz: FAZ 10: Ultra-MLOps, Dağıtık Eğitim, Triton GPU Kernel ve BÜYÜK FİNAL 201.")

    print("\n" + "=" * 110)
    print("✓ Day 180: MULTIMODAL OMNI BENCHMARK SUITE BAŞARIYLA TAMAMLANDI! (FAZ 9 BÜYÜK FİNALİ MÜHÜRLENDİ)")
    print("=" * 110)


if __name__ == "__main__":
    main()
