"""
Day 201: 201 GÜNLÜK BÜYÜK FİNAL - Mini-Omni Reasoner v1.0 (Multimodal + CoT + MoE + Triton) Ana Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.mini_omni_model import MiniOmniReasonerModel
from src.omni_reasoning_motoru import ChainOfThoughtReasoner
from src.omni_benchmark_profilleyici import OmniBenchmarkProfilleyici
from src.gorsellestirici import OmniGrandFinaleGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 201 (FAZ 10): 201 GÜNLÜK BÜYÜK FİNAL - MINI-OMNI REASONER v1.0 (MASTER AI & MLOPS GRAND FINALE)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Birleşik Mini-Omni Reasoner Modelinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Birleşik Mini-Omni Reasoner v1.0 Mimarisi Başlatılıyor...")
    model = MiniOmniReasonerModel(vocab_size=1000, embed_dim=128, num_layers=2)
    toplam_parametre = sum(p.numel() for p in model.parameters())

    print(f"  • Model Adı                   : Mini-Omni Reasoner v1.0")
    print(f"  • Toplam Parametre Sayısı     : {toplam_parametre:,} Parametre")
    print(f"  • Çok Modlu Girdi Katmanı     : Görüntü (64-dim) + Ses (32-dim) + Metin Token")
    print(f"  • Dikkat Mekanizması          : Özel Triton FlashAttention-2 & Fused RMSNorm")
    print(f"  • İleri Besleme Ağı           : 4 Uzmanlı Top-2 Seyrek MoE (Görsel, Matematik, Mantık, Dil)")
    print("  ✓ Model Mimarisi Başarıyla Oluşturuldu!")

    # -------------------------------------------------------------
    # ADIM 2: Çok Modlu CoT Akıl Yürütme ve Test-Time Search
    # -------------------------------------------------------------
    print("\n[2/4] Çok Modlu Problem Üzerinde Test-Time CoT Akıl Yürütme Yürütülüyor...")
    reasoner = ChainOfThoughtReasoner(model=model)
    problem = "Görseldeki geometrik açıyı ve sesli ipucunu birleştirerek hipotenüs uzunluğunu hesapla."

    cozum = reasoner.solve_multimodal_problem(query=problem, has_vision=True, has_audio=True)

    print(f"\n  • Girdi Problemi              : {cozum['query']}")
    print(f"  • Modaliteler                 : Görüntü (Açık), Ses (Açık), Metin (Açık)")
    print(f"  • TTFT (İlk Token Gecikmesi)  : {cozum['ttft_ms']:.2f} ms")
    print(f"  • TPOT (Token Başı Süre)      : {cozum['tpot_ms']:.2f} ms / token")
    print(f"  • Toplam Çıkarım Süresi       : {cozum['toplam_sure_ms']:.2f} ms")
    print(f"  • Çıkarım Akışı:\n{cozum['yanit']}\n")

    # -------------------------------------------------------------
    # ADIM 3: 4 Amiral Gemisi Benchmark Paketi Değerlendirmesi
    # -------------------------------------------------------------
    print("[3/4] 4 Amiral Gemisi Benchmark Paketinde Mini-Omni Reasoner Değerlendiriliyor...")
    benchmark_raporu = OmniBenchmarkProfilleyici.calistir_buyuk_final_benchmarki()

    print("-" * 115)
    print(f"{'Benchmark Paketi':<22} | {'Kategori':<30} | {'Doğruluk Skoru':<16} | {'TTFT Gecikmesi':<16} | {'Hız'}")
    print("-" * 115)
    for g in benchmark_raporu["gorev_sonuclari"]:
        print(
            f"{g['benchmark_id']:<22} | "
            f"{g['kategori']:<30} | "
            f"%{g['dogruluk']:>13.1f}   | "
            f"{g['ttft_ms']:>13.2f} ms | "
            f"{1000.0/g['tpot_ms']:>6.1f} tok/s"
        )
    print("-" * 115)
    print(f"  🏆 Genel Ortalama Doğruluk Skoru : %{benchmark_raporu['genel_ortalama_dogruluk']:.1f} (SOTA)")
    print(f"  ⚡ Triton FlashAttention-2 Hız   : {benchmark_raporu['triton_flashattention_hizlanma']:.1f}x Hızlanma")
    print(f"  🧠 Top-2 MoE Hesaplama Tasarrufu : %{benchmark_raporu['moe_hesaplama_tasarrufu_yuzde']:.0f} FLOPs Tasarrufu")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Büyük Final Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Büyük Final Şampiyonluk Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "mini_omni_grand_finale_paneli.png")

    OmniGrandFinaleGorsellestirici.teshis_paneli_olustur(
        benchmark_raporu=benchmark_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Büyük Final Şampiyonluk Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("🏆 DAY 201: 201 GÜNLÜK MASTER ROADMAP BÜYÜK FİNALİ EN ÜST SEVİYEDE BAŞARIYLA TAMAMLANDI! 🏆")
    print("=" * 115)


if __name__ == "__main__":
    main()
