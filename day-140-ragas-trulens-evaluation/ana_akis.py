"""
Day 140: Ragas & TruLens RAG Değerlendirmesi ve FAZ 7 BÜYÜK FİNALİ Ana Akışı.
RAG Triad (Faithfulness, Answer Relevance, Context Recall) ile uçtan uca mimari benchmark testi.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.ragas_trulens_degerlendirici import RagasTruLensDegerlendirici
from src.gorsellestirici import RAGEvaluationGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 140: Ragas & TruLens RAG Evaluation Framework - FAZ 7 BÜYÜK FİNALİ")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    degerlendirici = RagasTruLensDegerlendirici()

    # -------------------------------------------------------------
    # ADIM 1: Tekil RAG Yanıtı İçin RAG Triad Değerlendirmesi
    # -------------------------------------------------------------
    print("\n[1/3] Örnek RAG Yanıtı İçin Ragas & TruLens RAG Triad Hesaplanıyor...")
    soru = "Vision Transformer dikkat mekanizmasını GPU üzerinde hızlandıran optimizasyon mimarisi nasıl çalışır?"
    baglam = [
        "Vision Transformer modelleri görüntü yamalarını Self-Attention dikkat mekanizması ile işler.",
        "FlashAttention algoritması GPU SRAM bellek bant genişliğini optimize ederek dikkat matrisini hızlandırır.",
    ]
    yanit = (
        "Vision Transformer modelleri Self-Attention dikkat mekanizması kullanır. "
        "FlashAttention algoritması ise GPU SRAM bellek optimizasyonu sağlayarak dikkat hesaplamasını hızlandırır."
    )
    referanslar = [
        "Vision Transformer modelleri Self-Attention dikkat mekanizması kullanır.",
        "FlashAttention GPU SRAM belleğini optimize ederek dikkat hesaplamasını hızlandırır.",
    ]

    tekil_sonuc = degerlendirici.tekil_degerlendir(
        soru=soru,
        yanit=yanit,
        getirilen_baglam=baglam,
        referans_dogrulari=referanslar,
    )

    print(f"  • Sadakat (Faithfulness / Groundedness)   : %{tekil_sonuc['faithfulness']:.1f}")
    print(f"  • Soru Uygunluğu (Answer Relevance)       : %{tekil_sonuc['answer_relevance']:.1f}")
    print(f"  • Bağlam Kapsama (Context Recall)          : %{tekil_sonuc['context_recall']:.1f}")
    print(f"  • Bağlam Hassasiyeti (Context Precision)  : %{tekil_sonuc['context_precision']:.1f}")
    print(f"  • HARMONİK RAG TRIAD SKORU                : %{tekil_sonuc['rag_triad_score']:.1f}")
    print(f"  • Halüsinasyon Oranı                      : %{tekil_sonuc['halusinasyon_orani']:.1f}")

    # -------------------------------------------------------------
    # ADIM 2: Faz 7 Boyunca Geliştirilen 4 RAG Mimarisinin Benchmarkı
    # -------------------------------------------------------------
    print("\n[2/3] FAZ 7'nin 4 Büyük RAG Mimarisi Karşılaştırılıyor (Comparative Benchmark)...")
    benchmark = degerlendirici.faz7_mimarilerini_karsilastir()

    print(f"\n{'MİMARİ':<38} | {'SADAKAT':<9} | {'UYGUNLUK':<9} | {'RECALL':<8} | {'PRECISION':<10} | {'RAG TRIAD'}")
    print("-" * 95)
    for mimari_idx, mimari_adi in enumerate(benchmark["mimariler"]):
        key = ["naive_rag", "semantic_hyde", "compression_rerank", "hybrid_graphrag"][mimari_idx]
        skorlar = benchmark["sonuclar"][key]
        print(
            f"{mimari_adi:<38} | "
            f"%{skorlar[0]:<8.1f} | "
            f"%{skorlar[1]:<8.1f} | "
            f"%{skorlar[2]:<7.1f} | "
            f"%{skorlar[3]:<9.1f} | "
            f"%{skorlar[4]:.1f}"
        )

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli FAZ 7 BÜYÜK FİNALİ Teşhis Panosu Çiziliyor...")
    gorsellestirici = RAGEvaluationGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "ragas_trulens_evaluation_paneli.png")
    gorsellestirici.pano_olustur(
        benchmark_sonuclari=benchmark,
        tekil_degerlendirme=tekil_sonuc,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("🏆 [FAZ 7 BÜYÜK FİNALİ] FAZ 7: OTONOM AI AJANLARI VE ADVANCED GRAPHRAG %100 BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
