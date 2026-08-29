"""
Day 233: Dinamik Araç Geri Getirme (Tool-RAG) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.tool_rag_motoru import (
    ToolDefinition,
    ToolRegistry,
    SemanticToolRetriever,
    DynamicToolAgent,
)
from src.rag_profilleyici import RAGProfilleyici
from src.gorsellestirici import RAGGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 233 (FAZ 12): DİNAMİK ARAÇ GERİ GETİRME MOTORU (TOOL-RAG) - SEMANTİK ŞEMA ENJEKSİYONU")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Kurumsal Araç Havuzu ve Retriever Kurulumu
    # -------------------------------------------------------------
    print("\n[1/4] Kurumsal Araç Havuzu ve Semantik İndeks Başlatılıyor...")
    registry = ToolRegistry()
    retriever = SemanticToolRetriever(registry)
    agent = DynamicToolAgent(registry, retriever)
    print(f"  ✓ Toplam Kayıtlı Araç Sayısı: {len(registry.araclar)} (Finans, DevOps, DB, Matematik, İletişim)")

    # -------------------------------------------------------------
    # ADIM 2: Finans ve Analiz Sorgusu (Tool-RAG Seçimi)
    # -------------------------------------------------------------
    sorgu_finans = "Tesla hisse senedi fiyatını getir ve 14 günlük RSI hesapla"
    print(f"\n[2/4] Kullanıcı İstemi: '{sorgu_finans}'")
    plan_finans = agent.planla_ve_sec(sorgu_finans, top_k=3)

    print("  • Getirilen Top-3 Araç:")
    for arac, skor in plan_finans["top_k_araclar"]:
        print(f"    - [{arac}] (Benzerlik Skoru: {skor})")
    print(f"  • Seçilen Birincil Araç : {plan_finans['secilen_birincil_arac']}")
    print(f"  • Token Tasarruf Oranı  : %{plan_finans['tasarruf_yuzdesi']}")

    # -------------------------------------------------------------
    # ADIM 3: Veritabanı ve Performans Sorgusu
    # -------------------------------------------------------------
    sorgu_db = "PostgreSQL veritabanında siparişler tablosunu sorgula ve indeksleri hızlandır"
    print(f"\n[3/4] Kullanıcı İstemi: '{sorgu_db}'")
    plan_db = agent.planla_ve_sec(sorgu_db, top_k=3)

    print("  • Getirilen Top-3 Araç:")
    for arac, skor in plan_db["top_k_araclar"]:
        print(f"    - [{arac}] (Benzerlik Skoru: {skor})")
    print(f"  • Seçilen Birincil Araç : {plan_db['secilen_birincil_arac']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Tool-RAG Teşhis Panosu Oluşturuluyor...")
    profil_raporu = RAGProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "tool_rag_paneli.png")

    RAGGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Tool-RAG Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 233 (FAZ 12): DİNAMİK ARAÇ GERİ GETİRME MOTORU (TOOL-RAG) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
