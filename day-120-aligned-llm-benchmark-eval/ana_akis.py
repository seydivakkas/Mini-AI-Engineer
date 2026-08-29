"""
Day 120: FAZ 6 BÜYÜK FİNALİ - Aligned LLM Benchmark & Chatbot Arena Şampiyonası Ana Akışı.
Tüm Faz 6 modellerinin (SFT, DPO, KTO, ORPO, SimPO, GRPO, Merged, Distilled) MT-Bench ve Elo turnuvası.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.faz6_modeller_benchmark import Faz6BenchmarkArenasi
from src.gorsellestirici import ArenaGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 120: PHASE 6 CAPSTONE GRAND FINALE - Aligned LLM Benchmark & Chatbot Arena Championship")
    print("=" * 105)

    # -------------------------------------------------------------
    # ADIM 1: Faz 6 Benchmark Arenası Başlatma
    # -------------------------------------------------------------
    print("\n[1/3] Faz 6 Modeller Şampiyonası ve MT-Bench Hakem Motoru Başlatılıyor...")
    arena = Faz6BenchmarkArenasi(seed=42)

    print(f"  * Turnuvaya Katılan Model Sayısı : {len(arena.MODELLER)} Model")
    print(f"  * Değerlendirilen Kategori Sayısı : {len(arena.hakem.KATEGORILER)} MT-Bench Kategorisi")
    print("  * Hakemlik ve Yanlılık Denetimi   : Swap Testli Pozisyon Yanlılığı (Position Bias) Telafisi")
    print("  * Sıralama Algoritması           : Bradley-Terry Dinamik Elo Derecelendirmesi (K=32)")

    # -------------------------------------------------------------
    # ADIM 2: Büyük Şampiyona Turnuvasının Koşturulması
    # -------------------------------------------------------------
    print("\n[2/3] Round-Robin Turnuva Karşılaşmaları ve MT-Bench Puanlaması Yürütülüyor...")
    rapor = arena.turnuvayi_kostur(mac_tur_sayisi=15)

    print("\n" + "=" * 105)
    print("                       🏆 CHATBOT ARENA DİNAMİK ELO LİDERLİK TABLOSU 🏆                       ")
    print("=" * 105)
    print(f"{'SIRA':<6} | {'MODEL ADI':<38} | {'ELO DERECESİ':<14} | {'TOPLAM MAÇ':<12} | {'KAZANMA ORANI (%)':<18}")
    print("-" * 105)
    for row in rapor["liderlik_tablosu"]:
        madalya = "🥇 " if row["sira"] == 1 else ("🥈 " if row["sira"] == 2 else ("🥉 " if row["sira"] == 3 else f"{row['sira']:<2} "))
        print(f"{madalya:<6} | {row['model_adi']:<38} | {row['elo']:>10.1f} Elo | {row['toplam_mac']:>10d} | %{row['kazanma_orani']:>14.1f}")
    print("-" * 105)

    print(f"\n[-] HAKEMLİK VE YANLILIK ANALİZİ:")
    print(f"  * Toplam Yapılan Karşılaşma       : {rapor['toplam_mac_sayisi']} Maç")
    print(f"  * Pozisyon Yanlılığı Tespit Oranı : %{rapor['pozisyon_yanliligi_tespit_orani']:.2f} (Swap testi ile başarıyla nötrlendi)")
    print(f"  * FAZ 6 ŞAMPİYONU                 : 👑 {rapor['sampiyon_model']} ({rapor['sampiyon_elo']:.1f} Elo)")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Faz 6 Capstone Teşhis Panosu Çiziliyor...")
    gorsellestirici = ArenaGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "faz6_capstone_benchmark_paneli.png",
    )
    gorsellestirici.pano_olustur(
        rapor,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 120: FAZ 6 BÜYÜK FİNALİ BAŞARIYLA TAMAMLANDI! TÜM MODELLER DERECELENDİRİLDİ.")
    print("=" * 105)


if __name__ == "__main__":
    main()
