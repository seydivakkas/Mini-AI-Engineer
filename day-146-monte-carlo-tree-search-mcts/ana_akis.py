"""
Day 146: Monte Carlo Tree Search (MCTS) Destekli LLM Düşünce Planlaması Ana Akışı.
UCT (Upper Confidence bounds for Trees) ile Game of 24 ve çok adımlı muhakeme planlaması.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.mcts_planlayici import MCTSDusuncePlanlayici
from src.gorsellestirici import MCTSGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 146: Monte Carlo Tree Search (MCTS + UCT) for LLM Reasoning (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    baslangic_sayilari = [4.0, 9.0, 10.0, 13.0]
    hedef = 24.0

    print(f"\n[1/3] Game of 24 Bulmacası Başlatılıyor: Sayılar = {baslangic_sayilari}, Hedef = {hedef}")

    # -------------------------------------------------------------
    # 1. MCTS PLANLAMA DÖNGÜSÜ (Selection, Expansion, Rollout, Backprop)
    # -------------------------------------------------------------
    print("\n[2/3] 4 Aşamalı MCTS Arama ve Planlama Motoru Çalıştırılıyor...")
    planlayici = MCTSDusuncePlanlayici(hedef=hedef, c_kesif=0.8, simulasyon_sayisi=250)
    sonuc = planlayici.planla(baslangic_sayilari)

    print("\n" + "-" * 95)
    print(f"  • Arama Algoritması        : {sonuc['algoritma']}")
    print(f"  • Çözüm Bulundu mu?        : {'EVET [OK]' if sonuc['cozum_bulundu_mu'] else 'HAYIR'}")
    print(f"  • Toplam Simülasyon Sayısı : {sonuc['toplam_simulasyon']}")
    print(f"  • Keşfedilen Düğüm Sayısı  : {sonuc['toplam_kesfedilen_dugum']}")
    print(f"  • Kök Ziyaret Sayısı       : {sonuc['kok_ziyaret_sayisi']}")
    print(f"  • Kök Ortalama Değeri Q    : {sonuc['kok_ortalama_q']}")
    print("  • MCTS Tarafından Bulunan Yol Adımları:")
    for i, adim in enumerate(sonuc["en_iyi_yol_adimlari"], start=1):
        print(f"      Adım {i}: {adim}")
    print(f"  • Nihai Sonuç              : {sonuc['nihai_sayi']} (HEDEF: {hedef})")
    print("-" * 95)

    # -------------------------------------------------------------
    # 2. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli MCTS Teşhis Panosu Üretiliyor...")
    gorsellestirici = MCTSGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "monte_carlo_tree_search_mcts_paneli.png")
    gorsellestirici.pano_olustur(mcts_sonucu=sonuc, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 146: MONTE CARLO TREE SEARCH (MCTS + UCT) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
