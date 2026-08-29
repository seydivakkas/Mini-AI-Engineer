"""
Day 144: Tree of Thoughts (ToT): BFS ve DFS Arama ile Düşünce Ağacı Gezintisi ve Geri İzleme (Backtracking) Ana Akışı.
Game of 24 bulmacası üzerinde Tree of Thoughts simülasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.tot_arama_motoru import TreeOfThoughtsMotoru
from src.gorsellestirici import TreeOfThoughtsGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 144: Tree of Thoughts (ToT) - BFS, DFS & Backtracking (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    baslangic_sayilari = [4.0, 9.0, 10.0, 13.0]
    hedef = 24.0

    print(f"\n[1/3] Game of 24 Bulmacası Başlatılıyor: Sayılar = {baslangic_sayilari}, Hedef = {hedef}")

    motor = TreeOfThoughtsMotoru(hedef=hedef)

    # -------------------------------------------------------------
    # ADIM 1: BFS (Genişlik Öncelikli Arama / Beam Search)
    # -------------------------------------------------------------
    print("\n--- [A] Tree of Thoughts (BFS Arama) ---")
    bfs_sonuc = motor.bfs_ara(baslangic_sayilari, beam_genisligi=4)

    print(f"  • Çözüm Durumu             : {'BULUNDU' if bfs_sonuc['cozum_bulundu_mu'] else 'BULUNAMADI'}")
    print(f"  • Toplam Keşfedilen Düğüm  : {bfs_sonuc['toplam_kesfedilen_dugum']}")
    print(f"  • Toplam Budanan Düğüm     : {bfs_sonuc['toplam_budanan_dugum']}")
    print("  • Çözüm Yolu Adımları      :")
    for i, adim in enumerate(bfs_sonuc["adim_gecmisi"], start=1):
        print(f"      Adım {i}: {adim}")
    print(f"  • Nihai Sonuç              : {bfs_sonuc['nihai_sayi']}")

    # -------------------------------------------------------------
    # ADIM 2: DFS (Derinlik Öncelikli Arama ve Backtracking)
    # -------------------------------------------------------------
    print("\n--- [B] Tree of Thoughts (DFS + Backtracking Arama) ---")
    dfs_sonuc = motor.dfs_ara(baslangic_sayilari, maks_derinlik=4)

    print(f"  • Çözüm Durumu             : {'BULUNDU' if dfs_sonuc['cozum_bulundu_mu'] else 'BULUNAMADI'}")
    print(f"  • Toplam Keşfedilen Düğüm  : {dfs_sonuc['toplam_kesfedilen_dugum']}")
    print(f"  • Toplam Budanan Düğüm     : {dfs_sonuc['toplam_budanan_dugum']}")
    print(f"  • Geri İzleme (Backtrack)  : {dfs_sonuc['geri_izleme_sayisi']} kez çıkmazdan geri dönüldü")
    print("  • Çözüm Yolu Adımları      :")
    for i, adim in enumerate(dfs_sonuc["adim_gecmisi"], start=1):
        print(f"      Adım {i}: {adim}")
    print(f"  • Nihai Sonuç              : {dfs_sonuc['nihai_sayi']}")

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Tree of Thoughts Teşhis Panosu Üretiliyor...")
    gorsellestirici = TreeOfThoughtsGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "tree_of_thoughts_bfs_dfs_paneli.png")
    gorsellestirici.pano_olustur(
        bfs_sonucu=bfs_sonuc,
        dfs_sonucu=dfs_sonuc,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("✓ Day 144: TREE OF THOUGHTS (BFS, DFS & BACKTRACKING) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
