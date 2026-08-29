"""
Day 288 (FAZ 15): LLM Akıl Yürütme ve Test-Zamanı Hesaplama (MCTS & PRM) Ana Akış Betiği.
Tree of Thoughts (ToT), Process Reward Model (PRM) ve OpenAI o1/o3 Düşünme Mimarisi.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.mcts_reasoning_motoru import ThoughtNode, ProcessRewardModel, MCTSReasoningEngine
from src.mcts_reasoning_profilleyici import MCTSReasoningProfilleyici
from src.gorsellestirici import MCTSReasoningGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 288 (FAZ 15): BÜYÜK DİL MODELLERİNDE AKIL YÜRÜTME — MCTS & PRM TEST-TIME COMPUTE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Düşünce Ağacı ve PRM'in Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Düşünce Ağacı (ToT) ve Süreç Ödül Modeli (PRM) Başlatılıyor...")
    problem = "2x + 6 = 14 denklemini çöz ve x değerini bul."
    print(f"  • Çözülecek Problem                  : {problem}")
    print(f"  • Arama Mimarisi                     : Tree of Thoughts (ToT) + MCTS")
    print(f"  • Adım Doğrulama Modeli              : Process Reward Model (PRM r(s_t) ∈ [0, 1])")

    # -------------------------------------------------------------
    # ADIM 2: Test-Zamanı MCTS Akıl Yürütme Arama Döngüsü
    # -------------------------------------------------------------
    print("\n[2/4] Test-Zamanı MCTS Arama Döngüsü Çalıştırılıyor (40 Simülasyon, UCB1)...")
    res = MCTSReasoningEngine.run_mcts_reasoning(problem, num_simulations=40)

    print(f"  • Gerçekleştirilen MCTS Simülasyonu  : {res['num_simulations']} Adım")
    print(f"  • Genişletilen Düğüm Sayısı          : {res['expanded_nodes']} Düşünce Düğümü")
    print(f"  • Budanan Mantıksal Hatalı Dal       : {res['pruned_branches']} Dal (Halüsinasyon Engellendi)")
    print(f"  • Nihai Doğru Çözüm Bulundu mu       : {'✓ EVET (x = 4)' if res['final_solution_found'] else '✗ HAYIR'}")
    print("  • Keşfedilen En İyi Düşünce Yolu    :")
    for idx, step in enumerate(res["best_path"], 1):
        print(f"      Adım {idx}: {step}")

    # -------------------------------------------------------------
    # ADIM 3: Akıl Yürütme Yöntemleri Kıyaslama Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Açgözlü Çıkarım vs CoT vs MCTS Test-Time Compute Kıyaslama Raporu...")
    profil = MCTSReasoningProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • 1. Direct Greedy Başarımı          : %{kars['matematik_mantik_basarisi_yuzde']['1. Direct Greedy']:.1f} (Halüsinasyon: %{kars['mantiksal_halusinasyon_orani']['1. Direct Greedy']:.1f})")
    print(f"  • 2. Standard CoT Başarımı           : %{kars['matematik_mantik_basarisi_yuzde']['2. Standard CoT']:.1f} (Halüsinasyon: %{kars['mantiksal_halusinasyon_orani']['2. Standard CoT']:.1f})")
    print(f"  • 3. MCTS + PRM Test-Time Başarımı   : %{kars['matematik_mantik_basarisi_yuzde']['3. MCTS + PRM Test-Time']:.1f} (Halüsinasyon: %{kars['mantiksal_halusinasyon_orani']['3. MCTS + PRM Test-Time']:.1f})")
    print(f"  • Doğruluk Kazancı                   : +%{kars['matematik_mantik_basarisi_yuzde']['3. MCTS + PRM Test-Time'] - kars['matematik_mantik_basarisi_yuzde']['2. Standard CoT']:.1f} (15x Daha Az Halüsinasyon)")
    print(f"  • Otonom Hata Düzeltme (Backtracking): %{kars['otonom_hata_duzeltme_yuzde']['3. MCTS + PRM Test-Time']:.1f}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli LLM Akıl Yürütme (MCTS & PRM) Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "mcts_reasoning_prm_paneli.png")

    MCTSReasoningGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ MCTS Akıl Yürütme Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 288 (FAZ 15): BÜYÜK DİL MODELLERİNDE AKIL YÜRÜTME (MCTS & PRM) MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
