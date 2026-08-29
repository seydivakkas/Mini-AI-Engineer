"""
Day 292 (FAZ 15): Otonom Bilimsel Keşif ve AI Scientist Ana Akış Betiği.
Sakana AI The AI Scientist, Otomatik Hipotez, Deney Simülasyonu, LaTeX ve Otonom Hakemlik.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.ai_scientist_motoru import (
    ResearchIdea,
    LiteratureNoveltyChecker,
    ExperimentRunner,
    LaTeXPaperGenerator,
    AutonomousPeerReviewer,
)
from src.ai_scientist_profilleyici import AIScientistProfilleyici
from src.gorsellestirici import AIScientistGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 292 (FAZ 15): OTONOM BİLİMSEL KEŞİF VE AI SCIENTIST — AUTONOMOUS DISCOVERY")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Otonom Hipotez ve Literatür Özgünlük Taraması
    # -------------------------------------------------------------
    print("\n[1/4] Otonom Bilimsel Fikir Üretiliyor ve arXiv Literatür Özgünlüğü Doğrulanıyor...")
    idea = ResearchIdea(
        title="Adaptive Sparse Attention via Dynamic Entropy Gating",
        hypothesis="Filtering low-entropy attention heads saves 60% compute with 0% accuracy drop",
        novelty_score=0.94,
    )
    novelty = LiteratureNoveltyChecker.check_novelty(idea)

    print(f"  • Makale Başlığı                     : \"{idea.title}\"")
    print(f"  • Temel Hipotez                      : \"{idea.hypothesis}\"")
    print(f"  • Literatür Özgünlük Skoru           : %{novelty * 100:.1f} (Özgün Fikir Onaylandı)")

    # -------------------------------------------------------------
    # ADIM 2: Otonom Deney Koşturma ve LaTeX Makale Derleme
    # -------------------------------------------------------------
    print("\n[2/4] Otonom Deney Koşturuluyor ve NeurIPS Standartlarında LaTeX Makalesi Derleniyor...")
    exp_res = ExperimentRunner.run_experiment(idea)
    paper_tex = LaTeXPaperGenerator.generate_latex(idea, exp_res)

    print(f"  • Doğruluk Artışı (Baseline -> Öneri): %{exp_res['baseline_acc']:.1f} -> %{exp_res['proposed_acc']:.1f}")
    print(f"  • FLOPs Hesaplama Tasarrufu          : %{exp_res['compute_savings']:.1f}")
    print(f"  • Derlenen LaTeX Makale Boyutu       : {len(paper_tex)} Karakter")

    # -------------------------------------------------------------
    # ADIM 3: Otonom Hakemlik ve Kıyaslama Raporu
    # -------------------------------------------------------------
    print("\n[3/4] NeurIPS Standartlarında Otonom Hakem Değerlendirmesi ve Kıyaslama Raporu...")
    review = AutonomousPeerReviewer.review_paper(paper_tex)
    profil = AIScientistProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • Hakem Sağlamlık (Soundness) Skoru  : {review['soundness']:.1f} / 10.0")
    print(f"  • Hakem Özgün Katkı (Contribution)   : {review['contribution']:.1f} / 10.0")
    print(f"  • Nihai Hakem Kararı                 : {review['decision']}")
    print(f"  • Araştırma Döngüsü Hızlanması       : 180 Gün -> 15 Dakika ({profil['hizlanma_orani']:,.0f}x Hızlı)")
    print(f"  • Araştırma Maliyeti                 : $50,000 -> $5.0 (10,000x Tasarruf)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli AI Scientist Otonom Keşif Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "ai_scientist_autonomous_discovery_paneli.png")

    AIScientistGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ AI Scientist Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 292 (FAZ 15): OTONOM BİLİMSEL KEŞİF (THE AI SCIENTIST) MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
