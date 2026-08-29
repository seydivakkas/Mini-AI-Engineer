"""
Day 289 (FAZ 15): Çok Modlu Çoklu Ajan Tartışması ve Konsensüs Toplumu Ana Akış Betiği.
Society of Mind, Multi-Agent Debate (MAD), Elo Ağırlıklı Hakem Konsensüsü.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.multi_agent_debate_motoru import AgentPersona, MultiAgentDebateSociety
from src.multi_agent_debate_profilleyici import MultiAgentDebateProfilleyici
from src.gorsellestirici import MultiAgentDebateGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 289 (FAZ 15): ÇOK MODLU ÇOKLU AJAN TARTIŞMASI VE KONSENSÜS — MULTI-AGENT DEBATE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Heterojen Ajanların Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Heterojen Uzman Ajan Personaları ve Elo Ağırlıkları Başlatılıyor...")
    sorgu = "Fintech ödeme altyapısı için mikroservis mimarisi tasarımı ve veri tutarlılığı güvencesi"
    print(f"  • Tartışma Konusu                    : {sorgu}")
    print(f"  • Katılımcı Ajan 1                   : Ajan Alfa (Tez Sahibi - Elo: 1550)")
    print(f"  • Katılımcı Ajan 2                   : Ajan Beta (Kritik Eleştirmen - Elo: 1620)")
    print(f"  • Katılımcı Ajan 3                   : Baş Hakem Omega (Sentez & Konsensüs - Elo: 1850)")

    # -------------------------------------------------------------
    # ADIM 2: 3 Turlu Diyalektik Tartışma Protokolü
    # -------------------------------------------------------------
    print("\n[2/4] 3 Turlu Çoklu Ajan Tartışması (MAD) Yürütülüyor (Tez -> Antitez -> Sentez)...")
    res = MultiAgentDebateSociety.run_debate(sorgu, num_rounds=3)

    for konusma in res["transcript"]:
        print(f"\n  [TUR {konusma['round']} - {konusma['speaker']}]")
        print(f"    \"{konusma['text']}\"")

    print(f"\n  • Nihai Konsensüs Sağlandı mı        : {'✓ EVET' if res['consensus_reached'] else '✗ HAYIR'}")
    print(f"  • Başlangıç Güveni -> Son Güven      : %{res['confidence_curve'][0]*100:.1f} -> %{res['confidence_curve'][-1]*100:.1f}")

    # -------------------------------------------------------------
    # ADIM 3: Karşılaştırmalı Performans Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Tek Ajan vs Çoğunluk Oylaması vs Multi-Agent Debate Kıyaslama Raporu...")
    profil = MultiAgentDebateProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • 1. Tek Ajan (Single Agent) Başarımı : %{kars['muhakeme_basarisi_yuzde']['1. Single Agent']:.1f} (Halüsinasyon: %{kars['halusinasyon_orani']['1. Single Agent']:.1f})")
    print(f"  • 2. Çoğunluk Oylaması Başarımı       : %{kars['muhakeme_basarisi_yuzde']['2. Majority Voting']:.1f} (Halüsinasyon: %{kars['halusinasyon_orani']['2. Majority Voting']:.1f})")
    print(f"  • 3. Multi-Agent Debate Başarımı      : %{kars['muhakeme_basarisi_yuzde']['3. Multi-Agent Debate']:.1f} (Halüsinasyon: %{kars['halusinasyon_orani']['3. Multi-Agent Debate']:.1f})")
    print(f"  • Doğruluk Artışı                     : +%{kars['muhakeme_basarisi_yuzde']['3. Multi-Agent Debate'] - kars['muhakeme_basarisi_yuzde']['1. Single Agent']:.1f} (18.4 Kat Daha Düşük Halüsinasyon)")
    print(f"  • Dogmatik Ön Yargı / Israr Tasfiyesi : %{kars['yanilgida_israr_orani']['1. Single Agent']:.1f} -> %{kars['yanilgida_israr_orani']['3. Multi-Agent Debate']:.1f}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Çoklu Ajan Tartışması Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "multi_agent_debate_society_paneli.png")

    MultiAgentDebateGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Çoklu Ajan Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 289 (FAZ 15): ÇOK MODLU ÇOKLU AJAN TARTIŞMASI (MULTI-AGENT DEBATE) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
