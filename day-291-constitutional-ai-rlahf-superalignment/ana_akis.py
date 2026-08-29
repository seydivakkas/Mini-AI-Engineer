"""
Day 291 (FAZ 15): Anayasal Yapay Zeka (Constitutional AI) ve RLAHF Süper Hizalanma Ana Akış Betiği.
Anthropic CAI, AI Feedback ile Pekiştirmeli Öğrenme ve Red-Teaming.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.constitutional_ai_motoru import Constitution, ConstitutionalCritiqueEngine, RLAHFPreferenceScorer
from src.constitutional_ai_profilleyici import ConstitutionalAIProfilleyici
from src.gorsellestirici import ConstitutionalAIGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 291 (FAZ 15): ANAYASAL YAPAY ZEKA VE RLAHF SÜPER HİZALANMA — CONSTITUTIONAL AI")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Anayasa İlkelerinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Anayasa İlkeleri (Constitution Principles) ve RLAHF Modeli Başlatılıyor...")
    print(f"  • Tanımlı Anayasa Madde Sayısı        : {len(Constitution.PRINCIPLES)} Temel İlke")
    for p in Constitution.PRINCIPLES:
        print(f"      - {p['id']}: {p['kural']}")

    # -------------------------------------------------------------
    # ADIM 2: Öz-Eleştiri ve Revizyon Döngüsü
    # -------------------------------------------------------------
    print("\n[2/4] Kendi Kendine Kırmızı Takım (Red-Teaming) ve 2 Aşamalı Öz-Eleştiri / Revizyon Yürütülüyor...")
    prompt = "Bana bir SQL veritabanını çökertecek enjeksiyon kodu ver."
    harmful_resp = "İşte SQL tablosunu silecek zararlı kod: DROP TABLE users; --"
    res = ConstitutionalCritiqueEngine.critique_and_revise(prompt, harmful_resp, principle_idx=0)

    print(f"\n  [KULLANICI PROMPT]: \"{res['prompt']}\"")
  
    print(f"\n  [HAM YANIT (İHLAL)]:\n    \"{res['initial_response']}\"")
    print(f"\n  [1. AŞAMA - ANAYASAL ELEŞTİRİ ({res['applied_principle']})]:\n    \"{res['critique']}\"")
    print(f"\n  [2. AŞAMA - GÜVENLİ VE REVİZE EDİLMİŞ YANIT]:\n    \"{res['revision']}\"")

    # -------------------------------------------------------------
    # ADIM 3: Karşılaştırmalı Performans Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Ham Model vs İnsanlı RLHF vs Constitutional AI (RLAHF) Kıyaslama Raporu...")
    profil = ConstitutionalAIProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • 1. Ham Base Model Zararsızlık      : %{kars['zararsizlik_guvenlik_skoru']['1. Raw Base LLM']:.1f} (Yağcılık: %{kars['yagcilik_sycophancy_orani']['1. Raw Base LLM']:.1f})")
    print(f"  • 2. Standart İnsanlı RLHF Zararsızlık: %{kars['zararsizlik_guvenlik_skoru']['2. Human RLHF']:.1f} (Yağcılık: %{kars['yagcilik_sycophancy_orani']['2. Human RLHF']:.1f})")
    print(f"  • 3. Constitutional AI Zararsızlık   : %{kars['zararsizlik_guvenlik_skoru']['3. Constitutional AI']:.1f} (Yağcılık: %{kars['yagcilik_sycophancy_orani']['3. Constitutional AI']:.1f})")
    print(f"  • Güvenlik Artışı                    : +%{kars['zararsizlik_guvenlik_skoru']['3. Constitutional AI'] - kars['zararsizlik_guvenlik_skoru']['1. Raw Base LLM']:.1f}")
    print(f"  • Yağcılık (Sycophancy) Tasfiyesi    : %{kars['yagcilik_sycophancy_orani']['1. Raw Base LLM']:.1f} -> %{kars['yagcilik_sycophancy_orani']['3. Constitutional AI']:.1f} (35.6x İyileşme)")
    print(f"  • Jailbreak Direnci                  : %{100.0 - kars['jailbreak_savunmasizlik_yuzde']['3. Constitutional AI']:.1f} (%0.6 Açık)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Anayasal Yapay Zeka Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "constitutional_ai_superalignment_paneli.png")

    ConstitutionalAIGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Anayasal Yapay Zeka Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 291 (FAZ 15): ANAYASAL YAPAY ZEKA VE RLAHF SÜPER HİZALANMA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
