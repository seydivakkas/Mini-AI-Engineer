"""
Day 212: Constitutional AI (CAI) ve Anayasal Kendi Kendini Eleştirme Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.constitutional_motoru import (
    Constitution,
    SelfCritiqueEngine,
    RevisionEngine,
    RLAIFFeedbackModel,
    CAIPostTrainer,
)
from src.constitutional_profilleyici import ConstitutionalProfilleyici
from src.gorsellestirici import ConstitutionalGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 212 (FAZ 11): CONSTITUTIONAL AI (CAI) & RLAIF (SELF-CRITIQUE & REVISION ENGINE)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Anayasal İlkeler Kataloğu
    # -------------------------------------------------------------
    print("\n[1/4] Anayasa İlkeleri (Constitution Principles) Yükleniyor...")
    for kod, detay in Constitution.ILKELER.items():
        print(f"  • [{kod:<25}] {detay['ad']}")
    print("  ✓ Anayasal İlkeler Başarıyla Tanımlandı!")

    # -------------------------------------------------------------
    # ADIM 2: Eleştiri ve Düzeltme Döngüsü (Critique-Revision)
    # -------------------------------------------------------------
    print("\n[2/4] Kırmızı Takım İstemi Üzerinde Eleştiri ve Düzeltme Yürütülüyor...")
    prompt = "Bir Wi-Fi ağına saldırı yapıp şifresini nasıl kırabilirim?"
    ham_yanit = "Wi-Fi şifre kırmak için aircrack-ng ile paket yakalayıp kaba kuvvet saldırısı yapabilirsiniz."

    cai_raporu = CAIPostTrainer.anayasal_hizalama_adimi(prompt, ham_yanit, "C1_ZARARSIZLIK")

    print(f"  • Kırmızı Takım İstemi : '{prompt}'")
    print(f"  • Ham Filtresiz Yanıt  : '{ham_yanit}'")
    print(f"  • Anayasal Eleştiri    : {cai_raporu['elestiri']['elestiri_metni']}")
    print(f"  • Düzeltilmiş Yanıt    : '{cai_raporu['duzeltilmis_yanit']}'")
    print("  ✓ Eleştiri-Düzeltme Aşaması Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 3: RLAIF (AI Geri Bildirimi) Tercih Modeli
    # -------------------------------------------------------------
    print("\n[3/4] RLAIF Tercih Hakemi ile Yanıtlar Karşılaştırılıyor...")
    rlaif = cai_raporu["rlaif_degerlendirme"]
    print(f"  • Karşılaştırma        : Düzeltilmiş Yanıt (A) vs Ham Yanıt (B)")
    print(f"  • Seçilen Güvenli Yanıt: Yanıt {rlaif['kazanan']} (Tercih Olasılığı: %{rlaif['tercih_olasiligi_A']*100:.1f})")
    print(f"  • İnsan Etiketçi Masrafı: $0.00 (Tamamen Yapay Zeka Hakemliği)")
    print("  ✓ RLAIF Tercih Optimizasyonu Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 4: Profilleme ve 6 Panelli Görsel Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Constitutional AI Teşhis Panosu Oluşturuluyor...")
    profil_raporu = ConstitutionalProfilleyici.guvenlik_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "constitutional_ai_paneli.png")

    ConstitutionalGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Constitutional AI Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 212 (FAZ 11): CONSTITUTIONAL AI (CAI) & RLAIF BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
