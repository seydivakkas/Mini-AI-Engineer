"""
Day 214: Length-Bias Cezalandırma ve Over-Thinking Önleme Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.length_bias_motoru import (
    LengthPenaltyObjective,
    OverthinkingDetector,
    AdaptiveLengthController,
    LengthRegularizedTrainer,
)
from src.length_bias_profilleyici import LengthBiasProfilleyici
from src.gorsellestirici import LengthBiasGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 214 (FAZ 11): LENGTH-BIAS CEZALANDIRMA & OVER-THINKING ÖNLEME (TOKEN VERİMLİLİK MİMARİSİ)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Dinamik Token Bütçesi Belirleme
    # -------------------------------------------------------------
    print("\n[1/4] Problem Karmaşıklığına Göre Dinamik Düşünce Bütçesi Belirleniyor...")
    ornek_sorular = [
        "2 + 2 kaç eder?",
        "Bir üçgenin kenarları 3, 4, 5 ise alanı kaçtır?",
        "Fermat'nın Son Teoremini cebirsel geometri yöntemleriyle ispatlayın.",
    ]
    for s in ornek_sorular:
        butce = AdaptiveLengthController.hedef_butce_belirle(s)
        print(f"  • Soru: '{s:<65}' -> Hedef Token Bütçesi: {butce} tok")
    print("  ✓ Dinamik Token Bütçeleri Başarıyla Hesaplandı!")

    # -------------------------------------------------------------
    # ADIM 2: Over-Thinking ve Gevezelik Tespiti
    # -------------------------------------------------------------
    print("\n[2/4] Döngüsel Düşünce (Over-Thinking) Analizi Yürütülüyor...")
    sisik_metin = (
        "Cevabı buldum 4. Dur bir dakika, tekrar kontrol edeyim. "
        "Tekrar kontrol edeyim. Baştan hesaplayalım. "
        "Emin olmak için bir daha bakalım. Sonuç yine 4."
    )
    analiz = OverthinkingDetector.analiz_et(sisik_metin)

    print(f"  • İncelenen Düşünce İzi: '{sisik_metin}'")
    print(f"  • Tespit Edilen Tekrarlar : {analiz['tekrar_sayisi']} adet")
    print(f"  • Cümle Tekrar Oranı      : %{analiz['tekrar_orani']*100:.1f}")
    print(f"  • Gevezelik Skoru         : {analiz['gevezelik_skoru']:.2f} / 1.00 (Over-thinking: {'⚠️ VAR' if analiz['overthinking_var_mi'] else '✅ TEMİZ'})")
    print("  ✓ Gevezelik ve Döngü Tespiti Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 3: Uzunluk Düzenlileştirmeli Ödül Değerlendirmesi
    # -------------------------------------------------------------
    print("\n[3/4] Menteşe (Hinge) Uzunluk Cezası ile Pareto Değerlendirmesi...")
    soru = "5*x = 20 denkleminde x kaçtır?"
    yanit = "<think>5*x = 20 -> x = 4</think>\nSonuç: \\boxed{4}"

    degerlendirme = LengthRegularizedTrainer.degerlendir(soru, yanit, dogru_mu=True)
    print(f"  • Gerçek Uzunluk        : {degerlendirme['uzunluk']} kelime")
    print(f"  • Hedef Bütçe           : {degerlendirme['hedef_butce']} kelime")
    print(f"  • Ham Doğruluk Ödülü    : {degerlendirme['ham_odul']:.2f}")
    print(f"  • Düzenlenmiş Hinge Ödül: {degerlendirme['duzenlenmis_odul']:.2f}")
    print(f"  • Token Verimlilik Skoru: {degerlendirme['verimlilik_skoru']:.2f} (Bits/Token)")
    print("  ✓ Pareto-Optimal Uzunluk Değerlendirmesi Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 4: Profilleme ve 6 Panelli Görsel Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Length-Bias ve Verimlilik Teşhis Panosu Oluşturuluyor...")
    profil_raporu = LengthBiasProfilleyici.verimlilik_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "length_bias_paneli.png")

    LengthBiasGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Length-Bias Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 214 (FAZ 11): LENGTH-BIAS CEZALANDIRMA & OVER-THINKING ÖNLEME BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
