"""
Day 208: Kural Tabanlı Doğrulayıcılar (Rule-Based Verifiers) ile Sembolik Matematik ve AST Doğrulama Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dogrulayici_motoru import (
    SymPyMathVerifier,
    PythonASTCodeVerifier,
    DeterministicUnitTestRunner,
    RuleBasedRewardEngine,
)
from src.dogrulayici_profilleyici import DogrulayiciProfilleyici
from src.gorsellestirici import DogrulayiciGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 208 (FAZ 11): RULE-BASED VERIFIERS (SYMPY & AST) HALLUCINATION-FREE REWARD ENGINE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: SymPy Sembolik Matematik Doğrulayıcısı Testi
    # -------------------------------------------------------------
    print("\n[1/4] SymPy Sembolik Cebirsel Eşdeğerlik Doğrulanıyor...")
    testler_matematik = [
        ("x^2 - 1", "(x-1)*(x+1)"),
        ("\\frac{\\sqrt{2}}{2}", "1/sqrt(2)"),
        ("3*x + 6 = 21", "5", "denklem"),
    ]
    for test in testler_matematik:
        if len(test) == 2:
            sonuc = SymPyMathVerifier.sembolik_esitlik_kontrolu(test[0], test[1])
            print(f"  • İfade Eşleşmesi : '{test[0]}' == '{test[1]}' -> {'✅ DOĞRU (Özdeş)' if sonuc else '❌ HATALI'}")
        else:
            sonuc = SymPyMathVerifier.denklem_cozumu_dogrula(test[0], test[1])
            print(f"  • Denklem Çözümü  : '{test[0]}' için x={test[1]} -> {'✅ DOĞRU ÇÖZÜM' if sonuc else '❌ HATALI'}")
    print("  ✓ SymPy Sembolik Motoru Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 2: Python AST Statik Sentaks ve Güvenlik Denetimi
    # -------------------------------------------------------------
    print("\n[2/4] Python AST Statik Sentaks ve Güvenlik Filtresi Test Ediliyor...")
    guvenli_kod = "def topla(a, b):\n    return a + b\n"
    zararli_kod = "import os\nos.system('dir')\n"

    g_sentaks, _ = PythonASTCodeVerifier.sentaks_dogrula(guvenli_kod)
    g_guvenlik, g_msg = PythonASTCodeVerifier.guvenlik_ve_yapi_denetle(guvenli_kod)
    z_guvenlik, z_msg = PythonASTCodeVerifier.guvenlik_ve_yapi_denetle(zararli_kod)

    print(f"  • Güvenli Kod Sentaksı  : {'✅ GEÇERLİ' if g_sentaks else '❌ HATALI'}")
    print(f"  • Güvenli Kod Durumu    : {'✅ ONAYLANDI' if g_guvenlik else '❌ RED'} ({g_msg})")
    print(f"  • Zararlı Kod Durumu    : {'❌ ENGELLENDİ' if not z_guvenlik else '⚠️ KAÇAK'} ({z_msg})")
    print("  ✓ Python AST Güvenlik Kalkanı Başarıyla Çalıştı!")

    # -------------------------------------------------------------
    # ADIM 3: Deterministik Birim Test ve Tam Ödül Motoru
    # -------------------------------------------------------------
    print("\n[3/4] Deterministik Birim Test Yürütücüsü ve RLVR Ödül Skoru...")
    ornek_kod = (
        "def asal_mi(n: int) -> bool:\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for i in range(2, int(n**0.5) + 1):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n"
    )
    testler = [(2, True), (3, True), (4, False), (17, True), (1, False)]
    test_sonucu = DeterministicUnitTestRunner.birim_testleri_calistir(
        ornek_kod, "asal_mi", testler
    )
    print(f"  • Birim Test Başarımı   : {test_sonucu['gecen_test']} / {test_sonucu['toplam_test']} Test Geçti (%{test_sonucu['basari_orani']*100:.0f})")

    # Matematiksel Format + Doğruluk Ödülü
    ornek_yanit = "<think>\nAdım adım işlem yapıldı.\n</think>\nSonuç: \\boxed{\\frac{1}{\\sqrt{2}}}"
    odul_raporu = RuleBasedRewardEngine.matematik_odulu_hesapla(ornek_yanit, "\\frac{\\sqrt{2}}{2}")
    print(f"  • Format Ödülü          : {odul_raporu['format_odulu']:.2f}")
    print(f"  • SymPy Doğruluk Ödülü  : {odul_raporu['dogruluk_odulu']:.2f}")
    print(f"  • Toplam Deterministik R: {odul_raporu['toplam_odul']:.2f} / 1.00")

    # -------------------------------------------------------------
    # ADIM 4: Profilleme ve 6 Panelli Görsel Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Kural Tabanlı Doğrulayıcı Teşhis Panosu Oluşturuluyor...")
    profil_raporu = DogrulayiciProfilleyici.tam_karsilastirma_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "rule_based_verifier_paneli.png")

    DogrulayiciGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Doğrulayıcı Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 208 (FAZ 11): KURAL TABANLI DOĞRULAYICILAR (RULE-BASED VERIFIERS) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
