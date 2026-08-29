"""
PyTest Birim Testleri - Day 208: Kural Tabanlı Doğrulayıcılar (SymPy & AST).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dogrulayici_motoru import (
    SymPyMathVerifier,
    PythonASTCodeVerifier,
    DeterministicUnitTestRunner,
    RuleBasedRewardEngine,
)
from src.dogrulayici_profilleyici import DogrulayiciProfilleyici
from src.gorsellestirici import DogrulayiciGorsellestirici


def test_sympy_exact_equivalence():
    """1. SymPy cebirsel olarak özdeş ifadeleri doğru (True) olarak doğrulamalıdır."""
    assert SymPyMathVerifier.sembolik_esitlik_kontrolu("x^2 - 1", "(x-1)*(x+1)")
    assert SymPyMathVerifier.sembolik_esitlik_kontrolu("2*x + 4", "2*(x + 2)")


def test_sympy_latex_fractions_and_roots():
    """2. SymPy LaTeX kesir ve kök ifadelerini başarıyla ayrıştırmalıdır."""
    assert SymPyMathVerifier.sembolik_esitlik_kontrolu("\\frac{\\sqrt{2}}{2}", "1/sqrt(2)")
    assert SymPyMathVerifier.sembolik_esitlik_kontrolu("\\frac{1}{2}", "0.5")


def test_sympy_equation_solution():
    """3. SymPy bir adayın denklem çözümünü (LHS == RHS) doğrulamalıdır."""
    assert SymPyMathVerifier.denklem_cozumu_dogrula("2*x + 5 = 15", "5")
    assert not SymPyMathVerifier.denklem_cozumu_dogrula("2*x + 5 = 15", "6")


def test_python_ast_valid_syntax():
    """4. PythonASTCodeVerifier geçerli sentakslı kodu onaylamalıdır."""
    kod = "def topla(a, b):\n    return a + b\n"
    gecerli, err = PythonASTCodeVerifier.sentaks_dogrula(kod)
    assert gecerli is True
    assert err is None


def test_python_ast_invalid_syntax():
    """5. PythonASTCodeVerifier hatalı sentaksı yakalamalıdır."""
    hatali_kod = "def hatali(:\n    return\n"
    gecerli, err = PythonASTCodeVerifier.sentaks_dogrula(hatali_kod)
    assert gecerli is False
    assert "Sentaks Hatası" in err


def test_python_ast_forbidden_imports():
    """6. PythonASTCodeVerifier yasaklı ve tehlikeli modül çağrılarını engellemelidir."""
    zararli = "import os\nos.system('calc')\n"
    guvenli, msg = PythonASTCodeVerifier.guvenlik_ve_yapi_denetle(zararli)
    assert guvenli is False
    assert "Yasaklı" in msg


def test_deterministic_unit_test_runner():
    """7. DeterministicUnitTestRunner birim testleri doğru çalıştırmalıdır."""
    kod = "def kare(x):\n    return x * x\n"
    testler = [(2, 4), (3, 9), (-4, 16)]
    sonuc = DeterministicUnitTestRunner.birim_testleri_calistir(kod, "kare", testler)

    assert sonuc["basarili"] is True
    assert sonuc["gecen_test"] == 3
    assert sonuc["toplam_test"] == 3


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. DogrulayiciGorsellestirici 6 panelli teşhis panosunu üretmelidir."""
    cikti = str(tmp_path / "test_verifier_paneli.png")
    profil = DogrulayiciProfilleyici.tam_karsilastirma_profili_cikar()

    DogrulayiciGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
