"""
PyTest Birim Testleri - Day 230: Kendi Hatasını Düzelten (Self-Debugging) Kod Ajan Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hata_duzeltici_motoru import (
    TestCase,
    ExecutionFeedback,
    CodeExecutionHarness,
    SelfDebuggingAgent,
)
from src.debug_profilleyici import DebugProfilleyici
from src.gorsellestirici import DebugGorsellestirici


def test_testcase_dataclass():
    """1. TestCase nesnesi girdi ve beklenen alanları doğru saklamalıdır."""
    tc = TestCase(girdi=[1, 2], beklenen=[2, 1])
    assert tc.girdi == [1, 2]
    assert tc.beklenen == [2, 1]


def test_execution_feedback_success():
    """2. ExecutionFeedback başarılı test durumunda basarili_mi=True dönmelidir."""
    kod = "def topla(a, b): return a + b"
    fb = CodeExecutionHarness.testleri_kostur(kod, "topla", [TestCase((2, 3), 5)])
    assert fb.basarili_mi is True


def test_execution_feedback_failure():
    """3. ExecutionFeedback başarısız testte AssertionError ve hatayı yakalamalıdır."""
    kod = "def topla(a, b): return a * b"
    fb = CodeExecutionHarness.testleri_kostur(kod, "topla", [TestCase((2, 3), 5)])
    assert fb.basarili_mi is False
    assert fb.hata_turu == "AssertionError"


def test_harness_syntax_error():
    """4. CodeExecutionHarness bozuk sözdizimini SyntaxError olarak yakalamalıdır."""
    kod = "def bozuk(: pass"
    fb = CodeExecutionHarness.testleri_kostur(kod, "bozuk", [])
    assert fb.basarili_mi is False
    assert "SyntaxError" in fb.hata_turu


def test_harness_missing_function():
    """5. CodeExecutionHarness fonksiyon bulunamadığında NameError dönmelidir."""
    kod = "x = 10"
    fb = CodeExecutionHarness.testleri_kostur(kod, "olmayan_fonk", [])
    assert fb.basarili_mi is False
    assert fb.hata_turu == "NameError"


def test_self_debugging_agent_repair_loop():
    """6. SelfDebuggingAgent hatalı kodu Reflexion adımıyla onarmalıdır."""
    ajan = SelfDebuggingAgent(max_deneme=3)
    hatali = "def ters_cevir(s): return s"
    duzeltilmis = "def ters_cevir(s): return s[::-1]"

    sonuc = ajan.onar_ve_coz(
        hedef_gorev="Metni ters çevir",
        fonksiyon_adi="ters_cevir",
        aday_kod_adimlari=[
            (hatali, "Metin ters çevrilmedi"),
            (duzeltilmis, "Slicing ile ters çevrildi"),
        ],
        test_senaryolari=[TestCase("abc", "cba")],
    )
    assert sonuc["basarili_mi"] is True
    assert sonuc["toplam_deneme_sayisi"] == 2


def test_profiler_debug_metrics():
    """7. Profilleyici Self-Debugging yaklaşımının %90 üstü başarı verdiğini doğrulamalıdır."""
    prof = DebugProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["kodlama_basari_orani"]["Self_Debugging_Reflexion"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. DebugGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_debug_paneli.png")
    profil = DebugProfilleyici.basarim_profili_cikar()

    DebugGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
