"""
GÜN 150: Sembolik Akıl Yürütme (Z3 & SymPy) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.sympy_sembolik_motor import SymPySembolikMotor
from src.z3_smt_cozucu import Z3SMTCozucu
from src.neuro_sembolik_kopru import NeuroSembolikKopru
from src.gorsellestirici import SembolikReasoningGorsellestirici


def test_sympy_denklem_cozumu():
    """SymPy'nin ikinci dereceden polinom köklerini kesin çözdüğünü test eder."""
    kokler = SymPySembolikMotor.denklem_coz("x**2 - 5*x + 6", "0", "x")
    assert 2.0 in kokler
    assert 3.0 in kokler
    assert len(kokler) == 2


def test_sympy_moduler_aritmetik():
    """SymPy modüler kongrüans çözümünü test eder."""
    x_val = SymPySembolikMotor.moduler_coz(a=3, b=0, m=5)
    assert x_val == 0
    assert (3 * x_val) % 5 == 0


def test_sympy_turev_ve_integral():
    """SymPy sembolik türev ve integral işlemlerini test eder."""
    turev = SymPySembolikMotor.turev_al("x**3", "x")
    assert "3*x**2" in turev

    integral = SymPySembolikMotor.integral_al("x**2", "x")
    assert "x**3/3" in integral or "x**3" in integral


def test_sympy_sadelestirme():
    """SymPy sembolik trigonometrik sadeleştirmeyi test eder."""
    sade = SymPySembolikMotor.sadelestir("sin(x)**2 + cos(x)**2")
    assert sade == "1"


def test_z3_smt_sopa_top():
    """Z3 SMT çözücünün Sopa ve Top problemini kesin olarak çözdüğünü test eder."""
    sonuc = Z3SMTCozucu.sopa_ve_top_coz()
    assert sonuc["sat_mi"] is True
    assert pytest.approx(sonuc["top"], 1e-3) == 0.05
    assert pytest.approx(sonuc["sopa"], 1e-3) == 1.05


def test_z3_smt_tam_sayi_kisitlari():
    """Z3 SMT çözücünün tam sayı kısıtlarını tatmin ettiğini test eder."""
    sonuc = Z3SMTCozucu.tam_sayi_kisit_coz(toplam=15, carpim=56)
    assert sonuc["sat_mi"] is True
    x, y = sonuc["x"], sonuc["y"]
    assert x + y == 15
    assert x * y == 56


def test_neuro_sembolik_kopru_kapsamli_ispat():
    """NeuroSembolikKopru'nün tüm matematiksel ve mantıksal ispatları başarıyla çalıştırdığını test eder."""
    rapor = NeuroSembolikKopru.calistir_kapsamli_ispat()
    assert rapor["tum_ispatlar_gecerli_mi"] is True
    assert len(rapor["sympy_kokler"]) == 2
    assert rapor["z3_sopa_top"]["sat_mi"] is True


def test_gorsellestirici_pano_uretme():
    """6 panelli Sembolik Reasoning teşhis panosunun PNG olarak başarıyla kaydedildiğini test eder."""
    rapor = NeuroSembolikKopru.calistir_kapsamli_ispat()
    gorsellestirici = SembolikReasoningGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_symbolic_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
