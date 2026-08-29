"""
GÜN 152: Biçimsel Mantık ve Teorem İspatı (Lean 4) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.lean4_taktik_motoru import Lean4TaktikMotoru, HedefDurumu
from src.formal_teorem_ureticisi import FormalTeoremUreticisi
from src.itp_dogrulayici import ITPDogrulayici
from src.gorsellestirici import Lean4Gorsellestirici


def test_hedef_durumu_yapisi():
    """HedefDurumu sınıfının doğru hipotez ve hedef tuttuğunu test eder."""
    h = HedefDurumu(1, {"n": "Nat"}, "n + 0 = n")
    assert h.hedef_id == 1
    assert h.hipotezler["n"] == "Nat"
    assert h.hedef_ifadesi == "n + 0 = n"


def test_lean4_taktik_motoru_induction():
    """Tümevarım (induction) taktiğinin hedefi 2 alt hedefe böldüğünü test eder."""
    motor = Lean4TaktikMotoru("add_zero", "n + 0 = n")
    assert len(motor.acik_hedefler) == 1

    kayit = motor.taktik_uygula("induction n")
    assert kayit["kalan_hedef_sayisi"] == 2
    assert len(motor.acik_hedefler) == 2


def test_lean4_taktik_motoru_rfl():
    """Reflexivity (rfl) taktiğinin bir alt hedefi kapattığını test eder."""
    motor = Lean4TaktikMotoru("add_zero", "n + 0 = n")
    motor.taktik_uygula("induction n")
    assert len(motor.acik_hedefler) == 2

    kayit = motor.taktik_uygula("rfl")
    assert kayit["kalan_hedef_sayisi"] == 1
    assert len(motor.acik_hedefler) == 1


def test_lean4_taktik_motoru_tam_ispat():
    """Taktik dizisinin tüm hedefleri kapatıp ispatı tamamladığını test eder."""
    motor = Lean4TaktikMotoru("add_zero", "n + 0 = n")
    motor.taktik_uygula("induction n")
    motor.taktik_uygula("rfl")
    son_kayit = motor.taktik_uygula("rw [hd]")

    assert son_kayit["kalan_hedef_sayisi"] == 0
    assert motor.ispatlandi_mi is True


def test_formal_teorem_ureticisi_autoformalization():
    """FormalTeoremUreticisi'nin Lean 4 sözdizimini doğru biçimsellendirdiğini test eder."""
    ureticisi = FormalTeoremUreticisi()
    teorem = ureticisi.teoremi_bicimsellestir("Her n için n + 0 = n olduğunu ispatla")

    assert "theorem add_zero" in teorem["lean4_kodu"]
    assert "induction n" in teorem["taktik_adimlari"]
    assert teorem["hedef_ifadesi"] == "n + 0 = n"


def test_itp_dogrulayici_tam_akis():
    """ITPDogrulayici'nin uçtan uca teorem ispatını başarıyla doğruladığını test eder."""
    sonuc = ITPDogrulayici.teoremi_ispatla_ve_dogrula("Doğal sayılar için toplama etkisiz eleman ispatı: n + 0 = n")

    assert sonuc["ispatlandi_mi"] is True
    assert sonuc["kalan_hedef_sayisi"] == 0
    assert sonuc["taktik_sayisi"] == 3


def test_itp_dogrulayici_kalan_hedef_kontrolu():
    """İspat tamamlandığında açık hedef kalmadığını (no goals left) test eder."""
    sonuc = ITPDogrulayici.teoremi_ispatla_ve_dogrula("n + 0 = n")
    son_adim = sonuc["adim_kayitlari"][-1]
    assert son_adim["kalan_hedef_sayisi"] == 0
    assert son_adim["ispatlandi_mi"] is True


def test_gorsellestirici_pano_uretme():
    """6 panelli Lean 4 teşhis panosunun PNG olarak başarıyla kaydedildiğini test eder."""
    sonuc = ITPDogrulayici.teoremi_ispatla_ve_dogrula("n + 0 = n")
    gorsellestirici = Lean4Gorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_lean4_pano.png")
        gorsellestirici.pano_olustur(sonuc, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
