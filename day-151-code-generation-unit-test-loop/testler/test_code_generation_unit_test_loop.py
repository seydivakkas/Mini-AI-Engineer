"""
GÜN 151: Test Odaklı Kod Üretimi (TDD) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.kod_ureticisi import KodUreticisi
from src.test_yurutucu import TestYurutucu
from src.tdd_dongusu_yoneticisi import TDDDongusuYoneticisi
from src.gorsellestirici import TDDGorsellestirici


def test_kod_ureticisi_ilk_taslak():
    """KodUreticisi'nin ilk taslak kod ürettiğini test eder."""
    ureticisi = KodUreticisi()
    taslak = ureticisi.ilk_kodu_uret("RLE")

    assert "def run_length_encoding" in taslak["kod"]
    assert taslak["tur"] == 1


def test_test_yurutucu_hatali_kod():
    """TestYurutucu'nün ilk taslaktaki hatayı yakalayıp Traceback verdiğini test eder."""
    ureticisi = KodUreticisi()
    taslak = ureticisi.ilk_kodu_uret("RLE")

    sonuc = TestYurutucu.testleri_kostur(taslak["kod"])
    assert sonuc["tum_testler_gecti_mi"] is False
    assert sonuc["gecen_sayisi"] < sonuc["toplam_test_sayisi"]
    assert sonuc["hata_raporu"] is not None


def test_kod_ureticisi_onarim():
    """KodUreticisi'nin hata raporunu analiz ederek yamalama monoloğu ürettiğini test eder."""
    ureticisi = KodUreticisi()
    onarim = ureticisi.kodu_onar("RLE", "eski_kod", "IndexError: string index out of range", 2)

    assert "<think>" in onarim["onarma_monologu"]
    assert "if not metin:" in onarim["kod"]
    assert onarim["tur"] == 2


def test_test_yurutucu_onarilmis_kod():
    """TestYurutucu'nün onarılmış kodu 4/4 PASSED olarak onayladığını test eder."""
    ureticisi = KodUreticisi()
    onarim = ureticisi.kodu_onar("RLE", "eski_kod", "hata", 2)

    sonuc = TestYurutucu.testleri_kostur(onarim["kod"])
    assert sonuc["tum_testler_gecti_mi"] is True
    assert sonuc["gecen_sayisi"] == 4
    assert sonuc["hata_raporu"] is None


def test_tdd_dongusu_tam_akis():
    """TDDDongusuYoneticisi'nin 2. turda kodu %100 başarıyla onardığını test eder."""
    yonetici = TDDDongusuYoneticisi()
    sonuc = yonetici.tdd_dongusunu_baslat("Run-Length Encoding")

    assert sonuc["toplam_tur"] == 2
    assert sonuc["basarili_mi"] is True
    assert "if not metin:" in sonuc["nihai_kod"]


def test_tdd_dongusu_bos_string_korumasi():
    """Nihai kodun boş string girdisinde boş string döndürdüğünü test eder."""
    yonetici = TDDDongusuYoneticisi()
    sonuc = yonetici.tdd_dongusunu_baslat("RLE")

    yerel = {}
    exec(sonuc["nihai_kod"], {}, yerel)
    fn = yerel["run_length_encoding"]
    assert fn("") == ""


def test_tdd_dongusu_son_karakter_tamponu():
    """Nihai kodun son karakter grubunu eksiksiz kodladığını test eder."""
    yonetici = TDDDongusuYoneticisi()
    sonuc = yonetici.tdd_dongusunu_baslat("RLE")

    yerel = {}
    exec(sonuc["nihai_kod"], {}, yerel)
    fn = yerel["run_length_encoding"]
    assert fn("AABBC") == "A2B2C1"
    assert fn("WWWWWW") == "W6"


def test_gorsellestirici_pano_uretme():
    """6 panelli TDD teşhis panosunun PNG olarak başarıyla kaydedildiğini test eder."""
    yonetici = TDDDongusuYoneticisi()
    sonuc = yonetici.tdd_dongusunu_baslat("RLE")

    gorsellestirici = TDDGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_tdd_pano.png")
        gorsellestirici.pano_olustur(sonuc, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
