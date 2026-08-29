"""
GÜN 156: Chain of Verification (CoVe) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.taslak_ureticisi import TaslakUreticisi
from src.cove_soru_planlayici import CoVESoruPlanlayici
from src.bagimsiz_dogrulayici import BagimsizDogrulayici
from src.cove_duzeltici_motor import CoVEDuzelticiMotor
from src.gorsellestirici import CoVEGorsellestirici


def test_taslak_ureticisi_iddialar():
    """Taslak üreticisinin olgusal iddiaları doğru çıkardığını test eder."""
    soru = "Mehmet Akif Ersoy nerede doğdu ve İstiklal Marşı'nı nerede yazdı?"
    sonuc = TaslakUreticisi.taslak_uret(soru)

    assert "Ankara'da doğmuştur" in sonuc["taslak_yanit"]
    assert len(sonuc["iddialar"]) == 3
    assert sonuc["iddialar"][0]["konu"] == "Doğum Yeri"


def test_cove_soru_planlayici_soru_sayisi():
    """Planlayıcının her iddia için tam 1 bağımsız doğrulama sorusu ürettiğini test eder."""
    iddialar = [
        {"iddia_id": "c1", "konu": "Doğum Yeri", "taslak_ifade": "Ankara'da doğmuştur"},
        {"iddia_id": "c2", "konu": "Kabul Yılı", "taslak_ifade": "1923 yılında kabul edilmiştir"},
    ]
    sorular = CoVESoruPlanlayici.sorulari_planla(iddialar)

    assert len(sorular) == 2
    assert "hangi şehirde" in sorular[0]["dogrulama_sorusu"]


def test_cove_soru_planlayici_onyargisizlik():
    """Üretilen soruların taslaktaki hatalı iddiayı kopyalamadan tarafsız sorulduğunu test eder."""
    iddialar = [{"iddia_id": "c1", "konu": "Doğum Yeri", "taslak_ifade": "Ankara'da doğmuştur"}]
    sorular = CoVESoruPlanlayici.sorulari_planla(iddialar)

    # "Ankara" kelimesi doğrulama sorusunda yer almamalıdır (Önyargı engelleme)
    assert "Ankara" not in sorular[0]["dogrulama_sorusu"]


def test_bagimsiz_dogrulayici_fakt_kontrolu():
    """Bağımsız doğrulayıcının doğru olgusal değerleri döndürdüğünü test eder."""
    sorular = [{"iddia_id": "c1", "konu": "Doğum Yeri", "dogrulama_sorusu": "Nerede doğdu?", "taslak_iddia": "Ankara"}]
    yanitlar = BagimsizDogrulayici.sorulari_yanitla(sorular)

    assert len(yanitlar) == 1
    assert "İstanbul" in yanitlar[0]["dogrulanmis_cevap"]


def test_bagimsiz_dogrulayici_celiski_tespiti():
    """Hatalı taslak ile bağımsız gerçek arasındaki çelişkinin tespit edildiğini test eder."""
    sorular = [{"iddia_id": "c1", "konu": "Doğum Yeri", "dogrulama_sorusu": "Nerede doğdu?", "taslak_iddia": "Ankara"}]
    yanitlar = BagimsizDogrulayici.sorulari_yanitla(sorular)

    assert yanitlar[0]["celiski_var_mi"] is True
    assert "DÜZELTİLDİ" in yanitlar[0]["durum"]


def test_cove_duzeltici_motor_tam_akis():
    """CoVE tam akışının halüsinasyonları düzeltip doğru metni ürettiğini test eder."""
    soru = "Mehmet Akif Ersoy biyografisi ve İstiklal Marşı"
    sonuc = CoVEDuzelticiMotor.calistir(soru)

    assert "İstanbul" in sonuc["duzeltilmis_yanit"]
    assert "Taceddin Dergâhı" in sonuc["duzeltilmis_yanit"]
    assert "12 Mart 1921" in sonuc["duzeltilmis_yanit"]
    assert sonuc["cove_dogruluk_orani"] == 100.0


def test_cove_duzeltici_motor_metrikler():
    """CoVE değerlendirme metriklerinin doğru hesaplandığını test eder."""
    soru = "Test Sorusu"
    sonuc = CoVEDuzelticiMotor.calistir(soru)

    assert sonuc["toplam_iddia_sayisi"] == 3
    assert sonuc["duzeltilen_iddia_sayisi"] == 3
    assert sonuc["halusinasyon_temizleme_orani"] == 100.0


def test_gorsellestirici_pano_uretme():
    """6 panelli CoVE teşhis panosunun PNG olarak kaydedildiğini test eder."""
    sonuc = CoVEDuzelticiMotor.calistir("Örnek Soru")
    gorsellestirici = CoVEGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_cove_pano.png")
        gorsellestirici.pano_olustur(sonuc, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
