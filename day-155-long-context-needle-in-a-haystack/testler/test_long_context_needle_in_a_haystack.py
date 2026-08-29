"""
GÜN 155: Needle In A Haystack (NIAH) Uzun Bağlam Değerlendirme Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import numpy as np

from src.samanlik_olusturucu import SamanlikOlusturucu
from src.niah_test_motoru import NIAHTestMotoru
from src.coklu_igne_akil_yurutucu import CokluIgneAkilYurutucu
from src.gorsellestirici import NIAHGorsellestirici


def test_samanlik_olusturucu_uzunluk_ve_igne():
    """Samanlık oluşturucunun istenen uzunlukta metin ürettiğini ve iğneyi içerdiğini test eder."""
    igne = "Gizli Kod: 998877"
    sonuc = SamanlikOlusturucu.samanlik_uret(
        hedef_kelime_sayisi=500,
        igne_metni=igne,
        derinlik_yuzdesi=0.5,
    )

    assert igne in sonuc["tam_dokuman"]
    assert sonuc["toplam_kelime_sayisi"] >= 400
    assert sonuc["derinlik_yuzdesi"] == 50.0


def test_samanlik_olusturucu_derinlik_oranlari():
    """İğnenin dokümanın başına (%0) ve sonuna (%100) doğru yerleştiğini test eder."""
    res_bas = SamanlikOlusturucu.samanlik_uret(200, "BAS_IGNE", 0.0)
    res_son = SamanlikOlusturucu.samanlik_uret(200, "SON_IGNE", 1.0)

    assert res_bas["ekleme_indeksi"] == 0
    assert res_son["ekleme_indeksi"] == res_son["toplam_paragraf_sayisi"] - 1


def test_niah_test_motoru_matris_boyutu():
    """NIAH test motorunun 8 bağlam x 11 derinlik matrisi ürettiğini test eder."""
    motor = NIAHTestMotoru()
    matris = motor.izgara_matrisini_hesapla()

    assert matris.shape == (8, 11)
    assert np.all(matris >= 0.0) and np.all(matris <= 1.0)


def test_niah_test_motoru_kucuk_baglam_tam_dogruluk():
    """Küçük bağlamlarda (<=16k) doğruluk oranının %100 olduğunu test eder."""
    motor = NIAHTestMotoru()
    matris = motor.izgara_matrisini_hesapla()

    # İlk 5 satır (1k, 2k, 4k, 8k, 16k)
    assert np.all(matris[:5, :] == 1.0)


def test_niah_test_motoru_lost_in_middle_dusus():
    """128k bağlamda orta derinlikteki doğruluğun uçlardan daha düşük olduğunu test eder."""
    motor = NIAHTestMotoru()
    matris = motor.izgara_matrisini_hesapla()

    satir_128k = matris[-1, :]
    uc_dogruluk = (satir_128k[0] + satir_128k[-1]) / 2.0
    orta_dogruluk = satir_128k[5] # %50 derinlik

    assert orta_dogruluk < uc_dogruluk


def test_coklu_igne_akil_yurutucu_sentez():
    """Çoklu iğne motorunun 3 ayrı ipucundan doğru matematiksel çıkarım yaptığını test eder."""
    sonuc = CokluIgneAkilYurutucu.coklu_igne_sentezle("Test dokümanı")

    assert sonuc["akil_yurutme_basarili_mi"] is True
    assert sonuc["igne_sayisi"] == 3
    assert sonuc["nihai_cevap"] == 37.5


def test_niah_tam_degerlendirme_istatistikleri():
    """NIAH tam değerlendirme raporundaki metriklerin doğru hesaplandığını test eder."""
    motor = NIAHTestMotoru()
    rapor = motor.tam_degerlendirme_yap()

    assert "ortalama_dogruluk" in rapor
    assert "orta_bolge_dogruluk" in rapor
    assert rapor["lost_in_middle_kaybi"] >= 0.0


def test_gorsellestirici_pano_uretme():
    """6 panelli NIAH teşhis panosunun PNG olarak kaydedildiğini test eder."""
    motor = NIAHTestMotoru()
    niah_sonuc = motor.tam_degerlendirme_yap()
    coklu_sonuc = CokluIgneAkilYurutucu.coklu_igne_sentezle("Örnek doküman")

    gorsellestirici = NIAHGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_niah_pano.png")
        gorsellestirici.pano_olustur(niah_sonuc, coklu_sonuc, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
