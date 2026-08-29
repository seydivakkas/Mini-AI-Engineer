"""
GÜN 165: OCR-Free Doküman ve Tablo Anlama Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.dokuman_metrik_degerlendirici import DokumanMetrikDegerlendirici
from src.dokuman_veri_kumesi import DokumanVeriKumesi
from src.donut_nougat_ayristirici import DonutNougatAyristirici
from src.gorsellestirici import DokumanGorsellestirici


def test_levenshtein_mesafesi_tam_eslesme():
    """Birebir aynı iki metinde mesafenin 0 olduğunu test eder."""
    s1 = r"\int e^x dx = e^x + C"
    s2 = r"\int e^x dx = e^x + C"
    assert DokumanMetrikDegerlendirici.levenshtein_mesafesi(s1, s2) == 0


def test_levenshtein_mesafesi_farkli_metin():
    """Karakter ekleme/değiştirme mesafesini doğru hesapladığını test eder."""
    assert DokumanMetrikDegerlendirici.levenshtein_mesafesi("kitten", "sitting") == 3


def test_normalized_edit_similarity_tam_benzerlik():
    """Tam eşleşmede Edit Similarity'nin 1.0 olduğunu test eder."""
    sim = DokumanMetrikDegerlendirici.normalized_edit_similarity("latex_formula", "latex_formula")
    assert sim == 1.0


def test_normalized_edit_similarity_tam_farklilik():
    """Tamamen farklı karakterlerde benzerliğin 0.0 olduğunu test eder."""
    sim = DokumanMetrikDegerlendirici.normalized_edit_similarity("abc", "xyz")
    assert sim == 0.0


def test_dokuman_veri_kumesi_senaryolar():
    """Veri setinin LaTeX formülleri, tablolar ve JSON fatura içerdiğini test eder."""
    senaryolar = DokumanVeriKumesi.senaryolari_getir()
    assert len(senaryolar) == 4
    tipler = [s["dokuman_tipi"] for s in senaryolar]
    assert "Akademik Formül (LaTeX)" in tipler
    assert "Markdown Tablosu" in tipler
    assert "Yapılandırılmış Fatura (JSON)" in tipler


def test_donut_nougat_ayristirici_degerlendirmesi():
    """Donut/Nougat motorunun senaryoları kusursuz ayrıştırdığını test eder."""
    rapor = DonutNougatAyristirici.dokumanlari_ayristir_ve_degerlendir()

    assert len(rapor["senaryo_sonuclari"]) == 4
    assert rapor["genel_ozet"]["ortalama_dogruluk_yuzdesi"] == 100.0
    assert rapor["genel_ozet"]["ortalama_edit_similarity"] == 1.0


def test_toplu_degerlendir_bos_liste():
    """Boş liste verildiğinde sıfır/boş skor döndüğünü test eder."""
    ozet = DokumanMetrikDegerlendirici.toplu_degerlendir([], [])
    assert ozet["toplam_dokuman"] == 0
    assert ozet["ortalama_edit_similarity"] == 0.0


def test_gorsellestirici_pano_uretme():
    """6 panelli OCR-Free doküman teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = DonutNougatAyristirici.dokumanlari_ayristir_ve_degerlendir()
    gorsellestirici = DokumanGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_doc_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
