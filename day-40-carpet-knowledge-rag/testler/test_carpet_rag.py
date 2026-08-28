"""
Day 40: Tekstil ve Üretim Teknik Dokümanları RAG Sistemi Birim Testleri.
"""

import os
import pytest
from src.sektor_korpusu import TEKSTIL_TEKNIK_KORPUS
from src.semantik_parcalayici import SemantikMetinParcalayici
from src.vektor_deposu import TekstilVektorDeposu
from src.rag_asistani import SektorelRAGAsistani
from src.gorsellestirici import SektorelRAGGorsellestirici


def test_sektor_korpusu_yukleme():
    """Korpusun geçerli teknik dokümanlar içerdiğini doğrular."""
    assert len(TEKSTIL_TEKNIK_KORPUS) >= 5
    for dok in TEKSTIL_TEKNIK_KORPUS:
        assert "dokuman_id" in dok
        assert "metin" in dok
        assert "kaynak_standart" in dok
        assert len(dok["metin"].strip()) > 50


def test_semantik_parcalayici_overlap():
    """Semantik parçalayıcının alt başlıkları koruduğunu doğrular."""
    parcalayici = SemantikMetinParcalayici(max_karakter=300, overlap_karakter=50)
    chunklar = parcalayici.korpus_parcala(TEKSTIL_TEKNIK_KORPUS)

    assert len(chunklar) >= len(TEKSTIL_TEKNIK_KORPUS)
    for ch in chunklar:
        assert "alt_baslik" in ch
        assert "dokuman_id" in ch
        assert ch["karakter_uzunlugu"] <= 350


def test_vektor_deposu_indeksleme():
    """Vektör deposunun sözlük oluşturup indeksleme yaptığını doğrular."""
    parcalayici = SemantikMetinParcalayici()
    chunklar = parcalayici.korpus_parcala(TEKSTIL_TEKNIK_KORPUS)

    depo = TekstilVektorDeposu()
    depo.indeksle(chunklar)

    assert len(depo.kelime_sozlugu) > 50
    assert depo.tfidf_matrisi.shape[0] == len(chunklar)


def test_vektor_deposu_kategori_filtresi():
    """Metadata kategori filtresinin doğru çalıştığını test eder."""
    parcalayici = SemantikMetinParcalayici()
    chunklar = parcalayici.korpus_parcala(TEKSTIL_TEKNIK_KORPUS)

    depo = TekstilVektorDeposu()
    depo.indeksle(chunklar)

    sonuclar = depo.sorgula("büküm iplik", top_k=5, kategori_filtresi="iplik_standardi")
    for s in sonuclar:
        assert s["chunk"]["kategori"] == "iplik_standardi"


def test_rag_asistani_dogrulanmis_yanit():
    """Kurutma sıcaklığı sorusunda doğru standardın alıntılandığını test eder."""
    parcalayici = SemantikMetinParcalayici()
    chunklar = parcalayici.korpus_parcala(TEKSTIL_TEKNIK_KORPUS)

    depo = TekstilVektorDeposu()
    depo.indeksle(chunklar)

    asistan = SektorelRAGAsistani(depo, guven_esigi=0.15)
    sonuc = asistan.yanit_uret("apre kurutma fiksaj sıcaklığı kaç derecedir?", top_k=2)

    assert sonuc["durum"] == "BASARILI_YANIT"
    assert "145°C - 155°C" in sonuc["yanit"] or "Kurutma ve Fiksaj" in sonuc["yanit"]
    assert len(sonuc["kaynaklar"]) > 0


def test_rag_asistani_reddetme_korumasi():
    """Alan dışı ilgisiz soruların güven eşiği altında kalarak reddedildiğini test eder."""
    parcalayici = SemantikMetinParcalayici()
    chunklar = parcalayici.korpus_parcala(TEKSTIL_TEKNIK_KORPUS)

    depo = TekstilVektorDeposu()
    depo.indeksle(chunklar)

    asistan = SektorelRAGAsistani(depo, guven_esigi=0.50)
    sonuc = asistan.yanit_uret("mars gezegenine roket fırlatma yakıtı nedir?", top_k=2)

    assert sonuc["durum"] == "REDDEDILDI_BILGI_YOK"
    assert "bulunmamaktadır" in sonuc["yanit"]


def test_sektorel_rag_gorsellestirici(tmp_path):
    """6 panelli görselleştiricinin çizim dosyası ürettiğini test eder."""
    parcalayici = SemantikMetinParcalayici()
    chunklar = parcalayici.korpus_parcala(TEKSTIL_TEKNIK_KORPUS)

    mock_sonuc = {
        "soru": "Test Soru",
        "durum": "BASARILI_YANIT",
        "yanit": "Test Yanıt",
        "en_yuksek_skor": 0.85,
        "kaynaklar": [{
            "dokuman_id": "DOC-1", "ana_baslik": "Başlık", "alt_baslik": "Alt",
            "kaynak_standart": "ISO", "kategori": "iplik_standardi", "skor": 0.85
        }]
    }

    cikis_path = str(tmp_path / "test_rag_panel.png")
    yol = SektorelRAGGorsellestirici.rag_paneli_ciz(mock_sonuc, chunklar, hedef_path=cikis_path)
    assert os.path.exists(yol)
