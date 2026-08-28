"""
Day 33: Hibrit Arama & Reciprocal Rank Fusion (RRF) Birim Testleri.
"""

import os
import pytest
from src.leksikal_motor import LeksikalBM25Motoru
from src.semantik_motor import SemantikVektorMotoru
from src.rrf_fuzor import RRFFuzor, PuanNormalizasyonFuzor
from src.hibrit_arama_yoneticisi import HibritAramaYoneticisi
from src.gorsellestirici import HibritAramaGorsellestirici


def test_leksikal_motor():
    """BM25 leksikal arama bileşeni testi."""
    bm25 = LeksikalBM25Motoru()
    bm25.dokuman_ekle("D1", "Python Programlama", "Python veri analizi ve makine ogrenmesi")
    bm25.dokuman_ekle("D2", "Veritabani", "SQL ve NoSQL veritabani sistemleri")

    res = bm25.ara("python makine", top_k=2)
    assert len(res) >= 1
    assert res[0]["doc_id"] == "D1"


def test_semantik_motor():
    """Dense semantik arama bileşeni testi."""
    sem = SemantikVektorMotoru(embed_dim=64)
    sem.dokuman_ekle("D1", "Derin Aglar", "Yapay sinir aglari ve geriye yayilim")
    sem.dokuman_ekle("D2", "Ag Guvenligi", "Siber saldirilar ve firewall kurallari")

    res = sem.ara("sinir aglari", top_k=2)
    assert len(res) >= 1
    assert res[0]["doc_id"] == "D1"


def test_rrf_fuzor_mantigi():
    """RRF formülü ve sıralama füzyon testi."""
    fuzor = RRFFuzor(k=60)
    bm25_list = [{"doc_id": "D1", "skor": 10.5, "baslik": "B1"}, {"doc_id": "D2", "skor": 5.0, "baslik": "B2"}]
    sem_list = [{"doc_id": "D2", "skor": 0.95, "baslik": "B2"}, {"doc_id": "D1", "skor": 0.80, "baslik": "B1"}]

    # D1: rank1(bm25) + rank2(sem) -> 0.5/61 + 0.5/62 = 0.008196 + 0.008064 = 0.01626
    # D2: rank2(bm25) + rank1(sem) -> 0.5/62 + 0.5/61 = 0.01626 (eşit)
    res = fuzor.birlestir({"bm25": bm25_list, "semantik": sem_list}, agirliklar={"bm25": 0.5, "semantik": 0.5}, top_k=2)
    assert len(res) == 2
    assert res[0]["skor"] == pytest.approx(res[1]["skor"], 1e-5)


def test_puan_normalizasyon_fuzor():
    """Min-Max skor normalizasyon füzyonu testi."""
    fuzor = PuanNormalizasyonFuzor()
    bm25_list = [{"doc_id": "D1", "skor": 10.0, "baslik": "B1"}, {"doc_id": "D2", "skor": 0.0, "baslik": "B2"}]
    sem_list = [{"doc_id": "D1", "skor": 1.0, "baslik": "B1"}, {"doc_id": "D2", "skor": 0.0, "baslik": "B2"}]

    res = fuzor.normalize_ve_birlestir({"bm25": bm25_list, "semantik": sem_list}, top_k=2)
    assert len(res) == 2
    assert res[0]["doc_id"] == "D1"
    assert res[0]["skor"] == pytest.approx(1.0, 1e-4)


def test_hibrit_arama_yoneticisi_entegrasyon():
    """Uçtan uca HibritAramaYoneticisi entegrasyon testi."""
    yonetici = HibritAramaYoneticisi(rrf_k=60, embed_dim=64)
    yonetici.toplu_dokuman_ekle([
        {"id": "D1", "baslik": "RAG Asistani", "icerik": "Vektor indeksleme ile soru cevaplama", "kategori": "NLP"},
        {"id": "D2", "baslik": "YOLO Dedektor", "icerik": "Nesne tespiti ve kutu regresyonu", "kategori": "CV"}
    ])

    cikis = yonetici.hibrit_ara("RAG ve vektor", top_k=1, fuzyon_yontemi="rrf")
    assert "final_sonuclar" in cikis
    assert len(cikis["final_sonuclar"]) == 1
    assert cikis["final_sonuclar"][0]["doc_id"] == "D1"


def test_hibrit_gorsellestirici(tmp_path):
    """6 panelli hibrit teşhis panosu çizim testi."""
    mock_cikis = {
        "sorgu": "test sorgu",
        "final_sonuclar": [
            {"doc_id": "D1", "baslik": "Test Doc 1", "skor": 0.016, "siralama_gecmisi": {"bm25": 1, "semantik": 2}}
        ],
        "bm25_sonuclari": [{"doc_id": "D1", "skor": 5.0}],
        "semantik_sonuclari": [{"doc_id": "D1", "skor": 0.9}]
    }
    cikis_path = str(tmp_path / "test_hibrit.png")
    yol = HibritAramaGorsellestirici.hibrit_panel_ciz(mock_cikis, hedef_path=cikis_path)
    assert os.path.exists(yol)
