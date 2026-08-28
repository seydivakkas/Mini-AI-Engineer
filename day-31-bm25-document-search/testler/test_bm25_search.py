"""
Day 31: BM25 Leksikal Belge Arama Motoru Birim Testleri.
"""

import os
import pytest
from src.tokenlestirici import MetinTokenlestirici
from src.ters_indeks import TersIndeks
from src.bm25_motoru import OkapiBM25Motoru
from src.arama_sunucusu import BelgeAramaSunucusu
from src.gorsellestirici import BM25Gorsellestirici


def test_tokenlestirici_normalizasyon():
    """Metin temizliği, noktalama ve stop-words filtreleme testi."""
    tok = MetinTokenlestirici()
    metin = "Derin Öğrenme, Yapay Zeka ve Bilgisayarla Görme!"
    tokenlar = tok.tokenlestir(metin)

    assert "derin" in tokenlar
    assert "ogrenme" in tokenlar or "öğrenme" in tokenlar
    assert "ve" not in tokenlar  # stop-word
    assert len(tokenlar) >= 3


def test_ters_indeks_ekleme_ve_istatistik():
    """Ters indeks ekleme, kelime sıklığı ve ortalama uzunluk testi."""
    indeks = TersIndeks()
    indeks.belge_ekle("DOC-1", "CNN", "Evrisimli aglar", ["cnn", "evrisimli", "aglar"])
    indeks.belge_ekle("DOC-2", "YOLO", "Nesne tespiti aglar", ["yolo", "nesne", "tespiti", "aglar"])

    assert indeks.belge_sayisi == 2
    assert indeks.ortalama_belge_uzunlugu == pytest.approx(3.5, 0.1)
    assert indeks.terim_gecen_belge_sayisi("aglar") == 2
    assert indeks.terim_gecen_belge_sayisi("cnn") == 1
    assert indeks.terim_frekansi("cnn", "DOC-1") == 1


def test_okapi_bm25_idf():
    """IDF hesaplamasının pozitifliği ve nadir terim önceliği testi."""
    indeks = TersIndeks()
    indeks.belge_ekle("D1", "A", "kedi kopek", ["kedi", "kopek"])
    indeks.belge_ekle("D2", "B", "kedi kus", ["kedi", "kus"])
    indeks.belge_ekle("D3", "C", "kedi balik", ["kedi", "balik"])

    bm25 = OkapiBM25Motoru(indeks)
    idf_kedi = bm25.idf_hesapla("kedi")  # Tüm belgelerde var
    idf_balik = bm25.idf_hesapla("balik")  # Sadece 1 belgede var

    assert idf_balik > idf_kedi
    assert idf_kedi >= 0.0


def test_okapi_bm25_sorgu_siralama():
    """BM25 sorgu uygunluk sıralaması testi."""
    indeks = TersIndeks()
    indeks.belge_ekle("D1", "Python", "Python programlama dili", ["python", "programlama", "dili"])
    indeks.belge_ekle("D2", "Java", "Java programlama dili nesne", ["java", "programlama", "dili", "nesne"])
    indeks.belge_ekle("D3", "Rust", "Rust sistem programlama dili", ["rust", "sistem", "programlama", "dili"])

    bm25 = OkapiBM25Motoru(indeks, k1=1.5, b=0.75)
    sonuclar = bm25.sorgula(["python", "programlama"], top_k=2)

    assert len(sonuclar) > 0
    assert sonuclar[0]["doc_id"] == "D1"
    assert "python" in sonuclar[0]["terim_katkilari"]


def test_arama_sunucusu_toplu_arama():
    """Üst seviye BelgeAramaSunucusu entegrasyon testi."""
    sunucu = BelgeAramaSunucusu()
    sunucu.toplu_belge_ekle([
        {"id": "DOC-A", "baslik": "Yapay Zeka", "icerik": "Yapay zeka modelleri gelisiyor"},
        {"id": "DOC-B", "baslik": "Veritabani", "icerik": "SQL veritabani optimizasyonu"}
    ])

    sonuc = sunucu.ara("yapay zeka", top_k=1)
    assert len(sonuc) == 1
    assert sonuc[0]["doc_id"] == "DOC-A"


def test_bm25_gorsellestirici(tmp_path):
    """6 panelli görselleştirici panosu çizim testi."""
    sonuclar = [
        {"doc_id": "D1", "baslik": "Test Dokuman", "skor": 4.5, "terim_katkilari": {"test": 2.5, "dokuman": 2.0}}
    ]
    istatistikler = {"belge_sayisi": 5, "ortalama_uzunluk": 12.0}
    cikis = str(tmp_path / "test_bm25_panel.png")

    yol = BM25Gorsellestirici.arama_panosu_ciz(
        arama_sonuclari=sonuclar,
        sorgu="test dokuman",
        indeks_istatistikleri=istatistikler,
        hedef_path=cikis
    )
    assert os.path.exists(yol)
