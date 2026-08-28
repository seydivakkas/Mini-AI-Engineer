"""
Day 32: Yoğun Vektör Tabanlı Semantik Arama Motoru Birim Testleri.
"""

import os
import numpy as np
import pytest
from src.vektorlestirici import CumleVektorlestirici
from src.vektor_indeksi import DuzVektorIndeksi
from src.semantik_arama_motoru import SemantikAramaMotoru
from src.gorsellestirici import SemantikAramaGorsellestirici


def test_vektorlestirici_boyut_ve_norm():
    """Vektörleştirici tensör boyutu ve L2 birim norm testi."""
    vek = CumleVektorlestirici(embed_dim=128)
    metinler = ["Yapay zeka modelleri", "Derin öğrenme ve bilgisayarla görme"]
    embs = vek.vektorlestir(metinler)

    assert embs.shape == (2, 128)
    norm1 = np.linalg.norm(embs[0])
    norm2 = np.linalg.norm(embs[1])
    assert norm1 == pytest.approx(1.0, 1e-4)
    assert norm2 == pytest.approx(1.0, 1e-4)


def test_duz_vektor_indeksi_ekleme_ve_ara():
    """Düz vektör indeksine ekleme ve kosinüs benzerliği sıralama testi."""
    indeks = DuzVektorIndeksi(boyut=4)
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float32)

    indeks.ekle("D1", v1, {"baslik": "Belge 1", "kategori": "A"})
    indeks.ekle("D2", v2, {"baslik": "Belge 2", "kategori": "B"})
    indeks.ekle("D3", v3, {"baslik": "Belge 3", "kategori": "A"})

    assert indeks.toplam_vektor_sayisi == 3

    # v1 sorgusu ile D1 tam eşleşmeli (skor ~ 1.0), D3 ikinci (~ 0.707), D2 üçüncü (~ 0.0)
    sonuclar = indeks.en_yakin_komsu_ara(v1, top_k=3)
    assert len(sonuclar) == 3
    assert sonuclar[0]["doc_id"] == "D1"
    assert sonuclar[0]["skor"] == pytest.approx(1.0, 1e-3)
    assert sonuclar[1]["doc_id"] == "D3"


def test_kategori_filtreleme():
    """Vektör aramasında metaveri kategori filtresi testi."""
    indeks = DuzVektorIndeksi(boyut=4)
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    indeks.ekle("D1", v1, {"baslik": "Doc1", "kategori": "Saglik"})
    indeks.ekle("D2", v1, {"baslik": "Doc2", "kategori": "Finans"})

    sonuclar = indeks.en_yakin_komsu_ara(v1, top_k=2, filtre_kategori="Finans")
    assert len(sonuclar) == 1
    assert sonuclar[0]["doc_id"] == "D2"
    assert sonuclar[0]["kategori"] == "Finans"


def test_semantik_arama_motoru_entegrasyon():
    """Üst seviye semantik arama motoru uçtan uca sorgu testi."""
    motor = SemantikAramaMotoru(embed_dim=64)
    motor.toplu_dokuman_ekle([
        {"id": "D1", "baslik": "Görüntü İşleme", "icerik": "Piksel filtreleri ve kenar bulma", "kategori": "CV"},
        {"id": "D2", "baslik": "Doğal Dil", "icerik": "Metin analizi ve dil modelleri", "kategori": "NLP"}
    ])

    sonuclar = motor.semantik_ara("metin analizi", top_k=1)
    assert len(sonuclar) == 1
    assert sonuclar[0]["doc_id"] == "D2"


def test_pca_temsil_uzayi():
    """PCA 2D indirgeme boyutu testi."""
    motor = SemantikAramaMotoru(embed_dim=32)
    for i in range(5):
        motor.dokuman_ekle(f"D{i}", f"Baslik {i}", f"Icerik {i} detay metni")

    pca_2d, doc_ids, kategoriler = motor.temsil_uzayi_pca_projeksiyonu()
    assert pca_2d.shape == (5, 2)
    assert len(doc_ids) == 5
    assert len(kategoriler) == 5


def test_semantik_gorsellestirici(tmp_path):
    """6 panelli semantik görselleştirici panosu çizim testi."""
    sonuclar = [{"doc_id": "D1", "baslik": "Test Dokuman", "skor": 0.92, "icerik": "Test icerik"}]
    pca_2d = np.array([[0.1, 0.2], [-0.3, 0.4]])
    doc_ids = ["D1", "D2"]
    kategoriler = ["A", "B"]
    capraz_matris = np.array([[0.9, 0.3], [0.2, 0.8]])
    capraz_sorgular = ["Sorgu 1", "Sorgu 2"]

    cikis_path = str(tmp_path / "test_semantik.png")
    cizim = SemantikAramaGorsellestirici.semantik_panel_ciz(
        arama_sonuclari=sonuclar,
        sorgu="test sorgu",
        pca_2d=pca_2d,
        doc_idler=doc_ids,
        kategoriler=kategoriler,
        capraz_benzerlik_matrisi=capraz_matris,
        capraz_sorgular=capraz_sorgular,
        hedef_path=cikis_path
    )
    assert os.path.exists(cizim)
