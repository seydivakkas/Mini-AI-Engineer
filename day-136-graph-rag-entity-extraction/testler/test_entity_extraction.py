"""
FAZ 7 GÜN 136: GraphRAG-1 Entity & Relationship Extraction Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.varlik_cikarici import Varlik, VarlikCikarici
from src.iliski_cikarici import IliskiUclusu, IliskiCikarici
from src.varlik_cozumleyici import VarlikCozumleyici
from src.bilgi_grafi_olusturucu import BilgiGrafiOlusturucu
from src.gorsellestirici import GraphRAGGorsellestirici


def test_varlik_cikarici_katalog():
    """VarlikCikarici'nin katalogdaki teknik varlıkları ve tiplerini başarıyla bulduğunu test eder."""
    metin = "Raft algoritması lider seçimi yapar. PostgreSQL ilişkisel veritabanı B-Tree indeksi kullanır."
    varliklar = VarlikCikarici.cikar(metin)

    isimler = [v.isim for v in varliklar]
    assert "Raft" in isimler
    assert "PostgreSQL" in isimler
    assert "B-Tree" in isimler


def test_varlik_cikarici_dinamik_kaliplar():
    """VarlikCikarici'nin büyük harfli yeni teknik terimleri yakaladığını test eder."""
    metin = "Modern sistemlerde Quantum Annealing işlemcileri kullanılır."
    varliklar = VarlikCikarici.cikar(metin)

    isimler = [v.isim for v in varliklar]
    assert any("Quantum" in name for name in isimler)


def test_iliski_cikarici_ucluler():
    """IliskiCikarici'nin varlıklar arası Özne-Yüklem-Nesne üçlülerini doğru kurduğunu test eder."""
    metin = "Raft protokolü Quorum kuralını uygular ve veri bölünmesini engeller."
    varliklar = [
        Varlik(isim="Raft", tip="ALGORITMA"),
        Varlik(isim="Quorum", tip="KAVRAM"),
    ]

    iliskiler = IliskiCikarici.cikar(metin, varliklar)
    assert len(iliskiler) >= 1
    assert iliskiler[0].ozne == "Raft"
    assert iliskiler[0].nesne == "Quorum"


def test_varlik_cozumleyici_kanonik_ad():
    """VarlikCozumleyici'nin eşanlamlıları ve kısaltmaları standart kanonik ada çevirdiğini test eder."""
    assert VarlikCozumleyici.kanonik_ad("ViT") == "Vision Transformer"
    assert VarlikCozumleyici.kanonik_ad("postgres") == "PostgreSQL"
    assert VarlikCozumleyici.kanonik_ad("LOB") == "Limit Order Book"


def test_varlik_cozumleyici_graf_guncelleme():
    """VarlikCozumleyici'nin varlıkları tekilleştirip ilişkileri kanonik isimlerle güncellediğini test eder."""
    varliklar = [
        Varlik(isim="ViT", tip="TEKNOLOJI"),
        Varlik(isim="Vision Transformer", tip="TEKNOLOJI"),
        Varlik(isim="Self-Attention", tip="ALGORITMA"),
    ]
    iliskiler = [
        IliskiUclusu(ozne="ViT", yuklem="KULLANIR", nesne="Self-Attention"),
    ]

    tekil_v, guncel_i, _ = VarlikCozumleyici.cozumle(varliklar, iliskiler)

    v_isimler = [v.isim for v in tekil_v]
    assert len(tekil_v) == 2
    assert "Vision Transformer" in v_isimler
    assert guncel_i[0].ozne == "Vision Transformer"


def test_bilgi_grafi_olusturucu_pipeline():
    """BilgiGrafiOlusturucu sınıfının uçtan uca düğümleri, kenarları ve derece merkeziliğini ürettiğini test eder."""
    metin = (
        "Raft konsensüs protokolü Quorum kuralını uygular. "
        "Vision Transformer modelleri Self-Attention mekanizmasını kullanır. "
        "PostgreSQL veritabanı B-Tree indeksleme desteği sunar."
    )

    olusturucu = BilgiGrafiOlusturucu()
    sonuc = olusturucu.metinden_graf_olustur(metin)

    assert sonuc["toplam_dugum_sayisi"] >= 4
    assert sonuc["toplam_kenar_sayisi"] >= 2
    assert len(sonuc["dugumler"]) == sonuc["toplam_dugum_sayisi"]
    assert "cikarim_suresi_ms" in sonuc


def test_benchmark_karsilastir_metrikleri():
    """GraphRAG-1 karşılaştırma metriklerinin eksiksiz olduğunu test eder."""
    olusturucu = BilgiGrafiOlusturucu()
    bench = olusturucu.benchmark_karsilastir()

    assert len(bench["metrikler"]) == 4
    assert bench["graphrag_entity_extraction"][0] > bench["standart_regex_ner"][0]


def test_graphrag_gorsellestirici_pano():
    """GraphRAGGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    olusturucu = BilgiGrafiOlusturucu()
    metin = "Raft algoritması Quorum kuralı ile çalışır. ViT mimarisi Self-Attention kullanır."
    sonuc = olusturucu.metinden_graf_olustur(metin)
    bench = olusturucu.benchmark_karsilastir()

    gorsellestirici = GraphRAGGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_graphrag_pano.png")
        gorsellestirici.pano_olustur(sonuc, bench, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
