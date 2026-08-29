"""
FAZ 7 GÜN 135: Contextual Compression & Dynamic Extraction for RAG Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.baglam_ayristirici import BaglamAyristirici, CumleBirimi
from src.semantik_sikistirici import SemantikBaglamSikistirici
from src.sikistirilmis_rag_getirici import SikistirilmisRAGGetirici
from src.gorsellestirici import ContextualCompressionGorsellestirici


def test_baglam_ayristirici_cumleler():
    """BaglamAyristirici sınıfının metni temiz cümlelere bölüp token tahmini yaptığını test eder."""
    metin = "Raft protokolü konsensüs sağlar. Lider düğüm günlüğü çoğaltır. Ağ bölünmesinde quorum kritiktir."
    cumleler = BaglamAyristirici.ayristir("DOC_1", metin)

    assert len(cumleler) == 3
    assert cumleler[0].doc_id == "DOC_1"
    assert cumleler[0].token_tahmini >= 1
    assert "Raft" in cumleler[0].metin


def test_semantik_sikistirici_puanlama():
    """SemantikBaglamSikistirici'nin cümle bazında kosinüs benzerliği hesapladığını test eder."""
    sikistirici = SemantikBaglamSikistirici(vektor_boyutu=64)
    cumleler = [
        CumleBirimi("D1", 1, "Raft algoritması lider seçimi yapar.", 36, 8),
        CumleBirimi("D1", 2, "Hava bugün çok güneşli ve sıcak.", 32, 7),
    ]

    puanli = sikistirici.cumleleri_puanla("Raft lider seçimi", cumleler)
    assert len(puanli) == 2
    assert puanli[0][1] > puanli[1][1]  # İlgili cümle daha yüksek skor almalı


def test_semantik_sikistirici_budama_ve_tasarruf():
    """Sıkıştırma motorunun alakasız cümleleri elediğini ve token tasarrufu ürettiğini test eder."""
    sikistirici = SemantikBaglamSikistirici(vektor_boyutu=64, esik_skoru=0.25)
    belgeler = [
        {
            "doc_id": "D1",
            "metin": (
                "PostgreSQL B-Tree indeksleme sorgu optimizasyonu sağlar. "
                "Öğle yemeğinde makarna yemek oldukça lezzetlidir. "
                "Ters indeksler tam metin aramalarında kullanılır."
            ),
        }
    ]

    sonuc = sikistirici.sikistir("PostgreSQL veritabanı indeksleri", belgeler)

    assert sonuc["toplam_cumle_sayisi"] == 3
    assert sonuc["secilen_cumle_sayisi"] < 3
    assert sonuc["elenen_cumle_sayisi"] >= 1
    assert sonuc["token_tasarrufu_yuzde"] > 0.0


def test_sikistirici_fallback_guvencesi():
    """Tüm cümleler eşik altında kalsa bile fallback olarak en iyi cümlenin korunduğunu test eder."""
    sikistirici = SemantikBaglamSikistirici(vektor_boyutu=64, esik_skoru=0.99)
    belgeler = [{"doc_id": "D1", "metin": "Birinci cümle veri. İkinci cümle bilgi."}]

    sonuc = sikistirici.sikistir("Kuantum Fiziği", belgeler)
    assert sonuc["secilen_cumle_sayisi"] >= 1


def test_sikistirilmis_rag_getirici_belge_ekle():
    """SikistirilmisRAGGetirici sınıfının belgeleri ekleyip ham vektör araması yaptığını test eder."""
    getirici = SikistirilmisRAGGetirici(vektor_boyutu=64)
    getirici.toplu_belge_ekle([
        {"doc_id": "D1", "metin": "Transformer öz-dikkat mekanizması ve QKV matrisleri."},
        {"doc_id": "D2", "metin": "Biyokimya ve hücre metabolizması döngüsü."},
    ])

    ham = getirici.ham_getir("Transformer dikkat mekanizması", top_k=1)
    assert len(ham) == 1
    assert ham[0]["doc_id"] == "D1"


def test_sorgula_ve_sikistir_entegrasyon():
    """Uçtan uca getirme ve bağlam sıkıştırma zincirinin çalıştığını test eder."""
    getirici = SikistirilmisRAGGetirici(vektor_boyutu=64, esik_skoru=0.20)
    getirici.belge_ekle(
        "DOC_01",
        "Raft konsensüs protokolü lider seçimi yürütür. Ağ bölünmesinde quorum şarttır. Yarın hava yağmurlu olacak."
    )

    sonuc = getirici.sorgula_ve_sikistir("Raft konsensüs quorum", top_k=1)

    assert sonuc["getirilen_ham_belge_sayisi"] == 1
    assert "islem_suresi_ms" in sonuc
    assert "[KAYNAK: DOC_01]" in sonuc["nihai_baglam"]


def test_benchmark_karsilastir_metrikleri():
    """Benchmark karşılaştırma metriklerinin eksiksiz olduğunu test eder."""
    getirici = SikistirilmisRAGGetirici()
    bench = getirici.benchmark_karsilastir()

    assert len(bench["metrikler"]) == 4
    assert bench["contextual_compression"][0] > bench["ham_baglam_rag"][0]


def test_contextual_compression_gorsellestirici_pano():
    """ContextualCompressionGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    getirici = SikistirilmisRAGGetirici(vektor_boyutu=64)
    getirici.belge_ekle("D1", "Derin öğrenme ve tensör optimizasyonu. Bahçede çiçekler açtı.")
    sonuc = getirici.sorgula_ve_sikistir("Derin öğrenme", top_k=1)
    bench = getirici.benchmark_karsilastir()

    gorsellestirici = ContextualCompressionGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_compression_pano.png")
        gorsellestirici.pano_olustur(sonuc, bench, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
