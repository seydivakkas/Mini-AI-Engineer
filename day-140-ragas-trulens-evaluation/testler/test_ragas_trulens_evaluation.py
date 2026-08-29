"""
FAZ 7 GÜN 140 BÜYÜK FİNALİ: Ragas & TruLens RAG Evaluation Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.sadakat_olcucu import SadakatOlcucu
from src.soru_uygunlugu_olcucu import SoruUygunluguOlcucu
from src.baglam_metrikleri_olcucu import BaglamMetrikleriOlcucu
from src.ragas_trulens_degerlendirici import RagasTruLensDegerlendirici
from src.gorsellestirici import RAGEvaluationGorsellestirici


def test_sadakat_olcucu_tam_sadakat():
    """SadakatOlcucu'nün bağlamla tamamen desteklenen iddialara 1.0 verdiğini test eder."""
    baglam = [
        "Vision Transformer mimarisi Self-Attention kullanır.",
        "FlashAttention GPU SRAM belleğini optimize ederek dikkat hesaplamasını hızlandırır.",
    ]
    yanit = "Vision Transformer mimarisi Self-Attention kullanır. FlashAttention algoritması GPU SRAM belleğini optimize eder."

    sonuc = SadakatOlcucu.olc(yanit, baglam)
    assert sonuc["sadakat_skoru"] == 1.0
    assert sonuc["halusinasyon_orani"] == 0.0
    assert len(sonuc["desteklenen_iddialar"]) == 2


def test_sadakat_olcucu_halusinasyon_tespiti():
    """SadakatOlcucu'nün bağlamda yer almayan uydurma iddiaları (halüsinasyon) tespit ettiğini test eder."""
    baglam = ["Raft konsensüs protokolü lider seçimi yapar."]
    yanit = "Raft protokolü lider seçimi yapar. Mars yüzeyindeki uzay istasyonları kuantum şifreleme kullanır."

    sonuc = SadakatOlcucu.olc(yanit, baglam)
    assert sonuc["sadakat_skoru"] < 1.0
    assert sonuc["halusinasyon_orani"] > 0.0
    assert len(sonuc["halusinasyon_iddialar"]) == 1


def test_soru_uygunlugu_olcucu_yuksek_skor():
    """SoruUygunluguOlcucu'nün soruyla doğrudan alakalı yanıtlara yüksek skor verdiğini test eder."""
    olcer = SoruUygunluguOlcucu()
    soru = "PostgreSQL veritabanında B-Tree indeksleme nasıl çalışır?"
    yanit = "PostgreSQL B-Tree indeksleri logaritmik arama ve aralık sorgularını optimize eder."

    sonuc = olcer.olc(soru, yanit)
    assert sonuc["soru_uygunlugu_skoru"] > 0.70


def test_baglam_metrikleri_recall_ve_precision():
    """BaglamMetrikleriOlcucu'nün Context Recall ve Context Precision değerlerini test eder."""
    getirilen_baglam = [
        "Vision Transformer modelleri görüntü yamalarını işler.",
        "Şirket içi yemek menüsü bugün güncellendi.",  # Alakasız
        "Self-Attention token korelasyonunu hesaplar.",
    ]
    referans_dogrulari = [
        "Vision Transformer görüntü yamalarını işler.",
        "Self-Attention token korelasyonunu hesaplar.",
    ]

    sonuc = BaglamMetrikleriOlcucu.olc(getirilen_baglam, referans_dogrulari)
    assert sonuc["context_recall"] == 1.0
    assert sonuc["context_precision"] > 0.50


def test_ragas_trulens_tekil_degerlendir_triad():
    """RagasTruLensDegerlendirici'nin harmonik RAG Triad skorunu hesapladığını test eder."""
    degerlendirici = RagasTruLensDegerlendirici()
    soru = "Raft protokolü nasıl çalışır?"
    baglam = ["Raft konsensüs protokolü lider seçimi ve log çoğaltması yapar."]
    yanit = "Raft konsensüs protokolü lider seçimi ve log çoğaltması yapar."
    referans = ["Raft lider seçimi ve log çoğaltması yapar."]

    sonuc = degerlendirici.tekil_degerlendir(soru, yanit, baglam, referans)
    assert sonuc["faithfulness"] == 100.0
    assert sonuc["rag_triad_score"] > 80.0
    assert "halusinasyon_orani" in sonuc


def test_faz7_mimarileri_benchmark_karsilastirma():
    """Faz 7 boyunca inşa edilen 4 mimarinin kıyaslama metriklerini test eder."""
    degerlendirici = RagasTruLensDegerlendirici()
    bench = degerlendirici.faz7_mimarilerini_karsilastir()

    assert len(bench["mimariler"]) == 4
    assert len(bench["metrik_adlari"]) == 5
    assert bench["sonuclar"]["hybrid_graphrag"][0] > bench["sonuclar"]["naive_rag"][0]


def test_halusinasyon_orani_dususu_karsilastirma():
    """Advanced Hybrid GraphRAG mimarisinin Naive RAG'e göre halüsinasyonu %1.8'e indirdiğini test eder."""
    degerlendirici = RagasTruLensDegerlendirici()
    bench = degerlendirici.faz7_mimarilerini_karsilastir()

    naive_halus = bench["halusinasyon_oranlari"][0]
    hybrid_halus = bench["halusinasyon_oranlari"][3]
    assert naive_halus == 37.5
    assert hybrid_halus == 1.8


def test_rag_evaluation_gorsellestirici_pano():
    """FAZ 7 büyük finali 6 panelli teşhis panosunun başarıyla üretildiğini test eder."""
    degerlendirici = RagasTruLensDegerlendirici()
    bench = degerlendirici.faz7_mimarilerini_karsilastir()
    tekil = degerlendirici.tekil_degerlendir(
        soru="ViT mimarisi",
        yanit="Vision Transformer Self-Attention kullanır.",
        getirilen_baglam=["Vision Transformer Self-Attention kullanır."],
        referans_dogrulari=["ViT Self-Attention kullanır."],
    )

    gorsellestirici = RAGEvaluationGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_final_pano.png")
        gorsellestirici.pano_olustur(bench, tekil, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
