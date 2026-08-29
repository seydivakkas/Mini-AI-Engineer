"""
FAZ 7 GÜN 139: Hybrid Vector + Graph RAG Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.vektor_getirici import VektorGetirici
from src.graf_getirici import GrafGetirici
from src.rrf_birlestirici import RRFBirlestirici
from src.hibrit_rag_yoneticisi import HibritRAGYoneticisi
from src.gorsellestirici import HybridRAGGorsellestirici


@pytest.fixture
def ornek_hibrit_veritabani():
    """Vektör ve Graf getirme testleri için örnek veri tabanı hazırlar."""
    belgeler = [
        {
            "id": "DOC_01_VIT",
            "metin": "Vision Transformer mimarisi Self-Attention mekanizmasını kullanır ve dikkat uygular.",
            "varliklar": ["Vision Transformer", "Self-Attention"],
        },
        {
            "id": "DOC_02_FLASH",
            "metin": "FlashAttention algoritması GPU SRAM bellek optimizasyonu ile Self-Attention'ı hızlandırır.",
            "varliklar": ["FlashAttention", "Self-Attention", "NVIDIA GPU"],
        },
        {
            "id": "DOC_03_RAFT",
            "metin": "Raft konsensüs protokolü Quorum kuralını uygular ve lider seçimi yapar.",
            "varliklar": ["Raft", "Quorum"],
        },
    ]
    graf_kenarlari = [
        {"ozne": "Vision Transformer", "yuklem": "KULLANIR", "nesne": "Self-Attention"},
        {"ozne": "Self-Attention", "yuklem": "HIZLANDIRIR", "nesne": "FlashAttention"},
        {"ozne": "FlashAttention", "yuklem": "CALISIR", "nesne": "NVIDIA GPU"},
        {"ozne": "Raft", "yuklem": "UYGULAR", "nesne": "Quorum"},
    ]
    return belgeler, graf_kenarlari


def test_vektor_getirici_indeks_ve_arama(ornek_hibrit_veritabani):
    """VektorGetirici sınıfının indeksleme ve kosinüs benzerliği sıralamasını test eder."""
    belgeler, _ = ornek_hibrit_veritabani
    getirici = VektorGetirici()
    getirici.indeksle(belgeler)

    sonuclar = getirici.ara("Vision Transformer dikkat modeli", top_k=2)
    assert len(sonuclar) == 2
    assert sonuclar[0]["id"] == "DOC_01_VIT"
    assert "vektor_skoru" in sonuclar[0]
    assert sonuclar[0]["vektor_sirasi"] == 1


def test_graf_getirici_varlik_ve_baglanti_aramasi(ornek_hibrit_veritabani):
    """GrafGetirici sınıfının tohum varlık ve 1-2 hop komşuluk aramasını test eder."""
    belgeler, graf_kenarlari = ornek_hibrit_veritabani
    belgeler_haritasi = {b["id"]: b for b in belgeler}

    getirici = GrafGetirici()
    getirici.indeksle(belgeler, graf_kenarlari)

    sonuclar = getirici.ara("Raft konsensüs", belgeler_haritasi, top_k=2)
    assert len(sonuclar) >= 1
    assert sonuclar[0]["id"] == "DOC_03_RAFT"
    assert sonuclar[0]["graf_sirasi"] == 1


def test_rrf_birlestirici_skorlama():
    """RRFBirlestirici'nin Reciprocal Rank Fusion skorlarını doğru hesapladığını test eder."""
    v_sonuclar = [
        {"id": "DOC_A", "vektor_sirasi": 1, "vektor_skoru": 0.95},
        {"id": "DOC_B", "vektor_sirasi": 2, "vektor_skoru": 0.85},
    ]
    g_sonuclar = [
        {"id": "DOC_B", "graf_sirasi": 1, "graf_skoru": 2.0},
        {"id": "DOC_A", "graf_sirasi": 2, "graf_skoru": 1.0},
    ]

    hibrit = RRFBirlestirici.birlestir(v_sonuclar, g_sonuclar, w_vec=0.5, w_graph=0.5, k_rrf=60)
    assert len(hibrit) == 2
    assert "rrf_skoru" in hibrit[0]
    assert hibrit[0]["nihai_sira"] == 1


def test_rrf_siralama_kaymasi_hesaplama():
    """RRF sonucunda sıralama kaymasının (rank shift) pozitif/negatif değer alabildiğini test eder."""
    v_sonuclar = [{"id": "DOC_X", "vektor_sirasi": 3, "vektor_skoru": 0.5}]
    g_sonuclar = [{"id": "DOC_X", "graf_sirasi": 1, "graf_skoru": 3.0}]

    hibrit = RRFBirlestirici.birlestir(v_sonuclar, g_sonuclar, w_vec=0.2, w_graph=0.8, k_rrf=60)
    assert hibrit[0]["nihai_sira"] == 1
    # 3. sıradan 1. sıraya yükseldi -> kayma = 3 - 1 = +2
    assert hibrit[0]["siralama_kaymasi"] == 2


def test_hibrit_yonetici_sorgu_tipi_yonlendirme():
    """HibritRAGYoneticisi'nin sorgu tipine göre dinamik ağırlık atadığını test eder."""
    yonetici = HibritRAGYoneticisi()

    tip1, wv1, wg1 = yonetici.sorgu_tipini_belirle("ViT dikkat mekanizmasını nasıl hızlandırır?")
    assert tip1 == "İLİŞKİSEL_COKLU_ATLAMA"
    assert wg1 > wv1

    tip2, wv2, wg2 = yonetici.sorgu_tipini_belirle("Transformer mimarisi genel kavramsal olarak nedir ve tanımla?")
    assert tip2 == "ANLAMSAL_KAVRAMSAL"
    assert wv2 > wg2


def test_hibrit_rag_uc_tan_uca_arama(ornek_hibrit_veritabani):
    """HibritRAGYoneticisi sınıfının uçtan uca hibrit arama icra ettiğini test eder."""
    belgeler, graf_kenarlari = ornek_hibrit_veritabani
    yonetici = HibritRAGYoneticisi()
    yonetici.indeksle(belgeler, graf_kenarlari)

    sonuc = yonetici.ara("Vision Transformer hızlandırma mekanizması", top_k=3)
    assert len(sonuc["hibrit_sonuclar"]) >= 2
    assert "getirme_suresi_ms" in sonuc
    assert "agirliklar" in sonuc


def test_hibrit_rag_benchmark_metrikleri():
    """Benchmark metriklerinin eksiksiz olduğunu ve hibrit yaklaşımın üstünlüğünü test eder."""
    yonetici = HibritRAGYoneticisi()
    bench = yonetici.benchmark_karsilastir()

    assert len(bench["metrikler"]) == 4
    assert bench["hibrit_rrf_rag"][0] > bench["saf_vektor"][0]
    assert bench["hibrit_rrf_rag"][0] > bench["saf_graf"][0]


def test_hybrid_rag_gorsellestirici_pano(ornek_hibrit_veritabani):
    """Teşhis panosu görselleştiricisinin 6 panelli PNG dosyasını ürettiğini test eder."""
    belgeler, graf_kenarlari = ornek_hibrit_veritabani
    yonetici = HibritRAGYoneticisi()
    yonetici.indeksle(belgeler, graf_kenarlari)

    sonuc = yonetici.ara("Vision Transformer")
    bench = yonetici.benchmark_karsilastir()

    gorsellestirici = HybridRAGGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_hybrid_pano.png")
        gorsellestirici.pano_olustur(bench, sonuc, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
