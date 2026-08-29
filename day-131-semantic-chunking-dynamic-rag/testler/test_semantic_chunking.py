"""
FAZ 7 GÜN 131: Semantic Chunking ve Dinamik RAG Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.cumle_ayristirici import CumleAyristirici, BaglamTamponlayici
from src.semantik_parcalayici import SemantikParcalayici
from src.rag_karsilastirici import RAGParcalamaKarsilastirici
from src.gorsellestirici import SemanticChunkingGorsellestirici


def test_cumle_ayristirici():
    """CumleAyristirici'nin metni noktalama işaretlerinden temiz cümlelere ayırdığını test eder."""
    metin = "Yapay zeka modelleri hızla gelişiyor. Özellikle Transformer mimarisi devrim yarattı! Peki sırada ne var? GraphRAG sistemleri."
    cumleler = CumleAyristirici.ayristir(metin)

    assert len(cumleler) == 4
    assert cumleler[0] == "Yapay zeka modelleri hızla gelişiyor."
    assert "GraphRAG" in cumleler[3]


def test_baglam_tamponlayici():
    """BaglamTamponlayici'nın kayan pencere ile komşu cümleleri birleştirdiğini test eder."""
    cumleler = ["Cümle bir.", "Cümle iki.", "Cümle üç."]
    tamponlu = BaglamTamponlayici.tampon_olustur(cumleler, tampon_boyutu=1)

    assert len(tamponlu) == 3
    assert "Cümle bir." in tamponlu[1]["birlestirilmis_baglam"]
    assert "Cümle üç." in tamponlu[1]["birlestirilmis_baglam"]


def test_metin_vektorlestir_ve_kosinus_mesafesi():
    """Semantik vektörlerin L2 normalize olduğunu ve kosinüs mesafelerinin [0, 2] aralığında çıktığını test eder."""
    parcalayici = SemantikParcalayici(vektor_boyutu=64)
    metinler = ["Derin öğrenme ve PyTorch tensörleri.", "Doğal dil işleme ve Transformerlar."]

    vektorler = parcalayici._metin_vektorlestir(metinler)
    assert vektorler.shape == (2, 64)

    mesafeler = parcalayici.kosinus_mesafesi_hesapla(vektorler)
    assert len(mesafeler) == 1
    assert 0.0 <= mesafeler[0] <= 2.0


def test_esik_degeri_belirleme():
    """Eşik belirleyicinin hem standart sapma hem yüzdelik dilim modunda geçerli değer ürettiğini test eder."""
    parcalayici_std = SemantikParcalayici(esik_yontemi="standart_sapma", esik_katsayisi=0.5)
    parcalayici_pct = SemantikParcalayici(esik_yontemi="yuzdelik_dilim", yuzdelik_dilim=75.0)

    mesafeler = [0.1, 0.2, 0.15, 0.85, 0.9, 0.2]

    esik1 = parcalayici_std.esik_degeri_belirle(mesafeler)
    esik2 = parcalayici_pct.esik_degeri_belirle(mesafeler)

    assert esik1 > 0.3
    assert esik2 > 0.3


def test_semantik_parcalama_cok_konulu_metin():
    """Farklı konulardan oluşan metnin anlamsal geçiş noktalarından başarıyla parçalandığını test eder."""
    metin = (
        "Bölüm 1: Evren ve Galaksiler. Samanyolu galaksisi milyarlarca yıldıza ev sahipliği yapar. "
        "Güneş sistemi bu galaksinin avcı kolunda yer alır. "
        "Bölüm 2: İtalyan Mutfağı ve Makarnalar. Geleneksel makarna durum buğdayı irmiğinden üretilir. "
        "Pesto sosu taze fesleğen ve çam fıstığı ile hazırlanır. "
        "Bölüm 3: Kuantum Fiziği ve Dolanıklık. Kuantum dolanıklığı parçacıkların anlık etkileşimidir. "
        "Einstein bu durumu uzaktan hayaletimsi etki olarak tanımlamıştır."
    )

    parcalayici = SemantikParcalayici(esik_yontemi="standart_sapma", esik_katsayisi=0.4)
    sonuc = parcalayici.parcala(metin)

    assert sonuc["toplam_cumle"] >= 6
    assert sonuc["toplam_parca"] >= 2
    assert len(sonuc["parcalar"]) == sonuc["toplam_parca"]


def test_sabit_parcalama_karsilastir():
    """RAGParcalamaKarsilastirici sabit parçalama fonksiyonunun metni karakter sınırlarından böldüğünü test eder."""
    metin = "Bu bir test metnidir. " * 30
    parcalar = RAGParcalamaKarsilastirici.sabit_boyutlu_parcala(metin, parca_boyutu=150, cakisim=20)

    assert len(parcalar) > 1
    assert parcalar[0]["karakter_sayisi"] <= 150


def test_benchmark_karsilastir_metrikleri():
    """RAG parçalama karşılaştırma metriklerinin eksiksiz olduğunu test eder."""
    bench = RAGParcalamaKarsilastirici.benchmark_karsilastir()
    assert len(bench["metrikler"]) == 4
    assert bench["semantik_parcalama"][0] > bench["sabit_parcalama"][0]


def test_semantic_chunking_gorsellestirici_pano():
    """SemanticChunkingGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    metin = "Cümle bir yapay zeka. Cümle iki makine öğrenimi. Cümle üç veri analizi. Cümle dört aşçılık sanatı."
    parcalayici = SemantikParcalayici()
    sonuc = parcalayici.parcala(metin)
    bench = RAGParcalamaKarsilastirici.benchmark_karsilastir()

    gorsellestirici = SemanticChunkingGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_semantic_chunking_pano.png")
        gorsellestirici.pano_olustur(sonuc, bench, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
