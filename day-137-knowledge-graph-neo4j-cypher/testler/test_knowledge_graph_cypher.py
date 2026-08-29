"""
FAZ 7 GÜN 137: GraphRAG-2 Knowledge Graph, Neo4j & Cypher Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.ozellikli_graf_deposu import OzellikliGrafDeposu
from src.cypher_ayristirici_ve_motor import CypherMotoru
from src.graf_gezgini import GrafGezgini
from src.gorsellestirici import CypherGraphGorsellestirici


@pytest.fixture
def ornek_graf():
    """Testler için örnek Labeled Property Graph (LPG) veritabanı hazırlar."""
    depo = OzellikliGrafDeposu()
    depo.dugum_ekle("Vision Transformer", etiket="TEKNOLOJI", aciklama="Görüntüleri işleyen ViT mimarisi.")
    depo.dugum_ekle("Self-Attention", etiket="ALGORITMA", aciklama="QKV matrisli dikkat mekanizması.")
    depo.dugum_ekle("FlashAttention", etiket="ALGORITMA", aciklama="IO farkındalı GPU bellek optimizasyonu.")
    depo.dugum_ekle("NVIDIA GPU", etiket="DONANIM", aciklama="SRAM ve HBM bellekli hızlandırıcı.")

    depo.kenar_ekle("Vision Transformer", "Self-Attention", "KULLANIR", agirlik=1.0)
    depo.kenar_ekle("Self-Attention", "FlashAttention", "HIZLANDIRIR", agirlik=1.0)
    depo.kenar_ekle("FlashAttention", "NVIDIA GPU", "CALISIR", agirlik=1.0)
    return depo


def test_ozellikli_graf_dugum_ve_kenar_ekleme(ornek_graf):
    """Graf deposuna düğüm ve kenar ekleme mekanizmasını test eder."""
    assert len(ornek_graf.tum_dugumler()) == 4
    assert len(ornek_graf.tum_kenarlar()) == 3
    vit = ornek_graf.dugum_getir("Vision Transformer")
    assert vit is not None
    assert vit.etiket == "TEKNOLOJI"


def test_ozellikli_graf_komsuluk_getirme(ornek_graf):
    """Düğümün giren ve çıkan komşuluklarını test eder."""
    cikanlar = ornek_graf.komsulari_getir("Vision Transformer", yon="OUT")
    assert len(cikanlar) == 1
    kenar, hedef = cikanlar[0]
    assert kenar.iliski_tipi == "KULLANIR"
    assert hedef.id == "Self-Attention"

    girenler = ornek_graf.komsulari_getir("Self-Attention", yon="IN")
    assert len(girenler) == 1
    assert girenler[0][1].id == "Vision Transformer"


def test_cypher_bir_hop_sorgusu(ornek_graf):
    """Cypher motorunun 1-hop 'MATCH (a)-[r:KULLANIR]->(b) WHERE a.id = ...' sorgusunu test eder."""
    motor = CypherMotoru(ornek_graf)
    sorgu = "MATCH (a)-[r:KULLANIR]->(b) WHERE a.id = 'Vision Transformer' RETURN b"
    sonuclar = motor.sorgula(sorgu)

    assert len(sonuclar) == 1
    assert sonuclar[0]["hedef"] == "Self-Attention"


def test_cypher_iki_hop_zincir_sorgusu(ornek_graf):
    """Cypher motorunun 2-hop zincir 'MATCH (a)-[r1]->(b)-[r2]->(c) WHERE a.id = ...' sorgusunu test eder."""
    motor = CypherMotoru(ornek_graf)
    sorgu = "MATCH (a)-[r1]->(b)-[r2]->(c) WHERE a.id = 'Vision Transformer' RETURN c"
    sonuclar = motor.sorgula(sorgu)

    assert len(sonuclar) == 1
    assert sonuclar[0]["baslangic"] == "Vision Transformer"
    assert sonuclar[0]["ara_dugum"] == "Self-Attention"
    assert sonuclar[0]["hedef_dugum"] == "FlashAttention"


def test_graf_gezgini_k_hop_komsuluk(ornek_graf):
    """Graf gezgininin BFS ile k-hop alt-grafını başarıyla çıkardığını test eder."""
    gezgini = GrafGezgini(ornek_graf)
    altgraf = gezgini.k_hop_komsuluk("Vision Transformer", max_derinlik=2)

    dugum_idleri = [d.id for d in altgraf["dugumler"]]
    assert "Vision Transformer" in dugum_idleri
    assert "Self-Attention" in dugum_idleri
    assert "FlashAttention" in dugum_idleri


def test_graf_gezgini_en_kisa_yol(ornek_graf):
    """Graf gezgininin iki uzak kavram arasındaki akıl yürütme yolunu (Shortest Path) bulduğunu test eder."""
    gezgini = GrafGezgini(ornek_graf)
    yol = gezgini.en_kisa_yol("Vision Transformer", "NVIDIA GPU")

    assert yol is not None
    assert yol == ["Vision Transformer", "Self-Attention", "FlashAttention", "NVIDIA GPU"]


def test_altgraf_baglami_olustur_markdown(ornek_graf):
    """LLM prompt bağlamı için Markdown serileştirmesini test eder."""
    gezgini = GrafGezgini(ornek_graf)
    baglam = gezgini.altgraf_baglami_olustur("Vision Transformer", max_derinlik=2)

    assert "### [BİLGİ GRAFI BAĞLAMI" in baglam
    assert "Vision Transformer" in baglam
    assert "Self-Attention" in baglam


def test_cypher_gorsellestirici_pano(ornek_graf):
    """GraphRAG-2 teşhis panosu görselleştiricisinin PNG dosyasını ürettiğini test eder."""
    gezgini = GrafGezgini(ornek_graf)
    altgraf = gezgini.k_hop_komsuluk("Vision Transformer", max_derinlik=2)
    yol = gezgini.en_kisa_yol("Vision Transformer", "FlashAttention")
    bench = gezgini.benchmark_karsilastir()

    gorsellestirici = CypherGraphGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_cypher_pano.png")
        gorsellestirici.pano_olustur(bench, yol or [], altgraf, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
