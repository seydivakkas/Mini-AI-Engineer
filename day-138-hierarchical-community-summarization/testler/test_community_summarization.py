"""
FAZ 7 GÜN 138: GraphRAG-3 Leiden Community Detection & Hierarchical Summarization Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.leiden_topluluk_tespiti import ToplulukKumesi, LeidenToplulukDedektoru
from src.hiyerarsik_ozetleyici import ToplulukRaporu, HiyerarsikOzetleyici
from src.kuresel_arama_motoru import KureselAramaMotoru
from src.gorsellestirici import CommunitySummarizationGorsellestirici


@pytest.fixture
def ornek_dugumler_ve_kenarlar():
    """Topluluk tespiti için örnek bilgi grafı hazırlar."""
    dugumler = [
        "Vision Transformer", "Self-Attention", "FlashAttention", "NVIDIA GPU",
        "Raft", "Quorum", "PostgreSQL", "B-Tree",
        "Limit Order Book", "FPGA"
    ]
    kenarlar = [
        ("Vision Transformer", "Self-Attention", 1.0),
        ("Self-Attention", "FlashAttention", 1.0),
        ("FlashAttention", "NVIDIA GPU", 1.0),
        ("Raft", "Quorum", 1.0),
        ("PostgreSQL", "B-Tree", 1.0),
        ("Limit Order Book", "FPGA", 1.0),
    ]
    detaylar = {
        "Vision Transformer": "ViT görüntü mimarisi.",
        "Self-Attention": "Dikkat matrisi.",
        "FlashAttention": "GPU bellek optimizasyonu.",
        "NVIDIA GPU": "Donanım hızlandırıcı.",
        "Raft": "Lider seçimi.",
        "Quorum": "Çoğunluk kuralı.",
        "PostgreSQL": "Veritabanı motoru.",
        "B-Tree": "Arama indeksi.",
        "Limit Order Book": "Emir defteri.",
        "FPGA": "Mikrosaniye donanım.",
    }
    return dugumler, kenarlar, detaylar


def test_leiden_topluluk_tespiti_seviye1(ornek_dugumler_ve_kenarlar):
    """Leiden dedektörünün Seviye-1 alt-alan kümelerini başarıyla tespit ettiğini test eder."""
    dugumler, kenarlar, _ = ornek_dugumler_ve_kenarlar
    hiyerarsi = LeidenToplulukDedektoru.tespit_et(dugumler, kenarlar)

    assert 1 in hiyerarsi
    assert len(hiyerarsi[1]) >= 2
    topluluk_adlari = [t.alan_adi for t in hiyerarsi[1]]
    assert any("Dikkat" in ad or "Yapay Zeka" in ad or "Öğrenme" in ad for ad in topluluk_adlari)


def test_leiden_topluluk_tespiti_seviye2_makro(ornek_dugumler_ve_kenarlar):
    """Leiden dedektörünün Seviye-2 Makro Kök Topluluğunu ürettiğini test eder."""
    dugumler, kenarlar, _ = ornek_dugumler_ve_kenarlar
    hiyerarsi = LeidenToplulukDedektoru.tespit_et(dugumler, kenarlar)

    assert 2 in hiyerarsi
    assert len(hiyerarsi[2]) == 1
    makro = hiyerarsi[2][0]
    assert len(makro.alt_topluluklar) >= 2


def test_modulerlik_skoru_hesaplama(ornek_dugumler_ve_kenarlar):
    """Modülerlik $Q$ skorunun kaliteli bir kümelemeye işaret ettiğini test eder."""
    dugumler, kenarlar, _ = ornek_dugumler_ve_kenarlar
    hiyerarsi = LeidenToplulukDedektoru.tespit_et(dugumler, kenarlar)
    q = LeidenToplulukDedektoru.modulerlik_hesapla(hiyerarsi[1], len(kenarlar))

    assert q > 0.70


def test_hiyerarsik_ozetleyici_rapor_uretimi(ornek_dugumler_ve_kenarlar):
    """HiyerarsikOzetleyici motorunun Seviye 1 ve 2 raporlarını ürettiğini test eder."""
    dugumler, kenarlar, detaylar = ornek_dugumler_ve_kenarlar
    hiyerarsi = LeidenToplulukDedektoru.tespit_et(dugumler, kenarlar)
    raporlar = HiyerarsikOzetleyici.raporlari_uret(hiyerarsi, detaylar)

    assert len(raporlar) >= 3
    assert any(r.seviye == 2 for r in raporlar.values())
    assert any(r.seviye == 1 for r in raporlar.values())


def test_kuresel_arama_map_asamasi(ornek_dugumler_ve_kenarlar):
    """Küresel arama motorunun Map aşamasında raporları doğru puanladığını test eder."""
    dugumler, kenarlar, detaylar = ornek_dugumler_ve_kenarlar
    hiyerarsi = LeidenToplulukDedektoru.tespit_et(dugumler, kenarlar)
    raporlar = HiyerarsikOzetleyici.raporlari_uret(hiyerarsi, detaylar)

    motor = KureselAramaMotoru(raporlar)
    sonuc = motor.kuresel_sorgula("Sistemdeki genel yapay zeka ve donanım mimarisi nasıldır?")

    assert len(sonuc["haritalanan_raporlar"]) == len(raporlar)
    assert sonuc["haritalanan_raporlar"][0]["skor"] > 0.0


def test_kuresel_arama_reduce_yanit(ornek_dugumler_ve_kenarlar):
    """Küresel arama motorunun Reduce aşamasında makro sentez yanıtı ürettiğini test eder."""
    dugumler, kenarlar, detaylar = ornek_dugumler_ve_kenarlar
    hiyerarsi = LeidenToplulukDedektoru.tespit_et(dugumler, kenarlar)
    raporlar = HiyerarsikOzetleyici.raporlari_uret(hiyerarsi, detaylar)

    motor = KureselAramaMotoru(raporlar)
    sonuc = motor.kuresel_sorgula("Tüm platformun temel mimari bileşenleri nelerdir?")

    assert len(sonuc["nihai_kuresel_yanit"]) > 50
    assert "###" in sonuc["nihai_kuresel_yanit"]


def test_kuresel_arama_benchmark_metrikleri(ornek_dugumler_ve_kenarlar):
    """Benchmark kıyaslama metriklerinin eksiksiz olduğunu test eder."""
    dugumler, kenarlar, detaylar = ornek_dugumler_ve_kenarlar
    hiyerarsi = LeidenToplulukDedektoru.tespit_et(dugumler, kenarlar)
    raporlar = HiyerarsikOzetleyici.raporlari_uret(hiyerarsi, detaylar)

    motor = KureselAramaMotoru(raporlar)
    bench = motor.benchmark_karsilastir()

    assert len(bench["metrikler"]) == 4
    assert bench["graphrag_hierarchical"][0] > bench["standart_vektor_rag"][0]


def test_community_summarization_gorsellestirici_pano(ornek_dugumler_ve_kenarlar):
    """Teşhis panosu görselleştiricisinin 6 panelli PNG dosyasını ürettiğini test eder."""
    dugumler, kenarlar, detaylar = ornek_dugumler_ve_kenarlar
    hiyerarsi = LeidenToplulukDedektoru.tespit_et(dugumler, kenarlar)
    raporlar = HiyerarsikOzetleyici.raporlari_uret(hiyerarsi, detaylar)

    motor = KureselAramaMotoru(raporlar)
    sonuc = motor.kuresel_sorgula("Genel mimari özet")
    bench = motor.benchmark_karsilastir()

    gorsellestirici = CommunitySummarizationGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_community_pano.png")
        gorsellestirici.pano_olustur(bench, hiyerarsi, sonuc, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
