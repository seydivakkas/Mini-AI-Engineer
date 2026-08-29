"""
FAZ 7 GÜN 133: HyDE (Hypothetical Document Embeddings) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.hipotez_ureticisi import HipotezUreticisi
from src.hyde_vektor_motoru import HyDEVektorMotoru
from src.hyde_getirici import HyDERAGGetirici
from src.gorsellestirici import HyDEGorsellestirici


def test_hipotez_ureticisi_tek_ve_coklu():
    """HipotezUreticisi'nin tekil ve çoklu varsayımsal pasajlar ürettiğini test eder."""
    sorgu = "Kuantum dolanıklığı ve Qubit süperpozisyonu nedir?"
    tek_h = HipotezUreticisi.tek_hipotez_uret(sorgu, perspektif="tanimsal_mekanik")
    coklu_h = HipotezUreticisi.coklu_hipotez_uret(sorgu, n=3)

    assert len(tek_h) > 50
    assert "kuantum" in tek_h.lower()
    assert len(coklu_h) == 3


def test_hyde_vektor_motoru_l2_norm():
    """HyDEVektorMotoru'nun ürettiği embedding'lerin L2 normunun 1.0 olduğunu test eder."""
    motor = HyDEVektorMotoru(vektor_boyutu=64)
    v = motor.metin_vektorlestir("Test metni")

    norm = float(torch.norm(v, p=2, dim=1).item())
    assert pytest.approx(norm, 1e-4) == 1.0


def test_hyde_centroid_vektoru_hesapla():
    """Centroid hesaplayıcının birden çok hipotez vektörünü ortalayıp normalize ettiğini test eder."""
    motor = HyDEVektorMotoru(vektor_boyutu=64)
    hipotezler = [
        "Birinci varsayımsal teknik belge.",
        "İkinci alternatif mimari açıklama.",
        "Üçüncü sistemik değerlendirme.",
    ]
    centroid = motor.hyde_centroid_vektoru_hesapla(hipotezler)

    assert centroid.shape == (1, 64)
    norm = float(torch.norm(centroid, p=2, dim=1).item())
    assert pytest.approx(norm, 1e-4) == 1.0


def test_hyde_getirici_belge_ekle_ve_boyut():
    """HyDERAGGetirici'nin belgeleri başarıyla ekleyip tensör boyutunu koruduğunu test eder."""
    getirici = HyDERAGGetirici(vektor_boyutu=64)
    getirici.toplu_belge_ekle([
        {"doc_id": "DOC_01", "metin": "Raft algoritması konsensüs sağlar.", "kategori": "dağıtık"},
        {"doc_id": "DOC_02", "metin": "Transformer öz-dikkat mekanizması.", "kategori": "nlp"},
    ])

    assert len(getirici.belgeler) == 2
    assert len(getirici.belge_embeddingleri) == 2


def test_standart_vektor_arama():
    """Standart soru embedding'i ile aramanın çalıştığını test eder."""
    getirici = HyDERAGGetirici(vektor_boyutu=64)
    getirici.belge_ekle("D1", "Raft konsensüs lider seçimi ve günlük çoğaltma.")
    getirici.belge_ekle("D2", "Evrişimli sinir ağları ve görüntü tanıma.")

    sonuclar = getirici.standart_arama("Raft konsensüs protokolü", top_k=1)
    assert len(sonuclar) == 1
    assert sonuclar[0]["doc_id"] == "D1"


def test_hyde_arama_pipeline():
    """HyDE aramasının hipotez üreterek gerçek belgeleri başarıyla getirdiğini test eder."""
    getirici = HyDERAGGetirici(vektor_boyutu=64)
    getirici.toplu_belge_ekle([
        {"doc_id": "DOC_01", "metin": "GPU Tensör Paralelizmi ve Megatron-LM All-Reduce iletişimi."},
        {"doc_id": "DOC_02", "metin": "Biyokimya ve protein katlanması enzim mekanizmaları."},
    ])

    sonuc = getirici.hyde_arama("GPU'lar arası tensör paralelizmi nasıl çalışır?", hipotez_sayisi=3, top_k=1)
    assert sonuc["hipotez_sayisi"] == 3
    assert len(sonuc["getirilen_belgeler"]) == 1
    assert sonuc["getirilen_belgeler"][0]["doc_id"] == "DOC_01"


def test_benchmark_karsilastir_metrikleri():
    """HyDE karşılaştırma metriklerinin eksiksiz olduğunu test eder."""
    getirici = HyDERAGGetirici()
    bench = getirici.benchmark_karsilastir()

    assert len(bench["metrikler"]) == 4
    assert bench["hyde_retrieval"][0] > bench["standart_dense"][0]


def test_hyde_gorsellestirici_pano():
    """HyDEGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    getirici = HyDERAGGetirici(vektor_boyutu=64)
    getirici.belge_ekle("D1", "Doğal dil işleme ve LLM token optimizasyonu.")
    std_sonuc = getirici.standart_arama("LLM tokenları")
    hyde_sonuc = getirici.hyde_arama("LLM tokenları", hipotez_sayisi=3)
    bench = getirici.benchmark_karsilastir()

    gorsellestirici = HyDEGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_hyde_pano.png")
        gorsellestirici.pano_olustur(hyde_sonuc, std_sonuc, bench, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
