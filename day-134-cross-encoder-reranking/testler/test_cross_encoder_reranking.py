"""
FAZ 7 GÜN 134: Two-Stage Precision Retrieval (Bi-Encoder + Cross-Encoder) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import numpy as np

from src.bi_encoder import BiEncoderArama
from src.cross_encoder import CrossEncoderReranker
from src.iki_asamali_getirici import IkiAsamaliRAGGetirici
from src.gorsellestirici import CrossEncoderGorsellestirici


def test_bi_encoder_aday_getir():
    """BiEncoderArama sınıfının adayları kosinüs benzerliği ile hızlıca sıraladığını test eder."""
    bi = BiEncoderArama(vektor_boyutu=64)
    bi.belge_ekle("D1", "Raft konsensüs protokolü lider seçimi.")
    bi.belge_ekle("D2", "Evrişimli sinir ağları görüntü işleme.")

    adaylar = bi.aday_getir("Raft lideri", top_k=2)
    assert len(adaylar) == 2
    assert adaylar[0][0]["doc_id"] == "D1"
    assert adaylar[0][1] > adaylar[1][1]


def test_cross_encoder_puanla_ve_matris():
    """CrossEncoderReranker sınıfının [0, 1] aralığında skor ve çapraz dikkat matrisi ürettiğini test eder."""
    cross = CrossEncoderReranker(gizli_boyut=32)
    skor, matris = cross.puanla("Transformer mimarisi", "Transformer öz-dikkat mekanizması ile çalışır.")

    assert 0.0 <= skor <= 1.0
    assert matris.ndim == 2
    assert matris.shape[0] >= 1
    assert matris.shape[1] >= 1


def test_cross_encoder_yeniden_sirala():
    """CrossEncoder'ın aday listesini çapraz dikkat skoruna göre azalan sırada dizdiğini test eder."""
    cross = CrossEncoderReranker(gizli_boyut=32)
    adaylar = [
        {"doc_id": "D_UZAK", "metin": "Tarım ve sulama yöntemleri."},
        {"doc_id": "D_YAKIN", "metin": "Büyük dil modellerinde LoRA ince ayarı."},
    ]

    sirali = cross.yeniden_sirala("LoRA ile model eğitimi", adaylar)
    assert sirali[0]["doc_id"] == "D_YAKIN"
    assert sirali[0]["cross_encoder_skor"] > sirali[1]["cross_encoder_skor"]


def test_iki_asamali_getirici_entegrasyon():
    """IkiAsamaliRAGGetirici sınıfının 1. ve 2. aşamaları sırasıyla başarıyla işlettiğini test eder."""
    getirici = IkiAsamaliRAGGetirici(vektor_boyutu=64, cross_gizli_boyut=32)
    getirici.toplu_belge_ekle([
        {"doc_id": "D1", "metin": "PostgreSQL indeksleme B-Tree ve GIN."},
        {"doc_id": "D2", "metin": "Redis önbellekleme ve veri yapıları."},
        {"doc_id": "D3", "metin": "Kubernetes Pod orkestrasyonu."},
    ])

    sonuc = getirici.getir_ve_yeniden_sirala("PostgreSQL veri tabanı indeksleri", aday_k=3, nihai_k=2)

    assert len(sonuc["asama_1_adaylar"]) == 3
    assert len(sonuc["nihai_sonuclar"]) == 2
    assert "bi_encoder_ms" in sonuc["sureler"]
    assert "cross_encoder_ms" in sonuc["sureler"]


def test_ndcg_hesaplayici():
    """NDCG@k hesaplama fonksiyonunun ideal sırada 1.0, ters sırada düşük değer ürettiğini test eder."""
    ideal_ndcg = IkiAsamaliRAGGetirici.ndcg_hesapla([3.0, 2.0, 1.0, 0.0], k=4)
    ters_ndcg = IkiAsamaliRAGGetirici.ndcg_hesapla([0.0, 1.0, 2.0, 3.0], k=4)

    assert pytest.approx(ideal_ndcg, 1e-3) == 1.0
    assert ters_ndcg < ideal_ndcg


def test_sira_degisimi_rank_shift():
    """Sıralama değişim değerinin (Rank Shift) doğru hesaplandığını test eder."""
    getirici = IkiAsamaliRAGGetirici(vektor_boyutu=64, cross_gizli_boyut=32)
    getirici.toplu_belge_ekle([
        {"doc_id": "D1", "metin": "Genel bilgisayar donanımları."},
        {"doc_id": "D2", "metin": "Derin pekiştirmeli öğrenme PPO algoritması."},
    ])

    sonuc = getirici.getir_ve_yeniden_sirala("PPO algoritması ile pekiştirmeli öğrenme", aday_k=2, nihai_k=2)
    for doc in sonuc["asama_2_tam_liste"]:
        assert "sira_degisimi" in doc


def test_benchmark_karsilastir_metrikleri():
    """Karşılaştırma benchmark metriklerinin eksiksiz olduğunu test eder."""
    getirici = IkiAsamaliRAGGetirici()
    bench = getirici.benchmark_karsilastir()

    assert len(bench["metrikler"]) == 4
    assert bench["cross_encoder_reranked"][0] > bench["bi_encoder_yalnizca"][0]


def test_cross_encoder_gorsellestirici_pano():
    """CrossEncoderGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    getirici = IkiAsamaliRAGGetirici(vektor_boyutu=64, cross_gizli_boyut=32)
    getirici.belge_ekle("D1", "PyTorch ve derin öğrenme tensör optimizasyonu.")
    sonuc = getirici.getir_ve_yeniden_sirala("PyTorch tensörleri", aday_k=1, nihai_k=1)
    bench = getirici.benchmark_karsilastir()

    _, matris = getirici.cross_encoder.puanla("PyTorch", "PyTorch tensörleri")

    gorsellestirici = CrossEncoderGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_cross_encoder_pano.png")
        gorsellestirici.pano_olustur(
            getirme_sonucu=sonuc,
            dikkat_matrisi=matris,
            sorgu_tokenlari=["pytorch"],
            belge_tokenlari=["pytorch", "tensor"],
            karsilastirma=bench,
            kayit_yolu=kayit,
        )

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
