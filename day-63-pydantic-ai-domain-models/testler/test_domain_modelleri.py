"""
Day 63: Pydantic v2 Tip Güvenli Domain Modelleri Birim Testleri.
"""

import pytest
import os
import numpy as np
from pydantic import ValidationError

from src.domain_modelleri import (
    GorselMetadatasi,
    BoundingBoxModeli,
    NesneTespitiSonucu,
    VektorEmbeddingSozlesmesi,
    InferenceIstekSozlesmesi,
    InferenceYanitSozlesmesi
)
from src.sozlesme_dogrulayici import SozlesmeDogrulayici, PydanticBenchmarkEngine
from src.gorsellestirici import PydanticGorsellestirici


def test_gorsel_metadatasi_gecerli_ve_gecersiz():
    """Görsel metaveri sözleşmesinin geçerli ve geçersiz girişlerini test eder."""
    # Geçerli
    meta = GorselMetadatasi(
        genislik=1920,
        yukseklik=1080,
        kanal_sayisi=3,
        format="PNG",
        dosya_boyutu_kb=1024.0
    )
    assert meta.toplam_piksel == 1920 * 1080
    assert meta.format == "PNG"

    # Geçersiz Format
    with pytest.raises(ValidationError):
        GorselMetadatasi(
            genislik=1920,
            yukseklik=1080,
            kanal_sayisi=3,
            format="BMP",  # Desteklenmeyen format
            dosya_boyutu_kb=1024.0
        )

    # Geçersiz Negatif Boyut
    with pytest.raises(ValidationError):
        GorselMetadatasi(
            genislik=-500,
            yukseklik=1080,
            kanal_sayisi=3,
            format="PNG",
            dosya_boyutu_kb=1024.0
        )


def test_bounding_box_geometrisi_ve_iou():
    """Sınır kutusu koordinat doğrulaması ve IoU hesaplamasını test eder."""
    # Geçerli Kutu
    b1 = BoundingBoxModeli(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    assert pytest.approx(b1.alan(), abs=1e-5) == 0.16

    # Ters Koordinat (x_min >= x_max) -> ValidationError
    with pytest.raises(ValidationError):
        BoundingBoxModeli(x_min=0.8, y_min=0.1, x_max=0.2, y_max=0.5)

    # IoU Testi
    b2 = BoundingBoxModeli(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    assert pytest.approx(b1.iou(b2), abs=1e-5) == 1.0


def test_vektor_embedding_l2_norm_dogrulamasi():
    """Embedding vektörlerinin L2-normalize birim uzunluk gereksinimini test eder."""
    raw = np.random.randn(512).astype(np.float32)
    norm = (raw / np.linalg.norm(raw)).tolist()

    # Geçerli Normalize Vektör
    emb = VektorEmbeddingSozlesmesi(vektor=norm, beklenen_boyut=512)
    assert len(emb.vektor) == 512

    # Normalize Edilmemiş Vektör -> ValidationError
    unnorm = (raw * 10.0).tolist()
    with pytest.raises(ValidationError):
        VektorEmbeddingSozlesmesi(vektor=unnorm, beklenen_boyut=512)


def test_nesne_tespiti_sonucu():
    """Nesne tespiti güven skoru ve sınır kutusu sözleşmesini test eder."""
    bbox = BoundingBoxModeli(x_min=0.2, y_min=0.3, x_max=0.7, y_max=0.8)
    tespit = NesneTespitiSonucu(sinif_adi="araba", guven_skoru=0.92, kutu=bbox)
    assert tespit.sinif_adi == "araba"
    assert tespit.guven_skoru == 0.92

    # Geçersiz Güven Skoru (> 1.0)
    with pytest.raises(ValidationError):
        NesneTespitiSonucu(sinif_adi="araba", guven_skoru=1.5, kutu=bbox)


def test_inference_istek_ve_yanit_sozlesmesi():
    """API istek ve yanıt sözleşmelerinin uçtan uca doğrulanmasını test eder."""
    payload_istek = {
        "istek_id": "req_vision_123456",
        "model_adi": "YOLOv8x",
        "gorsel_meta": {
            "genislik": 1280,
            "yukseklik": 720,
            "kanal_sayisi": 3,
            "format": "WEBP",
            "dosya_boyutu_kb": 850.0
        },
        "nms_esigi": 0.45,
        "guven_esigi": 0.60
    }
    istek, err = SozlesmeDogrulayici.dogrula_istek(payload_istek)
    assert err is None
    assert istek is not None
    assert istek.model_adi == "YOLOv8x"


def test_sozlesme_dogrulayici_json_schema():
    """LLM Structured Tool Calling için JSON Schema üretimini test eder."""
    schema = SozlesmeDogrulayici.json_sema_uret(InferenceIstekSozlesmesi)
    assert "properties" in schema
    assert "istek_id" in schema["properties"]
    assert "gorsel_meta" in schema["properties"]


def test_benchmark_ve_gorsellestirici(tmp_path):
    """Pydantic benchmark motoru ve 6 panelli teşhis panosu üretimini test eder."""
    bench = PydanticBenchmarkEngine.calistir_benchmark(num_samples=100)
    assert bench["dogrulama_qps"] > 0
    assert bench["hata_yakalama_orani_yuzde"] == 100.0

    pano_yolu = os.path.join(tmp_path, "test_pydantic_pano.png")
    cikis = PydanticGorsellestirici.panel_ciz(bench, hedef_path=pano_yolu)
    assert os.path.exists(cikis)
