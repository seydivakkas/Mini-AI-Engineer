"""
FastAPI Asenkron Çıkarım Servisi Birim ve Entegrasyon Testleri (Day 98).
Tüm endpoint'ler TestClient ile test edilir (%100 PASSED).
"""

import io
import base64
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.api_uygulamasi import app
from src.servis_yoneticisi import ServisYoneticisi


@pytest.fixture(scope="session")
def test_istemcisi():
    """FastAPI TestClient fikstürü."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def ornek_gorsel_bytes():
    """Test için 32x32 RGB sentetik PNG görsel baytları."""
    img = Image.new("RGB", (32, 32), color=(70, 130, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_kok_dizin_endpoint(test_istemcisi):
    """GET / endpoint'inin 200 OK ve karşılama mesajı döndürdüğünü test eder."""
    yanit = test_istemcisi.get("/")
    assert yanit.status_code == 200
    veri = yanit.json()
    assert veri["durum"] == "CALISIYOR"
    assert "saglik_kontrolu" in veri


def test_health_ve_readiness_probelari(test_istemcisi):
    """GET /health, /healthz ve /ready probelarının HEALTHY döndürdüğünü test eder."""
    for ep in ["/health", "/healthz", "/ready"]:
        yanit = test_istemcisi.get(ep)
        assert yanit.status_code == 200
        veri = yanit.json()
        assert veri["status"] == "HEALTHY"
        assert veri["model_loaded"] is True
        assert "cihaz" in veri
        assert "calisma_suresi_sn" in veri


def test_metadata_endpoint(test_istemcisi):
    """GET /metadata endpoint'inin model mimari ve etiket bilgilerini doğru döndürdüğünü test eder."""
    yanit = test_istemcisi.get("/metadata")
    assert yanit.status_code == 200
    veri = yanit.json()
    assert veri["model_adi"] == "MiniViT-v1.0"
    assert veri["parametre_sayisi"] > 0
    assert veri["sinif_sayisi"] == 10
    assert "0" in veri["id2label"] or 0 in veri["id2label"]


def test_metrics_endpoint(test_istemcisi):
    """GET /metrics endpoint'inin gecikme ve istek sayılarını raporladığını test eder."""
    yanit = test_istemcisi.get("/metrics")
    assert yanit.status_code == 200
    veri = yanit.json()
    assert "toplam_istek_sayisi" in veri
    assert "p50_gecikme_ms" in veri
    assert "p99_gecikme_ms" in veri


def test_predict_multipart_dosya_yukleme(test_istemcisi, ornek_gorsel_bytes):
    """POST /predict endpoint'inin dosya yüklemesi ile başarılı Top-K çıkarımı yaptığını test eder."""
    dosyalar = {"file": ("test_resim.png", ornek_gorsel_bytes, "image/png")}
    yanit = test_istemcisi.post("/predict?top_k=5", files=dosyalar)

    assert yanit.status_code == 200
    veri = yanit.json()
    assert veri["durum"] == "basarili"
    assert "en_iyi_tahmin" in veri
    assert len(veri["top_k_tahminler"]) == 5
    assert veri["gecikme_ms"] > 0.0
    assert 0.0 <= veri["en_iyi_tahmin"]["olasilik"] <= 1.0


def test_predict_base64_json_istegi(test_istemcisi, ornek_gorsel_bytes):
    """POST /predict/base64 endpoint'inin Base64 JSON girdisini başarıyla sınıflandırdığını test eder."""
    b64_str = base64.b64encode(ornek_gorsel_bytes).decode("utf-8")
    govde = {"base64_goruntu": f"data:image/png;base64,{b64_str}", "top_k": 3}

    yanit = test_istemcisi.post("/predict/base64", json=govde)

    assert yanit.status_code == 200
    veri = yanit.json()
    assert veri["durum"] == "basarili"
    assert len(veri["top_k_tahminler"]) == 3


def test_predict_batch_coklu_dosya(test_istemcisi, ornek_gorsel_bytes):
    """POST /predict/batch endpoint'inin çoklu dosya yüklemesini toplu olarak işlediğini test eder."""
    dosyalar = [
        ("files", ("resim1.png", ornek_gorsel_bytes, "image/png")),
        ("files", ("resim2.png", ornek_gorsel_bytes, "image/png")),
    ]
    yanit = test_istemcisi.post("/predict/batch?top_k=5", files=dosyalar)

    assert yanit.status_code == 200
    veri = yanit.json()
    assert veri["durum"] == "basarili"
    assert veri["toplam_goruntu"] == 2
    assert len(veri["sonuclar"]) == 2


def test_gecersiz_girdi_hata_yonetimi(test_istemcisi):
    """Geçersiz bozuk dosya yüklendiğinde 400 Bad Request döndürüldüğünü test eder."""
    dosyalar = {"file": ("bozuk.txt", b"Bu bir resim degildir!", "text/plain")}
    yanit = test_istemcisi.post("/predict", files=dosyalar)

    assert yanit.status_code == 400
    veri = yanit.json()
    assert "detail" in veri
