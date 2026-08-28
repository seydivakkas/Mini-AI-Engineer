"""
Day 35: FastAPI ile Asenkron AI Model Servisi Birim Testleri.
"""

import os
import io
import pytest
from fastapi.testclient import TestClient
from src.servis_uygulamasi import app
from src.gorsellestirici import FastAPIServisGorsellestirici


@pytest.fixture
def client():
    return TestClient(app)


def test_healthz_endpoint(client):
    """Sağlık kontrolü /healthz testi."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["durum"] == "aktif"
    assert "MiniMetinSiniflandirici" in data["yuklu_modeller"]
    assert "X-Process-Time-Ms" in response.headers


def test_predict_text_valid(client):
    """Metin sınıflandırma ve embedding çıkarma testi."""
    istek = {
        "metin": "Derin Ogrenme ve Yapay Sinir Aglari",
        "kategori": "NLP",
        "embedding_iste": True
    }
    response = client.post("/api/v1/predict/text", json=istek)
    assert response.status_code == 200
    data = response.json()
    assert "tahmin_edilen_etiket" in data
    assert 0.0 <= data["olasilik"] <= 1.0
    assert len(data["vektor_embedding"]) == 64


def test_predict_text_validation_error(client):
    """Girdi doğrulama hatası (422) testi."""
    istek = {"metin": "ab"}  # min_length=3 kuralı ihlal edilir
    response = client.post("/api/v1/predict/text", json=istek)
    assert response.status_code == 422
    data = response.json()
    assert data["kod"] == 422
    assert "Doğrulama Hatası" in data["hata"]


def test_predict_image_upload(client):
    """Görsel yükleme ve analiz testi."""
    sahte_bayt = b"\x00\xff\xaa" * 20
    files = {"dosya": ("test.png", io.BytesIO(sahte_bayt), "image/png")}
    response = client.post("/api/v1/predict/image", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["dosya_adi"] == "test.png"
    assert len(data["tespit_edilen_nesneler"]) >= 1


def test_rag_query_endpoint(client):
    """RAG doküman sorgulama testi."""
    istek = {"soru": "YOLO nasil calisir?", "top_k": 1}
    response = client.post("/api/v1/rag/query", json=istek)
    assert response.status_code == 200
    data = response.json()
    assert data["durum"] == "BASARILI"
    assert len(data["kaynaklar"]) >= 1


def test_telemetry_endpoint(client):
    """Telemetri sayacı testi."""
    # İstek yapıp telemetriyi tetikle
    client.get("/healthz")
    response = client.get("/api/v1/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert data["toplam_istek"] >= 1


def test_servis_gorsellestirici(tmp_path):
    """6 panelli servis teşhis panosu çizim testi."""
    telemetri_mock = {
        "endpoint_istek_dagilimi": {"/predict/text": 10, "/healthz": 5}
    }
    ornek_mock = {
        "tum_olasiliklar": {"A": 0.7, "B": 0.3},
        "baskin_renk": [100, 150, 200]
    }
    cikis_path = str(tmp_path / "test_servis.png")
    yol = FastAPIServisGorsellestirici.servis_paneli_ciz(telemetri_mock, ornek_mock, hedef_path=cikis_path)
    assert os.path.exists(yol)
