"""
Docker ve Yük Testi Birim ve Entegrasyon Testleri (Day 99).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import io
import tempfile
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.api_uygulamasi import app
from src.yuk_testi_motoru import YukTestiMotoru
from src.gorsellestirici import DockerYukGorsellestirici


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def ornek_gorsel_bytes():
    img = Image.new("RGB", (32, 32), color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_dockerfile_ve_compose_varligi():
    """Dockerfile ve docker-compose yapılandırmalarının doğruluğunu test eder."""
    kok_dizin = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dockerfile = os.path.join(kok_dizin, "Dockerfile")
    compose = os.path.join(kok_dizin, "docker-compose.yml")

    assert os.path.exists(dockerfile)
    assert os.path.exists(compose)

    with open(dockerfile, "r", encoding="utf-8") as f:
        df_content = f.read()
        assert "HEALTHCHECK" in df_content
        assert "appuser" in df_content

    with open(compose, "r", encoding="utf-8") as f:
        dc_content = f.read()
        assert "minivit-api" in dc_content
        assert "locust" in dc_content


def test_locustfile_tanimlari():
    """locustfile.py dosyasının task'lerini ve kullanıcı sınıflarını test eder."""
    kok_dizin = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    locustfile = os.path.join(kok_dizin, "locustfile.py")
    assert os.path.exists(locustfile)

    with open(locustfile, "r", encoding="utf-8") as f:
        content = f.read()
        assert "MiniViTKullanicisi" in content
        assert "test_predict_multipart" in content
        assert "test_predict_base64" in content


def test_api_health_endpoint(test_client):
    """GET /health endpoint'inin 200 OK ve HEALTHY döndürdüğünü test eder."""
    yanit = test_client.get("/health")
    assert yanit.status_code == 200
    veri = yanit.json()
    assert veri["status"] == "HEALTHY"
    assert veri["model_loaded"] is True


def test_api_predict_endpoint(test_client, ornek_gorsel_bytes):
    """POST /predict endpoint'inin çıkarım yaptığını test eder."""
    files = {"file": ("test.png", ornek_gorsel_bytes, "image/png")}
    yanit = test_client.post("/predict?top_k=5", files=files)
    assert yanit.status_code == 200
    veri = yanit.json()
    assert veri["durum"] == "basarili"
    assert len(veri["top_k_tahminler"]) == 5


def test_api_base64_endpoint(test_client, ornek_gorsel_bytes):
    """POST /predict/base64 endpoint'inin Base64 veriyi işlediğini test eder."""
    import base64
    b64 = base64.b64encode(ornek_gorsel_bytes).decode("utf-8")
    yanit = test_client.post("/predict/base64", json={"base64_goruntu": b64, "top_k": 3})
    assert yanit.status_code == 200
    veri = yanit.json()
    assert veri["durum"] == "basarili"
    assert len(veri["top_k_tahminler"]) == 3


@pytest.mark.asyncio
async def test_eszamanli_seviye_testi_motoru():
    """YukTestiMotoru'nun tek seviyede yük testi koşturduğunu test eder."""
    motor = YukTestiMotoru(app)
    res = await motor.eszamanli_seviye_testi(kullanici_sayisi=5, kullanici_basina_istek=2)
    assert res["toplam_istek"] == 10
    assert res["hata_sayisi"] == 0
    assert res["hata_orani_yuzde"] == 0.0
    assert res["throughput_rps"] > 0.0


@pytest.mark.asyncio
async def test_basamakli_yuk_testi_motoru():
    """YukTestiMotoru'nun basamaklı yük testi koşturduğunu test eder."""
    motor = YukTestiMotoru(app)
    sonuclar = await motor.basamakli_yuk_testi(kullanici_basamaklari=[1, 3, 5])
    assert len(sonuclar) == 3
    for s in sonuclar:
        assert s["hata_orani_yuzde"] == 0.0


def test_gorsellestirici_pano_uretme():
    """6 panelli teşhis panosunun oluşturulduğunu test eder."""
    gorsellestirici = DockerYukGorsellestirici(dpi=100)
    ornek_sonuclar = [
        {"kullanici_sayisi": 1, "throughput_rps": 120, "p50_ms": 3.0, "p90_ms": 4.0, "p99_ms": 5.0, "toplam_istek": 10, "basarili_sayisi": 10, "hata_sayisi": 0},
        {"kullanici_sayisi": 5, "throughput_rps": 350, "p50_ms": 3.5, "p90_ms": 4.8, "p99_ms": 6.2, "toplam_istek": 20, "basarili_sayisi": 20, "hata_sayisi": 0},
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit_yolu = os.path.join(tmp_dir, "test_docker_paneli.png")
        gorsellestirici.pano_olustur(ornek_sonuclar, kayit_yolu=kayit_yolu)
        assert os.path.exists(kayit_yolu)
        assert os.path.getsize(kayit_yolu) > 1000
