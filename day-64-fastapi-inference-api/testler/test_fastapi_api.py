"""
Day 64: FastAPI İnference API ve Model Lifespan Birim Testleri.
"""

import pytest
import os
import httpx
import asyncio

from src.api_servisi import olustur_uygulama
from src.batch_kuyruk_yoneticisi import DinamikBatchKuyrugu
from src.gorsellestirici import FastAPIGorsellestirici


@pytest.mark.asyncio
async def test_kok_dizin_endpoint():
    """Kök dizin GET / endpoint'ini test eder."""
    app = olustur_uygulama()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/")
            assert res.status_code == 200
            data = res.json()
            assert data["durum"] == "aktif"
            assert "dokumantasyon" in data


@pytest.mark.asyncio
async def test_saglik_kontrolu_endpoint():
    """Sağlık ve Lifespan GET /saglik kontrolünü test eder."""
    app = olustur_uygulama()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/saglik")
            assert res.status_code == 200
            data = res.json()
            assert data["durum"] == "saglikli"
            assert data["hazir"] is True
            assert data["model_adi"] == "MiniVision-YOLOv8-Embedder"


@pytest.mark.asyncio
async def test_tekil_tahmin_endpoint():
    """Tekil çıkarım POST /v1/tahmin/tekil endpoint'ini test eder."""
    app = olustur_uygulama()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "istek_id": "test_req_001",
                "gorsel_meta": {"genislik": 1280, "yukseklik": 720, "format": "JPEG"},
                "nms_esigi": 0.45,
                "guven_esigi": 0.50
            }
            res = await client.post("/v1/tahmin/tekil", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["istek_id"] == "test_req_001"
            assert len(data["tespitler"]) > 0
            assert data["embedding"]["beklenen_boyut"] == 512


@pytest.mark.asyncio
async def test_toplu_tahmin_endpoint():
    """Toplu çıkarım POST /v1/tahmin/toplu (Batch) endpoint'ini test eder."""
    app = olustur_uygulama()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "batch_id": "test_batch_01",
                "istekler": [
                    {
                        "istek_id": f"batch_item_{i:02d}",
                        "gorsel_meta": {"genislik": 1920, "yukseklik": 1080, "format": "PNG"},
                        "nms_esigi": 0.45,
                        "guven_esigi": 0.50
                    }
                    for i in range(4)
                ]
            }
            res = await client.post("/v1/tahmin/toplu", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["toplam_islenen"] == 4
            assert len(data["sonuclar"]) == 4


@pytest.mark.asyncio
async def test_pydantic_gecersiz_istek_engelleme():
    """Geçersiz payload gönderildiğinde 422 HTTP hatası dönmesini test eder."""
    app = olustur_uygulama()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Geçersiz format ve negatif boyut
            gecersiz_payload = {
                "istek_id": "short",  # min_length=6 ihlali
                "gorsel_meta": {"genislik": -100, "yukseklik": 720, "format": "GIF"}
            }
            res = await client.post("/v1/tahmin/tekil", json=gecersiz_payload)
            assert res.status_code == 422


@pytest.mark.asyncio
async def test_dinamik_batch_kuyrugu():
    """Asenkron Dinamik Batch Kuyruğunu test eder."""
    app = olustur_uygulama()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test"):
            motor = app.state.model_motoru
            kuyruk = DinamikBatchKuyrugu(model_motoru=motor, maks_batch_boyutu=8, maks_bekleme_ms=10.0)
            kuyruk.baslat()

            async def worker(idx: int):
                return await kuyruk.tahmin_kuyruga_ekle({"istek_id": f"q_{idx}"})

            sonuclar = await asyncio.gather(*[worker(i) for i in range(16)])
            await kuyruk.durdur()

            assert len(sonuclar) == 16
            assert all(s["basarili"] for s in sonuclar)


def test_gorsellestirici_paneli(tmp_path):
    """6 panelli teşhis panosunun oluşturulmasını test eder."""
    bench = {
        "tekil_qps": 850.0,
        "batch_qps": 4200.0,
        "tekil_ortalama_gecikme_ms": 1.18,
        "batch_istek_basi_gecikme_ms": 0.24,
        "hizlanma_orani": 4.94
    }
    hedef = os.path.join(tmp_path, "test_fastapi_pano.png")
    cikis = FastAPIGorsellestirici.panel_ciz(bench, hedef_path=hedef)
    assert os.path.exists(cikis)
