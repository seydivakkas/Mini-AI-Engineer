"""
FastAPI Asenkron Çıkarım Servisi ve Endpoint Tanımları (Day 98).
"""

import io
import asyncio
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image

from .semalar import (
    TahminYaniti,
    TopluTahminYaniti,
    Base64Istegi,
    SaglikYaniti,
    ModelMetaveriYaniti,
)
from .servis_yoneticisi import ServisYoneticisi


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Servis yaşam döngüsü yöneticisi (Başlatma ve Kapanış)."""
    # Model belleğe yüklenir ve ısıtılır
    yonetici = ServisYoneticisi.get_instance()
    app.state.yonetici = yonetici
    yield
    # Temizlik işlemleri (gerekirse)


app = FastAPI(
    title="MiniViT v1.0 Asenkron Çıkarım Servisi",
    version="1.0.0",
    description="Vision Transformer tabanlı CIFAR-10 görüntü sınıflandırma mikroservisi.",
    lifespan=lifespan,
)


@app.get("/", tags=["Genel"])
async def kok_dizin():
    """Servis genel karşılama endpoint'i."""
    return {
        "servis": "MiniViT v1.0 Asenkron Çıkarım API",
        "durum": "CALISIYOR",
        "dokumantasyon": "/docs",
        "saglik_kontrolu": "/health",
    }


@app.get("/health", response_model=SaglikYaniti, tags=["Sağlık & Gözlemlenebilirlik"])
@app.get("/healthz", response_model=SaglikYaniti, tags=["Sağlık & Gözlemlenebilirlik"])
@app.get("/ready", response_model=SaglikYaniti, tags=["Sağlık & Gözlemlenebilirlik"])
async def saglik_kontrolu():
    """Kubernetes Liveness & Readiness Probeları için sağlık kontrolü."""
    yonetici: ServisYoneticisi = app.state.yonetici
    return yonetici.saglik_raporu()


@app.get("/metadata", response_model=ModelMetaveriYaniti, tags=["Genel"])
async def model_metaveri():
    """Model mimarisi ve sınıf etiketleri metaverisini döner."""
    yonetici: ServisYoneticisi = app.state.yonetici
    return yonetici.metaveri_raporu()


@app.get("/metrics", tags=["Sağlık & Gözlemlenebilirlik"])
async def servis_metrikleri():
    """İstek sayıları ve gecikme yüzdelikleri (P50, P90, P99)."""
    yonetici: ServisYoneticisi = app.state.yonetici
    return yonetici.metrik_raporu()


@app.post("/predict", response_model=TahminYaniti, tags=["Çıkarım"])
async def tekli_tahmin(
    file: UploadFile = File(..., description="Sınıflandırılacak görüntü dosyası (JPEG/PNG/WebP)"),
    top_k: int = Query(5, ge=1, le=10, description="Döndürülecek Top-K sınıf sayısı"),
):
    """Multipart dosya yüklemesi ile tekli görüntü sınıflandırma."""
    yonetici: ServisYoneticisi = app.state.yonetici

    try:
        icerik = await file.read()
        gorsel = Image.open(io.BytesIO(icerik))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Geçersiz görsel dosyası: {str(e)}",
        )

    # Event loop'u bloklamamak için çıkarımı iş parçacığı havuzuna devret
    sonuc = await asyncio.to_thread(yonetici.tahmin_et, gorsel, top_k)
    return sonuc


@app.post("/predict/base64", response_model=TahminYaniti, tags=["Çıkarım"])
async def base64_tahmin(istek: Base64Istegi):
    """Base64 kodlu JSON gövdesi ile tekli görüntü sınıflandırma."""
    yonetici: ServisYoneticisi = app.state.yonetici

    try:
        gorsel = yonetici.base64_coz(istek.base64_goruntu)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Base64 dizgisi çözülemedi: {str(e)}",
        )

    sonuc = await asyncio.to_thread(yonetici.tahmin_et, gorsel, istek.top_k)
    return sonuc


@app.post("/predict/batch", response_model=TopluTahminYaniti, tags=["Çıkarım"])
async def toplu_tahmin(
    files: List[UploadFile] = File(..., description="Sınıflandırılacak görüntü dosyaları listesi"),
    top_k: int = Query(5, ge=1, le=10, description="Döndürülecek Top-K sınıf sayısı"),
):
    """Çoklu dosya yüklemesi ile toplu görüntü sınıflandırma."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="En az bir görüntü dosyası yüklenmelidir.",
        )

    yonetici: ServisYoneticisi = app.state.yonetici
    gorseller: List[Image.Image] = []

    for f in files:
        try:
            icerik = await f.read()
            img = Image.open(io.BytesIO(icerik))
            gorseller.append(img)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dosya okuma hatası ({f.filename}): {str(e)}",
            )

    sonuc = await asyncio.to_thread(yonetici.toplu_tahmin_et, gorseller, top_k)
    return sonuc
