"""
FastAPI Mikroservis Uygulaması (Day 99 - Container Load Testing).
"""

import io
import time
import base64
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from pydantic import BaseModel, Field

from .konfigurasyon import MiniViTConfig
from .model import MiniViTForImageClassification


class TahminOgesi(BaseModel):
    sinif_adi: str
    sinif_id: int
    olasilik: float


class TahminYaniti(BaseModel):
    durum: str = "basarili"
    en_iyi_tahmin: TahminOgesi
    top_k_tahminler: List[TahminOgesi]
    gecikme_ms: float
    model_surumu: str = "v1.0"


class Base64Istegi(BaseModel):
    base64_goruntu: str
    top_k: int = Field(5, ge=1, le=10)


class ServisState:
    """Model durumunu ve metriklerini tutan sınıf."""
    def __init__(self):
        self.config = MiniViTConfig()
        self.cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MiniViTForImageClassification(self.config).to(self.cihaz)
        self.model.eval()
        self.baslangic = time.time()
        self.toplam_istek = 0
        self.gecikmeler: List[float] = []

        self.mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        self.std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

        # Isınma
        dummy = torch.randn(1, 3, 32, 32, device=self.cihaz)
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy)

    def tahmin_et(self, img: Image.Image, top_k: int = 5) -> TahminYaniti:
        t0 = time.perf_counter()
        img_rgb = img.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        arr = (np.array(img_rgb, dtype=np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.cihaz)

        with torch.no_grad():
            logits = self.model(tensor).logits
            probs = F.softmax(logits, dim=-1)[0]

        top_k = min(top_k, 10)
        top_probs, top_indices = torch.topk(probs, k=top_k)
        t1 = time.perf_counter()
        gecikme = (t1 - t0) * 1000.0

        self.toplam_istek += 1
        self.gecikmeler.append(gecikme)

        items = []
        for p, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            idx_int = int(idx)
            items.append(TahminOgesi(
                sinif_adi=self.config.id2label.get(idx_int, f"LABEL_{idx_int}"),
                sinif_id=idx_int,
                olasilik=float(p),
            ))

        return TahminYaniti(
            durum="basarili",
            en_iyi_tahmin=items[0],
            top_k_tahminler=items,
            gecikme_ms=round(gecikme, 2),
            model_surumu="v1.0",
        )


servis_state = ServisState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.servis = servis_state
    yield


app = FastAPI(
    title="MiniViT Docker Load Testing API",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.servis = servis_state


@app.get("/health", tags=["Gözlemlenebilirlik"])
@app.get("/healthz", tags=["Gözlemlenebilirlik"])
async def saglik():
    state: ServisState = app.state.servis
    return {
        "status": "HEALTHY",
        "model_loaded": True,
        "cihaz": state.cihaz.type,
        "toplam_istek": state.toplam_istek,
        "calisma_suresi_sn": round(time.time() - state.baslangic, 2),
    }


@app.get("/metrics", tags=["Gözlemlenebilirlik"])
async def metrikler():
    state: ServisState = app.state.servis
    g = state.gecikmeler if state.gecikmeler else [0.0]
    return {
        "toplam_istek": state.toplam_istek,
        "p50_ms": round(float(np.percentile(g, 50)), 2),
        "p90_ms": round(float(np.percentile(g, 90)), 2),
        "p99_ms": round(float(np.percentile(g, 99)), 2),
    }


@app.post("/predict", response_model=TahminYaniti, tags=["Çıkarım"])
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bozuk görsel: {str(e)}")

    state: ServisState = app.state.servis
    res = await asyncio.to_thread(state.tahmin_et, img, top_k)
    return res


@app.post("/predict/base64", response_model=TahminYaniti, tags=["Çıkarım"])
async def predict_base64(istek: Base64Istegi):
    try:
        b64 = istek.base64_goruntu
        if "," in b64:
            b64 = b64.split(",")[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Geçersiz base64: {str(e)}")

    state: ServisState = app.state.servis
    res = await asyncio.to_thread(state.tahmin_et, img, istek.top_k)
    return res
