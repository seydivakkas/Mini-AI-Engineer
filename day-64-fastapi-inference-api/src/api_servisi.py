"""
FastAPI Üretim Seviyesi Model Çıkarım API Servisi (Production Inference Service).
"""

from typing import List, Optional, Dict, Any
import time
from fastapi import FastAPI, HTTPException, Request, Depends, status
from pydantic import BaseModel, Field, ConfigDict

from .lifespan_yoneticisi import model_lifespan
from .model_motoru import YapayZekaModelMotoru


# ---------------------------------------------------------------------
# Pydantic Girdi / Çıktı Şemaları
# ---------------------------------------------------------------------
class GorselGirdisi(BaseModel):
    model_config = ConfigDict(extra="forbid")
    genislik: int = Field(default=1920, gt=0, le=7680)
    yukseklik: int = Field(default=1080, gt=0, le=4320)
    format: str = Field(default="JPEG", pattern="^(JPEG|PNG|WEBP)$")


class TekilTahminIstegi(BaseModel):
    model_config = ConfigDict(extra="forbid")
    istek_id: str = Field(min_length=6, description="Benzersiz istek kimliği")
    gorsel_meta: GorselGirdisi = Field(default_factory=GorselGirdisi)
    nms_esigi: float = Field(default=0.45, ge=0.0, le=1.0)
    guven_esigi: float = Field(default=0.50, ge=0.0, le=1.0)


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class NesneTespiti(BaseModel):
    sinif_adi: str
    guven_skoru: float
    kutu: BoundingBox


class EmbeddingCiktisi(BaseModel):
    vektor: List[float]
    beklenen_boyut: int


class TekilTahminYaniti(BaseModel):
    istek_id: str
    model_adi: str
    tespitler: List[NesneTespiti]
    embedding: Optional[EmbeddingCiktisi] = None
    gecikme_ms: float
    basarili: bool = True


class TopluTahminIstegi(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_id: str = Field(min_length=6, description="Toplu istek kimliği")
    istekler: List[TekilTahminIstegi] = Field(min_length=1, max_length=64, description="En fazla 64 istek")


class TopluTahminYaniti(BaseModel):
    batch_id: str
    toplam_islenen: int
    toplam_gecikme_ms: float
    ortalama_istek_gecikme_ms: float
    sonuclar: List[TekilTahminYaniti]


# ---------------------------------------------------------------------
# Bağımlılık Enjeksiyonu (Dependency Injection)
# ---------------------------------------------------------------------
def model_motoru_al(request: Request) -> YapayZekaModelMotoru:
    """FastAPI Request state üzerinden başlatılmış model motorunu döndürür."""
    if not hasattr(request.app.state, "model_motoru") or not request.app.state.hazir:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model motoru henüz başlatılmadı veya hazır değil!"
        )
    return request.app.state.model_motoru


# ---------------------------------------------------------------------
# FastAPI Uygulama Fabrikası
# ---------------------------------------------------------------------
def olustur_uygulama() -> FastAPI:
    """Lifespan yöneticisi ve üretim endpoint'leri ile FastAPI uygulamasını oluşturur."""
    uygulama = FastAPI(
        title="Vision & Embedding AI Inference API",
        description="Üretim Seviyesi FastAPI İnference, Model Lifespan ve Batch Prediction Servisi",
        version="1.0.0",
        lifespan=model_lifespan
    )

    @uygulama.get("/", tags=["Sistem"])
    async def kok_dizin():
        return {
            "servis": "Vision & Embedding AI Inference API",
            "surum": "1.0.0",
            "durum": "aktif",
            "dokumantasyon": "/docs"
        }

    @uygulama.get("/saglik", tags=["Sistem"])
    async def saglik_kontrolu(request: Request):
        calisma_zamani_s = round(time.time() - getattr(request.app.state, "baslatma_zamani", time.time()), 2)
        hazir = getattr(request.app.state, "hazir", False)
        return {
            "durum": "saglikli" if hazir else "baslatiliyor",
            "hazir": hazir,
            "calisma_suresi_s": calisma_zamani_s,
            "model_adi": getattr(request.app.state, "model_motoru", None).model_adi if hazir else None
        }

    @uygulama.post("/v1/tahmin/tekil", response_model=TekilTahminYaniti, tags=["Cikarim"])
    async def tekil_tahmin(
        istek: TekilTahminIstegi,
        motor: YapayZekaModelMotoru = Depends(model_motoru_al)
    ):
        """Tekil görsel çıkarımı gerçekleştirir."""
        sonuc = motor.tekil_tahmin(
            istek_id=istek.istek_id,
            genislik=istek.gorsel_meta.genislik,
            yukseklik=istek.gorsel_meta.yukseklik,
            nms_esigi=istek.nms_esigi,
            guven_esigi=istek.guven_esigi
        )
        return sonuc

    @uygulama.post("/v1/tahmin/toplu", response_model=TopluTahminYaniti, tags=["Cikarim"])
    async def toplu_tahmin(
        toplu_istek: TopluTahminIstegi,
        motor: YapayZekaModelMotoru = Depends(model_motoru_al)
    ):
        """Maksimum 64 elemanlı toplu görsel çıkarımı (Batch Inference) gerçekleştirir."""
        start_t = time.perf_counter()
        ham_istekler = [req.model_dump() for req in toplu_istek.istekler]
        sonuclar = motor.toplu_tahmin(ham_istekler)
        total_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "batch_id": toplu_istek.batch_id,
            "toplam_islenen": len(sonuclar),
            "toplam_gecikme_ms": round(total_ms, 2),
            "ortalama_istek_gecikme_ms": round(total_ms / max(len(sonuclar), 1), 2),
            "sonuclar": sonuclar
        }

    @uygulama.get("/v1/metrikler", tags=["Gozlemlenebilirlik"])
    async def servis_metrikleri(motor: YapayZekaModelMotoru = Depends(model_motoru_al)):
        """İnference motorunun anlık telemetri metriklerini döndürür."""
        ort_gecikme = (motor.toplam_gecikme_ms / max(motor.toplam_istek_sayisi, 1))
        return {
            "toplam_islenen_istek": motor.toplam_istek_sayisi,
            "toplam_islenen_batch": motor.toplam_batch_sayisi,
            "ortalama_gecikme_ms": round(ort_gecikme, 2)
        }

    return uygulama


app = olustur_uygulama()
