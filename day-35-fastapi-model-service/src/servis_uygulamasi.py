"""
FastAPI Asenkron AI Model Servisi & REST API Uygulaması.
"""

from typing import Dict, Any, List
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, status, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .semalar import (
    SaglikYaniti,
    MetinTahminIstegi,
    MetinTahminYaniti,
    GorselAnalizYaniti,
    RAGSorguIstegi,
    RAGSorguYaniti,
    HataDetayi
)
from .model_motoru import AIModelMotoru


class TelemetriKaydedici:
    """Sunucu istek sayılarını ve gecikmelerini tutan bellek içi telemetri."""

    def __init__(self):
        self.toplam_istek: int = 0
        self.gecikmeler: List[float] = []
        self.endpoint_sayaclari: Dict[str, int] = {}

    def kaydet(self, endpoint: str, gecikme_ms: float):
        self.toplam_istek += 1
        self.gecikmeler.append(gecikme_ms)
        self.endpoint_sayaclari[endpoint] = self.endpoint_sayaclari.get(endpoint, 0) + 1


telemetri = TelemetriKaydedici()
model_motoru: AIModelMotoru = None


def arka_plan_gunluk_kaydi(endpoint: str, veri_ozeti: str, gecikme_ms: float):
    """Yanıt döndükten sonra çalışan asenkron arka plan görevi."""
    telemetri.kaydet(endpoint, gecikme_ms)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama açılış ve kapanış yaşam döngüsü (Lifespan)."""
    global model_motoru
    model_motoru = AIModelMotoru(device="cpu")
    yield
    model_motoru = None


def servis_olustur() -> FastAPI:
    """FastAPI uygulama örneğini yapılandırır."""
    uygulama = FastAPI(
        title="Day 35: Asenkron AI Model Servisi",
        description="FastAPI, Pydantic v2 ve PyTorch tabanlı üretim seviyesi AI REST API.",
        version="1.0.0",
        lifespan=lifespan
    )

    # -------------------------------------------------------------
    # 1. Gecikme Ölçüm Middleware (Process Time Header)
    # -------------------------------------------------------------
    @uygulama.middleware("http")
    async def gecikme_middleware(request: Request, call_next):
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        gecikme_ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{gecikme_ms:.2f}"
        return response

    # -------------------------------------------------------------
    # 2. Özel Hata Yakalayıcılar (Exception Handlers)
    # -------------------------------------------------------------
    @uygulama.exception_handler(RequestValidationError)
    async def dogrulama_hatasi_yakala(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "hata": "Girdi Doğrulama Hatası",
                "mesaj": str(exc.errors()[0]["msg"]) if exc.errors() else "Geçersiz şema",
                "kod": 422
            }
        )

    # -------------------------------------------------------------
    # 3. Uç Noktalar (Endpoints)
    # -------------------------------------------------------------
    @uygulama.get("/healthz", response_model=SaglikYaniti, tags=["Sistem"])
    async def saglik_kontrolu():
        """Liveness ve Readiness probu."""
        return SaglikYaniti(
            durum="aktif",
            versiyon="1.0.0",
            cihaz="cpu",
            yuklu_modeller=["MiniMetinSiniflandirici", "GorselAnalizMotoru", "MiniRAG"]
        )

    @uygulama.post("/api/v1/predict/text", response_model=MetinTahminYaniti, tags=["Metin AI"])
    async def metin_tahmini_yap(
        istek: MetinTahminIstegi,
        arka_plan: BackgroundTasks
    ):
        """Asenkron metin konu sınıflandırması ve embedding çıkarımı."""
        global model_motoru
        if model_motoru is None:
            model_motoru = AIModelMotoru(device="cpu")

        sonuc = await model_motoru.metin_tahmin_et(istek.metin, istek.embedding_iste)
        arka_plan.add_task(arka_plan_gunluk_kaydi, "/api/v1/predict/text", istek.metin[:30], sonuc["gecikme_ms"])
        return MetinTahminYaniti(**sonuc)

    @uygulama.post("/api/v1/predict/image", response_model=GorselAnalizYaniti, tags=["Görüntü AI"])
    async def gorsel_analiz_yap(
        arka_plan: BackgroundTasks,
        dosya: UploadFile = File(..., description="Analiz edilecek görsel dosyası")
    ):
        """Görsel yükleme ve nesne tespiti / renk analizi."""
        global model_motoru
        if model_motoru is None:
            model_motoru = AIModelMotoru(device="cpu")

        if not dosya.filename:
            raise HTTPException(status_code=400, detail="Dosya adı boş olamaz.")

        icerik = await dosya.read()
        sonuc = await model_motoru.gorsel_analiz_et(dosya.filename, icerik)
        arka_plan.add_task(arka_plan_gunluk_kaydi, "/api/v1/predict/image", dosya.filename, sonuc["gecikme_ms"])
        return GorselAnalizYaniti(**sonuc)

    @uygulama.post("/api/v1/rag/query", response_model=RAGSorguYaniti, tags=["RAG Soru-Cevap"])
    async def rag_sorgusu_yap(
        istek: RAGSorguIstegi,
        arka_plan: BackgroundTasks
    ):
        """RAG doküman sorgulama ve kaynak atıflı yanıt."""
        global model_motoru
        if model_motoru is None:
            model_motoru = AIModelMotoru(device="cpu")

        sonuc = await model_motoru.rag_sorgula(istek.soru, istek.top_k)
        arka_plan.add_task(arka_plan_gunluk_kaydi, "/api/v1/rag/query", istek.soru[:30], sonuc["gecikme_ms"])
        return RAGSorguYaniti(**sonuc)

    @uygulama.get("/api/v1/telemetry", tags=["Telemetri"])
    async def telemetri_al():
        """Sistem istek sayıları ve ortalama gecikme metrikleri."""
        gecikmeler = telemetri.gecikmeler
        ort_gecikme = sum(gecikmeler) / len(gecikmeler) if gecikmeler else 0.0
        return {
            "toplam_istek": telemetri.toplam_istek,
            "ortalama_gecikme_ms": float(ort_gecikme),
            "endpoint_istek_dagilimi": telemetri.endpoint_sayaclari
        }

    return uygulama


app = servis_olustur()
