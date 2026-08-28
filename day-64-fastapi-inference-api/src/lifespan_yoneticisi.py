"""
FastAPI Modern Model Yaşam Döngüsü (Lifespan Context Manager).
"""

from contextlib import asynccontextmanager
import time
from fastapi import FastAPI

from .model_motoru import YapayZekaModelMotoru


@asynccontextmanager
async def model_lifespan(app: FastAPI):
    """
    FastAPI uygulamasının başlatılma (startup) ve kapatılma (shutdown) yaşam döngüsünü yönetir.
    Eski @app.on_event("startup") yerine modern async context manager standardıdır.
    """
    # -------------------------------------------------------------
    # 1. Başlangıç (Startup): Model Yükleme ve Isınma (Warmup)
    # -------------------------------------------------------------
    baslangic = time.time()
    motor = YapayZekaModelMotoru(model_adi="MiniVision-YOLOv8-Embedder", embedding_boyutu=512)
    motor.yukle_ve_isinma()

    # Uygulama durumuna (app.state) global bağımlılık olarak enjekte et
    app.state.model_motoru = motor
    app.state.baslatma_zamani = baslangic
    app.state.hazir = True

    try:
        yield
    finally:
        # -------------------------------------------------------------
        # 2. Kapanış (Shutdown): Kaynakları ve GPU Belleğini Boşaltma
        # -------------------------------------------------------------
        app.state.hazir = False
        if hasattr(app.state, "model_motoru"):
            app.state.model_motoru.yuklendi = False
