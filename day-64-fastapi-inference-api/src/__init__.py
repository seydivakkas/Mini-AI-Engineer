"""
Day 64: Üretim Seviyesi FastAPI İnference, Model Yaşam Döngüsü (lifespan) ve Batch Prediction Paketi.
"""

from .model_motoru import YapayZekaModelMotoru
from .lifespan_yoneticisi import model_lifespan
from .api_servisi import app, olustur_uygulama
from .batch_kuyruk_yoneticisi import DinamikBatchKuyrugu
from .gorsellestirici import FastAPIGorsellestirici

__all__ = [
    "YapayZekaModelMotoru",
    "model_lifespan",
    "app",
    "olustur_uygulama",
    "DinamikBatchKuyrugu",
    "FastAPIGorsellestirici"
]
