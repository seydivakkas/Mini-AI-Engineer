"""Day 20: TensorFlow/Keras ile Derin Öğrenme Görsel Sınıflandırma Modülü.

Conv2D, MaxPooling, Flatten, Dense ve Dropout katmanları ile modern CNN mimarisi,
veri ön işleme, model eğitimi ve teşhis görselleştirme araçları.
"""

import os
# Keras backend ayarı (varsayılan torch veya tensorflow)
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

from src.model_mimari import build_cnn_model
from src.veri_hazirlayici import VeriHazirlayici
from src.egitici import ModelEgitici, EgitimSonucu
from src.gorsellestirici import CNNGorsellestirici
from src.aktivasyon_cikarici import AraKatmanAktivasyonCikarici

__all__ = [
    "build_cnn_model",
    "VeriHazirlayici",
    "ModelEgitici",
    "EgitimSonucu",
    "CNNGorsellestirici",
    "AraKatmanAktivasyonCikarici",
]
