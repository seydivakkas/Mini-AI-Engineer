"""
Hugging Face Spaces Canlı Gradio Giriş Noktası (app.py).
Hugging Face Spaces üzerinde doğrudan çalıştırılarak canlı web demo sunar.
"""

import os
import sys
import torch

# Modül yolunu ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.konfigurasyon import MiniViTConfig
from src.model import MiniViTForImageClassification
from src.dagitim_yoneticisi import HfDagitimYoneticisi, MiniViTPipeline
from src.canli_demo import GradioDemoOlusturucu


def app_baslat():
    """Gradio demo uygulamasını başlatır."""
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_paketi")

    dagitimci = HfDagitimYoneticisi()

    if os.path.exists(model_dizini) and os.path.exists(os.path.join(model_dizini, "model.safetensors")):
        pipe = dagitimci.yukle_ve_pipeline_kur(model_dizini)
    else:
        # Yedek sıfırdan model başlatma
        config = MiniViTConfig()
        model = MiniViTForImageClassification(config).to(cihaz)
        pipe = MiniViTPipeline(model, config)

    demo_olusturucu = GradioDemoOlusturucu(pipe)
    demo = demo_olusturucu.arayuz_olustur()
    return demo


if __name__ == "__main__":
    demo = app_baslat()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
