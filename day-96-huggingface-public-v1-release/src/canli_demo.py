"""
Hugging Face Spaces Canlı Gradio Demo Modülü (Day 96).
MiniViT v1.0 modeli için etkileşimli Gradio Web kullanıcı arayüzü inşa eder.
"""

from typing import Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
import gradio as gr

from .dagitim_yoneticisi import MiniViTPipeline


class GradioDemoOlusturucu:
    """Hugging Face Spaces uyumlu interaktif Gradio demo motoru."""

    def __init__(self, pipeline: MiniViTPipeline):
        self.pipeline = pipeline

    def siniflandir(self, goruntu: Image.Image) -> Tuple[Dict[str, float], str]:
        """Gradio girdi görüntüsünü alır, model tahminlerini ve çıkarım gecikmesini döndürür."""
        if goruntu is None:
            return {}, "Lütfen bir görüntü yükleyin."

        sonuclar = self.pipeline(goruntu, top_k=5)
        if not sonuclar:
            return {}, "Tahmin üretilemedi."

        # Gradio Label formatı: {etiket: olasilik}
        olasilik_haritasi = {item["label"]: round(item["score"], 4) for item in sonuclar}
        gecikme_ms = sonuclar[0]["gecikme_ms"]
        gecikme_bilgisi = f"⏱️ Model Çıkarım Gecikmesi: {gecikme_ms:.2f} ms"

        return olasilik_haritasi, gecikme_bilgisi

    def arayuz_olustur(self) -> gr.Blocks:
        """Hugging Face Spaces için şık ve modern Gradio Blocks arayüzü oluşturur."""
        baslik = "Mini Vision Transformer (MiniViT v1.0) — Canlı CIFAR-10 Sınıflandırma"
        aciklama = (
            "Bu demo, sıfırdan eğitilmiş ve **Hugging Face Model Hub** standartlarında paketlenmiş "
            "**MiniViT v1.0** Vision Transformer mimarisinin canlı çıkarım arayüzüdür.\n\n"
            "**Sınıflar (CIFAR-10):** Uçak, Otomobil, Kuş, Kedi, Geyik, Köpek, Kurbağa, At, Gemi, Kamyon."
        )

        with gr.Blocks(title="MiniViT v1.0 Canlı Demo", theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"# 🤖 {baslik}")
            gr.Markdown(aciklama)

            with gr.Row():
                with gr.Column():
                    girdi_gorsel = gr.Image(type="pil", label="Girdi Görüntüsü Yükleyin (veya Çizin)")
                    siniflandir_butonu = gr.Button("🚀 Görüntüyü Sınıflandır", variant="primary")

                with gr.Column():
                    cikti_etiketler = gr.Label(num_top_classes=5, label="Tahmin Olasılıkları (Top-5)")
                    cikti_gecikme = gr.Textbox(label="Performans Bilgisi", interactive=False)

            siniflandir_butonu.click(
                fn=self.siniflandir,
                inputs=[girdi_gorsel],
                outputs=[cikti_etiketler, cikti_gecikme],
            )

            gr.Markdown("---")
            gr.Markdown(
                "**Geliştirici:** Seydi Eryılmaz (@seydivakkas) | "
                "**Lisans:** Özel Lisans — Tüm Hakları Saklıdır (c) 2026"
            )

        return demo
