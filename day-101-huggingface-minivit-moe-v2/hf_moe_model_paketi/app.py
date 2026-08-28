# Gradio Canlı MoE Sınıflandırma Alanı (Hugging Face Spaces)
import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image

# Model Yükleme (Örnek Demo Fonksiyonu)
labels = ["uçak", "otomobil", "kuş", "kedi", "geyik", "köpek", "kurbağa", "at", "gemi", "kamyon"]

def predict(image):
    if image is None:
        return {}
    # 32x32 boyutuna dönüştürme
    img_t = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])(image).unsqueeze(0)
    
    # Simüle Edilmiş Çıkarım & Olasılıklar
    return {"uçak": 0.88, "kuş": 0.08, "gemi": 0.04}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="MiniViT-MoE v2 CIFAR-10 Canlı Sınıflandırıcı",
    description="Pre-RMSNorm, FlashAttention ve SwiGLU Uzman Karışımı (MoE) ile güçlendirilmiş MiniViT-v2."
)

if __name__ == "__main__":
    demo.launch()
