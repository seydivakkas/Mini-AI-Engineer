"""
Locust Eşzamanlı Yük ve Stres Testi Tanımları (Day 99).
"""

import io
import base64
from PIL import Image
from locust import HttpUser, task, between


class MiniViTKullanicisi(HttpUser):
    """MiniViT API servisine yük oluşturan simüle edilmiş istemci."""
    wait_time = between(0.01, 0.05)  # İstekler arası bekleme (10ms - 50ms)

    def on_start(self):
        """Kullanıcı başlatıldığında sentetik test görselini hazırlar."""
        img = Image.new("RGB", (32, 32), color=(60, 120, 210))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.img_bytes = buf.getvalue()
        self.b64_str = base64.b64encode(self.img_bytes).decode("utf-8")

    @task(6)
    def test_predict_multipart(self):
        """En sık kullanılan tekli görsel çıkarım endpoint'i."""
        files = {"file": ("test.png", self.img_bytes, "image/png")}
        self.client.post("/predict?top_k=5", files=files, name="/predict (Multipart)")

    @task(3)
    def test_predict_base64(self):
        """Base64 formatında JSON çıkarım endpoint'i."""
        payload = {"base64_goruntu": self.b64_str, "top_k": 3}
        self.client.post("/predict/base64", json=payload, name="/predict/base64 (JSON)")

    @task(1)
    def test_health_check(self):
        """Kubernetes sağlık kontrolü endpoint'i."""
        self.client.get("/health", name="/health")
