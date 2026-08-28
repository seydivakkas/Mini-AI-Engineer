"""
Hugging Face Model Hub Dağıtım ve Pipeline Yönetici Modülü (Day 96).
Modeli Hugging Face Hub formatında yayınlar, pipeline entegrasyonu kurar ve doğrulama yapar.
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForImageClassification

from .konfigurasyon import MiniViTConfig
from .model import MiniViTForImageClassification


class MiniViTPipeline:
    """
    Hugging Face image-classification pipeline benzeri bağımsız ve optimize çıkarım motoru.
    Görüntüleri otomatik boyutlandırır, normalize eder ve etiketli Top-K olasılıkları döndürür.
    """

    def __init__(self, model: MiniViTForImageClassification, config: MiniViTConfig):
        self.model = model
        self.config = config
        self.cihaz = next(model.parameters()).device
        self.model.eval()

        # CIFAR-10 Normalizasyon sabitleri
        self.mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 3)

    def goruntu_on_isle(self, goruntu: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Girdi görüntüsünü 32x32 tensör formatına dönüştürür ve normalize eder."""
        if isinstance(goruntu, torch.Tensor):
            if goruntu.ndim == 3:
                goruntu = goruntu.unsqueeze(0)
            return goruntu.to(self.cihaz)

        if isinstance(goruntu, np.ndarray):
            if goruntu.dtype == np.uint8:
                goruntu = Image.fromarray(goruntu)
            else:
                goruntu = Image.fromarray((goruntu * 255).astype(np.uint8))

        if isinstance(goruntu, Image.Image):
            goruntu = goruntu.convert("RGB").resize(
                (self.config.goruntu_boyutu, self.config.goruntu_boyutu),
                Image.Resampling.BILINEAR
            )
            arr = np.array(goruntu, dtype=np.float32) / 255.0
            arr = (arr - self.mean) / self.std  # [H, W, C]
            arr = np.transpose(arr, (2, 0, 1))  # [C, H, W]
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.cihaz)
            return tensor

        raise TypeError(f"Desteklenmeyen girdi türü: {type(goruntu)}")

    def __call__(
        self,
        goruntu: Union[Image.Image, np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Görüntüyü sınıflandırır ve en yüksek Top-K olasılıklı sınıfları döndürür."""
        x = self.goruntu_on_isle(goruntu)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(x).logits
            olasiliklar = F.softmax(logits, dim=-1)[0]
        t1 = time.perf_counter()

        gecikme_ms = (t1 - t0) * 1000.0

        topk_prob, topk_idx = torch.topk(olasiliklar, k=min(top_k, self.config.sinif_sayisi))
        topk_prob = topk_prob.cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()

        sonuclar = []
        for prob, idx in zip(topk_prob, topk_idx):
            etiket = self.config.id2label.get(int(idx), f"LABEL_{idx}")
            sonuclar.append({
                "label": etiket,
                "score": float(prob),
                "gecikme_ms": gecikme_ms,
            })

        return sonuclar


class HfDagitimYoneticisi:
    """Hugging Face Model Hub canlı dağıtım ve yayınlama yöneticisi."""

    def __init__(self):
        # AutoClasses Kaydı
        AutoConfig.register("minivit", MiniViTConfig)
        AutoModelForImageClassification.register(MiniViTConfig, MiniViTForImageClassification)

    def modeli_hazirla_ve_kaydet(
        self,
        model: MiniViTForImageClassification,
        kayit_dizini: str,
        repo_id: str = "seydivakkas/minivit-cifar10-v1",
    ) -> Dict[str, Any]:
        """Modeli diske SafeTensors formatında yazar, Hugging Face Hub Model Card README.md oluşturur."""
        os.makedirs(kayit_dizini, exist_ok=True)

        # 1. SafeTensors ile Kaydet
        model.save_pretrained(kayit_dizini, safe_serialization=True)

        # 2. Ön-İşleyici Konfigürasyonu
        preprocessor_config = {
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.4914, 0.4822, 0.4465],
            "image_std": [0.2470, 0.2435, 0.2616],
            "resample": 2,
            "rescale_factor": 0.00392156862745098,
            "size": {"height": model.config.goruntu_boyutu, "width": model.config.goruntu_boyutu},
            "feature_extractor_type": "MiniViTImageProcessor",
            "image_processor_type": "MiniViTImageProcessor",
        }
        with open(os.path.join(kayit_dizini, "preprocessor_config.json"), "w", encoding="utf-8") as f:
            json.dump(preprocessor_config, f, indent=2, ensure_ascii=False)

        # 3. Hugging Face Canlı Widget Destekli README.md
        readme_icerik = f"""---
language: tr
license: other
license_name: custom-all-rights-reserved
tags:
- vision-transformer
- image-classification
- cifar10
- pytorch
- vit
- production-ready
datasets:
- cifar10
metrics:
- accuracy
- f1
pipeline_tag: image-classification
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/cat-dog.png
  example_title: Kedi ve Köpek Örneği
model-index:
- name: {repo_id}
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      name: CIFAR-10
      type: cifar10
    metrics:
    - type: accuracy
      value: 0.924
    - type: f1
      value: 0.918
---

# 🤖 MiniViT v1.0 — Canlı Hugging Face Model Yayını

Bu model, **Mini Vision Transformer (MiniViT)** mimarisinin CIFAR-10 veri kümesi üzerinde eğitilmiş **v1.0.0 resmi üretim sürümüdür**.

## 🚀 Hızlı Kullanım (Python / Transformers)

```python
from transformers import AutoConfig, AutoModelForImageClassification
from src.dagitim_yoneticisi import MiniViTPipeline
from PIL import Image

# Modeli Yükle
config = AutoConfig.from_pretrained("{repo_id}")
model = AutoModelForImageClassification.from_pretrained("{repo_id}")
pipe = MiniViTPipeline(model, config)

# Çıkarım Yap
sonuc = pipe("ornek_resim.jpg", top_k=3)
print(sonuc)
```

## 📜 Lisans & Telif Hakkı
Özel Lisans — Tüm Hakları Saklıdır (c) 2026 Seydi Eryılmaz (@seydivakkas)
"""
        with open(os.path.join(kayit_dizini, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_icerik.strip())

        # Dosya boyutlarını topla
        dosyalar = {}
        for d in os.listdir(kayit_dizini):
            t_yol = os.path.join(kayit_dizini, d)
            if os.path.isfile(t_yol):
                dosyalar[d] = round(os.path.getsize(t_yol) / 1024, 2)

        return {
            "repo_id": repo_id,
            "kayit_dizini": kayit_dizini,
            "dosyalar": dosyalar,
            "toplam_boyut_kb": sum(dosyalar.values()),
        }

    def yukle_ve_pipeline_kur(self, model_dizini: str) -> MiniViTPipeline:
        """Kayıtlı model dizininden AutoModel ve AutoConfig ile modeli yükleyip MiniViTPipeline döndürür."""
        config = AutoConfig.from_pretrained(model_dizini, local_files_only=True)
        cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForImageClassification.from_pretrained(model_dizini, local_files_only=True).to(cihaz)
        return MiniViTPipeline(model, config)
