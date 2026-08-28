---
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
- name: seydivakkas/minivit-cifar10-v1
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
config = AutoConfig.from_pretrained("seydivakkas/minivit-cifar10-v1")
model = AutoModelForImageClassification.from_pretrained("seydivakkas/minivit-cifar10-v1")
pipe = MiniViTPipeline(model, config)

# Çıkarım Yap
sonuc = pipe("ornek_resim.jpg", top_k=3)
print(sonuc)
```

## 📜 Lisans & Telif Hakkı
Özel Lisans — Tüm Hakları Saklıdır (c) 2026 Seydi Eryılmaz (@seydivakkas)