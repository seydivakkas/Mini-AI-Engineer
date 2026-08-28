---
language:
- en
- tr
license: other
license_name: all-rights-reserved
license_link: LICENSE
tags:
- vision-transformer
- image-classification
- minivit
- pytorch
- safetensors
pipeline_tag: image-classification
widget:
- src: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg
  example_title: Kedi
---

# 🤗 seydivakkas/minivit-cifar10-v1

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Model Type: MiniViT](https://img.shields.io/badge/Architecture-MiniViT-blue.svg?style=flat-square)](#)
[![Format: SafeTensors](https://img.shields.io/badge/Format-SafeTensors-success.svg?style=flat-square)](#)

Bu model, **101 Günlük Yapay Zeka & MLOps Master Serisi** kapsamında geliştirilmiş, hafif ve yüksek performanslı bir **MiniViT (Vision Transformer)** görsel sınıflandırma modelidir.

## 🚀 Hızlı Başlangıç (Inference with Hugging Face)

```python
from transformers import AutoConfig, AutoModelForImageClassification
import torch

# 1. Modeli Hub veya Yerel Dizinden Yükle
model_yolu = "seydivakkas/minivit-cifar10-v1"
model = AutoModelForImageClassification.from_pretrained(model_yolu, trust_remote_code=True)
model.eval()

# 2. Örnek Çıkarım
dummy_pixel = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    outputs = model(dummy_pixel)
    tahmin_sinif = outputs.logits.argmax(dim=-1).item()

print(f"Tahmin Edilen Sınıf: {tahmin_sinif}")
```

## 📌 Model Konfigürasyonu
- **Görüntü Boyutu:** `32x32`
- **Yama Boyutu (Patch Size):** `4x4`
- **Gizli Boyut (Hidden Dim):** `128`
- **Katman Sayısı:** `4`
- **Dikkat Başlığı:** `4`
- **Sınıf Sayısı:** `10`

## 📜 Lisans ve Telif Hakkı
```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
