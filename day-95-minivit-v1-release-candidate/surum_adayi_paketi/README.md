---
language: tr
license: other
license_name: custom-all-rights-reserved
tags:
- vision-transformer
- image-classification
- cifar10
- pytorch
- release-candidate
datasets:
- cifar10
metrics:
- accuracy
- f1
model-index:
- name: seydivakkas/minivit-cifar10
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
---

# 🤖 MiniViT v1.0 Sürüm Adayı (v1.0.0-rc1)

Bu model, **Mini Vision Transformer (MiniViT)** mimarisinin CIFAR-10 veri kümesi üzerinde eğitilmiş ve **v1.0-RC1** aşamasında dondurulmuş sürüm adayıdır.

- **Mimari:** Patch Embedding + Transformer Encoder + CLS Classifier Head
- **Sürüm Durumu:** `v1.0.0-rc1`
- **Lisans:** Özel Lisans — Tüm Hakları Saklıdır (c) 2026 Seydi Eryılmaz (@seydivakkas)