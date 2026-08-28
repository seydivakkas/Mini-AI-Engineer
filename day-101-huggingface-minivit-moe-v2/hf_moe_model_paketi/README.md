---
language:
- tr
- en
license: other
license_name: all-rights-reserved
license_link: LICENSE
tags:
- vision
- image-classification
- vision-transformer
- mixture-of-experts
- moe
- swiglu
- rmsnorm
- flash-attention
- pytorch
datasets:
- cifar10
metrics:
- accuracy
model-index:
- name: seydivakkas/minivit-moe-v2-cifar10
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      type: cifar10
      name: CIFAR-10
    metrics:
    - type: accuracy
      value: 86.8
      name: Test Accuracy (%)
---

# MiniViT-MoE v2: Sparse Mixture of Experts Vision Transformer

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)
[![Model Type: MoE Vision Transformer](https://img.shields.io/badge/architecture-Sparse%20MoE%20v2-purple.svg?style=flat-square)](#)
[![Accuracy](https://img.shields.io/badge/test_accuracy-86.8%25-brightgreen.svg?style=flat-square)](#)

MiniViT-MoE v2, **101 Günlük Yapay Zeka, Bilgisayarlı Görü, LLM/RAG ve MLOps Mühendisliği Master Programı**'nın Büyük Finalinde geliştirilmiş, üretim seviyesinde **Sparse Mixture of Experts (MoE)** Vision Transformer modelidir.

## Mimari Yenilikleri
- **Sparse MoE Katmanı**: 4 Uzman (Experts), Top-2 Routing.
- **Modern LLM Yapı Taşları**: Pre-RMSNorm, SwiGLU Uzmanları, PyTorch 2.0 SDPA (FlashAttention-2).
- **Yük Dengeleme Kaybı**: Switch Transformer / Mixtral standardında $\mathcal{L}_{\text{aux}}$ desteği.

## Model İstatistikleri
- **Toplam Parametre Kapasitesi**: 2,376,458 Parametre
- **Aktif Çıkarım Parametresi**: 1,328,906 Parametre (%50 FLOPs Tasarrufu)
- **P50 Çıkarım Gecikmesi**: 167.50 ms
- **Throughput**: 95 FPS

## Lisans
Özel Lisans — Tüm Hakları Saklıdır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
