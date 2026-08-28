# 101 Günlük Yapay Zeka, Bilgisayarlı Görü, LLM/RAG ve MLOps Mühendisliği Master Projesi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Roadmap Status](https://img.shields.io/badge/Roadmap-101%2F101%20(%25100)%20Completed-gold?style=flat-square)](#)
[![Tests](https://img.shields.io/badge/tests-800%2B%20PASSED-success?style=flat-square)](#)

> **Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.**  
> Bu depo; sıfırdan ileri seviyeye derin öğrenme, tensör matematiği, konvolüsyonel sinir ağları, nesne tespiti/segmentasyon, doğal dil işleme, LLM ince ayar, RAG sistemleri, Vision Transformer'lar, model sıkıştırma, gözlemlenebilirlik, mikroservis mimarisi, yük testleri ve **Sparse Mixture of Experts (MoE)** dağıtımına kadar 101 günlük uçtan uca üretim seviyesinde (production-grade) mühendislik müfredatını içerir.

---

## 🏛️ Müfredat ve Faz Mimarisi

```
====================================================================================================
           101 GÜNLÜK YAPAY ZEKA VE MLOPS MASTER PROGRAMI — MİMARİ FAZLAR
====================================================================================================
• FAZ 1 (Gün 01 - 20) : Python Temelleri, PyTorch Çekirdeği, Tensör Matematiği ve CNN Mimarileri
• FAZ 2 (Gün 21 - 40) : Bilgisayarlı Görü, Nesne Tespiti (YOLO), Segmentasyon ve Veri Boru Hatları
• FAZ 3 (Gün 41 - 65) : Doğal Dil İşleme (NLP), Transformer, LLM Fine-Tuning ve RAG Mimarileri
• FAZ 4 (Gün 66 - 85) : Vision Transformers (ViT), Çok Modlu (Multimodal) Modeller ve CLIP
• FAZ 5 (Gün 86 - 101): Model Sıkıştırma (Pruning/INT8/FP16), Güvenilirlik, Determinizm, FastAPI,
                        Docker Konteynerleştirme, Yük Testleri, SwiGLU/RMSNorm ve BÜYÜK FİNAL: MoE v2!
====================================================================================================
```

---

## 📁 101 Günlük Tam Proje Dizini

| Gün | Modül / Proje Adı | Açıklama |
|---|---|---|
| **01-20** | `day-01` .. `day-20` | **FAZ 1**: Python Optimizasyonu, Autograd, CNN, ResNet, VGG, Veri Artırma |
| **21-40** | `day-21` .. `day-40` | **FAZ 2**: YOLOv8, U-Net Segmentasyon, Multi-Task Learning, Albumentations |
| **41-65** | `day-41` .. `day-65` | **FAZ 3**: NLP, Tokenizer, Self-Attention, BERT, GPT, LoRA Fine-Tuning, RAG Boru Hattı |
| **66-85** | `day-66` .. `day-85` | **FAZ 4**: Vision Transformer (ViT), Patch Embedding, MAE, CLIP, Multimodal Sistemler |
| **86** | [`day-86-model-pruning`](./day-86-model-pruning) | Yapısal Olmayan (Unstructured) ve Yapısal (Structured L1-Norm) Pruning |
| **87** | [`day-87-post-training-quantization`](./day-87-post-training-quantization) | INT8 Dinamik ve Statik Post-Training Kuantizasyon |
| **88** | [`day-88-quantization-aware-training`](./day-88-quantization-aware-training) | FakeQuantize ve Straight-Through Estimator ile QAT |
| **89** | [`day-89-onnx-tensorrt-export`](./day-89-onnx-tensorrt-export) | ONNX Runtime Dinamik Eksen Dışa Aktarımı ve TensorRT Optimizasyonu |
| **90** | [`day-90-dynamic-batching-inference`](./day-90-dynamic-batching-inference) | Asenkron Kuyruk ve Dinamik Batching Çıkarım Motoru |
| **91** | [`day-91-ai-observability`](./day-91-ai-observability) | P50/P95/P99 Gecikme, Throughput ve KS-Testi Veri Kayması (Drift) İzleme |
| **92** | [`day-92-final-training-contract`](./day-92-final-training-contract) | Deterministik Eğitim Sözleşmesi, Seed Yönetimi ve Donanım İzolasyonu |
| **93** | [`day-93-final-evaluation-model-card`](./day-93-final-evaluation-model-card) | Kapsamlı Değerlendirme, Hata Matrisi ve Standart Model Card |
| **94** | [`day-94-hugging-face-integration`](./day-94-hugging-face-integration) | Hugging Face Hub Entegrasyonu, Safetensors ve ImageProcessor |
| **95** | [`day-95-minivit-v1-release-candidate`](./day-95-minivit-v1-release-candidate) | MiniViT v1 Sürüm Adayı (RC) ve Uçtan Uca Regresyon Testleri |
| **96** | [`day-96-huggingface-public-v1-release`](./day-96-huggingface-public-v1-release) | MiniViT v1.0 Canlı Hugging Face Dağıtımı ve Pipeline Demosu |
| **97** | [`day-97-reproducible-inference`](./day-97-reproducible-inference) | Deterministik Çıkarım, Bit-Düzeyi SHA-256 Hash Doğrulama Testleri |
| **98** | [`day-98-fastapi-inference-service`](./day-98-fastapi-inference-service) | Üretime Hazır Asenkron FastAPI Servisi ve `/health`, `/ready` Probları |
| **99** | [`day-99-container-load-testing`](./day-99-container-load-testing) | Çok Aşamalı Docker Konteynerleştirme ve Locust Eşzamanlı Yük Testleri |
| **100** | [`day-100-modern-architecture-ablations`](./day-100-modern-architecture-ablations) | SwiGLU, RMSNorm ve SDPA (FlashAttention-2) ile MiniViT Ablasyon Analizleri |
| **101** | [`day-101-huggingface-minivit-moe-v2`](./day-101-huggingface-minivit-moe-v2) | **BÜYÜK FİNAL**: MiniViT Mixture of Experts (MoE) v2 Hugging Face Dağıtımı |

---

## 🚀 Başlangıç ve Çalıştırma

### 1. Gereksinimleri Yükleme

```bash
pip install -r day-101-huggingface-minivit-moe-v2/gereksinimler.txt
```

### 2. Büyük Final MoE Modelini Çalıştırma

```bash
# Birim testleri koşturun (8/8 PASSED)
pytest day-101-huggingface-minivit-moe-v2/testler -v

# Canlı dağıtım paketini ve teşhis panosunu üretin
python day-101-huggingface-minivit-moe-v2/ana_akis.py
```

### 3. FastAPI Servisini ve Docker Konteynerini Çalıştırma

```bash
# FastAPI Servisi (Day 98)
uvicorn day-98-fastapi-inference-service.src.api_uygulamasi:app --host 0.0.0.0 --port 8000

# Docker Konteyneri (Day 99)
docker build -t minivit-servis:latest day-99-container-load-testing/
docker run -p 8000:8000 minivit-servis:latest
```

---

## 📜 Lisans

```
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
amaçlı olarak paylaşılmıştır.

YASAKLAR:
  1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
  2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
  3. Alt lisanslanamaz, satılamaz veya devredilemez.
  4. Tersine mühendislik yapılamaz.

İZİN VERİLEN KULLANIM:
  - GitHub üzerinde görüntüleme ve okuma.
  - Kişisel öğrenim amacıyla kodu inceleme (kopyalamadan).

YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
İzin talepleri için: GitHub @seydivakkas
```
