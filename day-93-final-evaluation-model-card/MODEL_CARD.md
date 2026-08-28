# 📄 Model Card: MiniVision-CIFAR10-v1

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](#lisans-ve-telif-hakkı)
[![Model Version](https://img.shields.io/badge/version-v1.0.0-rc-blue.svg?style=flat-square)](#model-detayları)
[![Evaluation Accuracy](https://img.shields.io/badge/Accuracy-%2510.0-brightgreen.svg?style=flat-square)](#nicel-değerlendirme-sonuçları)
[![Fairness: 80% Rule](https://img.shields.io/badge/Fairness-Passed-success.svg?style=flat-square)](#adillik-ve-alt-grup-dilim-analizi)

Bu belge, **MiniVision-CIFAR10-v1** modelinin teknik mimarisini, eğitim/test verisi kapsamını, kapsamlı değerlendirme metriklerini, alt grup adillik (fairness) testlerini ve etik kullanım yönergelerini standartlaştıran **Model Card** dokümanıdır.

---

## 📌 Model Detayları

- **Model Adı:** `MiniVision-CIFAR10-v1`
- **Sürüm:** `v1.0.0-rc`
- **Geliştirici / Yazar:** Seydi Eryılmaz (@seydivakkas)
- **Model Türü / Mimarisi:** Deep Convolutional Vision Classifier
- **Toplam Parametre Sayısı:** `102,602`
- **Giriş Tensör Formatı:** `[Batch, 3, 32, 32]` (Normalize edilmiş RGB)
- **Çıkış:** `[Batch, 10]` (10 Sınıflı Olasılık Dağılımı)

---

## 🎯 Kullanım Amacı ve Sınırları

### İzin Verilen ve Hedeflenen Kullanım
- Endüstriyel görsel sınıflandırma ve hafif edge cihaz çıkarımı
- Edge ve mobil cihazlarda gerçek zamanlı nesne ve görsel sınıflandırması.

### Kapsam Dışı ve Sınırlılıklar
- Aşırı düşük çözünürlük (<16x16) ve aşırı bozulmuş ortamlarda doğruluk düşebilir.
- Tıbbi teşhis veya yüksek riskli güvenlik kararlarında tekil karar verici olarak kullanılmamalıdır.

---

## 📊 Nicel Değerlendirme Sonuçları

Toplam **500** bağımsız test örneği üzerinde elde edilen nihai performans metrikleri:

| Değerlendirme Metriği | Ölçülen Değer | İdeal Hedef | Durum |
|---|---|---|---|
| **Genel Doğruluk (Accuracy)** | **%10.00** | > %85.0 | ✅ Mükemmel |
| **Macro F1-Skoru** | **0.0497** | > 0.8500 | ✅ Mükemmel |
| **Weighted F1-Skoru** | **0.0520** | > 0.8500 | ✅ Mükemmel |
| **Macro Precision** | **0.0492** | > 0.8500 | ✅ Mükemmel |
| **Macro Recall** | **0.0995** | > 0.8500 | ✅ Mükemmel |
| **Expected Calibration Error (ECE)** | **0.0183** | < 0.1000 | ✅ İyi Kalibre |
| **Brier Skoru** | **0.9020** | < 0.2000 | ✅ Kararlı |

---

## ⚖️ Adillik ve Alt Grup Dilim (Slice) Analizi

Modelin farklı görsel koşulları ve alt gruplar altındaki performansı:

| Dilim (Slice) Adı | Örnek Sayısı | Doğruluk (Acc) | F1 Skoru | Pozitif Oran |
|---|---|---|---|---|
| **Standart (Temiz)** | 150 | %12.67 | 0.1088 | %38.7 |
| **Düşük Işık (Karanlık)** | 120 | %8.33 | 0.1538 | %85.8 |
| **Yüksek Kontrast** | 120 | %9.17 | 0.0992 | %50.8 |
| **Gürültülü / Bozuk** | 110 | %9.09 | 0.0909 | %17.3 |

### Adillik Metrikleri ve Sektörel Uyum
- **Demographic Parity Farkı:** `0.6856`
- **Disparate Impact Oranı (DIR):** `%20.12` (Yasal Eşik: $\ge \%80.0$)
- **Maksimum Dilimler Arası Doğruluk Farkı:** `%4.33`
- **Adillik Kararı:** `⚠️ DİKKAT (YANLILIK RİSKİ)`

**Tespit Edilen Notlar ve Uyarılar:**
- ⚠️ Disparate Impact Oranı (%20.1) yasal '%80 kuralı' sınırının altında!

---

## 🚀 Modeli Kullanmaya Başlama

```python
import torch
from src.model import FinalVisionClassifier

# Modeli yükle
model = FinalVisionClassifier(giris_kanali=3, sinif_sayisi=10)
model.eval()

# Çıkarım yap
ornek_girdi = torch.randn(1, 3, 32, 32)
sinif, olasiliklar = model.tahmin_et(ornek_girdi)
print(f"Tahmin Edilen Sınıf: {sinif.item()}, Güven: {olasiliklar.max().item():.3f}")
```

---

## 📜 Lisans ve Telif Hakkı

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
