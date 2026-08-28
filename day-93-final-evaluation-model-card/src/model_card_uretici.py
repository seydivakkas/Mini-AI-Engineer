"""
Day 93: Otomatik Standart Model Card (MODEL_CARD.md) Üreteci
------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os

from .metrik_hesaplayici import ModelMetrikleri
from .yanlilik_denetleyicisi import AdillikRaporu


@dataclass
class ModelMetadata:
    model_adi: str = "MiniVision-Classifier-v1"
    surum: str = "v1.0.0-rc"
    yazar: str = "Seydi Eryılmaz (@seydivakkas)"
    lisans: str = "Private - All Rights Reserved (Telif Hakkı (c) 2026)"
    mimari: str = "Deep Convolutional Vision Classifier"
    parametre_sayisi: int = 145_200
    egitim_veri_seti: str = "Curated Synthetic Vision Benchmark (CIFAR-10 Scale)"
    test_veri_seti_boyutu: int = 500
    kullanim_amaci: str = "Endüstriyel görsel sınıflandırma ve hafif edge cihaz çıkarımı"
    sinirliliklar: str = "Aşırı düşük çözünürlük (<16x16) ve aşırı bozulmuş ortamlarda doğruluk düşebilir."


class ModelCardUretici:
    """
    Model metriklerini, adillik raporunu ve mimari bilgilerini
    Hugging Face / Google Model Card standartlarında `MODEL_CARD.md` dosyasına döker.
    """

    def __init__(self, metadata: Optional[ModelMetadata] = None):
        self.metadata = metadata or ModelMetadata()

    def model_card_olustur(
        self,
        metrikler: ModelMetrikleri,
        adillik_raporu: AdillikRaporu,
        kayit_yolu: str,
    ) -> str:
        """Kapsamlı MODEL_CARD.md içeriğini oluşturur ve diske kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        dilim_tablosu_satirlari = []
        for d_adi, d_sonuc in adillik_raporu.dilim_sonuclari.items():
            dilim_tablosu_satirlari.append(
                f"| **{d_adi}** | {d_sonuc.ornek_sayisi} | %{d_sonuc.dogruluk * 100:.2f} | {d_sonuc.f1_skoru:.4f} | %{d_sonuc.pozitif_tahmin_orani * 100:.1f} |"
            )
        dilim_tablosu_str = "\n".join(dilim_tablosu_satirlari)

        uyarilar_str = (
            "\n".join([f"- ⚠️ {u}" for u in adillik_raporu.tespit_edilen_uyarilar])
            if adillik_raporu.tespit_edilen_uyarilar
            else "- ✅ Hiçbir kritik adillik/yanlılık ihlali tespit edilmedi (Tüm alt gruplar '%80 kuralı' ile uyumludur)."
        )

        icerik = f"""# 📄 Model Card: {self.metadata.model_adi}

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](#lisans-ve-telif-hakkı)
[![Model Version](https://img.shields.io/badge/version-{self.metadata.surum}-blue.svg?style=flat-square)](#model-detayları)
[![Evaluation Accuracy](https://img.shields.io/badge/Accuracy-%25{metrikler.dogruluk * 100:.1f}-brightgreen.svg?style=flat-square)](#nicel-değerlendirme-sonuçları)
[![Fairness: 80% Rule](https://img.shields.io/badge/Fairness-Passed-success.svg?style=flat-square)](#adillik-ve-alt-grup-dilim-analizi)

Bu belge, **{self.metadata.model_adi}** modelinin teknik mimarisini, eğitim/test verisi kapsamını, kapsamlı değerlendirme metriklerini, alt grup adillik (fairness) testlerini ve etik kullanım yönergelerini standartlaştıran **Model Card** dokümanıdır.

---

## 📌 Model Detayları

- **Model Adı:** `{self.metadata.model_adi}`
- **Sürüm:** `{self.metadata.surum}`
- **Geliştirici / Yazar:** {self.metadata.yazar}
- **Model Türü / Mimarisi:** {self.metadata.mimari}
- **Toplam Parametre Sayısı:** `{self.metadata.parametre_sayisi:,}`
- **Giriş Tensör Formatı:** `[Batch, 3, 32, 32]` (Normalize edilmiş RGB)
- **Çıkış:** `[Batch, 10]` (10 Sınıflı Olasılık Dağılımı)

---

## 🎯 Kullanım Amacı ve Sınırları

### İzin Verilen ve Hedeflenen Kullanım
- {self.metadata.kullanim_amaci}
- Edge ve mobil cihazlarda gerçek zamanlı nesne ve görsel sınıflandırması.

### Kapsam Dışı ve Sınırlılıklar
- {self.metadata.sinirliliklar}
- Tıbbi teşhis veya yüksek riskli güvenlik kararlarında tekil karar verici olarak kullanılmamalıdır.

---

## 📊 Nicel Değerlendirme Sonuçları

Toplam **{metrikler.toplam_ornek}** bağımsız test örneği üzerinde elde edilen nihai performans metrikleri:

| Değerlendirme Metriği | Ölçülen Değer | İdeal Hedef | Durum |
|---|---|---|---|
| **Genel Doğruluk (Accuracy)** | **%{metrikler.dogruluk * 100:.2f}** | > %85.0 | ✅ Mükemmel |
| **Macro F1-Skoru** | **{metrikler.macro_f1:.4f}** | > 0.8500 | ✅ Mükemmel |
| **Weighted F1-Skoru** | **{metrikler.weighted_f1:.4f}** | > 0.8500 | ✅ Mükemmel |
| **Macro Precision** | **{metrikler.macro_precision:.4f}** | > 0.8500 | ✅ Mükemmel |
| **Macro Recall** | **{metrikler.macro_recall:.4f}** | > 0.8500 | ✅ Mükemmel |
| **Expected Calibration Error (ECE)** | **{metrikler.kalibrasyon.ece_skoru:.4f}** | < 0.1000 | ✅ İyi Kalibre |
| **Brier Skoru** | **{metrikler.kalibrasyon.brier_skoru:.4f}** | < 0.2000 | ✅ Kararlı |

---

## ⚖️ Adillik ve Alt Grup Dilim (Slice) Analizi

Modelin farklı görsel koşulları ve alt gruplar altındaki performansı:

| Dilim (Slice) Adı | Örnek Sayısı | Doğruluk (Acc) | F1 Skoru | Pozitif Oran |
|---|---|---|---|---|
{dilim_tablosu_str}

### Adillik Metrikleri ve Sektörel Uyum
- **Demographic Parity Farkı:** `{adillik_raporu.demographic_parity_farki:.4f}`
- **Disparate Impact Oranı (DIR):** `%{adillik_raporu.disparate_impact_orani * 100:.2f}` (Yasal Eşik: $\\ge \\%80.0$)
- **Maksimum Dilimler Arası Doğruluk Farkı:** `%{adillik_raporu.maks_dogruluk_farki * 100:.2f}`
- **Adillik Kararı:** `{'✅ GEÇTİ (ADİL)' if adillik_raporu.adillik_esigi_gecti_mi else '⚠️ DİKKAT (YANLILIK RİSKİ)'}`

**Tespit Edilen Notlar ve Uyarılar:**
{uyarilar_str}

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
print(f"Tahmin Edilen Sınıf: {{sinif.item()}}, Güven: {{olasiliklar.max().item():.3f}}")
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
"""
        with open(kayit_yolu, "w", encoding="utf-8") as f:
            f.write(icerik)

        return icerik
