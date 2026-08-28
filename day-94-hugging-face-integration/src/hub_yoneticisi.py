"""
Day 94: Hugging Face Model Hub Paketleme, SafeTensors ve AutoClass Yöneticisi
-----------------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import json
import torch
from transformers import AutoConfig, AutoModelForImageClassification

from .konfigurasyon import MiniViTConfig
from .model import MiniViTForImageClassification


@dataclass
class ModelPaketBilgisi:
    paket_dizini: str
    config_dosyasi: str
    safetensors_dosyasi: str
    preprocessor_dosyasi: str
    hub_model_card_dosyasi: str
    toplam_parametre: int
    dosya_boyutlari_kb: Dict[str, float]
    sayisal_uyumluluk_dogrulandi: bool
    maks_hata_farki: float


class HubPaketleyici:
    """
    MiniViT modelini Hugging Face ekosistemine (AutoClasses, SafeTensors, Preprocessor)
    uygun şekilde paketler, yerel dizine kaydeder ve geri yükleme testlerini doğrular.
    """

    def __init__(self):
        self._autoclass_kayit_yap()

    def _autoclass_kayit_yap(self):
        """AutoConfig ve AutoModelForImageClassification sistemlerine özel mimariyi tanıtır."""
        try:
            AutoConfig.register("minivit", MiniViTConfig)
        except ValueError:
            pass  # Zaten kayıtlıysa yoksay

        try:
            AutoModelForImageClassification.register(MiniViTConfig, MiniViTForImageClassification)
        except ValueError:
            pass

    def modeli_paketle_ve_kaydet(
        self,
        model: MiniViTForImageClassification,
        kayit_dizini: str,
        repo_adi: str = "seydivakkas/minivit-cifar10-v1",
    ) -> ModelPaketBilgisi:
        """Modeli, SafeTensors ağırlıklarını, konfigürasyonu ve Hub Model Card'ı kaydeder."""
        os.makedirs(kayit_dizini, exist_ok=True)

        # 1. Hugging Face Standart save_pretrained (SafeTensors ile)
        model.save_pretrained(kayit_dizini, safe_serialization=True)

        # 2. Ön-işleyici (Image Processor) Konfigürasyonu
        preprocessor_bilgisi = {
            "image_processor_type": "MiniViTImageProcessor",
            "do_resize": True,
            "size": {"height": model.config.goruntu_boyutu, "width": model.config.goruntu_boyutu},
            "do_rescale": True,
            "rescale_factor": 0.00392156862745098,  # 1/255
            "do_normalize": True,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }
        preprocessor_yolu = os.path.join(kayit_dizini, "preprocessor_config.json")
        with open(preprocessor_yolu, "w", encoding="utf-8") as f:
            json.dump(preprocessor_bilgisi, f, indent=2)

        # 3. Hub README.md (Hugging Face Metadata Header ile)
        model_card_yolu = os.path.join(kayit_dizini, "README.md")
        self._hub_readme_olustur(model_card_yolu, repo_adi, model.config)

        # 4. Dosya Yolları ve Boyutları
        config_yolu = os.path.join(kayit_dizini, "config.json")
        safetensors_yolu = os.path.join(kayit_dizini, "model.safetensors")

        dosya_boyutlari = {}
        for d in [config_yolu, safetensors_yolu, preprocessor_yolu, model_card_yolu]:
            if os.path.exists(d):
                dosya_boyutlari[os.path.basename(d)] = os.path.getsize(d) / 1024.0

        # 5. Geri Yükleme ve Sayısal Doğrulama Testi
        uyumluluk, maks_fark = self.sayisal_uyumluluk_dogrula(model, kayit_dizini)

        toplam_parametre = sum(p.numel() for p in model.parameters())

        return ModelPaketBilgisi(
            paket_dizini=kayit_dizini,
            config_dosyasi=config_yolu,
            safetensors_dosyasi=safetensors_yolu,
            preprocessor_dosyasi=preprocessor_yolu,
            hub_model_card_dosyasi=model_card_yolu,
            toplam_parametre=toplam_parametre,
            dosya_boyutlari_kb=dosya_boyutlari,
            sayisal_uyumluluk_dogrulandi=uyumluluk,
            maks_hata_farki=maks_fark,
        )

    def sayisal_uyumluluk_dogrula(
        self,
        orijinal_model: MiniViTForImageClassification,
        kayit_dizini: str,
    ) -> Tuple[bool, float]:
        """Diske kaydedilen modeli AutoModel ile yükleyip çıkarım çıktılarının birebir eşitliğini test eder."""
        cihaz = next(orijinal_model.parameters()).device
        yuklenen_model = AutoModelForImageClassification.from_pretrained(kayit_dizini, local_files_only=True).to(cihaz)
        orijinal_model.eval()
        yuklenen_model.eval()

        # Deterministik test tensörü
        torch.manual_seed(42)
        test_x = torch.randn(2, 3, orijinal_model.config.goruntu_boyutu, orijinal_model.config.goruntu_boyutu, device=cihaz)

        with torch.no_grad():
            cikis_orj = orijinal_model(test_x).logits
            cikis_yuk = yuklenen_model(test_x).logits

        fark = torch.max(torch.abs(cikis_orj - cikis_yuk)).item()
        uyumlu_mu = fark < 1e-5

        return uyumlu_mu, fark

    def _hub_readme_olustur(self, dosya_yolu: str, repo_adi: str, config: MiniViTConfig):
        """Hugging Face Hub standart YAML metadata başlığı içeren README.md oluşturur."""
        hub_icerik = f"""---
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

# 🤗 {repo_adi}

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Model Type: MiniViT](https://img.shields.io/badge/Architecture-MiniViT-blue.svg?style=flat-square)](#)
[![Format: SafeTensors](https://img.shields.io/badge/Format-SafeTensors-success.svg?style=flat-square)](#)

Bu model, **101 Günlük Yapay Zeka & MLOps Master Serisi** kapsamında geliştirilmiş, hafif ve yüksek performanslı bir **MiniViT (Vision Transformer)** görsel sınıflandırma modelidir.

## 🚀 Hızlı Başlangıç (Inference with Hugging Face)

```python
from transformers import AutoConfig, AutoModelForImageClassification
import torch

# 1. Modeli Hub veya Yerel Dizinden Yükle
model_yolu = "{repo_adi}"
model = AutoModelForImageClassification.from_pretrained(model_yolu, trust_remote_code=True)
model.eval()

# 2. Örnek Çıkarım
dummy_pixel = torch.randn(1, 3, {config.goruntu_boyutu}, {config.goruntu_boyutu})
with torch.no_grad():
    outputs = model(dummy_pixel)
    tahmin_sinif = outputs.logits.argmax(dim=-1).item()

print(f"Tahmin Edilen Sınıf: {{tahmin_sinif}}")
```

## 📌 Model Konfigürasyonu
- **Görüntü Boyutu:** `{config.goruntu_boyutu}x{config.goruntu_boyutu}`
- **Yama Boyutu (Patch Size):** `{config.yama_boyutu}x{config.yama_boyutu}`
- **Gizli Boyut (Hidden Dim):** `{config.gizli_boyut}`
- **Katman Sayısı:** `{config.katman_sayisi}`
- **Dikkat Başlığı:** `{config.dikkat_baslik_sayisi}`
- **Sınıf Sayısı:** `{config.sinif_sayisi}`

## 📜 Lisans ve Telif Hakkı
```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
"""
        with open(dosya_yolu, "w", encoding="utf-8") as f:
            f.write(hub_icerik)
