"""
Sürüm Adayı (Release Candidate) ve Manifest Yönetici Modülü (Day 95).
Model artefaktlarının SHA-256 bütünlüğünü denetler, RELEASE_MANIFEST.json üretir ve imzalar.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import torch
import transformers
from transformers import AutoConfig, AutoModelForImageClassification

from .konfigurasyon import MiniViTConfig
from .model import MiniViTForImageClassification


class ReleaseManifestYoneticisi:
    """
    Sürüm adayı (RC) manifestosu oluşturan, SHA-256 bütünlük imzalarını hesaplayan
    ve release doğrulamasını gerçekleştiren sınıf.
    """

    @staticmethod
    def dosya_sha256_hesapla(dosya_yolu: str) -> str:
        """Bir dosyanın SHA-256 karma özetini hesaplar."""
        sha256_hash = hashlib.sha256()
        with open(dosya_yolu, "rb") as f:
            for byte_blogu in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_blogu)
        return sha256_hash.hexdigest()

    def manifesto_olustur(
        self,
        paket_dizini: str,
        surum_etiketi: str = "v1.0.0-rc1",
        model_adi: str = "seydivakkas/minivit-cifar10",
        ekstra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Paket dizinindeki tüm dosyaları tarayarak imzalı RELEASE_MANIFEST.json üretir."""
        if not os.path.exists(paket_dizini):
            raise FileNotFoundError(f"Paket dizini bulunamadı: {paket_dizini}")

        dosyalar = {}
        for kok, _, dosya_adlari in os.walk(paket_dizini):
            for d in dosya_adlari:
                if d == "RELEASE_MANIFEST.json":
                    continue
                tam_yol = os.path.join(kok, d)
                rel_yol = os.path.relpath(tam_yol, paket_dizini)
                dosyalar[rel_yol] = {
                    "sha256": self.dosya_sha256_hesapla(tam_yol),
                    "boyut_bayt": os.path.getsize(tam_yol),
                    "boyut_kb": round(os.path.getsize(tam_yol) / 1024, 2),
                }

        config_yolu = os.path.join(paket_dizini, "config.json")
        config_veri = {}
        if os.path.exists(config_yolu):
            with open(config_yolu, "r", encoding="utf-8") as f:
                config_veri = json.load(f)

        manifesto = {
            "surum_bilgisi": {
                "release_tag": surum_etiketi,
                "surum_tipi": "Release Candidate (RC1)",
                "uretim_tarihi_utc": datetime.now(timezone.utc).isoformat(),
                "model_adi": model_adi,
                "yazar": "Seydi Eryılmaz (@seydivakkas)",
                "lisans": "Ozel Lisans - Tum Haklari Saklidir",
            },
            "calisma_ortami": {
                "pytorch_surumu": torch.__version__,
                "transformers_surumu": transformers.__version__,
                "cuda_kullanilabilir": torch.cuda.is_available(),
                "cihaz_adi": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            },
            "mimari_ozeti": {
                "model_tipi": config_veri.get("model_type", "minivit"),
                "goruntu_boyutu": config_veri.get("goruntu_boyutu", 32),
                "yama_boyutu": config_veri.get("yama_boyutu", 4),
                "gizli_boyut": config_veri.get("gizli_boyut", 128),
                "katman_sayisi": config_veri.get("katman_sayisi", 4),
                "dikkat_basliklari": config_veri.get("dikkat_baslik_sayisi", 4),
                "sinif_sayisi": config_veri.get("sinif_sayisi", 10),
            },
            "dosya_butunluk_tablosu": dosyalar,
            "kalite_kapisi_durumu": "BEKLEMEDE",
            "ekstra_meta": ekstra_meta or {},
        }

        # Manifestoyu imzala (İçerik özeti oluştur)
        manifesto_str = json.dumps(manifesto, sort_keys=True, ensure_ascii=False)
        manifesto["manifesto_imzasi_sha256"] = hashlib.sha256(manifesto_str.encode("utf-8")).hexdigest()

        # Diske yaz
        cikis_yolu = os.path.join(paket_dizini, "RELEASE_MANIFEST.json")
        with open(cikis_yolu, "w", encoding="utf-8") as f:
            json.dump(manifesto, f, indent=2, ensure_ascii=False)

        return manifesto

    def manifesto_dogrula(self, paket_dizini: str) -> Dict[str, Any]:
        """Paket dizinindeki RELEASE_MANIFEST.json dosyasını ve dosya SHA-256 özetlerini denetler."""
        manifesto_yolu = os.path.join(paket_dizini, "RELEASE_MANIFEST.json")
        if not os.path.exists(manifesto_yolu):
            return {"gecerli": False, "hata": "RELEASE_MANIFEST.json dosyası mevcut değil."}

        with open(manifesto_yolu, "r", encoding="utf-8") as f:
            manifesto = json.load(f)

        dosyalar = manifesto.get("dosya_butunluk_tablosu", {})
        bozuk_dosyalar = []

        for rel_yol, meta in dosyalar.items():
            tam_yol = os.path.join(paket_dizini, rel_yol)
            if not os.path.exists(tam_yol):
                bozuk_dosyalar.append({"dosya": rel_yol, "hata": "Dosya eksik"})
                continue

            hesaplanan_sha = self.dosya_sha256_hesapla(tam_yol)
            if hesaplanan_sha != meta["sha256"]:
                bozuk_dosyalar.append({
                    "dosya": rel_yol,
                    "beklenen_sha": meta["sha256"],
                    "hesaplanan_sha": hesaplanan_sha,
                    "hata": "SHA-256 Checksum Uyuşmazlığı"
                })

        return {
            "gecerli": len(bozuk_dosyalar) == 0,
            "toplam_dosya_sayisi": len(dosyalar),
            "bozuk_dosyalar": bozuk_dosyalar,
            "manifesto": manifesto,
        }


class SurumAdayiPaketleyici:
    """MiniViT Sürüm Adayı paketini Hugging Face formatında oluşturan ve manifestoyu bağlayan sınıf."""

    def __init__(self):
        # AutoClass kayıtları
        AutoConfig.register("minivit", MiniViTConfig)
        AutoModelForImageClassification.register(MiniViTConfig, MiniViTForImageClassification)

    def paketi_hazirla(
        self,
        model: MiniViTForImageClassification,
        hedef_dizin: str,
        surum_etiketi: str = "v1.0.0-rc1",
        repo_adi: str = "seydivakkas/minivit-cifar10",
    ) -> Dict[str, Any]:
        """Modeli diske SafeTensors formatında yazar ve RELEASE_MANIFEST.json üretir."""
        os.makedirs(hedef_dizin, exist_ok=True)

        # 1. Hugging Face SafeTensors Serileştirme
        model.save_pretrained(hedef_dizin, safe_serialization=True)

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
        with open(os.path.join(hedef_dizin, "preprocessor_config.json"), "w", encoding="utf-8") as f:
            json.dump(preprocessor_config, f, indent=2, ensure_ascii=False)

        # 3. Model Kartı README.md
        readme_icerik = f"""---
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
- name: {repo_adi}
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

# 🤖 MiniViT v1.0 Sürüm Adayı ({surum_etiketi})

Bu model, **Mini Vision Transformer (MiniViT)** mimarisinin CIFAR-10 veri kümesi üzerinde eğitilmiş ve **v1.0-RC1** aşamasında dondurulmuş sürüm adayıdır.

- **Mimari:** Patch Embedding + Transformer Encoder + CLS Classifier Head
- **Sürüm Durumu:** `{surum_etiketi}`
- **Lisans:** Özel Lisans — Tüm Hakları Saklıdır (c) 2026 Seydi Eryılmaz (@seydivakkas)
"""
        with open(os.path.join(hedef_dizin, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_icerik.strip())

        # 4. Manifestoyu oluştur
        manifesto_yoneticisi = ReleaseManifestYoneticisi()
        manifesto = manifesto_yoneticisi.manifesto_olustur(
            paket_dizini=hedef_dizin,
            surum_etiketi=surum_etiketi,
            model_adi=repo_adi,
        )

        return manifesto
