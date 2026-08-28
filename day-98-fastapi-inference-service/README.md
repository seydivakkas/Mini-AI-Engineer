# 🚀 Day 98: MiniViT v1.0 Üretime Hazır Yüksek Performanslı Asenkron FastAPI Servisi

Bu proje, **101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği Master Serisi** kapsamında; MiniViT v1.0 Vision Transformer modelini üretim ortamında Kubernetes, Docker ve mikroservis mimarilerine tam uyumlu, asenkron, Pydantic v2 veri doğrulamalı ve `/health` probelarına sahip bir REST API olarak sunmak amacıyla geliştirilmiştir.

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)
![Pydantic](https://img.shields.io/badge/Pydantic-v2.x-E92063?style=flat-square)
![Uvicorn](https://img.shields.io/badge/ASGI-Uvicorn-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=flat-square)

---

## 📑 İçindekiler
- [Teorik ve Mimari Temeller](#-teorik-ve-mimari-temeller)
  - [1. Neden Bu Sistem Kullanılır? (Mühendislik Gerekçesi)](#1-neden-bu-sistem-kullanılır-mühendislik-gerekçesi)
  - [2. Ne Gibi Sorunları Çözer? (Çözülen Darboğazlar)](#2-ne-gibi-sorunları-çözer-çözülen-darboğazlar)
  - [3. Ne Konuda Eksik Kalır? (Limitler & Riskler)](#3-ne-konuda-eksik-kalır-limitler--riskler)
  - [4. Alternatif Yaklaşımlar & Karşılaştırma](#4-alternatif-yaklaşımlar--karşılaştırma)
- [API Endpoint Sözleşmesi (OpenAPI 3.0)](#-api-endpoint-sözleşmesi-openapi-30)
- [Mimari Yapı & Dosya Düzeni](#-mimari-yapı--dosya-düzeni)
- [Hızlı Başlangıç & Kurulum](#-hızlı-başlangıç--kurulum)
- [Testler ve Doğrulama](#-testler-ve-doğrulama)
- [Lisans](#-lisans)

---

## 🧠 Teorik ve Mimari Temeller

Üretim seviyesinde bir AI modelinin REST API olarak sunulmasında en kritik mimari zorluk, CPU/GPU yoğun model çıkarımının web sunucusunun asenkron olay döngüsünü (event loop) kilitlemesini engellemek ve Kubernetes orkestrasyon sistemine `/health`, `/healthz`, `/ready` uç noktalarıyla anlık durum bildirmektir.

### 1. Neden Bu Sistem Kullanılır? (Mühendislik Gerekçesi)
- **Non-Blocking Asenkron Çıkarım:** FastAPI ve ASGI mimarisi sayesinde, model çıkarımı `asyncio.to_thread` ile arka plan iş parçacığı havuzuna devredilir; böylece I/O işlemleri ve sağlık kontrolü istekleri asla gecikmez.
- **Tip Güvenliği ve Şema Doğrulama:** Pydantic v2 şemaları ile gelen ve giden tüm JSON/Multipart verileri sıkı tip denetiminden geçer.
- **Kubernetes Liveness & Readiness Probeları:** K8s podları ayağa kalkarken model belleğe yüklenip ısınmadan trafiğe açılmaz (`readiness probe`).

### 2. Ne Gibi Sorunları Çözer? (Çözülen Darboğazlar)
- **Event Loop Bloklanması (Server Freezing):** Ağır tensör matris çarpımları sırasında web sunucusunun yeni HTTP bağlantılarını kabul edememesi sorununu çözer.
- **Geçersiz Girdi Hataları:** Bozuk JPEG veya geçersiz Base64 dizgileri model katmanına ulaşmadan FastAPI seviyesinde `400 Bad Request` ile temiz şekilde reddedilir.

### 3. Ne Konuda Eksik Kalır? (Limitler & Riskler)
- **Dinamik Batching:** Saf FastAPI tek başına Triton Inference Server veya TorchServe gibi gelişmiş dinamik GPU kuyruk birleştirme (dynamic batching) motoru içermez.
- **Bellek Yönetimi:** Çok sayıda eşzamanlı büyük görsel yüklemesinde sunucu RAM tüketiminin izlenmesi gerekir.

### 4. Alternatif Yaklaşımlar & Karşılaştırma

| Servis Mimarisi | Geliştirme Kolaylığı | Asenkron I/O | Dinamik Batching | Kubernetes Uyumluluğu |
|---|---|---|---|---|
| **FastAPI + Uvicorn (Bizim)** | **Çok Yüksek (Python)** | **Evet (asyncio)** | **Uygulama Seviyesi** | **Tam Uyumlu** |
| **Flask + Gunicorn** | Yüksek | Hayır (Senkron) | Hayır | Kısmi |
| **Triton Inference Server** | Düşük (C++/Protobuf) | Evet | Evet (Dahili) | Tam Uyumlu |
| **TorchServe** | Orta (Java/Python) | Kısmi | Evet | Tam Uyumlu |

---

## 📡 API Endpoint Sözleşmesi (OpenAPI 3.0)

| Yöntem | Endpoint | Açıklama | Başarılı Yanıt |
|---|---|---|---|
| `GET` | `/` | Servis karşılama ve genel durum | `200 OK` |
| `GET` | `/health` | Kubernetes Liveness & Readiness Probu | `200 OK (HEALTHY)` |
| `GET` | `/metadata` | Model parametreleri ve CIFAR-10 etiketleri | `200 OK` |
| `GET` | `/metrics` | P50, P90, P99 gecikme ve toplam istek metrikleri | `200 OK` |
| `POST` | `/predict` | Multipart Form-Data (Görsel dosyası yükleme) | `200 OK (Top-K Tahmin)` |
| `POST` | `/predict/base64` | Base64 kodlu JSON gövdesi ile tahmin | `200 OK (Top-K Tahmin)` |
| `POST` | `/predict/batch` | Çoklu dosya yüklemesi ile toplu tahmin | `200 OK (Toplu Liste)` |

---

## 📂 Mimari Yapı & Dosya Düzeni

```
day-98-fastapi-inference-service/
├── LICENSE
├── gereksinimler.txt
├── README.md
├── ana_akis.py
├── src/
│   ├── __init__.py
│   ├── konfigurasyon.py
│   ├── model.py
│   ├── semalar.py
│   ├── servis_yoneticisi.py
│   ├── api_uygulamasi.py
│   └── gorsellestirici.py
├── testler/
│   ├── __init__.py
│   └── test_fastapi_servis.py
└── ciktilar/
    └── fastapi_servis_paneli.png
```

---

## 🚀 Hızlı Başlangıç & Kurulum

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Canlı FastAPI Sunucusunu Başlatın
uvicorn src.api_uygulamasi:app --host 0.0.0.0 --port 8000 --reload

# Veya test benchmark ve teşhis panosu akışını çalıştırın
python ana_akis.py
```

### Örnek cURL İstekleri:
```bash
# 1. Sağlık Kontrolü
curl -X GET http://localhost:8000/health

# 2. Tekli Görsel Tahmini
curl -X POST "http://localhost:8000/predict?top_k=5" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@kedi.png"
```

---

## 🧪 Testler ve Doğrulama

```bash
pytest testler/ -v
```

Çıktı Özeti:
```
testler/test_fastapi_servis.py::test_kok_dizin_endpoint PASSED
testler/test_fastapi_servis.py::test_health_ve_readiness_probelari PASSED
testler/test_fastapi_servis.py::test_metadata_endpoint PASSED
testler/test_fastapi_servis.py::test_metrics_endpoint PASSED
testler/test_fastapi_servis.py::test_predict_multipart_dosya_yukleme PASSED
testler/test_fastapi_servis.py::test_predict_base64_json_istegi PASSED
testler/test_fastapi_servis.py::test_predict_batch_coklu_dosya PASSED
testler/test_fastapi_servis.py::test_gecersiz_girdi_hata_yonetimi PASSED

======================== 8 passed in 11.07s ========================
```

---

## 📜 Lisans

```
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
