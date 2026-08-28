# Day 35: FastAPI ile Asenkron AI Model Servisi & REST API (FastAPI AI Model Service)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.0+-E92063.svg?style=flat-square&logo=pydantic)](https://docs.pydantic.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; derin öğrenme ve üretken yapay zeka modellerinin üretim ortamında (production) yüksek eşzamanlılık (high concurrency), ultra düşük gecikme ve katı tip güvenliği ile sunulmasını sağlayan **FastAPI Asenkron AI Model Servisi ve REST API** mimarisidir.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Neden FastAPI ile Model Servisleme?
Flask veya Django gibi geleneksel senkron web çatıları, her istek için bir thread kilitler ve yüksek trafik altında kolayca darboğaza (bottleneck) girer. **FastAPI** ise:
1. **ASGI (Asynchronous Server Gateway Interface):** Tek bir process üzerinde binlerce eşzamanlı I/O bağlantısını (ağ, veritabanı, dosya yükleme) non-blocking olarak yönetir.
2. **Pydantic v2 Tabanlı Tip Güvenliği:** JSON istek gövdelerini otomatik ayrıştırır, doğrular ve OpenAPI/Swagger şemalarını dinamik üretir.
3. **Otomatik Dokümantasyon:** `/docs` (Swagger UI) ve `/redoc` arayüzlerini sıfır ek yapılandırmayla sunar.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │               İSTEMCİLER (WEB, MOBİL, IOT)               │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │ HTTP / REST İstekleri
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  FASTAPI ASGI SUNUCUSU (UVICORN / STARLETTE EVENT LOOP)                      │
        │  - Gecikme Ölçüm Middleware (X-Process-Time-Ms)                             │
        │  - Pydantic v2 İstek Doğrulama & Hata Yakalama (422 / 500)                   │
        │  - Liveness & Readiness Probları (/healthz)                                  │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
        ┌───────────────────────────────┐                 ┌───────────────────────────────┐
        │  I/O YOĞUN GÖREVLER (ASYNC)   │                 │  CPU/GPU YOĞUN MODEL ÇIKARIMI │
        │  - Dosya Okuma & Yükleme      │                 │  (asyncio.to_thread Havuzu)   │
        │  - Arka Plan Loglama          │                 │  - PyTorch Tensor Çıkarımı    │
        │  - Telemetri & Metrikler      │                 │  - RAG Semantik Eşleme        │
        └───────────────────────────────┘                 └───────────────────────────────┘
                                                 │
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  TİP GÜVENLİ JSON / GÖRSEL ANALİZ REST YANITI            │
                    └──────────────────────────────────────────────────────────┘
```

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **FastAPI** | *FastAPI Framework* | Python tip ipuçlarına dayalı, asenkron (asyncio) destekli, yüksek performanslı ve otomatik OpenAPI/Swagger dokümantasyonlu web çerçevesi. |
| **ASGI Sunucusu (Uvicorn)** | *Asynchronous Server Gateway Interface* | Eşzamanlı binlerce HTTP isteğini bloke olmadan yönetebilen modern asenkron Python web sunucusu. |
| **Pydantic Şemaları** | *Pydantic Request/Response Models* | Gelen istek gövdelerinin ve dönen JSON yanıtlarının tip güvenliğini ve veri doğrulamasını sağlayan modeller. |
| **Sağlık Kontrolü (`/health`)** | *Health Check Endpoint* | Kubernetes veya yük dengeleyicilerin servisin ayakta olup olmadığını ve modelin yüklendiğini kontrol ettiği uç nokta. |

---

## 2. Kritik Mimari Kural: CPU-Bound Model Çıkarımında Event Loop Bloklanması

Derin öğrenme modellerinde ileri besleme (`model.forward()`) saf CPU/GPU matris çarpımıdır (CPU-bound).
- **Yanlış Yaklaşım:** `async def` içine doğrudan ağır tensör işlemi koymak. Bu, Python'un tek ana event loop'unu dondurur ve sunucuya gelen diğer tüm HTTP istekleri kuyrukta bekler.
- **Doğru Yaklaşım (Thread Pool Offloading):** Ağır model çıkarımlarını `asyncio.to_thread()` fonksiyonu ile FastAPI'nin arka plandaki `ThreadPoolExecutor` havuzuna aktarmak veya standart senkron `def` endpoint tanımlamaktır.

```python
# Doğru Asenkron Çıkarım Örneği
async def metin_tahmin_et(self, metin: str):
    # Event loop'u bloklamadan thread havuzunda koştur
    return await asyncio.to_thread(self._senkron_cikarim, metin)
```

---

### 3. API Uç Noktaları ve Özellik Tablosu

| Metot | Uç Nokta (Endpoint) | Girdi Şeması | Çıktı Şeması | Açıklama |
|---|---|---|---|---|
| `GET` | `/healthz` | *Yok* | `SaglikYaniti` | K8s liveness & readiness kontrolü |
| `POST` | `/api/v1/predict/text` | `MetinTahminIstegi` | `MetinTahminYaniti` | Metin sınıflandırma ve 64-d embedding |
| `POST` | `/api/v1/predict/image` | `UploadFile (multipart)` | `GorselAnalizYaniti` | Görsel yükleme, nesne tespiti & RGB |
| `POST` | `/api/v1/rag/query` | `RAGSorguIstegi` | `RAGSorguYaniti` | RAG doküman soru-cevap |
| `GET` | `/api/v1/telemetry` | *Yok* | `JSON` | İstek sayaçları ve ortalama gecikme |

---

## 🛠️ Dizin Yapısı

```
day-35-fastapi-model-service/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # fastapi, uvicorn, pydantic, httpx, torch, pytest
├── ana_akis.py                      # REST API test istemcisi ve canlı test betiği
├── README.md                        # Detaylı mimari ve üretim kılavuzu (220+ Satır)
├── src/
│   ├── __init__.py
│   ├── semalar.py                   # Pydantic v2 istek/yanıt doğrulama şemaları
│   ├── model_motoru.py              # Asenkron çoklu modalite AI motoru
│   ├── servis_uygulamasi.py         # FastAPI ASGI uygulaması, middleware ve routerlar
│   └── gorsellestirici.py           # 6 panelli servis analiz panosu (Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_fastapi_service.py      # 7 adet kapsamlı API birim testi
└── ciktilar/
    └── fastapi_servis_paneli.png    # 6 panelli servis analiz görseli
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Canlı Geliştirme Sunucusunun Başlatılması (Uvicorn)
```bash
uvicorn src.servis_uygulamasi:app --host 0.0.0.0 --port 8000 --reload
```
*API Dokümantasyonu için tarayıcıda `http://localhost:8000/docs` adresini açın.*

### 3. Ana Akışın Çalıştırılması
```bash
python ana_akis.py
```

### 4. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/servis_uygulamasi.py` içerisine bir **"Rate Limiter (İstek Sınırlandırma Middleware)"** ekleyerek aynı IP adresinden dakikada 60'tan fazla istek gelmesi durumunda `HTTP 429 Too Many Requests` yanıtı döndüren mekanizma kurmak.

**Tamamlanan Çözüm:**
```python
from collections import defaultdict
import time
from fastapi import Request, HTTPException

IP_SAYACLARI = defaultdict(list)

@app.middleware("http")
async def rate_limiter_middleware(request: Request, call_next):
    istemci_ip = request.client.host if request.client else "127.0.0.1"
    suan = time.time()
    # Son 60 saniyedeki istekleri filtrele
    IP_SAYACLARI[istemci_ip] = [t for t in IP_SAYACLARI[istemci_ip] if suan - t < 60.0]
    
    if len(IP_SAYACLARI[istemci_ip]) >= 60:
        raise HTTPException(status_code=429, detail="Çok fazla istek yapıldı. Lütfen bekleyin.")
        
    IP_SAYACLARI[istemci_ip].append(suan)
    return await call_next(request)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Üretim ortamında (Kubernetes / Docker) bir PyTorch derin öğrenme modelini FastAPI ile servis ederken neden Uvicorn worker sayısını rastgele çok yüksek (örneğin 32 worker) ayarlamamalıyız?

> **Mentor Cevabı:**
> 1. **RAM ve VRAM Tüketimi (Out Of Memory - OOM):** Her Uvicorn worker'ı ayrı bir Python process'idir ve her biri PyTorch model ağırlıklarını belleğe (RAM/VRAM) baştan kopyalar. 1 GB'lık bir model 32 worker ile 32 GB bellek tüketerek sunucuyu çökertebilir.
> 2. **CPU Çekirdek Yarışması (Thread Contention):** PyTorch kendi içinde OpenMP/MKL ile çoklu thread kullanır. Çok sayıda worker process aynı anda CPU çekirdekleri için yarışarak bağlam değiştirme (context switching) maliyetini fırlatır ve gecikmeyi artırır.
> 3. **Optimum Kural:** Genellikle worker sayısı `GPU Sayısı` kadar (GPU varsa) veya CPU için `Worker = 2 - 4` tutulup, model içi batching (Dynamic Batching) veya Triton / TorchServe mimarileri tercih edilir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
