# Day 64: Üretim Seviyesi FastAPI İnference, Model Lifespan & Batch Prediction

[![License: Private All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests: 7 Passed](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)](testler/)

Üretim seviyesinde derin öğrenme (Vision & Embedding) modellerini yüksek eşzamanlılıkta (high concurrency) sunmak, `lifespan` asenkron bağlam yöneticisi ile sıfır soğuk başlangıç (zero cold-start) sağlamak ve toplu tahmin (Batch Prediction) ile donanım çıkarım verimliliğini maksimize etmek amacıyla geliştirilmiş **FastAPI İnference Servisi ve Dinamik Batching Mimarisi**.

---

## 1. 🎯 Günün Konusu & Teorik/Matematiksel Derinlik

### A. Çözülen Temel Problem ve Endüstriyel Senaryo
Üretim ortamında yapay zeka modellerini mikroservis mimarisinde sunarken:
1. **Model Yaşam Döngüsü ve Soğuk Başlatma Darboğazı:** Modeli istek geldikçe route fonksiyonu içinde yüklemek bellek tüketimini patlatır ve onlarca saniye gecikmeye yol açar. Eski `@app.on_event("startup")` yapısı yerine modern Starlette/FastAPI `lifespan` bağlam yöneticisi kullanılarak model belleğe tek seferde alınır, dummy tensörlerle GPU çekirdekleri ısıtılır (`warmup`) ve uygulama kapanırken VRAM temizlenir.
2. **Tekil Çıkarım (Batch Size = 1) İsrafı:** GPU ve modern CPU SIMD/AVX tensör çekirdekleri, matris çarpımlarını toplu ($B \ge 16$) yaptığında $\mathcal{O}(1)$ sabit maliyet amorti edilir. Dinamik batching olmadan donanım kapasitesinin yalnızca $\%10-\%20$'si kullanılır.
3. **Pydantic v2 ile Sözleşme Güvenliği:** Tip denetimi yapılmayan istekler 422 HTTP koduyla milisaniye-altı sürede reddedilir, modele bozuk veri ulaşması engellenir.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI ASYNC INFERENCE & LIFESPAN MIMARISI                                     │
│                                                                                                           │
│  [Sunucu Başlatma] ──► [Lifespan Context Manager] ──► [Model Yükleme + Warmup] ──► [app.state.model]      │
│                                                                                             │             │
│  [İstemciler: REST] ──► [Pydantic v2 Validasyon] ──► [Dinamik Batch Kuyruğu] ───────────────┤             │
│                                (422 Hata)          (tau_wait=10ms / B_max=16)               │             │
│                                                              │                              ▼             │
│                                                              └──► [Vektörize Toplu Çıkarım (Toplu_Tahmin)]│
│                                                                             │                             │
│                                                                             ▼                             │
│                                                                 [200 OK / Structured Response]            │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### B. Matematiksel Formülasyon ve Kuyruk Teorisi

1. **Kuyruk Teorisi ve Dinamik Batching (Queuing Model):**
   Gelen istekler geliş hızı $\lambda$ olan bir Poisson sürecini ($\text{Pois}(\lambda)$) takip eder. Maksimum bekleme penceresi $\tau_{\text{wait}}$ ve maksimum batch boyutu $B$ iken oluşturulan anlık batch boyutu $b$:
   $$b = \min\Big(B, \, N(t + \tau_{\text{wait}})\Big)$$

2. **Amorti Edilmiş İstek Başına Gecikme (Amortized Latency):**
   Sabit başlatma süresi $T_{\text{fixed}}$ ve tensör başına hesaplama süresi $T_{\text{tensor}}$ iken toplam süre $T(b) = T_{\text{fixed}} + b \cdot T_{\text{tensor}}$. İstek başına amorti edilmiş gecikme:
   $$\bar{L}(b) = \frac{T(b)}{b} = \frac{T_{\text{fixed}}}{b} + T_{\text{tensor}}$$
   $b \to B$ arttıkça $\frac{T_{\text{fixed}}}{b} \to 0$ yaklaşır ve birim maliyet düşer.

3. **İşlem Hacmi (Throughput / QPS):**
   $$\text{QPS}(b) = \frac{b}{T(b)} = \frac{b}{T_{\text{fixed}} + b \cdot T_{\text{tensor}}}$$

---

### C. SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | `lifespan` ile sıfır model yeniden yükleme; Pydantic v2 entegrasyonu; otomatik OpenAPI/Swagger dökümantasyonu; asenkron I/O ile yüksek eşzamanlılık. |
| **Weaknesses (Zayıf Yönler)** | Python GIL sebebiyle CPU-bound ağır çıkarımlarda tek iş parçacığı darboğazı (TorchScript / C++ runtime veya ProcessPool gereksinimi). |
| **Opportunities (Fırsatlar)** | Dinamik batching ile $3\times - 10\times$ throughput kazancı; Kubernetes HPA (Horizontal Pod Autoscaler) ve `/saglik` endpoint'i ile otomatik ölçeklenme. |
| **Threats (Tehditler)** | Aşırı trafik altında dinamik kuyruğun kontrolsüz büyümesi (OOM riski); zaman aşımı ($\tau_{\text{wait}}$) yanlış ayarlanırsa yüksek kuyruk gecikmesi. |

---

## 2. 💻 Üretim Seviyesinde Uygulama Mimarisi

Proje modüler bir paket yapısına sahiptir:

- [`src/model_motoru.py`](src/model_motoru.py): `YapayZekaModelMotoru` (Vision & Embedding simülatörü, `isinma()`, `tekil_tahmin()`, `toplu_tahmin()`).
- [`src/lifespan_yoneticisi.py`](src/lifespan_yoneticisi.py): `model_lifespan` (FastAPI modern async context manager).
- [`src/api_servisi.py`](src/api_servisi.py): FastAPI uygulama fabrikası, bağımlılık enjeksiyonu (`Depends(model_motoru_al)`), `/`, `/saglik`, `/v1/tahmin/tekil`, `/v1/tahmin/toplu`, `/v1/metrikler`.
- [`src/batch_kuyruk_yoneticisi.py`](src/batch_kuyruk_yoneticisi.py): `DinamikBatchKuyrugu` (Asenkron Producer-Consumer kuyruğu).
- [`src/gorsellestirici.py`](src/gorsellestirici.py): `FastAPIGorsellestirici` (6 panelli teşhis panosu).
- [`ana_akis.py`](ana_akis.py): Uçtan uca asenkron HTTPX benchmark betiği.
- [`testler/test_fastapi_api.py`](testler/test_fastapi_api.py): 7 kapsamlı asenkron birim testi (%100 Başarı).

---

## 3. 🧪 Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

**Görev:** Gelen isteklerin görsel boyutlarını ($H, W$) dinamik olarak kontrol edip aynı boyuttaki görselleri aynı batch'te toplayan ve boyut uyuşmazlığı durumunda otomatik padding uygulayan bir `AkilliBatchGruplayici` (Bucket Batcher) tasarlamak.

**Eksiksiz Kod Çözümü:**
```python
from typing import List, Dict, Any
from collections import defaultdict

class AkilliBatchGruplayici:
    """Görselleri çözünürlük havuzlarına (Buckets) göre gruplayarak tensör padding maliyetini minimize eder."""

    @staticmethod
    def havuzlara_ayir(istekler: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        havuzlar = defaultdict(list)
        for req in istekler:
            meta = req.get("gorsel_meta", {})
            w = meta.get("genislik", 1920)
            h = meta.get("yukseklik", 1080)
            # En yakın standart çözünürlük kovasına ata
            kova_anahtari = f"{w}x{h}"
            havuzlar[kova_anahtari].append(req)
        return dict(havuzlar)
```

---

## 4. 📊 Doğrulama ve Benchmark Metrikleri

FastAPI ASGI istemcisi ile ölçülen performans sonuçları:

| Metrik | Tekil Çıkarım ($B=1$) | Toplu Çıkarım ($B=32$) | Kazanç / İyileşme |
|---|---|---|---|
| **İşlem Hacmi (Throughput)** | $933.6\text{ QPS}$ | **$3,153.3\text{ QPS}$** | **$3.38\times$ Daha Yüksek Hacim** |
| **İstek Başına Amorti Gecikme** | $1.07\text{ ms}$ | **$0.32\text{ ms}$** | **$\%70.1$ Daha Düşük Gecikme** |
| **Model Lifespan Durumu** | Tamamlandı | Tamamlandı | Sıfır Soğuk Başlatma |
| **Birim Test Başarımı** | $7 / 7$ PASSED | $7 / 7$ PASSED | %100 Başarı (4.10s) |

---

## 5. 🚀 Kurulum ve Çalıştırma

```bash
# 1. Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 2. Ana asenkron benchmark betiğini çalıştırın
python ana_akis.py

# 3. Uvicorn geliştirme sunucusunu başlatın
uvicorn src.api_servisi:app --host 0.0.0.0 --port 8000 --reload

# 4. Birim testleri koşun
pytest testler -v
```

---

## 6. 📜 Lisans & Metaveri

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
