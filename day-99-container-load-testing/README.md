# 🐳 Day 99: MiniViT v1.0 Docker Konteynerleştirme ve Locust/k6 ile Eşzamanlı Yük/Stres Testleri

Bu proje, **101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği Master Serisi** kapsamında; MiniViT v1.0 FastAPI servisinin üretim seviyesinde hafif, güvenli (non-root) ve çok aşamalı (multi-stage) bir Docker konteynerine paketlenmesini, `docker-compose.yml` ile orkestre edilmesini ve Locust asenkron yük motoruyla $1 \sim 100$ eşzamanlı kullanıcı altında doygunluk (saturation) ve SLA stres testlerine tabi tutulmasını amaçlar.

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Multi--Stage-blue?style=flat-square&logo=docker)
![Locust](https://img.shields.io/badge/LoadTesting-Locust-green?style=flat-square)
![Throughput](https://img.shields.io/badge/Peak%20Throughput-400%2B%20RPS-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-100%25%20Passing-success?style=flat-square)

---

## 📑 İçindekiler
- [Teorik ve MLOps Temelleri](#-teorik-ve-mlops-temelleri)
  - [1. Neden Bu Sistem Kullanılır? (Mühendislik Gerekçesi)](#1-neden-bu-sistem-kullanılır-mühendislik-gerekçesi)
  - [2. Ne Gibi Sorunları Çözer? (Çözülen Darboğazlar)](#2-ne-gibi-sorunları-çözer-çözülen-darboğazlar)
  - [3. Ne Konuda Eksik Kalır? (Limitler & Riskler)](#3-ne-konuda-eksik-kalır-limitler--riskler)
  - [4. Alternatif Yük Testi Araçları & Karşılaştırma](#4-alternatif-yük-testi-araçları--karşılaştırma)
- [Matematiksel ve İstatistiksel Metrikler](#-matematiksel-ve-i̇statistiksel-metrikler)
- [Mimari Yapı & Dosya Düzeni](#-mimari-yapı--dosya-düzeni)
- [Hızlı Başlangıç & Kurulum](#-hızlı-başlangıç--kurulum)
- [Testler ve Doğrulama](#-testler-ve-doğrulama)
- [Lisans](#-lisans)

---

## 🧠 Teorik ve MLOps Temelleri

Bir yapay zeka modelinin canlı üretime çıkmadan önceki son adımı, konteynerleştirilmiş ortamda eşzamanlı yüksek trafik altında nasıl davrandığının modellenmesidir.

### 1. Neden Bu Sistem Kullanılır? (Mühendislik Gerekçesi)
- **Konteyner İzolasyonu & Güvenlik:** `python:3.11-slim` tabanlı multi-stage Dockerfile ile gereksiz derleme araçları üretim imajından atılır; `appuser (UID: 1000)` ile root yetkisi olmadan çalıştırılır.
- **Kapasite ve Doygunluk Planlaması (Capacity Planning):** Sistemin kaç RPS'den sonra kuyruklanmaya (queuing) başladığı ve gecikmenin katlandığı (saturation point) tespit edilir.
- **K8s Yatay Pod Otomatik Ölçekleme (HPA) Eşiklerinin Belirlenmesi:** Yük testi sonuçlarına göre CPU/RPS eşik kuralı (ör. pod başına 300 RPS) netleştirilir.

### 2. Ne Gibi Sorunları Çözer? (Çözülen Darboğazlar)
- **Trafik Patlamalarında Çökme (Cascading Failures):** Servisin 100 eşzamanlı kullanıcıda bile %0 hata oranıyla tüm istekleri karşıladığı kanıtlanır.
- **Şişirilmiş Docker İmaj Boyutları:** Çok aşamalı (multi-stage) derleme ile imaj boyutu optimize edilir.

### 3. Ne Konuda Eksik Kalır? (Limitler & Riskler)
- **Donanım Kaynak Sınırları:** Tek bir Docker konteyneri (2 worker) CPU/GPU sınırlarına ulaştığında kuyruk gecikmeleri artar; K8s Cluster HPA ile çoklu pod ölçeklemesi gerektirir.

### 4. Alternatif Yük Testi Araçları & Karşılaştırma

| Yük Test Aracı | Dil / Tanımlama | UI Dashboard | Dağıtık Yükleme | Python Ekosistemi |
|---|---|---|---|---|
| **Locust (Bizim)** | **Python (`locustfile.py`)** | **Evet (Web UI)** | **Evet (Master/Worker)** | **Mükemmel** |
| **k6** | JavaScript | Hayır (CLI/Cloud) | Evet | Düşük |
| **Apache JMeter** | XML / GUI | Kısmi | Evet | Orta |
| **wrk / vegeta** | CLI / C | Hayır (CLI) | Hayır | Yok |

---

## 📐 Matematiksel ve İstatistiksel Metrikler

### 1. Throughput ve İstek/Saniye (RPS)
Belirli bir $T$ süresi boyunca tamamlanan $N_{\text{basarili}}$ istek sayısı:

$$\text{RPS} = \frac{N_{\text{basarili}}}{T}$$

### 2. Gecikme Yüzdelikleri (Percentiles: P50, P90, P99)
$n$ adet sıralanmış gecikme ölçümü $L = [l_1, l_2, \dots, l_n]$ ($l_1 \le l_2 \le \dots \le l_n$) için $P$. yüzdelik indeksi:

$$k = \left\lceil \frac{P}{100} \times n \right\rceil, \quad \text{Percentile}_P = l_k$$

---

## 📂 Mimari Yapı & Dosya Düzeni

```
day-99-container-load-testing/
├── LICENSE
├── gereksinimler.txt
├── README.md
├── Dockerfile                  # Multi-Stage Production Dockerfile
├── docker-compose.yml          # FastAPI + Locust Master Orchestration
├── locustfile.py               # Locust HTTP User Scenarios
├── ana_akis.py
├── src/
│   ├── __init__.py
│   ├── konfigurasyon.py
│   ├── model.py
│   ├── api_uygulamasi.py
│   ├── yuk_testi_motoru.py
│   └── gorsellestirici.py
├── testler/
│   ├── __init__.py
│   └── test_yuk_testi.py
└── ciktilar/
    └── docker_yuk_testi_paneli.png
```

---

## 🚀 Hızlı Başlangıç & Kurulum

```bash
# 1. Bağımlılıkları Yükleyin
pip install -r gereksinimler.txt

# 2. Docker Konteynerini Derleyin ve Başlatın
docker-compose up --build -d

# 3. Locust Web Arayüzüne Erişin
# http://localhost:8089

# 4. Yük Testi Simülasyonu ve Teşhis Panosunu Çalıştırın
python ana_akis.py
```

---

## 🧪 Testler ve Doğrulama

```bash
pytest testler/ -v
```

Çıktı Özeti:
```
testler/test_yuk_testi.py::test_dockerfile_ve_compose_varligi PASSED
testler/test_yuk_testi.py::test_locustfile_tanimlari PASSED
testler/test_yuk_testi.py::test_api_health_endpoint PASSED
testler/test_yuk_testi.py::test_api_predict_endpoint PASSED
testler/test_yuk_testi.py::test_api_base64_endpoint PASSED
testler/test_yuk_testi.py::test_eszamanli_seviye_testi_motoru PASSED
testler/test_yuk_testi.py::test_basamakli_yuk_testi_motoru PASSED
testler/test_yuk_testi.py::test_gorsellestirici_pano_uretme PASSED

======================== 8 passed in 11.80s ========================
```

---

## 📜 Lisans

```
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
