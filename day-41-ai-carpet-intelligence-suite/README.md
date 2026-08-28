# Day 41: Uçtan Uca Çoklu Görev Halı Zekası Paketi (AI Carpet Intelligence Suite)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-9.5+-005571.svg?style=flat-square)](https://python-pillow.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; Sektörel Halı ve Tekstil Zekası serimizin (Faz 2B) büyük dönüm noktası olan **Uçtan Uca Çoklu Görev Halı Zekası Paketidir (AI Carpet Intelligence Suite)**. Canlı dokuma tezgahı veya konveyör bandı kamerasından gelen tek bir halı görüntüsü üzerinde **Renk Ayrıştırma**, **Görsel Arama**, **Kusur Tespiti** ve **Sektörel RAG Reçete Danışmanı** modüllerini konsolide ederek tek bir fabrika yönetim panosunda birleştirir.

---

## 📖 Mentorluk Dersi ve Konsolide Sistem Mimarisi

```
                        ┌──────────────────────────────────────────────────────────┐
                        │      CANLI DOKUMA HATTI KAMERA GÖZLEMİ (400x300 RGB)     │
                        └────────────────────────────┬─────────────────────────────┘
                                                     │
                                                     ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                MERKEZİ HALI ZEKASI ORKESTRATÖRÜ (HaliZekasiOrkestrator)                      │
    └──────┬───────────────────────┬───────────────────────────────┬────────────────────────┬──────┘
           │                       │                               │                        │
           ▼                       ▼                               ▼                        ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐ ┌──────────────────────────┐
│  1. RENK ZEKASI     │ │  2. GÖRSEL ARAMA    │ │  3. KUSUR TESPİTİ   │ │  4. SEKTÖREL RAG       │
│ - CIELAB K-Means K=5│ │ - 3B HSV Histogramı │ │ - Kalıntı Isı Harita│ │ - TS EN ISO Standartı  │
│ - İplik Sarfiyat %  │ │ - GLCM Haralick Doku│ │ - Morfolojik Aç/Kapa│ │ - Kusur İçin Otomatik  │
│ - Delta-E 2000 Uyum │ │ - Top-K Ürün Eşleme │ │ - Bounding Box & AR │ │   Reçete & Çözüm Getir │
└──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘ └─────────────┬────────────┘
           │                       │                       │                          │
           └───────────────────────┴───────────┬───────────┴──────────────────────────┘
                                               │
                                               ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │              KONSOLİDE YÖNETİCİ KALİTE RAPORU VE FABRİKA KARAR PROTOKOLÜ                     │
    │  - Genel Kalite Skoru: %68.0 / 100                                                           │
    │  - Karar: PARTI_RED_URETIMI_DURDUR (Kritik İplik Kopması & Yağ Lekesi Nedeniyle)             │
    │  - Otomatik Reçete: Vandewiele Dokuma Çözgü Tansiyonu 45 cN'e düşürülmeli                   │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Entegre Edilen 4 Temel Modülün Görevleri

1. **Modül 1 (Renk Zekası - CIELAB & $\Delta E_{00}$):** Halı yüzeyindeki 5 ana iplik rengini ayırır, her ipliğin desendeki yüzdesel alanını ($P_i$) hesaplar ve kurumsal iplik kartelası (`YARN-101` .. `YARN-106`) ile ISO tolerans eşlemesi yapar.
2. **Modül 2 (Görsel Arama - HSV & GLCM Haralick):** Dokuma deseni ve mikro-doku özelliklerini harmanlayarak iç ürün kataloğundan en yakın tasarım eşleşmesini getirir.
3. **Modül 3 (Dokuma Hataları & Anomali Tespiti):** Referans/lokal Gauss kalıntı haritası çıkarır, ikili morfolojik açma/kapama ile gürültüyü temizler, kontur analizi ile hataları (`IPLIK_KOPMASI`, `YAG_BOYA_LEKESI`) sınıflandırır.
4. **Modül 4 (Sektörel RAG & Reçete Danışmanı):** Tespit edilen her hata için otomatik olarak TS EN ISO 2060 ve Vandewiele dokuma kılavuzlarını tarayarak teknisyene anında çözüm önerisi üretir.

---

### Konsolide Teftiş Çıktıları Tablosu

| Analiz Modülü | Çıkarılan Metrik / Sonuç | Durum / Karar |
|---|---|---|
| **1. Renk Zekası** | 4 İplik Ayrıştırıldı (Krem %43.8, Bordo %24.7, Mavi %14.2, Yeşil %8.6) | $\Delta E_{00} < 2.0$ (**Mükemmel Uyum**) |
| **2. Görsel Arama** | `CARPET-CLASSIC-01` (Hereke Klasik Madalyonlu Bordo) | **%98.40 Benzerlik** (Katalog Uyumlu) |
| **3. Kusur Tespiti** | 2 Kusur: 1x İplik Kopması (1100 px), 1x Yağ Lekesi (1600 px) | **2 Kritik Hata** (Riskli) |
| **4. Sektörel RAG** | Vandewiele Çözgü Tansiyonu: $45 \pm 5\text{ cN}$, Ultrasonik Leke Temizleme | **Otomatik Çözüm Eklendi** |
| **5. Fabrika Kararı** | **Genel Kalite Skoru: %65.0 / 100** | **PARTI_RED_URETIMI_DURDUR** |

---

## 🛠️ Dizin Yapısı

```
day-41-ai-carpet-intelligence-suite/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, pillow, matplotlib, seaborn, pytest
├── ana_akis.py                      # 4 modülü konsolide eden ana yürütme betiği
├── README.md                        # 220+ satır sektörel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── orkestrator.py               # Merkezi orkestratör ve kalite karar motoru
│   ├── gorsellestirici.py           # 6 panelli konsolide fabrika kontrol panosu
│   └── moduller/
│       ├── __init__.py
│       ├── renk_motoru.py           # CIELAB K-Means ve Delta-E 2000 katalog eşleyici
│       ├── arama_motoru.py          # HSV + GLCM Haralick çoklu özellik görsel arama
│       ├── kusur_motoru.py          # Anomali, morfoloji ve kontur kusur sınıflandırıcı
│       └── rag_motoru.py            # TS EN ISO standartları ve otomatik reçete danışmanı
├── testler/
│   ├── __init__.py
│   └── test_carpet_suite.py         # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── hali_zeka_paketi_paneli.png  # 6 panelli yüksek çözünürlüklü fabrika kontrol panosu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Çok Modlu Entegre Sistem** | *End-to-End Multimodal Suite* | Renk analizi, görsel arama, kusur tespiti ve teknik RAG asistanını tek bir platformda birleştiren konsolide mimari. |
| **Mikroservis Orkestrasyonu** | *Microservice Pipeline Orchestration* | FastAPI arka uç servisleri ile Streamlit kullanıcı arayüzü arasındaki asenkron veri iletişimi. |
| **Uçtan Uca Kalite Güvencesi** | *Full-Stack AI Quality Assurance* | Tüm modüllerin eşzamanlı ve çökme korumalı çalıştığını doğrulayan entegrasyon testleri. |
| **Kurumsal Dağıtım Hazırlığı** | *Production Readiness* | Büyük ölçekli tekstil fabrikalarında ve e-ticaret platformlarında canlıya alınabilir mimari tasarım. |

---

## 2. Ana Akışın Çalıştırılması
```bash
python ana_akis.py
```

### 3. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/orkestrator.py` içerisine **"Üretim Hattı Canlı JSON Telemetri Akışı (Live JSON Telemetry Exporter)"** fonksiyonu ekleyerek fabrikanın ERP/SCADA ve MQTT sistemlerine uygun anlık JSON durum paketi üretmek.

**Tamamlanan Çözüm:**
```python
def telemetri_paketi_uret(self, teftis_raporu: dict) -> dict:
    return {
        "timestamp_ms": 1724838000000,
        "hat_id": "WEAVING-LINE-04",
        "kalite_skoru": teftis_raporu["genel_kalite_skoru"],
        "hat_karari": teftis_raporu["fabrika_karari"],
        "hata_sayisi": teftis_raporu["kusur_tespiti"]["kusur_sayisi"],
        "kritik_alarm": teftis_raporu["kusur_tespiti"]["kritik_kusur_sayisi"] > 0,
        "iplik_sayisi": teftis_raporu["renk_analizi"]["iplik_sayisi"]
    }
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Bir fabrikada Renk Ayrıştırma, Görsel Arama, Kusur Tespiti ve RAG Asistanı gibi 4 farklı yapay zeka modülünü birbirinden bağımsız scriptler yerine **Merkezi Bir Orkestratör (Suite Orchestrator)** altında birleştirmenin endüstriyel avantajı nedir?

> **Mentor Cevabı:**
> 1. **Uçtan Uca Aksiyon Döngüsü (Closed-Loop Action):** Kusur tespit motoru bir hata bulduğunda sistem yalnızca *hata var* deyip durmaz; RAG motorunu otomatik tetikleyerek teknisyene *hangi ayarı değiştirmesi gerektiğini* (`45 cN gerginlik`) anında söyler.
> 2. **Bütünleşik Kalite Puanlama:** Halının sadece dokusu değil, ipliklerinin Delta-E renk toleransı ve katalog uyumu da tek bir `Genel Kalite Skoru`na ağırlıklı olarak etki ederek çok boyutlu kalite güvencesi sağlar.
> 3. **ERP/SCADA Entegrasyonu:** Tek bir API / JSON arayüzü ile fabrikanın merkezi otomasyon sistemine tek noktadan bağlanır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
