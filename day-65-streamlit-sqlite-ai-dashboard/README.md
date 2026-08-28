# Day 65: SQLite Destekli CRUD, Model Çıkarım Logları ve Kalıcı AI Yönetim Paneli

[![License: Private All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62.0-FF4B4B.svg)](https://streamlit.io/)
[![SQLite 3](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg)](https://www.sqlite.org/)
[![Tests: 7 Passed](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)](testler/)

Üretim ortamındaki yapay zeka ve bilgisayarlı görü modellerinin çıkarım telemetrisini (Inference Telemetry), tespit sonuçlarını ve insan geri bildirimlerini (Human-in-the-Loop) kalıcı ve asenkron ilişkisel veritabanında saklamak, görselleştirmek ve yönetmek için geliştirilmiş **Streamlit + SQLite AI Model Yönetim Paneli**.

---

## 1. 🎯 Günün Konusu & Teorik/Matematiksel Derinlik

### A. Çözülen Temel Problem ve Endüstriyel Senaryo
Yapay zeka modelleri canlıya alındığında:
1. **Sessiz Model Bozulması (Silent Model Degradation) ve Veri Sapması (Data Drift):** Modelin zaman içindeki güven skorları düşebilir, yanlış pozitif oranları artabilir veya belirli sınıflar nadirleşebilir. Bellekte tutulan loglar pod yeniden başladığında kaybolur.
2. **Denetim İzi (Audit Trail) ve İnsan Denetimi (Human-in-the-Loop):** Kritik endüstriyel kalite kontrol ve tıp sistemlerinde uzman denetçilerin model kararlarını incelemesi, hatalı çıkarımları etiketlemesi ve yeniden eğitim için veri havuzuna aktarması gerekir.
3. **Gömülü ve Hafif İlişkisel Depolama (Embedded SQLite WAL):** Ağır harici veritabanı kümeleri (Postgres/Cassandra) kurmak yerine, yerel diskte çalışan ve Write-Ahead Logging (`PRAGMA journal_mode=WAL;`) ile eşzamanlı okumalara olanak tanıyan SQLite, uç/kenar (Edge) cihazlarda sıfır konfigürasyonla çalışır.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      SQLITE & STREAMLIT AI ÇIKARIM YÖNETİM VE DENETİM MİMARİSİ                            │
│                                                                                                           │
│  [FastAPI / Model Çıkarımı] ──► [AIVeritabaniYoneticisi (SQLite WAL)] ──► [cikarim_loglari Tablosu]       │
│                                              │                                    │                       │
│                                              └──► [B-Tree İndeksleri]             └──► [nesne_tespitleri] │
│                                                          │                                                │
│                                                          ▼                                                │
│  [Streamlit Dashboard] ◄────── [SQL Filtreleme / Kayan Ortalama] ◄────────────────────────────────────────┤
│           │                                                                                               │
│           ├──► [1. KPI Kartları: İstek, Gecikme, Güven]                                                   │
│           ├──► [2. İnteraktif DataFrame & Filtreleme]                                                     │
│           ├──► [3. Telemetri Grafikleri & Sınıf Dağılımı]                                                 │
│           └──► [4. İnsan Denetimi: Geri Bildirim ve Etiketleme (Human-in-the-Loop)]                       │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### B. Matematiksel Formülasyon ve İstatistiksel Analiz

1. **B-Tree İndeks Arama Karmaşıklığı:**
   $N$ adet çıkarım günlüğü ve $B$ dallanma faktörü için `istek_id`, `model_adi` ve `olusturma_tarihi` indeksli arama süresi:
   $$\mathcal{O}_{\text{Search}} = \mathcal{O}(\log_B N)$$

2. **Kayan Pencere Telemetri İstatistikleri (Sliding Window Moving Average):**
   Gecikme süresi $L_i$ ve $W$ pencere boyutu için anlık kayan ortalama $\mu_L(W)$ ve standart sapma $\sigma_L(W)$:
   $$\mu_L(W) = \frac{1}{|W|} \sum_{i \in W} L_i, \quad \sigma_L(W) = \sqrt{\frac{1}{|W|} \sum_{i \in W} (L_i - \mu_L)^2}$$

3. **Düşük Güven Anomali Oranı (Low-Confidence Alarm Ratio):**
   $$\text{Alarm Oranı} = \frac{1}{|W|} \sum_{i \in W} \mathbb{I}(\text{Güven}_i < \gamma_{\text{threshold}})$$

---

### C. SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Sıfır konfigürasyon; gömülü tek dosya veritabanı (`.db`); WAL modu ile eşzamanlı yüksek hızlı okuma; Streamlit ile koddan doğrudan interaktif gösterge paneli. |
| **Weaknesses (Zayıf Yönler)** | SQLite'ın tek yazıcı (single-writer) mimarisi sebebiyle saniyede on binlerce eşzamanlı yazmada kilit beklemesi (`database is locked`); Streamlit'in her widget etkileşiminde tüm betiği yeniden çalıştırması. |
| **Opportunities (Fırsatlar)** | Model sapması (drift) ve anomalilerin erken tespiti; uç cihazlarda (Edge AI) bağımsız denetim kaydı tutma; Human-in-the-Loop ile aktif öğrenme (Active Learning) veri seti toplama. |
| **Threats (Tehditler)** | Çok büyük veri tabanlarında ($>10\text{ GB}$) disk I/O darboğazı; yetersiz yedekleme durumunda disk bozulması riski. |

---

## 2. 💻 Üretim Seviyesinde Uygulama Mimarisi

Proje modüler bir paket yapısına sahiptir:

- [`src/veritabani_yoneticisi.py`](src/veritabani_yoneticisi.py): `AIVeritabaniYoneticisi` (SQLite WAL bağlantısı, B-Tree indeksleme, atomik CRUD işlemleri, metrik agregasyonu).
- [`src/analiz_motoru.py`](src/analiz_motoru.py): `AITelemetriAnalizci` (Pandas telemetri analizi, sınıf frekansları, düşük güven anomalileri).
- [`src/app_dashboard.py`](src/app_dashboard.py): Streamlit interaktif kullanıcı paneli (KPI kartları, log tablosu, grafikler, geri bildirim formu).
- [`src/gorsellestirici.py`](src/gorsellestirici.py): `DashboardGorsellestirici` (6 panelli teşhis panosu üreticisi).
- [`ana_akis.py`](ana_akis.py): Uçtan uca veritabanı oluşturma, veri doldurma, CRUD doğrulama ve görselleştirme betiği.
- [`testler/test_sqlite_dashboard.py`](testler/test_sqlite_dashboard.py): 7 kapsamlı birim testi (%100 Başarı).

---

## 3. 🧪 Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

**Görev:** Belirli bir zaman aralığındaki (ör. son 24 saat) ve belirli bir güven eşiğinin altındaki çıkarımları çekip, bunları aktif öğrenme (Active Learning) için JSON formatında dışa aktaran bir `AktifOgrenmeVeriToplayici` metodu geliştirmek.

**Eksiksiz Kod Çözümü:**
```python
import pandas as pd
import json
from src.veritabani_yoneticisi import AIVeritabaniYoneticisi

class AktifOgrenmeVeriToplayici:
    """Modelin zorlandığı (düşük güvenli) çıkarımları aktif öğrenme havuzuna aktarır."""

    @staticmethod
    def zor_ornekleri_ihrac_et(db: AIVeritabaniYoneticisi, maks_guven: float = 0.65, hedef_dosya: str = "zor_ornekler.json") -> int:
        sorgu = """
            SELECT c.istek_id, c.model_adi, c.ortalama_guven, n.sinif_adi, n.guven_skoru, n.x_min, n.y_min, n.x_max, n.y_max
            FROM cikarim_loglari c
            LEFT JOIN nesne_tespitleri n ON c.id = n.cikarim_id
            WHERE c.ortalama_guven <= ?
            ORDER BY c.id DESC
        """
        with db._baglanti_al() as conn:
            df = pd.read_sql_query(sorgu, conn, params=[maks_guven])
            if df.empty:
                return 0
            df.to_json(hedef_dosya, orient="records", indent=2)
            return len(df)
```

---

## 4. 📊 Doğrulama ve Benchmark Metrikleri

300 sentetik çıkarım ve 600+ nesne tespiti üzerinde ölçülen telemetri sonuçları:

| Metrik | Ölçülen Değer | Birim / Açıklama |
|---|---|---|
| **Veritabanı Modu** | SQLite 3 (WAL) | Eşzamanlı Okuma Desteği |
| **Toplam Loglanan İstek** | $299$ | Adet Model Çıkarımı |
| **Toplam Tespit Edilen Nesne** | $613$ | Adet Sınır Kutusu |
| **Ortalama Model Gecikmesi** | **$1.79\text{ ms}$** | Milisaniye Seviyesinde |
| **Ortalama Model Güveni** | **$\%79.3$** | Beta Dağılımı $[\alpha=8, \beta=2]$ |
| **İnsan Denetim Kayıtları** | $3$ | Human-in-the-Loop Onayı |
| **Birim Test Başarımı** | **$7 / 7$ PASSED** | %100 Başarı (4.80s) |

---

## 5. 🚀 Kurulum ve Çalıştırma

```bash
# 1. Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 2. Veritabanını oluşturun ve test akışını çalıştırın
python ana_akis.py

# 3. Streamlit Yönetim Panelini başlatın
streamlit run src/app_dashboard.py

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
