# Day 121: ReAct (Reasoning + Acting) Otonom AI Ajanı ve Scratchpad Bellek Mimarisi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7 BAŞLANGICI: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu proje; dil modellerini salt metin üreten statik sistemlerden, dış dünyayla dinamik araçlar aracılığıyla etkileşime geçen otonom karar vericilere dönüştüren **ReAct (Reasoning + Acting)** desenini, **Düşünce-Eylem-Gözlem (Thought-Action-Observation) Döngüsü**nü, **Araç Kayıt Defteri**ni ve **Scratchpad Bellek Tamponu**nu sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Düşün, Harekete Geç, Gözlemle: ReAct Devrimi"

Diyelim ki bir LLM'e *"Türkiye'nin başkenti neresidir ve 2024 nüfusu kaçtır?"* diye sordunuz.
- **Standart LLM (CoT):** Sadece kendi hafızasındaki eski eğitim verilerine güvenir. Nüfus sayısını kafadan uydurur (**Halüsinasyon**).
- **Sadece Araç Çağıran Model (Act-Only):** Düşünmeden körlemesine arama yapar; arama sonucunu nasıl yorumlayacağını ve bir sonraki adımı nasıl planlayacağını bilemez.
- **ReAct Ajanı (Reasoning + Acting - Yao et al., 2023):** Bir insan araştırmacı gibi çalışır:
  1. 💭 **Thought 1 (Düşünce):** *"Önce başkenti bulmalıyım."*
  2. ⚡ **Action 1 (Eylem):** `AramaMotoru[türkiye başkenti]`
  3. 👁️ **Observation 1 (Gözlem):** *"Ankara'dır."*
  4. 💭 **Thought 2 (Düşünce):** *"Başkent Ankara. Şimdi Ankara'nın 2024 nüfusunu aramalıyım."*
  5. ⚡ **Action 2 (Eylem):** `AramaMotoru[ankara nüfus]`
  6. 👁️ **Observation 2 (Gözlem):** *"5,803,482 kişi."*
  7. 🏁 **Final Answer (Nihai Yanıt):** *"Türkiye'nin başkenti Ankara'dır ve nüfusu 5.8 milyondur."*

```
     KULLANICI GÖREVİ (Task Prompt)                     DIŞ DÜNYA & ARAÇLAR (Tools)
 ┌────────────────────────────────────┐            ┌────────────────────────────────────┐
 │ Soru: "Ankara'nın nüfusu kaçtır?"  │            │ • Arama Motoru (Search API)        │
 └─────────────────┬──────────────────┘            │ • Hesap Makinesi (AST Calculator)  │
                   │                               │ • Python / SQL Kod Çalıştırıcı     │
                   ▼                               └─────────────────┬──────────────────┘
      [REACT OTONOM DÖNGÜSÜ]                                         │
 ┌────────────────────────────────────────┐                          │
 │ 1. Thought: Durumu analiz et           │                          │
 │ 2. Action : Araç seç ve çağır ─────────┼──────────────────────────┘
 │ 3. Observation: Sonucu al ve oku <─────┼──────────────────────────┐
 │ 4. Scratchpad : Hafızayı güncelle      │                          │
 └─────────────────┬──────────────────────┘                          │
                   ▼                                                 │
          [HEDEF TAMAMLANDI MI?] ── (Hayır: Yeni Düşünce Üret) ──────┘
                   │
                (Evet)
                   ▼
       [FİNAL YANIT: Final Answer]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & ReAct Paradigması
ReAct (Reasoning + Acting), düşünme (Reasoning Trace) ve eylemde bulunma (Action Execution) süreçlerini birbirine bağlar:
- **Sinerji:** Düşünceler eylemleri planlar ve gözlemleri sentezler; eylemler ise dış dünyadan taze ve doğrulanmış gözlemler toplayarak düşünceleri günceller.

### 2. Scratchpad Bellek ve Bağlam Yönetimi
- **Scratchpad:** Ajanın her adımda ürettiği `Thought`, `Action` ve aldığı `Observation` metinlerini kronolojik olarak biriktiren çalışma alanıdır.
- **Sliding Window:** Bağlam penceresi (Context Window) dolduğunda en eski adımları özetleyerek veya pencereyi kaydırarak token taşmasını önler.

### 3. Dinamik Araç Entegrasyonu ve Hata Geri Bildirimi (Self-Correction)
- **Güvenli Araç İcrası:** AST tabanlı hesap makinesi ve sanal çalışma alanı ile tehlikeli sistem komutları (`import os`, `exec`) engellenir.
- **Self-Correction:** Bir araç hata döndürdüğünde (örn. `Geçersiz ifade`), ajan bu hatayı `Observation` olarak okur ve sonraki adımda alternatif bir sorgu üreterek kendini düzeltir.

### 4. Mimari Kıyaslama (CoT vs Act-Only vs ReAct)
- **Doğruluk:** CoT (%62.5) -> Act-Only (%54.0) -> **ReAct (%94.8)** (+%51.6 Artış).
- **Halüsinasyon:** CoT (%37.5) -> **ReAct (%4.2)** (-%88.8 Düşüş).
- **Hata Kurtarma:** CoT (%0.0) -> **ReAct (%88.4)**.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **ReAct** | Reasoning (Akıl Yürütme) ve Acting (Eylem) paradigmalarını birleştiren ajan deseni. |
| **Thought** | Ajanın mevcut durumunu, amacını ve bir sonraki eylemini planlayan içsel akıl yürütme metni. |
| **Action** | Ajanın belirli bir aracı belirli parametrelerle çağırmak için ürettiği komut bloğu. |
| **Observation** | Çağrılan aracın çalışması sonucu dönen harici veri veya hata mesajı. |
| **Scratchpad** | Ajanın anlık çalışma belleği; geçmiş adımların prompta geri beslendiği tampon alan. |
| **Final Answer** | Ajanın görevi başarıyla tamamladığını belirten nihai yanıt belirteci. |
| **Tool Registry** | Ajanın kullanabileceği tüm araçları ve şemalarını barındıran kayıt defteri. |
| **Trajectory** | Bir ajanın başlangıçtan nihai yanıta kadar izlediği tüm düşünce-eylem-gözlem patikası. |
| **Self-Correction** | Hatalı gözlem alan ajanın hatayı anlayıp alternatif bir eylem planlaması süreci. |
| **Context Window** | Dil modelinin aynı anda işleyebileceği maksimum token sınırıdır. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %94.8 yüksek görev başarısı.       │ • Çok turlu LLM çağrısı nedeniyle   │
 │ • Halüsinasyonu %4.2'ye düşürme.     │   çıkarım gecikmesi ve token maliyeti│
 │ • Hata aldığında kendini düzeltebilme│ • Döngüye girme (Infinite Loop) riski│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kurumsal API'ler, veritabanları ve │ • Dış araçların çökmesi veya yanıt   │
 │   arama motorlarıyla canlı entegrasyon│   vermemesi durumunda gecikme.       │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/react_ajan_paneli.png` dosyası üretilir:
1. **ReAct Çok Adımlı Yürütme Trajectory (Zaman Çizelgesi)**
2. **Çok Adımlı Görev Doğruluk Oranı (% Başarı)**
3. **Halüsinasyon ve Yanlış Bilgi Oranı (%)**
4. **Araç Hatasından Sonra Kendini Kurtarma Oranı (%)**
5. **Görevlerde Araç Kullanım Frekansı (%)**
6. **ReAct Ajan Mimarisi ve Trajectory Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# ReAct otonom ajanını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
