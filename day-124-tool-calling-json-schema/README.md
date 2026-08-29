# Day 124: JSON Schema Destekli Tip Güvenli Tool Calling & Grammar-Constrained Decoding

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; dil modellerinin serbest metin çıktılarındaki sözdizim ve tip uyuşmazlıklarını ortadan kaldıran **OpenAI/Anthropic Tool Calling JSON Schema Standardı**, **Pydantic Benzeri Kesin Tip Doğrulaması (Strict Type Safety)**, **Grammar-Constrained Decoding Durum Makinesi** ve **Hata Onarımlı JSON Ayrıştırıcı** mimarisini sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Ajanların Dış Dünyaya Açılan Güvenli Kapısı: Tool Calling"

Eski nesil LLM ajanlarında modele bir araç çağırtmak için `Eylem: Hesapla(2+2)` gibi serbest metinler yazdırılır ve regex ile ayrıştırılırdı. Ancak bu yaklaşım:
1. Modelin parametre tiplerini karıştırmasına (örn. tamsayı yerine `"iki"` yazması),
2. Zorunlu alanları unutmasına,
3. Bozuk veya eksik parantezli JSON üretmesine (%32 hata oranı) yol açıyordu.

**Modern Function/Tool Calling Çözümü:**
- 📜 **JSON Schema Tanımı:** Modele izin verilen araçlar ve parametre tipleri (`type`, `properties`, `required`, `enum`, `minimum`, `maximum`) kesin kurallarla verilir.
- 🛡️ **Grammar-Constrained Decoding:** Model çıktı üretirken geçersiz JSON tokenları logit maskeleme ile engellenir (%100 geçerli JSON).
- 🔧 **Hata Onarımı (Self-Healing):** Markdown kod çitleri (````json ... ````), fazlalık virgüller (*trailing comma*) ve tek tırnaklar otomatik düzeltilir.
- 🚦 **Tip Validatörü & Dispatcher:** Gelen argümanlar şemadan geçtikten sonra güvenle Python fonksiyonuna aktarılır.

```
      [Kullanıcı İstemi] ──> [LLM + JSON Schema] ──> [Grammar Masking]
                                                            │
                                                     (Geçerli JSON)
                                                            ▼
      [Araç Sonucu] <── [Python Fonksiyonu] <── [Şema Doğrulama]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Serbest Metinden Tip Güvenliğine: JSON Schema ve Function Calling Mimarisi
- **Klasik Regex Ajanları:** Metin manipülasyonuna dayalıdır; model ufak bir format sapması yaptığında tüm ajan akışı çöker.
- **Tip Güvenli Tool Calling:** Fonksiyon adı ve parametreler yapılandırılmış sözleşmelerle (*Contract-Driven Architecture*) tanımlanır.

### 2. OpenAI / Anthropic Endüstri Standardı Şema Spesifikasyonu
- Araçlar `{"type": "function", "function": {"name": "...", "parameters": {...}}}` biçiminde standart JSON Schema olarak tanımlanır.
- `enum` kısıtlamaları halüsinasyon argümanları tamamen engeller.

### 3. Grammar-Constrained Decoding (GBNF / Outlines / Jsonformer) ve Sıfır Sözdizim Hatası
- Üretim sırasında bir sonraki token olasılıkları gramer durum makinesine (*Grammar State Machine*) göre filtrelenir.
- Sonuç: Sözdizimsel geçerlilik **%68.2'den %100.0'a** çıkar; argüman tip hatası **%28.5'ten %0.0'a** düşer.

### 4. Hata Toleranslı Ayrıştırma, Otomatik Onarım ve Self-Healing Doğrulama
- Markdown blokları, eksik kapanış parantezleri ve tek tırnaklı Python sözlük çıktıları anında düzeltilerek ajan kesintisi önlenir.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Tool Calling** | LLM'in harici API ve fonksiyonları yapılandırılmış formatta çağırma yeteneği. |
| **JSON Schema** | JSON verilerinin yapısını, tiplerini ve kurallarını belirleyen standart. |
| **Strict Type Safety** | Argümanların belirtilen tiplere (string, int, bool) mutlak uyum zorunluluğu. |
| **Grammar-Constrained Decoding** | Üretimi belirli bir gramere (JSON/GBNF) zorlayan kısıtlı token örnekleme. |
| **Logit Masking** | Gramere uymayan geçersiz tokenların logitlerini $-\infty$ yaparak elenmesi. |
| **Tool Dispatcher** | Şema doğrulamasından geçen argümanları hedef Python fonksiyonuna yönlendiren modül. |
| **Trailing Comma** | JSON listeleri veya nesnelerinin sonundaki geçersiz fazlalık virgül. |
| **Self-Healing Parser** | Hatalı sözdizimini otomatik tamir edip geçerli JSON nesnesine dönüştüren motor. |
| **Enum Constraint** | Parametrenin yalnızca önceden belirlenmiş değer kümesinden seçilebilmesi kısıtı. |
| **OpenAI Function Spec** | OpenAI ve modern LLM'lerin desteklediği standart araç arayüz formatı. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %100 sözdizimsel geçerli JSON.     │ • Çok sayıda araç şeması tanımlandık-│
 │ • %0 argüman tip uyuşmazlık hatası.  │   ça prompt token maliyetinin artması│
 │ • Halüsinasyonu engelleyen enumlar.  │ • Grammar decoding kütüphanelerinin  │
 │ • Otomatik markdown & hata onarımı.  │   bazı çıkarım motorlarında ek yükü. │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kurumsal ERP, CRM, Borsa ve API    │ • Eksik veya hatalı şema tanımların- │
 │   entegrasyonlarında sıfır hata.     │   da ajanın kilitlenme riski.        │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/tool_calling_paneli.png` dosyası üretilir:
1. **JSON Sözdizimsel Geçerlilik Oranı (% Başarı)**
2. **Argüman Tip Uyumsuzluk Hatası (%)**
3. **Zorunlu Alan (Required Field) Eksikliği (%)**
4. **Şema Doğrulamasından Geçen Örnek Çağrılar**
5. **OpenAI Function JSON Schema Mimarisi**
6. **Tip Güvenli Tool Calling Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Tip güvenli araç çağırma motorunu çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
