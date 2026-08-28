# Day 63: Pydantic v2 ile Tip Güvenli Girdi/Çıktı Sözleşmeleri & AI Domain Modelleri

[![License: Private All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.13.3-green.svg)](https://docs.pydantic.dev/)
[![Tests: 7 Passed](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)](testler/)

Üretim seviyesi Yapay Zeka, Bilgisayarlı Görü ve LLM sistemlerinde uçtan uca veri bütünlüğünü (Data Integrity) sağlamak, mikrosaniye seviyesinde doğrulama gecikmesiyle çalışmak ve LLM çıktılarını yapılandırılmış sözleşmelere (Structured Contracts) bağlamak amacıyla geliştirilmiş **Pydantic v2 Tip Güvenli AI Domain Modelleri ve Doğrulama Motoru**.

---

## 1. 🎯 Günün Konusu & Teorik/Matematiksel Derinlik

### A. Çözülen Temel Problem ve Endüstriyel Senaryo
Modern yapay zeka boru hatlarında (AI Pipelines) ve mikromimari servislerde:
1. **Şema Sapması (Schema Drift) ve Veri Bozulması:** Tip denetimi yapılmayan gevşek Python sözlükleri (`dict`) veya gevşek JSON verileri, eksik anahtarlar (`KeyError`), geçersiz aralıklar ($[0, 1]$ dışındaki olasılıklar) veya geometrik tutarsızlıklar ($x_{\min} \ge x_{\max}$) sebebiyle model çıkarım anında veya veritabanı yazma aşamasında sistem çökmelerine yol açar.
2. **LLM Çıktı Stokastisitesi:** Büyük Dil Modellerinin ürettiği JSON yanıtları sözleşmelere uymayabilir. Pydantic v2 `model_validate_json()` ile Rust çekirdeğinde (`pydantic-core`) milisaniye-altı ($< 10\,\mu\text{s}$) sürede doğrulanır.
3. **Vektör & Hiperküre Kısıtları:** Cosine Similarity arama motorlarına giren embedding vektörlerinin $L_2$-normu ($\|e\|_2 = 1.0$) olmak zorundadır. Aksi halde FAISS ve benzeri indekslerde arama sonuçları bozulur.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          PYDANTIC v2 AI DOMAIN SÖZLEŞME VE DOĞRULAMA MİMARİSİ                             │
│                                                                                                           │
│  [Giriş: Ham JSON / API İstek] ──► [Pydantic v2 Rust Çekirdeği] ──► [Tip Güvenli Domain Modelleri]       │
│                                           │                                │                             │
│                                     (Hata Var mı?)                         ├──► GorselMetadatasi         │
│                                      /          \                          ├──► BoundingBoxModeli        │
│                                  (EVET)        (HAYIR)                     ├──► NesneTespitiSonucu       │
│                                    ▼              ▼                        ├──► VektorEmbeddingSozlesmesi│
│                             ValidationError   Model Çıkarımı               └──► InferenceYanitSozlesmesi │
│                           (422 Unprocessable) (YOLOv8 / ViT)                                              │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### B. Matematiksel Formülasyon ve Doğrulama Cebiri

1. **Küme Teorisi ve Projeksiyon Filtresi:**
   Ham yapılandırılmamış payload uzayı $\mathcal{U}$'dan tip güvenli geçerli durum uzayı $\mathcal{X}$'e izdüşüm:
   $$\mathcal{V}(u) = \begin{cases} x \in \mathcal{X}, & \text{eğer } \forall f \in \text{Fields}(x), \Phi_f(u_f) = \text{True} \\ \bot (\text{ValidationError}), & \text{aksi halde} \end{cases}$$

2. **Geometrik BoundingBox Tutarlılığı ve IoU:**
   $$x_{\min} < x_{\max} \quad \land \quad y_{\min} < y_{\max} \quad \forall (x, y) \in [0.0, 1.0]^2$$
   $$\text{IoU}(A, B) = \frac{\text{Alan}(A \cap B)}{\text{Alan}(A \cup B)} = \frac{\max(0, x_2^{\cap} - x_1^{\cap}) \cdot \max(0, y_2^{\cap} - y_1^{\cap})}{\text{Alan}(A) + \text{Alan}(B) - \text{Alan}(A \cap B)}$$

3. **Birim Hiperküre $L_2$-Norm Kısıtı:**
   $$\|e\|_2 = \sqrt{\sum_{i=1}^d e_i^2} \in [1.0 - \epsilon, 1.0 + \epsilon], \quad \epsilon = 0.05$$

---

### C. SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Rust tabanlı `pydantic-core` ile $20\times - 50\times$ doğrulama hızı; OpenAPI ve JSON Schema ile LLM Structured Outputs tam uyumu; `extra='forbid'` ile sıfır veri kirliliği. |
| **Weaknesses (Zayıf Yönler)** | Katı tip dönüşümleri sebebiyle örtük (implicit) tip dönüşümlerinin kalkması; özel validatörlerin (`mode='after'`) dikkatli tasarlanma gereksinimi. |
| **Opportunities (Fırsatlar)** | Mikroservisler arası asenkron veri transferinde sıfır hata garantisi; LLM Function Calling ve Tool Use çıktılarında %100 doğruluk; otomatik OpenAPI dökümantasyonu. |
| **Threats (Tehditler)** | `extra='allow'` yapılandırmasının açık unutulması durumunda sessiz veri kirlenmesi; çok derin iç içe modellerde aşırı serileştirme yükü. |

---

## 2. 💻 Üretim Seviyesinde Uygulama Mimarisi

Proje modüler bir paket yapısına sahiptir:

- [`src/domain_modelleri.py`](src/domain_modelleri.py): `GorselMetadatasi`, `BoundingBoxModeli`, `NesneTespitiSonucu`, `VektorEmbeddingSozlesmesi`, `InferenceIstekSozlesmesi`, `InferenceYanitSozlesmesi`.
- [`src/sozlesme_dogrulayici.py`](src/sozlesme_dogrulayici.py): `SozlesmeDogrulayici` (doğrulama & JSON Schema), `PydanticBenchmarkEngine` (Rust çekirdek hız testi).
- [`src/gorsellestirici.py`](src/gorsellestirici.py): `PydanticGorsellestirici` (6 panelli teşhis panosu).
- [`ana_akis.py`](ana_akis.py): Uçtan uca yürütme betiği.
- [`testler/test_domain_modelleri.py`](testler/test_domain_modelleri.py): 7 kapsamlı birim testi (%100 Başarı).

---

## 3. 🧪 Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

**Görev:** Birden fazla nesne tespitinin listesini içeren bir grupta, aynı sınıf için IoU değeri $0.90$'dan büyük olan kopyaları otomatik tespit eden ve ayıklayan bir `@field_validator('tespitler')` kuralı yazmak.

**Eksiksiz Kod Çözümü:**
```python
from typing import List
from pydantic import BaseModel, field_validator
from src.domain_modelleri import NesneTespitiSonucu

class BirlestirilmisTespitListesi(BaseModel):
    tespitler: List[NesneTespitiSonucu]

    @field_validator("tespitler")
    @classmethod
    def cakisan_kopyalari_filtrele(cls, tespit_listesi: List[NesneTespitiSonucu]) -> List[NesneTespitiSonucu]:
        """Aynı sınıfa ait ve IoU > 0.90 olan kopyaları eler (Soft-NMS benzeri filtre)."""
        filtrelenmis = []
        for yeni in tespit_listesi:
            cakisiyor = False
            for mevcut in filtrelenmis:
                if yeni.sinif_adi == mevcut.sinif_adi and yeni.kutu.iou(mevcut.kutu) > 0.90:
                    cakisiyor = True
                    break
            if not cakisiyor:
                filtrelenmis.append(yeni)
        return filtrelenmis
```

---

## 4. 📊 Doğrulama ve Benchmark Metrikleri

10,000 sentetik API istek payload'ı ile gerçekleştirilen Rust Core test sonuçları:

| Metrik | Ölçülen Değer | Birim / Açıklama |
|---|---|---|
| **Test Edilen Payload** | $10,000$ | Adet Sentetik İstek |
| **Doğrulama Hızı (Validation Throughput)** | **$143,074$** | Payload / Saniye |
| **Tekil Doğrulama Gecikmesi** | **$6.99$** | Mikrosaniye ($\mu\text{s}$) |
| **Serileştirme Hızı (`model_dump_json`)** | **$470,781$** | Payload / Saniye |
| **Geçersiz Veri Engelleme Oranı** | **$\%100.00$** | Tam Güvenlik |
| **Birim Test Başarımı** | **$7 / 7$ PASSED** | %100 Başarı |

---

## 5. 🚀 Kurulum ve Çalıştırma

```bash
# 1. Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 2. Ana yürütme betiğini çalıştırın
python ana_akis.py

# 3. Birim testleri koşun
pytest testler -v
```

---

## 6. ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Pydantic v2'de `@model_validator(mode='before')` ile `@model_validator(mode='after')` arasındaki fark nedir ve LLM/AI boru hatlarında hangisi ne zaman tercih edilmelidir?

> **Mentor Cevabı:**
> 1. **`mode='before'` (Ham Veri Normalizasyonu):** Pydantic henüz hiçbir alanın tipini dönüştürmeden önce ham `dict` veya `Any` nesnesine uygulanır. LLM'lerin ürettiği Markdown bloklarını (` ```json ... ``` `), string olarak gelen sayıları veya büyük/küçük harf tutarsızlıklarını temizlemek (data sanitization) için kullanılır.
> 2. **`mode='after'` (Geometrik ve Mantıksal Bütünlük):** Tüm alanlar kendi tiplerine başarıyla ayrıştırıldıktan sonra (`self: Model`) çalışır. Örneğin $x_{\min} < x_{\max}$ geometrik kontrolü, $L_2$-norm denetimi veya başlangıç zamanı $<$ bitiş zamanı gibi çoklu alanların birbiriyle ilişkisini doğrulamak için kullanılır.

---

## 7. 📜 Lisans & Metaveri

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
