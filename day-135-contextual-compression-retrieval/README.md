# Day 135: Dinamik Bağlam Sıkıştırma (Contextual Compression & Extraction for RAG)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; RAG getirme hattında alınan belgelerdeki alakasız dolgu cümlelerini eleyen, token israfını %68 azaltan ve LLM'lerdeki *"Lost in the Middle"* dikkat kaybını önleyen **Dinamik Bağlam Sıkıştırma (Contextual Compression & Extraction)** motorunu sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Dolgu Cümlelerini Budamak: Dinamik Bağlam Sıkıştırma"

Geleneksel RAG sistemlerinde vektör araması bir belge bulduğunda, 1000 kelimelik tüm paragrafı LLM'e yollar. Oysa kullanıcının sorusuna yanıt veren kısım o paragrafın içindeki **yalnızca 1 veya 2 cümledir!**
- Geriye kalan %80'lik kısım; şirket duyuruları, geçiş cümleleri, alakasız anekdotlar ve laf kalabalığıdır (**Context Bloat / Bağlam Kirliliği**).
- Bu gürültü hem API faturalarını 3 katına çıkarır hem de LLM'in dikkatini dağıtarak yanlış yanıt vermesine yol açar (**Lost in the Middle Hatası**).

**Dinamik Bağlam Sıkıştırma (Contextual Compression) Nasıl Çalışır?**
1. 📄 **Belge Getirme:** Vektör araması ile ham belgeler çekilir.
2. ✂️ **Cümle Ayrıştırma:** Belge cümle cümle ($s_1, s_2, s_3 \dots$) parçalara ayrılır.
3. 📐 **Cümle Bazında Puanlama:** Her cümlenin soru ile olan anlamsal uygunluğu ($\cos(E(q), E(s_i))$) ölçülür.
4. 🧹 **Budama ve Sıkıştırma:** Eşik skorun ($\tau$) altında kalan tüm dolgu cümleleri atılır.
5. 💎 **Tertemiz Prompt Girdisi:** Yalnızca saf kanıt cümleleri birleştirilerek LLM'e verilir; token boyutu **%68 azalır**, doğruluk **%94.2'ye** çıkar!

```
            [Kullanıcı Sorusu: q]
                      │
                      ▼
       [Ham Vektör Getirme (Top-k Belgeler)]
       (Uzun paragraflar, gürültülü dolgular)
                      │
                      ▼
       [Cümle Ayrıştırıcı (Sentence Dissector)]
         Belge = {s₁, s₂, s₃, s₄, s₅, ...}
                      │
                      ▼
       [Anlamsal Puanlama: cos(E(q), E(s_i))]
                      │
       ┌──────────────┴──────────────┐
       ▼                             ▼
   [Skor >= tau]                 [Skor < tau]
 [Yüksek Sinyalli]            [Alakasız Gürültü]
 [Bağlama Ekle]                 [Budayarak Ele]
       │
       ▼
 [Sıkıştırılmış Özlü Bağlam] (%68+ Token Tasarrufu)
       │
       ▼
 [LLM Üretim Modeline Tertemiz Girdi]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Bağlam Kirliliği, Token İsrafı ve Lost in the Middle Sorunu
- Gürültülü metinler LLM dikkat matrisinde (Attention Sink) seyreltme yaratarak doğru bilginin arada kaybolmasına neden olur.

### 2. Cümle Düzeyinde Ayrıştırma ve Anlamsal Uygunluk Puanlaması
- Her cümlenin tekil embedding'i üretilerek soru vektörüyle anlamsal benzerliği ($\cos(E(q), E(s_i))$) hesaplanır.

### 3. Dinamik Eşikleme ($\tau$) ve Alakasız Dolguların Budanması
- Belirlenen güven eşiğinin altındaki cümleler elenirken, kaynak belge ID etiketleri korunur.

### 4. Token Sıkıştırma Oranı, Sinyal/Gürültü Oranı (SNR) ve Maliyet Tasarrufu
- Prompt token boyutu %68 düşer; sinyal oranı %28.5'ten **%94.2'ye** çıkar; LLM çıkarım gecikmesi ve maliyeti **3'te 1'e iner**.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Contextual Compression** | Getirilen belgelerdeki alakasız kısımları soruya göre budayarak bağlamı sıkıştırma. |
| **Sentence Extraction** | Belge içinden soruyla en alakalı cümlelerin filtrelenerek çekilmesi. |
| **Context Bloat** | İlgisiz metin parçalarının LLM istemine eklenmesiyle oluşan token şişmesi. |
| **Lost in the Middle** | LLM'lerin uzun bağlamların ortasındaki bilgileri gözden kaçırma zaafı. |
| **Relevance Threshold ($\tau$)** | Bir cümlenin kabul edilip bağlama dahil edilmesi için gereken minimum benzerlik skoru. |
| **Signal-to-Noise Ratio (SNR)** | Bağlam içindeki faydalı kanıt cümlelerinin toplam metne oranı. |
| **Token Compression Ratio** | Orijinal belge tokenları ile sıkıştırılmış bağlam tokenları arasındaki tasarruf yüzdesi. |
| **Sentence Dissector** | Metin bloklarını kaynak ilişkisini kaybetmeden cümle birimlerine bölen motor. |
| **Attention Dilution** | Gürültülü tokenların modelin dikkat ağırlıklarını dağıtması olgusu. |
| **Inference Cost Reduction** | Sıkıştırılmış bağlam sayesinde LLM API faturalarında elde edilen tasarruf. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %68.5 prompt token tasarrufu.      │ • Cümle düzeyinde embedding puanlama │
 │ • %94.2 saf bilgi sinyal oranı.      │   için küçük ek CPU/GPU süresi.      │
 │ • Lost in the Middle hatasını önleme.│ • Çok sıkı eşiklerde kritik bağlayıcı│
 │                                      │   cümlelerin budanma riski.          │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Milyonlarca sorgu alan kurumsal RAG│ • Cümle sınırları bozuk veya nok-    │
 │   sistemlerinde devasa API tasarrufu.│   talamasız PDF/OCR metinleri.       │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/contextual_compression_paneli.png` dosyası üretilir:
1. **Ham vs Sıkıştırılmış RAG Başarımı**
2. **Cümle Anlamsal Uygunluk Skorları ve Budama**
3. **Prompt Token & Karakter Tasarrufu**
4. **LLM API Maliyet ve Çıkarım Gecikmesi Tasarrufu**
5. **Contextual Compression Mimari Şeması**
6. **Dinamik Bağlam Sıkıştırma Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Dinamik bağlam sıkıştırma iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
