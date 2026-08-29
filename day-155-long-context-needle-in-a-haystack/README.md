# Day 155: Needle In A Haystack (NIAH) Uzun Bağlam Değerlendirme & Akıl Yürütme Motoru

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; uzun bağlamlı temel modellerin (Claude 3.5 Sonnet, Gemini 1.5 Pro, GPT-4o, Llama-3-70B-128k) $1k \dots 128k+$ token uzunluğundaki devasa dokümanlarda bilgiyi geri çağırma kalitesini ölçen **Needle In A Haystack (NIAH)** testini, **2D Retrieval Accuracy Isı Haritasını (Heatmap)**, **"Lost in the Middle"** zaafını ve **Çoklu İğne Akıl Yürütme (Multi-Needle Reasoning)** mimarisini sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Needle In A Haystack (Samanlıkta İğne)" Testi Nedir ve Neden Hayatidir?
- **Sorun:** Bir yapay zeka modeli "128.000 token bağlam destekliyorum" diyebilir; ancak 100 sayfalık bir PDF'in tam ortasına gizlenmiş tek satırlık kritik bir bilgiyi (örn: *"Şirketin gizli PIN kodu: 89341"*) bulup çıkarabiliyor mu?
- **Lost in the Middle Olgusu (Liu et al., 2023):**
  Transformer modellerindeki Dikkat (Self-Attention) mekanizması, dokümanın başına (Primacy bias) ve sonuna (Recency bias) aşırı odaklanır. Dokümanın tam ortasında (%40-%60 derinlik) yer alan bilgiler ise dikkat zayıflaması sebebiyle kolayca unutulur.
- **NIAH Testi:**
  $1k \dots 128k$ token bağlamı ve $\%0 \dots \%100$ derinlik ızgarasında (Grid) modeli test ederek her koordinat için bir doğruluk skoru çıkarır ve 2D renkli bir ısı haritası (Yeşil = Başarılı, Kırmızı = Unutuldu) üretir.

```
         NEEDLE IN A HAYSTACK TEST PIPELINE
  [1. Doküman Üretimi (1k - 128k Token Arkaplan)]
                       │
                       ▼
  [2. İğne Enjeksiyonu (%0 ... %100 Derinlik)]
    --- KRİTİK BİLGİ: X Şifresi = 89341 ---
                       │
                       ▼
  [3. LLM Uzun Bağlam Dikkat Mekanizması]
    Baştaki Bilgi (%0-%20)   : Yüksek Dikkat (Primacy)
    Ortadaki Bilgi (%40-%60) : Düşük Dikkat (Lost-Middle)
    Sondaki Bilgi (%80-%100) : Yüksek Dikkat (Recency)
                       │
                       ▼
  [4. 2D Isı Haritası ve Sağlamlık Skoru Üretimi]
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: 2D Izgara Değerlendirmesi ($L \times D$)
- Bağlam uzunluğu kümesi $\mathcal{L} = \{1k, 2k, \dots, 128k\}$ ve derinlik kümesi $\mathcal{D} = \{0\%, 10\%, \dots, 100\%\}$ için doğruluk matrisi:
  $$\mathbf{M}_{i, j} = \text{Score}\big(\text{LLM}(\text{Haystack}(L_i, \text{Needle at } D_j)), \text{GroundTruth}\big) \in [0, 1]$$

### B. "Lost in the Middle" Dikkat Dağılımı ve U-Eğrisi
- Attention ağırlıkları $A_{i, j} = \text{softmax}(Q_i K_j^T / \sqrt{d_k})$ mesafeye göre asimetrikleşir. Doküman büyüdükçe orta bölgedeki ($D \approx 50\%$) sorgu-anahtar benzerliği seyrekleşir.

### C. Çoklu İğne Akıl Yürütme (Multi-Needle Reasoning-in-a-Haystack)
- Sadece tek bir bilgiyi bulmak yerine, dokümanın farklı derinliklerine dağıtılmış ($15\%, 50\%, 85\%$) 3 ayrı öncülü birleştirerek nihai matematiksel/mantıksal çıkarım yapma.

### D. Uzun Bağlam Gecikme ve KV-Cache Analizi
- $O(N)$ ve $O(N^2)$ bellek ayak izi artışı nedeniyle $1k$ token çıkarımı $15\text{ ms}$ sürerken, $128k$ token çıkarımı $2450\text{ ms}$ seviyesine ulaşır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Needle In A Haystack (NIAH)** | Uzun metin içinde gizlenmiş tekil bir bilginin geri çağrılma başarısını ölçen test. |
| **Lost in the Middle** | LLM'lerin uzun dokümanların ortasındaki bilgiyi hatırlamakta zorlanması olgusu. |
| **Context Window** | Bir modelin tek seferde girdi olarak işleyebileceği maksimum token sayısı. |
| **Retrieval Accuracy** | İğne bilgisinin doğru ve eksiksiz biçimde geri çağrılma yüzdesi. |
| **Primacy Bias** | Modelin metnin en başındaki token'lara orantısız yüksek dikkat vermesi. |
| **Recency Bias** | Modelin metnin en sonundaki token'lara daha taze olduğu için yüksek dikkat vermesi. |
| **Multi-Needle Reasoning** | Farklı derinliklere dağılmış çoklu bilgileri toplayıp birleştirerek akıl yürütme. |
| **RoPE Scaling** | Döner Pozisyonel Kodlama (Rotary Positional Embedding) frekansını uzun bağlama uyarlama. |
| **KV-Cache Memory** | Çıkarım sırasında önceki tüm token'ların Key-Value matrislerini saklayan GPU belleği. |
| **FlashAttention-2** | GPU bellek bant genişliğini optimize eden donanım farkındalıklı dikkat algoritması. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Modelin gerçek uzun bağlam hafıza  │ • Çok büyük bağlamlarda (1M+) test  │
 │   kapasitesini 2D görselleştirme.    │   etmenin yüksek GPU maliyeti.       │
 │ • RAG vs Long-Context tercihi için   │ • Sentetik iğnelerin gerçek dünya    │
 │   objektif kıyaslama metriği.        │   karmaşıklığını tam yansıtamaması.  │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Hukuki sözleşme analizi, tıbbi     │ • Doküman içinde birden çok çelişkili│
 │   tarihçe taraması ve kod tabanı     │   iğne olduğunda halüsinasyon        │
 │   okuma (SWE-bench) yetkinliği.      │   oluşma riski.                      │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/long_context_needle_in_a_haystack_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
