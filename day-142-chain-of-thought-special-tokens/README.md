# Day 142: Açık Akıl Yürütme Akışı (<think> ... </think>), Düşünce Tokenizasyonu ve Self-Consistency

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; DeepSeek-R1 ve OpenAI o1 mimarilerinde standartlaşan **Özel Akıl Yürütme Tokenleri (`<think>`, `</think>`, `<step>`)**, **Düşünce Tokenizasyonu & Ayrıştırma Motoru** ve **Self-Consistency (Çoklu Akıl Yürütme Yollarında Çoğunluk Oylaması)** boru hattını içermektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ `<think>` Tokenleri ve Self-Consistency Neden Devrim Yarattı?
1. **Düşünceyi Yanıttan Ayırma (`<think> ... </think>`):**
   Modelin sesli düşünmesi (Chain-of-Thought) ile kullanıcıya verilen nihai yanıt özel etiketlerle birbirinden ayrılır. Model `<think>` etiketleri arasında ara adımları dener, denklemleri çözer, çelişkileri fark edip kendini düzeltir; `</think>` etiketinden sonra sadece net ve temiz cevabı sunar.
2. **Self-Consistency (Çoğunluk Oylaması):**
   Tek bir düşünce yoluna güvenmek yerine, belirli bir sıcaklıkta ($T=0.7$) modele soru $K=5$ kez çözdürülür:
   - Yol 1: Cebirsel denklem kurdu -> **0.05**
   - Yol 2: Farktan yola çıktı -> **0.05**
   - Yol 3: Varsayım ve çelişki testi yaptı -> **0.05**
   - Yol 4: Birim dönüştürdü -> **0.05**
   - Yol 5: Hızlı sezgisel çıkarma yaptı -> **0.10** (Sapan Yanıt!)
   Çoğunluk oylaması yapıldığında **%80 konsensüsle 0.05 seçilir**, sapan hatalar elenir!

```
               [Kullanıcı Sorusu: x]
                         │
                         ▼
        [Sıcaklık Örneklemesi: T=0.7, K=5]
      ┌──────────┬──────────┬──────────┬──────────┐
      ▼          ▼          ▼          ▼          ▼
    [Yol 1]    [Yol 2]    [Yol 3]    [Yol 4]    [Yol 5]
   <think>    <think>    <think>    <think>    <think>
    $0.05      $0.05      $0.05      $0.05      $0.10
      └──────────┴──────────┬──────────┴──────────┘
                            ▼
              [Self-Consistency Oylaması]
              • Oy Dağılımı: 4 x $0.05, 1 x $0.10
              • Konsensüs Skoru: %80.0
                            │
                            ▼
             [KONSENSÜS YANITI: $0.05 (DOĞRU)]
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Özel Token Tabanlı Düşünce İzolasyonu
- Modelin sözlüğüne eklenen `<think>` (ID: 32000) ve `</think>` (ID: 32001) tokenleri, çıkarım anında metin ayrıştırmasını deterministik kılar ve SFT/RL eğitiminde düşünce maskelemesine olanak tanır.

### B. Self-Consistency ve Akıl Yürütme Marjinalizasyonu (Wang et al.)
- Sıcaklık örneklemesi ($T > 0$) ile üretilen bağımsız akıl yürütme yolları $\mathcal{T}_1, \dots, \mathcal{T}_K$ marjinalize edilerek nihai yanıt çoğunluk oyuyla belirlenir:
  $$\hat{y} = \arg\max_{a \in \mathcal{A}} \sum_{i=1}^K \mathbb{I}(\text{Yanıt}_i = a)$$

### C. Konsensüs Güven Skoru (Consensus Score)
- Kazanan yanıtın aldığı oy oranına göre sistemin güven derecesi hesaplanır:
  $$\text{Consensus\_Score} = \frac{\max_{a} \text{Oy}(a)}{K} \in [0, 1]$$
  Konsensüs skoru $\ge \%60$ ise yanıt güvenli kabul edilir.

### D. Sapan Yol (Outlier / Hallucinated Path) Tespiti ve Gürültü Filtreleme
- Çoğunluktan ayrılan azınlık yollar (örn: $1/5$ oranındaki $0.10$ tahmini) otomatik olarak izole edilir ve elenir.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Special Reasoning Tokens** | Modelin içsel düşünme akışını sınırlayan `<think>` ve `</think>` gibi ayrık belirteçler. |
| **Chain-of-Thought (CoT)** | Karmaşık problemleri adım adım ara mantıksal ifadelere bölerek çözme tekniği. |
| **Self-Consistency** | Aynı soru için birden çok CoT yolu örnekleyip çoğunluk oyuyla nihai cevabı seçme yöntemi. |
| **Majority Voting** | Örneklenen farklı akıl yürütme yollarından en çok tekrar eden yanıtı belirleme işlemi. |
| **Marginalization** | Ara düşünce adımlarının çeşitliliğini göz ardı ederek sadece nihai cevabın olasılık dağılımını birleştirme. |
| **Consensus Score** | Çoğunluk oyunun toplam örnek sayısına oranı ile hesaplanan güven metriği. |
| **Reasoning Trajectory** | Problemin başlangıcından nihai yanıta kadar izlenen belirli bir düşünce adımları dizisi. |
| **Thinking Loss Masking** | Eğitim sırasında düşünce ve yanıt bloklarına farklı kayıp ağırlıkları uygulama tekniği. |
| **Sampling Temperature ($T$)** | Akıl yürütme yollarında yaratıcı ve alternatif stratejiler keşfetmek için kullanılan rastlantısallık katsayısı. |
| **Outlier Reasoning Path** | Çoğunluk konsensüsünden sapan, mantık hatası veya gürültü içeren izole düşünce yolu. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Sapan gürültülü yolları filtreleyen│ • K adet yol örneklemenin hesaplama  │
 │   %80-100 konsensüs güvenilirliği.   │   ve token maliyeti (K katı compute).│
 │ • Tamamen şeffaf ve denetlenebilir   │ • Yanıt gecikmesinin tekil sorguya   │
 │   <think> akıl yürütme blokları.     │   göre artması.                      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Matematik, kod sentezi, tıp ve     │ • Tüm yolların aynı yanılgıya düşmesi│
 │   hukuk gibi kritik doğruluk alanları│   durumunda (Systemic Bias) risk.    │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/chain_of_thought_special_tokens_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
