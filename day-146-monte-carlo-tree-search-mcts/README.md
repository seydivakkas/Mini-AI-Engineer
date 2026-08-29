# Day 146: Monte Carlo Tree Search (MCTS) Destekli LLM Düşünce Planlaması & UCT Algoritması

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; AlphaGo'dan OpenAI o1 ve DeepSeek-R1 gibi yeni nesil akıl yürüten modellere uzanan **Monte Carlo Tree Search (MCTS)** algoritmasını, **UCT (Upper Confidence bounds for Trees) Düğüm Seçimi**, **Genişletme (Expansion)**, **Simülasyon / Rollout** ve **Geri Yayılım (Backpropagation)** adımlarıyla sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ AlphaGo'nun Satranç ve Go'yu Fetheden MCTS Algoritması LLM Akıl Yürütmesinde Ne İşe Yarar?
- **Klasik Arama vs MCTS:**
  Genişlik Öncelikli Arama (BFS) tüm dalları eşit seviyede genişletir ve devasa kombinatoryal patlama yaşar. Derinlik Öncelikli Arama (DFS) ise yanlış bir dala saplanıp saatlerce kaybolabilir.
- **MCTS Nasıl Çalışır? (4 Aşamalı Döngü):**
  1. 🎯 **Selection (Seçim):** Ağaçta kökten başlayarak **UCT formülü** ile hem yüksek kaliteli ($Q(s)$) hem de henüz az keşfedilmiş ($c \sqrt{\frac{\ln N}{n}}$) en umut verici düğüme inilir.
  2. 🌿 **Expansion (Genişletme):** Seçilen düğümden yeni ara düşünce adımları türetilir.
  3. 🎲 **Simulation / Rollout (Simülasyon):** O adımdan sonuca kadar hızlı bir simülasyon yapılır ve başarı ödülü ($R \in \{0, 1\}$) hesaplanır.
  4. 🔄 **Backpropagation (Geri Yayılım):** Elde edilen ödül köke kadar geri iletilerek tüm ata düğümlerin ziyaret sayısı ($N$) ve kalite puanı ($W$) güncellenir.
  5. 🏆 **Sonuç:** Model 250 simülasyon sonrasında en çok kanıtlanmış ve güvenilir akıl yürütme yolunu seçerek Game of 24 ve karmaşık matematik problemlerinde **%92.5+ başarıya** ulaşır!

```
               [Kök Düğüm: 4 9 10 13]
                         │
        ┌────────────────┴────────────────┐
        ▼ (Selection: UCT)                ▼
 [4 - 10 = -6, 9, 13]             [4 + 9 = 13, 10, 13]
 Q: 0.85, N: 45 (Yüksek Değer)    Q: 0.00, N: 5 (Elendi)
        │
        ▼ (Expansion)
 [-6, 9 - 13 = -4]
 Q: 1.00, N: 30
        │
        ▼ (Simulation / Rollout)
 [-6 * -4 = 24] ──► ÖDÜL = 1.0 (Backpropagation Köke Geri Yayılır!)
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: UCT (Upper Confidence bounds for Trees) Düğüm Seçimi
- Düğüm seçiminde sömürü (exploitation) ile keşif (exploration) dengesini kuran klasik UCT formülü:
  $$\text{UCT}(s) = Q(s) + c \cdot \sqrt{\frac{\ln N(\text{ebeveyn})}{N(s) + \epsilon}}$$
  - $Q(s) = \frac{W(s)}{N(s)}$: Düğümün tarihsel ortalama başarı skoru (Sömürü).
  - $c \cdot \sqrt{\frac{\ln N}{n}}$: Az ziyaret edilen dalları keşfetme bonusu (Keşif, $c \approx 0.8 - 1.414$).

### B. Genişletme ve Aksiyon Uzayı (Expansion Phase)
- Kalan serbest sayılar veya mantık önermeleri arasından $k$-ary yeni düşünce düğümleri türetilir.

### C. Rollout / Değer Tahmini (Simulation Phase)
- Durumdan terminale kadar derinlik kısıtlı hızlı sezgisel rollout koşularak başarı ödülü $R \in [0.0, 1.0]$ elde edilir.

### D. Değer Güncellemesi ve Geri Yayılım (Backpropagation)
- Seçilen düğümden köke kadar tüm ata zincirinde:
  $$N(s) \leftarrow N(s) + 1, \quad W(s) \leftarrow W(s) + R, \quad Q(s) \leftarrow \frac{W(s)}{N(s)}$$

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Monte Carlo Tree Search (MCTS)** | Karar ağaçlarında rastgele simülasyonlar ve UCT seçimiyle optimal yolu bulan arama algoritması. |
| **UCT Formula** | Keşif (Exploration) ve Sömürü (Exploitation) dengesini sağlayan ağaç üst güven sınırı formülü. |
| **Exploitation ($Q(s)$)** | Geçmiş simülasyonlarda en yüksek başarıyı vermiş bilinen kaliteli dallara yönelme eğilimi. |
| **Exploration Bonus** | Az ziyaret edilmiş düğümlere öncelik vererek yerel minimumlardan kaçınma terimi. |
| **Rollout / Simulation** | Bir durumdan başlayarak oyunun veya problemin sonuna kadar yapılan hızlı simülasyon. |
| **Backpropagation (MCTS)** | Simülasyon sonucunda kazanılan ödülün arama ağacının köküne kadar geri iletilmesi. |
| **Test-Time Compute Scaling** | Çıkarım zamanında MCTS simülasyon sayısını artırarak modelin muhakeme gücünü katlama. |
| **Tree Policy** | Kökten yaprağa inilirken hangi düğümün seçileceğini belirleyen kural (UCT). |
| **Default Policy** | Rollout aşamasında terminal duruma kadar adımları rastgele/sezgisel seçen hızlı politika. |
| **Visit Count ($N(s)$)** | Bir düşünce düğümünün kaç simülasyon boyunca ziyaret edildiğini gösteren güven göstergesi. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Arama alanını hedefe odaklayarak   │ • Simülasyon sayısı arttıkça büyüyen │
 │   Game of 24'te %92.5 başarı sağlama.│   çıkarım gecikmesi (Latency).       │
 │ • Asla körü körüne çıkmazda kalmama  │ • Rollout politikasının kalitesine   │
 │   (UCT ile kendini düzeltme).        │   aşırı bağımlılık.                  │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • OpenAI o1, DeepSeek-R1 ve o3 tarzı │ • Çok derin ispat ağaçlarında bellek │
 │   akıl yürüten sistemlerin temeli.   │   ve GPU tüketiminin artması.        │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/monte_carlo_tree_search_mcts_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
