# Day 157: Soru Zorluğuna Göre Dinamik Hesaplama ve Token Bütçesi Tahsisi (Dynamic Compute Allocation)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; OpenAI o1 ve DeepSeek-R1 felsefesiyle, her kullanıcı sorgusuna sabit maksimum bütçe harcamak yerine soru zorluğuna göre dinamik test-time compute tahsis eden (**Easy vs Hard Query Routing**), GPU maliyetini ve gecikmesini $\%60-\%85$ optimize eden mimariyi hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Dinamik Compute Tahsisi (Dynamic Compute Allocation)" Nedir ve Neden Hayati Önemdedir?
- **Sorun (Sabit Bütçe İsrafı):**
  Bir yapay zeka sistemine *"Türkiye'nin başkenti neresidir?"* diye sorduğunuzda, modelin 4.096 token boyunca düşünüp ağaç araması yapması gereksizdir. Bu durum hem kullanıcıyı 5 saniye bekletir hem de sunucu faturasını onlarca kat artırır.
- **Çözüm (Zorluğa Göre Akıllı Rotalama):**
  1. **Kolay Soru (Trivia/Doğrudan Bilgi):** $32$ token (Doğrudan Yanıt / System 1 $\to 40\text{ ms}$).
  2. **Orta Soru (Aritmetik/2-3 Adım):** $512$ token (Standart Düşünce Zinciri CoT $\to 320\text{ ms}$).
  3. **Zor Soru (Teorem/Algoritma/AIME):** $4096$ token (Derin Ağaç Araması MCTS $\to 2.400\text{ ms}$).

```
         DYNAMIC COMPUTE & TOKEN ROUTING
  [Girdi Sorusu] ────────┐
                         ▼
            [Zorluk & Entropi Tahmincisi]
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     [KOLAY: %15]   [ORTA: %45]    [ZOR: %85]
          │              │              │
     System 1        Standart      Derin Arama
     Doğrudan         CoT Zinciri    MCTS + Ağaç
     32 Token        512 Token     4096 Token
     (40 ms)         (320 ms)      (2400 ms)
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Dinamik Token Bütçesi Tahsis Fonksiyonu
- Zorluk skoru $D(q) \in [0, 1]$ için tahsis edilen bütçe:
  $$B(q) = \text{clip}\left(B_{\text{min}} \cdot \exp(\alpha \cdot D(q)), B_{\text{min}}, B_{\text{max}}\right)$$
- Kolay sorgularda $B_{\text{min}} = 32$, zor sorgularda $B_{\text{max}} = 4096$ olarak ölçeklenir.

### B. İlk-Token Logit Entropisi ($\mathcal{H}$) ve Belirsizlik Analizi
- Prompt zorluğu, ilk adımlardaki sonraki token olasılık dağılımının entropisi ile kestirilebilir:
  $$\mathcal{H}(X) = -\sum_{i} P(w_i \mid q) \log P(w_i \mid q)$$
- Düşük entropi $\implies$ Bilinen gerçek $\implies$ Düşük bütçe tahsisi.

### C. Maliyet ve Gecikme Pareto Analizi
- 6 farklı zorluktaki soru seti üzerinde sabit bütçe toplam $24.576$ token tüketirken, dinamik rotalama ile $9.280$ token harcanmıştır ($\%62.2$ Tasarruf, $2.61\times$ Hızlanma).

### D. Çıkarım Zamanı Hesaplama Skalalaması (Inference Scaling Law)
- Kolay görevlerde ek token başarımı artırmaz (Doyum noktası). Karmaşık görevlerde ise her $2\times$ token artışı logaritmik doğruluk kazancı sağlar.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Dynamic Compute Allocation** | Soru zorluğuna göre çıkarım zamanı bütçesini dinamik olarak belirleme. |
| **Easy vs Hard Routing** | Girdiyi doğrudan hafif modeller/modlar ile ağır akıl yürütme motorları arasında yönlendirme. |
| **Test-Time Compute** | Modelin cevabı üretmeden önce düşündüğü token bütçesi ve arama süresi. |
| **System 1 vs System 2** | Hızlı refleksif yanıtlar (System 1) ile derin adımlı akıl yürütme (System 2) ayrımı. |
| **Logit Entropy** | Modelin sonraki kelime tahminindeki belirsizlik ve varyans ölçüsü. |
| **Pareto Efficiency** | Doğruluktan ödün vermeden minimum maliyet ve gecikme sağlama durumu. |
| **Token Budgeting** | Bir oturumda üretilecek maksimum düşünce ve yanıt tokenı sınırı. |
| **Inference Cost Optimization** | GPU çalışma süresi ve API token maliyetlerini düşürme mühendisliği. |
| **MCTS Rollout Depth** | Ağaç aramasında bir hipotezi test etmek için inilen maksimum derinlik. |
| **Prompt Complexity Score** | Girdinin içerdiği kısıt ve matematiksel kavram yoğunluğu skoru. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • GPU ve API maliyetlerinde %60-%80  │ • Çok nadir durumlarda zor bir      │
 │   oranında radikal tasarruf.         │   sorunun yanlışlıkla kolay olarak   │
 │ • Kullanıcı gecikmesinde 2.6x+ hız.  │   sınıflandırılması riski.           │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Hibrit model mimarilerinde         │ • Rotalayıcı modelin kendi gecikme   │
 │   (Small Router + Heavy Reasoner)    │   ek maliyeti (Router latency).      │
 │   milyonlarca dolar altyapı karı.    │                                      │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/dynamic_compute_allocation_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
