# Day 143: Self-Consistency: Çoklu Akıl Yürütme Yollarında Sıcaklık Örneklemesi ($T$), Ağırlıklı Oylama ve Entropi Analizi

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; akıl yürüten LLM modellerinde (Reasoning LLMs) **Sıcaklık Örneklemesi ($T \in \{0.0, 0.3, 0.7, 1.2\}$)**, **Yol Güven Ağırlıklı Çoğunluk Oylaması (Soft Weighted Majority Voting)**, **Shannon Tahmin Entropisi ($H(Y|x)$)** ve **Epistemik Belirsizlik Ölçümü** boru hattını içermektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Sıcaklık Parametresi ($T$) Akıl Yürütmeyi Nasıl Etkiler?
1. **$T = 0.0$ (Greedy / Deterministik):** Model her zaman en yüksek olasılıklı tek bir kelimeyi seçer. Çeşitlilik sıfırdır; model ilk adımda yanlış bir varsayımla başlarsa geri dönemez.
2. **$T = 0.3$ (Konservatif):** Küçük varyasyonlar üretir ancak çoğu zaman benzer yolları tekrarlar.
3. **$T = 0.7$ (Optimal Akıl Yürütme Sıcaklığı):** Model hem cebirsel hem aritmetik hem de varsayım denetimi gibi farklı çözüm stratejilerini keşfeder. Doğru yanıt tüm geçerli stratejilerin kesişiminde yer alır.
4. **$T = 1.2$ (Kaotik / Aşırı Yaratıcı):** Sıcaklık çok yüksek olduğunda model mantık dışı adımlar atar ve halüsinasyon riski artar.
5. **Shannon Entropisi ($H(Y|x)$):** Örneklenen yanıtlar tek bir cevapta birleşiyorsa entropi sıfıra yakındır (Yüksek Güven). Yanıtlar darmadağınıksa ($0.05, 0.10, 0.55$) entropi yükselir ve modelin kararsız olduğu anlaşılır!

```
               [Kullanıcı Sorusu: x]
                         │
                         ▼
             [Sıcaklık Rejimleri Analizi]
     ┌───────────────┬───────────────┬───────────────┐
     ▼               ▼               ▼               ▼
  [T=0.0]         [T=0.3]         [T=0.7]         [T=1.2]
  Greedy        Konservatif       Optimal         Kaotik
 Entropi: 0.0   Entropi: 0.28   Entropi: 0.49   Entropi: 1.38
 Doğruluk: %60  Doğruluk: %80   Doğruluk: %100  Doğruluk: %70
                     │               │
                     └───────┬───────┘
                             ▼
              [Soft Weighted Majority Voting]
              • Skor(a) = sum( P(trajectory_i) )
              • Kazanan: $0.05 (%90+ Ağırlıklı Güven)
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Sıcaklık Ayarlı Softmax Olasılık Dağılımı
- Çözüm yolları arasındaki seçim olasılığı sıcaklık parametresi $T$ ile ölçeklendirilir:
  $$P(\tau_i \mid x, T) = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$$

### B. Soft Weighted Majority Voting (Ağırlıklı Marjinalizasyon)
- Her yolun sadece varlığı (Hard Voting) değil, o yolun üretim log-olasılığı $P(\tau_i)$ dikkate alınarak ağırlıklı toplam hesaplanır:
  $$\text{Score}_{\text{weighted}}(a) = \frac{\sum_{i=1}^N P(\tau_i) \cdot \mathbb{I}(\hat{y}_i = a)}{\sum_{i=1}^N P(\tau_i)}$$

### C. Shannon Tahmin Entropisi ve Epistemik Belirsizlik ($H(Y|x)$)
- Oylama dağılımının bilgi entropisi hesaplanarak modelin eminlik düzeyi ölçülür:
  $$H(Y \mid x) = - \sum_{y \in \mathcal{Y}} p(y \mid x) \log_2 p(y \mid x)$$
  $H(Y|x) < 0.50$ bit ise model yüksek güvenli; $H(Y|x) \ge 1.20$ bit ise halüsinasyon riskindedir.

### D. Gini Kirliliği ve Saflık Analizi
- Dağılımın saflığı $\text{Gini} = 1 - \sum p(y)^2$ formülüyle denetlenir.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Sampling Temperature ($T$)** | Modelin olasılık dağılımının dikliğini veya düzlüğünü kontrol eden hiperparametre. |
| **Soft Weighted Voting** | Yanıtları yol olasılıkları veya model güven skorlarıyla çarparak marjinalize etme yöntemi. |
| **Hard Majority Voting** | Olasılıklara bakılmaksızın sadece her yolun tek bir oy sayıldığı klasik çoğunluk oylaması. |
| **Predictive Entropy ($H(Y\|x)$)** | Modelin verdiği farklı yanıtların belirsizlik ve kaos derecesini ölçen Shannon metriği. |
| **Epistemic Uncertainty** | Modelin bilgi eksikliğinden veya çelişkili çıkarımlarından kaynaklanan içsel belirsizlik. |
| **Gini Impurity** | Rastgele seçilen iki cevabın farklı olma olasılığını gösteren saflık ölçüsü. |
| **Log-Probability ($\log P(\tau)$)** | Bir düşünce yolunun token başına kümülatif koşullu logaritmik olasılığı. |
| **Temperature Sweep** | En yüksek akıl yürütme başarımını veren sıcaklığı bulmak için yapılan tarama. |
| **Confidence Calibration** | Modelin belirttiği güven skoru ile gerçek doğruluk oranının örtüşmesi durumu. |
| **Path Collapse** | Sıcaklık çok düşük olduğunda modelin tek bir hatalı düşünce yoluna kilitlenmesi riski. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %100 doğruluğa ulaşan optimal T=0.7│ • Sıcaklık yükseldikçe token başına  │
 │   akıl yürütme dengesi.              │   hesaplama maliyetinin sürmesi.     │
 │ • Epistemik belirsizliği (Entropi)   │ • Çok yüksek sıcaklıkta (T=1.2)      │
 │   matematiksel olarak raporlama.     │   halüsinasyon gürültüsünün artması. │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Üretim ortamında belirsiz yanıtları│ • Promptun aşırı kısıtlayıcı olması  │
 │   insan denetçisine (HITL) yönlendirme│   durumunda sıcaklık etkisinin düşmesi│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/self_consistency_majority_voting_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
