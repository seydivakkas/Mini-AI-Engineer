# Day 159: Nedensellik Analizi (Causal Inference & Reasoning): Neden-Sonuç Grafı (DAG), Do-Calculus ve Karşıgelişçi Akıl Yürütme

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; yapay zekanın sadece ilişkisel korelasyonlara (Correlation $\neq$ Causation) takılmasını engelleyen, Judea Pearl'ün **Nedensellik Merdiveni (Ladder of Causation: Gözlem $\to$ Müdahale $\to$ Karşıgelişçi)**, **Causal DAG**, **Do-Calculus ($P(Y \mid \text{do}(X))$)** ve **Counterfactual Reasoning** mimarisini sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Korelasyon $\neq$ Nedensellik" ve "Judea Pearl Nedensellik Merdiveni" Nedir?
- **Sorun (Korelasyon Yanılgısı - Simpson Paradoksu):**
  İstatistiksel bir model "Dondurma satışları arttığında boğulma vakaları artar" ilişkisini yakalayabilir. Ancak dondurma satışını yasaklamak boğulmaları durdurmaz; çünkü asıl ortak neden (Confounder) "Sıcak Hava"dır!
- **Çözüm (Nedensellik Merdiveni - 3 Seviye):**
  1. **Seviye 1: Gözlem / İlişkilendirme ($P(Y \mid X)$):** *"X olduğunda Y ne olur?"* (Standart LLM / İstatistik).
  2. **Seviye 2: Müdahale / Eylem ($P(Y \mid \text{do}(X))$):** *"X'i aktif olarak değiştirirsem (müdahale edersem) Y ne olur?"* (Do-Calculus / A-B Testi).
  3. **Seviye 3: Karşıgelişçi Akıl Yürütme ($P(Y_{X=x'} \mid X=x, Y=y)$):** *"Eğer geçmişte X yerine X' yapsaydık, sonuç Y ne olurdu?"* (Retrospektif Analiz / "Ya öyle olmasaydı?").

```
                  [Z: Yaş Grubu]
                  (Konfondör)
                    /       \
         (Backdoor)/         \
                  v           v
            [X: İlaç] ──────> [Y: İyileşme]
           (Müdahale) (Saf    (Sonuç)
                       Etki)
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Backdoor Kriteri ve Do-Calculus Müdahale Formülü
- Konfondör $Z$ üzerinden akan sahte ilişkiyi (Backdoor path) bloke etmek için:
  $$P(Y \mid \text{do}(X = x)) = \sum_{z} P(Y \mid X = x, Z = z) P(Z = z)$$
- Gözlemsel korelasyon ilacın etkisini yanıltıcı şekilde $+\%34.0$ gösterirken, Do-Calculus gerçek saf nedensel etkiyi (ATE) $+\%10.0$ olarak kanıtlar!

### B. Simpson Paradoksu ve Konfondör Bloklama
- Yaş ($Z$) değişkeni hem ilaç alma kararını ($Z \to X$) hem de iyileşme potansiyelini ($Z \to Y$) etkilediği için veriyi stratifiye etmeden analiz etmek ölümcül medikal hatalara yol açar.

### C. 3 Basamaklı Karşıgelişçi Algoritması (Pearl's Three-Step Counterfactuals)
1. **Abduction (Geri Çıkarım):** Bireysel arka plan ($Z = \text{Genç}$) sabitlenir.
2. **Action (Müdahale):** Bireyin aldığı ilaç sanal olarak iptal edilir ($X = 0$).
3. **Prediction (Tahmin):** $P(Y_{X=0} = 1 \mid Z=0, X=1, Y=1) = \%80.0$ olarak hesaplanır.

### D. Zorunluluk Olasılığı (Probability of Necessity - PN)
- İyileşmenin doğrudan ilacın varlığına borçlu olma ihtimali:
  $$\text{PN} = \frac{P(Y=1 \mid X=1, Z) - P(Y=1 \mid X=0, Z)}{P(Y=1 \mid X=1, Z)} = \frac{0.90 - 0.80}{0.90} = 11.1\%$$

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Causal Inference (Nedensel Çıkarım)** | Veriler arasındaki korelasyonlardan bağımsız olarak gerçek neden-sonuç bağını bulma disiplini. |
| **Ladder of Causation** | Judea Pearl'ün 3 basamaklı bilişsel modeli: Gözlem $\to$ Müdahale $\to$ Karşıgelişçi. |
| **Causal DAG** | Değişkenler arasındaki yönlü nedensel akışı temsil eden asiklik çizge. |
| **Confounder (Karıştırıcı/Ortak Neden)** | Hem nedeni hem de sonucu etkileyerek sahte korelasyon üreten üçüncü değişken. |
| **Do-Calculus** | Aktif bir müdahale yapıldığında ($\text{do}(X)$) sistemin davranışını hesaplayan cebir. |
| **Backdoor Criterion** | Bir müdahalenin etkisini hesaplamak için kapatılması gereken konfondör yolları kuralı. |
| **ATE (Average Treatment Effect)** | Bir tedavinin/müdahalenin popülasyon genelindeki net ortalama nedensel etkisi. |
| **Counterfactual (Karşıgelişçi)** | Gerçekleşmiş bir olayın aksini ("Ya öyle olmasaydı?") varsayan hipotetik akıl yürütme. |
| **Abduction** | Gözlemlenen verilerden bireye özgü gizli arka plan durumunu tespit etme adımı. |
| **Simpson's Paradox** | Alt gruplarda geçerli olan bir eğilimin, gruplar birleştirildiğinde tersine dönmesi yanılsaması. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Sahte korelasyonları eleyerek      │ • Doğru Causal DAG grafını kurmak    │
 │   hatasız karar verme yetisi.        │   için derin alan uzmanlığı          │
 │ • Medikal, finans ve hukukta         │   (Domain Knowledge) gereksinimi.    │
 │   hayat kurtaran karşıgelişçi analiz.│                                      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • LLM'lerin nedensel muhakeme        │ • Gözlemlenmeyen konfondörlerin      │
 │   motorlarıyla güçlendirilerek       │   (Unobserved Confounders) varlığı   │
 │   bilimsel keşifler yapması.         │   halinde yanlılık riski.            │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/causal_reasoning_dag_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
