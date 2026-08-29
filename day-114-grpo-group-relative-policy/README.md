# Day 114: Group Relative Policy Optimization (GRPO) ile Critic'siz LLM Akıl Yürütme Eğitimi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; **DeepSeek-R1** ve **DeepSeekMath** modellerinin devrimsel temelini oluşturan, devasa Critic (Value Network) ağını tamamen ortadan kaldıran, **Grup İçi Göreli Ödül Normalizasyonu ($A_i = \frac{r_i - \bar{r}}{\sigma_r}$)** ve kural tabanlı akıl yürütme ödülleriyle çalışan **Group Relative Policy Optimization (GRPO)** algoritmasını sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Critic'siz Grup Akıl Yürütme ve DeepSeek-R1 Mucizesi"

Geleneksel PPO (RLHF) algoritmasında LLM eğitmek çok masraflıydı:
- Eğer ana modelimiz 70 milyar parametreli bir LLM ise, durumların ne kadar iyi olduğunu tahmin etmek için aynı büyüklükte (70B) bir **Critic (Değer Ağı - $V_\phi$)** tutmak zorundaydık.
- Bu durum GPU belleğini tüketiyor ve karmaşık matematik/kodlama sorularında Critic ağı çoğu zaman yanlış değerler tahmin edip eğitimi bozuyordu.

**DeepSeek'in geliştirdiği GRPO (Group Relative Policy Optimization)** bu problemi dâhice bir fikirle çözdü:
1. 👥 **Grup Halinde Düşünme (Group Rollouts):** Model tek bir matematik sorusu aldığında, bu soru için paralel olarak $G$ adet farklı çözüm üretir (örneğin $G=8, 64$ veya $128$ farklı deneme).
2. 🎯 **Kendi İçinde Yarışma (Relative Advantage):** Bir Critic ağına gerek yoktur! Üretilen $G$ çözümün ödülleri (kural tabanlı doğruluk ve format) hesaplanır. Grubun ortalamasından ($\bar{r}$) daha iyi olan çözümler **pozitif avantaj ($A_i > 0$)**, daha kötü olanlar ise **negatif avantaj ($A_i < 0$)** alır:
   $$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}$$
3. 💡 **Aha Moment (Kendi Kendine Düzeltme):** DeepSeek-R1-Zero, hiçbir insan müdahalesi veya SFT olmadan, sadece doğru cevaba verilen kural tabanlı ödüllerle `<think>...</think>` etiketleri arasında düşünmeyi, hatasını fark edip geri dönmeyi ("Wait, that's wrong, let me recalculate...") kendi kendine keşfetmiştir!

```
       PPO (DEĞER AĞI - CRITIC İLE)                                 GRPO (CRITIC-FREE - GRUP Z-SCORE İLE)
 ┌──────────────────────────────────────────────┐       ┌──────────────────────────────────────────────┐
 │ • Actor: 70B Model                           │       │ • Policy (Actor): 70B Model                  │
 │ • Critic (Value V): 70B Model (Ekstra VRAM!) │       │ • Critic: YOK (0 Parametre - %65 Tasarruf!)  │
 │ • Avantaj: GAE-lambda (Değer hatası içerir)  │       │ • Avantaj: Grup Z-Score (r_i - mean(r))/std  │
 │ • Rollout: 1-2 Adet (Pahalı)                 │       │ • Rollout: G=8, 64, 128 Grup Halinde         │
 └──────────────────────────────────────────────┘       └──────────────────────────────────────────────┘
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Critic-Free Grup Göreli Avantaj
GRPO her bir girdi $q$ için $G$ adet yanıt üretir: $\{o_1, o_2, \dots, o_G\}$. Her bir yanıt $r_i$ ödülü alır.
$$A_i = \frac{r_i - \frac{1}{G}\sum_{j=1}^G r_j}{\sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar{r})^2 + \epsilon}}$$
Bu z-score dönüşümü, karmaşık değer fonksiyonlarına ihtiyaç duymadan sıfır yanlı (unbiased) bir referans noktası sağlar.

### 2. Kırpılmış Taşıyıcı Politika Hedefi ve Token Bazlı KL Cezası
Politika oranı $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})}$ olmak üzere:
$$\mathcal{L}_{\text{clip}} = \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t} A_i, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) A_i \right)$$

Modelin dilden uzaklaşmasını engelleyen Schulman yansız KL tahmincisi:
$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$

### 3. Kural Tabanlı Akıl Yürütme Ödülleri (Rule-Based Reasoning Rewards)
DeepSeek-R1 insan geri bildirimli sübjektif ödül modelleri yerine nesnel kurallar kullanır:
- **Doğruluk Ödülü:** Matematiksel sonucun veya kod derleyicisinin doğruluğu ($+1.0$).
- **Format Ödülü:** Düşünme adımlarının `<think>...</think>` ve nihai cevabın `<answer>...</answer>` etiketleri içinde olması ($+0.5$).

### 4. DeepSeek-R1 ve DeepSeek-R1-Zero Devrimi
- **Aha Moment:** Model pekiştirmeli öğrenmeyle uzun düşünme zincirleri oluşturmayı, ara adımları doğrulamayı ve alternatif yollar denemeyi otonom öğrenmiştir.
- **Hesaplama Tasarrufu:** Critic modelinin olmaması, serbest kalan VRAM'in daha büyük grup boyutlarına ($G=64$) ve daha uzun akıl yürütme dizilerine (32k+ context) ayrılmasını sağlamıştır.

---

## 📊 PPO vs GRPO (DeepSeek-R1) Kıyaslama Tablosu

| Kriter | PPO (RLHF) | GRPO (DeepSeek-R1) |
|:---|:---|:---|
| **Critic Modeli ($V_\phi$)** | Var (Model boyutunda devasa ağ) | **YOK (0 Parametre)** |
| **Avantaj Tahmini** | GAE-$\lambda$ (Değer hatası içerir) | **Grup İçi Z-Score ($(r-\bar{r})/\sigma$)** |
| **Grup Örnekleme ($G$)** | 1 - 2 Rollout | **$G = 8, 64, 128$ Paralel Rollout** |
| **Bellek (VRAM) Kullanımı** | %100 (Actor + Critic + Ref + RM) | **%35 (Sadece Policy + Ref)** |
| **Akıl Yürütme (Reasoning) Uyumu** | Düşük / Kararsız | **Lider (SOTA Reasoning - R1)** |

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **GRPO** | Critic ağı olmadan grup içi göreli ödül normalizasyonu ile çalışan pekiştirmeli öğrenme algoritması. |
| **Group Rollout ($G$)** | Tek bir girdi prompt'u için modelden eşzamanlı olarak üretilen $G$ adet paralel çıktı kümesi. |
| **Group-Relative Advantage ($A_i$)** | Grubun ortalaması ve standart sapması kullanılarak hesaplanan Z-Score avantaj değeri. |
| **Critic-Free Architecture** | Değer ağı ($V$) olmadan doğrudan grup istatistiklerine dayanan hafif mimari. |
| **Rule-Based Reward** | Matematik/kod gibi nesnel alanlarda doğrulanabilir kurallara dayalı kesin ödül mekanizması. |
| **Clipped Surrogate Objective** | Politika güncellemesinin aşırı büyük adımlar atmasını engelleyen $[1-\epsilon, 1+\epsilon]$ kırpma mekanizması. |
| **Schulman KL Estimator** | $\frac{\pi_{\text{ref}}}{\pi} - \log \frac{\pi_{\text{ref}}}{\pi} - 1$ formülüyle hesaplanan kesin pozitif KL sapma tahmincisi. |
| **Aha Moment** | Modelin pekiştirmeli öğrenme sırasında kendi hatasını fark edip düzeltmeyi keşfettiği akıl yürütme anı. |
| **Chain of Thought (CoT)** | Modelin cevaba ulaşmadan önce `<think>` etiketleri arasında kurduğu mantıksal ara adımlar dizisi. |
| **Clip Fraction** | Olasılık oranının kırpılma bölgesine giren token adımlarının yüzdesi. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Critic ağı yok -> %65 VRAM         │ • G çok küçük seçilirse (G < 4)      │
 │   tasarrufu ve devasa grup rollout.  │   grup varyansı gürültülü olabilir.  │
 │ • Değer kestirimi hatalarından muaf. │ • Yalnızca kural tabanlı doğrulanabilir│
 │ • Akıl yürütme ve matematikte SOTA.  │   görevlerde (Math/Code) çok etkilidir.│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • 671B MoE modellerini düşük GPU ile │ • Açık uçlu yaratıcı yazarlıkta      │
 │   hizalayıp akıl yürütme kazandırma. │   kural tabanlı ödül tasarlamak zordur.│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/grpo_reasoning_paneli.png` dosyası üretilir:
1. **GRPO Toplam & Politika Kayıp Eğrileri**
2. **Kural Tabanlı Akıl Yürütme Ödül Gelişimi (Mean & Std)**
3. **Politika Referans KL Sapması ($D_{\text{KL}}$)**
4. **Taşıyıcı Oran Kırpılma Oranı (%)**
5. **PPO vs GRPO Mimari Kıyas Matrisi**
6. **GRPO Matematik Kartı & DeepSeek-R1 Kararı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Ana akıl yürütme eğitimi ve görselleştirmeyi koşturun
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
