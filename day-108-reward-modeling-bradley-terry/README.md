# Day 108: Bradley-Terry Tercih Modellemesi & Skaler Ödül Modeli (Reward Modeling)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; Takviyeli Öğrenme (RLHF) ve doğrudan hizalama (DPO) süreçlerinin temel taşı olan **Bradley-Terry Tercih Modeli**, **Çiftli Karşılaştırma ($y_w \succ y_l$)**, **Skaler Ödül Fonksiyonu ($r_\psi(x, y)$)** ve **Reward Hacking Analizi** mekanizmalarını sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Satranç Elo Puanı ve Jüri Hakemliği Analojisi"

Bir yapay zeka modeline "Bu cevaba 10 üzerinden puan ver" dediğinizde insanlar da modeller de çok tutarsız davranır: Kimi 7 verir, kimi 9 verir.

Ancak bir jüriye iki cevabı yan yana koyup **"Hangisi daha iyi?" ($y_w \succ y_l$)** diye sorduğunuzda insanların uzlaşma oranı **%95'in üzerine** çıkar!

1. ♟️ **Satranç Elo Mantığı (Bradley-Terry):** Tıpkı satrançta Magnus Carlsen bir acemiyi yendiğinde Elo farkının kazanma olasılığını belirlemesi gibi, Bradley-Terry modeli de cevabın ödül farkına ($\Delta r = r_w - r_l$) bakar:
   $$P(y_w \succ y_l) = \frac{1}{1 + e^{-(r_w - r_l)}}$$
2. 🎯 **Skaler Ödül Başlığı (Scalar Score Head):** Model metni baştan sona okur. Cümlenin bittiği son noktada (EOS token) durup tek bir skaler not ($r \in \mathbb{R}$) basar: "Aferin (+2.4 puan)" veya "Çok kötü (-1.8 puan)".
3. 🚨 **Reward Hacking (Goodhart Yasası):** "Bir metrik hedef haline gelirse, iyi bir metrik olmaktan çıkar." Model gerçekte kaliteli cevap vermek yerine sadece uzun yazarak veya kibar sözler tekrarlayarak ödül modelini kandırmaya çalışabilir.

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Bradley-Terry Matematiksel Modellemesi
Tercih veri seti $\mathcal{D} = \{(x, y_w, y_l)\}_{i=1}^N$ üzerinde negatif log-likelihood kaybı:
$$\mathcal{L}_{\text{RM}}(\psi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\psi(x, y_w) - r_\psi(x, y_l) - m) \right] + \lambda \|r\|^2$$

Burada $m \ge 0$ tercih marjini (margin), $\lambda \|r\|^2$ ise ödüllerin aşırı büyümesini engelleyen L2 regülarizasyonudur.

### 2. Skaler Ödül Başlığı ve Son Token (EOS) Havuzlaması
Standart dil modelleri kelime kelime tahmin yaparken ($[B, S, V]$), Ödül Modeli tüm cümlenin bağlamını özetleyen son geçerli token'ın (EOS) gizli durumunu alır ($h_{\text{last}} \in \mathbb{R}^d$) ve 1D skaler projeksiyonla tek bir sayıya dönüştürür:
$$r(x, y) = W_{\text{score}}^T h_{\text{last}}$$

### 3. Reward Hacking, Overoptimization & Goodhart Yasası Analizi
RLHF eğitimi sırasında politika modeli (Actor), ödül modelinin açıklarını keşfedebilir. Örneğin içeriği boş ama gereksiz uzun veya aşırı övgü dolu yanıtlar yüksek ödül alabilir. Bunun önüne geçmek için:
- **KL Cezası (KL Penalty):** Referans modelden uzaklaşmayı cezalandırma.
- **Marjinli Ayrışma:** $r_w - r_l > 0.5$ şartı koşarak ödül dağılımını kalibre etme.

### 4. Endüstriyel Entegrasyon (InstructGPT, LLaMA-2-Chat, Claude RLHF)
- **OpenAI InstructGPT / ChatGPT:** PPO takviyeli öğrenme için 6B/175B parametreli ödül modelleri kullandı.
- **Anthropic Claude (Constitutional AI):** İnsan tercihleri yerine AI-destekli (RLAIF) çiftli karşılaştırmalarla ödül modelleri eğitti.
- **Hugging Face TRL (`RewardTrainer`):** Bradley-Terry marjin kaybı desteğiyle açık kaynak standart oldu.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Reward Model (RM)** | Bir prompt ve yanıt çiftine kalite/güvenilirlik derecesi olarak skaler puan ($r \in \mathbb{R}$) atayan model. |
| **Bradley-Terry Modeli** | Çiftli karşılaştırmalarda kazanma olasılığını ödül farklarının lojistik fonksiyonu olarak tanımlayan olasılık modeli. |
| **Pairwise Preference ($y_w \succ y_l$)** | Aynı soruya verilen iki yanıttan birinin kazanmış (chosen), diğerinin reddedilmiş (rejected) olması. |
| **Scalar Score Head** | Transformer gizli durumunu 1 boyutlu skaler puana eşleyen tek katmanlı lineer projeksiyon başlığı. |
| **Last-Token (EOS) Pooling** | Bir dizinin son anlamlı token'ının gizli durum vektörünü cümlenin genel temsili olarak seçme işlemi. |
| **Reward Margin ($m$)** | Kazanan yanıtın kaybedenden en az belirli bir farkla ayrışmasını zorunlu kılan ceza eşiği. |
| **Reward Hacking** | Modelin gerçek insan niyetini karşılamak yerine ödül fonksiyonundaki açıkları istismar ederek hile yapması. |
| **Goodhart's Law** | "Bir ölçü hedef haline geldiğinde, iyi bir ölçü olmaktan çıkar" ilkesi (Ödül aşırı optimizasyonu). |
| **Pairwise Ranking Accuracy** | Modelin $r_w > r_l$ koşulunu doğru sağladığı çiftlerin toplam veri setine oranı (%). |
| **RLAIF** | İnsan jürisi yerine güçlü bir LLM (örn. Claude 3.5 / GPT-4) kullanarak çiftli tercih etiketleri üretme yöntemi. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • İnsan tercihlerini doğrudan skaler │ • Tek bir skaler sayı çok boyutlu    │
 │   matematiksel puana dönüştürme.     │   kaliteyi (doğruluk, üslup)         │
 │ • %95+ çiftli sıralama doğruluğu.    │   özetlemekte yetersiz kalabilir.    │
 │ • PPO ve DPO için vazgeçilmez temel. │ • Reward Hacking'e karşı hassasiyet. │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Sentetik AI jürisi (RLAIF) ile     │ • Ödül modelinin eğitildiği verideki │
 │   milyonlarca tercih verisi üretme.  │   önyargıların politika modeline     │
 │ • Marjinli optimizasyonla stabilite. │   büyütülerek aktarılması riski.     │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/bradley_terry_reward_paneli.png` dosyası üretilir:
1. **Bradley-Terry Kayıp Eğrisi (Epoklar Boyunca Düşüş)**
2. **Çiftli Sıralama Doğruluğu (% Accuracy: %50 -> %98+)**
3. **Ödül Puanlarının Epoklar Boyunca Ayrışması ($r_w$ vs $r_l$)**
4. **Bradley-Terry Olasılık ve Karar Eşiği Eğrisi ($P(y_w \succ y_l) = \sigma(\Delta r)$)**
5. **Bradley-Terry ve Skaler Başlık Matematiksel Formül Kartı**
6. **Stajyer Notu & Reward Model Endüstri Karar Sertifikası**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Ana kıyaslama ve görselleştirme akışını koşturun
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
