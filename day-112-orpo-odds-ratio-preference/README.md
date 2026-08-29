# Day 112: Odds Ratio Preference Optimization (ORPO) ile Monolitik LLM Hizalaması

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; ayrı bir Supervised Fine-Tuning (SFT) aşamasını ve dondurulmuş Referans Modeli tamamen ortadan kaldıran, **Tek Aşamalı Monolitik SFT + Tercih Hizalaması (Monolithic SFT + Alignment)** sağlayan **Odds Ratio Preference Optimization (ORPO)** algoritmasını sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Bahis Oranları (Odds) ve Tek Aşamalı Usta Çırak Analojisi"

Geleneksel LLM hizalama süreci 2 aşamalı bir okul gibidir:
1. **Aşama 1 (SFT):** Öğrenciye sadece doğru cevaplar ezberletilir. Ancak öğrenci bu sırada farkında olmadan bazı kötü konuşma kalıplarını da kapabilir (Cross-Entropy Degeneration).
2. **Aşama 2 (DPO / PPO):** Öğrenciye ikinci bir dondurulmuş hoca (Reference Model) eşliğinde doğru ve yanlış cevaplar kıyaslatılır. Bu hem 2 kat zaman alır hem de GPU'da 2 model tutmayı gerektirir.

**ORPO (Odds Ratio Preference Optimization)** ise usta-çırak ilişkisini tek bir aşamaya indirir:
- Çırak doğru cevabı öğrenirken ($\mathcal{L}_{\text{SFT}}$), ustanın gösterdiği yanlış cevabın üretilme **bahis oranını (odds)** eşzamanlı olarak düşürür!

1. 🎲 **Bahis Oranı Mantığı (Odds):** Bir cevabın üretilme olasılığının ($P$), üretilmeme olasılığına ($1-P$) oranıdır:
   $$\text{odds}(y) = \frac{P(y)}{1 - P(y)}$$
2. 🚀 **Sıfır Referans Model (Zero Ref Model):** GPU'da ikinci bir kopya modele gerek yoktur. Yalnızca tek bir model eğitilir (%50 DPO'ya, %75 PPO'ya göre bellek tasarrufu!).
3. ⏱️ **Tek Aşama (Single Stage):** SFT bittiğinde model aynı anda hizalanmış (aligned) olarak çıkar.

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Odds Ratio Matematiksel Modellemesi
Bir modelin $y_w$ yanıtını $y_l$ yanıtına tercih etme gücü Log-Odds oranıyla ölçülür:
$$\log \text{OR}_\theta(y_w, y_l) = \log \left( \frac{P_\theta(y_w \mid x)}{1 - P_\theta(y_w \mid x)} \right) - \log \left( \frac{P_\theta(y_l \mid x)}{1 - P_\theta(y_l \mid x)} \right)$$

Odds Ratio cezası:
$$\mathcal{L}_{\text{OR}}(\theta) = - \log \sigma \left( \log \text{OR}_\theta(y_w, y_l) \right)$$

### 2. Monolitik SFT + Tercih Kaybı
ORPO iki hedefi tek bir kayıp fonksiyonunda birleştirir:
$$\mathcal{L}_{\text{ORPO}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta; y_w) + \lambda_{\text{OR}} \mathcal{L}_{\text{OR}}(\theta; y_w, y_l)$$

Burada $\mathcal{L}_{\text{SFT}}$ modelin dili ve talimatları öğrenmesini sağlarken, $\lambda_{\text{OR}} \mathcal{L}_{\text{OR}}$ reddedilen yanıtlardaki istenmeyen token olasılıklarını eşzamanlı olarak bastırır.

### 3. Sıfır Referans Model ve GPU Bellek Devrimi
- **PPO:** GPU'da 4 model (Actor, Critic, Ref, RM) -> 100% VRAM.
- **DPO:** GPU'da 2 model (Policy, Ref) -> 50% VRAM.
- **ORPO:** GPU'da yalnızca **1 model** (Trainable Policy) -> **25% VRAM!**

### 4. Endüstriyel Entegrasyon (Mistral-NeMo, Llama-3-ORPO, TRL `ORPOTrainer`)
- **Mistral AI & NVIDIA (Mistral-NeMo-12B):** Hizalama hattında ORPO kullanarak tek aşamada SFT+Alignment gerçekleştirdi.
- **Hugging Face TRL (`ORPOTrainer`):** Açık kaynak topluluğunda en hızlı ve en hafif tercih eğitim motoru olarak yaygınlaştı.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Odds (Bahis Oranı)** | Bir olayın gerçekleşme olasılığının gerçekleşmeme olasılığına oranı: $p / (1-p)$. |
| **Odds Ratio (OR)** | İki farklı sonucun bahis oranlarının birbirine bölünmesiyle elde edilen bağıl üstünlük ölçüsü. |
| **Log-Odds Ratio** | Odds oranının doğal logaritması: $\log \text{odds}(y_w) - \log \text{odds}(y_l)$. |
| **ORPO** | Referans model kullanmadan SFT ve tercih hizalamasını tek aşamada birleştiren monolitik algoritma. |
| **Monolithic Training** | Ayrı aşamalar (SFT -> DPO) yerine tek bir geçişte tüm hedefleri optimize eden eğitim yapısı. |
| **Cross-Entropy Degeneration** | Yalnızca SFT yapıldığında modelin istenmeyen/kalitesiz token kalıplarına da olasılık ataması sorunu. |
| **Negative Log-Likelihood (NLL)** | SFT aşamasında hedef yanıttaki doğru token olasılıklarını maksimize eden kayıp fonksiyonu. |
| **Odds Penalty ($\lambda_{\text{OR}}$)** | Log-Odds cezasının SFT kaybına göre ağırlığını belirleyen ölçek katsayısı (genellikle 0.1 - 1.0). |
| **Zero Reference Architecture** | Dondurulmuş bir referans modeli GPU belleğinde tutmaya ihtiyaç duymayan hafif mimari. |
| **Per-Token Log-Probability** | Dizideki her bir token adımına atanan normalize edilmiş logaritmik olasılık değeri. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Tek aşamalı SFT + Hizalama.        │ • lambda_OR katsayısı yanlış seçilirse│
 │ • Referans model yok -> %50-%75      │   SFT öğrenim kalitesini bozabilir.  │
 │   VRAM tasarrufu (Tek Model!).       │ • Çok uzun yanıtlarda log-odds       │
 │ • Çapraz entropi bozulmasını önler.  │   sayısal hassasiyet gerektirir.     │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Tek GPU'da büyük modelleri (70B)   │ • Aşırı gürültülü tercih çiftlerinde │
 │   doğrudan hizalayabilme imkanı.     │   SFT kaybının da etkilenmesi.       │
 │ • Eğitim süresini yarı yarıya indirme│                                      │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/orpo_alignment_paneli.png` dosyası üretilir:
1. **Monolitik ORPO Kayıp Eğrileri (Total, SFT NLL, Odds Ratio $\mathcal{L}_{\text{OR}}$)**
2. **Çiftli Sıralama Doğruluğu (% Accuracy: %50 -> %100)**
3. **Log-Odds Oranı Gelişimi ($\log \text{OR}(y_w, y_l)$)**
4. **Hizalama Yöntemleri GPU Model Sayısı ve VRAM Kıyası (4 -> 2 -> 1 Model)**
5. **ORPO Matematik ve Formül Kartı**
6. **Stajyer Notu & ORPO Karar Sertifikası**

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
