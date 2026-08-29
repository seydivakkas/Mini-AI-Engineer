# Day 113: Simple Preference Optimization (SimPO) ile Hafif LLM Hizalaması

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; DPO'nun dondurulmuş referans model ($\pi_{\text{ref}}$) bağımlılığını ve çıkarım (inference) uyumsuzluğunu tamamen ortadan kaldıran, doğrudan **Ortalama Token Log-Olasılıkları ve Hedef Marjin ($\gamma$)** ile çalışan **Simple Preference Optimization (SimPO - NeurIPS 2024)** algoritmasını sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Hedef Marjin ve Referanssız Hafif Tercih Devrimi"

DPO algoritmasını öğrenirken "Ne güzel, Reward Modeline gerek kalmadı!" demiştik. Ancak DPO'nun da can sıkıcı iki problemi vardı:
1. **İkinci Bir Model (Referans Model):** GPU belleğinde dondurulmuş bir kopya model ($\pi_{\text{ref}}$) tutmak zorundaydık. Bu da VRAM ihtiyacını 2 katına çıkarıyordu.
2. **Çıkarım (Inference) Çelişkisi:** DPO modeli $\frac{P_\theta(y)}{P_{\text{ref}}(y)}$ oranını artırmaya çalışır. Fakat modeli gerçek hayatta çalıştırırken (Greedy decoding, sampling) sadece $P_\theta(y)$ olasılığına bakarız, referans modeli sormayız! Bu durum eğitim hedefi ile gerçek kullanım arasında bir kopukluk yaratır.

**SimPO (Simple Preference Optimization)** bu karmaşayı çözen zekice bir sadelik sunar:
- **Ödülü Doğrudan Tanımla:** Ödül, modelin o cümleye verdiği **ortalama token log-olasılığıdır**:
  $$r(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y \mid x)$$
- **Hedef Marjin ($\gamma$):** İyi cevabın ödülü ($r_w$), kötü cevabın ödülünden ($r_l$) sadece büyük olmakla kalmamalı, aralarında en az $\gamma$ kadar bir güvenlik marjini olmalıdır ($r_w - r_l > \gamma$).

```
       DPO (REFERANS MODEL ORANI)                                   SimPO (DOĞRUDAN ORTALAMA LOGP + MARJİN)
 ┌──────────────────────────────────────────────┐       ┌──────────────────────────────────────────────┐
 │ • Ödül: beta * log( pi_theta / pi_ref )      │       │ • Ödül: beta / |y| * log pi_theta(y|x)       │
 │ • Referans Model: ZORUNLU (2 Model GPU'da)   │       │ • Referans Model: YOK (1 Model GPU'da!)      │
 │ • Çıkarım Uyumu: UYUMSUZ (Oran vs Doğrudan)  │       │ • Çıkarım Uyumu: MÜKEMMEL (Doğrudan logp)    │
 │ • Uzunluk Yanlılığı: Var (Uzun cevap seçer)  │       │ • Uzunluk Yanlılığı: SIFIR (1/|y| ile norm)  │
 └──────────────────────────────────────────────┘       └──────────────────────────────────────────────┘
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & DPO'nun Çıkarım Çelişkisi (Inference Discrepancy)
DPO'da model optimize edilirken $\pi_\theta / \pi_{\text{ref}}$ oranı maksimize edilir. Ancak gerçek çıkarımda $\pi_{\text{ref}}$ kullanılmaz; model doğrudan $\pi_\theta$'ya göre token üretir. SimPO ödülü doğrudan ortalama token olasılığı olarak modellediği için eğitim hedefi ile çıkarım davranışı %100 örtüşür.

### 2. Uzunluk Normalizasyonu ($\frac{1}{|y|}$) ile Uzunluk Yanlılığı İptali
Ham log-olasılık toplamları uzun dizilerde negatif olarak birikir. DPO bazen uzun ve gereksiz laf kalabalığı yapan yanıtları tercih edebilir. SimPO ortalama token log-olasılığını ($\frac{1}{|y|} \sum \log \pi$) kullanarak uzunluk manipülasyonunu tamamen engeller.

### 3. Hedef Ödül Marjini ($\gamma$) Dinamiği
SimPO kayıp fonksiyonu:
$$\mathcal{L}_{\text{SimPO}}(\theta) = - \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right)$$

Burada $\gamma > 0$ (genellikle $0.5 - 1.4$), tercih edilen yanıtın reddedilen yanıta göre belirgin ve açık bir üstünlük sağlamasını zorunlu kılar.

### 4. Endüstriyel Liderlik (AlpacaEval 2.0, Arena-Hard, Hugging Face TRL)
- **AlpacaEval 2.0 & Arena-Hard:** SimPO, DPO ve PPO'yu geride bırakarak açık kaynak modellerde (Llama-3-8B-Instruct tabanlı SimPO) en yüksek Win-Rate skorlarına ulaşmıştır.
- **Hugging Face TRL:** `CPOTrainer` ve doğrudan SimPO modülleriyle endüstri standardı haline gelmiştir.

---

## 📊 4'lü Hizalama Yöntemi Kıyaslama Tablosu

| Kriter | PPO (RLHF) | DPO | ORPO | SimPO |
|:---|:---|:---|:---|:---|
| **GPU'da Eşzamanlı Model** | 4 Model (Actor, Critic, Ref, RM) | 2 Model (Policy, Ref) | 1 Model (Policy) | **1 Model (Policy)** |
| **Referans Model ($\pi_{\text{ref}}$)** | Zorunlu | Zorunlu | Yok | **Yok** |
| **Uzunluk Normalizasyonu** | Yok | Opsiyonel | Var | **Doğal Zorunlu ($1/\|y\|$)** |
| **Hedef Marjin ($\gamma$)** | Yok | Yok | Yok | **Var ($\gamma > 0$)** |
| **Çıkarım Uyumu (Inference)** | Orta | Uyumsuz | Uyumlu | **Mükemmel Uyumlu** |
| **VRAM Kullanımı** | %100 | %50 | %25 | **%25 (Ultra Hafif)** |
| **Benchmark Performansı** | Standart | Yüksek | Çok Yüksek | **Lider (SOTA)** |

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **SimPO** | Referans modelsiz, uzunluk normalize edilmiş ve hedef marjinli doğrudan tercih optimizasyonu. |
| **Target Margin ($\gamma$)** | Tercih edilen ve reddedilen yanıt ödülleri arasında hedeflenen minimum pozitif fark. |
| **Implicit Reward ($r$)** | SimPO'da doğrudan modelin ortalama token log-olasılığı olarak tanımlanan skaler ödül. |
| **Length Normalization** | Dizi uzunluğuna bölerek uzunluk yanlılığını ve gevezeliği (verbosity) engelleyen normalizasyon. |
| **Inference Discrepancy** | Eğitimde optimize edilen metrik ile test/çıkarım anında kullanılan metrik arasındaki uyumsuzluk. |
| **Reward Scaling ($\beta$)** | Ortalama log-olasılığı skaler ödüle dönüştüren eğim/sıcaklık parametresi (genellikle 2.0 - 2.5). |
| **Margin Violation Rate** | $\Delta r = r_w - r_l < \gamma$ şartını sağlayamayan örneklerin yüzdesi. |
| **Log-Sigmoid Loss** | Marjin farkını olasılık dağılımına dönüştürerek gradyan akışı sağlayan negatif log-sigmoid kaybı. |
| **Zero Reference Model** | Dondurulmuş bir referans modeli GPU belleğinde tutmaya ihtiyaç duymayan mimari. |
| **Win Rate** | AlpacaEval veya Arena-Hard gibi benchmark testlerinde rakip modellere karşı kazanma oranı. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Referans model yok -> %50 VRAM     │ • gamma ve beta hiperparametreleri   │
 │   tasarrufu ve 2 kat hızlı eğitim.   │   veri setine göre ayar gerektirir.  │
 │ • Uzunluk yanlılığını tamamen keser. │ • SFT yapılmamış baz modellerde tek  │
 │ • Çıkarım olasılığıyla %100 uyumlu.  │   başına dil yeteneği kazandırmaz.   │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • AlpacaEval 2.0 ve Arena-Hard       │ • Aşırı yüksek gamma değeri gradyan  │
 │   benchmarklarında SOTA sonuçlar.    │   doygunluğuna (saturation) yol açar.│
 │ • 70B+ modelleri tek sunucuda eğitme.│                                      │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/simpo_alignment_paneli.png` dosyası üretilir:
1. **SimPO Kayıp Eğrisi ($\mathcal{L}_{\text{SimPO}} = -\log \sigma(\Delta r - \gamma)$)**
2. **Örtük Ödüllerin Ayrışması ($r_w$ vs $r_l$)**
3. **Ödül Marjini ($\Delta r$) ve Hedef Marjin ($\gamma=0.5$) Aşımı**
4. **Doğruluk (%) ve Marjin İhlal Oranı (%)**
5. **PPO vs DPO vs ORPO vs SimPO Kıyas Matrisi**
6. **SimPO Matematik Kartı & Karar Sertifikası**

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
