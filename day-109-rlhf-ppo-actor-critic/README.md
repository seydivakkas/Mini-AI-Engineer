# Day 109: Proximal Policy Optimization (PPO) ile İleri LLM Hizalama

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; ChatGPT ve InstructGPT devriminin kalbinde yer alan **4 Modelli PPO Mimarisi (Actor, Critic, Reference Model, Reward Model)**, **Generalized Advantage Estimation (GAE-$\lambda$)**, **Token Bazlı KL Divergence Cezası** ve **PPO Kırpılmış Amaç Fonksiyonu (Clipped Objective)** mekanizmalarını sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Tiyatro Oyuncusu, Eleştirmen, Hakem ve Hafıza Muhafızı Analojisi"

Bir dil modelini sadece takviyeli öğrenmeyle eğitmek, sahneye çıkan bir oyuncuyu serbest bırakmaya benzer. Eğer kontrol mekanizması olmazsa oyuncu sadece alkış almak (Reward Hacking) için abartılı hareketler yapıp ana metni unutur (Catastrophic Forgetting).

Bu yüzden PPO mimarisinde **4 aktör aynı anda sahnede çalışır:**

1. 🎭 **Aktör (Actor - $\pi_\theta$):** Sahnede doğaçlama replik (token) üreten başrol oyuncusudur. PPO ile sürekli kendini geliştirir.
2. 🧐 **Eleştirmen (Critic / Value Network - $V_\phi$):** Oyuncunun her söylediği cümlenin gelecekte ne kadar alkış toplayacağını anlık tahmin eden ($V(s_t)$) tecrübeli tiyatro eleştirmenidir.
3. 🏆 **Ödül Modeli (Reward Model - $r_\psi$):** Sahne bittiğinde seyircinin ve jürinin verdiği nihai skaler karne notudur ($R_{\text{RM}}$).
4. 🧠 **Hafıza Muhafızı (Reference Model - $\pi_{\text{ref}}$):** Oyuncunun tiyatro okulundan yeni mezun olduğu orijinal dondurulmuş halidir. Aktörün replikleri bu temelden çok uzaklaştığında devreye **KL Cezası (KL Penalty)** sokarak "Karakterden çıkma!" uyarısı yapar.

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & 4-Modelli PPO Mimarisi
LLM PPO döngüsünde 4 model eşzamanlı olarak bellek ve hesaplama paylaşır:
- **Actor ($\pi_\theta$):** $\theta$ parametreleri güncellenir.
- **Critic ($V_\phi$):** $\phi$ parametreleri MSE ile güncellenir.
- **Reference Model ($\pi_{\text{ref}}$):** Dondurulmuş ($\nabla = 0$).
- **Reward Model ($r_\psi$):** Dondurulmuş ($\nabla = 0$).

### 2. Token Bazlı KL Cezası ve Kararlı Hizalama
Modelin ödül modelini kandırarak (Reward Hacking) anlamsız metinler üretmesini engellemek için her token adımında KL cezası verilir:
$$R_t = \begin{cases} -\beta (\log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t})), & t < T \\ -\beta (\log \pi_\theta(y_T) - \log \pi_{\text{ref}}(y_T)) + R_{\text{RM}}(x, y), & t = T \end{cases}$$

### 3. Generalized Advantage Estimation (GAE-$\lambda$)
Zaman adımları boyunca varyansı düşürüp yanlılığı (bias) dengelemek için GAE kullanılır:
$$\delta_t = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
$$\hat{A}_t = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}$$

### 4. PPO Kırpılmış Amaç Fonksiyonu (Clipped Objective)
Politika güncellemesinin aşırı büyük adımlarla çökmesini önlemek için olasılık oranı $r_t(\theta) = \frac{\pi_\theta(y_t)}{\pi_{\text{old}}(y_t)}$ kırpılır:
$$\mathcal{L}_{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t\right) \right]$$

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Actor Network ($\pi_\theta$)** | Metin ve yanıt token'ları üreten ana üretici dil modeli politikası. |
| **Critic Network ($V_\phi$)** | Her token durumunun gelecekteki beklenen kümülatif ödülünü $V(s)$ kestiren değer ağı. |
| **Reference Model ($\pi_{\text{ref}}$)** | SFT sonrası dondurulan ve politikanın dağılımdan kopmasını engelleyen başlangıç LLM'i. |
| **KL Divergence Penalty ($\beta$)** | Aktörün referans modelin token olasılık dağılımından sapmasını engelleyen ceza katsayısı. |
| **GAE ($\text{GAE-}\lambda$)** | Zamansal fark (TD) hatalarını üssel ağırlıklarla birleştirerek varyansı düşüren avantaj kestiricisi. |
| **PPO Ratio ($r_t(\theta)$)** | Yeni politikanın token olasılığının eski politikaya oranı: $\exp(\log \pi_{\text{new}} - \log \pi_{\text{old}})$. |
| **PPO Clipping ($\epsilon$)** | Politika oranını $[1-\epsilon, 1+\epsilon]$ bandında tutarak güvenli adım atan kırpma fonksiyonu. |
| **Rollout Buffer** | Aktörün ortamla (promptlar) etkileşime girip topladığı token, logprob, değer ve ödül havuzu. |
| **Mode Collapse** | LLM'in tek bir kalıp cevaba takılıp kalarak çeşitliliğini tamamen kaybetmesi durumu. |
| **Value Head Loss** | Eleştirmen ağının tahmin ettiği durum değerleri ile gerçekleşen getiriler ($\hat{R}_t$) arasındaki MSE kaybı. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • İnsan tercihlerine en yüksek       │ • 4 modelin aynı anda GPU'da         │
 │   uyum ve akıcı yanıt kalitesi.      │   bulunması yüksek VRAM gerektirir.  │
 │ • GAE ile düşük varyanslı eğitim.    │ • Hiperparametre hassasiyeti (beta,  │
 │ • KL kontrolü ile stabil kalma.      │   lr, clip_eps) oldukça yüksektir.   │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Ajanların araç kullanımı (tool     │ • DPO/GRPO gibi referans modelsiz    │
 │   calling) ve akıl yürütme eğitimi.  │   veya değer ağsız alternatiflerin   │
 │ • Çok turlu diyalog hizalaması.      │   daha az maliyetli olması.          │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/ppo_actor_critic_paneli.png` dosyası üretilir:
1. **PPO Model Ödül Artışı (Mean Reward vs Steps)**
2. **Referans Modelden Sapma (Token Bazlı $\mathbb{D}_{\text{KL}}$ Kontrolü)**
3. **PPO Politika Kırpma Oranı (% Clip Fraction)**
4. **Actor Politika ve Critic Değer Ağı Kayıp Eğrileri**
5. **4-Modelli PPO & GAE-$\lambda$ Matematik Kartı**
6. **Stajyer Notu & PPO RLHF Karar Sertifikası**

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
