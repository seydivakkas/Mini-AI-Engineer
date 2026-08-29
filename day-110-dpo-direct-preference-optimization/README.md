# Day 110: Direct Preference Optimization (DPO) ile Kapalı Form LLM Hizalaması

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; modern açık kaynak LLM dünyasını (Llama 3, Mistral, Zephyr, Gemma) fetheden, takviyeli öğrenmeyi (RLHF) doğrudan Supervised Fine-Tuning basitliğine indirgeyen **Direct Preference Optimization (DPO)** algoritmasını, **Örtük Ödül (Implicit Reward)** formülasyonunu ve **DPO vs PPO Kıyaslama Laboratuvarını** sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Çift Hatlı Terazi ve Ödül Modelsiz Hizalama Analojisi"

Geleneksel PPO (RLHF) yöntemi, bir şefe yemek pişirtip ardından ayrı bir gurmeye (Reward Model) ve restorandaki bir denetçiye (Critic) sormaya benzer. Bu süreçte mutfakta 4 kişi çalışır, masraflar ikiye katlanır ve gurme ile şef arasında anlaşmazlıklar (Reward Hacking) çıkar!

**DPO (Direct Preference Optimization)** ise şefe iki tabağı yan yana gösterir:
- "Bu kazanan tabak ($y_w$), bu da reddedilen tabak ($y_l$)."
- DPO, şefin tarifindeki (log-olasılık) kazanan bileşenlerin oranını artırırken kaybeden bileşenlerin oranını doğrudan düşürür!

1. ⚖️ **Ayrı Ödül Modeli Yok:** Ödül, modelin kendi olasılıkları içinde **örtük (implicit)** olarak zaten gizlidir:
   $$\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$
2. ⚡ **%50 Daha Az Bellek:** Critic ağı ve Reward Modeli tamamen silinir. GPU'da sadece eğitilen model ($\pi_\theta$) ve dondurulmuş referans ($\pi_{\text{ref}}$) kalır.
3. 🎯 **Süpervizeli Eğitim Kararlılığı:** Rollout/Örnekleme aşaması olmadığı için eğitim tıpkı SFT gibi deterministik, kararlı ve ultra hızlıdır.

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & DPO Matematiksel Türetimi
Takviyeli öğrenmedeki optimal politika kapalı formda:
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Buradan ödül $r(x, y)$ çekilip Bradley-Terry olasılığına yazıldığında $Z(x)$ paydaları birbirini götürür ve DPO kaybı elde edilir:
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### 2. Örtük Ödül (Implicit Reward) ve Marjin Ayrışması
DPO'da modelin cevaba verdiği örtük ödül:
$$\hat{r}_\theta(x, y) = \beta (\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x))$$
Model öğrendikçe $\hat{r}_\theta(x, y_w) > 0$ ve $\hat{r}_\theta(x, y_l) < 0$ olarak ayrışır. Marjin $\Delta \hat{r} = \hat{r}_w - \hat{r}_l$ hızla büyüyerek modelin tercih doğruluğunu **%98+** seviyesine taşır.

### 3. DPO vs PPO Mimari ve Hesaplama Kıyaslaması
| Özellik | PPO (RLHF) | DPO (Direct Alignment) |
|:---|:---|:---|
| **Gerekli Model Sayısı** | 4 (Actor, Critic, Ref, RM) | 2 (Policy, Ref) |
| **VRAM Tüketimi** | Çok Yüksek (100%) | Düşük (%50 Tasarruf) |
| **Örnekleme (Rollout)** | Zorunlu ($O(N \cdot T)$ maliyet) | Yok (Off-policy Dataset) |
| **Eğitim Kararlılığı** | Yüksek Hiperparametre Hassasiyeti | Kararlı SFT Benzeri Kayıp |

### 4. Endüstriyel Entegrasyon (Llama-3, Mistral, Zephyr, HuggingFace TRL)
- **Hugging Face TRL (`DPOTrainer`):** Açık kaynak topluluğunun birincil tercih hizalama standardı oldu.
- **Zephyr-7B & Mistral-7B-Instruct:** DPO kullanarak 70B modelleri MT-Bench ve AlpacaEval'de geride bırakan ilk küçük modeller oldu.
- **Meta Llama 3:** Hizalama hattında SFT sonrası çok adımlı DPO + PPO hibrit mimarisi kullandı.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Direct Preference Optimization (DPO)** | Takviyeli öğrenme ödül modellemesini doğrudan dil modelinin log-olasılık kaybına indirgeyen kapalı form hizalama algoritması. |
| **Implicit Reward ($\hat{r}$)** | DPO tarafından optimize edilen modelin referans modele göre log-olasılık farkı: $\beta(\log \pi_\theta - \log \pi_{\text{ref}})$. |
| **Log-Ratio** | Kazanan veya kaybeden yanıtın politika/referans log-olasılıkları oranı: $\log(\pi_\theta(y) / \pi_{\text{ref}}(y))$. |
| **Reference Model ($\pi_{\text{ref}}$)** | Politikanın dilden kopmasını engelleyen dondurulmuş SFT temel modeli. |
| **Temperature / Scaling Parameter ($\beta$)** | Tercih kaybının referans modelden sapmaya karşı duyarlılığını belirleyen katsayı (genellikle 0.05 - 0.2). |
| **Chosen Response ($y_w$)** | Verilen prompt için insan veya jüri tarafından tercih edilen kaliteli kazanmış yanıt. |
| **Rejected Response ($y_l$)** | Verilen prompt için elenen, kalitesiz veya hatalı bulunmuş yanıt. |
| **Length Normalization** | Uzun yanıtlara yönelik yapay log-olasılık düşüşünü engellemek için toplam log-olasılığı token sayısına bölme işlemi. |
| **Label Smoothing** | DPO çıktısındaki aşırı özgüveni ve logit patlamalarını engellemek için eklenen düzenlileştirici. |
| **Implicit Error Weight** | Modelin $y_l$'yi $y_w$'ye tercih ettiği durumlarda gradyanı büyüten $\sigma(\hat{r}_l - \hat{r}_w)$ çarpanı. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Reward Model ve Critic ağı         │ • Veri setindeki gürültü ve hatalı   │
 │   eğitme zorunluluğunu kaldırır.     │   etiketlere karşı duyarlıdır.       │
 │ • %50 VRAM tasarrufu, yüksek hız.    │ • Uzun yanıtları yapay olarak        │
 │ • SFT gibi kararlı yakınsama.        │   kayırma (length bias) eğilimi.     │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Sentetik veri setleriyle (Ultra-   │ • Dağılım dışı (OOD) yanıtlarda      │
 │   Feedback) hızlı model iterasyonu.  │   PPO kadar keşif (exploration)      │
 │ • LoRA/QLoRA ile tek GPU'da eğitim.  │   yapamaması.                        │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/dpo_alignment_paneli.png` dosyası üretilir:
1. **DPO Log-Oranı Kayıp Eğrisi (Epoklar Boyunca Düşüş)**
2. **Çiftli Sıralama Doğruluğu (% Pairwise Accuracy: %50 -> %98+)**
3. **Örtük Ödüllerin ve Marjinin Ayrışması ($\hat{r}_w$ vs $\hat{r}_l$ & $\Delta \hat{r}$)**
4. **DPO vs PPO Model Sayısı ve VRAM Tasarrufu (%50 Kazanç)**
5. **DPO Kapalı Form ve Örtük Ödül Matematik Kartı**
6. **Stajyer Notu & DPO Karar Sertifikası**

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
