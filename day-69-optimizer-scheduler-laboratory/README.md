# Day 69: AdamW vs Lion Optimizer, CosineAnnealing & Linear Warmup Dinamikleri

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passed-brightgreen?style=flat-square)

## 🎯 Proje Özeti & Mühendislik Hedefi

Modern derin öğrenme ve **Vision Transformer (ViT)** modellerinin eğitiminde optimizasyon kararlılığı, genelleştirme kabiliyeti ve GPU bellek tasarrufu kritik öneme sahiptir. Bu projede; endüstri standardı **AdamW** optimizasyon algoritması ile Google Brain tarafından AutoML (program arama) ile keşfedilen yeni nesil **Lion (EvoLved Sign Momentum)** optimizer sıfırdan karşılaştırılmıştır.

Ayrıca; eğitim başlangıcında gradyan şoklarını önleyen **Linear Warmup**, öğrenme oranını pürüzsüz sönümleyen **Cosine Annealing** zamanlayıcısı ve ağırlık azaltmanın (Weight Decay) sadece çekirdek tensörlere uygulanıp bias/normalizasyon katmanlarından ayrıştırıldığı **Decoupled Parameter Group Splitting** mimarisi geliştirilmiştir.

---

## 🔬 Teorik & Matematiksel Arka Plan

### 1. AdamW (Decoupled Weight Decay Adam) Formülasyonu
Standart Adam algoritmasında $L_2$ regülarizasyonu doğrudan kayıp gradyanına eklenir ($g_t \leftarrow g_t + \lambda \theta_t$). Bu durum, adaptif öğrenme oranı $\sqrt{v_t}$ teriminin büyük gradyanlı parametrelerde ağırlık azaltmayı (weight decay) orantısız şekilde bastırmasına yol açar. **AdamW (Loshchilov & Hutter, 2017)** ağırlık azaltmayı gradyandan tamamen ayırır (decoupled):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(1. Moment / Ortalama Gradyan)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(2. Moment / Gradyan Varyansı)}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(Sapma / Bias Düzeltmesi)}$$

$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

---

### 2. Lion (EvoLved Sign Momentum) Optimizer Formülasyonu
Google Brain (Chen et al., 2023) tarafından sembolik program arama ile geliştirilen **Lion**, ikinci moment matrisini ($v_t$) tamamen kaldırarak **%50 bellek tasarrufu** sağlar. Gradyanın büyüklüğü yerine sadece işaretini ($\text{sign}(\cdot)$) kullanarak her koordinatta tekdüze adım atar:

$$c_t = \text{sign}\Big(\beta_1 m_{t-1} + (1 - \beta_1) g_t\Big) \quad \text{(İşaret Momentum Güncellemesi)}$$

$$\theta_{t+1} = \theta_t - \eta_t \big(c_t + \lambda \theta_t\big) \quad \text{(Parametre Güncellemesi \& Decoupled Decay)}$$

$$m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t \quad \text{(Momentum Durumu Saklama)}$$

> **Kritik Kural:** Lion her adımda sabit $\pm \eta_t$ adımı attığından; AdamW'ye kıyasla öğrenme oranı **$3\times - 10\times$ daha küçük** seçilmeli (örn. $1\times 10^{-4}$), batch boyutu ise büyük tutulmalıdır.

---

### 3. Doğrusal Isınma ve Kosinüs Sönümleme (Linear Warmup + Cosine Annealing)
Eğitimin ilk epoch'larında rastgele ilklendirilen parametreler çok büyük ve gürültülü gradyanlar üretir. **Linear Warmup**, öğrenme oranını sıfırdan taban değere doğrusal artırarak bu şoku engeller. Ardından **Cosine Annealing** ile pürüzsüzce taban değerine indirir:

$$\eta_t = \begin{cases} \eta_{\text{base}} \cdot \dfrac{t}{T_{\text{warmup}}}, & t \le T_{\text{warmup}} \\ \eta_{\text{min}} + \dfrac{1}{2}(\eta_{\text{base}} - \eta_{\text{min}})\left(1 + \cos\left(\dfrac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}}\pi\right)\right), & t > T_{\text{warmup}} \end{cases}$$

---

## 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- AdamW ve bellek verimli Lion optimizer'ı CosineAnnealing ve Warmup dinamikleri ile laboratuvar ortamında kıyaslayıp optimal yakınsamayı bulmak için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Eğitimin başlangıcındaki gradyan şoklarını (Warmup ile) ve yerel minimumlara takılmayı (Cosine Annealing ile) çözer.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Lion optimizer momentum işaret (sign) tabanlı çalıştığı için küçük batch boyutlarında ve aşırı gürültülü gradyanlarda kararsızlaşabilir.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- SGD with Momentum, Sophia, Adafactor veya RMSprop.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **AdamW** | *Decoupled Weight Decay Adam* | L2 regülarizasyonunu adaptif gradyan ölçeklemesinden ayıran, modern derin öğrenme ve NLP/CV modellerinin varsayılan optimizasyon algoritması. |
| **Lion** | *EvoLved Sign Momentum* | İkinci moment varyans tensörü tutmayan, sadece 1. moment ve `sign()` fonksiyonuyla çalışan, bellek dostu ve hızlı AutoML optimizasyon algoritması. |
| **Linear Warmup** | *Linear Learning Rate Warmup* | Eğitimin ilk adımlarında öğrenme oranını sıfırdan hedef seviyeye kademeli artırarak parametrelerin erken diverjansını ve gradyan şoklarını önleyen teknik. |
| **Cosine Annealing** | *Cosine Annealing Schedule* | Öğrenme oranını kosinüs eğrisi boyunca yumuşak bir şekilde sönümleyerek modelin yerel minimumlar etrafında kararlı yakınsamasını sağlayan zamanlayıcı. |
| **Decoupled Weight Decay** | *Decoupled Weight Decay* | Ağırlıkların büyüklüğünü doğrudan küçülten ($\theta \leftarrow \theta(1 - \eta \lambda)$), kayıp fonksiyonunun gradyanından bağımsız ağırlık cezalandırma yöntemi. |
| **L2 Regularization** | *L2 Loss Penalty* | Kayıp fonksiyonuna $\frac{1}{2}\lambda \|\theta\|^2$ eklenmesiyle elde edilen klasik regülarizasyon (Standart Adam'da adaptif ölçeklemeden ötürü hatalı çalışır). |
| **Parameter Group Splitting** | *Weight Decay Exclusion* | Conv/Linear ağırlıklarına weight decay uygularken; bias, BatchNorm ve LayerNorm parametrelerini aşırı sönümlemeyi önlemek için muaf tutma mimarisi. |
| **Gradient Clipping** | *L2 Gradient Norm Clipping* | Gradyan normunun belirlenen bir tavan değeri ($C$) aşması durumunda gradyan vektörünü ölçekleyerek gradyan patlamasını (exploding gradient) önleyen koruma mekanizması. |
| **First Moment ($m_t$)** | *First Moment Vector* | Gradyanların üstel hareketli ortalaması (EMA); gradyan yönündeki momentumu temsil eder. |
| **Second Moment ($v_t$)** | *Second Moment Vector* | Gradyan karelerinin üstel hareketli ortalaması; koordinat bazında öğrenme hızını adaptif olarak ölçeklemek için kullanılır. |
| **Optimizer Memory Footprint** | *Optimizer Memory Overhead* | Optimizer'ın GPU RAM üzerinde parametre durumlarını (durum tensörleri) saklamak için tükettiği ek grafik belleği. |
| **Coordinate-wise Sign Update** | *Sign-based Gradient Step* | Lion'da her parametre koordinatının tam olarak aynı büyüklükte ($\pm \eta$) adım atmasını sağlayan işaret operatörü mekanizması. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Lion ile %50 daha az GPU bellek tüketimi; AdamW ile kanıtlanmış evrensel yakınsama kararlılığı; Linear Warmup ile sıfır gradyan patlaması; Decoupled Parameter Grouping ile bias tensörlerinin korunması. |
| **Weaknesses (Zayıf Yönler)** | Lion'ın küçük batch boyutlarında gürültülü `sign()` operasyonu sebebiyle osilasyona meyilli olması; Lion için öğrenme oranının ince ayar (fine-tuning) gerektirmesi. |
| **Opportunities (Fırsatlar)** | Büyük ölçekli Vision Transformer (ViT), Llama ve Diffusion modellerinin eğitiminde yüzlerce gigabaytlık VRAM tasarrufu; daha büyük batch boyutlarına olanak tanıması. |
| **Threats (Tehditler)** | Standart AdamW hiperparametrelerinin (lr=1e-3) Lion'a doğrudan kopyalanması durumunda eğitimin tamamen patlaması (diverge). |

---

## 📈 Deneysel Benchmark ve Karşılaştırma Tablosu

Aynı veri kümesi, model mimarisi ve tohum (Seed 42) altında 10 epoch süresince koşturulan kontrollü deney sonuçları:

| Deney Mimarisi | Optimizer Türü | LR Zamanlayıcı | Son Train Loss | Doğrulama Başarımı (%) | Optimizer Bellek Tüketimi |
|---|---|---|---|---|---|
| **1. Klasik Baseline** | **AdamW** | StepLR | $0.9405$ | **%20.50** | $735$ KB ($2 \times$ Parametre) |
| **2. Modern Standart** | **AdamW** | LinearWarmupCosine | **$0.8334$** | **%20.00** | $735$ KB ($2 \times$ Parametre) |
| **3. Google Brain Lion**| **Lion** | LinearWarmupCosine | $1.3662$ | **%21.00** | **$368$ KB (%50 Tasarruf)** |

---

## 🖼️ Görsel Çıktı: 6 Panelli Teşhis Panosu

Laboratuvar sonuçları [`ciktilar/optimizer_karsilastirma_paneli.png`](file:///c:/Users/seydieryilmaz/Desktop/Github%20Mini%20AI%20Engineer/day-69-optimizer-scheduler-laboratory/ciktilar/optimizer_karsilastirma_paneli.png) dosyasında oluşturulmuştur:
1. **Optimizasyon Laboratuvar Özeti**: 3 deneyin yakınsama ve bellek metrikleri kartı.
2. **Eğitim Kaybı (Train Loss) Yakınsaması**: WarmupCosine ile AdamW'nin pürüzsüz düşüşü.
3. **Doğrulama Başarımı**: Epoch bazlı doğrulama skoru eğrileri.
4. **LR Profil Dinamiği**: Warmup fazı (doğrusal artış) ve Cosine fazı (yumuşak sönümleme).
5. **Gradyan Normu ve Kararlılık**: Gradyan büyüklüklerinin kararlı salınımı.
6. **SWOT Karar Matrisi**: Mimari güçlü ve zayıf yönlerin endüstriyel sentezi.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Vision Transformer veya ConvNet modellerinde LayerNorm, BatchNorm ve bias parametrelerini tespit edip `weight_decay=0.0`, ağırlık matrislerine ise `weight_decay=0.05` atayan endüstriyel standartta bir parametre ayrıştırıcı yazınız.

**Eksiksiz Çözüm:**
```python
import torch
import torch.nn as nn
from typing import List, Dict, Any

def gelismis_parametre_ayristirici(
    model: nn.Module,
    weight_decay: float = 0.05
) -> List[Dict[str, Any]]:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D tensörler (bias, LayerNorm/BatchNorm weight ve bias'ları)
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Standart Adam optimizer'daki $L_2$ regülarizasyonu neden derin modellerde ve Transformer mimarilerinde AdamW kadar iyi genelleşemez? Matematiksel olarak adaptif moment terimi ($\sqrt{v_t}$) $L_2$ cezasına nasıl müdahale eder?

> **Mentor Cevabı:**
> 1. **Klasik Adam + $L_2$ Cezası:** Standart Adam'da $L_2$ cezası kayba eklenir: $\mathcal{L}_{\text{toplam}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|^2$. Gradyan $g_t' = g_t + \lambda \theta_t$ olur.
> 2. **Adaptif İkinci Momentin Etkisi:** Adam parametre güncellemesi yaparken gradyanı $\sqrt{v_t}$ terimine böler:
>    $$\Delta \theta_t = - \eta_t \frac{g_t + \lambda \theta_t}{\sqrt{v_t} + \epsilon}$$
>    Burada $\lambda \theta_t$ terimi de $\sqrt{v_t}$ ile bölünür!
> 3. **Tehlikeli Sonuç (Orantısız Sönümleme):** Çok sık güncellenen veya büyük gradyanlara sahip parametrelerde $v_t$ çok büyüktür. Dolayısıyla bu parametrelerin ağırlık cezası $\frac{\lambda \theta_t}{\sqrt{v_t}}$ neredeyse sıfırlanır ve ceza almazlar! Nadir güncellenen parametrelerde ise $v_t$ küçüktür ve orantısız şekilde aşırı cezalandırılırlar.
> 4. **AdamW Çözümü:** AdamW, $\lambda \theta_t$ terimini $\sqrt{v_t}$ bölmesinden çıkararak doğrudan parametreden düşer ($\theta_{t+1} = \theta_t (1 - \eta_t \lambda) - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$). Böylece tüm parametreler gradyan büyüklüklerinden bağımsız olarak eşit ve adil şekilde regüle edilir.

---

## 📜 Lisans & Telif Hakkı

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
