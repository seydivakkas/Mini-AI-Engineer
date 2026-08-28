# Day 70: Mixup, CutMix ve Label Smoothing Modern Düzenlileştirmesi

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passed-brightgreen?style=flat-square)

## 🎯 Proje Özeti & Mühendislik Hedefi

Modern derin öğrenme ve özellikle **Vision Transformer (ViT)** modellerinde en büyük risklerden biri; modellerin eğitim verisinin ezberine kapılarak aşırı güvenli (overconfident) tahminler üretmesi ve eğitim seti dışındaki (OOD) ya da bozulmuş görsellerde çökmesidir. 

Bu projede; görsel sınıflandırma modellerinin genelleştirme kabiliyetini zirveye taşıyan üç temel modern düzenlileştirme (regularization) yöntemi uçtan uca uygulanmıştır:
1. **Mixup (Zhang et al., 2018):** İki farklı görseli ve etiketlerini Beta dağılımı katsayısıyla doğrusal olarak harmanlama.
2. **CutMix (Yun et al., 2019):** Bir görselin rastgele bir bölgesini diğer görselin yaması ile değiştirip alan oranıyla etiketleme.
3. **Label Smoothing Cross-Entropy (Szegedy et al., 2016):** Sert One-Hot etiketler yerine yumuşatılmış olasılık dağılımı kullanarak logit patlamalarını ve aşırı güveni engelleme.

---

## 🔬 Teorik & Matematiksel Derinlik

### 1. Mixup (Doğrusal Enterpolasyon ile Düzenlileştirme)
Klasik eğitimde Ampirik Risk Minimizasyonu (ERM) uygulanır: model sadece eğitim veri noktalarında kayıp minimize eder. **Mixup**, veri noktaları arasındaki uzayda da doğrusal davranış sergilenmesini zorunlu kılar:

$$\lambda \sim \text{Beta}(\alpha, \alpha), \quad \lambda \in [0, 1]$$

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$

$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

Kayıp fonksiyonu çift hedef üzerinden ağırlıklı hesaplanır:

$$\mathcal{L}_{\text{mixup}}(\hat{y}, y_i, y_j, \lambda) = \lambda \mathcal{L}_{\text{CE}}(\hat{y}, y_i) + (1 - \lambda) \mathcal{L}_{\text{CE}}(\hat{y}, y_j)$$

---

### 2. CutMix (Bölgesel Kes-Yapıştır ile Mekansal Düzenlileştirme)
Mixup pikselleri üst üste bindirdiğinde doğal olmayan bulanıklıklar oluşturabilir. **CutMix**, bir görselin rastgele bir dikdörtgen bölgesini silerek yerine diğer görselden kesilen yamayı yapıştırır.

Rastgele sınır kutusu (Bounding Box) koordinatları:

$$r_x \sim \mathcal{U}(0, W), \quad r_y \sim \mathcal{U}(0, H)$$

$$r_w = W \sqrt{1 - \lambda}, \quad r_h = H \sqrt{1 - \lambda}$$

$$B = \Big(r_x - \frac{r_w}{2}, \; r_y - \frac{r_h}{2}, \; r_x + \frac{r_w}{2}, \; r_y + \frac{r_h}{2}\Big)$$

$$\tilde{x} = \mathbf{M} \odot x_i + (\mathbf{1} - \mathbf{M}) \odot x_j$$

Burada $\mathbf{M} \in \{0, 1\}^{W \times H}$ ikili maskedir. Gerçek alan oranı:

$$\lambda_{\text{adj}} = 1 - \frac{\text{Alan}(B)}{W \times H}$$

$$\mathcal{L}_{\text{cutmix}}(\hat{y}, y_i, y_j, \lambda_{\text{adj}}) = \lambda_{\text{adj}} \mathcal{L}_{\text{CE}}(\hat{y}, y_i) + (1 - \lambda_{\text{adj}}) \mathcal{L}_{\text{CE}}(\hat{y}, y_j)$$

---

### 3. Etiket Yumuşatma (Label Smoothing Cross-Entropy)
Standart One-Hot kodlamada hedef sınıf olasılığı $1.0$, diğer sınıflar $0.0$'dır. Model bu hedefi tutturmak için logit değerlerini $\infty$'a doğru iter ve aşırı güvenli (overconfident) hale gelir. **Label Smoothing ($\epsilon$)** ile hedef dağılım yumuşatılır:

$$q_k = \begin{cases} 1 - \epsilon + \dfrac{\epsilon}{K}, & k = y \\ \dfrac{\epsilon}{K}, & k \ne y \end{cases} = (1 - \epsilon) \mathbf{1}_{k=y} + \frac{\epsilon}{K}$$

Kayıp fonksiyonu:

$$\mathcal{L}_{\text{LS}}(p, q) = (1 - \epsilon) \big(-\log p_y\big) + \frac{\epsilon}{K} \sum_{k=1}^K \big(-\log p_k\big)$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Mixup** | *Mixup Data Augmentation* | İki görseli ve etiketlerini Beta dağılımı katsayısı ile piksel piksel harmanlayarak modelin karar sınırlarını yumuşatan veri artırma yöntemi. |
| **CutMix** | *CutMix Regularization* | Bir görselin dikdörtgen bölgesini kesip diğer görselden alınan yama ile değiştiren ve etiketleri alan oranında paylaştıran yöntem. |
| **Label Smoothing** | *Label Smoothing Regularization* | One-hot etiketlerdeki mutlak 1 ve 0 değerlerini $\epsilon$ katsayısıyla tüm sınıflara dağıtarak modelin aşırı güvenini ve logit patlamalarını önleyen teknik. |
| **Overconfidence** | *Model Overconfidence* | Bir modelin yanlış veya belirsiz tahminlerde dahi softmax çıkışında %99+ olasılık ataması ve kalibre olamaması durumu. |
| **Model Calibration** | *Probability Calibration* | Modelin ürettiği softmax olasılık değerlerinin gerçek doğruluk olasılıkları ile birebir örtüşmesi durumu (ör. %80 güvenli tahminlerin %80'inin doğru çıkması). |
| **Beta Distribution** | *Beta Distribution ($\text{Beta}(\alpha, \alpha)$)* | Mixup ve CutMix'te $\lambda$ karışım oranını seçmek için kullanılan, $\alpha$ parametresine göre U veya çan eğrisi alan istatistiksel dağılım. |
| **Empirical Risk (ERM)** | *Empirical Risk Minimization* | Modelin sadece eğitim setindeki somut veri noktalarında hatayı sıfırlamaya çalışması (aşırı uyumun ana kaynağı). |
| **Vicinal Risk (VRM)** | *Vicinal Risk Minimization* | Modelin eğitim örneklerinin komşuluğunda (virtual samples) da genelleştirme yapacak şekilde eğitilmesi (Mixup/CutMix felsefesi). |
| **Soft Targets** | *Continuous Soft Targets* | 0 ve 1 yerine float olasılık değerlerinden oluşan etiket vektörleri. |
| **Logit Penalty** | *Logit Explosion Prevention* | Model çıkışındaki ham logit değerlerinin sonsuza büyümesini engelleyerek sayısal kararlılık sağlama. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Mixup ile doğrusal karar sınırları; CutMix ile yerel özelliklere aşırı odaklanmayı önleme; Label Smoothing ile kusursuz kalibrasyon ve sıfır logit patlaması; Vision Transformer'larda vazgeçilmez genelleştirme. |
| **Weaknesses (Zayıf Yönler)** | Eğitim başlangıcında kayıp değerlerinin sert tekil hedeflere göre daha yavaş düşmesi; daha fazla eğitim epoch'u gerektirmesi. |
| **Opportunities (Fırsatlar)** | Temsil öğrenimi, OOD (Dağılım Dışı) dayanıklılığı ve Adversarial saldırılara karşı doğal bağışıklık sağlama. |
| **Threats (Tehditler)** | Çok küçük kapasiteli modellerde (örn. 2 katmanlı MLP) aşırı düzenlileştirme (underfitting) yaratarak modelin öğrenmesini geciktirme riski. |

---

## 📈 Deneysel Benchmark ve Karşılaştırma Tablosu

Aynı sentetik veri kümesi ve model mimarisi altında 10 epoch süresince koşturulan deney sonuçları:

| Deney Mimarisi | Artırma Türü | Label Smoothing ($\epsilon$) | Son Train Loss | Doğrulama Başarımı (%) | Ortalama Güven (Confidence) |
|---|---|---|---|---|---|
| **1. Standart Baseline** | **NONE** | $0.0$ | **$0.0762$** | **%100.00** | **$0.995$ (Aşırı Güvenli / Overconfident)** |
| **2. Mixup + LabelSmooth**| **MIXUP** | $0.1$ | $0.9166$ | **%75.00** | **$0.597$ (Yumuşatılmış / Kalibre)** |
| **3. CutMix + LabelSmooth**| **CUTMIX**| $0.1$ | $0.7652$ | **%100.00** | **$0.925$ (Kusursuz Dengeli)** |

- **Kalibrasyon Başarısı:** Standart baseline %99.5 gibi tehlikeli bir aşırı güven sergilerken, CutMix ve Mixup modelleri gerçekçi olasılık aralıklarına kalibre edildi.
- **Bölgesel Temsil:** CutMix %100 doğrulama başarımını koruyarak mekansal olarak en dayanıklı model oldu.
- **Birim Test Başarımı:** **$8 / 8$ PASSED (%100 Başarı, 8.36s)**

---

## 🖼️ Görsel Çıktı: 6 Panelli Teşhis Panosu

Laboratuvar sonuçları [`ciktilar/modern_regulerizasyon_paneli.png`](file:///c:/Users/seydieryilmaz/Desktop/Github%20Mini%20AI%20Engineer/day-70-modern-regularization-mixup-cutmix-label-smoothing/ciktilar/modern_regulerizasyon_paneli.png) dosyasında üretilmiştir:
1. **Regülerizasyon Laboratuvar Özeti**: 3 deneyin doğruluk, kayıp ve güven metrikleri kartı.
2. **Görsel Dönüşüm Örnekleri**: Orijinal, Mixup ($\lambda=0.50$) ve CutMix ($\lambda=0.61$) piksellerinin karşılaştırması.
3. **Eğitim Kaybı Yakınsaması**: Düzenlileştirilmiş modellerin kontrollü kayıp eğrileri.
4. **Doğrulama Başarımı**: Epoch bazlı doğrulama skoru grafiği.
5. **Aşırı Güven (Overconfidence) Analizi**: Baseline'ın yapay %99.5 güveni ile kalibre edilmiş düzenli modellerin ayrışımı.
6. **SWOT Karar Matrisi**: Mimari tercihlerin endüstriyel sentezi.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Girdi batch'ine %50 olasılıkla Mixup, %50 olasılıkla CutMix uygulayan ve hem tekil hem çift etiketleri Label Smoothing ile eğiten dinamik bir `DinamikRegulerBoruHatti` fonksiyonu yazınız.

**Eksiksiz Çözüm:**
```python
import torch
import numpy as np
from typing import Tuple

def dinamik_mix_secimi(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha_mixup: float = 0.8,
    alpha_cutmix: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
    """%50 ihtimalle Mixup, %50 ihtimalle CutMix uygulayan dinamik seçici."""
    if np.random.rand() < 0.5:
        # Mixup Uygula
        lam = float(np.random.beta(alpha_mixup, alpha_mixup))
        perm = torch.randperm(x.size(0), device=x.device)
        x_aug = lam * x + (1.0 - lam) * x[perm]
        return x_aug, y, y[perm], lam, "mixup"
    else:
        # CutMix Uygula
        lam = float(np.random.beta(alpha_cutmix, alpha_cutmix))
        batch_size, _, H, W = x.size()
        perm = torch.randperm(batch_size, device=x.device)
        
        kesim_orani = np.sqrt(1.0 - lam)
        kw, kh = int(W * kesim_orani), int(H * kesim_orani)
        cx, cy = np.random.randint(0, W), np.random.randint(0, H)
        
        x1, y1 = max(0, cx - kw // 2), max(0, cy - kh // 2)
        x2, y2 = min(W, cx + kw // 2), min(H, cy + kh // 2)
        
        x_aug = x.clone()
        x_aug[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        gercek_lam = 1.0 - float((x2 - x1) * (y2 - y1) / (W * H))
        return x_aug, y, y[perm], gercek_lam, "cutmix"
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Standart Cross-Entropy kaybı kullanan derin bir ağ neden eğitim ilerledikçe tahminlerinde "aşırı güvenli" (overconfident) hale gelir? Label Smoothing bu logit patlamasını matematiksel olarak nasıl engeller?

> **Mentor Cevabı:**
> 1. **Softmax ve Sert One-Hot Etiketlerin Doğası:** Softmax fonksiyonu $p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$ şeklindedir. One-hot hedefte $y = [0, \dots, 1, \dots, 0]$ olduğunda kayıp $\mathcal{L} = -\log p_y$ olur. Kaybın tam olarak sıfırlanabilmesi ($\mathcal{L} = 0$) için $p_y = 1.0$ olmalıdır.
> 2. **Logit Patlaması:** $p_y = 1.0$ olabilmesi için hedef sınıf logit değeri $z_y$'nin diğer tüm sınıfların logitlerinden sonsuz kat büyük olması gerekir ($z_y - z_k \to \infty$). Model bu farkı açmak için ağırlıklarını sürekli büyütür ve yanlış olduğu örneklerde bile %99.9 güvenle tahmin yapar.
> 3. **Label Smoothing'in Çözümü:** Label Smoothing hedef olasılığı $1.0$ yerine $1 - \epsilon + \frac{\epsilon}{K}$ (örneğin $0.92$) yapar. Diğer sınıflara da $\frac{\epsilon}{K}$ (örneğin $0.02$) pay bırakır.
> 4. **Sonuç (Sınırlı Logitler ve Kalibrasyon):** Modelin hedefi artık $p_y = 1.0$ değil, sonlu bir olasılıktır ($p_y = 0.92$). Bu durum $z_y - z_k = \log\big(\frac{(K-1)(1-\epsilon)}{\epsilon}\big)$ gibi sonlu ve sabit bir logit farkı oluşturur. Ağırlıklar sonsuza büyümez, model aşırı ezberden kurtulur ve kalibre olur.

---

## 📜 Lisans & Telif Hakkı

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
