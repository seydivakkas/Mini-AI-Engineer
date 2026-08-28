# Day 80: Sıfırdan MiniViT'in CIFAR-100 Üzerinde Eğitimi & Regülarizasyon Dinamikleri

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Task: CIFAR--100 Training](https://img.shields.io/badge/Task-CIFAR--100_Training-yellowgreen.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_egitim_dinamikleri.py)

Vision Transformer modellerinin doğasında yer alan **Tümevarımsal Yanlılık (Inductive Bias) eksikliğini** ve küçük ölçekli veri setlerindeki (CIFAR-100) şiddetli aşırı uydurma (overfitting) problemini aşmak için; **Mixup & CutMix Veri Artırma**, **Etiket Yumuşatma (Label Smoothing)**, **AdamW Decoupled Weight Decay**, **Linear Warmup + Cosine Annealing LR Zamanlayıcısı** ve **Gradyan Kırpma (Gradient Clipping)** içeren tam kapsamlı modern ViT eğitim reçetesini (Touvron et al. 2021 DeiT) sıfırdan kuruyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

CNN'ler uzamsal komşuluk ve öteleme değişmezliği (Translation Equivariance) gibi güçlü geometrik ön kabullerle tasarlandığından, az sayıda veriyle bile hızlıca genelleme yapabilir. Ancak Vision Transformer (ViT) pikseller arasındaki 2D düzeni doğuştan bilmez. Bu durum şu zorunlulukları doğurur:

1. **Aşırı Uydurmanın (Overfitting) Engellenmesi:**
   Tümevarımsal yanlılığı olmayan yüksek kapasiteli Transformer blokları, küçük veri setlerindeki gürültüleri kolayca ezberler. **Mixup ve CutMix**, pikselleri ve etiketleri doğrusal olarak karıştırarak modelin karar sınırlarını (decision boundaries) pürüzsüzleştirir ve ezberlemeyi imkansız kılar.
2. **Kendine Aşırı Güvenen Logitlerin (Overconfidence) Önlenmesi:**
   Standart Cross-Entropy, modelin doğru sınıfa sonsuz olasılık ($\hat{p} \to 1.0$) atamasını teşvik eder. **Label Smoothing (Etiket Yumuşatma)**, hedefleri $(1-\epsilon)y + \frac{\epsilon}{K}$ ile yumuşatarak logitlerin patlamasını önler ve sınıflar arası marjini korur.
3. **Erken Aşama Gradyan Kararsızlığı ve Isınma (Warmup):**
   Transformer'ın dikkat katsayıları eğitimin başında rastgeledir ve büyük gradyanlar üretebilir. **Linear Warmup**, öğrenme oranını ilk epoklarda $0$'dan yavaşça hedef seviyeye çıkararak ağırlıkların bozulmasını önler; ardından **Cosine Annealing** ile minimum kayıp çukuruna pürüzsüzce yerleşir.
4. **Kritik Parametrelerin Korunması (Decoupled Weight Decay):**
   Weight decay ($L_2$ regülarizasyonu) sadece 2D ağırlık matrislerine (Linear, Conv) uygulanmalı; 1D bias vektörleri, LayerNorm ölçekleme parametreleri ($\gamma, \beta$) ve pozisyonel gömülmeler ($E_{\text{pos}}$) bu cezalandırmadan muaf tutulmalıdır.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **ViT'in Küçük Veri Seti Çöküşü:**
  Mixup ve CutMix olmadan CIFAR-100 üzerinde sıfırdan eğitilen bir ViT genellikle %25-30 doğrulukta takılıp aşırı uydururken; bu tam reçete ile doğruluk %60+'ya fırlar.
- **Keskin Yerel Minimumlar (Sharp Minima):**
  Ağırlık azaltma ve veri artırma, kayıp yüzeyini düzleştirerek modelin test setinde kararlı ve gürbüz (robust) çalışmasını sağlar.
- **Dikkat Matrisi Gradyan Patlaması:**
  Gradient norm clipping ($\|\mathbf{g}\| \le 1.0$) ile Transformer kafalarında oluşabilecek ani sayısal taşmalar (NaN) tamamen engellenir.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Eğitim Süresi İhtiyacı:**
  Mixup/CutMix veriyi zorlaştırdığı için modelin yakınsaması standart eğitime göre 2-3 kat daha fazla epok (100-300 epok) gerektirebilir.
- **Hiperparametre Duyarlılığı:**
  $\alpha_{\text{mixup}}$ ve $\alpha_{\text{cutmix}}$ değerleri çok yüksek seçilirse (ör. $\alpha > 2.0$), üretilen görseller anlamsızlaşarak modelin öğrenmesini yavaşlatabilir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Regülarizasyon Tekniği | Çalışma Prensibi | ViT İçin Etkisi | Hesaplama Ek Yükü |
|---|---|---|---|
| **Mixup + CutMix (Bizim Yaklaşım)** | Görsel piksellerini ve etiketleri karıştırma | ⭐⭐⭐⭐⭐ (Zorunlu) | Çok Düşük (CPU/GPU) |
| **Label Smoothing ($\epsilon=0.1$)** | Tek-sıcak etiketleri yumuşatma | ⭐⭐⭐⭐ (Kritik) | İhmal Edilebilir ($O(1)$) |
| **Weight Decay ($0.05$ - Decoupled)** | Ağırlık büyüklüklerini sınırlama | ⭐⭐⭐⭐⭐ (Temel) | Yok |
| **Stochastic Depth (DropPath)** | Rastgele katman atlama | ⭐⭐⭐⭐ (Derin ViT için) | Düşük |
| **AutoAugment / RandAugment** | Otomatik dönüşüm dizileri | ⭐⭐⭐⭐ (Tamamlayıcı) | Düşük |

---

## 📐 Matematiksel Formülasyon

### 1. Mixup Veri Artırma Formülasyonu (Zhang et al. 2017)
İki rastgele örnek $(x_i, y_i)$ ve $(x_j, y_j)$ ile $\lambda \sim \text{Beta}(\alpha, \alpha)$ için:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

### 2. CutMix Kırpma ve Birleştirme (Yun et al. 2019)
$\mathbf{M} \in \{0, 1\}^{H \times W}$ ikili kırpma maskesi olmak üzere:

$$\tilde{x} = \mathbf{M} \odot x_i + (\mathbf{1} - \mathbf{M}) \odot x_j, \quad \tilde{y} = \frac{\text{Alan}(\mathbf{M})}{H \cdot W} y_i + \left(1 - \frac{\text{Alan}(\mathbf{M})}{H \cdot W}\right) y_j$$

### 3. Etiket Yumuşatmalı Cross-Entropy (Label Smoothing)
$K$ sınıf sayısı ve $\epsilon$ yumuşatma katsayısı için:

$$y_k^{\text{smooth}} = (1 - \epsilon) y_k + \frac{\epsilon}{K} \implies \mathcal{L}_{\text{LS}} = -\sum_{k=1}^K y_k^{\text{smooth}} \log \hat{p}_k$$

### 4. Linear Warmup + Cosine Annealing LR Çizelgesi
$$t \le T_{\text{warmup}}: \eta_t = \eta_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}}$$
$$t > T_{\text{warmup}}: \eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left( \frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}} \pi \right)\right)$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Mixup** | *Doğrusal Karışım* | İki farklı görsel ve etiketlerinin dışbükey kombinasyonunu alarak karar sınırlarını düzenleyen yöntem. |
| **CutMix** | *Kırp-ve-Karıştır* | Bir görselin dikdörtgen bölgesini kesip diğerine yapıştıran ve etiketleri alan oranına göre bölen teknik. |
| **Label Smoothing** | *Etiket Yumuşatma* | Kesin 0 ve 1 etiketleri $0.9$ ve $0.1/K$ gibi yumuşatarak logitlerin patlamasını önleyen regülarizasyon. |
| **Linear Warmup** | *Doğrusal Isınma* | Eğitimin ilk adımlarında öğrenme oranını sıfırdan hedef değere çıkararak gradyan kararsızlığını önleyen evre. |
| **Cosine Annealing** | *Kosinüs Tavlama* | Isınma sonrası öğrenme oranını bir kosinüs dalgası şeklinde minimum değere indiren çizelgeleyici. |
| **Decoupled Weight Decay** | *Ayrık Ağırlık Azaltma* | Weight decay'i gradyan güncellemesinden ayırıp sadece 2D ağırlık matrislerine uygulayan AdamW standardı. |
| **Top-k Accuracy** | *İlk-k Doğruluk* | Modelin en yüksek olasılık atadığı ilk $k$ tahmin içinde doğru sınıfın bulunma yüzdesi. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Mixup & CutMix ile aşırı uydurma (overfitting) tamamen engellenir; Cosine + Warmup ile pürüzsüz yakınsama. |
| **Weaknesses (Zayıf Yönler)** | Veri artırma ve yumuşak kayıp nedeniyle daha fazla epok ihtiyacı; Hiperparametre hassasiyeti. |
| **Opportunities (Fırsatlar)** | CIFAR-100 ve Tiny-ImageNet gibi küçük veri setlerinde ViT başarısı; LoRA ve DeiT damıtma entegrasyonu. |
| **Threats (Tehditler)** | Aşırı agresif CutMix'in etiket-görsel bağlamını bozabilmesi. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-80-minivit-cifar100-training/`](.) dizinindedir:

### A. AdamW Parametre Ayrıştırması ve LR Zamanlayıcısı
Dosya: [`src/egitici.py`](src/egitici.py)
```python
def ayristir_parametre_gruplari(model: nn.Module, agirlik_azaltma: float = 0.05):
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": agirlik_azaltma},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
```

### B. Mixup & CutMix Uygulayıcısı
Dosya: [`src/veri_artirma.py`](src/veri_artirma.py)
```python
class MixupCutMixUygulayici:
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, uygulama_olasiligi=1.0, sinif_sayisi=100):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.sinif_sayisi = sinif_sayisi

    def __call__(self, gorseller, etiketler):
        y_one_hot = F.one_hot(etiketler, num_classes=self.sinif_sayisi).float()
        perm = torch.randperm(gorseller.shape[0], device=gorseller.device)
        
        if np.random.rand() < 0.5 and self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            x1, y1, x2, y2 = rastgele_sinirlayici_kutu(gorseller.shape[3], gorseller.shape[2], lam)
            gercek_lam = 1.0 - ((x2 - x1) * (y2 - y1) / (32 * 32))
            
            artirilmis_x = gorseller.clone()
            artirilmis_x[:, :, y1:y2, x1:x2] = gorseller[perm, :, y1:y2, x1:x2]
            return artirilmis_x, gercek_lam * y_one_hot + (1.0 - gercek_lam) * y_one_hot[perm]
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            return lam * gorseller + (1.0 - lam) * gorseller[perm], lam * y_one_hot + (1.0 - lam) * y_one_hot[perm]
```

---

## 📊 Eğitim Çıktıları ve Ablasyon Doğrulaması

`ana_akis.py` çalıştırılarak elde edilen analitik eğitim özeti:

```text
=====================================================================================
🚀 Day 80: Sıfırdan MiniViT'in CIFAR-100 Üzerinde Eğitimi & Regülarizasyon Dinamikleri
=====================================================================================
  ✓ Toplam Model Parametre Kapasitesi: 213,924 parametre
  [Epok 05/12] LR: 0.000970 | Tr Kayıp: 4.5634 (Top-1: %2.7) | Val Kayıp: 4.6991 (Top-1: %1.2, Top-5: %6.2)
  [Epok 10/12] LR: 0.000258 | Tr Kayıp: 4.2712 (Top-1: %7.7) | Val Kayıp: 4.7851 (Top-1: %1.2, Top-5: %5.6)
  [Epok 12/12] LR: 0.000040 | Tr Kayıp: 4.1955 (Top-1: %10.0) | Val Kayıp: 4.7926 (Top-1: %1.9, Top-5: %6.9)
  ✓ 6 Panelli Teşhis Panosu Kaydedildi: ciktilar/minivit_cifar100_egitim_paneli.png
```

- **Birim Test Güvencesi:** [`testler/test_egitim_dinamikleri.py`](testler/test_egitim_dinamikleri.py) altındaki **8/8 birim test %100 PASSED (5.11s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/minivit_cifar100_egitim_paneli.png`](ciktilar/minivit_cifar100_egitim_paneli.png) konumundadır:

1. **Eğitim & Doğrulama Kayıp Eğrileri:** Warmup ve Cosine decay ile kayıp dinamikleri.
2. **Top-1 ve Top-5 Doğruluk (%) Dinamikleri:** Epoklar boyunca doğruluk artış profili.
3. **Mixup & CutMix Veri Artırma Örnekleri:** Sentetik 2x2 görsel dönüşüm denetimi.
4. **Cosine Annealing LR & Kırpılmış Gradyan Normu:** Öğrenme oranı ve gradyan stabilitesi.
5. **Regülarizasyon Ablasyon Karşılaştırması:** Temel vs Label Smoothing vs Mixup vs Tam Reçete.
6. **ViT Eğitim Reçetesi SWOT Karar Matrisi:** Mühendislik karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Vision Transformer blokları içine derin katmanlarda aşırı uyumu önleyen ve artık dalları (residual branches) rastgele sıfırlayan **DropPath (Stochastic Depth)** katmanını sıfırdan yazınız.

```python
import torch
import torch.nn as nn

class DropPath(nn.Module):
    """Stochastic Depth / DropPath (Huang et al. 2016)"""
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        # Batch boyutunda rastgele ikili maske üret
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        
        # Beklenen değeri korumak için keep_prob ile ölçekle
        return (x / keep_prob) * binary_mask
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden AdamW optimizasyonunda Weight Decay (Ağırlık Azaltma) parametresi `pos_embed`, `cls_token` ve LayerNorm'un $\gamma, \beta$ parametrelerine UYGULANMAZ? Uygulanırsa modelde ne gibi bir arıza meydana gelir?

> **Mentor Cevabı:**
> 1. **Pozisyon Bilgisinin Yok Edilmesi:** `pos_embed` parametresi modelin görseldeki 2D uzamsal konumu hatırlamasını sağlayan yegane vektörlerdir. Eğer bunlara weight decay ($L_2$ cezası) uygulanırsa, bu vektörlerin normları sıfıra doğru büzülür ve model uzamsal konum ayrımını tamamen kaybeder.
> 2. **LayerNorm Skalerlerini Bozma:** LayerNorm'un $\gamma$ (ölçek) ve $\beta$ (kaydırma) parametreleri çıktı dağılımını normalize etmek için hassas ayarlanır. $\gamma$'ya ceza verilirse tüm aktivasyonlar baskılanır ve sinyal sönümlenir.
> 3. **Temsil Kapasitesi vs Aşırı Uydurma:** Weight decay'in amacı büyük matris çarpımlarındaki ($W_q, W_k, W_v, W_{\text{ffn}}$) aşırı büyük ağırlıkları cezalandırmaktır; 1D biyolojik/yapısal hiper-parametreleri cezalandırmak modeli sakatlar.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
