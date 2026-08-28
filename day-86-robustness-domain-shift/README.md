# Day 86: Görsel Bozulmalar (Bulanıklık/Gürültü) Altında Model Dayanıklılığı & Domain Shift

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Benchmark: Common Corruptions ICLR 2019](https://img.shields.io/badge/Benchmark-ImageNet--C%2FCIFAR--10--C-orange.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_robustness.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin beşinci gününde; Dan Hendrycks & Thomas Dietterich (ICLR 2019) *"Benchmarking Neural Network Robustness to Common Corruptions and Perturbations"* ve Hendrycks et al. (ICLR 2020) *"AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty"* makaleleri ışığında derin görme modellerinin **Görsel Bozulmalar (Gürültü, Bulanıklık, Hava Durumu, Dijital Artefaktlar) Altındaki Dayanıklılığını (Robustness)** ve **Alan Kaymasını (Domain Shift)** sıfırdan kurup analiz ediyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Standart derin öğrenme modelleri laboratuvar koşullarında (i.i.d. — bağımsız ve özdeş dağılmış temiz test kümelerinde) %95+ doğruluk elde etse de gerçek üretim ortamlarında (otonom araç kameraları, güvenlik CCTV, medikal röntgenler, mobil sensörler) ciddi performans çöküşü yaşar.

Bu sistem şu bilimsel ilkelerle çalışır:

1. **Dağılım Kayması (Covariate Shift / Domain Shift):**
   Eğitim veri dağılımı $P_{\text{train}}(X)$ ile canlı ortam veri dağılımı $P_{\text{test}}(X)$ arasındaki istatistiksel fark nedeniyle koşullu olasılık $P(Y|X)$ korunsa dahi modelin öznitelik manifoldları kırılır.
2. **Kestirme Yol (Spurious Texture Shortcut Bias - Geirhos et al. 2019):**
   Standart CNN'ler ve Transformer'lar nesnenin gerçek geometrik şekli yerine yüksek frekanslı doku (texture) kestirmelerine aşırı uyum (overfitting) gösterir. Bulanıklık, yağmur veya kamera gürültüsü bu yüksek frekansları sildiği an standart model çöker.
3. **Standartlaştırılmış Dayanıklılık Metrikleri (mCE & Rel-mCE):**
   Farklı model mimarilerini (ResNet, ViT, ConvNeXt) 8 temel bozulma tipi ve 5 şiddet seviyesi (Severity 1..5) altında adil şekilde karşılaştırmak için **Mean Corruption Error (mCE)** metriği kullanılır.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Canlı Ortam Şoklarının (Production Drop) Önceden Tespiti:**
  Modeli üretime almadan önce sis, yağmur, hareket bulanıklığı, sensör gürültüsü ve JPEG sıkıştırma altında nasıl tepki vereceğini önceden stres testine tabi tutar.
- **Kırılgan Karar Sınırlarının Pürüzsüzleştirilmesi:**
  AugMix ve Perturbation tutarlılık kaybı (Jensen-Shannon Divergence - JSD) ile modelin görsel varyasyonlara karşı değişmez (invariant) temsil öğrenmesini sağlar.
- **Güvenlik-Kritik Sistemlerde Sıfır Hata Toleransı:**
  Otonom sürüş ve savunma sanayii uygulamalarında hava muhalefeti kaynaklı yanlış sınıflandırmaları minimize eder.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Temiz Doğruluk (Clean Accuracy) vs Dayanıklılık (Robustness) Takası:**
  Aşırı agresif bozulma artırmaları (heavy data augmentation) temiz laboratuvar test kümesindeki doğrulukta %1-%2 oranında küçük bir gerilemeye (trade-off) yol açabilir.
- **Hesaplama ve Eğitim Süresi:**
  Çoklu bozulma zincirleri ve tutarlılık kayıpları (JSD) ileri/geri geçiş (forward/backward pass) maliyetini 2-3 kat artırabilir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Dayanıklılık Yaklaşımı | Ek Eğitim Maliyeti | mCE Düşürme Gücü | Temiz Doğruluk Etkisi | Genelleme Alanı |
|---|---|---|---|---|
| **AugMix + JSD (Bizim Yöntem)** | **Orta (2x Forward)** | **YÜKSEK (-%10 ile -%25 mCE)** | **Nötr / Pozitif (+%0.5)** | **Tüm Bozulma Tipleri** |
| **Standart Veri Artırma (RandAugment)** | Düşük (1x Forward) | Orta (-%5 ile -%10 mCE) | Pozitif (+%1.0) | Sınırlı Dönüşümler |
| **Adversarial Training (PGD/FGSM)** | Çok Yüksek (8-10x Forward)| Düşük (Yalnızca $L_\infty$ gürültüsü) | Negatif (-%5 ile -%10) | Yalnızca Saldırı Gürültüsü |
| **Stylized-ImageNet (Shape Bias)** | Yüksek (Style Transfer) | Yüksek (-%15 mCE) | Hafif Negatif (-%2.0) | Doku Değişmezliği |
| **Derin Öznitelik Denoising (Feature Denoise)**| Yüksek (Non-local Bloklar)| Yüksek (-%12 mCE) | Nötr | Gürültü ve Sis |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       BOZULMA VE ALAN KAYMASI (CORRUPTIONS & DOMAIN SHIFT) STRES TESTİ                    │
│                                                                                                           │
│       Temiz Görüntü: x                                                                                    │
│          │                                                                                                │
│          ├───────────────────────────────────────────────────────────────────────┐                        │
│          ▼                                                                       ▼                        │
│     [ STANDART YOL ]                                                   [ BOZULMA MOTORU ]                 │
│     Temiz Test Doğruluğu:                                              c ∈ {Noise, Blur, Digital, Weather}│
│     Acc_clean = %100 - E_clean                                         s ∈ {1, 2, 3, 4, 5}                │
│                                                                                  │                        │
│                                                                                  ▼                        │
│                                                                        Bozulmuş Görüntü: x_{c, s}         │
│                                                                                  │                        │
│                                                                                  ▼                        │
│                                                                        [ STRES TESTİ VE mCE ANALİZİ ]     │
│                                                                        mCE = 1/|C| sum_c (1/5 sum_s E_cs) │
│                                                                        Rel-mCE = mCE - E_clean            │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Bozulma Hatası ve Ortalama Bozulma Hatası (mCE)
Hendrycks & Dietterich (2019) formülasyonu:

$$E_{c, s} = 100 - \text{Top1\_Acc}(c, s)$$

$$\text{mCE} = \frac{1}{|C|} \sum_{c \in C} \left( \frac{1}{5} \sum_{s=1}^5 E_{c, s} \right)$$

### 2. Göreceli Bozulma Hatası (Relative mCE)
Modelin temiz doğruluk avantajından bağımsız, saf bozulma kaynaklı performans kaybı:

$$\text{Rel-mCE} = \text{mCE} - E_{\text{clean}}$$

### 3. AugMix Tutarlılık Kaybı (Jensen-Shannon Divergence - JSD)
Temiz $x$ ve bozulmuş $x_{\text{aug}}$ girdilerinin tahmin olasılık dağılımları $p_{\text{clean}}$ ve $p_{\text{aug}}$ olsun. Ortalama dağılım $M = \frac{1}{2}(p_{\text{clean}} + p_{\text{aug}})$:

$$\mathcal{L}_{\text{AugMix}} = \mathcal{L}_{\text{CE}}(x, y) + \mathcal{L}_{\text{CE}}(x_{\text{aug}}, y) + \lambda \cdot \text{JSD}(p_{\text{clean}} \parallel p_{\text{aug}})$$

$$\text{JSD}(p_{\text{clean}} \parallel p_{\text{aug}}) = \frac{1}{2} D_{\text{KL}}(p_{\text{clean}} \parallel M) + \frac{1}{2} D_{\text{KL}}(p_{\text{aug}} \parallel M)$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Domain Shift (Covariate Shift)**| *Alan / Dağılım Kayması* | Eğitim ve test verilerinin girdi dağılımları ($P(X)$) arasındaki istatistiksel uyumsuzluk. |
| **Common Corruptions** | *Yaygın Görsel Bozulmalar* | Kamera gürültüsü, sis, yağmur, hareket bulanıklığı gibi gerçek hayatta sıkça karşılaşılan 15-20 doğal bozulma tipi. |
| **Severity Level ($s$)** | *Bozulma Şiddet Seviyesi* | Bozulmanın şiddetini kademelendiren 1'den (en hafif) 5'e (en ağır) kadar olan standart derecelendirme. |
| **Mean Corruption Error (mCE)**| *Ortalama Bozulma Hatası* | Bir modelin tüm bozulma tipleri ve şiddet seviyelerindeki ortalama hata oranı. |
| **Relative mCE (Rel-mCE)** | *Göreceli Bozulma Hatası* | Modelin temel temiz hatası çıkarıldıktan sonra yalnızca bozulmalardan kaynaklanan net hata payı. |
| **Shape Bias vs Texture Bias** | *Şekil vs Doku Yanlılığı* | İnsanların nesneleri şekle göre tanımasına karşın derin modellerin kırılgan doku desenlerine aşırı odaklanması olgusu. |
| **AugMix** | *Artırma Karıştırma* | Rastgele görsel dönüşüm zincirlerini paralel karıştırıp JSD tutarlılık kaybıyla modeli dayanıklı kılan eğitim yöntemi. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Gerçek dünya gürültüsü ve hava koşullarında istikrarlı başarım; mCE ve Rel-mCE metrikleriyle objektif dayanıklılık ölçümü; Otonom sürüş ve güvenlik kameralarında sıfır hata toleransı. |
| **Weaknesses (Zayıf Yönler)** | Bozulma artırma (AugMix) eğitimi ek eğitim süresi gerektirir; Aşırı agresif bozulma uygulanırsa temiz doğrulukta (%1-2) küçük takas oluşabilir. |
| **Opportunities (Fırsatlar)** | Endüstriyel kalite kontrol ve CCTV kameralarında güvenilirlik; OOD tespiti ve Kalibrasyon katmanlarıyla birleşerek tam güvenlik kalkanı. |
| **Threats (Tehditler)** | Eğitilmemiş çok farklı yeni bozulma tiplerinde genelleme açığı oluşabilir. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-86-robustness-domain-shift/`](.) dizinindedir:

### A. 8 Temel Bozulma Simülatörü (PyTorch)
Dosya: [`src/bozulma_motoru.py`](src/bozulma_motoru.py)
```python
class GorselBozulmaMotoru:
    @staticmethod
    def gaussian_noise(x: torch.Tensor, siddet: int = 1) -> torch.Tensor:
        std_listesi = [0.10, 0.20, 0.35, 0.50, 0.70]
        std = std_listesi[siddet - 1]
        return x + torch.randn_like(x) * std

    @staticmethod
    def gaussian_blur(x: torch.Tensor, siddet: int = 1) -> torch.Tensor:
        sigmalar = [0.6, 1.0, 1.5, 2.2, 3.0]
        sigma = sigmalar[siddet - 1]
        k_size = 2 * int(4 * sigma + 0.5) + 1
        ax = torch.arange(-k_size // 2 + 1., k_size // 2 + 1.)
        xx = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel_1d = xx / xx.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0).to(x.device)
        kernel_2d = kernel_2d.repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, kernel_2d, padding=k_size // 2, groups=x.size(1))
```

---

## 📊 Deneysel Sonuçlar ve Doğrulama Çıktıları

`ana_akis.py` koşturularak elde edilen deneysel stres testi karşılaştırması:

```text
=====================================================================================
🚀 Day 86: Görsel Bozulmalar Altında Model Dayanıklılığı & Domain Shift Laboratuvarı
=====================================================================================
📌 Çalışma Ortamı Cihazı: CUDA

[1/4] Model 1: Standart Model (Bozulmasız Temiz Veri) Eğitiliyor...
[2/4] Model 2: Dayanıklı Model (Perturbation / Robust Augmentation) Eğitiliyor...
[3/4] 8 Bozulma Tipi ve 5 Şiddet Seviyesinde Kapsamlı Stres Testi Koşuluyor...

=================================================================
📊 MODEL DAYANIKLILIK (ROBUSTNESS) METRİK KARŞILAŞTIRMASI
=================================================================
  Metrik                       | Standart Model | Dayanıklı Model
-----------------------------------------------------------------
  Temiz Test Doğruluğu (Clean) | %100.00       | %100.00      
  Bozulma Altı Ort. Doğruluk   | %49.85        | %57.05       
  Mean Corruption Error (mCE)  | %50.15        | %42.95       
  Relative mCE (Rel-mCE)       | %50.15        | %42.95       
=================================================================
  🚀 Dayanıklı Modelin Bozulmalar Altındaki Doğruluk Üstünlüğü: +%7.20

✓ 6 Panelli Teşhis Panosu Kaydedildi: ciktilar/robustness_domain_shift_paneli.png
```

- **+7.20% Bozulma Altı Doğruluk Üstünlüğü:** Dayanıklı Model (Perturbation Training & JSD), 8 farklı bozulma ve 5 şiddet seviyesinde mCE hatasını %50.15'ten %42.95'e indirerek kırılganlığı bertaraf etmiştir.
- **Birim Test Güvencesi:** [`testler/test_robustness.py`](testler/test_robustness.py) altındaki **8/8 birim test %100 PASSED (7.42s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/robustness_domain_shift_paneli.png`](ciktilar/robustness_domain_shift_paneli.png) konumundadır:

1. **Model Dayanıklılığı ve Bozulma Mimarisi:** Laboratuvar vs Gerçek Dünya Dayanıklılık Açığı (The Robustness Gap) diyagramı.
2. **8 Bozulma Tipi ve Matematiksel Modelleri:** Gürültü, Bulanıklık, Hava Koşulları ve Dijital Bozulma modelleri.
3. **Bozulma Şiddetine Göre Doğruluk Düşüşü (Clean ──> s5):** Standart model vs Dayanıklı model düşüş eğrileri.
4. **Bozulma Tipine Göre Ortalama Doğruluk:** 8 farklı bozulma kategorisinde model performansları.
5. **mCE ve Rel-mCE Hata Karşılaştırması:** Standart Model (%50.15 mCE) vs Dayanıklı Model (%42.95 mCE).
6. **Model Dayanıklılık SWOT Karar Matrisi:** Endüstriyel karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Görüntüdeki yüksek frekanslı doku yanlılığını (Texture Bias) kırıp şekil yanlılığını (Shape Bias) artırmak için **Fourier Spektrum Filtreleme (Fourier High-Frequency Perturbation)** fonksiyonu yazınız.

```python
import torch

def fourier_frekans_gurultusu(x: torch.Tensor, esik_yaricap: float = 8.0, siddet: float = 0.5) -> torch.Tensor:
    """
    2D Fourier uzayına geçerek yüksek frekans bileşenlerine rastgele faz pertürbasyonu uygular.
    """
    # 2D Gerçel Fourier Dönüşümü (RFFT2)
    fft_x = torch.fft.rfft2(x, norm="ortho")
    b, c, h, w_fft = fft_x.shape
    
    # Düşük frekans merkezli yarıçap maskesi
    y_coords = torch.fft.fftfreq(h)[:, None]
    x_coords = torch.fft.rfftfreq(w_fft * 2 - 2)[None, :]
    r = torch.sqrt(y_coords**2 + x_coords**2)
    yuksek_frekans_maskesi = (r > (esik_yaricap / h)).to(x.device)
    
    # Yüksek frekanslara rastgele faz kayması ekle
    rastgele_faz = torch.randn_like(fft_x) * siddet
    fft_x_bozuk = fft_x.clone()
    fft_x_bozuk[:, :, yuksek_frekans_maskesi] += rastgele_faz[:, :, yuksek_frekans_maskesi]
    
    # Ters Fourier Dönüşümü (IRFFT2)
    return torch.fft.irfft2(fft_x_bozuk, s=(x.shape[-2], x.shape[-1]), norm="ortho")
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Temiz bir test kümesinde %98 doğruluk alan bir derin görme modeli, neden basit bir Gauss bulanıklığı veya hafif kamera gürültüsü altında %40 doğruluğa kadar çökebilir? Bu kırılganlığın kök nedeni nedir?

> **Mentor Cevabı:**
> 1. **Doku Yanlılığı (Texture vs Shape Bias):** İnsan beyni nesneleri global şekil geometrisine göre (ör. kedinin kulak şekli, arabanın tekerlek yerleşimi) tanır. Ancak derin evrişimli (CNN) modeller standart eğitimde kayıp fonksiyonunu en hızlı düşürmek için görseldeki yüksek frekanslı piksel dokularını (ör. kürk deseni, asfalt mikro-pürüzleri) "kestirme yol" (shortcut) olarak ezberler.
> 2. **Yüksek Frekansların Tahribatı:** Gauss bulanıklığı bir alçak geçiren filtredir (low-pass filter) ve tüm yüksek frekans dokularını yok eder. Model sadece dokuya bağımlı olduğu için şekli yorumlayamaz ve tahmin çatlar.
> 3. **Çözüm (AugMix & JSD):** Modeli eğitim esnasında Fourier/Görsel pertürbasyonlarına maruz bırakarak ve JSD tutarlılık kaybı uygulayarak, modelin dikkati kırılgan dokulardan değişmez yapısal şekil özniteliklerine (Shape Bias) kaydırılır.

---

### 📌 Git & Yol Haritası Güncellemesi:
- **Tamamlanan:** Gün 86 (`day-86-robustness-domain-shift`) başarıyla tamamlandı.
- **Sıradaki Gün:** **Day 87: MLflow / Weights & Biases ile Merkezi Deney Takibi ve Artefakt Kayıt Sistemi (`day-87-experiment-registry`)**.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
