# Day 56: Edge Cihazlar İçin Sıfırdan Hafif CNN, Depthwise Separable Conv ve Analitik FLOPs Hesabı (TinyVisionCNN Edge Profiler)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 56. gününde geliştirilen **Edge Cihazlar ve Gömülü Sistemler İçin Sıfırdan Hafif CNN Mimarisi (TinyVisionCNN) ve Analitik FLOPs/MACs Profilleme Motorudur**. Mobil SoC'ler, Raspberry Pi, Jetson Nano ve mikrodenetleyiciler gibi kısıtlı donanımlarda yüksek doğruluk ve milisaniye seviyesinde gecikme elde etmek için **Derinlik Ayrışımlı Konvolüsyon (Depthwise Separable Convolution)** ve **Global Ortalama Havuzlama (Global Average Pooling)** tekniklerini sıfırdan hayata geçirir.

---

## 📖 Mentorluk Dersi ve Edge AI Mimari Teorisi

### 1. Geleneksel Konvolüsyonun Edge Darboğazı

Standart bir 2D konvolüsyon katmanı ($K \times K \times C_{\text{in}} \times C_{\text{out}}$), hem uzamsal (spatial) filtrelemeyi hem de kanallar arası (cross-channel) kombinasyonu tek bir matris çarpımında birleştirir.

Girdi tensörü $D_F \times D_F \times M$ ve çıktı tensörü $D_F \times D_F \times N$ olduğunda standart konvolüsyonun hesaplama maliyeti:
$$\text{FLOPs}_{\text{Standart}} = 2 \cdot D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$$

Bu durum, kanal sayısı $M, N$ arttıkça karesel bir hesaplama patlamasına yol açar.

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Evrişimli katmanların (Conv2D, BatchNorm, ReLU, MaxPool, GlobalAvgPool) parametre ve tensör boyut dinamiklerini sıfırdan inceleyip kavramak için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Kara kutu kütüphane çağrıları yerine tensör boyut hesaplamalarını ($W_{out} = \lfloor (W - K + 2P)/S \rfloor + 1$) tam kontrol altına alır.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Büyük ResNet veya ConvNeXt mimarileri kadar derin değildir; kapasitesi küçük görevlerle sınırlıdır.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- ResNet-18, MobileNetV3 veya SqueezeNet.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Edge AI / TinyML** | *Edge Artificial Intelligence* | Kısıtlı bellek ve işlem gücüne sahip mikrodenetleyici veya uç cihazlarda (mobil, IoT) çalışacak ultra hafif CNN mimarileri tasarlama. |
| **Toplu Normalizasyon (BatchNorm)** | *Batch Normalization (`nn.BatchNorm2d`)* | Her mini-batch'in ara katman aktivasyonlarını normalize ederek iç kovaryans kaymasını önleyen ve eğitimi hızlandıran katman. |
| **Rezidüel Atlama (Residual Shortcut)** | *Residual Identity Shortcut* | Girdiyi katman çıktısına doğrudan ekleyerek kaybolan gradyanları (Vanishing Gradients) engelleyen mimari köprü. |
| **Global Ortalama Havuzlama (GAP)** | *Global Average Pooling (`AdaptiveAvgPool2d`)* | Öznitelik haritalarını tek bir değere indirgeyerek Flatten katmanının yol açtığı devasa parametre patlamasını önleyen yöntem. |

---

## 2. Derinlik Ayrışımlı Konvolüsyon (Depthwise Separable Convolution)

MobileNet mimarisinin temelini oluşturan bu teknik, konvolüsyon işlemini iki bağımsız aşamaya çarpanlarına ayırır (factorization):

1. **Depthwise Konvolüsyon ($K \times K \times 1 \times M$ with `groups=M`):**
   - Her giriş kanalına yalnızca bir adet $K \times K$ uzamsal filtre uygulanır. Kanallar arasında hiçbir bilgi akışı olmaz.
   $$\text{FLOPs}_{\text{Depthwise}} = 2 \cdot D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$$
2. **Pointwise Konvolüsyon ($1 \times 1 \times M \times N$):**
   - $1 \times 1$ konvolüsyon ile farklı kanallardaki özellikler doğrusal olarak harmanlanır.
   $$\text{FLOPs}_{\text{Pointwise}} = 2 \cdot M \cdot N \cdot D_F \cdot D_F$$

```
    ┌───────────────────────────────────────────────────────────────────────────────────────────┐
    │                                1. STANDART KONVOLÜSYON (AĞIR)                             │
    │  [Girdi: HxW x C_in] ───────────────► [3x3 x C_in x C_out Conv] ─────────────► [HxW x C_out]│
    └───────────────────────────────────────────────────────────────────────────────────────────┘

                                                ▼  AYRIŞTIRMA (FACTORIZATION)

    ┌───────────────────────────────────────────────────────────────────────────────────────────┐
    │                    2. DERİNLİK AYRIŞIMLI KONVOLÜSYON (DEPTHWISE SEPARABLE)                │
    │  [Girdi: HxW x C_in] ──► [3x3 Depthwise (groups=C_in)] ──► [1x1 Pointwise] ──► [HxW x C_out] │
    │  (Uzamsal Filtreleme)                              (Kanal Harmanlama)                     │
    └───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Matematiksel Tasarruf İspatı

Derinlik ayrışımlı konvolüsyonun standart konvolüsyona göre hesaplama oranı:
$$\text{Oran} = \frac{\text{FLOPs}_{\text{Depthwise}} + \text{FLOPs}_{\text{Pointwise}}}{\text{FLOPs}_{\text{Standart}}} = \frac{D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2}{D_K^2 \cdot M \cdot N \cdot D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}$$

$3 \times 3$ filtre ($D_K = 3$) ve $N=64$ çıkış kanalı için:
$$\text{Oran} = \frac{1}{64} + \frac{1}{9} \approx 0.0156 + 0.1111 = 0.1267 \implies \mathbf{\%87.3 \text{ FLOPs Tasarrufu! } (8\times - 9\times \text{ Daha Az İşlem})}$$

---

### 4. Global Average Pooling (GAP) vs Flatten Dense Katmanları

Klasik CNN modellerinde (VGG, Standart CNN), konvolüsyon özellikleri $128 \times 8 \times 8 = 8192$ boyutunda düzleştirilip (Flatten) $256$'lık bir Linear katmana bağlanır. Bu tek bir katman $8192 \times 256 = \mathbf{2,097,152 \text{ Parametre}}$ ($8.4\text{ MB}$) tüketir!

**TinyVisionCNN Yaklaşımı:**
- Özellik haritası $128 \times 8 \times 8$'den `AdaptiveAvgPool2d((1, 1))` ile doğrudan $128 \times 1 \times 1$'e indirgenir (**0 Parametre**).
- Sınıflandırıcı yalnızca $128 \times 10 = \mathbf{1,280 \text{ Parametre}}$ tüketir!

---

## 🛠️ Dizin Yapısı

```
day-56-tinyvisioncnn/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca analitik FLOPs, parametre ve gecikme kıyaslama betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── modeller.py                  # DerinlikAyrisimliKonvolusyon, StandartCNN, TinyVisionCNN
│   ├── profil_motoru.py             # FLOPsProfilMotoru (Forward hook tabanlı kesin MACs/FLOPs hesaplayıcı)
│   └── gorsellestirici.py           # 6-Panelli Edge AI Teşhis Panosu (TinyVision Profiler Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_tinyvision.py           # 7 adet birim test (Tümü Başarılı: %100 PASSED)
└── ciktilar/
    └── tinyvision_profil_paneli.png # 6 panelli yüksek çözünürlüklü teşhis panosu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışın Çalıştırılması
```bash
python ana_akis.py
```

### 3. Birim Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📊 Karşılaştırmalı Performans ve Profilleme Tablosu

| Metrik / Performans Ölçütü | Standart CNN | TinyVisionCNN | Tasarruf & Hızlanma Oranı |
|---|---|---|---|
| **Toplam Parametre Sayısı** | $2,213,290$ | **$93,898$** | **$\%95.8$ Tasarruf ($23.6\times$ Daha Hafif)** |
| **Model Bellek Boyutu** | $8.44\text{ MB}$ | **$0.36\text{ MB}$ ($366.8\text{ KB}$)** | **Edge ROM / Flash Uyumlu** |
| **Toplam MACs** | $14,976,512$ | **$2,168,064$** | **$\%85.5$ Azalma** |
| **Toplam MFLOPs** | $29.95\text{ MFLOPs}$ | **$4.34\text{ MFLOPs}$** | **$\%85.5$ Hesaplama Tasarrufu ($6.9\times$ Az İşlem)** |
| **CPU Çıkarım Gecikmesi** | $3.24\text{ ms}$ | **$0.95\text{ ms}$** | **$3.4\times$ Hızlanma ($>1000\text{ FPS}$)** |
| **Edge Dağıtım Uygunluğu** | Düşük (Yüksek RAM/FLOPs) | **Mükemmel (A+)** | **Raspberry Pi / Cortex-M Uyumlu** |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** MobileNetV2'deki **Inverted Residual Block (Ters Artık Blok)** ve **Squeeze-and-Excitation (SE) Kanal Dikkat Mekanizmasını** birleştiren ultra hafif bir `SEInvertedResidualBlock` modülü geliştirmek.

**Tamamlanan Kod Çözümü:**
```python
import torch
import torch.nn as nn

class SEInvertedResidualBlock(nn.Module):
    """Genişletme (Expansion), Depthwise Conv, Squeeze-and-Excitation ve Residual bağlantılı blok."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand_ratio: int = 4):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
        hidden_dim = in_c * expand_ratio

        # 1. Aşama: 1x1 Genişletme (Expansion)
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()

        # 2. Aşama: 3x3 Depthwise Konvolüsyon
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        # 3. Aşama: Squeeze-and-Excitation (SE Kanal Dikkati)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )

        # 4. Aşama: 1x1 Doğrusal Daraltma (Linear Bottleneck)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = out * self.se(out)
        out = self.project(out)
        return (identity + out) if self.use_res else out
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Depthwise Separable Konvolüsyon teorik olarak $\%85 - \%90$ daha az FLOPs işlem gerektirmesine rağmen, bazı eski GPU veya TPU donanımlarında standart konvolüsyona göre beklenen hızlanmanın altında kalmasının (Arithmetic Intensity problemi) sebebi nedir?

> **Mentor Cevabı:**
> 1. **Aritmetik Yoğunluk (Arithmetic Intensity):** Aritmetik yoğunluk, belleğe erişilen her bayt başına yapılan matematiksel işlem sayısıdır ($\text{FLOPs} / \text{Memory Access Bytes}$). Standart konvolüsyon yüksek FLOPs içerdiği için GPU tensör çekirdeklerini yüksek yoğunlukla besler.
> 2. **Depthwise Konvolüsyonda Bellek Bant Genişliği Darboğazı (Memory Bound):** Depthwise aşamasında her kanal ayrı ayrı belleğe okunur ve çok az işlem yapılıp tekrar belleğe yazılır. Bu durum GPU'yu hesaplama odaklı (Compute-bound) olmaktan çıkarıp bellek bant genişliği odaklı (Memory-bandwidth bound) hale getirir.
> 3. **Çözüm:** Modern gömülü donanımlar (Apple Neural Engine, ARM Ethos, Jetson TensorRT) Depthwise işlemlerini SRAM üzerinde ara belleğe alarak (Kernel Fusion) bu darboğazı tamamen çözer ve $3\times - 8\times$ net hızlanma sağlar.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
