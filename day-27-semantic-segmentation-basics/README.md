# Day 27: Anlamsal Bölütleme Temelleri (Semantic Segmentation Basics — U-Net)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; piksel düzeyinde sınıflandırma (Pixel-Wise Classification) problemi olan **Anlamsal Bölütleme (Semantic Segmentation)** görevini sıfırdan ele alır. **U-Net Mimarisi (Encoder-Decoder & Skip Connections)**, **Piksel Düzeyinde Cross-Entropy + Soft Dice Hibrit Kaybı (Combo Loss)**, **Mean IoU (Jaccard İndeksi)** ve **Piksel Hata Haritası Analizini (Error Heatmap)** kapsayan 6 panelli endüstri standardı bir teşhis panosu (Diagnostic Dashboard) sunar.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Görsel Sınıflandırma vs Nesne Tespiti vs Anlamsal Bölütleme
Görsel sınıflandırma *"Görselde ne var?"*, nesne tespiti *"Nesneler nerede ve boyutları ne?"* sorularına cevap ararken; **Anlamsal Bölütleme (Semantic Segmentation)** görseldeki her bir piksele $(i, j)$ bağımsız sınıf etiketi atar ($f: \mathbb{R}^{H \times W \times C} \to \{0, \dots, K-1\}^{H \times W}$). 

- **Görsel Sınıflandırma:** Tekil etiket ($y \in \{1, \dots, C\}$).
- **Nesne Tespiti:** Sınırlayıcı kutular (Bounding Box: $[x, y, w, h, c]$).
- **Anlamsal Bölütleme:** Her piksel için sınıf matrisi. Medikal lezyon sınırları, otonom araçlarda yol/şerit/yaya piksel sınırları ve biyomedikal hücre analizlerinde milimetrik sınır hassasiyeti sağlar.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │               GİRİŞ GÖRSELİ (128x128x3)                  │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
       [MİMARİ & AKTARIM]                                              [KAYIP & METRİKLER]
 ┌─────────────────────────────┐                                ┌───────────────────────────────────┐
 │ 1. U-Net Encoder            │                                │ 4. Dice Loss (Sınıf Dengesizliği) │
 │    - MaxPool ile Çözünürlük │                                │    - 1 - (2|Y∩Y^| / (|Y|+|Y^|))   │
 │ 2. U-Net Decoder            │                                │ 5. Combo Loss (CE + Dice)         │
 │    - ConvTranspose2D Upsam. │                                │    - α * CE + (1-α) * Dice        │
 │ 3. Atlama Bağlantıları      │                                │ 6. Mean IoU (Jaccard İndeksi)     │
 │    - Skip Connections (Cat) │                                │    - TP / (TP + FP + FN)          │
 └─────────────────────────────┘                                └───────────────────────────────────┘
```

---

### 2. U-Net Mimari Anatomisi (Ronneberger et al., 2015)

```
Giriş: (3, H, W)
  │
  ├── [Encoder 1: 32 Kanal]  ──────────────────────── (Skip 1) ───────────► [Decoder 1: 32 Kanal] ──► Çıkış: (K, H, W)
  │        ▼ MaxPool(2)                                                            ▲ ConvTranspose(2)
  ├── [Encoder 2: 64 Kanal]  ──────────────── (Skip 2) ───────────────────► [Decoder 2: 64 Kanal]
  │        ▼ MaxPool(2)                                                            ▲ ConvTranspose(2)
  ├── [Encoder 3: 128 Kanal] ──────── (Skip 3) ───────────────────────────► [Decoder 3: 128 Kanal]
  │        ▼ MaxPool(2)                                                            ▲ ConvTranspose(2)
  └── [Encoder 4: 256 Kanal] ── (Skip 4) ─────────────────────────────────► [Decoder 4: 256 Kanal]
           ▼ MaxPool(2)                                                            ▲ ConvTranspose(2)
      [Darboğaz (Bottleneck): 512 Kanal] ──────────────────────────────────────────┘
```

#### Neden Atlama Bağlantıları (Skip Connections) Hayatidir?
1. **Daralan Yol (Encoder / Contracting Path):** Evrişim (Conv2D) + ReLU + MaxPool2D katmanları ile mekansal çözünürlük düşürülür ($H, W \to H/2, W/2$). Amacı zengin anlamsal öznitelikler ve geniş bağlam (**"Ne var? / What"**) öğrenmektir.
2. **Genişleyen Yol (Decoder / Expansive Path):** Yukarı Evrişim (ConvTranspose2D veya Bilinear Upsample) ile çözünürlük artırılır. Amacı konum bilgisini (**"Nerede? / Where"**) yeniden üretmektir.
3. **Skip Connections:** Encoder katmanlarındaki yüksek çözünürlüklü düşük seviyeli kenar ve doku detaylarını doğrudan Decoder'daki ilgili katmana kanal bazında ekler (`torch.cat`). Downsampling sırasında kaybolan keskin sınırların eksiksiz kurtarılmasını sağlar.

---

### 3. Kayıp Fonksiyonları & Sınıf Dengesizliği Çözümü

#### A. Piksel Cross-Entropy Kaybı
$$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^K y_{i,c} \log(\hat{p}_{i,c})$$
*Dezavantaj:* Arka plan pikselleri görselin $\%90$'ını kapladığında model küçük nesneleri/hücreleri yok sayar.

#### B. Çok Sınıflı Soft Dice Kaybı
$$\text{Dice}_c = \frac{2 \sum_i p_{i,c} y_{i,c} + \epsilon}{\sum_i p_{i,c} + \sum_i y_{i,c} + \epsilon}, \quad \mathcal{L}_{\text{Dice}} = 1 - \frac{1}{K} \sum_{c=1}^K \text{Dice}_c$$
*Avantaj:* Piksel sayısından bağımsız kesişim oranına odaklandığı için dengesiz maskelerde mükemmeldir.

#### C. Hibrit Kayıp (Combo Loss)
$$\mathcal{L}_{\text{Combo}} = \alpha \mathcal{L}_{\text{CE}} + (1 - \alpha) \mathcal{L}_{\text{Dice}}$$

---

### 4. Değerlendirme Metrikleri: Mean IoU (mIoU) & Dice Katsayısı

$$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \quad \text{mIoU} = \frac{1}{K} \sum_{c=1}^K \text{IoU}_c$$

$$\text{Dice}_c = \frac{2 TP_c}{2 TP_c + FP_c + FN_c} = \frac{2 \text{IoU}_c}{1 + \text{IoU}_c}$$

---

## 📊 Deneysel Sonuçlar ve Metrik Tablosu

Sistem `ana_akis.py` üzerinden 5 epok eğitildiğinde elde edilen metrikler:

```
================================================================================
>>> U-Net Eğitim İlerlemesi (Combo Loss = 0.5*CE + 0.5*Dice)
================================================================================
[*] Epok [01/05] | Train Loss: 0.6339 | Val Loss: 0.8468 | Val mIoU: %25.94 | Pix Acc: %73.89
[*] Epok [02/05] | Train Loss: 0.4106 | Val Loss: 0.7118 | Val mIoU: %29.45 | Pix Acc: %75.97
[*] Epok [03/05] | Train Loss: 0.3558 | Val Loss: 0.6145 | Val mIoU: %51.46 | Pix Acc: %89.65
[*] Epok [04/05] | Train Loss: 0.3237 | Val Loss: 0.5293 | Val mIoU: %55.11 | Pix Acc: %91.76
[*] Epok [05/05] | Train Loss: 0.2912 | Val Loss: 0.4428 | Val mIoU: %72.25 | Pix Acc: %95.26
```

### Sınıf Bazında Performans Dağılımı

| Sınıf Adı | IoU (Jaccard İndeksi) | Dice (F1-Score) |
|---|---|---|
| **Arka Plan** | %98.6 | %99.3 |
| **Hücre Gövdesi** | %79.9 | %88.8 |
| **Çekirdek (Nucleus)** | %38.3 | %55.1 |
| **GENEL ORTALAMA** | **%72.25 (mIoU)** | **%81.07 (mDice)** |

- **Piksel Uyuşmazlık (Hata) Oranı:** Örnek test görseli üzerinde yalnızca **%4.44**.

---

## 🛠️ Dizin Yapısı

```
day-27-semantic-segmentation-basics/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, numpy, seaborn, matplotlib, pytest
├── ana_akis.py                      # Uçtan uca veri, U-Net eğitimi ve teşhis akışı
├── README.md                        # Detaylı teorik ve mentorluk dokümantasyonu
├── src/
│   ├── __init__.py
│   ├── unet_modeli.py               # U-Net mimarisi (Encoder, Decoder, Skip Connections)
│   ├── kayip_ve_metrikler.py        # Dice Loss, Combo Loss, mIoU ve Pixel Acc motoru
│   ├── sentetik_veri_yoneticisi.py  # Sentetik hücresel doku veri seti & DataLoader
│   ├── egitici.py                   # PyTorch eğitim ve validasyon döngüsü yöneticisi
│   └── gorsellestirici.py           # 6 panelli görselleştirme panosu çizici
├── testler/
│   ├── __init__.py
│   └── test_bolutleme.py            # 6 adet kapsamlı birim test
└── ciktilar/
    └── bolutleme_teshis_paneli.png  # 6 panelli teşhis panosu görseli
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

### 3. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/kayip_ve_metrikler.py` içinde sınıf ağırlıklı Combo Loss (`WeightedComboLoss`) mekanizmasını tanımlayıp nadir sınıfların (ör. Çekirdek) kaybını artırmak.

**Çözüm:**
```python
class WeightedComboLoss(nn.Module):
    def __init__(self, class_weights=[0.1, 0.3, 0.6], alpha=0.5):
        super().__init__()
        self.weights = torch.tensor(class_weights)
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=self.weights)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return self.alpha * self.ce(inputs, targets) + (1 - self.alpha) * self.dice(inputs, targets)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Medikal bir görüntüleme veri setinde (örneğin $\%97$ sağlıklı doku, $\%3$ tümör pikselleri) yalnızca Cross-Entropy kaybı kullanarak eğitilen bir U-Net modeli neden tümör sınırlarını öğrenemez? Soft Dice kaybı veya Combo Loss kullanmak bu sorunu nasıl çözer ve U-Net mimarisindeki Atlama Bağlantıları (Skip Connections) kaldırılırsa segmentasyon maskelerinde ne tür bir bozulma gözlenir?

> **Cevap:**
> 1. **Sınıf Dengesizliği & Soft Dice / Combo Loss Çözümü:** Piksel düzeyindeki Cross-Entropy kaybı ($\mathcal{L}_{\text{CE}}$) her pikseli eşit ağırlıkla değerlendirir. Piksellerin $\%97$'si sağlıklı doku olduğunda, model tüm piksellere "sağlıklı" diyerek kaybı $\%97$ oranında düşürebilir ve gradient sinyali küçük tümör alanına ulaşmaz. **Soft Dice kaybı ($\mathcal{L}_{\text{Dice}}$)** ise tekil pikseller yerine doğrudan kümülatif kesişim kümesine ($\frac{2 |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}$) odaklanır. Tümör alanı ne kadar küçük olursa olsun, kesişimdeki tek bir eksiklik kayıpta devasa bir sıçramaya sebep olur. **Combo Loss ($\alpha \mathcal{L}_{\text{CE}} + (1-\alpha) \mathcal{L}_{\text{Dice}}$)** ise hem piksel düzeyinde olasılık kalibrasyonunu korur hem de küçük nesnelerin kaçırılmasını engeller.
> 2. **Atlama Bağlantılarının (Skip Connections) Kritik Rolü:** Encoder katmanında yapılan her MaxPool işlemi çözünürlüğü yarıya düşürerek yüksek frekanslı ince kenar ve doku detaylarını kaybettirir. Eğer Skip Connections kaldırılırsa (Autoencoder mimarisine dönüşürse), Decoder katmanı orijinal boyuta dönerken kenar sınırlarını hassas çizen düşük seviyeli özelliklere erişemez. Sonuç olarak tahmin edilen segmentasyon maskelerinin nesne sınırları **bulanıklaşır (blurry boundaries)** ve ince yapılar (damarlar, çekirdek zarları) tamamen silinir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
