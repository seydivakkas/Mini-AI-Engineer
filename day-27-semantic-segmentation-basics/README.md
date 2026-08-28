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
- **Sınıflandırma:** Görselin tamamı için tek bir etiket ($y \in \{1, \dots, C\}$).
- **Nesne Tespiti:** Nesnelerin etrafına dikdörtgen kutu çizme ($[x, y, w, h]$).
- **Anlamsal Bölütleme:** Görseldeki her bir $(i, j)$ pikseline bağımsız sınıf atama ($f: \mathbb{R}^{H \times W \times C} \to \{0, \dots, K-1\}^{H \times W}$). Medikal lezyon sınırları, otonom araç yol/şerit segmentasyonu ve biyomedikal hücre analizinde milimetrik sınır hassasiyeti sağlar.

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
- Encoder (Daralan Yol), mekansal çözünürlüğü azaltarak yüksek seviyeli anlamsal bağlamı (**"Ne var? / What"**) öğrenir; ancak piksel koordinat konumlarını kaybeder.
- Decoder (Genişleyen Yol), çözünürlüğü artırarak konum bilgisini (**"Nerede? / Where"**) yeniden üretir.
- **Skip Connections**, Encoder'daki ham kenar ve doku detaylarını doğrudan Decoder'daki ilgili katmana kanal bazında ekleyerek (`torch.cat`), downsampling sırasında kaybolan keskin sınırların eksiksiz kurtarılmasını sağlar.

---

### 3. Kayıp Fonksiyonları & Sınıf Dengesizliği Çözümü

#### A. Piksel Cross-Entropy Kaybı
$$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^K y_{i,c} \log(\hat{p}_{i,c})$$
*Dezavantaj:* Arka plan pikselleri görselin $\%90$'ını kapladığında model küçük nesneleri yok sayar.

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

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
