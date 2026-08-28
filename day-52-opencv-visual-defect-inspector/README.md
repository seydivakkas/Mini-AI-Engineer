# Day 52: OpenCV ile Kural Tabanlı Görsel Kusur & Bulanıklık Tespiti (Rule-Based Visual Defect & Blur Inspector)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 52. gününde geliştirilen **Otomatik Optik Teftiş (Automated Optical Inspection - AOI) ve Bulanıklık Tespit Motorudur**. Yüksek hızlı endüstriyel üretim hatlarında ve konveyör bantlarda ağır derin öğrenme modellerine gerek kalmadan $(<10\text{ ms})$ Laplacian varyansı, 2D FFT frekans spektrumu, morfolojik Top-Hat/Black-Hat filtreleri ve kontur geometrisi ile yüzey kusurlarını (yağ lekesi, iplik çekiği, çizik, delik) ayrıştırır.

---

## 📖 Mentorluk Dersi ve Kural Tabanlı Görü Teorisı

### 1. Neden Kural Tabanlı ve Frekans Analizi Kullanılır?

Endüstriyel üretim hatlarında saniyede 30-60 kare işleyen yüksek hızlı kameralarda her kareyi ağır bir YOLO veya Segmentation modeline göndermek GPU maliyetini ve gecikmeyi (latency) artırır. **Kural tabanlı ön filtreleme (Rule-based Pre-filter)** şu avantajları sağlar:

1. **Laplacian Varyansı ile Odak/Bulanıklık Tespiti (Focus Measure):**
   - Net bir görüntüde pikseller arası geçişler (kenarlar) diktir ve ikinci dereceden türev varyansı yüksektir ($FM > 150$).
   - Kamera lensi kirlendiğinde veya odak bozulduğunda kenarlar yumuşar ve varyans dramatik biçimde düşer ($FM < 60$).

2. **2D Hızlı Fourier Dönüşümü (FFT) Frekans Analizi:**
   - Uzamsal görüntüyü ($I(x,y)$) 2D frekans uzayına ($F(u,v)$) dönüştürür.
   - Merkezdeki düşük frekanslı arka plan enerjisi maskelenerek **Yüksek Frekans Enerji Oranı (HFR)** hesaplanır. Net desenlerde yüksek frekans enerjisi $\%15$'in üzerindeyken, bulanık görsellerde $\%3$'ün altına iner.

3. **Morfolojik Top-Hat & Black-Hat Kusur Ayrıştırması:**
   - Dokulu tekstil veya halı zeminlerinde homojen olmayan aydınlatmayı bastırmak için:
     - **Top-Hat:** Açma (Opening) operasyonundan farkı alarak zemin üzerindeki parlak iplik çekiklerini ve beyaz çizikleri yakalar.
     - **Black-Hat:** Kapama (Closing) operasyonundan farkı alarak zemin üzerindeki koyu yağ lekelerini, delikleri ve yanıkları yakalar.

```
                           ┌──────────────────────────────────────────────────────────┐
                           │          CANLI KAMERA / TEKSTİL GÖRSELİ                  │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      BulaniklikAnalizoru (Laplacian Varyansı & 2D FFT)                            │
    │  - Laplacian İkinci Dereceden Türev Haritası ve Varyans Skoru Hesaplanır                          │
    │  - 2D FFT ile Yüksek Frekans Güç Spektrumu (HFR) Denetlenir -> Bulanık ise Erken Red              │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      MorfolojikKusurDedektoru (Top-Hat / Black-Hat & Kontur Analizi)              │
    │  - Top-Hat (Parlak Çizik) + Black-Hat (Koyu Leke) Birleştirilir                                   │
    │  - İkili Maske -> Kontur Geometrisi (Dairesellik & En-Boy Oranı) ile Sınıflandırma                │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 6-PANELLİ AOI KUSUR TEFTİŞ VE BULANIKLIK PANELİ (Day 52)                          │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **CLAHE** | *Contrast Limited Adaptive Histogram Equalization* | Görseli küçük bloklara bölerek yerel kontrastı artıran ve gürültü patlamasını sınırlandıran adaptif histogram eşitleme. |
| **Canny Kenar Dedektörü** | *Canny Edge Detector* | Gradyan yoğunluğu, histerezis eşiklemesi ve maksimum olmayan bastırma ile ince ve kesintisiz kenarlar bulan algoritma. |
| **Kontur Hiyerarşisi** | *Contour Hierarchy (`RETR_TREE`)* | İç içe geçmiş nesne sınırlarının ebeveyn-çocuk ilişkilerini çıkararak delik ve lekeleri sınıflandırma. |
| **Kusur Şiddet Skoru** | *Defect Severity Scoring* | Kusurun alanı, kontrast farkı ve kenar yoğunluğunu birleştirerek üretilen kural tabanlı kalite puanı. |

---

## 2. Matematiksel Formülasyonlar

#### A. Laplacian Operatörü ve Varyans Odak Ölçütü
$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
$$\text{Focus Measure (FM)} = \text{Var}(\nabla^2 I) = \frac{1}{N} \sum_{x,y} \left( \nabla^2 I(x,y) - \mu_{\nabla^2 I} \right)^2$$

#### B. 2D Ayrık Fourier Dönüşümü (2D DFT/FFT)
$$F(u,v) = \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} I(x,y) \cdot e^{-i 2\pi \left( \frac{ux}{H} + \frac{vy}{W} \right)}$$
$$\text{High-Frequency Ratio (HFR)} = \frac{\sum_{(u,v) \in \text{HighFreq}} |F(u,v)|^2}{\sum_{(u,v)} |F(u,v)|^2} \times 100$$

#### C. Kontur Dairesellik (Circularity / Form Factor)
$$\text{Circularity} = \frac{4\pi \cdot \text{Alan}}{\text{Çevre}^2} \quad (\text{Daire: } 1.0, \quad \text{İnce Çizik: } < 0.3)$$

---

## 🛠️ Dizin Yapısı

```
day-52-opencv-visual-defect-inspector/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # opencv-python, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca sentetik doku, bulanıklık ve kusur analiz betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── bulaniklik_analizoru.py      # BulaniklikAnalizoru (Laplacian Varyansı, 2D FFT & Tenengrad)
│   ├── kusur_tespit_motoru.py       # MorfolojikKusurDedektoru (Top-Hat/Black-Hat, Kontur Geometrisi)
│   └── gorsellestirici.py           # 6-Panelli Teşhis Panosu (AOI Defect Inspector Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_defect_inspector.py     # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── kusur_teftis_paneli.png      # 6 panelli yüksek çözünürlüklü teşhis panosu
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

## 📊 Kusur Türleri ve Geometrik Sınıflandırma Tablosu

| Kusur Türü | Tip Kodu | Geometrik Kriter | Renk Kodu | Tipik Endüstriyel Sebep |
|---|---|---|---|---|
| **Leke / Delik** | `LEKE_DELIK` | $\text{Dairesellik} > 0.55$ | Kırmızı (`#e74c3c`) | Makine Yağ Damlaması, İğne Kırılması |
| **İplik Çekiği / Çizik** | `CIZIK_IPLIK_CEKIGI` | $\text{En-Boy Oranı} > 2.5 \text{ veya } < 0.4$ | Turuncu (`#e67e22`) | Çözgü Teli Kopması, Mekik Çizmesi |
| **Yüzey Anomalisi** | `YUZEY_ANOMALISI` | Düzensiz Kontur Geometrisi | Mor (`#9b59b6`) | Düzensiz Baskı, Hav Yığılması |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Görüntüyü $N \times M$ ızgara bölgelerine (grid patches / ROIs) bölerek lokal bulanıklık veya bölgesel dokuma bozulmalarını tespit eden bir **"Multi-ROI Local Focus Scanner"** geliştirmek.

**Tamamlanan Çözüm:**
```python
def bolgesel_odak_taramasi(gri_img: np.ndarray, izgara: tuple = (4, 4)) -> np.ndarray:
    """Görseli ızgaralara bölerek her bölgenin bağımsız Laplacian varyans haritasını çıkarır."""
    h, w = gri_img.shape
    r_h, r_w = h // izgara[0], w // izgara[1]
    skor_matrisi = np.zeros(izgara, dtype=float)

    for i in range(izgara[0]):
        for j in range(izgara[1]):
            parca = gri_img[i * r_h : (i + 1) * r_h, j * r_w : (j + 1) * r_w]
            skor_matrisi[i, j] = round(float(cv2.Laplacian(parca, cv2.CV_64F).var()), 1)

    return skor_matrisi
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Yüksek hızlı bir endüstriyel konveyör bant üzerinde hareket eden dokuma ürünlerde, standart Laplacian varyansı bazen hareket bulanıklığı (motion blur) ile kamera odak bulanıklığını (defocus blur) ayırt edemez. **2D FFT Frekans Spektrumu** bu iki bulanıklık türünü yönsel (directional) olarak nasıl kesin şekilde ayrıştırır?

> **Mentor Cevabı:**
> 1. **Frekans Spektrumunun Yönsel Özelliği:** 2D FFT uzamsal frekansları 360 derece yönsel olarak ayrıştırır. Kamera optik odak bulanıklığında (defocus), frekans enerjisi tüm yönlerde dairesel ve izotropik olarak zayıflar (merkezde dairesel parlama kalır).
> 2. **Hareket Bulanıklığı (Motion Blur) İmzası:** Eğer bant yatay hareket ederken bulanıklık oluşursa, yatay frekanslar çökerken dikey kenar frekansları korunur. FFT spektrumunda tek bir eksene dik uzanan belirgin çizgisel (anizotropik sinc) şeritler oluşur. Bu sayede FFT spektrumu yön analiziyle hareket hızı ayarsızlığı ile lens kirliliği kesin olarak ayırt edilir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
