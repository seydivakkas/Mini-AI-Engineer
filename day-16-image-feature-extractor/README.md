# Day 16: Geleneksel Görsel Öznitelik Çıkarımı (Image Feature Extractor: SIFT, ORB, HOG, LBP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.24+-orange.svg?style=flat-square)](https://scikit-image.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; derin öğrenme (CNN ve Vision Transformer) öncesi klasik bilgisayarlı görünün temelini oluşturan ve günümüzde robotik SLAM (eşzamanlı konumlandırma ve haritalama), mobil artırılmış gerçeklik (AR), panoramik görüntü dikişi ve kenar yapay zeka (Edge AI) sistemlerinde yüksek hız ve kaynak verimliliği nedeniyle aktif olarak kullanılan **en temel 4 geleneksel öznitelik çıkarıcıyı (SIFT, ORB, HOG ve LBP)** kapsamlı matematiksel altyapıları, bellek analizleri ve çalışma hızı kıyaslamalarıyla sunar.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Görüntüleri pikselleriyle doğrudan karşılaştırmak (Ham RGB/Gri piksel karşılaştırması) feci bir hatadır:
- Kamera 1 derece döndüğünde veya nesne 1 metre uzaklaştığında piksel matrisinin matematiksel korelasyonu sıfıra yaklaşır.
- Ortam ışığı değiştiğinde piksel değerleri tamamen kayar.

Bu nedenle bilgisayarlı görü mühendisleri **öznitelik çıkarımı (Feature Extraction)** yöntemlerini geliştirmiştir. Amaç: **Döndürmeye (Rotation), Ölçeğe (Scale), Perspektife (Affine) ve Aydınlatma Değişimlerine (Illumination)** karşı bağışık, nesneyi ve dokuyu tekil olarak tanımlayan matematiksel parmak izleri (vektörler) üretmektir.

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Görsellerden renk histogramları, Haralick doku özellikleri ve kenar yoğunlukları çıkararak sayısal öznitelik vektörleri oluşturmak için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Büyük görüntü matrislerini makine öğrenimi modellerinin işleyebileceği kompakt 1D vektörlere dönüştürür.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- El yapımı (hand-crafted) özniteliklerdir; derin ağların öğrendiği hiyerarşik anlamsal özellikleri tam yakalayamaz.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- CNN Backbone öznitelikleri (ResNet, EfficientNet) veya Vision Transformer Gömüleri.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **SIFT** | *Scale-Invariant Feature Transform* | Görüntülerdeki ölçek ve dönme değişimlerine dayanıklı anahtar nokta tespiti ve 128 boyutlu gradyan histogram tanımlayıcısı üreten algoritma. |
| **ORB** | *Oriented FAST and Rotated BRIEF* | FAST köşe bulucu ile yönlendirilmiş BRIEF ikili tanımlayıcıyı birleştiren, patent kısıtı olmayan yüksek hızlı öznitelik çıkarıcı. |
| **Ölçek Uzayı (Scale Space)** | *Scale-Space Representation* | Görüntünün ardışık Gauss piramitleriyle küçültülüp bulanıklaştırılarak farklı ölçeklerdeki detayların analiz edilmesi. |
| **Öznitelik Eşleme** | *Feature Matching (Hamming / L2)* | İki görsel arasındaki benzer anahtar noktaların tanımlayıcı vektörleri üzerinden en yakın komşulukla eşleştirilmesi. |

---

## 2. Dört Büyük Algoritmanın Matematiksel ve Algoritmik Mantığı

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  GELENEKSEL GÖRSEL ÖZNİTELİK ÇIKARICILAR                     │
├───────────────┬───────────────────────────────┬─────────────────────────────┤
│ Algoritma     │ Temsil Türü                   │ Mesafe / Eşleştirme Metriği │
├───────────────┼───────────────────────────────┼─────────────────────────────┤
│ SIFT (1999)   │ Yerel Nokta (128-B Float)     │ Öklid / L2 Mesafesi         │
│ ORB (2011)    │ Yerel Nokta (256-Bit Binary)  │ Hamming Mesafesi (POPCNT)   │
│ HOG (2005)    │ Yoğun Gradyan (Dense Vector)  │ Cosine / L2 (SVM Girdisi)   │
│ LBP (2002)    │ Mikro Doku Histogramı         │ Chi-Square (Ki-Kare)        │
└───────────────┴───────────────────────────────┴─────────────────────────────┘
```

#### A. SIFT (Scale-Invariant Feature Transform - David Lowe)
Ölçek ve döndürmeden tamamen bağımsız çalışan, bilgisayarlı görünün en saygın algoritmasıdır:
1. **Ölçek Uzayı ve Gauss Farkı (Difference of Gaussians - DoG):**
   Görüntü ardışık Gauss çekirdekleriyle ($\sigma, k\sigma, k^2\sigma$) bulanıklaştırılır ve birbirinden çıkarılır:
   $$DoG(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y)$$
   DoG piramidinde $3 \times 3 \times 3$ (uzamsal ve ölçek komşuluğu) ekstremum araması ile ölçekten bağımsız anahtar noktalar tespit edilir.
2. **Kenar Yanıtlarının Elenmesi:**
   Hessian matrisi $H$ kullanılarak asal eğrilik oranı $\frac{\text{Tr}(H)^2}{\text{Det}(H)} < \frac{(r+1)^2}{r}$ kontrol edilir; zayıf kontrastlı ve kenar üzerindeki kararsız noktalar elenir.
3. **Baskın Yönelim Atama (Orientation Assignment):**
   Anahtar noktanın etrafındaki gradyan yönelimleri bir histogramda toplanır ve zirve yapan açı $\theta$ noktaya atanır (**Döndürme Bağımsızlığı**).
4. **128 Boyutlu Tanımlayıcı (Descriptor):**
   Nokta etrafındaki $16 \times 16$'lık bölge $4 \times 4$'lük 16 alt hücreye bölünür. Her hücrede 8 yönelimli gradyan histogramı hesaplanır:
   $$4 \times 4 \times 8 = \mathbf{128\text{-boyutlu float vektör}}$$
   Vektör L2 normuna bölünerek normalize edilir ve doyum/ışık kırılmalarını gidermek için $0.2$ tavan değerinde kırpılıp (clip) tekrar normalize edilir.

#### B. ORB (Oriented FAST and Rotated BRIEF - Rublee et al.)
SIFT'in patentli olduğu dönemde geliştirilen, robotik ve mobil cihazlarda **gerçek zamanlı (Real-Time)** çalışabilen patent-free algoritmadır:
1. **O-FAST (Oriented FAST):** 16 piksellik Bresenham çemberi üzerinde merkez pikselden belirgin şekilde parlak/karanlık ardışık $N=9$ piksel kontrol edilerek çok hızlı köşe noktaları bulunur.
2. **Yoğunluk Ağırlık Merkezi (Intensity Centroid):**
   Yamanın (patch) momentleri hesaplanır:
   $$m_{pq} = \sum_{x, y} x^p y^q I(x, y), \quad C = \left(\frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}}\right)$$
   Merkezden ağırlık merkezine uzanan vektör açısı: $\theta = \text{atan2}(m_{01}, m_{10})$.
3. **rBRIEF (Rotated BRIEF):**
   Önceden belirlenmiş piksel ikilileri $\theta$ açısına göre döndürülür ve ikili testler yapılır:
   $$\tau(p; x, y) = \begin{cases} 1 & \text{eğer } p(x) < p(y) \\ 0 & \text{diğer durumda} \end{cases}$$
   256 adet ikili test sonucunda **256-bitlik (32 byte)** ikili bir tanımlayıcı üretilir.
4. **Hamming Mesafesi Üstünlüğü:**
   İki ORB vektörünün mesafesi kayan nokta Öklid formülü yerine CPU'nun donanımsal `XOR` ve `POPCNT` (bit sayma) komutlarıyla mikrosaniyeler mertebesinde hesaplanır!

#### C. HOG (Histogram of Oriented Gradients - Dalal & Triggs)
İnsan, yaya ve araç tespiti için nesnenin dış hatlarını ve yapısal sınırlarını kodlayan global/yarı-yerel bir tanımlayıcıdır:
1. Görüntünün yatay $G_x$ ve dikey $G_y$ Sobel gradyanları hesaplanır.
2. Görüntü $8 \times 8$ piksellik hücrelere (cells) ayrılır; her hücrede $[0^\circ, 180^\circ]$ aralığında 9 kutulu gradyan yönelim histogramı oluşturulur (büyüklükle ağırlıklandırılır).
3. Komşu $2 \times 2$ hücreler (bloklar) birleştirilerek yerel aydınlatma değişimlerine karşı $L_2\text{-norm}$ ile normalize edilir.
4. Tüm bloklar tek bir uzun $1D$ özellik vektörü halinde uç uca eklenir (Örn: Destek Vektör Makineleri - SVM sınıflandırıcısı için).

#### D. LBP (Local Binary Patterns - Ojala et al.)
Yüz tanıma, ahşap/mermer/kumaş dokusu analizi ve sahtecilik (liveness detection) tespitinde kullanılan mikro doku betimleyicisidir:
1. Her merkez piksel $g_c$, etrafındaki $P$ komşusu ($g_p$) ile yarıçap $R$ üzerinde karşılaştırılır:
   $$s(g_p - g_c) = \begin{cases} 1 & g_p \ge g_c \\ 0 & g_p < g_c \end{cases}$$
   $$\text{LBP}_{P, R}(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^p$$
2. **Uniform LBP:** İkili dizilimde $0 \to 1$ veya $1 \to 0$ geçiş sayısı en fazla 2 olan desenlerdir (çizgi, köşe, benek gibi temel dokuları temsil eder). $P=8$ için olası 256 deseni 59 (veya 10 ana sınıf) kompakt kutuya indirger.
3. Monotonik gri tonlama ışık değişimlerinden kesinlikle etkilenmez.

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Mesafe Metriği Uyuşmazlığı:**
   SIFT tanımlayıcıları için **Öklid (L2) Normu** kullanılırken, ORB ikili tanımlayıcıları için kesinlikle **Hamming Mesafesi** kullanılmalıdır. ORB üzerinde L2 mesafesi koşturmak anlamsız sonuçlara ve ağır performans kayıplarına yol açar.
2. **Giriş Görüntüsü Kanalı:**
   SIFT, ORB, HOG ve LBP algoritmalarının tamamı tek kanallı gri tonlama (Grayscale) matrisler üzerinde çalışır. Renkli BGR matrisi doğrudan verildiğinde gradyan ve parlaklık tutarlılığı bozulur veya istisna fırlatılır.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
                          Girdi Gri Görüntü (H x W)
                                     │
      ┌──────────────────┬───────────┴───────────┬──────────────────┐
      ▼                  ▼                       ▼                  ▼
  ┌──────────┐     ┌───────────┐           ┌───────────┐      ┌───────────┐
  │   SIFT   │     │    ORB    │           │    HOG    │      │    LBP    │
  │ DoG & L2 │     │FAST&rBRIEF│           │ Sobel &   │      │ P=8, R=1  │
  │ Piramidi │     │  Hamming  │           │ Blok Norm │      │  Uniform  │
  └────┬─────┘     └─────┬─────┘           └─────┬─────┘      └─────┬─────┘
       │                 │                       │                  │
       ▼                 ▼                       ▼                  ▼
(N, 128) Float    (N, 32) UInt8           (D,) Float32        (10,) Doku
Tanımlayıcı       İkili Vektör            Şekil Vektörü       Histogramı
       │                 │                       │                  │
       └─────────────────┼───────────────────────┼──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │     OznitelikGorsellestirici      │
        │     4 Panelli Karşılaştırma       │
        └────────────────┬──────────────────┘
                         │
                         ▼
       [ciktilar/oznitelik_analiz_paneli.png]
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

1. **[`src/oznitelik_cikarici.py`](./src/oznitelik_cikarici.py):**
   - `GorselOznitelikCikarici`: SIFT, ORB, HOG ve LBP çıkarımlarını güvenli arabellek yönetimi ve çalışma süresi telemetrisi ile yürüten ana sınıf.
   - `OznitelikOzeti`: Algoritma performansını, bellek tüketimini ve çıktı boyutlarını saklayan veri yapısı.
2. **[`src/gorsellestirici.py`](./src/gorsellestirici.py):**
   - `OznitelikGorsellestirici`: SIFT ölçek/yönelim dairelerini, ORB FAST noktalarını, HOG gradyan yön haritasını ve LBP doku desenini 4 panelli Matplotlib çizelgesi olarak çizen motor.
3. **[`ana_akis.py`](./ana_akis.py):**
   - Kumaş dokusu, yıldız, daire ve baklava desenleri içeren zengin test görselini üreten, 4 algoritmayı koşturup kıyaslama tablosunu konsola basan yürütücü betik.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
==============================================================================
>>> AŞAMA 1: Zengin Geometrik ve Dokusal Test Görselinin Üretilmesi
==============================================================================
[+] Görüntü Çözünürlüğü         : 320 x 320 piksel (Tek Kanal - Gri)
[+] Test Sahnesi Öğeleri        : Kumaş mikro dokusu, sekizgen yıldız, baklava ve daireler

==============================================================================
>>> AŞAMA 2: SIFT, ORB, HOG ve LBP Algoritmalarının Koşturulması
==============================================================================

==============================================================================
>>> AŞAMA 3: Algoritmik Karşılaştırma ve Telemetri Raporu
==============================================================================
Algoritma  | Nokta Adedi  | Çıktı Boyutu     | Veri Tipi  | Bellek     | Süre (ms) 
------------------------------------------------------------------------------
SIFT       | 503          | (503, 128)       | float32    | 257,536 B  | 21.00     
ORB        | 411          | (411, 32)        | uint8      | 13,152 B   | 3.34      
HOG        | Yoğun (Dense) | (54756,)         | float32    | 219,024 B  | 184.33    
LBP        | Yoğun (Dense) | (10,)            | float32    | 40 B       | 16.27     
------------------------------------------------------------------------------

[+] Mühendislik Çıkarımları:
    * ORB, SIFT'e kıyasla yaklaşık 6.3x kat DAHA HIZLI çalıştı.
    * ORB tanımlayıcıları, SIFT'e göre 19.6x kat DAHA AZ RAM tüketti (Binary Hamming!).
    * HOG 54,756 boyutlu global şekil vektörü üretti (SVM sınıflandırıcıları için ideal).
    * LBP yalnızca 10 boyutlu kompakt bir doku histogramı ile yüzeyi özetledi.

==============================================================================
>>> AŞAMA 4: 4 Panelli Analiz Çizelgesinin Kaydedilmesi
==============================================================================
[V] Öznitelik analiz paneli kaydedildi: oznitelik_analiz_paneli.png
[V] Kayıt Konumu: day-16-image-feature-extractor/ciktilar/oznitelik_analiz_paneli.png

[V] Day 16: Geleneksel Görsel Öznitelik Çıkarımı başarıyla tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Hamming Mesafesi Tabanlı İkili ORB Eşleştirici (Brute-Force Hamming Matcher)**

İki farklı görüntü arasındaki nesneyi tanımak için çıkarılan ORB özniteliklerini eşleştirmemiz gerekir.

### Görev Tanımı:
[`src/oznitelik_cikarici.py`](./src/oznitelik_cikarici.py) sınıfına şu metodu eklemeni bekliyorum:

```python
@staticmethod
def orb_eslestir(
    tanimlayici_1: np.ndarray,
    tanimlayici_2: np.ndarray,
    maks_mesafe: int = 50
) -> List[Tuple[int, int, int]]:
    """İki görüntüden çıkarılan ORB tanımlayıcılarını Hamming mesafesiyle eşleştirir."""
```

### Beklenen Kurallar:
1. `cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)` kullanarak iki küme arasındaki karşılıklı en yakın eşleşmeleri bulmalıdır.
2. Hamming mesafesi `maks_mesafe` eşiğinden büyük olan zayıf eşleşmeleri filtrelemelidir.
3. Eşleşmeleri en düşük mesafeden en yükseğe doğru sıralı olarak döndürmelidir.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> SIFT algoritmasında anahtar nokta etrafındaki yama doğrudan piksel parlaklık değerleri olarak saklanmak yerine neden $4 \times 4 \times 8 = 128$ boyutlu bir **gradyan yönelim histogramına** dönüştürülür?  
> Ayrıca tanımlayıcı vektörünün L2 normalizasyonundan sonra **$0.2$ eşiğinde kırpılması (clipping)** ve tekrar normalize edilmesi aydınlatmadaki hangi fiziksel/kamera olgusunu telafi etmek içindir?

---

## 📂 Dizin Yapısı

```
day-16-image-feature-extractor/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (opencv-python, numpy, scikit-image, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve görsel üretim akışı
├── ciktilar/                   # Üretilen 4 panelli öznitelik karşılaştırma görseli
│   └── oznitelik_analiz_paneli.png
├── src/
│   ├── __init__.py
│   ├── oznitelik_cikarici.py   # SIFT, ORB, HOG ve LBP çıkarım motoru
│   └── gorsellestirici.py      # 4 panelli Matplotlib çizelge motoru
└── testler/
    └── test_oznitelikler.py    # 7 adet birim testi (7 passed in 1.96s)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışı Çalıştırma
```bash
python ana_akis.py
```

### 3. Testleri Koşma
```bash
python -m pytest testler/test_oznitelikler.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
