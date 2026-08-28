# Day 14: Görsellerdeki Desen ve Motiflerin Ayrıştırılması (Motif Segmentation & Contour Analytics)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; geleneksel Türk halı/kilimleri, çinileri, tekstil kumaşları ve endüstriyel baskılı devre kartlarındaki (PCB) karmaşık desenleri **istatistiksel Otsu ikili eşiklemesi, morfolojik gürültü filtreleme, Suzuki-Abe kontur topolojisi, eksenel ve döndürülmüş sınırlayıcı kutular (Bounding Boxes) ile şekil analitiği (Dairesellik, Solidity, En/Boy oranı)** uygulayarak bağımsız münferit nesneler halinde ayrıştıran üretim kalitesinde bir motif izolasyon motorudur.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Bir halı veya seramik fabrikasında, tekstil tasarım stüdyosunda veya dijital arşivlerde:
- Bir görselin içinde onlarca farklı yıldız, çiçek, rozet, madalyon veya sembolik motif bulunur.
- Bu motiflerin tek tek taranıp **vektörize edilmesi, e-ticaret için kataloglanması, kusur denetiminden geçirilmesi veya benzer motif arama motoruna girdi verilmesi** gerekir.
- Elle tek tek motif kırpmak binlerce görselde imkansızdır.

**Kullanım Alanları:**
- **Kültürel Miras ve Müze Envanteri:** Anadolu kilimlerindeki motifleri (elibelinde, koçboynuzu, bereket) otomatik kırpıp etiketlemek.
- **Tekstil Kalite Kontrolü:** Kumaş baskısındaki desen kaymalarını, kopuklukları ve eksik basılan motifleri tespit etmek.
- **Biyomedikal Görüntüleme:** Mikroskop altındaki hücreleri, bakterileri veya kan damarlarını tekil konturlarla saymak ve şekil indekslerini ölçmek.

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Görseldeki özel desen ve motifleri renk eşikleme ve morfolojik kontur operasyonları ile arka plandan izole etmek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Karmaşık arka planlardan hedef motiflerin etiketli veri gerektirmeden kural tabanlı olarak çıkarılmasını sağlar.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Benzer renkteki arka plan detaylarını motife dahil edebilir; aydınlatma değişimlerine karşı hassastır.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- U-Net Segmentasyonu, Watershed veya GrabCut.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Otsu Eşikleme** | *Otsu's Thresholding* | Görüntüdeki sınıflar arası varyansı maksimize ederek optimum ikili (binary) eşik değerini otomatik bulan algoritma. |
| **Morfolojik Açma (Opening)** | *Morphological Opening (Erosion + Dilation)* | Küçük beyaz gürültü noktalarını yok ederken ana nesne boyutunu koruyan morfolojik filtre. |
| **Morfolojik Kapama (Closing)** | *Morphological Closing (Dilation + Erosion)* | Nesneler içindeki küçük delikleri ve çatlakları dolduran morfolojik filtre. |
| **Bağlantılı Bileşenler** | *Connected Component Labeling* | İkili maske üzerindeki birbirine komşu beyaz pikselleri gruplayarak bağımsız motif nesneleri çıkaran algoritma. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. Otsu Eşikleme Algoritması (Optimal Binarization)
Görüntüdeki arka plan ve ön plan piksellerini ayıracak sabit bir eşik seçmek yerine, histogramı inceleyerek **sınıflar arası varyansı ($\sigma_b^2$) maksimize eden** optimal $T^*$ eşiğini otomatik bulur:

$$\sigma_b^2(t) = \omega_0(t) \cdot \omega_1(t) \cdot \left(\mu_0(t) - \mu_1(t)\right)^2$$

- $\omega_0(t), \omega_1(t)$: $t$ eşiğinin altındaki (arka plan) ve üstündeki (ön plan) piksellerin kümülatif olasılıkları.
- $\mu_0(t), \mu_1(t)$: İki sınıfın ortalama gri ton değerleri.
- **Optimal Eşik:**
  $$T^* = \arg\max_{0 \le t < 256} \sigma_b^2(t)$$
Bu yöntem hiçbir kullanıcı parametresi gerektirmeden en temiz kontrast ayrımını matematiksel olarak garanti eder.

#### B. Morfolojik Açma (Opening) ve Kapatma (Closing)
İkili maske üretildikten sonra:
1. **Açma (Opening - $A \circ B$):** Önce aşınma (erosion) sonra genişleme (dilation). Küçük parazit beyaz noktaları (tuz gürültüsünü) siler.
2. **Kapatma (Closing - $A \bullet B$):** Önce genişleme sonra aşınma. Motif içindeki mikro delikleri kapatır ve kenarları birleştirir.

#### C. Suzuki-Abe Kontur Takip Topolojisi (`RETR_EXTERNAL`)
İkili görüntüdeki sınır pikselleri yönlü kenar takibi ile poligon zincirlerine (`contours`) dönüştürülür.
`cv2.RETR_EXTERNAL` bayrağı yalnızca en dış sınırları toplar; böylece motifin içindeki desenler veya delikler ayrı birer motif sanılmaz.

#### D. Şekil Deskriptörleri ve Matematiksel Metrikler

1. **Dairesellik / Yuvarlaklık (Circularity - $C$):**
   $$C = \frac{4 \pi \cdot \text{Alan}}{\text{Çevre}^2}$$
   - Kusursuz bir çemberde $C = 1.0$'dır.
   - Şekil uzadıkça veya köşeli/yıldız gibi girintili hale geldikçe $C \to 0$'a yaklaşır.

2. **Doluluk Oranı (Solidity / Konvekslik Oranı - $S$):**
   $$S = \frac{\text{Alan}(C)}{\text{Alan}(\text{ConvexHull}(C))}$$
   - $\text{ConvexHull}(C)$: Konturu saran en küçük dışbükey lastik bant (gövde).
   - $S \approx 1.0$: Şekil dolgun ve dışbükeydir (Daire, kare, elips).
   - $S < 0.85$: Şeklin derin girintileri ve çıkıntıları vardır (Sekiz köşeli yıldız veya elibelinde motifi!).

3. **Sınırlayıcı Kutular (Bounding Boxes):**
   - **Düz Eksenli Kutu (AABB):** `(x, y, w, h) = cv2.boundingRect(cnt)`
   - **Döndürülmüş Minimum Alanlı Kutu (OBB):** Açılı duran motifler için minimum alanı kapsayan `cv2.minAreaRect(cnt)` ve 4 köşe noktası `cv2.boxPoints`.

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Çift Tepeli (Bimodal) Olmayan Histogramlar:**
   Otsu algoritması arka plan ile ön planın iki tepe oluşturduğu görüntülerde kusursuzdur. Eğer sahnede degrade (aydınlatma gradyanı) varsa, Otsu global tek bir eşikle görüntünün yarısını karartabilir (Böyle durumlarda Adaptif Eşikleme gerekir).
2. **Alan Eşiklerinin Belirlenmesi:**
   Görüntüdeki 5-10 piksellik gürültülerin veya görüntünün tamamını kaplayan çerçevenin motif sanılmaması için `min_alan` ve `maks_alan_orani` dinamik filtreleri zorunludur.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
                    Girdi Görüntüsü (Çini / Kilim / Kumaş)
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │  Gri Seviye + Gauss Yumuşatma   │
                    │  (Yüksek Frekans Gürültü Süzme) │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │      Otsu İkili Eşikleme        │
                    │     (Varyans Maksimizasyonu)    │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │     Morfolojik Temizleme        │
                    │   Açma (Gürültü Sil) + Kapatma  │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   Dış Kontur Tespiti & Filtre   │
                    │ (Alan Süzgeci: min < Alan < max)│
                    └────────────────┬────────────────┘
                                     │
        ┌────────────────────────────┴────────────────────────────┐
        ▼                                                         ▼
[Şekil Analitiği Hesaplama]                               [Görsel İzolasyon]
- Alan (Area) & Çevre (ArcLength)                         - Kırpılmış Görsel (ROI)
- Dairesellik: 4π*Alan / Çevre^2                          - Kırpılmış İkili Maske
- Solidity: Alan / ConvexHull_Alanı                       - Bounding Boxes (Düz & Açılı)
- Ağırlık Merkezi (Moments)
        │                                                         │
        └────────────────────────────┬────────────────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │      MotifGorsellestirici       │
                    │  - Panel 1: Orijinal Görsel     │
                    │  - Panel 2: Otsu Maskesi        │
                    │  - Panel 3: Kutulu Tespitler    │
                    │  - Panel 4: Kırpılmış Galeri    │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                   [ciktilar/motif_segmentasyon_paneli.png]
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

1. **[`src/motif_ayristirici.py`](./src/motif_ayristirici.py):**
   - `MotifBilgisi`: Motif geometrisi, kutuları, dairesellik/solidity skorları ve kırpılmış ROI matrislerini tutan veri sınıfı.
   - `MotifAyristirici`: Otsu eşikleme, morfolojik temizleme, dış kontur tespiti ve şekil analitiği motoru.
2. **[`src/gorsellestirici.py`](./src/gorsellestirici.py):**
   - `MotifGorsellestirici`: 4 panelli analiz raporunu ve alt satırda kırpılmış münferit motif galerisini diske çizen modül.
3. **[`ana_akis.py`](./ana_akis.py):**
   - 5 farklı geleneksel geometrik motif içeren sentetik Anadolu çini sahnesini üreten ve analizi yürüten konsol betiği.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
============================================================================
>>> AŞAMA 1: Sentetik Motif Sahnesinin Oluşturulması
============================================================================
[+] Sahne Çözünürlüğü         : 420 x 420 piksel (3 kanal)
[+] Sahne İçeriği             : 5 Farklı Geometrik Anadolu Motifi
    * Yıldız Madalyon (Merkez)
    * Baklava Eşkenar Dörtgen (Sol-Üst)
    * Dairesel Rozet (Sağ-Üst)
    * Geometrik Üçgen (Sol-Alt)
    * Açılı Elips Rozet (Sağ-Alt)

============================================================================
>>> AŞAMA 2: Otsu Eşikleme ve Morfolojik Filtreleme
============================================================================
[V] Otsu Optimal Eşik Değeri (T*): 103.0

============================================================================
>>> AŞAMA 3: Kontur Ayrıştırma ve Şekil Analitiği
============================================================================
[V] Tespit Edilen Geçerli Motif Adedi: 4
----------------------------------------------------------------------------
ID    | Alan (px)  | Çevre    | Dairesellik  | Solidity   | Sınırlayıcı Kutu (x,y,w,h)
----------------------------------------------------------------------------
M-3   | 9540       | 419.6    | 0.681        | 0.836      | (147, 147, 127, 127)
M-1   | 6048       | 301.9    | 0.834        | 0.989      | (278, 281, 105, 79)
M-4   | 6048       | 291.1    | 0.897        | 0.988      | (286, 46, 89, 89)
M-2   | 5096       | 339.2    | 0.557        | 0.991      | (40, 271, 100, 100)
----------------------------------------------------------------------------
[+] En Yüksek Dairesellik : Motif M-4 (Skor: 0.897 - Dairesel Rozet!)
[+] En Girintili Motif    : Motif M-3 (Solidity: 0.836 - Köşeli Yıldız!)

============================================================================
>>> AŞAMA 4: 4 Panelli Görsel Raporun ve Kırpılmış Galerinin Kaydedilmesi
============================================================================
[V] Motif analiz paneli başarıyla kaydedildi: motif_segmentasyon_paneli.png
[V] Kayıt Konumu: day-14-motif-segmentation/ciktilar/motif_segmentasyon_paneli.png

[V] Day 14: Görsellerdeki Desen ve Motiflerin Ayrıştırılması başarıyla tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Hu Momentleri ile Döndürme ve Ölçekten Bağımsız Motif Eşleme**

Dairesellik ve Solidity yararlı olsa da, bir motif $45^\circ$ döndürüldüğünde veya kamera yaklaştığında ölçeği değiştiğinde şekil kimliğini koruyan 7 adet **Hu Değişmez Momenti (Hu Invariant Moments)** kullanılır.

### Görev Tanımı:
[`src/motif_ayristirici.py`](./src/motif_ayristirici.py) içerisindeki `MotifBilgisi` sınıfına ve hesaplama döngüsüne Hu momentlerini entegre etmeni bekliyorum:

```python
@staticmethod
def hu_momentleri_hesapla(kontur: np.ndarray) -> np.ndarray:
```

### Beklenen Kurallar:
1. `cv2.moments(kontur)` ile merkezi momentleri hesaplamalı.
2. `cv2.HuMoments(moments)` ile 7 boyutlu Hu moment dizisini çıkarmalı.
3. Çok küçük logaritmik değerlerle rahat çalışabilmek için $-\text{sign}(h) \cdot \log_{10}(|h|)$ dönüşümü uygulayarak 7 elemanlı float dizisini döndürmelidir.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> Otsu eşikleme algoritması, görüntünün bütünü için tek bir global eşik ($T^*$) hesaplar.  
> Bir tekstil fabrikasında bant üzerindeki kumaşın **sol tarafı güçlü lamba ile aydınlatılırken sağ tarafı gölgede kaldığında** global Otsu algoritması neden çöker? Bu sorunu aşmak için hangi eşikleme yöntemi (Thresholding paradigm) kullanılmalıdır?

---

## 📂 Dizin Yapısı

```
day-14-motif-segmentation/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (opencv-python, numpy, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve görsel üretim akışı
├── ciktilar/                   # Üretilen 4 panelli analiz paneli
│   └── motif_segmentasyon_paneli.png
├── src/
│   ├── __init__.py
│   ├── motif_ayristirici.py    # Otsu, kontur, morfoloji ve şekil analitiği
│   └── gorsellestirici.py      # 4 panelli Matplotlib çizelge motoru
└── testler/
    └── test_motif_segmentasyon.py # 7 adet birim testi (7 passed in 1.11s)
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
python -m pytest testler/test_motif_segmentasyon.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
