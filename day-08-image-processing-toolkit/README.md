# Day 08: OpenCV Tabanlı Temel Görüntü İşleme Araç Seti (Image Processing Toolkit)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; bilgisayarlı görü ve nesne algılama sistemlerinde görüntüleri derin öğrenme modellerine hazırlamak veya geleneksel görü algoritmalarıyla geometrik analizler yapmak için kullanılan **2B Konvolüsyon Filtrelerini, Gauss Yumuşatmasını, Sobel Yönlü Kenar Gradyanlarını ve Matematiksel Morfolojik Dönüşümleri (Aşınma, Genişleme, Açma, Kapatma, Morfolojik Gradyan)** üretim seviyesinde bir araya getiren kapsamlı bir araç setidir.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Bir derin öğrenme modeline (YOLO, Mask R-CNN veya OCR) ham kamera görüntüsünü doğrudan vermek çoğu zaman felaketle sonuçlanır:
- Fabrika ortamındaki titreşimler ve yüksek ISO gürültüleri (yüksek frekanslı parazitler), modellerin yanlış nesne tespit etmesine yol açar.
- Bir barkodun veya elektronik devre kartının iletken hatlarında mikro kopukluklar veya delikler varsa, nesne segmentasyonu parçalanır.
- Bir nesnenin sınır çizgilerini (contour) netleştirmeden boyut ölçümü (ölçüm metrolojisi) yapılamaz.

Klasik görüntü işleme teknikleri; derin öğrenme öncesi **ön işleme (preprocessing)** ve sonrasındaki **maske temizleme (postprocessing)** süreçlerinin vazgeçilmez temel taşıdır.

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Görüntü filtreleme, kontrast ayarlama, kırpma ve gürültü temizleme işlevlerini yeniden kullanılabilir modüler bir araç setinde toplamak için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Bilgisayarlı görü projelerinde tekrar eden temel görüntü işleme kodlarını standartlaştırır ve hata payını sıfırlar.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Gelişmiş derin öğrenme tabanlı restorasyon veya süper çözünürlük operasyonlarını kapsamaz.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- Albumentations, torchvision.transforms veya imgaug.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Uzamsal Evrişim** | *Spatial 2D Convolution* | Bir filtre çekirdeğinin (kernel) görüntü üzerinde kaydırılarak yerel piksel pencereleriyle çarpılıp toplanması işlemi. |
| **Gaussian Filtresi** | *Gaussian Blur Smoothing* | Görüntüdeki yüksek frekanslı gürültüleri iki boyutlu normal dağılım ağırlıklarıyla pürüzsüzleştiren alçak geçiren filtre. |
| **Sobel Kenar Filtresi** | *Sobel Gradient Filter* | Yatay ($G_x$) ve dikey ($G_y$) yönlerde birinci derece türev alarak parlaklık gradyanlarını ve kenarları saptayan operatör. |
| **Laplacian Filtresi** | *Laplacian Operator* | İkinci derece türev alarak görüntüdeki ani yoğunluk değişimlerini ve kenar geçişlerini tespit eden izotropik filtre. |
| **Dolgulama (Padding)** | *Zero / Reflect Padding* | Evrişim sonrasında görsel sınırlarında boyut kaybını önlemek için kenarlara piksel ekleme tekniği. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. 2 Boyutlu Ayrık Konvolüsyon (Discrete 2D Convolution)
Bir $K \times K$ boyutundaki çekirdek (kernel / filtre matrisi), görüntünün üzerinde piksel piksel kaydırılarak yerel komşulukların ağırlıklı toplamı hesaplanır:

$$(I * K)(x, y) = \sum_{i=-a}^a \sum_{j=-b}^b I(x - i, y - j) \cdot K(i, j)$$

- **Keskinleştirme Çekirdeği (Sharpening Kernel):**
  Merkez piksele yüksek pozitif ağırlık verip çevre pikselleri çıkararak kenar kontrastını artırır:
  $$K_{\text{keskin}} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

---

#### B. Gauss Bulanıklaştırma (Gaussian Smoothing / Blur)
Yüksek frekanslı sensör gürültüsünü bastırmak için merkeze en yakın piksellere en yüksek, uzaklaştıkça çan eğrisi şeklinde azalan ağırlıklar verir:

$$G(x, y) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

- **Özellikleri:** Çekirdek boyutları ($W, H$) daima **tek sayı (odd)** olmalıdır ($3 \times 3, 5 \times 5, 7 \times 7$). Çift sayılarda tam merkez piksel bulunamaz.
- Standart sapma ($\sigma$) arttıkça görüntü daha homojen ve yumuşak hale gelir.

---

#### C. Sobel Yönlü Gradyan ve Kenar Büyüklüğü
Görüntüdeki ani piksel parlaklık değişimleri (birinci dereceden türev) kenarları oluşturur.

1. **Yatay Gradyan ($G_x$ - Dikey Kenarları Yakalar):**
   $$K_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}$$

2. **Dikey Gradyan ($G_y$ - Yatay Kenarları Yakalar):**
   $$K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

3. **Birleşik Kenar Büyüklüğü ve Açısı:**
   $$G = \sqrt{G_x^2 + G_y^2} \quad \text{veya yaklaşık olarak} \quad |G_x| + |G_y|$$
   $$\theta = \arctan2(G_y, G_x)$$

---

#### D. Matematiksel Morfoloji Operatörleri
İkili (siyah-beyaz) veya gri seviye görüntülerde yapısal element ($B$) kullanılarak nesne geometrisi dönüştürülür:

1. **Aşınma (Erosion - $A \ominus B$):**
   Yapısal element nesnenin sınırlarında gezdirilir; nesne sınırlarını inceltir, izole mikro beyaz gürültüleri yok eder.
2. **Genişleme (Dilation - $A \oplus B$):**
   Nesne sınırlarını dışarı doğru genişletir; kopuk çizgileri birleştirir.
3. **Açma (Opening - $A \circ B = (A \ominus B) \oplus B$):**
   Önce aşınma, sonra genişleme. **Nesnenin genel boyutunu değiştirmeden arka plandaki küçük beyaz parazitleri yok eder.**
4. **Kapatma (Closing - $A \bullet B = (A \oplus B) \ominus B$):**
   Önce genişleme, sonra aşınma. **Nesnenin içindeki küçük siyah delikleri, oyukları ve yarıkları doldurur.**
5. **Morfolojik Gradyan ($(A \oplus B) - (A \ominus B)$):**
   Genişlemiş görüntüden aşınmış görüntü çıkarıldığında geriye sadece **nesnenin dış sınır konturu** kalır!

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Sobel Operasyonunda `uint8` Taşması (Underflow / Saturation):**
   Açık renkten koyu renge geçildiğinde türev negatiftir (ör. $-180$). Eğer Sobel doğrudan `uint8` dizisine yazılırsa negatif değerler $0$'a yuvarlanır! **Görüntünün kenarlarının yarısı tamamen yok olur!**
   - *Çözüm:* Gradyanlar önce `CV_64F` (float64) olarak hesaplanmalı, ardından mutlak değeri alınıp `cv2.convertScaleAbs()` ile `uint8`'e dönüştürülmelidir.
2. **Kenar Çıkarmadan Önce Bulanıklaştırma Yapmamak:**
   Gürültülü bir görselde doğrudan Sobel uygularsanız, her gürültü pikseli devasa bir kenar gibi görünür. **Kural: Kenar çıkarmadan önce mutlaka Gauss filtresi uygulanmalıdır!**

---

## 📌 Mimari Tasarım ve Akış Şeması

```
                      Ham Kamera / Test Görüntüsü
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │    GaussBulaniklastirici (5x5)       │
               │   (Yüksek Frekans Gürültü Bastırma)  │
               └──────────────────┬───────────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────────┐             ┌─────────────────────────────┐
│    SobelKenarTespitEdici    │             │      İkili Eşikleme         │
│  (CV_64F Gradyan Büyüklüğü) │             │    (Binary Thresholding)    │
└──────────────┬──────────────┘             └──────────────┬──────────────┘
               │                                           │
         [Kenar Haritası]                                  ▼
                                            ┌─────────────────────────────┐
                                            │      MorfolojikIslemci      │
                                            └──────────────┬──────────────┘
                                                           │
                                   ┌───────────────────────┼───────────────────────┐
                                   ▼                       ▼                       ▼
                           [Açma (Opening)]       [Kapatma (Closing)]    [Morfolojik Gradyan]
                            Arka Plan Tozları       Nesne İçi Delikleri   Dış Sınır İskeleti
                            Temizlendi              Dolduruldu
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

1. **[`src/filtreler.py`](./src/filtreler.py):**
   - `KonvolusyonFiltresi`: Herhangi bir 2B ağırlık matrisini görüntü üzerine kaydırır.
   - `GaussBulaniklastirici`: Tek boyutlu çekirdek kontrolleriyle Gauss gürültü filtresi uygular.
   - `SobelKenarTespitEdici`: `CV_64F` korumasıyla yatay ($G_x$), dikey ($G_y$) ve birleşik büyüklüğü ($G$) hesaplar.
2. **[`src/morfoloji.py`](./src/morfoloji.py):**
   - `MorfolojikIslemci`: Dikdörtgen, elips veya artı yapısal elementleriyle Aşınma, Genişleme, Açma, Kapatma ve Morfolojik Gradyan işlemlerini yürütür.
3. **[`src/gorsellestirici.py`](./src/gorsellestirici.py):**
   - `IslemePaneliUreteci`: Başlıklı 9 panelli ızgarayı headless olarak PNG formatında oluşturur.
4. **[`ana_akis.py`](./ana_akis.py):**
   - Sentetik elektronik çip ve iletken hatları üreterek tüm boru hattını çalıştıran ana betik.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
==========================================================================
>>> AŞAMA 1: Sentetik Endüstriyel Görüntünün Üretimi
==========================================================================
[+] Görsel Boyutu      : (256, 256) (Yükseklik x Genişlik)
[+] Piksel Veri Tipi   : uint8
[+] Ortalama Parlaklık : 65.31

==========================================================================
>>> AŞAMA 2: Gauss Bulanıklaştırma ve Keskinleştirme Konvolüsyonu
==========================================================================
[V] 5x5 Gauss Yumuşatması başarıyla uygulandı (Yüksek frekanslı gürültü bastırıldı).
[V] 3x3 Özel Keskinleştirme Konvolüsyonu uygulandı.

==========================================================================
>>> AŞAMA 3: Sobel Gradyan ve Kenar Büyüklüğü Analizi
==========================================================================
[+] Yatay Kenar (Gx) Max Değer    : 255
[+] Dikey Kenar (Gy) Max Değer    : 255
[+] Birleşik Büyüklük (G) Max     : 255

==========================================================================
>>> AŞAMA 4: Matematiksel Morfoloji Operasyonları (Açma, Kapatma, Gradyan)
==========================================================================
[V] Açma (Opening)   : Arka plan beyaz gürültüleri yok edildi.
[V] Kapatma (Closing): Çip içi mikro siyah delikler dolduruldu.
[V] Morf. Gradyan    : Dış çevre konturları ayrıştırıldı.

==========================================================================
>>> AŞAMA 5: 9 Panelli Görsel Raporun Diske Kaydedilmesi
==========================================================================
[V] Görsel panel başarıyla üretildi: goruntu_isleme_paneli.png
[V] Kayıt Konumu: day-08-image-processing-toolkit/ciktilar/goruntu_isleme_paneli.png

[V] Day 8: OpenCV Tabanlı Temel Görüntü İşleme Araç Seti tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Laplasyen İkinci Dereceden Türev ve Sıfır Geçişi (Laplacian Zero-Crossing)**

Sobel operatörü birinci dereceden türev alırken, **Laplasyen ($\nabla^2 I$)** operatörü görüntünün ikinci dereceden türevini alarak tüm yönlerdeki kenarları tek bir çekirdekle yakalar.

### Görev Tanımı:
[`src/filtreler.py`](./src/filtreler.py) dosyasına `LaplasyenFiltresi` adında yeni bir sınıf eklemeni bekliyorum:

```python
class LaplasyenFiltresi:
    @staticmethod
    def kenarlari_cikar(
        gorsel: np.ndarray,
        cekirdek_boyutu: int = 3
    ) -> np.ndarray:
```

### Beklenen Kurallar:
1. Girdi görselini gri tonlamaya çevirmeli.
2. İkinci dereceden türevde sayısal taşmayı önlemek için `cv2.Laplacian` fonksiyonunu `cv2.CV_64F` derinliğinde çalıştırmalı.
3. Sonucu `cv2.convertScaleAbs()` ile `uint8`'e dönüştürerek döndürmelidir.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> Kenar tespitinde **Sobel (1. Dereceden Türev)** ile **Laplasyen (2. Dereceden Türev)** karşılaştırıldığında:  
> Neden gürültülü (noisy) görüntülerde Laplasyen filtresi Sobel'e kıyasla gürültüyü katbekat daha fazla büyütür? Endüstride bu sorunu çözmek için neden doğrudan Laplasyen yerine **LoG (Laplacian of Gaussian)** filtresi kullanılır?

---

## 📂 Dizin Yapısı

```
day-08-image-processing-toolkit/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (opencv-python, numpy, matplotlib, pytest)
├── ana_akis.py                 # Konsol çalıştırma ve panel üretim akışı
├── ciktilar/                   # Üretilen 9 panelli analiz görseli
│   └── goruntu_isleme_paneli.png
├── src/
│   ├── __init__.py
│   ├── filtreler.py            # Konvolüsyon, Gauss ve Sobel sınıfları
│   ├── morfoloji.py            # Aşınma, Genişleme, Açma, Kapatma, Gradyan
│   └── gorsellestirici.py      # 9 panelli Matplotlib grid çizici
└── testler/
    └── test_goruntu_isleme.py  # 7 adet birim testi (7 passed in 0.75s)
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
python -m pytest testler/test_goruntu_isleme.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
