# Day 01: NumPy Görüntü Analizörü ve Piksel İstatistikleri

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje, harici görüntü işleme kütüphanelerine (OpenCV, Pillow vb.) bağımlı kalmadan, doğrudan **NumPy** çok boyutlu dizi (ndarray) mimarisi üzerinden dijital görüntülerin piksel düzeyinde incelenmesini, renk kanalı ayrıştırmasını, ağırlıklı gri tonlama dönüşümünü, istatistiksel profil çıkarımını ve derin öğrenme modellerine hazır normalizasyon boru hatlarını sıfırdan inşa eder.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Yapay zeka ve derin öğrenme modelleri (CNN'ler, Vision Transformer'lar, Difüzyon modelleri) "resimleri" görmez; yalnızca **sayılardan oluşan çok boyutlu tensörleri (N-boyutlu dizileri)** işler. 
Endüstride görüntü işleme hatlarında yapılan en büyük hata, görselleri doğrudan harici kütüphanelerin kara kutu fonksiyonlarına bırakıp tensörün bellekteki düzenini, veri tipini ve sayısal sınırlarını göz ardı etmektir. 

NumPy seviyesinde piksel matrislerini yönetebilmek; özel veri artırma (augmentation) hatları yazarken, medikal görüntülerde (16-bit DICOM) veya uydu görüntülerinde (çok bantlı multispektral) çalışırken ve çıkarım (inference) hızını optimize ederken hayati önem taşır.

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **ndarray** | *N-Dimensional Array* | NumPy'ın C düzeyinde bitişik bellek bloklarında tuttuğu, homojen veri tipli ve yüksek hızlı vektörize işlemleri destekleyen temel çok boyutlu dizi yapısı. |
| **HWC vs CHW** | *Tensor Channel Order* | Görüntü tensörlerinin bellek düzenidir. OpenCV/NumPy satır-öncelikli (Yükseklik, Genişlik, Kanal - HWC) kullanırken PyTorch kanal-öncelikli (CHW) düzeni zorunlu kılar. |
| **uint8 Taşması** | *Integer Overflow/Underflow* | 8-bit işaretsiz tamsayıların [0, 255] aralığından taşarak (ör. 200+100=44) görselde piksellerin aniden kararmasına yol açan sayısal taşma hatası. |
| **C-Contiguous** | *C-Contiguous Memory Layout* | Dizi elemanlarının bellekte satır satır ardışık olarak dizildiği, CPU önbellek (L1/L2 cache) isabet oranını maksimize eden bellek yerleşimi. |
| **Lüminans Ağırlığı** | *Luminance Weighting (ITU-R BT.601)* | İnsan retinasının yeşil renge olan yüksek duyarlılığını modelleyen ağırlıklı gri tonlama dönüşüm katsayıları (0.299R + 0.587G + 0.114B). |
| **Epsilon Dengeleme** | *Epsilon Numerical Stabilization* | Bölme işlemlerinde paydanın sıfır olmasıyla oluşacak NaN veya Inf değerleri engellemek için eklenen küçük sayısal sabit (örn. 1e-7). |
| **Z-Skoru Standartlaştırması** | *Z-Score Normalization* | Piksel dağılımının ortalamasını 0, varyansını 1 yaparak model gradyanlarının kararlı yakınsamasını sağlayan standartlaştırma tekniği. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. Tensör Düzeni ve Boyutlar ($H \times W \times C$)
Renkli bir dijital görüntü 3 boyutlu bir tensördür:
- **$H$ (Yükseklik / Satır Sayısı - Y Ekseni):** Dikeydeki piksel adedi.
- **$W$ (Genişlik / Sütun Sayısı - X Ekseni):** Yataydaki piksel adedi.
- **$C$ (Kanal Sayısı - Renk Derinliği):** RGB için $3$ (Kırmızı, Yeşil, Mavi), Gri için $1$, Saydamlık varsa RGBA için $4$.

Bellekte bu tensör ardışık baytlar halinde tutulur. NumPy'da varsayılan olarak **C-contiguous** (satır-öncelikli / row-major) bellek yerleşimi kullanılır:
$$\text{Bellek Adresi}(y, x, c) = \text{Taban} + (y \cdot W \cdot C + x \cdot C + c) \times \text{Eleman\_Bayt\_Boyutu}$$

#### B. Ağırlıklı Lüminans Gri Ton Dönüşümü (ITU-R BT.601)
Bir RGB görseli gri seviyeye dönüştürürken üç kanalın basit aritmetik ortalamasını almak ($\frac{R+G+B}{3}$) insan gözünün biyolojik algısına uymaz. İnsan retinasındaki koni hücreleri yeşil ışığa son derece duyarlıyken, mavi ışığa karşı çok daha az hassastır. Bu sebeple endüstri standardı olan **ağırlıklı lüminans** formülü kullanılır:

$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

#### C. Normalizasyon ve Standartlaştırma Dinamikleri
Modellerin gradyan inişi (Gradient Descent) sırasında ağırlıklarının kararlı güncellenmesi için pikseller $[0, 255]$ tamsayı aralığından çıkarılmalıdır:
1. **Min-Max Doğrusal Ölçekleme ($[0, 1]$ veya $[-1, 1]$):**
   $$X_{\text{norm}} = a + \frac{X - X_{\min}}{(X_{\max} - X_{\min}) + \epsilon} \cdot (b - a)$$
   *(Buradaki $\epsilon = 10^{-8}$, görsel tekdüze/tek renk olduğunda $0/0$ tanımsızlığını önler).*

2. **Z-Skoru Standartlaştırması (Kanal Bazlı):**
   Görselin ortalamasını $0$, standart sapmasını $1$ yapar ($N(0, 1)$):
   $$Z_c = \frac{X_c - \mu_c}{\sigma_c + \epsilon}$$

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **`uint8` Sayısal Taşması (Integer Overflow/Underflow):**
   Piksel değerleri varsayılan olarak 8-bit işaretsiz tamsayıdır (`np.uint8`, aralık: $0-255$). Eğer iki pikseli toplar veya çarparsanız:
   $$200 + 100 = 300 \xrightarrow{\text{uint8}} 300 \pmod{256} = 44$$
   Parlaklaşması gereken piksel aniden simsiyah olur! Matematiksel işlem öncesinde veri tipi mutlaka `float32`'ye çevrilmeli ve işlem sonunda `np.clip(deger, 0, 255).astype(np.uint8)` uygulanmalıdır.
2. **Görünüm (View) vs. Kopya (Copy) Farkı:**
   `alt_matris = gorsel[0:50, 0:50]` ifadesi bellekte yeni bir dizi oluşturmaz, sadece orijinal dizinin bir **görünümünü (slice view)** referans alır. `alt_matris` üzerindeki bir değişiklik orijinal görseli de bozar. İzolasyon gerektiğinde daima `.copy()` kullanılmalıdır.
3. **Sıfıra Bölme Hatası (Zero Division & NaN Üretimi):**
   Tamamen siyah bir görselde veya sabit arka planda varyans sıfırdır ($\sigma = 0$). Paydaya küçük bir sayısal dengeleyici ($\epsilon = 10^{-7}$) eklenmezse tensör `NaN` veya `Inf` değerlerle dolar ve derin öğrenme modelini patlatır.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
   Girdi Görsel Matrisi (H x W x C, uint8)
                    │
                    ▼
       ┌────────────────────────┐
       │  NumPyGoruntuAnalizoru │
       └───────────┬────────────┘
                   │
    ┌──────────────┼──────────────┬──────────────────┐
    ▼              ▼              ▼                  ▼
[Kanal Ayrıştırma] [Gri Dönüşüm] [İstatistikler]   [Normalizasyon]
- Kırmızı Matrisi  - BT.601       - Min, Max, Ort.   - Min-Max [0, 1]
- Yeşil Matrisi    - Ağırlıklı    - Medyan, Varyans  - Min-Max [-1, 1]
- Mavi Matrisi       Lüminans     - Çeyreklikler     - Z-Skoru N(0, 1)
```

---

## 💻 Konsol Çalıştırma Çıktısı

```text
=================================================================
>>> AŞAMA 1: Sentetik Test Görseli Üretimi ve Matris Boyutları
=================================================================
[+] Üretilen Görsel Şekli (Shape)    : (128, 128, 3) -> (Yükseklik, Genişlik, Kanal)
[+] Veri Tipi (Dtype)                : uint8
[+] Toplam Eleman Sayısı             : 49,152 değer

=================================================================
>>> AŞAMA 2: Düşük Seviyeli Bellek Yerleşimi ve Strides (Adımlar)
=================================================================
  * boyut_sekli              : (128, 128, 3)
  * adimlar_strides          : (384, 3, 1)
  * c_surekli_mi             : True

=================================================================
>>> AŞAMA 5: Kapsamlı İstatistiksel Analiz Raporu
=================================================================
Genel Çözünürlük     : 128x128 (16,384 piksel)
Bellek Tüketimi      : 48.00 KB
-----------------------------------------------------------------
Kanal      | Min   | Max   | Ortalama   | Medyan   | Std Sapma 
-----------------------------------------------------------------
Kirmizi    | 0     | 255   | 127.50     | 127.5    | 127.50    
Yesil      | 0     | 255   | 127.50     | 127.5    | 127.50    
Mavi       | 0     | 255   | 63.75      | 0.0      | 110.42    
-----------------------------------------------------------------

=================================================================
>>> AŞAMA 6: Piksel Değer Normalizasyonu Deneyleri
=================================================================
[+] Min-Max [0, 1]  -> Min: 0.0000, Max: 1.0000, Veri Tipi: float32
[+] Min-Max [-1, 1] -> Min: -1.0000, Max: 1.0000, Veri Tipi: float32
[+] Z-Skoru (Kanal Bazlı):
    - Kırmızı Z-Dağılımı -> Ortalama: 0.0000, Std Sapma: 1.0000
    - Yeşil   Z-Dağılımı -> Ortalama: 0.0000, Std Sapma: 1.0000
    - Mavi    Z-Dağılımı -> Ortalama: 0.0000, Std Sapma: 1.0000

=================================================================
>>> AŞAMA 7: Sayısal Taşma (Overflow) Koruması Testi
=================================================================
[+] 1.5x Parlaklık Sonrası Max Değer: 255 (Taşma engellendi, 255'te sınırlandı)
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Kontrast Germe (Contrast Stretching / Percentile Clipping)**

Endüstride düşük kontrastlı veya çok karanlık çekilmiş görüntülerde (ör. güvenlik kameraları veya endüstriyel kalite kontrol bantları) dinamik aralığı artırmak için **yüzdelik tabanlı kontrast germe (Contrast Stretching)** uygulanır.

### Görev Tanımı:
[`src/goruntu_analizoru.py`](./src/goruntu_analizoru.py) içerisine şu imzaya sahip yeni bir metod eklemeni istiyorum:

```python
def kontrast_ger(
    self,
    alt_yuzdelik: float = 2.0,
    ust_yuzdelik: float = 98.0
) -> np.ndarray:
```

### Beklenen Mantık ve Kurallar:
1. `np.percentile()` kullanarak görselin en alt %2'lik ve en üst %98'lik piksel eşik değerlerini ($P_{\text{alt}}$ ve $P_{\text{ust}}$) belirlemeli.
2. Değerleri $[P_{\text{alt}}, P_{\text{ust}}]$ aralığına `np.clip()` ile kırpmalı (aykırı uç pikselleri elemek için).
3. Ardından bu aralığı $[0, 255]$ aralığına doğrusal olarak yaymalı:
   $$X_{\text{yeni}} = \frac{X_{\text{kirpilmis}} - P_{\text{alt}}}{P_{\text{ust}} - P_{\text{alt}} + \epsilon} \times 255$$
4. Sonucu `uint8` tipinde döndürmeli.
5. `alt_yuzdelik >= ust_yuzdelik` durumunda `ValueError` fırlatmalı.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> Bir CNN (Evrişimli Sinir Ağı) modelini eğitirken görsel piksellerini doğrudan `[0, 255]` aralığında vermek yerine, neden **Z-Skoru Standartlaştırması** ($Z = \frac{X - \mu}{\sigma}$) uygulayarak veriyi sıfır merkezli (zero-centered) ve birim varyanslı hale getiririz? Bu işlemin ağırlık güncellemeleri (weight update) ve aktivasyon fonksiyonları (ör. Sigmoid, ReLU) üzerindeki etkisi nedir?

---

## 📂 Dizin Yapısı

```
day-01-numpy-image-analyzer/
├── LICENSE                     # Özel Tüm Hakları Saklıdır lisans dosyası
├── README.md                   # Proje ders ve teknik dokümantasyonu
├── gereksinimler.txt           # Python bağımlılıkları
├── ana_akis.py                 # Konsol çalıştırma betiği
├── src/
│   ├── __init__.py             # Paket tanımlayıcısı
│   ├── goruntu_analizoru.py    # NumPyGoruntuAnalizoru çekirdek sınıfı
│   └── yardimcilar.py          # Sentetik görsel ve bellek araçları
└── testler/
    └── test_analizor.py        # Kapsamlı pytest birim testleri (8 passed)
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
python -m pytest testler/test_analizor.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
