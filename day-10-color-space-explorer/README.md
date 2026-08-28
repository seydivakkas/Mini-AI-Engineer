# Day 10: Renk Uzayları Gezgini ve Gölgeye Dayanıklı Segmentasyon (Color Space Explorer)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; bilgisayarlı görüde ve robotik algı sistemlerinde değişken aydınlatma, yansıma ve derin gölgeler altında hedef nesneleri kararlı bir şekilde takip edebilmek için **RGB, HSV, CIELAB ve YCrCb** renk uzaylarını kanal bazında ayrıştıran, RGB'nin ışık zaafını kanıtlayan ve **OpenCV Ton (Hue) döngüsünü çözen çift aralıklı kırmızı segmentasyonu** ile **CIELAB $\Delta E^*$ algısal renk mesafesi motorunu** üretim kalitesinde sunan bir renk bilimi laboratuvarıdır.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Görüntü işlemede yeni başlayanların yaptığı en yaygın hata:
> *"Kırmızı bir nesneyi bulmak istiyorsam, $R > 150$ ve $G < 50$ ve $B < 50$ yaparım!"*

Gerçek dünyada bir fabrika bandına bulutlu havada güneş vurabilir, robotun kolu parçanın üzerine gölge düşürebilir ya da oda lambası sarı ışık yayabilir:
- Gölge düştüğünde, kırmızı nesnenin üzerindeki piksel değerleri örneğin $(R=240, G=20, B=20)$'den aniden $(R=90, G=10, B=10)$'a düşer!
- $R > 150$ kuralı gölgedeki kırmızı nesneyi **tamamen yok sayar**. Eşiği $R > 80$ yaparsanız bu sefer tüm gri arka planı kırmızı sanırsınız!

**Çözüm:** Rengi oluşturan saf kromatik bilgiyi (chrominance / hue) parlaklık bilgisinden (luminance / value) ayıran renk uzaylarını (HSV, LAB, YCrCb) kullanmaktır.

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- RGB, BGR, HSV ve LAB renk uzayları arasındaki matematiksel dönüşümleri inceleyerek ışık değişimlerinden bağımsız renk analizi yapmak için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- RGB uzayında gölge veya parlama nedeniyle değişen renk değerlerinin yarattığı segmentasyon hatalarını çözer.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Dokusal desenleri veya nesne geometrilerini hesaba katmaz; yalnızca saf renk koordinatlarını dönüştürür.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- CIELUV, YCrCb veya Öznitelik Bazlı Renk Gömme (Color Embeddings).

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **RGB Renk Uzayı** | *RGB Color Space* | Kırmızı, Yeşil ve Mavi ışık bileşenlerinin toplanmasıyla oluşan aygıt bağımlı kartezyen renk küpü. |
| **HSV Renk Uzayı** | *HSV (Hue, Saturation, Value)* | İnsan algısına daha uygun; Renk Özü ($H$), Doygunluk ($S$) ve Parlaklık ($V$) ayrımı sunan silindirik renk modeli. |
| **CIELAB Uzayı** | *CIE L*a*b* Color Space* | Işıklılık ($L^*$), Yeşil-Kırmızı ($a^*$) ve Mavi-Sarı ($b^*$) eksenlerinden oluşan algısal olarak tekdüze (perceptually uniform) renk uzayı. |
| **Delta E ($\Delta E$)** | *Color Difference Formula* | İki renk arasındaki algısal farkı Öklid uzayında ölçen metrik ($\Delta E < 1.0$ fark insan gözüyle ayırt edilemez). |
| **Renk Gamutu** | *Color Gamut* | Belirli bir renk modelinin veya ekran donanımının üretebildiği toplam renk alt kümesi. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. RGB Uzayı (Donanım Bağımlı / Katmanlı Model)
Kameraların sensör pikselleri ve ekranlar doğrudan Kırmızı (R), Yeşil (G) ve Mavi (B) fotodiyotlarla çalışır.
- **Sorun:** Renk tonu ve aydınlık iç içe geçmiştir. Işık şiddeti $I$ kat azaldığında her üç kanal da orantılı olarak küçülür: $[I \cdot R, I \cdot G, I \cdot B]$. Renk tespiti ışık dalgalanmalarına karşı son derece kırılgandır.

#### B. HSV Uzayı (Silindirik Algı Modeli)
İnsan gözünün rengi tarif etme biçimine (Renk tonu, Canlılık, Aydınlık) dayanır:
1. **Ton (Hue - $H$):** Saf renk dalga boyu. $0^\circ$ ile $360^\circ$ arasındaki açısal koordinattır ($0^\circ$ Kırmızı, $120^\circ$ Yeşil, $240^\circ$ Mavi).
   - **⚠️ OpenCV Kuralı:** 8-bit işaretsiz tamsayı (`uint8`) en fazla $255$ alabildiği için OpenCV açıyı $2$'ye böler: **$H \in [0, 179]$!**
2. **Doygunluk (Saturation - $S \in [0, 255]$):** Rengin saflığı veya grilikten uzaklığı ($0$ gri, $255$ saf canlı renk).
3. **Değer (Value - $V \in [0, 255]$):** Işık şiddeti / aydınlık ($0$ zifiri karanlık, $255$ maksimum parlaklık).

- **💡 Gölgeye Bağışıklık:** Bir kırmızı nesnenin üzerine gölge düştüğünde $V$ değeri $240$'tan $70$'e düşebilir; fakat **$H$ (Ton) değeri $0-5^\circ$ arasında sabit kalır!**

#### C. CIELAB ($L^* a^* b^*$) Uzayı (Algısal Düzgünlük / Perceptual Uniformity)
Uluslararası Aydınlatma Komisyonu (CIE) tarafından insan gözünün biyolojik renk algısına birebir uyacak şekilde tasarlanmıştır:
- **$L^*$ (Aydınlık / Lightness):** $0$ (siyah) ile $100$ (beyaz) arası.
- **$a^*$ Ekseni:** Negatif değerler **Yeşil**, pozitif değerler **Kırmızı/Macenta**.
- **$b^*$ Ekseni:** Negatif değerler **Mavi**, pozitif değerler **Sarı**.

- **CIE76 $\Delta E^*$ Renk Mesafesi:** İki renk ($L_1, a_1, b_1$) ve ($L_2, a_2, b_2$) arasındaki algısal Öklid farkıdır:
  $$\Delta E^* = \sqrt{(L_1 - L_2)^2 + (a_1 - a_2)^2 + (b_1 - b_2)^2}$$
  - $\Delta E^* < 1.0$: İnsan gözü iki rengi birbirinden ayırt edemez.
  - $1.0 < \Delta E^* < 5.0$: Dikkatli bakıldığında hafif fark sezilir.
  - $\Delta E^* > 20.0$: Tamamen farklı iki renk.

#### D. YCrCb Uzayı (Video Sıkıştırma ve Ten Algılama)
Televizyon yayıncılığı ve JPEG sıkıştırmasında kullanılır:
- **$Y$ (Luma / Lüminans):** İnsan gözünün detay algısını sağlayan siyah-beyaz aydınlık.
- **$Cr$ (Chroma Red):** $R - Y$ farkı (Kırmızı yoğunluğu).
- **$Cb$ (Chroma Blue):** $B - Y$ farkı (Mavi yoğunluğu).
- İnsan ten rengi ırktan bağımsız olarak $Cr \in [133, 173]$ ve $Cb \in [77, 127]$ aralığına kümelenir.

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Kırmızı Renk Tonunun Sıfır Noktasında Sarılması (Hue Wrap-Around):**
   Kırmızı renk renk çemberinin tam başlangıcında ($0^\circ$) ve bitişinde ($360^\circ$) yer alır.
   OpenCV'de bu durum kırmızının iki uca bölünmesine neden olur:
   - Başlangıç: $H \in [0, 10]$
   - Bitiş: $H \in [170, 179]$
   Eğer tek bir aralık tanımlarsanız ($[0, 10]$), **kırmızı piksellerin neredeyse yarısını kaçırırsınız!** İki aralık için iki ayrı maske üretilip mantıksal `cv2.bitwise_or(maske1, maske2)` ile birleştirilmelidir.
2. **Kromatik Mesafe İçin $L^*$ Kanalını Hariç Tutmak:**
   Eğer gölgede kalmış bir nesneyi rengine göre bulmak istiyorsanız, CIELAB formülünde $L^*$ aydınlık farkını dahil etmemelisiniz; yalnızca **$\Delta C^*_{ab} = \sqrt{(\Delta a^*)^2 + (\Delta b^*)^2}$** kromatik farkını kullanmalısınız.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
                      Gölge ve Işık Geçişli BGR Görüntü
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    RenkUzayiDonusturucu     │
                      └──────────────┬──────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  [RGB Kanalları]             [HSV Kanalları]              [LAB Kanalları]
- R (Gölgede Çöker)         - Ton H (Gölgede Sabit!)     - L* (Aydınlık)
- G                         - Doygunluk S                - a* (Yeşil - Kırmızı)
- B                         - Değer V (Gölgeyi Taşır)    - b* (Mavi - Sarı)
                                     │                            │
                                     ▼                            ▼
                      ┌─────────────────────────────┐  ┌─────────────────────────────┐
                      │   Çift Aralık Kırmızı       │  │    CIELAB Kromatik          │
                      │       Segmentasyonu         │  │    Delta-E Eşiklemesi       │
                      │ [0-10] OR [170-179]         │  │    sqrt(Δa^2 + Δb^2)        │
                      └──────────────┬──────────────┘  └──────────────┬──────────────┘
                                     │                                │
                                     └────────────────┬───────────────┘
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │   12 Panelli Analiz Raporu  │
                                       │  (Gölgeye Rağmen Net Nesne) │
                                       └──────────────┬──────────────┘
                                                      │
                                                      ▼
                                       [ciktilar/renk_uzaylari_
                                           analiz_paneli.png]
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

1. **[`src/renk_donusturucu.py`](./src/renk_donusturucu.py):**
   - `RenkUzayiDonusturucu`: RGB, HSV, CIELAB ve YCrCb dönüşümleri ve 3-kanal ayrıştırması.
   - `RenkSegmentasyoncu`: Çift aralıklı kırmızı maskeleme, aydınlık bağımsız CIELAB Delta-E algısal renk mesafesi ve ikili maske uygulama.
2. **[`src/gorsellestirici.py`](./src/gorsellestirici.py):**
   - `RenkUzayiGorsellestirici`: 12 panelli zengin Matplotlib çizelgesini diske kaydeder.
3. **[`ana_akis.py`](./ana_akis.py):**
   - Çapraz gölge geçişli sentetik endüstriyel sahne üreterek RGB zaafını ve HSV/LAB başarısını kanıtlayan konsol yürütücüsü.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
==========================================================================
>>> AŞAMA 1: Sentetik Sahnenin Oluşturulması ve Işık Değişimi
==========================================================================
[+] Çözünürlük               : (256, 256, 3)
[+] Sahne Özelliği           : Çapraz sert gölge geçişi
[+] Kırmızı Nesne Konumu     : Merkez (Yarısı gölgede, yarısı ışıkta!)

==========================================================================
>>> AŞAMA 2: RGB Uzayının Gölge Zaafı Kanıtı
==========================================================================
  * Işıktaki Kırmızı Nesne R Değeri : 171
  * Gölgedeki Kırmızı Nesne R Değeri: 152  <-- Aşırı düştü!
  * Arka Plan Nötr Gri R Değeri     : 151
>>> Sonuç: RGB uzayında 'R > 150' eşiği koyarsanız, gölgedeki kırmızı nesneyi
    tamamen kaçırırsınız; 'R > 80' yaparsanız tüm gri arka planı kırmızı sanırsınız!

==========================================================================
>>> AŞAMA 3: HSV Uzayında Renk Tonu (Hue) İzolasyonu
==========================================================================
  * Işıktaki Kırmızı Ton (H) Değeri : 0°
  * Gölgedeki Kırmızı Ton (H) Değeri: 0°
[V] Gözlem: Parlaklık 3 kat düşse bile Ton (Hue) değeri 0-5° arasında SABİT KALDI!

==========================================================================
>>> AŞAMA 4: Çift Aralık Kırmızı Segmentasyonu ve Hedef Çıkarma
==========================================================================
[V] Kırmızı maske başarıyla üretildi. Yakalanan piksel: 4513 adet.
[V] Nesne hem gölgede hem ışıkta tek parça halinde eksiksiz izole edildi!

==========================================================================
>>> AŞAMA 5: CIELAB Kromatik Delta-E ile Yeşil Nesnenin Algısal Tespiti
==========================================================================
[V] CIELAB kromatik mesafesiyle yeşil parça tespit edildi: 2595 piksel.

==========================================================================
>>> AŞAMA 6: 12 Panelli Analiz Raporunun Kaydedilmesi
==========================================================================
[V] 12 Panelli analiz çizelgesi kaydedildi: renk_uzaylari_analiz_paneli.png
[V] Tam Dosya Yolu: day-10-color-space-explorer/ciktilar/renk_uzaylari_analiz_paneli.png

[V] Day 10: Renk Uzayları Gezgini ve Segmentasyon başarıyla tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: YCrCb Tabanlı Evrensel İnsan Teni Dedektörü (Skin Detector)**

Görüntü işlemede el, parmak veya yüz takibinde ışık değişimlerinden etkilenmeyen en güçlü yöntem **YCrCb renk uzayındaki krominans ($Cr, Cb$) elipsidir**. Ten rengi ırktan bağımsız olarak belirli bir kromatik aralığa toplanır.

### Görev Tanımı:
[`src/renk_donusturucu.py`](./src/renk_donusturucu.py) içerisindeki `RenkSegmentasyoncu` sınıfına şu fonksiyonu eklemeni bekliyorum:

```python
@classmethod
def ten_rengi_maskesi(
    cls,
    gorsel_bgr: np.ndarray,
    cr_aralik: Tuple[int, int] = (133, 173),
    cb_aralik: Tuple[int, int] = (77, 127)
) -> np.ndarray:
```

### Beklenen Kurallar:
1. Görüntüyü `cv2.COLOR_BGR2YCrCb` ile YCrCb uzayına dönüştürmelidir.
2. $Y$ (Luma/Parlaklık) kanalını tamamen serbest bırakmalı ($0-255$).
3. Yalnızca $Cr$ ve $Cb$ kanallarını belirtilen sınırlar içinde filtreleyerek (`cv2.inRange`) ten maskesini ikili (binary) olarak döndürmelidir.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> İki renk arasındaki farkı ölçmek istediğimizde neden **RGB uzayındaki Öklid mesafesi** insan gözünü yanıltır da **CIELAB uzayındaki $\Delta E^*$ mesafesi** insan algısıyla birebir örtüşür?  
> *(İpucu: "Algısal Düzgünlük / Perceptual Uniformity" kavramını ve MacAdam Elipslerini düşünün).*

---

## 📂 Dizin Yapısı

```
day-10-color-space-explorer/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (opencv-python, numpy, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve 12 panelli görselleştirme yürütücüsü
├── ciktilar/                   # Üretilen görsel analiz raporu
│   └── renk_uzaylari_analiz_paneli.png
├── src/
│   ├── __init__.py
│   ├── renk_donusturucu.py     # RenkUzayiDonusturucu ve RenkSegmentasyoncu sınıfları
│   └── gorsellestirici.py      # Matplotlib 12 panelli çizelge motoru
└── testler/
    └── test_renk_uzaylari.py   # 7 adet birim testi (7 passed in 1.61s)
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
python -m pytest testler/test_renk_uzaylari.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
