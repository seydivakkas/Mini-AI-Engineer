# Day 09: Görüntü Histogramı Analizörü ve Kontrast İyileştirme (Image Histogram Analyzer & CLAHE)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; bilgisayarlı görü ve dijital görüntü işlemede düşük ışıklı, sisli, arkadan aydınlatmalı (backlit) veya aşırı parlamalı endüstriyel kamera görüntülerinde piksel frekans dağılımlarını, **Kümülatif Dağılım Fonksiyonunu (CDF)** ve **Shannon Bilgi Entropisini** inceleyen; renk bozulmasına (chromatic aberration) yol açmadan **Global Histogram Eşitleme** ve **Kontrast Sınırlı Uyarlanabilir Histogram Eşitleme (CLAHE)** teknikleriyle dinamik aralığı maksimize eden kapsamlı bir analiz laboratuvarıdır.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Bir endüstriyel kalite kontrol bandında veya medikal görüntüleme sisteminde (ör. X-Ray, ultrason, termal kamera) sensörden gelen görüntü genellikle çok dar bir parlaklık aralığına sıkışır:
- Kumaş dokuma hatlarında kumaş rengi koyuysa tüm pikseller $[20, 60]$ aralığında toplanır.
- Bir röntgen filminde tüm kemik ve doku yoğunlukları çok benzer gri tonlardadır.
- Derin öğrenme nesne tespit modelleri (ör. YOLO, RetinaNet), düşük kontrastlı bölgelerdeki zayıf kenarları ve dokuları kaçırır.

Histogram analizi, görüntünün "ışık parmak izini" çıkararak gizli detayları görünür kılmanın matematiksel yoludur.

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Piksel Histogramı** | *Image Pixel Histogram* | Görüntüdeki her bir parlaklık seviyesinin ($0-255$) frekansını (kaç kez tekrarlandığını) gösteren ayrık dağılım. |
| **Kümülatif Dağılım Fonksiyonu** | *Cumulative Distribution Function (CDF)* | Piksellerin belirli bir parlaklık seviyesine kadar olan toplam birikimli olasılığını gösteren fonksiyon. |
| **Histogram Eşitleme** | *Histogram Equalization (HE)* | CDF dönüşümü kullanarak dar dinamik aralıktaki kontrastı tüm gri ton skalasına yayarak görsel belirginliği artıran yöntem. |
| **Dinamik Aralık** | *Dynamic Range* | Görüntüdeki en karanlık piksel ile en aydınlık piksel arasındaki sinyal gücü oranı. |
| **Kontrast Genişletme** | *Contrast Stretching* | Piksel değerlerini $[I_{\min}, I_{\max}]$ aralığından doğrusal olarak $[0, 255]$ aralığına açma işlemi. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. Piksel Yoğunluk Histogramı ve Olasılık Dağılımı
Bir görüntünün histogramı $H(k)$, $k \in [0, 255]$ parlaklık değerine sahip piksel adedini temsil eder. Normalleştirilmiş histogram ise piksel olasılık yoğunluk fonksiyonudur (PDF):

$$P(r_k) = \frac{n_k}{N \times M}$$
*(Burada $n_k$, $k$ parlaklığındaki piksel sayısı; $N \times M$ toplam çözünürlüktür).*

#### B. Kümülatif Dağılım Fonksiyonu (CDF - Cumulative Distribution Function)
Piksel değerlerinin $0$'dan $k$'ya kadar olan birikimli olasılık toplamıdır:

$$CDF(k) = \sum_{j=0}^k P(r_j) = \frac{\sum_{j=0}^k n_j}{N \times M}$$
- $CDF(0) \ge 0$, $CDF(255) = 1.0$ (monotonik artan fonksiyon).

#### C. Global Histogram Eşitleme (Global Histogram Equalization)
Amaç, dar bir alana yığılmış histogramı tüm $[0, 255]$ aralığına **tekdüze (uniform)** olarak yaymaktır. Dönüşüm fonksiyonu doğrudan normalize CDF'e dayanır:

$$s_k = T(r_k) = \text{round}\left( \frac{CDF(k) - CDF_{\min}}{1.0 - CDF_{\min}} \times 255 \right)$$

- **Sonuç:** Kümülatif dağılım eğrisi düz bir doğrusal çizgiye ($45^\circ$) yaklaşır, dinamik aralık $255$'e ulaşır.
- **Kritik Dezavantajı:** Tüm görüntüye küresel tek bir formül uyguladığı için, arka plandaki önemsiz gürültüleri aşırı parlatır ve zaten parlak olan kısımları patlatarak (over-exposure / washout) detay kaybına yol açar!

#### D. Kontrast Sınırlı Uyarlanabilir Histogram Eşitleme (CLAHE)
Global eşitlemenin zayıflıklarını gidermek için Karel Zuiderveld (1994) tarafından geliştirilmiştir:
1. **Karolara Bölme (Tiling):** Görüntü küçük yerel ızgaralara (ör. $8 \times 8 = 64$ bağımsız blok) bölünür.
2. **Kontrast Kırpma (Contrast Limiting / Clip Limit):** Her karonun histogramında belirli bir eşiği (`clipLimit`, ör. $2.5$) aşan piksel yığınları kırpılır ve kırpılan miktar tüm kutucuklara eşit olarak dağıtılır. Bu adım **arka plan gürültüsünün patlamasını engeller**.
3. **Yerel Eşitleme:** Her karo kendi içinde eşitlenir.
4. **Çift Doğrusal Enterpolasyon (Bilinear Interpolation):** Karoların sınır çizgilerinde oluşabilecek kareli yapaylıkları (blocking artifacts) yok etmek için komşu pikseller enterpolasyonla pürüzsüzce harmanlanır.

#### E. Shannon Bilgi Entropisi (Shannon Entropy)
Görüntünün taşıdığı ortalama bilgi miktarını (zenginliğini) bit cinsinden ölçer:

$$H = -\sum_{k=0}^{255} P(r_k) \cdot \log_2(P(r_k) + \epsilon)$$
- Entropi ne kadar yüksekse ($> 5.5$ bit), görüntüde o kadar fazla doku, detay ve ayrıştırılabilir gri seviye vardır!

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Renkli Görüntüde RGB Kanallarını Ayrı Ayrı Eşitlemek (Renk Kayması Faciası):**
   Eğer R, G ve B kanallarına bağımsız olarak `cv2.equalizeHist()` uygularsanız, kanalların birbirine olan renk oranları (chrominance) tamamen bozulur. Yüzler yeşile, gökyüzü mora dönebilir!
   - *Endüstriyel Çözüm:* Görüntü **YCrCb** veya **LAB** renk uzayına dönüştürülür. Renk kanallarına dokunulmaz; **yalnızca Aydınlık ($Y$ veya $L$) kanalı eşitlenir** ve tekrar BGR'ye çevrilir!
2. **Aşırı Yüksek Clip Limit Seçimi:**
   CLAHE'de `clipLimit` parametresini $10.0$ gibi çok yüksek seçmek, onu tekrar global eşitlemeye yaklaştırır ve parazitleri canlandırır. Güvenli aralık $2.0 - 4.0$'tür.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
                    Düşük Kontrastlı Kamera Görüntüsü
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    HistogramHesaplayici   │
                      │ (Histogram, CDF, Entropi) │
                      └─────────────┬─────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│     Global Histogram Eşitleme   │           │         CLAHE İyileştirme       │
│    - CDF Tabanlı Yayma          │           │   - 8x8 Izgara Karoları         │
│    - Renkli ise: YCrCb (Y Kanal)│           │   - Tepe Kırpma (clipLimit=2.5) │
│                                 │           │   - Çift Doğrusal Enterpolasyon │
│                                 │           │   - Renkli ise: LAB (L Kanalı)  │
└────────────────┬────────────────┘           └────────────────┬────────────────┘
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │  HistogramGorsellestirici │
                          │     (2x3 Analiz Paneli)   │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          [ciktilar/histogram_analiz_
                                  raporu.png]
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

1. **[`src/histogram_motoru.py`](./src/histogram_motoru.py):**
   - `HistogramHesaplayici`: 1B kanal histogramı, RGB 3-kanal histogramı, normalize CDF eğrisi ve Shannon Entropisi / RMS Kontrast hesaplar.
   - `KontrastIyilestirici`: Renk uzayı korumalı (YCrCb / LAB) Global Histogram Eşitleme ve CLAHE motoru.
2. **[`src/gorsellestirici.py`](./src/gorsellestirici.py):**
   - `HistogramGorsellestirici`: 3 görseli ve altlarında hem çubuk histogramlarını hem de ikiz eksende kırmızı CDF eğrilerini içeren 2x3 panel üretir.
3. **[`ana_akis.py`](./ana_akis.py):**
   - Düşük kontrastlı karanlık X-ray / kalite kontrol test görüntüsü üreterek metrikleri hesaplayan ve kıyaslayan ana yürütücü.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
==========================================================================
>>> AŞAMA 1: Düşük Kontrastlı Sentetik Görselin İncelenmesi
==========================================================================
[+] Çözünürlük         : (256, 256)
[+] Min - Max Piksel   : [35.0, 230.0]
[+] Dinamik Aralık     : 195.0
[+] RMS Kontrast (Std) : 31.95
[+] Shannon Entropisi  : 4.916 bit

==========================================================================
>>> AŞAMA 2: Global Histogram Eşitleme (Global Equalization)
==========================================================================
[V] Tüm görselin kümülatif olasılık fonksiyonu (CDF) doğrusal dağıtıldı.
[+] Yeni Dinamik Aralık: 255.0
[+] Yeni RMS Kontrast  : 76.22
[+] Yeni Entropi       : 4.916 bit

==========================================================================
>>> AŞAMA 3: CLAHE (Kontrast Sınırlı Uyarlanabilir Eşitleme)
==========================================================================
[V] Görüntü 8x8 karolara bölündü, yerel kontrast dengelendi ve parazit kırpıldı.
[+] Yeni Dinamik Aralık: 202.0
[+] Yeni RMS Kontrast  : 32.9
[+] Yeni Entropi       : 5.629 bit

==========================================================================
>>> AŞAMA 4: Karşılaştırmalı Metrik Tablosu
==========================================================================
Yöntem Adı                | Dinamik Aralık   | RMS Kontrast   | Shannon Entropisi
--------------------------------------------------------------------------
Ham Görsel                | 195.0            | 31.95          | 4.916 bit
Global Eşitleme           | 255.0            | 76.22          | 4.916 bit
CLAHE (Adaptive)          | 202.0            | 32.9           | 5.629 bit
--------------------------------------------------------------------------

==========================================================================
>>> AŞAMA 5: 2x3 Histogram & CDF Raporunun Kaydedilmesi
==========================================================================
[V] Karşılaştırma raporu kaydedildi: histogram_analiz_raporu.png
[V] Tam Yol: day-09-image-histogram-analyzer/ciktilar/histogram_analiz_raporu.png

[V] Day 9: Görüntü Histogramı Analizörü ve Kontrast İyileştirme tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Histogram Eşleştirme / Belirleme (Histogram Matching / Specification)**

Bazen görüntünün histogramını rastgele tekdüze (flat) yapmak istemeyiz; **belirli bir referans görselin renk/kontrast dağılımını hedef görsele aktarmak** isteriz (ör. gündüz çekilmiş bir fotoğrafı gün batımı tonlarına uydurmak).

### Görev Tanımı:
[`src/histogram_motoru.py`](./src/histogram_motoru.py) içerisindeki `KontrastIyilestirici` sınıfına şu metodu eklemeni bekliyorum:

```python
@classmethod
def histogram_eslestir(
    cls,
    kaynak_gorsel: np.ndarray,
    referans_gorsel: np.ndarray
) -> np.ndarray:
```

### Beklenen Kurallar:
1. Kaynak ve referans görsellerin CDF eğrilerini ($CDF_{\text{kaynak}}$ ve $CDF_{\text{referans}}$) hesaplamalıdır.
2. Her bir gri seviye $s$ için, $CDF_{\text{referans}}(z) \approx CDF_{\text{kaynak}}(s)$ olacak şekilde en yakın eşleşen $z$ değerini bularak $256$ elemanlı bir arama tablosu (Look-Up Table - LUT) kurmalıdır.
3. `cv2.LUT()` fonksiyonu ile kaynak görseli bu arama tablosundan geçirerek referansın kontrast karakteristiğine dönüştürmelidir.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> Renkli bir fotoğrafta kontrastı artırmak istediğimizde, neden görseli doğrudan **RGB** kanallarında eşitlemek yerine önce **LAB** veya **YCrCb** renk uzayına çevirip sadece **L** veya **Y** kanalını eşitleriz? Eğer birisi BGR kanallarına tek tek bağımsız `cv2.equalizeHist()` uygularsa elde edeceği görüntüde nasıl bir görsel bozulma (artefact) meydana gelir?

---

## 📂 Dizin Yapısı

```
day-09-image-histogram-analyzer/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (opencv-python, numpy, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve görselleştirme üretim akışı
├── ciktilar/                   # Üretilen 2x3 histogram raporu
│   └── histogram_analiz_raporu.png
├── src/
│   ├── __init__.py
│   ├── histogram_motoru.py     # HistogramHesaplayici ve KontrastIyilestirici sınıfları
│   └── gorsellestirici.py      # Matplotlib 2x3 histogram & CDF panel çizici
└── testler/
    └── test_histogram.py       # 7 adet birim testi (7 passed in 1.94s)
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
python -m pytest testler/test_histogram.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
