# Day 13: Geometrik Dönüşümler ve Perspektif Düzeltme (Perspective Correction & Homography)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; mobil belge tarama uygulamalarında (CamScanner vb.), plaka tanıma sistemlerinde (ALPR) ve endüstriyel robotik montaj hatlarında açılı ve çarpık duran nesneleri **projektif geometri, 4 nokta köşe sıralama algoritması ve 3x3 Homografi matrisi ($H$) ile tam ortogonal kuşbakışı (Bird's-Eye / Orthographic View) düzleme taşıyan** üretim seviyesinde bir geometrik düzeltme motorudur.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Bir kamera ile masadaki faturanın, duvardaki tablonun, yerdeki bir halının veya hareket halindeki bir aracın plakasının fotoğrafı çekildiğinde:
- Nesne 3 boyutlu dünyada dikdörtgen olsa bile, **iğne deliği kamera projeksiyonu (Pinhole Camera Model)** nedeniyle 2 boyutlu sensör düzleminde **çarpık bir yamuk (Quadrilateral)** olarak belirir.
- Paralel çizgiler sonsuzdaki kaçış noktalarında (Vanishing Points) kesişir.
- Bu çarpıklık giderilmeden yapılan OCR (Optik Karakter Tanıma), desen eşleştirme veya boyut ölçümü **hatalı sonuçlar üretir.**

**Kullanım Alanları:**
- **Mobil Belge Tarama:** Masaya eğik konmuş makbuz veya kimlik kartını taranmış PDF gibi düzleştirmek.
- **Otonom Araçlar ve Robotik:** Kamera görüntüsünü "Kuşbakışı Görünüm" (Bird's-Eye View - IPM) haline getirerek şerit çizgilerini ve zemin engellerini ölçmek.
- **Spor Analitiği:** Sahadaki oyuncuların taktiksel konumlarını belirlemek için açılı TV kamerasını 2B saha planına izdüşürmek.

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Açılı çekilmiş belgeleri ve düzlemsel yüzeyleri 4 köşe noktası ve perspektif dönüşüm matrisi ile kuş bakışı (top-down) görünüme getirmek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Kamera açısının neden olduğu yamukluk ve distorsiyonları gidererek OCR ve desen analizinin kusursuz çalışmasını sağlar.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Yüzey düzlemsel (planar) olmadığında (örneğin bükülmüş kağıt) 2D homografi matrisi yetersiz kalır.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- Thin Plate Spline (TPS), 3D Mesh Düzeltme veya Derin Belge Düzeltme Ağları (DocUNet).

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Homografi Matrisi** | *Homography Matrix ($3 \times 3$)* | Düzlemsel bir yüzeyin bir kamera açısından başka bir kamera açısına projektif dönüşümünü tanımlayan 8 serbestlik dereceli matris. |
| **Perspektif Dönüşümü** | *Perspective Warp Transform* | Açılı çekilmiş bir doküman veya halıyı dik kuşbakışı (top-down) görünümüne getiren 2D projektif haritalama. |
| **Köşe Sıralama Algoritması** | *4-Point Convex Polygon Ordering* | Tespit edilen 4 köşe noktasını saat yönünde (Sol-Üst, Sağ-Üst, Sağ-Alt, Sol-Alt) sıralayan geometrik algoritma. |
| **Bilineer İnterpolasyon** | *Bilinear Interpolation* | Dönüşüm sonrası kesirli piksel koordinatlarındaki renk değerini komşu 4 pikselin ağırlıklı ortalamasıyla hesaplayan enterpolasyon. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. Homojen Koordinatlar ve Projektif Düzlem
2 boyutlu Kartezyen düzlemindeki bir $(x, y)$ noktası, projektif geometride 3 boyutlu homojen koordinat vektörü olarak temsil edilir:
$$\mathbf{x} = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

#### B. 3x3 Homografi Matrisi ($H$)
İki düzlem arasındaki perspektif izdüşümü $3 \times 3$ boyutunda bir homografi matrisi $H$ ile modellenir:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

- **Serbestlik Derecesi (DOF):** Matriste 9 eleman vardır; ancak ölçek serbestisi nedeniyle ($h_{33} = 1$) serbestlik derecesi **8'dir**.
- **Neden En Az 4 Nokta Çifti Gerekir?**
  Her nokta eşleşmesi $(x_i, y_i) \to (x'_i, y'_i)$ kartezyen denklem sistemine 2 bağımsız lineer denklem kazandırır:
  $$x'_i = \frac{h_{11}x_i + h_{12}y_i + h_{13}}{h_{31}x_i + h_{32}y_i + 1}, \quad y'_i = \frac{h_{21}x_i + h_{22}y_i + h_{23}}{h_{31}x_i + h_{32}y_i + 1}$$
  8 bilinmeyeni tekil olarak çözebilmek için: $4 \times 2 = 8$ denklem, yani **doğrudaş olmayan (non-collinear) tam 4 nokta çifti** şarttır!
  Bu denklem sistemi Doğrudan Lineer Dönüşüm (DLT - Direct Linear Transformation) ve Tekil Değer Ayrışımı (SVD) ile çözülür (`cv2.getPerspectiveTransform`).

#### C. 4 Nokta Matematiksel Sıralama Algoritması
Köşeler rastgele sırayla geldiğinde dönüşümün kelebek gibi bükülmemesi için saat yönünde sıralanmalıdır:
1. **Sol-Üst (Top-Left):** $x + y$ koordinat toplamının **en küçük** olduğu nokta.
2. **Sağ-Alt (Bottom-Right):** $x + y$ koordinat toplamının **en büyük** olduğu nokta.
3. **Sağ-Üst (Top-Right):** $y - x$ koordinat farkının **en küçük** olduğu nokta.
4. **Sol-Alt (Bottom-Left):** $y - x$ koordinat farkının **en büyük** olduğu nokta.

#### D. Hedef Kuşbakışı Boyutların Hesaplanması
Perspektif bozulmada yakın kenar geniş, uzak kenar dar görünür. Düzeltilmiş çıktının çözünürlüğü bozulmamış maksimum Öklid uzunluklarıyla belirlenir:
$$W_{\text{hedef}} = \max\left(\|\text{Sağ-Alt} - \text{Sol-Alt}\|_2, \|\text{Sağ-Üst} - \text{Sol-Üst}\|_2\right)$$
$$H_{\text{hedef}} = \max\left(\|\text{Sağ-Üst} - \text{Sağ-Alt}\|_2, \|\text{Sol-Üst} - \text{Sol-Alt}\|_2\right)$$

#### E. Tersine Eşleme (Backward Mapping) ve Bi-kübik İnterpolasyon
Görüntü dönüştürülürken kaynak pikseller hedef piksele ileriye doğru taşınırsa (Forward Mapping) hedefte delikler ve boşluklar kalır.
Bu yüzden `cv2.warpPerspective` **Tersine Eşleme (Backward Mapping)** kullanır:
Hedef görüntüdeki her $(x', y')$ koordinatı için kaynak koordinat $\mathbf{x} = H^{-1} \mathbf{x}'$ hesaplanır ve çevredeki 16 piksel kullanılarak **Bi-kübik (Bicubic) interpolasyon** ile pürüzsüz piksel değeri üretilir.

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Köşe Sıralamasının Karışması:**
   Eğer köşe noktalarını rastgele sırayla verirseniz, homografi matrisi görüntüyü çapraz bükerek ters yüz eder.
2. **Tekil Matris (Singular Matrix / Determinant Sıfır):**
   Eğer seçilen 4 noktadan 3 tanesi aynı doğru üzerindeyse (Collinear), homografi matrisinin determinantı sıfıra yaklaşır ve tersi alınamaz (DLT çöker).
3. **Affin ve Perspektif Ayrımı:**
   Affin dönüşüm 6 DOF'a sahiptir ve paralel çizgileri paralel tutar ($h_{31}=h_{32}=0$). Perspektif dönüşüm ise 8 DOF'a sahiptir ve paralel çizgilerin kesişmesine (kaçış noktalarına) izin verir.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
                      Açılı / Çarpık Görüntü (Kamera Çekimi)
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    4 Köşe Koordinatları     │
                      │    (Rastgele Sıralı Giriş)  │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │     noktalari_sirala()      │
                      │  [Sol-Üst, Sağ-Üst,         │
                      │   Sağ-Alt, Sol-Alt]         │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  hedef_boyutlari_hesapla()  │
                      │   W = max(L_alt, L_ust)     │
                      │   H = max(L_sag, L_sol)     │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  Homografi Matrisi Çözümü   │
                      │    H = getPerspective...    │
                      │   (3x3 Projektif Matris)    │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │       warpPerspective       │
                      │    (Ters Eşleme & Bi-kübik) │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  PerspektifGorsellestirici  │
                      │  - Açılı Görsel + Köşeler   │
                      │  - Kuşbakışı Çıktı          │
                      │  - 3x3 H Isı Haritası       │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                   [ciktilar/perspektif_duzeltme_paneli.png]
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

1. **[`src/perspektif_duzeltici.py`](./src/perspektif_duzeltici.py):**
   - `PerspektifDuzeltici`: 4 köşe sıralama algoritması ($x+y$ ve $y-x$), hedef kuşbakışı boyut kestirimi, 3x3 homografi çözümü ve affin açı eğikliği düzeltici (Deskewing).
2. **[`src/gorsellestirici.py`](./src/gorsellestirici.py):**
   - `PerspektifGorsellestirici`: Orijinal açılı sahneyi köşe etiketleriyle, kuşbakışı düzeltilmiş ortogonal çıktıyı ve 3x3 homografi matrisinin ısı haritasını 3 panelli Matplotlib çizelgesi olarak kaydeder.
3. **[`ana_akis.py`](./ana_akis.py):**
   - Ahşap parke zemin üzerine 3D açıyla yerleştirilmiş sentetik motifli halı sahnesini simüle eden ve perspektif düzeltmeyi yürüten konsol yürütücüsü.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
==========================================================================
>>> AŞAMA 1: Perspektif Bozulmalı Sentetik Sahnenin İncelenmesi
==========================================================================
[+] Sahne Çözünürlüğü         : 450 x 400 piksel
[+] Tespit Edilen Bozuk Köşeler (Rastgele Giriş):
    * Nokta #1: (x=90.0, y=60.0)
    * Nokta #2: (x=340.0, y=95.0)
    * Nokta #3: (x=390.0, y=360.0)
    * Nokta #4: (x=45.0, y=325.0)

==========================================================================
>>> AŞAMA 2: 4 Köşe Noktasının Saat Yönünde Matematiksel Sıralanması
==========================================================================
[V] Sol-Üst (Top-Left)        -> (x=90.0, y=60.0)
[V] Sağ-Üst (Top-Right)       -> (x=340.0, y=95.0)
[V] Sağ-Alt (Bottom-Right)    -> (x=390.0, y=360.0)
[V] Sol-Alt (Bottom-Left)     -> (x=45.0, y=325.0)

==========================================================================
>>> AŞAMA 3: Hedef Kuşbakışı Çözünürlüğünün Hesaplanması
==========================================================================
[+] Hesaplanmış İdeal Genişlik (W) : 347 piksel
[+] Hesaplanmış İdeal Yükseklik (H): 270 piksel
[+] En/Boy Oranı (Aspect Ratio)    : 1.285

==========================================================================
>>> AŞAMA 4: 3x3 Homografi Matrisinin Çözümü ve Görüntü Dönüşümü
==========================================================================
[+] 3x3 Homografi Projeksiyon Matrisi (H):
    [   1.3782e+00    2.3404e-01   -1.3808e+02 ]
    [  -2.0405e-01    1.4575e+00   -6.9086e+01 ]
    [  -3.6065e-04    1.4960e-03    1.0000e+00 ]
[V] Homografi Determinantı: 2.1744e+00 (Tersinir ve geçerli dönüşüm!)
[V] Düzeltilmiş Çıktı Boyutu: 347 x 270 px

==========================================================================
>>> AŞAMA 5: Analiz Panelinin ve Isı Haritasının Kaydedilmesi
==========================================================================
[V] Perspektif analiz paneli kaydedildi: perspektif_duzeltme_paneli.png
[V] Kayıt Konumu: day-13-perspective-correction/ciktilar/perspektif_duzeltme_paneli.png

[V] Day 13: Geometrik Dönüşümler ve Perspektif Düzeltme başarıyla tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Otomatik Belge Dörtgeni Tespiti (Canny + approxPolyDP)**

Kullanıcının köşe noktalarını elle tıklaması yerine, bir belgenin/kartın 4 köşesini görüntüden otomatik olarak yakalayan algoritmayı kurmanı bekliyorum.

### Görev Tanımı:
[`src/perspektif_duzeltici.py`](./src/perspektif_duzeltici.py) sınıfına şu metodu eklemeni bekliyorum:

```python
@classmethod
def otomatik_kose_tespiti(cls, gorsel_bgr: np.ndarray) -> Optional[np.ndarray]:
```

### Beklenen Adımlar:
1. Görüntüyü gri tonlamaya çevirip Gauss bulanıklığı (`cv2.GaussianBlur`) uygula.
2. Canny kenar dedektörü ile kenarları bul.
3. `cv2.findContours` ile en büyük dış konturu seç.
4. `cv2.arcLength` ve **Douglas-Peucker algoritması (`cv2.approxPolyDP`)** ile konturu sadeleştir (`epsilon = 0.02 * cevre`).
5. Eğer elde edilen çokgen tam 4 köşeli ve dışbükey (convex) ise `(4, 2)` boyutunda köşe noktalarını döndür; aksi halde `None` dön.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> $3 \times 3$ Homografi matrisindeki **$h_{31}$ ve $h_{32}$ katsayıları** geometrik olarak ne anlama gelir?  
> Neden bu iki katsayı sıfır olduğunda ($h_{31} = 0, h_{32} = 0$) dönüşüm **Affin Dönüşüme (Affine Transformation)** indirgenir ve paralel çizgiler paralel kalmaya devam eder?

---

## 📂 Dizin Yapısı

```
day-13-perspective-correction/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (opencv-python, numpy, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve görsel üretim akışı
├── ciktilar/                   # Üretilen 3 panelli analiz paneli
│   └── perspektif_duzeltme_paneli.png
├── src/
│   ├── __init__.py
│   ├── perspektif_duzeltici.py # Köşe sıralama, homografi ve deskewing motoru
│   └── gorsellestirici.py      # 3 panelli Matplotlib çizelge motoru
└── testler/
    └── test_perspektif.py      # 7 adet birim testi (7 passed in 1.06s)
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
python -m pytest testler/test_perspektif.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
