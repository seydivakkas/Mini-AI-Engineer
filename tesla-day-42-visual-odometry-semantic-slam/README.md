# 🚗 Tesla FSD Otonom Sürüş | Gün 42: Görsel Odometri (Visual Odometry) ve Semantik SLAM

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![VisualOdometry](https://img.shields.io/badge/SLAM-3D--to--2D%20PnP%20+%20RANSAC-red.svg?style=flat-square)](https://www.tesla.com/)
[![Semantic](https://img.shields.io/badge/AI-Semantic%20Dynamic%20Masking-blue.svg?style=flat-square)](https://www.sae.org/)
[![LoopClosure](https://img.shields.io/badge/Optimization-Loop%20Closure%20Drift%20Reset-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"42. günümüze hoş geldin stajyer!  
> HD Haritalara (High Definition Pre-Built Maps) bağımlı kalmadan, bilinmeyen bir şehirde veya karmaşık bir kavşakta Tesla aracının santimetre hassasiyetinde nerede olduğunu ve yolun nasıl devam ettiğini kestirmesi **Görsel SLAM (Simultaneous Localization and Mapping)** ile mümkündür:  
> 1. **3D-to-2D PnP (Perspective-n-Point):** Kamera koordinatlarında görülen 2D pikseller ile 3D dünya noktaları eşleştirilerek aracın rotasyonu ($R$) ve ötelemesi ($t$) hesaplanır.  
> 2. **RANSAC Outlier Ayıklama:** Hatalı eşleşen piksel gürültüleri elenerek $< 1.5\text{ piksel}$ yeniden izdüşüm hatası elde edilir.  
> 3. **Semantik Dinamik Nesne Maskelemesi:** Yanımızda hareket eden otobüs veya yayalar harita noktası olarak eklenirse SLAM sapar. Derin öğrenme segmentasyonu ile dinamik nesneler maskelenir ve yalnızca statik binalar/zemin kullanılır.  
> 4. **Döngü Kapatma (Loop Closure):** Araç daha önce geçtiği bir sokağa geri döndüğünde görsel parmak izini tanır ve yol boyunca biriken kümülatif yönelim sürüklenmesini anında sıfırlar.  
> Bugün haritasız otonom navigasyonun beynini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 3D-to-2D Kamera Perspektif İzdüşüm Modeli

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \mathbf{K} \cdot \left( \mathbf{R} \cdot \mathbf{P}_{\text{world}} + \mathbf{t} \right)$$

$$u = \frac{f_x \cdot X_c}{Z_c} + c_x, \quad v = \frac{f_y \cdot Y_c}{Z_c} + c_y$$

### 2. Yeniden İzdüşüm Hatası (Reprojection Error) ve Optimizasyon

$$e_{\text{reproj}} = \sum_{i=1}^N \left\| \mathbf{u}_i - \pi(\mathbf{K}, \mathbf{R}, \mathbf{t}, \mathbf{P}_i) \right\|^2$$

$$\arg\min_{\mathbf{R}, \mathbf{t}} e_{\text{reproj}}$$

### 3. Döngü Kapatma (Loop Closure) Mesafe Kriteri

$$d_{\text{loop}} = \left\| \mathbf{t}_{\text{current}} - \mathbf{t}_{\text{keyframe}_k} \right\| < \tau_{\text{loop}} \implies \text{Drift Reset}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Önceden taranmış pahalı LiDAR HD haritalarına ihtiyaç duymadan, aracın kendi kameralarıyla anlık olarak hem haritayı inşa edip hem de içinde konumlanmasını sağlamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Tekerlek Kayması Bağımsızlığı:** Tekerlekler buzda patinaj yapsa dahi görsel zemin akışı ile gerçek yer hareketi kusursuz kestirildi.
- **Dinamik Nesne Ayrıştırması:** Yan şeritte giden araçların SLAM algoritmasını yanıltması semantik maskeleme ile engellendi.
- **Kümülatif Drift Sıfırlama:** Loop Closure sayesinde 50 km'lik sürüş sonunda biriken sapmalar başlangıç noktasına dönüldüğünde sıfırlandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Doku Fakirliği:** Gece ışıksız düz asfalt veya bembeyaz karla kaplı yollarda görsel köşe/öznitelik sayısı düşebilir.
- **Hesaplama Yükü:** Yüzlerce anahtar kare ile küresel Bundle Adjustment (BA) NPU ve CPU'yu yoğun kullanır.

### 4. Alternatifler Nelerdir? (Alternatives)
- **LiDAR SLAM:** Yüksek geometrik doğruluk sunar ancak çok pahalıdır ve semantik bilgi taşımaz.
- **Wheel + IMU Dead Reckoning:** Düşük hesaplama gücü ister ancak zamanla sınırsız sürüklenir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Visual Odometry (VO)** | Ardışık kamera karelerindeki görsel değişimleri inceleyerek aracın 6-DoF pozunu kestirme. |
| **SLAM** | Eşzamanlı Konumlandırma ve Haritalama (Simultaneous Localization and Mapping). |
| **PnP (Perspective-n-Point)** | Bilinen $n$ adet 3D dünya noktası ve 2D kamera izdüşümünden kamera pozunu $[R|t]$ çözme problemi. |
| **RANSAC** | Rastgele örnekleme ile gürültülü veriler içindeki en tutarlı model alt-kümesini (Inlier) bulan algoritma. |
| **Reprojection Error** | 3D harita noktasının tahmin edilen kamera matrisiyle düzleme izdüşümü ile gerçek 2D piksel arasındaki fark. |
| **Keyframe (Anahtar Kare)**| Yeterli öteleme veya dönüş gerçekleştiğinde haritaya referans olarak kaydedilen kritik kamera karesi. |
| **Loop Closure** | Aracın daha önce bulunduğu bir konuma tekrar geldiğini tespit edip tüm harita grafını optimize etme işlemi. |
| **Semantic Masking** | Derin öğrenme ile dinamik nesneleri (arabalar, yayalar) tespit edip SLAM öznitelik havuzundan çıkarma. |
| **Bundle Adjustment (BA)** | Tüm kamera pozlarını ve 3D nokta konumlarını eşzamanlı olarak optimize eden doğrusal olmayan en küçük kareler yöntemi. |
| **Pose Graph** | Anahtar karelerin düğüm, bağıl kamera dönüşümlerinin kenar olduğu küresel optimizasyon grafı. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • HD Haritaya ihtiyaç duymadan sıfırdan haritalama    | • Dokusuz ortamlarda (sis, kar) öznitelik kaybı       |
| • Semantik filtreleme ile hareketli araç direnci     | • Yüksek çözünürlüklü optimizasyon işlemci yükü       |
| • < 1.0 px yeniden izdüşüm hassasiyeti                |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • BEV Transformer ve 3D Voxel Occupancy ile           | • Aniden değişen aşırı aydınlatma koşulları           |
|   birleşerek uçtan uca FSD dünya modeli oluşturma     |   (Tünel çıkışı ani parlama)                          |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Görsel Odometri ve Semantik SLAM Akışı

```
[ Ham Kamera Karesi ] ===> [ Derin Öğrenme Segmentasyonu ] ===> [ Statik Zemin Maskesi ]
                                                                        |
                                                                        v
                                                          [ 2D-3D Öznitelik Eşleme ]
                                                                        |
                                                                        v
                                                          [ PnP + RANSAC Poz Kestirimi ]
                                                                        |
                                                                        v
                                                        [ Keyframe & Loop Closure Denetimi ]
                                                                        |
                                                                        v
                                                        [ Sürüklenmesiz Küresel Yörünge ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Görsel Odometri ve SLAM simülasyon akışını çalıştırın
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
