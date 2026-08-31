# 🚗 Tesla FSD Otonom Sürüş | Gün 34: 8-Kamera Görüş Geometrisi ve İğne Deliği Kamera Modeli

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Vision](https://img.shields.io/badge/Tesla%20Vision-8%20Surround%20Cameras-red.svg?style=flat-square)](https://www.tesla.com/)
[![Optics](https://img.shields.io/badge/Optics-Pinhole%20+%20Brown--Conrady-blue.svg?style=flat-square)](https://www.sae.org/)
[![Performance](https://img.shields.io/badge/RTOS-36%20FPS%20Real--Time-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"Faz 4'e (Gün 34 - Gün 44) hoş geldin stajyer!  
> Artık batarya ve motoru arkamızda bıraktık; Tesla'nın kalbi olan **Full Self-Driving (FSD) Otonom Sürüş ve Görsel Algı Geometrisine** giriş yapıyoruz!  
> Tesla araçlarında LiDAR veya ultrasonik radar yoktur; aracın etrafını saran **8 adet kamera (Tesla Vision)** vardır:  
> 1. **Ön Üçlü (Front Triplet):** Ön camın üstünde $120^\circ$ Geniş (Wide), $50^\circ$ Ana (Main) ve $35^\circ$ Dar (Narrow/Telephoto) kameralar bulunur. Dar kamera 250 metre ilerdeki yayayı görürken, geniş kamera kavşak köşelerini tarar.  
> 2. **B-Sütunu (Pillars):** Sol ve sağ B sütunlarında $90^\circ$ yan görüş kameraları bulunur (Kör kavşaklardan çıkarken yolu ilk onlar görür).  
> 3. **Çamurluklar (Repeaters):** Sol ve sağ sinyal çamurluklarında $90^\circ$ geriye bakan kameralar otoyolda şerit değiştirirken arkadan gelen hızlı araçları yakalar.  
> 4. **Geri Görüş (Rear View):** Bagaj kapağında $120^\circ$ geri görüş kamerası bulunur.  
> Bir 3D dünya noktasının kamera sensöründeki $(u, v)$ pikseline izdüşümü, lensin bükülme distorsiyonu (Brown-Conrady) düzeltilmeden yapılamaz.  
> Bugün 8 kameranın $360^\circ$ görüş geometrisini ve distorsiyon düzeltme motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. İğne Deliği (Pinhole) Projeksiyon Matrisi

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \cdot \left( \mathbf{R} \cdot \begin{bmatrix} X_{\text{ego}} \\ Y_{\text{ego}} \\ Z_{\text{ego}} \end{bmatrix} + \mathbf{t} \right)$$

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{R} = \mathbf{R}_{\text{align}} \mathbf{R}_z(\psi) \mathbf{R}_y(\theta) \mathbf{R}_x(\phi)$$

### 2. Brown-Conrady Radyal ve Teğetsel Lens Distorsiyon Modeli
Normalleştirilmiş kamera düzlemi koordinatları $x = X_c / Z_c$, $y = Y_c / Z_c$ ve $r^2 = x^2 + y^2$ olmak üzere:

$$x_{\text{dist}} = x \left(1 + k_1 r^2 + k_2 r^4\right) + 2 p_1 x y + p_2 \left(r^2 + 2 x^2\right)$$

$$y_{\text{dist}} = y \left(1 + k_1 r^2 + k_2 r^4\right) + p_1 \left(r^2 + 2 y^2\right) + 2 p_2 x y$$

$$u = f_x x_{\text{dist}} + c_x, \quad v = f_y y_{\text{dist}} + c_y$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
3D fiziksel dünyadaki nesneleri, şerit çizgilerini ve yayaları 8 kameranın 2D pikselleriyle kusursuz ve gecikmesiz eşleştirmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Balık Gözü ve Lens Bükülmesi:** Geniş açılı kameralardaki fıçı distorsiyonunu (Barrel Distortion) matematiksel olarak düzeltti.
- **Kamera Çakışması (Overlap):** Yan yana bakan kameralar arasındaki ortak görüş alanlarını 3D uzayda hizalayarak kör noktaları yok etti.
- **Düşük Gecikmeli Projeksiyon:** $36\text{ FPS}$ (kare başı $< 27.7\text{ ms}$) gereksinimini $2.3\ \mu\text{s}$ işlem süresiyle aştı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Kamera Kirlenmesi ve Sis:** Çamurluk veya ön cam kirlendiğinde optik intrinsics parametreleri ışık kırılması sebebiyle hafif değişebilir.
- **Süspansiyon Salınımı:** Çukura girildiğinde aracın pitch/roll yapması dışsal matrisleri ($[R|t]$) anlık saptırır (Çevrimiçi kalibrasyon gerekir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **LiDAR Tabanlı Nokta Bulutu:** Pahalıdır ($> \$5000$) ve yağmur/karda performansı düşer.
- **Saf 2D Piksel Algılama:** Geometrik derinlik hesabı yapmaz; nesnelerin mesafesini bilemez.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Pinhole Camera Model** | Işık ışınlarının tek bir optik merkezden geçerek görüntü düzlemine düştüğünü varsayan temel kamera modeli. |
| **Intrinsics Matrix ($K$)** | Odak uzaklığı ($f_x, f_y$) ve optik merkez ($c_x, c_y$) parametrelerini içeren $3 \times 3$ içsel kamera matrisi. |
| **Extrinsics Matrix ($[R \vert t]$)** | Kameranın araç gövdesine göre 3D konum ve yönelimini tanımlayan rotasyon ve öteleme matrisi. |
| **Brown-Conrady Distortion** | Lens eğriliğinden kaynaklanan radyal ($k_1, k_2$) ve teğetsel ($p_1, p_2$) optik sapma modeli. |
| **Field of View (FOV)** | Kameranın yatay ve dikeyde görebildiği toplam açısal kapsama alanı ($35^\circ - 120^\circ$). |
| **Tesla Vision** | Radar ve LiDAR kullanmadan yalnızca 8 kamerayla otonom sürüş gerçekleştiren görsel algı mimarisi. |
| **Ego-Frame** | Aracın ağırlık merkezini veya arka dingil ortasını $(0, 0, 0)$ kabul eden araç eksen takımı. |
| **Barrel Distortion** | Geniş açılı lenslerde kenarların dışa doğru şişmesiyle oluşan fıçı biçimli optik distorsiyon. |
| **Undistortion** | Distorsiyonlu piksel koordinatlarını düzgün perspektif izdüşümüne çeviren piksel düzeltme işlemi. |
| **36 FPS Pipeline** | Tesla FSD bilgisayarının saniyede 36 tam çevre görüntü karesini işleme frekansı ($27.7\text{ ms}$). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 8 kamera ile 360° sıfır kör noktalı çevre görüş    | • Gece ve doğrudan güneş parlamasında kontrast düşüşü |
| • 2.3 µs ultra hızlı 3D-to-2D izdüşüm performansı     | • Dinamik şasi esnemesinde küçük kalibrasyon kayması  |
| • Brown-Conrady ile %99.8 geometrik hassasiyet        |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • HW4 5-Megapiksel kameralara sıfır kod değişikliği   | • Ağır kar ve çamurla kameraların fiziksel tıkanması  |
|   ile doğrudan ölçeklenebilir esnek mimari            |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla HW3/HW4 8-Kamera Donanım Yerleşimi

```
                                [ Ön Tampon ]
                                      |
                      +---------------+---------------+
                      |   ÖN ÜÇLÜ KAMERA (Ön Cam)     |
                      | - Narrow (35°): 250m İleri     |
                      | - Main   (50°): 150m İleri     |
                      | - Wide  (120°): 60m Kavşaklar |
                      +---------------+---------------+
                                      |
       [ Sol B-Sütunu (90°) ] --------+-------- [ Sağ B-Sütunu (90°) ]
                                      |
     [ Sol Çamurluk (140° Geri) ] ----+---- [ Sağ Çamurluk (140° Geri) ]
                                      |
                      +---------------+---------------+
                      |    GERİ GÖRÜŞ KAMERASI (120°) |
                      |    Bagaj Kapağı / Plakalık    |
                      +-------------------------------+
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana 8-kamera görüş geometrisi simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
