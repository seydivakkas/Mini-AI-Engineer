# 🚗 Tesla FSD Otonom Sürüş | Gün 37: Kuşbakışı (Bird’s Eye View - BEV) Temsili ve Homografi Projeksiyonları

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![BEV](https://img.shields.io/badge/FSD-Bird's%20Eye%20View%20(BEV)-red.svg?style=flat-square)](https://www.tesla.com/)
[![IPM](https://img.shields.io/badge/Math-Inverse%20Perspective%20Mapping-blue.svg?style=flat-square)](https://www.sae.org/)
[![Homography](https://img.shields.io/badge/Transform-3x3%20Planar%20Homography-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"37. günümüze hoş geldin stajyer!  
> İnsan sürücüler yolu ön camdan perspektif olarak görür: Yakındaki şeritler geniştir, uzaktaki şeritler tek bir noktada (Vanishing Point) birleşir. Ancak bir otonom sürüş yörünge planlayıcısının (Motion Planner) perspektif görüntüyle yön bulması imkânsızdır!  
> Aracın tekerleklerini kaç derece çevireceğini hesaplamak için yolu **Kuşbakışı (Bird's Eye View - BEV)** düzleminde, yani aracın yukarısından aşağıya doğru $100 \times 100\text{ metrelik}$ metrik bir harita olarak görmemiz gerekir:  
> 1. **Düzlemsel Homografi Matrisi ($H_{3 \times 3}$):** Yol yüzeyini düz kabul ettiğimizde ($Z_{\text{road}} = 0$), kamera pikselleri $(u, v)$ ile metrik yol koordinatları $(X, Y)$ arasında $3 \times 3$ tersinir bir dönüşüm kurulur.  
> 2. **Inverse Perspective Mapping (IPM):** Kamera montaj yüksekliği ($h_c = 1.35\text{ m}$), eğim açısı (pitch $-2^\circ$) ve odak uzaklığıyla perspektif bükülmesi tersine çevrilir. Perspektifte üçgen gibi kapanan otoyol şeritleri BEV düzleminde tam paralel $3.75\text{ metre}$ aralıklı düz çizgilere dönüşür.  
> 3. **Ufuk Çizgisi Kırpması:** Gökyüzü ve ufuk üstü pikseller sonsuzluğa gideceği için matematiksel olarak filtrelenir.  
> Bugün Tesla FSD'nin kuşbakışı yol dönüştürücüsünü ve IPM motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Zemin Düzlemi ($Z_{\text{road}} = 0$) Homografi Matrisi

$$\mathbf{H}_{\text{road} \to \text{cam}} = \mathbf{K} \cdot \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t}_{\text{cam}} \end{bmatrix}_{3 \times 3}$$

$$\mathbf{H}_{\text{cam} \to \text{road}} = \mathbf{H}_{\text{road} \to \text{cam}}^{-1}$$

### 2. Piksel Koordinatından Metrik BEV Düzlemine Dönüşüm

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \mathbf{H}_{\text{cam} \to \text{road}} \cdot \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \implies X_{\text{longitudinal}} = \frac{x'}{w'}, \quad Y_{\text{lateral}} = \frac{y'}{w'}$$

### 3. Ufuk Çizgisi Sınırı (Horizon Cutoff)

$$v_{\text{horizon}} = c_y - f_y \cdot \tan(-\theta_{\text{pitch}})$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Perspektif yanılsamasını ortadan kaldırıp yol çizgilerini, kaldırımları ve engelleri doğrudan metrik metre cinsinden ($X, Y$) haritalayarak Model Predictive Control (MPC) yörünge planlayıcısına girdi sağlamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Perspektif Daralması:** Uzaktaki şeritlerin daralarak birleşmesi sorununu ortadan kaldırdı; şeritlerin her mesafede paralel kalmasını sağladı.
- **Doğrudan Metrik Mesafe:** Piksel saymak yerine "Sol şerit $1.87\text{ metre}$ solda, viraj yarıçapı $250\text{ metre}$" şeklinde kesin fiziksel ölçümler üretti.
- **Sıfır Gecikme:** $3 \times 3$ matris çarpımı ile kare başına $12.5\ \mu\text{s}$ ultra hızlı dönüşüm sağladı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Düzlem Varsayımı (Planar Assumption):** Yokuş yukarı veya dik kasislerde zemin $Z = 0$ olmadığı için uzaktaki nesneler aşırı uzamış (Streaking Effect) görünür.
- **Yüksek Nesneler:** Ağaçlar, köprüler veya tırlar BEV düzlemine projeksiyonda yere yapışık devasa gölgeler gibi uzar.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Doğrudan 2D Piksel Tabanlı Şerit Takibi:** Viraj yarıçapını ve tekerlek açısını metrik olarak hesaplayamaz.
- **Transformer Tabanlı 3D Occupancy Network (BEVFormer):** Yükseltileri çözer fakat NPU donanım yükü daha fazladır.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Bird’s Eye View (BEV)** | Aracın ve çevresinin kuşbakışı (üstten aşağı) metrik $X, Y$ koordinat düzlemindeki temsili. |
| **Inverse Perspective Mapping (IPM)** | Kamera perspektif görüntüsünü zemin düzlemi parametreleriyle üstten görünüme çeviren geometrik dönüşüm. |
| **Planar Homography ($H$)** | İki düzlem arasındaki izdüşümsel ilişkiyi tanımlayan $3 \times 3$ tersinir projektif matris. |
| **Vanishing Point (Kaçış Noktası)** | Perspektif görüntüsünde paralel doğruların sonsuzda birleştiği ufuk noktası. |
| **Horizon Line** | Kameranın optik eğimine bağlı olarak gökyüzü ile yeryüzünü ayıran ufuk çizgisi ($v_{\text{horizon}}$). |
| **Streaking Effect** | Düzlem dışı dikey nesnelerin IPM ile geriye izdüşümünde BEV üzerinde sonsuza doğru uzaması kusuru. |
| **Ego-Centered Grid** | Aracın ağırlık merkezini $(0, 0)$ alan metrik ızgara haritası ($0.1\text{ m/piksel}$). |
| **Lane Width ($3.75\text{ m}$)** | Otoyol standartlarında şerit çizgileri arasındaki sabit fiziksel genişlik açıklığı. |
| **Pitch Angle ($\theta$)** | Kameranın ufka göre aşağı/yukarı bakış eğim açısı (Tesla'da tipik $-2^\circ$). |
| **Round-Trip Consistency** | Bir noktanın 2D'den BEV'e ve tekrar 2D'ye dönüştürüldüğünde sıfır hata vermesi doğrulaması. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Şeritleri paralel metrik uzaya kusursuz dönüştürme  | • Yokuş ve tümseklerde zemin düzlemi varsayımı sapması|
| • 12.5 µs ile saniyede 80.000+ kare işleme kapasitesi  | • Dikey nesnelerde uzama (Streaking) kusuru           |
| • Yörünge planlayıcı (MPC) ile doğrudan uyumluluk     |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • IMU dinamik pitch/roll kompanzasyonu ile yokuşlarda | • Hızlanma ve sert frenlerde aracın öne/arkaya        |
|   çevrimiçi düzlem düzeltme entegrasyonu              |   yatmasıyla anlık şerit mesafesi titremesi           |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ 2D Perspektiften Metrik BEV'e IPM Dönüşüm Mimarisi

```
       [ 2D Kamera Görüntüsü ]
       (Perspektif Şerit Çizgileri)
                   |
                   v
   +-------------------------------+
   |   Ufuk Çizgisi Kırpma (v_h)   |
   |   (Gökyüzü Noktalarını Filtrele)
   +---------------+---------------+
                   |
                   v
   +-------------------------------+
   |   Düzlemsel Homografi (H^-1)  |
   |   X_bev = x'/w', Y_bev = y'/w'|
   +---------------+---------------+
                   |
                   v
       [ Metrik BEV Yol Haritası ]
       (Paralel Şeritler, 3.75m Açıklık)
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana BEV ve Homografi simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
