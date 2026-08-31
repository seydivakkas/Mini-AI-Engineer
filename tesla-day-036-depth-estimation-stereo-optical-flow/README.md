# 🚗 Tesla FSD Otonom Sürüş | Gün 36: Derinlik Tahmini (Depth Estimation) ve Geometrik Optik Akış

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Depth](https://img.shields.io/badge/Vision-Stereo%20Disparity%20Z%3DfB%2Fd-red.svg?style=flat-square)](https://www.tesla.com/)
[![Flow](https://img.shields.io/badge/Kinematics-Lucas--Kanade%20Optical%20Flow-blue.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/AEB-Time--To--Contact%20(TTC)-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"36. günümüze hoş geldin stajyer!  
> Tesla'nın 'Pure Vision' (Saf Görsel) felsefesinde LiDAR sensörü bulunmaz. Peki bir Tesla, önündeki kamyonun $25.4\text{ metre}$ mesafede olduğunu ve $12\text{ m/s}$ hızla yaklaştığını nasıl bu kadar kesin bilir?  
> 1. **Stereo Disparity ($Z = \frac{f \cdot B}{d}$):** Yan yana iki kameranın (veya aynı kameranın hareket halindeki iki ardışık karesinin) pikselleri arasındaki kayma miktarı ($d$), mesafeyle ters orantılıdır. Yakındaki nesneler piksellerde devasa kayarken, ufuktaki dağlar neredeyse hiç kıpırdamaz.  
> 2. **Karesel Derinlik Belirsizliği ($\sigma_Z \propto Z^2$):** Optiğin temel kuralı gereği, mesafe 2 katına çıktığında derinlik hatası 4 katına ($Z^2$) fırlar. Bu yüzden uzak mesafelerde dar açılı telefoto kamera ($f = 1800\text{px}$) kullanılır.  
> 3. **Lucas-Kanade Optik Akış (Optical Flow):** Görüntüdeki parlaklık gradyanları ($I_x, I_y, I_t$) çözülerek piksellerin 2D hız vektörü kestirilir.  
> 4. **Çarpışma Süresi (Time-To-Contact - TTC):** Optik akışın merkezden dışa doğru radyal patlama hızından (Expansion Divergence), mutlak mesafeyi bilmesek bile kaç saniye sonra çarpışacağımız doğrudan hesaplanır (Otomatik Acil Fren - AEB tetikleyicisi).  
> Bugün Tesla Vision'ın derinlik ve hareket tahmin motorunu inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Stereo Disparity ve Metrik Derinlik

$$Z = \frac{f_{\text{px}} \cdot B_{\text{baseline}}}{d_{\text{disparity}}}$$

### 2. Karesel Derinlik Belirsizlik Modeli

$$\sigma_Z = \left| \frac{\partial Z}{\partial d} \right| \sigma_d = \frac{f \cdot B}{d^2} \sigma_d = \frac{Z^2}{f \cdot B} \sigma_d$$

### 3. Lucas-Kanade Diferansiyel Optik Akış Denklemi

$$\begin{bmatrix} I_x(p_1) & I_y(p_1) \\ \vdots & \vdots \\ I_x(p_n) & I_y(p_n) \end{bmatrix} \begin{bmatrix} u_t \\ v_t \end{bmatrix} = - \begin{bmatrix} I_t(p_1) \\ \vdots \\ I_t(p_n) \end{bmatrix} \implies \mathbf{v} = \left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T \mathbf{b}$$

### 4. Çarpışma Süresi (Time-To-Contact) ve Optik Genişleme

$$\text{TTC} = \frac{Z(t)}{-\dot{Z}(t)} = \frac{1}{\nabla \cdot \vec{v}_{\text{optical flow}}}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Radar ve LiDAR donanımlarına ihtiyaç duymadan, yalnızca kamera piksellerinin geometrik hareketinden ve stereo farkından milimetrik 3D mesafe ve yaklaşma hızı çıkarmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Hayalet Frenleme (Phantom Braking):** Radar sinyallerinin köprü altındaki metal levhaları araba sanıp aracı aniden durdurması sorununu görsel optik akışla çözdü.
- **Acil Fren (AEB) Güvenliği:** TTC hesabı ile $1.8\text{ saniye}$ kala acil durum frenini deterministik olarak tetikledi.
- **Karesel Belirsizlik Filtrelemesi:** Mesafeye bağlı artan $\sigma_Z$ belirsizliğini Kalman filtresinde kovaryans olarak kullanarak ölçüm hatalarını süzdü.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Dokusuz Yüzeyler (Aperture Problem):** Düz beyaz bir duvarda veya pürüzsüz asfaltta $I_x, I_y$ gradyanları sıfır olduğu için optik akış tekil hale gelebilir.
- **Uzak Mesafe Hassasiyeti:** $100\text{ metre}$ üzerinde stereo taban çizgisi ($50\text{ cm}$) yetersiz kalır (Monoküler derinlik ağı gerektirir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Fiziksel LiDAR Sensörü:** Pahalı, aerodinamiği bozan tavan çıkıntıları yaratır ve yağmurda saçılır.
- **Ultrasonik Sensörler:** Yalnızca $0-2\text{ metre}$ menzile sahiptir; yüksek hızda çalışamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Disparity ($d$)** | Bir nesnenin sol ve sağ kamera görüntülerindeki yatay piksel koordinat farkı ($u_L - u_R$). |
| **Baseline ($B$)** | İki stereo kamera arasındaki yatay fiziksel ayrıklık mesafesi ($0.5\text{ m}$). |
| **Optical Flow** | Ardışık görüntü kareleri arasında nesnelerin piksel bazlı 2D hareket vektör alanı. |
| **Lucas-Kanade** | Yerel bir piksel komşuluğunda akışın sabit olduğunu varsayan en küçük kareler tabanlı akış çözücüsü. |
| **Time-To-Contact (TTC)** | Mevcut hızla devam edildiğinde öndeki engele çarpana kadar kalan süre ($s$). |
| **Cost Volume** | Her piksel ve olası disparity adımı için eşleşme maliyetini saklayan 3D tensör. |
| **Semi-Global Matching (SGM)** | 1D çizgiler boyunca dinamik programlama ile pürüzsüz derinlik haritası üreten algoritma. |
| **Aperture Problem** | Dokusuz veya tek doğrultulu kenarlarda optik akış yönünün tekil (belirsiz) kalması durumu. |
| **Quadratic Uncertainty** | Mesafenin karesiyle ($Z^2$) orantılı olarak artan stereo derinlik hata payı. |
| **Expansion Divergence** | Bir nesneye yaklaşıldığında piksellerin dışa doğru radyal patlama yayılma hızı. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • LiDAR maliyeti olmadan santimetre hassasiyetli mesafe| • 100m üzerinde karesel belirsizliğin artması        |
| • 5.5 µs ultra hızlı Disparity-to-Depth çözümü        | • Dokusuz yüzeylerde optik akış belirsizliği          |
| • TTC ile doğrudan donanımsal AEB fren tetikleme      |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • FSD V12 Uçtan Uca Yapay Zeka modellerinde derinlik  | • Aşırı yağmur ve siste piksel gradyan kontrastının   |
|   ve hız tensörü olarak NPU'ya beslenme imkanı        |   düşerek optik akışın kaybolması                     |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Stereo Derinlik ve Optik Akış Hattı

```
   [ Sol Kamera Karesi ]                 [ Sağ Kamera Karesi ]
             \                                 /
              \                               /
               v                             v
       +---------------------------------------------+
       |   Epipolar Blok Eşleme (Disparity Engine)   |
       |   d = u_L - u_R  ====>  Z = (f * B) / d     |
       +----------------------+----------------------+
                              |
                              v
       +---------------------------------------------+
       |   Lucas-Kanade 2D Optik Akış & TTC Motoru   |
       |   - Hız Vektörü: [u_t, v_t]                 |
       |   - Çarpışma Süresi: TTC = Z / v_rel        |
       +----------------------+----------------------+
                              |
                              v
               [ FSD Acil Fren (AEB) Tetikleyici ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana derinlik ve optik akış simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
