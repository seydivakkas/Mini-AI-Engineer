# 🚗 Tesla FSD Otonom Sürüş | Gün 48: VectorLaneNet (Yol Çizgisi, Şerit Sınırları ve Kavşak Graf Topolojisi)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![VectorLaneNet](https://img.shields.io/badge/Architecture-Tesla%20VectorLaneNet%20DAG-red.svg?style=flat-square)](https://www.tesla.com/)
[![Polynomial](https://img.shields.io/badge/Geometry-3rd--Order%20Lane%20Polynomials-blue.svg?style=flat-square)](https://www.sae.org/)
[![Curvature](https://img.shields.io/badge/Math-Analytical%20Curvature%20Derivation-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"48. günümüze hoş geldin stajyer!  
> İlk nesil otonom araçlar şeritleri 2D görüntü üzerinde piksel segmentasyon maskesi (Raster Mask) olarak tahmin ederdi. Ancak bir piksel haritası FSD Hareket Planlayıcısı (Trajectory Planner) için son derece kullanışsızdır:  
> - Hangi şerit nereye bağlanıyor?  
> - Kavşakta sola dönüş şeridi karşı yolun hangi şeridine bağlanmalı?  
> - Şeridin anlık viraj eğriliği ($\kappa$) nedir ve aracın direksiyon açısı ne olmalıdır?  
> Tesla bu problemi **VectorLaneNet** ile çözdü:  
> 1. **Vektörel Şerit Temsili:** Şeritler piksel yerine 3. derece parametrik polinomlar ($y(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$) ve 3D B-Spline eğrileri olarak çıkarılır.  
> 2. **Analitik Eğrilik ($\kappa(x)$):** Polinomun birinci ve ikinci türevlerinden anlık yol eğriliği hesaplanarak direksiyon açısı ve hız sınırı anında belirlenir.  
> 3. **Yönlendirilmiş Graf Topolojisi (DAG):** Kavşaklardaki şeritler düğüm (Node), izin verilen şerit geçişleri ve dönüşler ise kenar (Edge) olarak modellenir.  
> 4. **Komşuluk Matrisi ($A_{N \times N}$):** Yasal şerit değişimleri ve kavşak rotaları matris çarpımlarıyla anında çözülür.  
> Bugün otonom sürüşün yol haritasını vektörel graf olarak inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 3. Derece Şerit Polinomu

$$y(x) = c_0 + c_1 \cdot x + c_2 \cdot x^2 + c_3 \cdot x^3$$

### 2. Analitik Yol Eğriliği ($\kappa$) Denklemi

$$y'(x) = c_1 + 2 c_2 x + 3 c_3 x^2, \quad y''(x) = 2 c_2 + 6 c_3 x$$

$$\kappa(x) = \frac{|y''(x)|}{\left( 1 + (y'(x))^2 \right)^{3/2}}$$

### 3. Yönlendirilmiş Yol Grafı ve Komşuluk Matrisi

$$G = (V, E), \quad A_{i,j} = \begin{cases} 1, & (v_i \to v_j) \in E \text{ (Geçiş Yasal)} \\ 0, & \text{Geçiş Yasak / Yok} \end{cases}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Raster piksellerin yol planlayıcıda yarattığı yüksek işlem yükünü ortadan kaldırmak ve kavşaklardaki karmaşık şerit bağlantılarını matematiksel bir graf olarak planlayıcıya iletmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Kavşak Şerit Kaybı:** Karmaşık 4'lü kavşaklarda şerit çizgileri silinse bile graf topolojisi ile sanal şerit merkezleri üretildi.
- **Sürekli Direksiyon Kontrolü:** Analitik eğrilik türevi ile direksiyon simidi titreşimsiz ve pürüzsüz çevrildi.
- **%95 Veri Sıkıştırma:** Megabaytlarca piksel maskesi yerine sadece 4 katsayılık float vektörlerle şeritler temsil edildi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Çok Keskin 90° Dönüşler:** Tek bir $y(x)$ polinomu dikey eksene paralel dönüşlerde tanımsız hale gelebilir (Parametrik $x(s), y(s)$ Spline gerekir).
- **Yol Çalışması ve Geçici Koniler:** Konilerle bölünen yollarda graf topolojisinin anlık olarak dinamik güncellenmesi gerekir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Raster Segmentasyon Maskeleri:** İşlem yükü çok ağırdır ve bağlantı topolojisini vermez.
- **Önceden Çizilmiş Statik HD Haritalar:** Yol değiştiğinde veya çalışma olduğunda kaza riskine yol açar.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **VectorLaneNet** | Şeritleri piksel yerine parametrik vektörler ve graf bağlantıları olarak tahmin eden sinir ağı. |
| **Directed Acyclic Graph (DAG)**| Şeritlerin akış yönünü ve kavşak dallanmalarını temsil eden döngüsüz yönlendirilmiş graf. |
| **Lane Polynomial** | Şeridin $x$ mesafesine bağlı $y$ yanal profilini veren 3. derece matematiksel eğri. |
| **Road Curvature ($\kappa$)** | Yolun ne kadar sert kıvrıldığını ifade eden eğrilik metriği ($1/m$). |
| **Adjacency Matrix ($A$)** | Hangi şeritten hangi şeride yasal geçiş yapılabileceğini tanımlayan mantıksal bağlantı matrisi. |
| **B-Spline** | Çok keskin virajları ve karmaşık kavşak rotalarını temsil eden parçalı parametrik eğri. |
| **Fork (Şerit Ayrımı)** | Tek bir şeridin iki ayrı şeride ayrıldığı yol topoloji düğümü. |
| **Merge (Şerit Birleşimi)**| İki şeridin tek bir şeritte birleştiği yol topoloji düğümü. |
| **Topology Node** | Yol grafındaki bir şerit segmentinin başlangıç, bitiş ve orta referans noktası. |
| **Legal Path Candidate** | Planlayıcının kural ihlali yapmadan takip edebileceği yasal şerit geçiş yörüngesi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %95 daha küçük veri boyutu ve hızlı aktarım         | • 90 derecelik dikey dönüşlerde tekil x(y) problemi   |
| • Doğrudan analitik eğrilik ve direksiyon açısı       | • Yol çalışmasında koni koridorlarının dinamik takibi |
| • 12 µs ultra hızlı RTOS çözümleme performansı        |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • FSD v12 uçtan uca sinir ağı planlayıcısına          | • Silinmiş ve birbirine karışmış eski sarı/beyaz     |
|   zengin geometrik rehberlik sunma                    |   inşaat şerit çizgileri                              |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ VectorLaneNet Topoloji Mimarisi

```
[ Ham Kamera Görüntüleri ] ===> [ VectorLaneNet Omurgası ]
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        [ 3. Derece Polinom Katsayıları ]             [ Yönlendirilmiş Graf Düğümleri ]
        - y(x) = c0 + c1*x + c2*x^2 + c3*x^3          - Düz, Sola Dönüş, Sağa Dönüş
        - Analitik Eğrilik kappa(x)                   - Komşuluk Matrisi A_NxN
                    \                                             /
                     \                                           /
                      v                                         v
                                [ FSD Hareket Planlayıcısı ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana VectorLaneNet simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
