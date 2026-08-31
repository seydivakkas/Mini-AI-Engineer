# 🚗 Tesla FSD Otonom Sürüş | Gün 56: Yol Planlama Temelleri: Hibrit A* (Hybrid A*) ve Voronoi Alanı ile Park Planlama

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Planning](https://img.shields.io/badge/Planner-Hybrid%20A%2A%203D%20Kinematics-red.svg?style=flat-square)](https://www.tesla.com/)
[![Autopark](https://img.shields.io/badge/Autopark-Parallel%20%26%20Perpendicular-blue.svg?style=flat-square)](https://www.sae.org/)
[![Voronoi](https://img.shields.io/badge/Safety-Voronoi%20Obstacle%20Field-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"56. günümüze ve FAZ 6'YA (Yörünge Planlama, Model Predictive Control & ISO 26262 ASIL-D) HOŞ GELDİN stajyer!  
> Önceki fazlarda aracın görmesini (Vision BEV) ve düşünmesini (Deep Learning AI) mükemmelleştirdik. Şimdi ise FSD'nin eyleme geçme, direksiyon ve pedalları yönetme (Planning & Control) evresindeyiz!  
> Standart A* veya Dijkstra algoritmaları 2D ızgarada çapraz yürüyen noktalar için tasarlanmıştır. Ancak bir Tesla otomobil yana doğru yürüyemez (Non-Holonomic Kısıt) ve bir dönüş yarıçapına ($R_{\min} = \frac{L}{\tan\delta_{\max}}$) sahiptir.  
> Tesla otonom park (Autopark) ve dar alan manevralarını **Hibrit A* (Hybrid A*)** ile çözer:  
> 1. **Sürekli Durum Uzayı ($x, y, \theta$):** Arama ızgarasının her hücresinde aracın fiziksel yönelme açısı ($\theta$) ve alt-piksel koordinatları saklanır.  
> 2. **Kinematik Bisiklet Modeli:** Tüm adım genişlemeleri direksiyon açısı kısıtlarına ($\delta \le 31.5^\circ$) uygun olarak türetilir.  
> 3. **Reeds-Shepp Eğrileri:** Hedef park cebine yaklaşıldığında ileri-geri analitik S-eğrileri ile arama derinliği anında sonlandırılır.  
> 4. **Voronoi Güvenlik Potansiyel Alanı:** Yan araçların tamponlarına ve kaldırımlara sürtmeyi önleyen itici güvenlik maliyeti uygulanır.  
> Bugün Tesla Autopark'ın matematiksel direksiyon beynini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Kinematik Bisiklet Modeli Adım Geçişi (Kinematic Bicycle Step)

$$x_{t+1} = x_t + v \cdot \cos(\psi_t) \cdot \Delta t$$

$$y_{t+1} = y_t + v \cdot \sin(\psi_t) \cdot \Delta t$$

$$\psi_{t+1} = \psi_t + \frac{v}{L} \cdot \tan(\delta_t) \cdot \Delta t$$

### 2. Hibrit A* Toplam Maliyet Fonksiyonu

$$f(n) = g(n) + h(n) + w_{\text{voronoi}} \cdot C_{\text{voronoi}}(x, y)$$

$$C_{\text{voronoi}}(x, y) = \frac{1}{\left( \min_i \|\mathbf{x} - \mathbf{x}_{\text{obs}, i}\| \right)^2}$$

### 3. Minimum Dönüş Yarıçapı ve Reeds-Shepp Kısıtı

$$R_{\min} = \frac{L}{\tan(\delta_{\max})}, \quad L = 2.875\text{ m (Model 3 Aks Mesafesi)}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Fiziksel aracın direksiyon ve aks geometrisine uygun, dar sokaklarda ve otoparklarda çarpışmasız ileri-geri manevra yörüngeleri üretmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Holonomik Olmayan Kilitlenmeler:** Standart grid aramasının ürettiği fiziksel olarak dönülemez keskin köşeleri ortadan kaldırdı.
- **Kaldırım ve Jant Çizilmeleri:** Voronoi güvenlik alanı sayesinde araç tekerlekleri kaldırım kenarından güvenli mesafede tutuldu.
- **Tek Hamlede Park Yörüngesi:** Reeds-Shepp analitik yayları ile park cebine milimetrik giriş sağlandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Yüksek Hız Dinamikleri:** 60 km/h üzeri hızlarda lastik kayma açıları (Slip Angle) kinematik modeli yetersiz bırakır (Dinamik Bisiklet Modeli gerekir).
- **Arama Süresi:** Çok büyük labirent otoparklarda heuristik tasarımının optimize edilmesi gerekir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **RRT* (Rapidly-exploring Random Trees):** Rastlantısaldır; her çalıştırmada farklı ve titreşimli yörünge üretebilir.
- **Saf Potansiyel Alanlar:** Yerel minimumlarda (Local Minima) kilitlenip aracın durmasına neden olabilir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Hybrid A*** | Sürekli durum uzayını ($x, y, \theta$) kinematik araç hareket primitifleriyle arayan otonom planlayıcı. |
| **Kinematic Bicycle Model**| Ön ve arka tekerlekleri tek bir çizgi üzerinde modelleyen temel araç hareket dinamiği. |
| **Non-Holonomic Constraint**| Bir aracın yanlamasına doğrudan ötelenemeyip sadece tekerlek yönünde ilerleyebilme kısıtı. |
| **Reeds-Shepp Curves** | İleri ve geri hareket edebilen araçlar için iki nokta arasındaki en kısa analitik yörünge eğrileri. |
| **Voronoi Diagram / Field**| Engellerden eşit uzaklıktaki güvenli merkez omurgayı çıkarıp aracı orada tutan potansiyel alan. |
| **Wheelbase ($L$)** | Aracın ön ve arka aks merkezleri arasındaki boyuna mesafe (Tesla Model 3: 2.875 m). |
| **Cross-Track Error ($e_{\text{lat}}$)**| Aracın takip etmesi gereken yörüngeden olan yanal sapma mesafesi. |
| **Heading Error ($e_{\psi}$)** | Aracın burnunun gitmesi gereken yörünge teğetine olan açısal farkı. |
| **Steering Angle ($\delta$)** | Ön tekerleklerin yönelme açısı (Maksimum Tesla sınırı: $\approx 31.5^\circ$). |
| **Autopark Trajectory** | Park cebine girmek için gereken S-eğrisi direksiyon ve vites (D/R) sekansı. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 kinematik uygulanabilir pürüzsüz yörünge       | • Yüksek hızlarda lastik kayma dinamiğini içermez     |
| • Voronoi güvenlik alanı ile sıfır jant sürtmesi      | • 3D arama uzayında bellek yönetimi optimizasyonu şart|
| • 15 µs ultra hızlı RTOS planlama döngüsü             |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Gelecekteki Robotaksi filolarında milimetrik        | • Park manevrası sırasında aniden aracın arkasına     |
|   otonom şarj istasyonuna yanaşma kontrolü            |   geçen dikkatsiz yayalar                             |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Hibrit A* Autopark Akış Şeması

```
[ Başlangıç Pozu (x, y, psi) & Hedef Park Cebi ]
                        |
                        v
    [ Sürekli Durum Uzayı (x, y, theta) Arama Izgarası ]
                        |
       +----------------+----------------+
       |                                 |
       v                                 v
[ Kinematik Bisiklet Adımları ]   [ Voronoi Potansiyel Alanı ]
- v = {-1.2, +0.5} m/s            - Güvenlik Tamponu
- delta = {-31.5°, 0°, +31.5°}    - Çarpışma Önleme
       \                                 /
        \                               /
         v                             v
[ Reeds-Shepp S-Eğrisi ile Kusursuz Park Yörüngesi ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Hibrit A* Autopark simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
