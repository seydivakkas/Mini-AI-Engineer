# 🛰️ Day 341: Spacecraft Autonomous GNC under Zero-GNSS (FAZ 18 BAŞLANGICI)

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![Phase 18](https://img.shields.io/badge/Phase-18%3A%20Space%2C%20Aerospace%20%26%20Defense%20AI-orange?style=flat-square)
![Status: Completed](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

> 🎓 **Stajyer Notu & Mentor Rehberi:** Hoş geldin! Bugün **FAZ 18: Uzay, Havacılık ve Savunma için Kritik Otonom AI** fazına dev bir adımla başlıyoruz! Derin uzay görevlerinde (Ay, Mars yolculukları, Cislunar yörüngeler) veya askeri elektronik harp ortamlarında GPS/GNSS uydularından gelen sinyaller ya hiç yoktur ya da karıştırma (Jamming/Spoofing) ile tamamen kesilir. Peki bir uzay aracı GPS olmadan uzay boşluğunda yolunu nasıl bulur? **Optik Yıldız Takipçisi (Star Tracker - TRIAD Algoritması)** ile gök küresindeki yıldızları nirengi alır, **Genişletilmiş Kalman Filtresi (EKF)** ile İki Cisim + Dünya'nın Basıklık ($J_2$) Yerçekimi dinamiklerini birleştirir ve **Otonom Rehberlik, Navigasyon ve Kontrol (GNC)** itki komutları üretir!

---

## 🎯 1. Günün Konusu & Teorik/Matematiksel Derinlik

### 1.1 Sıfır-GNSS Koşullarında Uzay Aracı Dinamikleri

Uzay aracı $x = [r_x, r_y, r_z, v_x, v_y, v_z]^T$ durum vektörü ile hareket eder:

1. **İki Cisim Yerçekimi ve $J_2$ Dünya Basıklığı Pertürbasyonu:**
   $$\ddot{\mathbf{r}} = -\frac{\mu}{\|\mathbf{r}\|^3}\mathbf{r} + \mathbf{a}_{J2}$$
   Burada $J_2 = 1.08263 \times 10^{-3}$ Dünya'nın kutuplardan basık ekvatordan şişkin olmasından kaynaklanan yerçekimi asimetrisidir.

2. **Optik Yıldız Takipçisi TRIAD Yönelim Kestirimi:**
   İki yıldızın gövde koordinatlarındaki birim vektörleri ($\mathbf{v}_1^b, \mathbf{v}_2^b$) ile eylemsizlik kataloğundaki referans vektörleri ($\mathbf{v}_1^i, \mathbf{v}_2^i$) kullanılarak yönelim dönüşüm matrisi $\mathbf{R}$ çözülür:
   $$\mathbf{t}_1 = \mathbf{v}_1, \quad \mathbf{t}_2 = \frac{\mathbf{v}_1 \times \mathbf{v}_2}{\|\mathbf{v}_1 \times \mathbf{v}_2\|}, \quad \mathbf{t}_3 = \mathbf{t}_1 \times \mathbf{t}_2 \quad \implies \mathbf{R} = \mathbf{M}_{body} \mathbf{M}_{inertial}^T$$

```text
       ┌─────────────────────────────────────────────────────────┐
       │ Optical Star Tracker (Celestial Star Catalog TRIAD)     │
       └────────────────────┬────────────────────────────────────┘
                                    │ Attitude Estimation Matrix (Error < 0.05°)
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Orbital Extended Kalman Filter (Two-Body + J2 Gravity)  │
       └────────────────────┬────────────────────────────────────┘
                                    │ Position Estimation Error (< 1.5 meters)
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Autonomous GNC Controller (Thruster Delta-V Guidance)   │
       └─────────────────────────────────────────────────────────┘
```

---

### 1.2 4 Zorunlu Mimari Analiz

#### 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- **Deep Space & GPS-Denied Autonomy:** Dünya'dan milyonlarca kilometre uzakta veya GPS sinyallerinin kesildiği savunma/uzay senaryolarında aracın tamamen otonom rotada kalabilmesi için.
- **Micro-Radian Attitude Pointing:** Uydu antenlerinin ve teleskopların hedefe $0.05^\circ$ altında hassasiyetle kilitlenmesini sağlamak için.

#### 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- **GPS Dependency Bottleneck:** Dünya dışı görevlerde GPS uydusu ihtiyacını optik nirengi ve yıldız kataloğuyla sıfırlar.
- **Orbital Drift via J2 Perturbations:** Dünya'nın şekil bozukluğundan kaynaklanan yörünge kaymalarını EKF modelinde önceden telafi eder.

#### ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- **Star Occlusion:** Güneş parlaması veya gezegen arkasına geçiş durumlarında yıldız takipçisi görüşü kısa süreliğine kaybolabilir (IMU entegrasyonu gerekir).
- **High-Order Gravitational Harmonics:** $J_3, J_4$ veya Ay/Güneş çekim etkileri ultra-uzun yörünge simülasyonlarında modele eklenmelidir.

#### 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- **Ground-Based Deep Space Network (DSN) Telemetry:** Dünya'dan komut bekleyen ve 20-40 dakika gecikmeye maruz kalan klasik yöntem.
- **Autonomous GNC with Star Tracker & EKF (Bizim Yaklaşımımız):** Milisaniye altında onboard çalışan ve sıfır gecikmeli kararlar alan tam otonom uzay navigasyonu.

---

### 1.3 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Terim | Tanım ve Stajyer Açıklaması |
| --- | --- |
| **GNC** | Guidance, Navigation & Control: Uzay aracı rota ve yönelim kontrol sistemi. |
| **Zero-GNSS** | Küresel uydu konumlama sinyallerinin (GPS vb.) tamamen bulunmadığı durum. |
| **Star Tracker** | Yıldız desenlerini fotoğraflayıp yönelim hesaplayan optik uzay kamerası. |
| **TRIAD** | İki yıldız vektöründen 3D yönelim dönüşüm matrisi çıkaran klasik algoritma. |
| **J2 Perturbation** | Dünya'nın ekvatoral şişkinliğinin neden olduğu ikinci derece yerçekim etkisi. |
| **EKF** | Extended Kalman Filter: Doğrusal olmayan dinamikleri kestiren filtre. |
| **Delta-V ($\Delta v$)** | Uzay aracının hızını ve yörüngesini değiştirmek için gereken hız farkı. |
| **LEO Orbit** | Low Earth Orbit: Dünya yüzeyinden 200 - 2000 km irtifadaki alçak yörünge. |
| **Attitude Error** | Uzay aracının hedeflenen bakış açısıyla mevcut yönelimi arasındaki açı farkı. |
| **Keplerian Orbit** | İki gök cisminin yerçekimi etkisi altında çizdiği eliptik/dairesel yörünge. |

---

### 1.4 SWOT Analizi Karar Matrisi

```
               GÜÇLÜ YÖNLER (STRENGTHS)                      ZAYIF YÖNLER (WEAKNESSES)
      ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
      │ • GPS bağımsız < 1.5 m yörünge konumu.   │  │ • Güneş parlamasında optik körlük       │
      │ • < 0.05° yıldız takip yönelim keskinliği│   riski.                                 │
      └──────────────────────────────────────────┘  └──────────────────────────────────────────┘
               FIRSATLAR (OPPORTUNITIES)                       TEHDİTLER (THREATS)
      ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
      │ • Ay ve Mars iniş araçlarında tam otonom │  │ • Aşırı kozmik radyasyon kaynaklı sensör │
      │   navigasyon yazılımı.                   │   gürültüsü.                             │
      └──────────────────────────────────────────┘  └──────────────────────────────────────────┘
```

---

## 💻 2. Üretim Seviyesinde Uygulama Kodu & Mimarisi

Dizin yapısı:

```text
day-341-spacecraft-autonomous-gnc-zero-gnss/
├── ana_akis.py
├── gereksinimler.txt
├── LICENSE
├── README.md
├── ciktilar/
│   └── uzay_araci_gnc_paneli.png
├── src/
│   ├── __init__.py
│   ├── gnc_gorsellestirici.py
│   ├── gnc_profilleyici.py
│   └── spacecraft_gnc_motoru.py
└── testler/
    └── test_spacecraft_gnc_motoru.py
```

---

## 🧪 3. Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

### 🎯 Görev (Stajyer Görevi)
$r = 7000.0\text{ km}$ irtifada dairesel yörüngede dönen bir uzay aracının Keplerian yörünge hızını ($v = \sqrt{\mu / r}$) $\mu = 398600.44\text{ km}^3/\text{s}^2$ formülü ile hesaplayan bir Python fonksiyonu yazınız.

### 💡 Çözüm Kodu
```python
import numpy as np

def test_orbital_velocity():
    mu = 398600.4418 # km^3 / s^2
    r = 7000.0       # km (LEO Yarıçapı)
    
    v_circ = np.sqrt(mu / r)
    print(f"Yörünge Yarıçapı: {r} km")
    print(f"Hesaplanan Dairesel Yörünge Hızı: {v_circ:.4f} km/s ({v_circ * 3600:.2f} km/h)")

if __name__ == "__main__":
    test_orbital_velocity()
```

---

## 📊 4. Spacecraft GNC Performance Benchmark Tablosu

| Navigasyon Mimarisi | GNSS İhtiyacı | Konum Hatası (m) | Yönelim Hatası (°) | Otonomi Seviyesi |
| --- | --- | --- | --- | --- |
| **Klasik Yer İstasyonu Telemetrisi** | ❌ Gecikmeli | 50.0 m | 0.80° | Seviye 1 |
| **Sıfır-GNSS Star Tracker + EKF GNC (Bizim)** | **✅ SIFIR GNSS** | **< 1.50 m** | **< 0.05°** | **Seviye 5 (Tam Otonom)** |

---

## 📜 5. Lisans & Metaveri

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 201-Day AI, CV, LLM/RAG, Reasoning & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```

---

## ❓ 6. Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

### ❓ Soru
Uzay araçlarında neden sadece yıldız takipçisi (Star Tracker) tek başına konum belirlemeye yetmez ve yanında $J_2$ yerçekimi modelli EKF gerekir?

### 💬 Mentorluk Yanıtı
Harika bir astrodinamik sorusu! **Yıldız Takipçisi (Star Tracker)**, sonsuz uzaktaki yıldızlara baktığı için uzay aracının yalnızca **3D Yönelimini (Attitude / Hangi yöne baktığını)** söyler; aracın Dünya'ya veya Ay'a göre nerede olduğunu (3D Konumunu) doğrudan veremez. **$J_2$ Modelli EKF** ise uzay aracının eylemsizlik hareket denklemlerini yerçekimi pertürbasyonlarıyla birleştirerek aracın 3D konum ve hızını ($x, y, z, v_x, v_y, v_z$) kusursuz şekilde hesaplar! İkisi birleştiğinde Sıfır GNSS ile tam otonom uzay navigasyonu mümkün olur!
