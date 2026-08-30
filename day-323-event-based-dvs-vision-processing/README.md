# 🧠 Day 323: Dynamic Vision Sensors (DVS) & Olay Tabanlı Görsel İşleme

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![Phase 17](https://img.shields.io/badge/Phase-17%3A%20Neuromorphic%20AI%20%26%20BCI-blueviolet?style=flat-square)
![Status: Completed](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

> 🎓 **Stajyer Notu & Mentor Rehberi:** Merhaba! Bugün bilgisayarlı görüde (Computer Vision) devrim yaratan yeni nesil kameralara adım atıyoruz: **Dynamic Vision Sensors (DVS)** yani Nöromorfik Kameralar. Standart kameralar (saniyede 30/60 kare çeken) anlamsız arka planları sürekli tekrar çekerken, DVS kameralar tıpkı insan gözü ağtabakası (retina) gibi sadece **değişen piksellerde mikrosaniye düzeyinde olay (event)** üretir. Bu rehberde DVS verisini işleme, zamansal SAE yüzeyi oluşturma ve Spiking ConvNet modelleri eğitmeyi adım adım öğreneceksin!

---

## 🎯 1. Günün Konusu & Teorik/Matematiksel Derinlik

### 1.1 DVS Kamera Mekanizması ve Matematiksel Modeli

Standart kameralar $I(x, y, t)$ parlaklık matrisini sabit frekansta (ör. $33 \text{ ms}$ aralıkla) pozlarken; DVS kamerasındaki her bir piksel bağımsız olarak logaritmik parlaklık değişimini izler:

$$\Delta L(x, y, t) = \ln I(x, y, t) - \ln I(x, y, t_{last})$$

Eğer bu değişim eşik değeri $C$'yi (contrast threshold) aşarsa, piksel anında asenkron bir **Olay (Event)** yayınlar:

$$e_k = (x_k, y_k, t_k, p_k)$$

- $x_k, y_k$: Olayın gerçekleştiği piksel koordinatları.
- $t_k$: Mikrosaniye ($\mu s$) çözünürlüklü zaman damgası.
- $p_k \in \{-1, +1\}$: Polarite (Parlaklık artışı $+1$ ON, azalışı $-1$ OFF).

```text
Standart Kamera: [ Frame 1 ] ──(33 ms)──> [ Frame 2 ] ──(33 ms)──> [ Frame 3 ] (Gereksiz Veri Israfı)
DVS Kamera:      e1 (x,y,t1,p) ──> e2 (x,y,t2,p) ──> e3 (x,y,t3,p) ... (Sadece Hareket Varsa Veri Aksın!)
```

---

### 1.2 Olay Dönüştürme ve Temsil Yöntemleri

1. **Surface of Active Events (SAE / Zamansal Sönümlenme Yüzeyi):**
   Her piksel için son olay zaman damgası $T_{last}(x, y, p)$ tutulur. Zaman ilerledikçe üstel olarak sönümlenir:
   
   $$S(x, y, p) = \exp\left( -\frac{t_{current} - T_{last}(x, y, p)}{\tau} \right)$$

2. **3D Voxel Grid Temsili:**
   Mikrosaniye seviyesindeki olay akışı $B$ adet zamansal kutuya (temporal bins) ayrıştırılarak $(2 \times B, H, W)$ boyutlu bir 3D tensör haline getirilir.

---

### 1.3 4 Zorunlu Mimari Analiz

#### 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- **Mikrosaniye Zamansal Çözünürlük ($\mu s$ Latency):** Hareket bulanıklığı (motion blur) olmadan yüksek hızlı nesneleri (ör. roketler, otonom drone'lar, sanayi robotları) takip etmeyi sağlar.
- **Yüksek Dinamik Menzil (High Dynamic Range - 120+ dB):** Aşırı karanlık veya aşırı parlak ışık koşullarında (güneş parlaması, tünel girişleri) körleşmeden çalışır.

#### 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- **Bant Genişliği ve Veri İsrafı (Data Redundancy Bottleneck):** Sabit arka planı tekrar tekrar çekmek yerine veri hacmini 10x-50x azaltır.
- **Motion Blur (Hareket Bulanıklığı):** Hızlı nesnelerde görüntü izi kalmasını engeller.

#### ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler me Dikkat Edilmesi Gerekenler)
- **Durgun Sahne Körlüğü (Static Scene Invisibility):** Kamera ve nesneler tamamen durduğunda hiçbir olay üretilmez (görüntü tamamen kaybolur).
- **Gürültü Duyarlılığı (Thermal Noise Events):** Sıcaklık artışlarında karanlıkta rastgele aşırı olay patlamaları oluşabilir.

#### 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- **Frame-based RGB Cameras:** Yüksek çözünürlüklü renkli görüntü (Yavaş, yüksek veri boyutu).
- **RGB-D / Time-of-Flight Cameras:** Derinlik haritası (Yüksek güç tüketimi).
- **DVS Neuromorphic Cameras (Bizim Yaklaşımımız):** Asenkron olay akışı (Ultra-hızlı, ultra-düşük güç).

---

### 1.4 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Terim | Tanım ve Stajyer Açıklaması |
| --- | --- |
| **DVS** | Dynamic Vision Sensor: Sadece parlaklık değişiminde olay üreten nöromorfik kamera. |
| **Event Tuple ($e_k$)** | $(x, y, t, p)$ boyutunda 4 bileşenli asenkron olay verisi. |
| **Polarity ($p$)** | Parlaklığın arttığını ($+1$) veya azaldığını ($-1$) belirten 1-bitlik yön bilgisi. |
| **SAE** | Surface of Active Events: Zamansal olarak üstel sönümlenen 2D olay yüzeyi. |
| **Voxel Grid** | Olay akışını Evrişimsel Ağların (CNN) işleyebileceği $C \times H \times W$ tensörüne dönüştürme. |
| **Contrast Threshold ($C$)** | Olay tetiklenmesi için gereken asgari logaritmik parlaklık değişimi. |
| **Motion Blur** | Klasik kameralarda hızlı harekette oluşan bulanıklık (DVS'de sıfırdır). |
| **HDR (Dynamic Range)** | DVS kameraların 120 dB üzerindeki geniş ışık toleransı. |
| **Events/sec (Throughput)** | DVS kamerasının saniyede ürettiği ve işlenen toplam olay hızı. |
| **Spiking ConvNet** | DVS Voxel Grid tensörlerini işleyen olaya duyarlı Spiking CNN. |

---

### 1.5 SWOT Analizi Karar Matrisi

```
               GÜÇLÜ YÖNLER (STRENGTHS)                      ZAYIF YÖNLER (WEAKNESSES)
      ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
      │ • Mikrosaniye seviyesinde ultra-düşük    │  │ • Durgun nesnelerde görünmezlik.         │
      │   gecikme (Sub-millisecond latency).    │  │ • Standart CNN kütüphaneleriyle doğrudan │
      │ • 120+ dB yüksek dinamik aralık (HDR).   │  │   uyumsuzluk (Voxel dönüşümü gerekir).  │
      └──────────────────────────────────────────┘  └──────────────────────────────────────────┘
               FIRSATLAR (OPPORTUNITIES)                       TEHDİTLER (THREATS)
      ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
      │ • Otonom araçlarda tünel ve gece sürüşü. │  │ • Sensör maliyetlerinin henüz yüksek     │
      │ • Yüksek hızlı İHA/Drone navigasyonu.    │  │   olması.                                │
      └──────────────────────────────────────────┘  └──────────────────────────────────────────┘
```

---

## 💻 2. Üretim Seviyesinde Uygulama Kodu & Mimarisi

Dizin yapısı:

```text
day-323-event-based-dvs-vision-processing/
├── ana_akis.py
├── gereksinimler.txt
├── LICENSE
├── README.md
├── ciktilar/
│   └── dvs_isleme_paneli.png
├── src/
│   ├── __init__.py
│   ├── dvs_motoru.py
│   ├── dvs_gorsellestirici.py
│   └── dvs_profilleyici.py
└── testler/
    └── test_dvs_motoru.py
```

---

## 🧪 3. Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

### 🎯 Görev (Stajyer Görevi)
SAE (Surface of Active Events) yüzeyinde sönümlenme zaman sabiti ($\tau_{us}$) $10,000 \mu s$'den $1,000 \mu s$'ye düşürüldüğünde, hızlı hareket eden nesnelerin izin kaybolup kaybolmadığını doğrulayan bir kod yazınız.

### 💡 Çözüm Kodu
```python
import numpy as np
from src.dvs_motoru import SurfaceOfActiveEvents

def test_sae_tau_decay():
    sae_long = SurfaceOfActiveEvents(height=16, width=16, tau_us=10000.0)
    sae_short = SurfaceOfActiveEvents(height=16, width=16, tau_us=1000.0)

    events = np.array([[8, 8, 1000.0, 1.0]], dtype=np.float32)

    # 5,000 us sonra sönümlenme (t_current = 6000 us)
    surf_long = sae_long.guncelle_ve_hesapla(events, t_current=6000.0)
    surf_short = sae_short.guncelle_ve_hesapla(events, t_current=6000.0)

    print(f"Uzun Tau (10ms) İzsiz Değer:  {surf_long[1, 8, 8]:.4f} (İz net şekilde belirgin)")
    print(f"Kısa Tau (1ms) İzsiz Değer:   {surf_short[1, 8, 8]:.4f} (İz tamamen söndü ve kayboldu)")

if __name__ == "__main__":
    test_sae_tau_decay()
```

---

## 📊 4. Veri Hacmi & Performans Benchmark Tablosu

| Metrik | DVS Nöromorfik Kamera | Standart RGB Kamera (60 FPS) | Faksiyonel Kazanç |
| --- | --- | --- | --- |
| **Zamansal Çözünürlük** | ~1-10 $\mu s$ | 16.6 $ms$ (60 FPS) | **1000x Daha Hızlı** |
| **Görüntü Veri Boyutu** | ~9.6 KB / 50ms | ~153.6 KB / 50ms | **16.0x Veri Sıkıştırması** |
| **Dinamik Menzil (HDR)** | 120+ dB | 60-70 dB | **Işık Toleransı İki Kat** |
| **Motion Blur** | Yok (Zero Blur) | Var | **Net Hareket Algılama** |

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
DVS nöromorfik kameralar durgun sahnelerde (stik görüntüde) neden hiçbir veri üretmez ve bu durum otonom araçlarda nasıl aşılır?

### 💬 Mentorluk Yanıtı
DVS kameralar sadece logaritmik parlaklık değişimi $\Delta L = \ln I(t) - \ln I(t_{last})$ eşik değeri $C$'yi aştığında olay yayınlar. Durgun nesnelerde $I(t)$ sabit kaldığı için $\Delta L = 0$ olur ve piksel hiçbir olay üretmez. 

Otonom araçlarda bu durum iki şekilde aşılır:
1. **Mikro-Sakkad Hareketi (Saccadic Camera Motion):** Kamera çok küçük genlikte yüksek frekanslı titreşim hareketi yapar; böylece durağan arka plan bile piksellerde olay tetikler (tıpkı gözlerimizin sürekli farkında olmadan yaptığı mikrosakkadlar gibi).
2. **Sensor Fusion (Hibrit Sensör Füzyonu):** DVS kamera yüksek hızlı olaylar için kullanılırken, geleneksel düşük FPS'li bir RGB kamera veya LiDAR ile durağan sahne derinliği ve renk bilgisi birleştirilir.
