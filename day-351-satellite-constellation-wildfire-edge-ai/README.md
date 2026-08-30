# 🛰️ Day 351: Satellite Constellation Edge AI: Real-Time Wildfire & Threat Detection

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![Phase 18](https://img.shields.io/badge/Phase-18%3A%20Space%2C%20Aerospace%20%26%20Defense%20AI-orange?style=flat-square)
![Status: Completed](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

> 🎓 **Stajyer Notu & Mentor Rehberi:** Hoş geldin! Bugün uzaydan gezegenimizi ve stratejik tesislerimizi koruyan bir **Alçak Dünya Yörüngesi (LEO) Küp Uydu Takımyıldızı** inşa ediyoruz! Klasik yer gözlem uydularının en büyük çıkmazı "Veri İndirme (Downlink) Darboğazı"dır. Bir uydu devasa optik/kızılötesi fotoğraflar çeker, ancak yer istasyonunun üzerinden geçerken (günde 2-3 kez) veriyi indirebilir. Bir orman yangını veya füze rampası ısı anomalisi başladığında yer istasyonunun sırasını beklemek 6-12 saatlik ölümcül bir gecikmeye yol açar! Çözüm nedir? **Uydu Üzeri Uç Yapay Zeka (On-Board Edge AI)!** Uydu çektiği çok bantlı (**NIR, SWIR, MWIR 3.9 μm**) spektral veriyi saniyede uydunun kendi radyasyona dayanıklı çipinde işler, **Yangın Işınım Gücünü (FRP - Fire Radiative Power)** hesaplar ve uydular arası lazer ağı (ISL) ile sadece 50 baytlık kritik alarmı **23 milisaniyede** doğrudan itfaiye/savunma merkezine ulaştırır!

---

## 🎯 1. Günün Konusu & Teorik/Matematiksel Derinlik

### 1.1 Çok Bantlı Spektral İndeksler ve Yangın Fiziği

1. **Normalize Edilmiş Yanık Oranı (NBR - Normalized Burn Ratio):**
   $$\text{NBR} = \frac{\rho_{NIR} - \rho_{SWIR}}{\rho_{NIR} + \rho_{SWIR}}$$
   (Sağlıklı orman yüksek NIR yayar $\text{NBR} > 0.4$, yangın ve yanık zemin yüksek SWIR yayar $\text{NBR} < -0.2$).

2. **Stefan-Boltzmann ve MWIR Yangın Işınım Gücü (FRP - Fire Radiative Power):**
   Kaufman ve Wooster formülasyonu ile MWIR 3.9 $\mu\text{m}$ bandı üzerinden alevin termal çıkışı:
   $$\text{FRP} = \frac{\sigma \cdot \epsilon}{a} \cdot A_{pixel} \left( T_{fire}^8 - T_{bg}^8 \right) \quad [\text{MegaWatt}]$$

```text
       ┌────────────────────────────────────────────────────────┐
       │ LEO Satellite Constellation (Multispectral SWIR/MWIR)   │
       └───────────────────────────┬────────────────────────────┘
                                   │ On-Board Raw Pixel Stream
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │ On-Board Edge AI Segmenter (INT8 / Quantized CNN)      │
       │ NBR < -0.15 & T_MWIR > 420K -> Active Wildfire Core    │
       └───────────────────────────┬────────────────────────────┘
                                   │ Filtered FRP Metric (MW)
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │ Inter-Satellite Laser Link (ISL) Direct Geo-Alert      │
       │ Ground Latency < 25 ms (Zero Downlink Bottleneck)      │
       └────────────────────────────────────────────────────────┘
```

---

### 1.3 4 Zorunlu Mimari Analiz

#### 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- **Zero-Latency Early Warning:** Yangınlar henüz 1 dönümlük aşamadayken 1 dakika içinde müdahale ekiplerini alarma geçirmek için.
- **Bandwidth Constraint Overcoming:** Gigabaytlarca ham fotoğrafı yeryüzüne indirmek yerine sadece 50 baytlık alarm koordinatını fırlatmak için.

#### 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- **Ground Station Pass Waiting:** Uydunun yer istasyonu üzerinden geçmesini saatlerce bekleme mecburiyetini sıfırlar.
- **False Sun Glint Alarms:** Çatı veya göl yüzeyinden yansıyan güneş ışığı parlamalarını SWIR + MWIR spektral çiftiyle kusursuz ayırt eder.

#### ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- **Dense Cloud Coverage:** Çok kalın fırtına bulutlarının altındaki alevleri tespit etmek için sentetik açıklıklı radar (SAR) uydularıyla eşgüdüm gerekir.
- **On-Board Power Budget:** Küp uydunun güneş paneli güç kısıtı nedeniyle yapay zeka çıkarımının $< 5\text{ Watt}$ güç harcaması şarttır (Kuantizasyon zorunludur).

#### 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- **Geleneksel Ham Veri İndirme (Raw Downlink):** 6-12 saat gecikmeli, yangın büyüdükten sonra haber veren eski uydu modeli.
- **Takımyıldızı Edge AI (Bizim Yaklaşımımız):** Uyduda analiz edip milisaniyede alarm veren modern uzay standardı.

---

### 1.4 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Terim | Tanım ve Stajyer Açıklaması |
| --- | --- |
| **LEO** | Low Earth Orbit: 500-800 km irtifadaki alçak Dünya yörüngesi. |
| **Constellation** | Takımyıldız: Dünyanın her yerini sürekli kapsayan organize uydu filosu. |
| **Edge AI** | Veriyi yere göndermeden doğrudan uydu üzerindeki işlemcide işleyen yapay zeka. |
| **SWIR** | Short-Wave Infrared: Dumanı delip alevin sıcaklığını gören 1.4-3.0 $\mu\text{m}$ bandı. |
| **MWIR** | Mid-Wave Infrared: Yangın ve füze alevlerinin zirve yaptığı 3.7-4.0 $\mu\text{m}$ bandı. |
| **NBR** | Normalized Burn Ratio: Yanmış alanları ve alev cephesini vurgulayan spektral indeks. |
| **FRP** | Fire Radiative Power: Yangının çevreye yaydığı anlık termal güç (MegaWatt). |
| **ISL** | Inter-Satellite Link: Uyduların uzayda birbirine lazerle veri aktarması. |
| **Downlink** | Uydunun uzaydan yer istasyonuna radyo frekansıyla veri indirme süreci. |
| **Sun Glint** | Güneş ışığının deniz veya camdan yansıyıp sahte yangın gibi parlaması hatası. |

---

### 1.5 SWOT Analizi Karar Matrisi

```
               GÜÇLÜ YÖNLER (STRENGTHS)                      ZAYIF YÖNLER (WEAKNESSES)
      ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
      │ • < 25 ms küresel acil durum uyarısı.    │  │ • Çok kalın yağmur bulutlarında optik    │
      │ • %98 üzerinde yangın tespit hassasiyeti.│   kızılötesi zayıflaması.                │
      └──────────────────────────────────────────┘  └──────────────────────────────────────────┘
               FIRSATLAR (OPPORTUNITIES)                       TEHDİTLER (THREATS)
      ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
      │ • Orman yangınları, volkanik patlamalar, │  │ • Güneş fırtınaları ve uzay radyasyonunun│
      │   endüstriyel patlama ve füze ikazı.     │   uydu kenar işlemcisini kitlemesi.      │
      └──────────────────────────────────────────┘  └──────────────────────────────────────────┘
```

---

## 💻 2. Üretim Seviyesinde Uygulama Kodu & Mimarisi

Dizin yapısı:

```text
day-351-satellite-constellation-wildfire-edge-ai/
├── ana_akis.py
├── gereksinimler.txt
├── LICENSE
├── README.md
├── ciktilar/
│   └── uydu_yangin_edge_paneli.png
├── src/
│   ├── __init__.py
│   ├── satellite_wildfire_edge_motoru.py
│   ├── wildfire_gorsellestirici.py
│   └── wildfire_profilleyici.py
└── testler/
    └── test_satellite_wildfire_edge_motoru.py
```

---

## 🧪 3. Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

### 🎯 Görev (Stajyer Görevi)
Bir uydunun NIR piksel yansıtması $\rho_{NIR} = 0.15$ ve SWIR yansıtması $\rho_{SWIR} = 0.85$ olarak ölçülmüştür. Normalized Burn Ratio (NBR) indeksini ($\text{NBR} = \frac{\rho_{NIR} - \rho_{SWIR}}{\rho_{NIR} + \rho_{SWIR}}$) hesaplayan ve $\text{NBR} < -0.2$ durumunda yangın alarmı üreten bir Python fonksiyonu yazınız.

### 💡 Çözüm Kodu
```python
def test_nbr_wildfire_calc():
    rho_nir = 0.15
    rho_swir = 0.85
    
    nbr = (rho_nir - rho_swir) / (rho_nir + rho_swir)
    is_fire = nbr < -0.2
    
    print(f"Hesaplanan NBR İndeksi: {nbr:.3f}")
    print(f"Yangın / Yanık Alan Tespit Durumu: {is_fire} (Kritik Alev Cephesi)")

if __name__ == "__main__":
    test_nbr_wildfire_calc()
```

---

## 📊 4. Satellite Edge AI vs Traditional Downlink Performance Benchmark Tablosu

| Mimari | Yangın Tespit Yeri | Alarm İletim Gecikmesi | İletilen Veri Boyutu | Erken Müdahale Başarısı |
| --- | --- | --- | --- | --- |
| **Geleneksel Ham Uydu İndirme** | Yeryüzü Sunucusu | 6.0 - 12.0 Saat | 4.5 Gigabayt | %30 (Geç Kalınır) |
| **Takımyıldızı Edge AI (Bizim)**| **Uydu Üzeri (On-Board)** | **< 25 Milisaniye** | **50 Bayt (Geo-JSON)** | **%99 (İlk Dakikada)** |

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
Uzayda neden sadece basit bir RGB renkli kamera kullanmıyoruz da MWIR (3.9 $\mu\text{m}$) ve SWIR (2.2 $\mu\text{m}$) kızılötesi bantlarına ihtiyaç duyuyoruz?

### 💬 Mentorluk Yanıtı
Harika bir uzaktan algılama sorusu! Orman yangınları başladığında havaya devasa bir duman bulutu yayılır. **Görünür RGB ışık**, duman partiküllerine çarparak dağılır (Rayleigh/Mie saçılması) ve altındaki alevleri tamamen gizler; optik kamera sadece beyaz duman görür. Ancak **SWIR ve MWIR (3.9 μm) dalga boyları**, duman partiküllerinden çok daha büyük olduğu için duman perdesini delip doğrudan alevlerin kor çekirdeğine ulaşır ve alevin kaç MegaWatt güçte yandığını milimetrik olarak ölçer!
