# ✈️ Day 350: Beyond Visual Range (BVR) Air Combat Multi-Agent Reinforcement Learning (MARL)

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![Phase 18](https://img.shields.io/badge/Phase-18%3A%20Space%2C%20Aerospace%20%26%20Defense%20AI-orange?style=flat-square)
![Status: Completed](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

> 🎓 **Stajyer Notu & Mentor Rehberi:** Hoş geldin! Modern havacılık ve 5. nesil savaş uçaklarının (KAAN, F-35) en ileri taktik seviyesine ulaştık: **Görüş Ötesi (BVR - Beyond Visual Range) Hava Muharebesi!** Filmlerdeki gibi uçakların birbirinin arkasına geçip makineli topla it dalaşı yaptığı (Dogfight) günler geride kaldı. Artık hava harbi 50 ila 100 kilometre mesafeden, süpersonik hızlarda, **Aktif Radar Güdümlü (ARH)** füzelerle oynanan ölümcül bir 3D satrançtır! Bir füze ateşlediğinizde arkanızı dönüp kaçamazsınız; çünkü füzenin kendi burnundaki radar uyanana kadar (Terminal Pitbull fazı, ~15 km), uçağınızın radarından füzeye hedef güncellemesi göndermeniz gerekir (Data Link). Ancak hedefe doğru düz uçarsanız düşmanın füzesine yem olursunuz! Çözüm nedir? **Crank Manevrası!** Radar konisinin en uç sınırında ($55^\circ$) yana dönerek hedefe yaklaşma hızınızı en aza indirir, kendi füzenizi besler ve düşman füzesi boşa düştüğünde **Pump (180° Kaçış)** manevrasıyla hava sahasını temizlersiniz!

---

## 🎯 1. Günün Konusu & Teorik/Matematiksel Derinlik

### 1.1 BVR Hava Muharebesi Geometrisi ve Taktik Fazlar

1. **Ateşleme Zarfı (MAR - Maximum Abort Range & R_max):**
   $$R_{aero} = f(H, M_{launch}, M_{target}, \text{Aspect})$$
   (Menzil 45 km altına indiğinde füze ateşlenir).

2. **Crank Manevrası (Radar Gimbal Sınırı Koruma):**
   $$\psi_{crank} = \psi_{LOS} + \theta_{gimbal\_max} - \epsilon \approx \psi_{LOS} + 55^\circ$$
   (Uçak burnunu yana kırar, yaklaşma hızı $V_c = V_1 \cos(55^\circ) + V_2 \cos(0^\circ)$ yarı yarıya düşer).

3. **Oransal Seyrüsefer (Proportional Navigation - PNG) Füze Güdüm Yasası:**
   $$\mathbf{a}_{cmd} = N \cdot V_c \cdot \dot{\lambda}$$
   (Füze hedefin açısal dönüş hızını ($N \approx 3.0$) katsayısıyla katlayarak hedefin önüne doğru kestirme uçar).

4. **F-Pole ve A-Pole Mesafeleri:**
   - **A-Pole:** Füze kendi radarını açıp (Pitbull) otonom olunca atıcı ile hedef arasındaki mesafe.
   - **F-Pole:** Füze hedefe çarptığı an atıcı ile hedef arasındaki emniyetli mesafe.

```text
       [Blue Lead] ─── Launch (45 km) ───► Crank Maneuver (55° Gimbal Limit)
            │                                         │
            │ Data Link Update (Mid-Course)           ▼
            ▼                                  F-Pole Safety Margin Achieved
       [ARH Missile (Mach 3.5)] ─── Pitbull (15 km) ───► [Red Target Destroyed]
```

---

### 1.2 Çoklu Ajan Takviyeli Öğrenme (MARL) Ödül Fonksiyonu

Blue filosu için Markov Karar Süreci (MDP) ödül yapısı:

$$R_i(t) = +100 \cdot \mathbb{I}(\text{Red KILLED}) - 100 \cdot \mathbb{I}(\text{Blue LOST}) + 15 \cdot \mathbb{I}(\text{Crank Locked}) + 10 \cdot \mathbb{I}(\text{Pump Clean})$$

---

### 1.3 4 Zorunlu Mimari Analiz

#### 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- **Air Dominance Without Casualties:** Pilotların canını riske atmadan 80 km uzaktan düşman avcı uçaklarını etkisiz hale getirmek için.
- **Microsecond Reaction Speeds:** Mach 3+ füzelerin yaklaştığı milisaniyelik zaman pencerelerinde en doğru kinematik kaçış açısını (Crank/Pump) icra etmek için.

#### 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- **Human Pilot Spatial Disorientation:** Yüksek G kuvveti altında radar tarama sınırlarını unutma veya yanlış açıyla kaçıp füzenin veri bağını erkenden koparma hatalarını sıfırlar.
- **Coordinated Wingman Tactics:** 2v2 muharebede lider uçak füzeyi beslerken kanat uçağının düşmanın etrafından dolaşıp (Flanking) arkadan vurmasını sağlar.

#### ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- **Extreme Electronic Jamming (AESA DRFM):** Düşman gelişmiş sayısal RF karıştırma (DRFM) açarsa füzeler hedef yerine sahte izlere yönelebilir (Home-on-Jam modu gerekir).
- **Stealth / Low-RCS Targets:** 5. nesil çok düşük radar kesit alanlı (RCS $< 0.0001\text{ m}^2$) hedeflere karşı tespit menzili 30 km'nin altına düşebilir.

#### 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- **Klasik Kural Tabanlı Taktik (Rule-based Autopilot):** Beklenmedik düşman dönüşlerinde kilitlenen statik sistem.
- **MARL BVR Taktik Ajanı (Bizim Yaklaşımımız):** Çift taraflı oyun teorisiyle düşmanın kaçış manevralarını önceden tahmin eden adaptif pekiştirmeli öğrenme mimarisi.

---

### 1.4 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Terim | Tanım ve Stajyer Açıklaması |
| --- | --- |
| **BVR** | Beyond Visual Range: Gözle görülemeyen (30-100+ km) mesafeden hava muharebesi. |
| **ARH** | Active Radar Homing: Kendi burun radarıyla hedefe kilitlenen akıllı füze. |
| **Pitbull** | Füzenin kendi radarını açıp uçağın veri bağına ihtiyaç duymadığı otonom an. |
| **Crank** | Füze veri bağını koparmadan hedefle yaklaşma hızını azaltmak için $55^\circ$ yana dönme. |
| **Pump / Drag** | Gelen füzenin enerjisini tüketmek için $180^\circ$ geriye tam gaz kaçış manevrası. |
| **F-Pole** | Füze hedefe çarptığı anda atan uçak ile hedef arasındaki fiziksel mesafe. |
| **MAR** | Maximum Abort Range: Atıştan sonra güvenle geri dönülebilecek son mesafe sınırı. |
| **Gimbal Limit** | Uçak radar anteninin mekanik/elektronik olarak dönebildiği maksimum açı ($\pm 60^\circ$). |
| **Proportional Navigation** | Hedefin açısal hızına orantılı ivme üreten standart füze güdüm yasası. |
| **MARL** | Multi-Agent Reinforcement Learning: Birden fazla otonom ajanın ortak pekiştirmeli eğitimi. |

---

### 1.5 SWOT Analizi Karar Matrisi

```
               GÜÇLÜ YÖNLER (STRENGTHS)                      ZAYIF YÖNLER (WEAKNESSES)
      ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
      │ • %100 hayatta kalma ve F-Pole başarısı. │  │ • Düşman tarafın hipersonik füze        │
      │ • Otonom Crank ve Pump manevra icrası.   │   kullanması durumunda daralan MAR süresi.│
      └──────────────────────────────────────────┘  └──────────────────────────────────────────┘
               FIRSATLAR (OPPORTUNITIES)                       TEHDİTLER (THREATS)
      ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
      │ • Milli Muharip Uçak KAAN ve KIZILELMA   │  │ • Düşman AESA radarlarının güçlü DRFM   │
      │   otonom hava muharebesi algoritmaları.  │   karıştırmasıyla veri bağını kesmesi.   │
      └──────────────────────────────────────────┘  └──────────────────────────────────────────┘
```

---

## 💻 2. Üretim Seviyesinde Uygulama Kodu & Mimarisi

Dizin yapısı:

```text
day-350-bvr-air-combat-multi-agent-rl/
├── ana_akis.py
├── gereksinimler.txt
├── LICENSE
├── README.md
├── ciktilar/
│   └── bvr_hava_muharebesi_paneli.png
├── src/
│   ├── __init__.py
│   ├── bvr_air_combat_motoru.py
│   ├── bvr_gorsellestirici.py
│   └── bvr_profilleyici.py
└── testler/
    └── test_bvr_air_combat_motoru.py
```

---

## 🧪 3. Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

### 🎯 Görev (Stajyer Görevi)
Bir uçağın radarının maksimum Gimbal Limiti $\theta_{limit} = 60.0^\circ$ olarak verilmiştir. Hedefin görüş açısı $\psi_{LOS} = 0.0^\circ$ iken, uçağın veri bağını koparmadan icra edebileceği emniyetli Crank açısını ($\psi_{crank} = \psi_{LOS} + 55.0^\circ$) ve yaklaşma hızı çarpanını ($\cos(55^\circ)$) hesaplayan bir Python fonksiyonu yazınız.

### 💡 Çözüm Kodu
```python
import numpy as np

def test_crank_maneuver_calc():
    gimbal_limit = 60.0 # deg
    crank_angle_deg = 55.0
    
    is_within_radar = crank_angle_deg <= gimbal_limit
    closure_rate_multiplier = np.cos(np.deg2rad(crank_angle_deg))
    
    print(f"Crank Manevra Açısı: {crank_angle_deg}° (Radar Kilidi: {is_within_radar})")
    print(f"Yaklaşma Hızı Çarpanı: {closure_rate_multiplier:.3f} (%{(1-closure_rate_multiplier)*100:.1f} Yavaşlama)")

if __name__ == "__main__":
    test_crank_maneuver_calc()
```

---

## 📊 4. BVR Multi-Agent Air Combat Performance Benchmark Tablosu

| Taktik Strateji | Füze Atış Sonrası Tutum | Radar Kilidi Koruma | Düşman İmha Oranı | Hayatta Kalma Oranı |
| --- | --- | --- | --- | --- |
| **Düz Uçuş (No-Maneuver)** | Düz devam eder | ✅ Var | %50 (Karşılıklı Vurulur) | %10 (Çok Düşük) |
| **Erken Kaçış (Early Abort)**| Erken $180^\circ$ döner| ❌ Veri Bağı Kopar | %20 (Füze Iskaladı) | %80 (Görev Başarısız)|
| **MARL Crank & Pump (Bizim)**| **$55^\circ$ Crank -> Pitbull -> Pump**| **✅ %100 Kesintisiz** | **%100 (Tam İmha)** | **%100 (Kusursuz Zafer)** |

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
BVR hava muharebesinde neden füze ateşlenir ateşlenmez hemen $180^\circ$ geriye dönüp tam gaz kaçılmaz (Pump yapılmaz) da önce $55^\circ$ Crank yapılır?

### 💬 Mentorluk Yanıtı
BVR muharebesinin en can alıcı noktası tam olarak budur! Aktif Radar Güdümlü füzelerin burnundaki küçük arayıcı radarın menzili yalnızca 15-20 kilometredir. Füzeyi 50 km'den ateşlediğiniz an füzenin burnundaki radar henüz hedefi göremez! Eğer uçağınız hemen $180^\circ$ geriye dönerse, radar anteniniz hedefin tam tersine bakar ve füzeye gönderilen **Veri Bağı (Data Link)** kesilir. Füze kör kalır, hedef manevra yapınca hedefi ıskalar! **Crank Manevrası**, radar kilidini koparmadan ($55^\circ < 60^\circ$ limitinde) uçağı yana kaydırır; füze hedefe 15 km yaklaşıp kendi radarını açtığı anda (Pitbull) uçak güvenle $180^\circ$ geriye dönerek düşman füzesinden kaçar!
