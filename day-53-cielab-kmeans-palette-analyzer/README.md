# Day 53: CIELAB Renk Uzayında K-Means & Delta-E 2000 Hassas Tolerans Analizi (Industrial Colorimetry & CIEDE2000 Tolerance Engine)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.21+-blue.svg?style=flat-square)](https://scikit-image.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 53. gününde geliştirilen **Endüstriyel Kolorimetri, Algısal Renk Paleti Çıkarıcı ve CIEDE2000 Kalite Tolerans Analiz Motorudur**. Tekstil, halı, otomotiv boya ve ekran kalibrasyonu gibi yüksek hassasiyet gerektiren endüstrilerde insan gözünün algısal doğrusal olmama durumunu (MacAdam Elipsleri) modelleyerek standart sRGB yerine **CIELAB ($L^*a^*b^*$)** uzayında K-Means kümeleme yapar ve **ISO/CIE 116-2019 ($\Delta E_{2000}$)** standardı ile parti renk sapmalarını mikron hassasiyetinde denetler.

---

## 📖 Mentorluk Dersi ve Algısal Kolorimetri Teorisı

### 1. Neden Standart sRGB Öklid Mesafesi Yetersizdir?

Standart sRGB uzayında renkler donanım odaklıdır (CRT/LCD ekranların fosfor/LED tepkileri). İnsan gözü (retinadaki koni hücreleri) renklere doğrusal tepki vermez:
- İnsan retinası **yeşil ve camgöbeği (cyan)** tonlarındaki mikro sapmalara son derece hassasken, doymuş **mavi ve kırmızı** bölgelerdeki büyük sayısal değişimleri çok daha az fark eder (MacAdam Elipsleri).
- Standart RGB üzerinde K-Means kümelemesi yapıldığında küme merkezleri görsel olarak yanlış temsil edilir ve hesaplanan Öklid mesafesi $\sqrt{(\Delta R)^2 + (\Delta G)^2 + (\Delta B)^2}$ gerçek insan gözü renk sapmasını yansıtamaz.

### 2. CIELAB ($L^*, a^*, b^*$) Algısal Olarak Üniform Renk Uzayı

1976 yılında Uluslararası Aydınlatma Komisyonu (CIE) tarafından insan algısına göre geliştirilmiştir:
- **$L^* \in [0, 100]$:** Algılanan Açıklık / Parlaklık (0: Saf Siyah, 100: Mükemmel Beyaz).
- **$a^* \in [-128, +127]$:** Yeşil (negatif) ile Kırmızı/Macenta (pozitif) ekseni.
- **$b^* \in [-128, +127]$:** Mavi (negatif) ile Sarı (pozitif) ekseni.

```
                           ┌──────────────────────────────────────────────────────────┐
                           │          REFERANS STANDART & ÜRETİM PARTİSİ              │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 RenkUzayiDonusturucu (sRGB -> Standart CIELAB D65 Dönüşümü)                       │
    │  - Pikseller Doğrusal Olmayan Algısal CIELAB (L*, a*, b*) Uzayına Taşınır                         │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 CIELABKMeansPaletAnalizoru (LAB Uzayında K-Means Kümeleme)                         │
    │  - K=4..8 Dominant Merkezler Çıkarılır, Görsel Baskınlık Yüzdeleri ve HEX Kodları Hesaplanır      │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 DeltaEHesaplayici (Delta-E 1976 vs ISO/CIE CIEDE2000 Formülasyonu)                │
    │  - Doygunluk (Chroma), Ton (Hue) ve Mavi Bölge Dönme (Rotation RT) Düzeltmeleri Hesaplanır        │
    │  - Endüstriyel Kalite Toleransı: PASS (<2.0 dE00), WARNING (2.0-5.0), REJECT (>=5.0)             │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │               6-PANELLİ KOLORİMETRİ VE DELTA-E 2000 TEŞHİS PANOSU (Day 53)                        │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Matematiksel Formülasyonlar

#### A. Delta-E 1976 ($\Delta E_{76}$ - Basit Öklid Mesafesi)
$$\Delta E_{76} = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}$$

#### B. CIEDE2000 Formülasyonu (ISO/CIE 116-2019 Standardı)
$$\Delta E_{00} = \sqrt{ \left(\frac{\Delta L'}{k_L S_L}\right)^2 + \left(\frac{\Delta C'}{k_C S_C}\right)^2 + \left(\frac{\Delta H'}{k_H S_H}\right)^2 + R_T \left(\frac{\Delta C'}{k_C S_C}\right)\left(\frac{\Delta H'}{k_H S_H}\right) }$$

Burada:
- **$S_L, S_C, S_H$:** Açıklık, Doygunluk ve Ton ağırlık fonksiyonları.
- **$R_T$:** Mavi bölgede ($h \approx 275^\circ$) insan gözünün eliptik yönelimini telafi eden dönme faktörü (Rotation function).
- **$k_L, k_C, k_H$:** Parametrik katsayılar (tekstil ve genel endüstri için $1.0$).

---

## 🛠️ Dizin Yapısı

```
day-53-cielab-kmeans-palette-analyzer/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # opencv-python, scikit-learn, scikit-image, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca kumaş palet çıkarımı ve Delta-E analiz boru hattı
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── renk_uzayi_donusturucu.py    # RenkUzayiDonusturucu (sRGB, CIELAB D65, HEX çift yönlü dönüşüm)
│   ├── cielab_kmeans_analizor.py    # CIELABKMeansPaletAnalizoru (LAB K-Means dominant palet çıkarımı)
│   ├── delta_e_hesaplayici.py       # DeltaEHesaplayici (Delta-E 76, CIEDE2000 ve kalite tolerans motoru)
│   └── gorsellestirici.py           # 6-Panelli Kolorimetri Teşhis Panosu (Palette & Tolerance Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_cielab_palette.py       # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── cielab_palet_tolerans_paneli.png # 6 panelli yüksek çözünürlüklü teşhis panosu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışın Çalıştırılması
```bash
python ana_akis.py
```

### 3. Birim Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📊 CIEDE2000 Endüstriyel Kalite Tolerans Tablosu

| Delta-E ($\Delta E_{00}$) | Seviye Kodu | Karar | İnsan Algısı ve Endüstriyel Anlam | Tipik Uygulama Alanı |
|---|---|---|---|---|
| **$< 1.0$** | `MUKEMMEL_ESLESME` | **PASS** | İnsan gözüyle ayırt edilemez (Imperceptible). | Lüks Otomotiv Boyası, Ekran Kalibrasyonu |
| **$1.0 \le \Delta E < 2.0$** | `TOLERANS_DAHILINDE` | **PASS** | Yalnızca uzman gözle yakından bakıldığında fark edilir. | Premium Tekstil, Mobilya Kaplama |
| **$2.0 \le \Delta E < 5.0$** | `KABUL_SINIRINDA` | **WARNING** | Standart gözlemci tarafından fark edilebilir renk sapması. | Seri Üretim Plastik, Ambalaj Baskısı |
| **$\ge 5.0$** | `KRITIK_RED` | **REJECT** | Belirgin renk uyumsuzluğu, hatalı boyama partisi. | Hatalı Ürün / Fire Ayrıştırma |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Farklı üretim partilerindeki ($N$ adet parti) renk kaymasını referans standarta göre takip eden ve lot trend sapması grafiği üreten **"Multi-Batch Color Drift Tracker"** fonksiyonu geliştirmek.

**Tamamlanan Çözüm:**
```python
def parti_sapma_takip_edici(ref_lab: np.ndarray, partiler_lab: list) -> list:
    """Üretim partilerinin referansa göre CIEDE2000 renk sapma trendini hesaplar."""
    trend_sonuclari = []
    for idx, p_lab in enumerate(partiler_lab):
        de00 = DeltaEHesaplayici.delta_e_2000(ref_lab, p_lab)
        tolerans = DeltaEHesaplayici.tolerans_degerlendir(de00)
        trend_sonuclari.append({
            "parti_no": idx + 1,
            "delta_e_2000": de00,
            "karar": tolerans["seviye"],
            "kod": tolerans["kod"]
        })
    return trend_sonuclari
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** $\Delta E_{76}$ (Öklid LAB mesafesi) $4.2$ birim çıkmasına rağmen, aynı renk çifti için $\Delta E_{2000}$ değeri neden $1.6$ (Tolerans Dahilinde / PASS) çıkabilir? Bu farkın temel matematiksel sebebi nedir?

> **Mentor Cevabı:**
> 1. **Doygunluk ve Ton Bağımlılığı:** $\Delta E_{76}$, LAB koordinatlarını Kartezyen uzayda doğrusal kabul eder. Ancak insan gözü yüksek doygunluklu (yüksek kroma $C^*$) canlı renklerdeki küçük kaymaları, soluk/pastel tonlardaki aynı sayısal kaymaya kıyasla çok daha zor fark eder.
> 2. **CIEDE2000 Ağırlıklandırması ($S_C$ ve $S_H$):** CIEDE2000 formülünde $S_C = 1 + 0.045\bar{C}'$ ve $S_H = 1 + 0.015\bar{C}' T$ paydada yer alır. Yüksek kromalı bölgelerde bu terimler mesafeyi normalize ederek insan algısına uygun hale getirir. Bu sayede $\Delta E_{76}$ yapay olarak yüksek bir hata gösterirken, $\Delta E_{2000}$ insan gözünün gerçek algısını yansıtarak gereksiz üretim firesini (False Rejection) önler.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
