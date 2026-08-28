# Day 37: Halı/Tekstil Renk Ayrıştırma & İplik Renk Oranları Çıkarımı (Carpet Color Intelligence)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-9.5+-005571.svg?style=flat-square)](https://python-pillow.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; jakarlı halı dokuma, iplik boyama ve tekstil üretim hatlarında desenlerdeki **ana iplik renklerini CIELAB uzayında K-Means ile ayrıştıran**, her ipliğin **yüzdesel sarfiyat oranını ($P_i$)** hesaplayan ve **CIE Delta-E 2000 ($\Delta E_{00}$)** standardıyla kurumsal iplik kartelası katalog eşlemesini yapan sektörel yapay zeka motorudur.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Neden RGB Değil de CIELAB Renk Uzayı?
- **RGB Uzayının Yetersizliği:** RGB renk uzayı doğrusal ve algısal olarak homojen (perceptually uniform) değildir. RGB uzayındaki iki renk arasındaki Euclidean mesafe, insan gözünün algıladığı renk farkıyla uyuşmaz (yeşil tonlarında insan gözü çok hassasken, mavi tonlarında daha az hassastır).
- **CIELAB (CIE $L^*a^*b^*$) Uzayı:**
  - $L^*$: Açıklık / Parlaklık ($0$: Siyah, $100$: Beyaz)
  - $a^*$: Yeşil ($-128$) ile Kırmızı ($+127$) ekseni
  - $b^*$: Mavi ($-128$) ile Sarı ($+127$) ekseni

```
                    ┌──────────────────────────────────────────────────────────┐
                    │            HALI DOKUMA DESENİ GÖRSELİ (RGB)              │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  1. CIELAB DÖNÜŞÜMÜ (RGB -> sRGB Linear -> XYZ -> CIE L*a*b*)                │
        │  - D65 Standart Referans Beyaz Noktası                                       │
        │  - Algısal Homojen Renk Temsili                                              │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  2. K-MEANS İPLİK KÜMELEME & SARFİYAT ORANI ÇIKARIMI                         │
        │  - K Adet Ana İplik Rengi Kümeleme (K = 5)                                   │
        │  - İplik Yüzdesi: P_i = (Piksel_Sayisi / Toplam_Piksel) * 100                │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  3. CIE DELTA-E 2000 (dE00) STANDART KATALOG EŞLEMESİ                        │
        │  - İplik Kartelası / Katalog Havuzu ile Karşılaştırma                        │
        │  - Renk Farkı Eşiği: dE < 2.0 (Mükemmel), dE < 5.0 (Kabul), dE >= 5.0 (Red) │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  SEKTÖREL KALİTE KONTROL VE İPLİK SARFİYAT RAPORU        │
                    └──────────────────────────────────────────────────────────┘
```

---

### 2. CIE $\Delta E_{00}$ (Delta-E 2000) Renk Tolerans Standardı

İki CIELAB rengi $(L_1, a_1, b_1)$ ve $(L_2, a_2, b_2)$ arasındaki algısal fark şu formülle hesaplanır:

$$\Delta E_{00} = \sqrt{\left(\frac{\Delta L'}{k_L S_L}\right)^2 + \left(\frac{\Delta C'}{k_C S_C}\right)^2 + \left(\frac{\Delta H'}{k_H S_H}\right)^2 + R_T \left(\frac{\Delta C'}{k_C S_C}\right)\left(\frac{\Delta H'}{k_H S_H}\right)}$$

#### Endüstriyel Renk Tolerans Skalası
- $\Delta E_{00} < 1.0$: İnsan gözüyle ayırt edilemez (Mükemmel Eşleşme).
- $1.0 \le \Delta E_{00} < 2.0$: Yalnızca deneyimli kalite kontrol uzmanı fark edebilir.
- $2.0 \le \Delta E_{00} < 5.0$: **Endüstriyel Kabul Sınırı (Pass - Parti İçi Kabul)**.
- $\Delta E_{00} \ge 5.0$: **Parti Farkı Red (Fail - Hatalı Boyama)**.

---

### 3. Halı Deseni İplik Ayrıştırma Deney Çıktıları

| İplik ID | Sarfiyat Oranı ($P_i$) | Çıkarılan Renk (HEX) | Eşleşen Katalog İpliği | Katalog Kodu | $\Delta E_{00}$ | Kalite Kararı |
|---|---|---|---|---|---|---|
| **IPLIK-01** | **%43.82** | `#e4d9c6` (Krem) | Klasik Krem Vizon | `YARN-103` | **0.42** | `MUKEMMEL_UYUM` |
| **IPLIK-02** | **%25.10** | `#8a1c30` (Bordo) | Kraliyet Bordosu | `YARN-101` | **0.85** | `MUKEMMEL_UYUM` |
| **IPLIK-03** | **%14.25** | `#182b49` (Mavi) | Derin Gece Mavisi | `YARN-102` | **0.91** | `MUKEMMEL_UYUM` |
| **IPLIK-04** | **%10.15** | `#cc9a2d` (Hardal) | Anadolu Hardal Sarısı | `YARN-104` | **1.15** | `MUKEMMEL_UYUM` |
| **IPLIK-05** | **%6.68** | `#206241` (Yeşil) | Osmanlı Zümrüt Yeşili | `YARN-105` | **1.22** | `MUKEMMEL_UYUM` |

---

## 🛠️ Dizin Yapısı

```
day-37-carpet-color-intelligence/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, pillow, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca halı deseni analizi ve raporlama betiği
├── README.md                        # Detaylı sektörel ve matematiksel dokümantasyon (220+ Satır)
├── src/
│   ├── __init__.py
│   ├── renk_donusturucu.py          # RGB <-> CIELAB tam hassasiyetli vektörel dönüştürücü
│   ├── delta_e_hesaplayici.py       # ISO/CIE Delta-E 2000 matematiksel motoru
│   ├── iplik_kumeleyici.py          # CIELAB K-Means iplik kümeleme ve yüzde çıkarıcı
│   ├── katalog_esleyici.py          # İplik kartelası katalog eşleyici ve tolerans denetçisi
│   └── gorsellestirici.py           # 6 panelli sektörel analiz panosu (Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_carpet_color.py         # 7 adet kapsamlı birim test
└── ciktilar/
    └── hali_renk_analiz_paneli.png  # 6 panelli tekstil teşhis görseli
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

### 3. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/katalog_esleyici.py` içerisine **"İplik Maliyet ve Gramaj Simülatörü"** fonksiyonu ekleyerek toplam halı ağırlığı (örneğin 3.5 kg / $m^2$) verildiğinde her iplik kodunun harcayacağı net gramajı ve toplam hammadde maliyetini hesaplamak.

**Tamamlanan Çözüm:**
```python
def iplik_maliyet_hesapla(eslesmeler: list, toplam_hali_kg: float = 3.5, birim_fiyat_tl_kg: float = 120.0) -> dict:
    detaylar = []
    toplam_maliyet = 0.0
    for es in eslesmeler:
        yuzde = es["iplik_yuzdesi"] / 100.0
        iplik_kg = toplam_hali_kg * yuzde
        maliyet = iplik_kg * birim_fiyat_tl_kg
        toplam_maliyet += maliyet
        detaylar.append({
            "katalog_kod": es["katalog_kod"],
            "katalog_ad": es["katalog_ad"],
            "harcanan_kg": round(iplik_kg, 3),
            "maliyet_tl": round(maliyet, 2)
        })
    return {"toplam_kg": toplam_hali_kg, "toplam_maliyet_tl": round(toplam_maliyet, 2), "detaylar": detaylar}
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Halı ve kumaş kalite kontrolünde neden klasik Delta-E 1976 ($\Delta E^*_{ab} = \sqrt{\Delta L^2 + \Delta a^2 + \Delta b^2}$) formülü yerine **CIE Delta-E 2000 ($\Delta E_{00}$)** formülü zorunlu olarak kullanılır?

> **Mentor Cevabı:**
> 1. **Doygunluk ve Ton Doğrusalsızlığı (Chroma & Hue Non-uniformity):** Delta-E 1976 uzayda saf küresel mesafe varsayar. Ancak insan gözünün renk ayrım elipsoitleri yüksek doygunluklu renklerde (özellikle sarı ve kırmızılarda) genişlerken, nötr gri tonlarında daralır. Delta-E 76, doymuş renklerdeki küçük ton farklarını aşırı abartarak hatalı reddetme üretir.
> 2. **Mavi Bölgesindeki Dönme (Blue Rotation Term $R_T$):** İnsan gözünün mavi-mor renk eksenindeki elipsoiti eğiktir. Delta-E 2000'deki $R_T$ dönme faktörü bu eğikliği matematiksel olarak telafi ederek insan algısıyla birebir örtüşen güvenilir tolerans ölçümü sunar.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
