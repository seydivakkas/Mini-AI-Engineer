# Day 12: Algısal Renk Benzerliği ve Arama Altyapısı (Color Similarity & Retrieval Engine)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-f89939.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; e-ticaret kataloglarında, moda/tekstil tasarımında ve içerik tabanlı görsel erişiminde (CBIR) kullanıcıların aradığı renk uyumuna en yakın ürünleri bulabilmek için **CIELAB uzayında CIE76 ve ISO/CIE standartlarındaki CIEDE2000 $\Delta E$ renk farkı metriklerini hesaplayan, farklı uzunluk ve ağırlıktaki paletleri çift yönlü ağırlıklı algısal mesafe ile eşleştiren ve tüm kataloğu benzerlik yüzdesine göre sıralayan** üretim kalitesinde bir arama ve öneri motorudur.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Görsel arama motorlarında (Pinterest, Google Lens, Trendyol vb.) en sık karşılaşılan arama senaryolarından biri:
> *"Oturma odamdaki kiremit kırmızısı koltuğum ve hardal sarısı perdemle uyumlu halıları göster!"*

Geleneksel derin öğrenme veya metin aramaları renk tonlarının hassas uyumunu kaçırabilir. İki görselin rengini kıyaslarken:
1. **RGB Öklid Mesafesi Çöker:** RGB uzayı algısal olarak homojen değildir (Non-uniform). İnsan gözü yeşil tonlarındaki küçük değişimlere aşırı duyarlıyken, mavi tonlarındaki büyük değişimleri fark etmeyebilir.
2. **Palet Eşleştirme Karmaşıklığı:** Sorgu paletinde 2 renk varken katalogdaki halıda 4 renk olabilir. Renklerin ağırlıkları (%60 vs %10) farklıdır. Basit bir vektör uzaklığı hesaplanamaz.

**Çözüm:** İnsan biyolojik algısına dayanan **CIELAB $\Delta E$** renk mesafesini ve **çift yönlü ağırlıklı palet eşleştirme (Bidirectional Weighted Matching)** algoritmasını kullanmaktır.

---

### 2. Matematiksel ve Algoritmik Mantık

#### A. CIELAB Renk Uzayı ve $\Delta E$ Farkı
1. **CIE76 (Klasik Öklid Farkı):**
   $$\Delta E^*_{76} = \sqrt{(L_1 - L_2)^2 + (a_1 - a_2)^2 + (b_1 - b_2)^2}$$
   - $\Delta E < 1.0$: İnsan gözü farkı ayırt edemez.
   - $1.0 < \Delta E < 3.0$: Çok dikkatli bakıldığında fark edilir.
   - $\Delta E > 10.0$: Tamamen farklı renkler.

2. **CIEDE2000 ($\Delta E_{00}$ — Altın Standart):**
   CIE76'nın insan gözünün ton (Hue) ve doygunluk (Chroma) non-lineerliklerini telafi etmek için geliştirilmiştir:
   $$\Delta E_{00} = \sqrt{\left(\frac{\Delta L'}{k_L S_L}\right)^2 + \left(\frac{\Delta C'}{k_C S_C}\right)^2 + \left(\frac{\Delta H'}{k_H S_H}\right)^2 + R_T \left(\frac{\Delta C'}{k_C S_C}\right)\left(\frac{\Delta H'}{k_H S_H}\right)}$$
   - $S_L, S_C, S_H$: Aydınlık, doygunluk ve ton ağırlıklandırma fonksiyonları.
   - $R_T$: Özellikle mavi-mor bölgedeki eksen kaymasını düzelten rotasyon terimi.

#### B. Çift Yönlü Ağırlıklı Palet Mesafesi (Bidirectional Weighted Distance)
$P = \{(c_i, w_i)\}$ sorgu paleti ve $Q = \{(q_j, v_j)\}$ katalog ürün paleti olsun:
1. **$P \to Q$ Mesafesi:** Sorgudaki her renk için katalogdaki en yakın renk bulunur ve sorgudaki ağırlığıyla çarpılır:
   $$D(P \to Q) = \sum_{i} w_i \min_{j} \Delta E(c_i, q_j)$$
2. **$Q \to P$ Mesafesi:** Katalogdaki her renk için sorgudaki en yakın renk bulunur:
   $$D(Q \to P) = \sum_{j} v_j \min_{i} \Delta E(q_j, c_i)$$
3. **Simetrik Palet Mesafesi:**
   $$D(P, Q) = \frac{D(P \to Q) + D(Q \to P)}{2}$$

#### C. Benzerlik Skoru Fonksiyonu (%0 - %100)
Mesafe sıfır olduğunda benzerlik tam %100 olmalı; mesafe büyüdükçe pürüzsüz biçimde azalmalıdır:
$$\text{Benzerlik} = 100.0 \times \exp\left(-\frac{D(P, Q)}{\sigma}\right)$$
- $\sigma = 25.0$ ölçeğinde:
  - $\Delta E = 1.5 \to \%94.2$ (Mükemmel eşleşme)
  - $\Delta E = 5.0 \to \%81.9$ (Çok iyi benzerlik)
  - $\Delta E = 15.0 \to \%54.9$ (Fark edilir ton farkı)
  - $\Delta E \ge 40.0 \to < \%20$ (Tamamen uyumsuz)

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Tek Yönlü Eşleştirme Yanılgısı:**
   Yalnızca $P \to Q$ yönünde bakarsanız; sorgudaki 1 rengi içeren fakat geri kalan %90'ı alakasız yeşil olan bir ürün %100 benzer çıkabilir! Simetri ($P \to Q$ ve $Q \to P$) bu açığı tamamen kapatır.
2. **Ağırlık Normalizasyonu:**
   Kullanıcı paletindeki ağırlıkların toplamı her zaman $1.0$ olmalıdır; aksi takdirde benzerlik skorları ölçek hatasına uğrar.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
       [Kullanıcı Sorgu Paleti]              [Katalogdaki Ürün Paletleri]
       (Kiremit %65, Hardal %35)            (5 Farklı Halı / Ürün Paleti)
                  │                                        │
                  ▼                                        ▼
      ┌────────────────────────────────────────────────────────┐
      │                  DeltaEHesaplayici                     │
      │        (RGB -> Standart CIELAB Float Dönüşümü)         │
      │        - CIE76 ve CIEDE2000 (ΔE00) Algoritmaları       │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │                 PaletBenzerlikMotoru                   │
      │    - A'dan B'ye Ağırlıklı En Yakın Eşleşme (D_AtoB)    │
      │    - B'den A'ya Ağırlıklı En Yakın Eşleşme (D_BtoA)    │
      │    - Simetrik Mesafe: (D_AtoB + D_BtoA) / 2            │
      │    - Benzerlik Skoru: 100 * exp(-D / σ)                │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │                RenkTabanliAramaMotoru                  │
      │       Kataloğu Tara -> Skorla -> Azalan Sırala         │
      │                 Top-3 En Benzer Ürün                   │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │                 AramaGorsellestirici                   │
      │       [Sorgu Görseli] vs [Top-K Eşleşen Ürünler]       │
      │    Palet Şeritleri ve Renkli Benzerlik Rozetleri       │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
                 [ciktilar/renk_arama_sonuclari.png]
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

1. **[`src/delta_e_hesaplayici.py`](./src/delta_e_hesaplayici.py):**
   - `DeltaEHesaplayici`: RGB'den $L^* \in [0, 100]$ CIELAB uzayına float dönüşümü, CIE76 ve CIEDE2000 algoritmaları.
2. **[`src/palet_eslestirici.py`](./src/palet_eslestirici.py):**
   - `PaletRengi`: RGB, ağırlık, HEX ve LAB özelliklerini tutan veri sınıfı.
   - `PaletBenzerlikMotoru`: Çift yönlü ağırlıklı palet mesafesi ve üstel benzerlik skoru hesaplayıcı.
3. **[`src/katalog_arama.py`](./src/katalog_arama.py):**
   - `KatalogUrunu`: Ürün kimliği, görseli ve renk paleti.
   - `RenkTabanliAramaMotoru`: Kataloğu tarayan ve Top-K benzer ürünü listeleyen arama motoru.
4. **[`src/gorsellestirici.py`](./src/gorsellestirici.py):**
   - `AramaGorsellestirici`: Sorgu ve bulunan ürünleri, renk şeritlerini ve benzerlik rozetlerini tek çizelgede çizen modül.
5. **[`ana_akis.py`](./ana_akis.py):**
   - 5 farklı halı içeren kataloğu oluşturan, sıcak tonlu sorguyu tarayan ve sıralayan konsol yürütücüsü.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
==========================================================================
>>> AŞAMA 1: Sentetik Katalog Ürünlerinin ve Renk Paletlerinin Tanımlanması
==========================================================================
[+] Kataloğa 5 adet ürün kaydedildi.
    - [HL-101] Ege Mavisi Klasik (Palet Rengi: 3 adet)
    - [HL-102] Sonbahar Toprak (Palet Rengi: 3 adet)
    - [HL-103] Akdeniz Güneşi (Palet Rengi: 3 adet)
    - [HL-104] İskandinav Minimal (Palet Rengi: 3 adet)
    - [HL-105] Tropik Zümrüt (Palet Rengi: 3 adet)

==========================================================================
>>> AŞAMA 2: Kullanıcı Sorgusunun Oluşturulması (Kiremit & Hardal Paleti)
==========================================================================
[+] Sorgu Paleti:
    * #C32D23 (RGB: (195, 45, 35)) -> Ağırlık: %65
    * #D7AF23 (RGB: (215, 175, 35)) -> Ağırlık: %35

==========================================================================
>>> AŞAMA 3: CIEDE2000 Algısal Benzerlik Taraması ve Sıralama
==========================================================================
Sıra  | Ürün ID  | Ürün Adı             | Benzerlik    | Delta-E    | En Yakın Eşleşme
--------------------------------------------------------------------------
#1    | HL-102   | Sonbahar Toprak      | %88.7        | 3.00       | #C32D23 -> #BE281E (Delta-E=1.49)
#2    | HL-103   | Akdeniz Güneşi       | %69.6        | 9.05       | #C32D23 -> #D23214 (Delta-E=5.23)
#3    | HL-104   | İskandinav Minimal   | %30.2        | 29.95      | #C32D23 -> #3C3C3C (Delta-E=31.35)
--------------------------------------------------------------------------

==========================================================================
>>> AŞAMA 4: Görsel Arama Paneli ve Benzerlik Rozetlerinin Kaydedilmesi
==========================================================================
[V] Arama sonuçları görsel paneli kaydedildi: renk_arama_sonuclari.png
[V] Kayıt Konumu: day-12-color-similarity-engine/ciktilar/renk_arama_sonuclari.png

[V] Day 12: Algısal Renk Benzerliği ve Arama Altyapısı başarıyla tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Renk Baskınlığına Göre Dinamik Eşikleme Filtresi (Relevance Cutoff)**

Arama motorlarında benzerliği çok düşük (örneğin $\text{Benzerlik} < \%40$) olan alakasız ürünlerin kullanıcıya hiç gösterilmemesi istenir.

### Görev Tanımı:
[`src/katalog_arama.py`](./src/katalog_arama.py) dosyasındaki `RenkTabanliAramaMotoru.arama_yap` metoduna bir eşik parametresi eklemeni bekliyorum:

```python
def filtreli_arama_yap(
    self,
    sorgu_paleti: List[PaletRengi],
    minimum_benzerlik_yuzdesi: float = 50.0,
    en_iyi_k: int = 5
) -> List[AramaSonucu]:
```

### Beklenen Kurallar:
1. Katalogdaki tüm ürünler için benzerlik skorunu hesaplamalı.
2. Skoru `minimum_benzerlik_yuzdesi` değerinin altında kalan ürünleri doğrudan eleyerek listeye almamalı.
3. Kalan sonuçları yine yüksekten düşüğe doğru sıralayıp en fazla `en_iyi_k` adedini döndürmelidir.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> CIEDE2000 formülünde yer alan **$R_T$ (Rotation Term)** terimi, özellikle renk çemberindeki **Mavi-Mor ($h' \approx 275^\circ$)** bölgesinde neden hayati bir düzeltme yapar? Bu terim olmasaydı klasik CIE76 veya CIE94 formülleri mavi renklerde insan gözünün algısıyla neden çelişiyordu?  
> *(İpucu: CIELAB uzayında mavi bölgedeki MacAdam elipslerinin ana ekseninin eğimini ve insan gözünün mavideki kroma/ton algı eksenini düşünün).*

---

## 📂 Dizin Yapısı

```
day-12-color-similarity-engine/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (opencv-python, numpy, scikit-learn, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve görsel arama üretim akışı
├── ciktilar/                   # Üretilen arama sonuçları paneli
│   └── renk_arama_sonuclari.png
├── src/
│   ├── __init__.py
│   ├── delta_e_hesaplayici.py   # RGB->LAB, CIE76 ve CIEDE2000 motoru
│   ├── palet_eslestirici.py    # Çift yönlü ağırlıklı palet eşleştirme ve benzerlik skoru
│   ├── katalog_arama.py        # Katalog indeksleme ve Top-K arama motoru
│   └── gorsellestirici.py      # Sorgu vs Sonuç karşılaştırma paneli
└── testler/
    └── test_renk_benzerligi.py # 7 adet birim testi (7 passed in 1.08s)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışı Çalıştırma
```bash
python ana_akis.py
```

### 3. Testleri Koşma
```bash
python -m pytest testler/test_renk_benzerligi.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
