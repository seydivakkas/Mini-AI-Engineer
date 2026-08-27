# Day 06: Kapsamlı Keşifçi Veri Analizi Laboratuvarı (EDA Lab)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.2+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.9+-11557c.svg?style=flat-square&logo=matplotlib)](https://matplotlib.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; bilgisayarlı görü ve yapay zeka öznitelik tablolarında **doğrusal (Pearson) ve monotonik (Spearman) korelasyon analizlerini**, makine öğrenimi modellerini zehirleyen **Çoklu Doğrusallık (Multicollinearity)** problemini **Varyans Şişme Faktörü (VIF)** ile tespit etmeyi ve **otomatik grafik görselleştirmelerini (Isı Haritası, Histogram, Saçılım)** sunucu dostu (headless) olarak üreten kapsamlı bir keşifçi veri analizi (EDA) laboratuvarıdır.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Bir görüntü sınıflandırma, nesne tespiti veya endüstriyel kalite kontrol modelini (ör. halı/kumaş dokuma hatası ve kusur alanı tahmini) eğitmeden önce şu kritik soruların yanıtlanması gerekir:
- İki öznitelik birbiriyle neredeyse %100 aynı bilgiyi mi taşıyor? (Örneğin: `iplik_sikligi` ile `dugum_sayisi`).
- Bu iki özniteliği aynı anda modele vermek neden zararlıdır?
- Hangi öznitelikler hedef değişkeni (kusur alanını) en güçlü şekilde açıklamaktadır?

Birbiriyle yüksek korelasyonlu değişkenler bir regresyon veya derin öğrenme modeline birlikte verildiğinde **Çoklu Doğrusallık (Multicollinearity)** oluşur; matrisin tersi sayısal olarak kararsızlaşır, modelin öğrendiği ağırlık katsayıları anlamsız dalgalanmalara uğrar ve test setindeki genelleme performansı çöker.

---

### 2. Matematiksel ve Algoritmik Temeller

#### A. Pearson vs. Spearman Korelasyonu
- **Pearson Korelasyon Katsayısı ($r$):** İki değişken arasındaki **doğrusal (lineer)** ilişkinin gücünü ve yönünü ölçer:
  $$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}$$
  - Değer aralığı: $[-1.0, 1.0]$.
  - Yalnızca doğrusal ilişkileri yakalar; eğrisel (non-linear) ilişkilerde sıfıra yakın çıkabilir.

- **Spearman Sıra Korelasyon Katsayısı ($\rho$):** Değerlerin kendisi yerine sıralamalarını (ranks) temel alarak **monotonik (artarken sürekli artan veya sürekli azalan) ilişkileri** ölçer:
  $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
  *(Burada $d_i = \text{rank}(x_i) - \text{rank}(y_i)$)*

- **💡 Endüstriyel Analiz İpucu:**
  Eğer bir değişken çiftinde **Pearson düşük ($r = 0.35$)** fakat **Spearman çok yüksekse ($\rho = 0.92$)**, bu durum değişkenler arasında doğrusal olmayan (örneğin logaritmik, karesel veya üstel) son derece güçlü bir fiziksel kural olduğunu kanıtlar!

---

#### B. Çoklu Doğrusallık ve Varyans Şişme Faktörü (VIF)
VIF; bir bağımsız değişkenin ($X_i$), diğer tüm bağımsız değişkenler tarafından ne kadar açıklandığını En Küçük Kareler (OLS) regresyonunun $R_i^2$ skoru ile ölçer:

$$VIF_i = \frac{1}{1 - R_i^2}$$

- **$VIF = 1$:** Değişken diğerleriyle tamamen ilişkisizdir (mükemmel ortogonalite).
- **$VIF < 5$:** Düşük riskli, model için güvenli öznitelik.
- **$5 \le VIF \le 10$:** Orta seviye doğrusallık (takip edilmeli).
- **$VIF > 10$:** **Kritik Çoklu Doğrusallık.** O değişken aslında diğer değişkenlerin bir kopyası veya doğrusal bir kombinasyonudur. Model eğitilmeden önce **kesinlikle elenmelidir!**

---

### 3. Dikkat Edilmesi Gereken Tuzaklar

1. **Anscombe Dörtlüsü (Anscombe's Quartet):**
   Aynı ortalamaya, aynı varyansa ve aynı korelasyon katsayısına ($r = 0.816$) sahip 4 farklı veri kümesi çizildiğinde; birinin doğrusal, birinin parabol, birinin dikey aykırı değerli olduğu görülür. **Sadece sayısal korelasyona güvenilemez; mutlaka grafik (Scatter Plot) çizilmelidir.**
2. **Korelasyon $\ne$ Nedensellik (Correlation $\ne$ Causation):**
   Yüksek korelasyon iki değişken arasında sebep-sonuç ilişkisi olduğunu kanıtlamaz; ikisini birden etkileyen gizli bir üçüncü faktör (confounding variable) olabilir.
3. **Simpson Paradoksu:**
   Veri alt gruplara ayrıldığında her grupta pozitif olan bir eğilim, tüm veri birleştirildiğinde negatif bir eğilime dönüşebilir. Bu sebeple kategorik ayrıştırma şarttır.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
                     Öznitelik Veri Çerçevesi (pd.DataFrame)
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │   KesifciVeriAnalizoru  │
                           └────────────┬────────────┘
                                        │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
[Pearson / Spearman]     [VIF Analizi]          [Hedef Değişken]
Doğrusal vs. Monotonik   Çoklu Doğrusallık      Öznitelik Korelasyonları
Korelasyon Matrisleri    (1 / (1 - R^2))        Kategori Dağılımları
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                   ┌─────────────────────────┐
                   │    EdaGrafikUreteci     │
                   │   (Headless Matplotlib) │
                   └────────────┬────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
[korelasyon_isi_        [ozellik_               [dokuma_hizi_vs_kusur_
 haritasi.png]           dagilimlari.png]        sacilim.png]
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

Proje 3 temel python modülünden oluşur:
1. **[`src/kesifci_analizor.py`](./src/kesifci_analizor.py):**
   - `korelasyon_analizi()`: Pearson ve Spearman matrislerini hesaplar, $|r| \ge \text{esik}$ çiftleri tespit eder.
   - `vif_analizi()`: Harici kütüphaneye muhtaç olmadan saf lineer cebir (`np.linalg.lstsq`) ile her sayısal sütun için $VIF = \frac{1}{1 - R^2}$ değerini çıkarır.
   - `hedef_iliskisi_analizi()`: Bağımlı değişken ile sayısal özniteliklerin korelasyonunu ve kategorik kırılımları üretir.
2. **[`src/grafik_ureteci.py`](./src/grafik_ureteci.py):**
   - `matplotlib.use('Agg')` ile GUI penceresi açmadan sunucularda ve CI/CD hatlarında arka planda yüksek çözünürlüklü grafikler üretir.
   - Korelasyon Isı Haritası, Histogram Tabloları ve Saçılım Grafikleri (Scatter Plot) oluşturup diske yazar.
3. **[`ana_akis.py`](./ana_akis.py):**
   - Tüm analizi gerçekçi endüstriyel dokuma metaverisi üzerinde baştan sona çalıştıran konsol yürütücüsü.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
==========================================================================
>>> AŞAMA 1: Korelasyon Matrisi ve Kritik Çiftlerin Tespiti
==========================================================================
Pearson Korelasyon Matrisi:
                    iplik_sikligi  ...  kusurlu_alan_mm2
iplik_sikligi               1.000  ...            -0.217
hali_agirligi               0.912  ...            -0.176
dugum_sayisi                1.000  ...            -0.218
dokuma_hizi                -0.035  ...             0.733
ortalama_parlaklik          0.010  ...            -0.069
kusurlu_alan_mm2           -0.217  ...             1.000

[!] Eşik Değerini Aşan Yüksek Korelasyonlu Çiftler (|r| >= 0.70):
    * iplik_sikligi   <---> hali_agirligi   : r =   0.91 [Doğrusal ve Monotonik Uyumlu]
    * iplik_sikligi   <---> dugum_sayisi    : r =   1.00 [Doğrusal ve Monotonik Uyumlu]
    * hali_agirligi   <---> dugum_sayisi    : r =   0.91 [Doğrusal ve Monotonik Uyumlu]
    * dokuma_hizi     <---> kusurlu_alan_mm2 : r =   0.73 [Doğrusal ve Monotonik Uyumlu]

==========================================================================
>>> AŞAMA 2: Çoklu Doğrusallık (Multicollinearity) ve VIF Analizi
==========================================================================
Öznitelik Adı          | VIF Skoru    | Risk Değerlendirmesi
--------------------------------------------------------------------------
iplik_sikligi          | 1456.17      | Kritik Çoklu Doğrusallık (Modelden Çıkarılmalı!)
hali_agirligi          | 5.99         | Orta Seviye Doğrusallık (Takip Edilmeli)
dugum_sayisi           | 1452.63      | Kritik Çoklu Doğrusallık (Modelden Çıkarılmalı!)
dokuma_hizi            | 2.24         | Düşük Risk (Güvenli)
ortalama_parlaklik     | 1.01         | Düşük Risk (Güvenli)
kusurlu_alan_mm2       | 2.35         | Düşük Risk (Güvenli)
--------------------------------------------------------------------------
>>> Kural: VIF > 10 olan sütunlar (ör. dugum_sayisi veya iplik_sikligi) model
    eğitilmeden önce elenmelidir; aksi halde model katsayıları kararsızlaşır.

==========================================================================
>>> AŞAMA 3: Hedef Değişken (Kusurlu Alan) İlişki Analizi
==========================================================================
Sayısal Değişkenlerin Kusur Alanı ile Korelasyonu:
    * iplik_sikligi       : -0.217
    * hali_agirligi       : -0.176
    * dugum_sayisi        : -0.218
    * dokuma_hizi         :  0.733  <-- En güçlü pozitif belirteç!
    * ortalama_parlaklik  : -0.069

Kumaş Tipine Göre Ortalama Kusurlu Alan Dağılımı:
    - kumas_tipi [Akrilik ]: 5.48 mm2
    - kumas_tipi [Ipek    ]: 5.69 mm2
    - kumas_tipi [Yun     ]: 5.47 mm2

==========================================================================
>>> AŞAMA 4: Grafiksel Görselleştirmelerin Diske Kaydedilmesi
==========================================================================
[V] 1. Isı Haritası Kaydedildi : korelasyon_isi_haritasi.png
[V] 2. Histogramlar Kaydedildi : ozellik_dagilimlari.png
[V] 3. Saçılım Grafiği Kaydedildi: dokuma_hizi_vs_kusur_sacilim.png
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Spearman vs. Pearson Ayrışma Dedektörü (Doğrusal Olmayan İlişki Avcısı)**

Endüstride veri analistlerinin en çok yanıldığı durumlardan biri, bir değişkenin diğeriyle ilişkisinin çok güçlü olmasına rağmen ilişkinin doğrusal olmamasından ötürü (ör. $y = x^3$ veya $y = e^x$) Pearson korelasyonunun düşük çıkmasıdır.

### Görev Tanımı:
[`src/kesifci_analizor.py`](./src/kesifci_analizor.py) içerisine şu fonksiyonu eklemeni bekliyorum:

```python
def dogrusal_olmayan_iliskileri_bul(
    self,
    fark_esigi: float = 0.25
) -> List[Tuple[str, str, float, float]]:
```

### Beklenen Kurallar:
1. Her sayısal sütun çifti için $|r_{\text{spearman}}| - |r_{\text{pearson}}|$ farkını hesaplamalıdır.
2. Eğer Spearman korelasyonunun mutlak değeri Pearson'dan `fark_esigi` kadar büyükse (örneğin $\rho = 0.90$ iken $r = 0.60$), bu çifti listeye eklemelidir.
3. Çıktı olarak `(sutun_1, sutun_2, pearson, spearman)` demetleri listesi döndürmelidir.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> Bir veri kümesinde iki özniteliğin VIF skoru $1000+$ çıkmışsa (kritik çoklu doğrusallık), bu iki öznitelikten birini modelden çıkarmadan doğrudan bir **Doğrusal Regresyon (Linear Regression)** veya **Lojistik Regresyon** eğitirsek katsayıların (weights/coefficients) işaretleri ve standart hataları (standard errors) üzerinde tam olarak ne olur?  
> Bu durum **Karar Ağaçlarını (Random Forest, XGBoost)** da aynı şekilde etkiler mi, neden?

---

## 📂 Dizin Yapısı

```
day-06-eda-lab/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (pandas, numpy, scipy, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve grafik üretim betiği
├── ciktilar/                   # Üretilen PNG grafik dosyaları
│   ├── korelasyon_isi_haritasi.png
│   ├── ozellik_dagilimlari.png
│   └── dokuma_hizi_vs_kusur_sacilim.png
├── src/
│   ├── __init__.py
│   ├── kesifci_analizor.py     # KesifciVeriAnalizoru ve VIF motoru
│   └── grafik_ureteci.py       # Headless Matplotlib çizim araçları
└── testler/
    └── test_eda.py             # 5 adet pytest birim testi (5 passed)
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
python -m pytest testler/test_eda.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
