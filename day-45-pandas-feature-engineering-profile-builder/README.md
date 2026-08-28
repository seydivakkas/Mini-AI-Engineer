# Day 45: Özellik Mühendisliği, Encoding, Ölçeklendirme ve Feature Store Profil Oluşturucu (Pandas Feature Engineering Profile Builder)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** yol haritamızın 45. gününde geliştirilen **Üretim Seviyesi Özellik Mühendisliği (Feature Engineering), Kategorik Kodlama, Sayısal Ölçeklendirme ve Feature Store Profil Oluşturucu Motorudur**. Ham tabüler verileri yüksek kaliteli özellik vektörlerine dönüştürür, veri sızıntısını (data leakage) engelleyen durum parametrelerini kaydeder ve modern MLOps platformları (Feast / Hopsworks) için **Feature Store Metadata Kataloğu** üretir.

---

## 📖 Mentorluk Dersi ve Özellik Mühendisliği Teorisı

### 1. Üretimde Özellik Mühendisliği ve Feature Store İhtiyacı

Makine öğrenimi modellerinin başarısı doğrudan girdi özelliklerinin bilgi yoğunluğuna ve temsil gücüne bağlıdır. Ham verilerin modele beslenmeden önce geçmesi gereken 4 kritik dönüşüm aşaması:

1. **Kategorik Kodlama (Categorical Encoding):**
   - **One-Hot Encoding (OHE):** Düşük kardinaliteli kategoriler ($k \le 10$) için ikili (binary) sütunlar.
   - **Frequency Encoding:** Her kategoriyi veri setindeki görülme sıklığına ($P(c) = \text{count}(c)/N$) eşler.
   - **Smoothed Target Encoding (Empirical Bayes):** Yüksek kardinaliteli kategorilerde hedef değişkenin ortalamasını, seyrek kategorilerdeki aşırı öğrenmeyi (overfitting) engelleyen bir yumuşatma ağırlığı ($m$) ile harmanlar.

2. **Sayısal Dönüşümler & Ölçeklendirme:**
   - **StandardScaler (Z-Score):** Ortalamayı 0, standart sapmayı 1 yapar: $z = (x - \mu) / \sigma$.
   - **RobustScaler (Medyan & IQR):** Aşırı uç değerlere (outliers) karşı dayanıklı normalizasyon: $z = (x - Q_2) / (Q_3 - Q_1)$.
   - **Log1p Dönüşümü:** Gelir, işlem hacmi gibi aşırı sağa çarpık (right-skewed) verileri Gauss çan eğrisine yaklaştırır: $\ln(1 + x)$.

3. **Domain Etkileşim ve Oran Özellikleri (Interaction Features):**
   - İki değişken arasındaki fiziksel veya finansal oranlar (örn: $\text{Borç} / \text{Gelir}$ oranı).

4. **Feature Store Metadata & Profilleme (Feast/Hopsworks):**
   - Üretim hattında çevrimdışı (offline training) ve çevrimiçi (online low-latency inference) özellik tanımlarının tutarlı olmasını sağlayan metadata sözleşmesi.

```
                           ┌──────────────────────────────────────────────────────────┐
                           │            HAM TABÜLER VERİ GİRDİSİ (DataFrame)          │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      ÖZELLİK MÜHENDİSLİĞİ VE DÖNÜŞÜM BORU HATTI                                   │
    │  - Kategorik: One-Hot, Frequency Encoding, Smoothed Target Encoding (m=15)                        │
    │  - Sayısal  : Log1p Transformation, StandardScaler (Z-Score), RobustScaler (IQR)                  │
    │  - Domain   : Borç/Gelir Oranı, Çapraz Etkileşim Terimleri                                        │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      FeatureStoreProfilci (Metadata Kataloğu & Feast Şeması)                      │
    │  - İstatistikler: Min, Max, Ortalama, Medyan, Çarpıklık (Skewness), Hedef Korelasyonu             │
    │  - Feature Registry: Feast-Uyumlu Feature View Tanımı & Tip Güvenliği                             │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 ÜRETİME HAZIR TÜRETİLMİŞ ÖZELLİK MATRİSİ (+6 YENİ ÖZNİTELİK)                      │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Matematiksel Formülasyonlar

#### A. Düzeltilmiş Hedef Kodlama (Smoothed Target Encoding)
$$S(c) = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n_c + m}$$
- $n_c$: $c$ kategorisinin örneklem sayısı
- $\bar{y}_c$: $c$ kategorisindeki hedef değişken ortalaması
- $\bar{y}_{\text{global}}$: Tüm veri setinin genel hedef ortalaması
- $m$: Düzeltme katsayısı (Smoothing weight, örn: $m=15$)

#### B. Dayanıklı Ölçekleyici (Robust Scaler)
$$z_i = \frac{x_i - \text{Median}(X)}{\text{IQR}(X)} = \frac{x_i - Q_{50}}{Q_{75} - Q_{25}}$$

---

## 🛠️ Dizin Yapısı

```
day-45-pandas-feature-engineering-profile-builder/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # pandas, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca özellik mühendisliği ve Feast profil akışı
├── README.md                        # 220+ satır teorik ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── kodlayicilar.py              # KategorikKodlayici (Target, Frequency, OHE)
│   ├── olcekleyiciler.py            # SayisalOlcekleyici (Standard, Robust, Log1p, Oranlar)
│   ├── ozellik_profili.py           # FeatureStoreProfilci (Metadata, İstatistik, Feast Şeması)
│   └── gorsellestirici.py           # 6 panelli teşhis panosu (Feature Engineering Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_feature_engineer.py     # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── ozellik_muhendisligi_paneli.png # 6 panelli yüksek çözünürlüklü teşhis panosu
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

## 📊 Türetilen Özellikler ve Korelasyon Tablosu

| Özellik Adı | Dönüşüm Türü | Min / Max | Çarpıklık (Skew) | Hedef Korelasyonu |
|---|---|---|---|---|
| `gelir_log1p` | Log1p Normalizasyonu | $7.85$ / $12.10$ | **$0.08$ (Simetrik)** | $-0.245$ |
| `meslek_grubu_target_enc` | Smoothed Target Enc | $0.21$ / $0.48$ | $-0.12$ | **$+0.412$ (Güçlü Sinyal)** |
| `borc_gelir_orani` | Etkileşim Oranı | $0.15$ / $6.80$ | $+1.14$ | **$+0.534$ (En Yüksek)** |
| `yas_std_scaled` | StandardScaler | $-1.85$ / $+1.92$ | $-0.02$ | $+0.118$ |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/ozellik_profili.py` içerisine Feast Feature Store için doğrudan çalıştırılabilir Python `FeatureView` kodunu üreten bir **"Feast FeatureView Code Generator"** fonksiyonu eklemek.

**Tamamlanan Çözüm:**
```python
def feast_python_tanimi_uret(feature_view_name: str, entities: list, features: list) -> str:
    lines = [
        "from feast import Entity, FeatureView, Field, FileSource, ValueType",
        "from feast.types import Float32, Int64, String",
        "",
        f'{entities[0]} = Entity(name="{entities[0]}", value_type=ValueType.INT64)',
        "",
        f"{feature_view_name} = FeatureView(",
        f'    name="{feature_view_name}",',
        f"    entities=[{entities[0]}],",
        "    schema=["
    ]
    for feat in features:
        lines.append(f'        Field(name="{feat}", dtype=Float32),')
    lines.append("    ],")
    lines.append('    source=FileSource(path="data/features.parquet")')
    lines.append(")")
    return "\n".join(lines)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden ham kategorik ortalamalarla Target Encoding yapmak **hedef veri sızıntısına (target leakage)** ve aşırı öğrenmeye (overfitting) sebep olur ve bu sorunu önlemek için neden **Smoothing ($m$)** ve **K-Fold Out-of-Fold (OOF)** stratejisi zorunludur?

> **Mentor Cevabı:**
> 1. **Seyrek Kategori Tuzağı (Rare Category Overfitting):** Eğer bir veri setinde sadece 1 kez geçen `meslek = 'ASTROFİZİKÇİ'` kategorisi varsa ve bu kişinin hedef değeri $y=1$ ise, ham target encoding bu kategoriye $1.0$ atar. Model bu özelliğe aşırı güvenir ancak test verisinde aynı meslekteki başka bir kişi $y=0$ olduğunda model çöker.
> 2. **Smoothing ve OOF Korunması:** Smoothed Target Encoding ($S(c) = \frac{n_c \bar{y}_c + m \bar{y}_{\text{global}}}{n_c + m}$), örneklem sayısı $n_c$ az olduğunda skoru genel ortalamaya ($\bar{y}_{\text{global}}$) çekerek varyansı düşürür. K-Fold Out-of-Fold uygulandığında ise her satırın hedef değeri kendi hesaplamasından çıkarılarak modelin kendi etiketini ezberlemesi (data leakage) kesin olarak engellenir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
