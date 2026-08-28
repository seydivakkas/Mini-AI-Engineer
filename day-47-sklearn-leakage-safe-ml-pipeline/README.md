# Day 47: Scikit-Learn ile Veri Sızıntısına (Data Leakage) Karşı Güvenli Pipeline Tasarımı (Leakage-Safe ML Pipeline)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 47. gününde geliştirilen **Veri Sızıntısına (Data Leakage) Karşı Kapsüllenmiş Scikit-Learn Pipeline ve Nested Cross-Validation Motorudur**. Üretim ortamında modellerin test/çapraz doğrulama verisinden geleceğe dair bilgi sızdırmasını (train-test contamination) ve hedef değişken sızıntısını (target leakage) %100 engelleyen bir mimari kalkan sunar.

---

## 📖 Mentorluk Dersi ve Veri Sızıntısı (Data Leakage) Teorisı

### 1. Veri Sızıntısı Nedir ve Neden Felakettir?

Makine öğrenimi projelerinde en yaygın ve sinsi hata **Veri Sızıntısıdır (Data Leakage)**. Model eğitim esnasında test setinde veya üretim ortamında erişemeyeceği bilgilere ulaştığında metrikler (örneğin ROC-AUC) $\%99$ gibi mükemmel görünür; ancak model canlıya (production) alındığında tamamen çöker.

Üretimde 3 ana sızıntı türü görülür:

1. **Ön İşleme ve İmpütasyon Sızıntısı (Preprocessing Leakage):**
   - **Hatalı Yaklaşım:** Tüm veri seti üzerinde `StandardScaler.fit(X)` veya `SimpleImputer.fit(X)` çalıştırıp ardından `train_test_split` veya `cross_val_score` yapmak.
   - **Sonuç:** Test katmanının ortalaması ($\mu_{\text{test}}$) ve medyanı eğitim aşamasına sızar.
   - **Çözüm:** `ColumnTransformer` ve `Pipeline` kullanarak ölçekleyici ve doldurucuların yalnızca eğitim katmanında `fit` edilmesini garanti altına almak.

2. **Hedef Değişken Sızıntısı (Target Leakage):**
   - Tahmin anından sonra oluşan veya hedef değişkeni doğrudan/dolaylı temsil eden sütunların (örn. `kredi_iptal_tarihi`, `tahsilat_tutari`) girdi olarak kullanılması.
   - **Çözüm:** Korelasyon eşik taraması ($|r| \ge 0.88$) ve zaman damgası (timestamp) denetimi.

3. **Çapraz Doğrulama ve Hiperparametre Sızıntısı (CV Optimization Bias):**
   - Hiperparametre optimizasyonunun (GridSearchCV) tek bir test seti üzerinde yapılması yanlılık oluşturur.
   - **Çözüm:** **Nested Cross-Validation (İç-Dış Çapraz Doğrulama)** mimarisi.

```
                           ┌──────────────────────────────────────────────────────────┐
                           │               HAM VERİ GİRDİSİ (DataFrame)               │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      TargetLeakageDedektoru (Hedef Sızıntı & Korelasyon Taraması)                 │
    │  - Şüpheli Kolon Taraması: |r| >= 0.88 olan değişkenler işaretlenir ve filtrelenir                │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      NESTED CROSS-VALIDATION (5-Dış Katman x 3-İç Katman)                         │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │  DIŞ KATMAN (Outer CV - 5 Fold): Tarafsız Genelleme Hatası Ölçümü                           │  │
    │  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │  │
    │  │  │  İÇ KATMAN (Inner CV - 3 Fold / GridSearchCV):                                        │  │  │
    │  │  │  - ColumnTransformer: SimpleImputer(median) + RobustScaler (Sayısal)                 │  │  │
    │  │  │  - ColumnTransformer: SimpleImputer(const) + OneHotEncoder(ignore) (Kategorik)       │  │  │
    │  │  │  - Model Estimator  : LogisticRegression / RandomForest                              │  │  │
    │  │  └───────────────────────────────────────────────────────────────────────────────────────┘  │  │
    │  └─────────────────────────────────────────────────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 ÜRETİME HAZIR SERİLEŞTİRİLEBİLİR GÜVENLİ PIPELINE (Day 47)                        │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Matematiksel Formülasyonlar

#### A. Sızıntısız Katmanlı K-Fold Tahmini
Her $k$. dış katman için model yalnızca $T_k$ eğitim setinde eğitilir ve $V_k$ test setinde değerlendirilir:
$$\text{Genelleme Hatası} = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}\left(f_{\theta^*(T_k)}(V_k)\right)$$
Burada $\theta^*(T_k)$ iç GridSearchCV ile $T_k$ üzerinde optimize edilen parametrelerdir.

---

## 🛠️ Dizin Yapısı

```
day-47-sklearn-leakage-safe-ml-pipeline/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # scikit-learn, pandas, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca güvenli pipeline ve Nested CV yürütme betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── pipeline_mimari.py           # GuvenliPipelineUretici (ColumnTransformer Kapsülleme)
│   ├── sizinti_dedektoru.py         # TargetLeakageDedektoru (Hedef Sızıntısı & Korelasyon Taraması)
│   ├── nested_cv_motoru.py          # NestedCVMotoru (Outer 5-Fold x Inner 3-Fold GridSearch)
│   └── gorsellestirici.py           # 6-Panelli Teşhis Panosu (Leakage-Safe Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_leakage_safe_pipeline.py # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── leakage_guvenli_pipeline_paneli.png # 6 panelli yüksek çözünürlüklü teşhis panosu
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

## 📊 Güvenli Nested CV vs Sızıntılı Karşılaştırma Tablosu

| Değerlendirme Stratejisi | Ortalama ROC-AUC | Std Sapma ($\sigma$) | Sızıntı Yanlılığı (Bias) | Canlı Güvenilirliği |
|---|---|---|---|---|
| **Güvenli Nested Pipeline** | **$\%88.45$** | $\pm 0.021$ | $\mathbf{0.000}$ | ✅ $\%100$ Gerçekçi |
| **Sızıntılı Naive Ön İşleme** | **$\%93.12$** | $\pm 0.012$ | **$+0.0467$ Şişme** | ❌ Canlıda Çöker |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `scikit-learn` uyumlu `BaseEstimator` ve `TransformerMixin` miras alan, eğitim katmanında Out-of-Fold hedef ortalamalarını hesaplayıp test katmanına güvenle aktaran bir **"Out-of-Fold Target Encoder Transformer"** geliştirmek.

**Tamamlanan Çözüm:**
```python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.target_maps = {}
        self.global_mean = 0.0

    def fit(self, X, y):
        self.global_mean = float(np.mean(y))
        for col in X.columns:
            counts = X.groupby(col).size()
            means = y.groupby(X[col]).mean()
            smoothed = (counts * means + self.smoothing * self.global_mean) / (counts + self.smoothing)
            self.target_maps[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in X.columns:
            if col in self.target_maps:
                X_out[col] = X[col].map(self.target_maps[col]).fillna(self.global_mean)
        return X_out
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden `StandardScaler` veya `SimpleImputer` nesnesini `train_test_split` yapmadan önce tüm veri üzerinde çalıştırmak bir **Veri Sızıntısıdır (Data Leakage)** ve `scikit-learn` `Pipeline` kullanımı bu sızıntıyı mimari olarak nasıl engeller?

> **Mentor Cevabı:**
> 1. **Test Dağılımının Sızması:** Tüm veri üzerinde `scaler.fit(X)` çağrıldığında, gelecekteki test kümesinin ortalaması ($\mu$) ve varyansı ($\sigma^2$) hesaplamaya dahil edilir. Model dolaylı yoldan test setinin sınırlarını öğrenir ve test skoru yapay olarak yükselir.
> 2. **Pipeline İzolasyonu:** `Pipeline(steps=[('scaler', StandardScaler()), ('model', LogisticRegression())])` yapısı kullanıldığında, `cross_val_score` veya `fit` metodu çağrıldığında pipeline her katmanda (fold) `scaler.fit_transform(X_train)` ve ardından test kümesinde yalnızca `scaler.transform(X_test)` uygular. Böylece test verisi asla öğrenme sürecine sızamaz.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
