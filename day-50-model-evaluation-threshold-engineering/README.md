# Day 50: Model Değerlendirme & Eşik Değeri Mühendisliği (Model Evaluation & Threshold Engineering)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 50. gününde geliştirilen **Olasılık Kalibrasyonu, $F_\beta$ Optimizasyonu, Maliyet-Fayda Karar Matrisi ve Eşik Değeri Mühendisliği (Threshold Engineering) Motorudur**. Ham model çıktı olasılıklarının kalibrasyonunu Brier Skoru ve ECE ile denetler, sabit $0.50$ eşiği yerine işletmenin asimetrik finansal kâr/zarar dengesini maksimize eden optimal karar eşiklerini ($\tau^*$) belirler.

---

## 📖 Mentorluk Dersi ve Eşik Mühendisliği Teorisı

### 1. Neden Olasılık $\ne$ Karardır? (Eşik Mühendisliğinin Önemi)

Yapay zeka modelleri (Lojistik Regresyon, XGBoost, Derin Sinir Ağları) bir girdi için $[0.0, 1.0]$ arasında olasılık ($\hat{p}$) üretir. Ancak bu olasılığı ikili bir eyleme (Onay / Red, Alarm / Temiz, Tedavi / İzlem) dönüştürmek bir **Karar Mühendisliği** sürecidir.

Kritik 3 Adım:

1. **Olasılık Kalibrasyonu ve Brier Skoru:**
   - Model $\%80$ risk diyorsa, bu gruptaki her 100 işlemden tam 80'i gerçekten riskli mi?
   - **Brier Skoru:** Tahmin edilen olasılık ile gerçekleşen olay arasındaki ortalama karesel hata:
     $$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$
   - **Expected Calibration Error (ECE):** Güvenilirlik eğrisi üzerinde modelin aşırı özgüven (overconfidence) sapmasını ölçer.

2. **$F_\beta$-Score Optimizasyonu:**
   - **$\beta = 0.5$ (Precision Odaklı):** Yanlış alarm maliyeti yüksekse (müşteri sürtünmesi).
   - **$\beta = 1.0$ (Dengeli):** Klasik harmonik ortalama.
   - **$\beta = 2.0$ (Recall Odaklı):** Kaçırılan risk maliyeti yüksekse (sağlık, siber saldırı).

3. **Maliyet-Fayda Karar Matrisi (Net Financial Utility):**
   - Her hücrenin parasal değeri farklıdır:
     - $B_{\text{TP}}$: Yakalanan risk kârı ($+3000\$$)
     - $B_{\text{TN}}$: Sorunsuz işlem kârı ($+20\$$)
     - $C_{\text{FP}}$: Yanlış alarm inceleme masrafı ($-100\$$)
     - $C_{\text{FN}}$: Kaçırılan dolandırıcılık hasarı ($-4500\$$)
   - **Net Kazanç:** $\text{Net Kazanç}(\tau) = \text{TP}(\tau) B_{\text{TP}} + \text{TN}(\tau) B_{\text{TN}} - \text{FP}(\tau) C_{\text{FP}} - \text{FN}(\tau) C_{\text{FN}}$

```
                           ┌──────────────────────────────────────────────────────────┐
                           │          HAM MODEL ÇIKTILARI (Olasılık Tahminleri)       │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      OlasilikKalibratoru (Brier Skoru & İzotonik Kalibrasyon)                     │
    │  - Güvenilirlik Eğrisi (Reliability Diagram) ve ECE Hatası Hesaplanır                             │
    │  - İzotonik Dönüşüm ile Kalibre Olasılık Dağılımı Elde Edilir                                     │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      EsikDegeriMuhendisi (F-Beta & Maliyet-Fayda Karar Motoru)                    │
    │  - F0.5, F1, F2 Eğrileri ve Zirve Noktaları Belirlenir                                            │
    │  - Finansal Net Kazanç Eğrisi Taranır -> Maksimum Net Kâr Sağlayan Eşik (tau*) Seçilir             │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 6-PANELLİ EŞİK DEĞERİ VE MALİYET TEŞHİS PANELİ (Day 50)                           │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Matematiksel Formülasyonlar

#### A. Expected Calibration Error (ECE)
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

#### B. $F_\beta$ Skoru
$$F_\beta = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

#### C. Decision Curve Analysis (Net Benefit)
$$\text{Net Benefit}(\tau) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left( \frac{\tau}{1 - \tau} \right)$$

---

## 🛠️ Dizin Yapısı

```
day-50-model-evaluation-threshold-engineering/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # scikit-learn, pandas, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca kalibrasyon, eşik optimizasyonu ve maliyet analizi betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── kalibrasyon_motoru.py        # OlasilikKalibratoru (Brier Skoru, ECE & İzotonik Kalibratör)
│   ├── esik_muhendisi.py            # EsikDegeriMuhendisi (F-Beta, Net Kazanç & DCA Analizi)
│   └── gorsellestirici.py           # 6-Panelli Teşhis Panosu (Threshold Engineering Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_threshold_engineering.py # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── esik_muhendisligi_paneli.png # 6 panelli yüksek çözünürlüklü teşhis panosu
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

## 📊 Eşik Değerleri ve Finansal Getiri Karşılaştırma Tablosu

| Strateji | Seçilen Karar Eşiği ($\tau^*$) | Recall ($\%$) | Precision ($\%$) | Net Finansal Kazanç ($) |
|---|---|---|---|---|
| **Varsayılan Sabit Eşik** | $\tau = 0.500$ | $\%78.5$ | $\%82.1$ | $\$194,500.00$ |
| **Dengeli $F_1$ Eşiği** | $\tau = 0.446$ | $\%83.2$ | $\%79.4$ | $\$208,200.00$ |
| **Recall Öncelikli $F_2$** | $\tau = 0.287$ | $\%94.8$ | $\%65.1$ | $\$226,400.00$ |
| **Finansal Optimize Eşik** | $\mathbf{\tau = 0.218}$ | $\mathbf{\%97.5}$ | $\%58.9$ | $\mathbf{\$238,700.00}$ (Zirve Kâr) |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Karar Eğrisi Analizi (DCA) çıktısını kullanarak modelin "Herkesi Tedavi Et / İncele" ve "Kimseyi İnceleme" stratejilerine kıyasla hangi risk aralıklarında net klinik/finansal fayda sağladığını raporlayan bir **"DCA Policy Decision Engine"** geliştirmek.

**Tamamlanan Çözüm:**
```python
def dca_politika_karsilastir(esikler, net_benefit_model, prevalans: float) -> dict:
    """Modelin müdahale aralığını ve referans stratejilere üstünlüğünü denetler."""
    ustun_aralik = []
    for th, nb in zip(esikler, net_benefit_model):
        nb_all = prevalans - (1 - prevalans) * (th / max(1 - th, 1e-6))
        if nb > max(0.0, nb_all):
            ustun_aralik.append(round(th, 2))
    return {
        "model_kullanilabilir_aralik": f"[{min(ustun_aralik):.2f}, {max(ustun_aralik):.2f}]" if ustun_aralik else "YOK",
        "maksimum_net_fayda": round(float(np.max(net_benefit_model)), 4)
    }
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Bir dolandırıcılık veya medikal teşhis yapay zeka modelinde, kaçırılan bir pozitif vakanın maliyeti ($C_{\text{FN}} = 4500\$$), yanlış alarm maliyetinden ($C_{\text{FP}} = 100\$$) 45 kat büyük olduğunda, neden standart $\tau = 0.50$ eşiği yerine **$\tau \approx 0.20$ gibi düşük bir karar eşiği** seçilmelidir?

> **Mentor Cevabı:**
> 1. **Asimetrik Risk Dengesi:** Karar teorisinde Bayes optimal eşiği $\tau^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}$ olarak türetilir. Verilen maliyetlerle $\tau^* = \frac{100}{100 + 4500} \approx 0.0217$ olur. Yani model bir işlemin dolandırıcılık olma olasılığını $\%3$ bile görse, onu incelemeye almak incelememekten daha kârlıdır.
> 2. **Finansal Net Fayda:** Eşik $0.50$'den $0.22$'ye düşürüldüğünde yanlış alarmlar (FP) bir miktar artar ancak kaçırılan devasa dolandırıcılık zararları (FN) neredeyse sıfırlanır. Sonuç olarak toplam net finansal kâr maksimum noktaya ulaşır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
