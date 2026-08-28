# Day 49: XGBoost ile Dengesiz Tabüler Risk Sınıflandırıcısı (XGBoost Tabular Risk Classifier)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg?style=flat-square&logo=xgboost)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 49. gününde geliştirilen **Aşırı Dengesiz Tabüler Verilerde XGBoost Risk ve Dolandırıcılık (Fraud) Sınıflandırıcısı ve TreeSHAP Açıklayıcısıdır**. Pozitif sınıf oranının $\%5$ veya daha düşük olduğu kritik endüstriyel/finansal risk görevlerinde `scale_pos_weight`, PR-AUC odaklı Erken Durdurma (Early Stopping), Karar Eşiği Mühendisliği (Threshold Optimization) ve yerel TreeSHAP özellik katkısı hesaplaması sunar.

---

## 📖 Mentorluk Dersi ve Dengesiz Veri Sınıflandırma Teorisı

### 1. Dengesiz Tabüler Verilerde Doğruluk (Accuracy) Tuzağı

Gerçek dünya dolandırıcılık, anomali ve donanım arızası tespitinde pozitif sınıf sıklıkla veri setinin yalnızca $\%1 - \%5$'ini oluşturur ($N_{\text{neg}} \gg N_{\text{pos}}$). Standart bir model her örneğe "Normal" (0) tahmini yaparak $\%95$ doğruluk (accuracy) elde edebilir; ancak asıl hedef olan riskli vakaların $\%100$'ünü kaçırır.

Bu problemi çözmek için 4 kritik mekanizma uygulanır:

1. **Maliyet Duyarlı Ağırlıklandırma (`scale_pos_weight`):**
   - XGBoost kayıp fonksiyonunda pozitif sınıfın gradyan ve hessian ağırlığını artırır:
     $$\text{scale\_pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$
   - Modelin azınlık sınıfını yanlış tahmin etmesine (False Negative) yüksek ceza uygular.

2. **Validation PR-AUC Tabanlı Erken Durdurma (Early Stopping):**
   - Dengesiz veri setlerinde ROC-AUC yerine **Precision-Recall (PR) AUC** metriği takip edilir.
   - Doğrulama PR-AUC skoru $P$ iterasyon boyunca iyileşmediğinde eğitim durdurularak aşırı öğrenme (overfitting) engellenir.

3. **Karar Eşiği Mühendisliği (Threshold Tuning):**
   - Standart $0.50$ olasılık eşiği yerine, $F_1$-skorunu veya iş biriminin maliyet matrisini ($C_{\text{FN}} \gg C_{\text{FP}}$) optimize eden optimal eşik $\tau^*$ seçilir.

4. **TreeSHAP ile Açıklanabilir Yapay Zeka (XAI):**
   - XGBoost'un yerel `pred_contribs=True` C++ motoruyla her özelliğin risk tahminine yaptığı Shapley katkısı hesaplanır:
     $$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{D} \phi_j(\mathbf{x})$$

```
                           ┌──────────────────────────────────────────────────────────┐
                           │          DENGESİZ RİSK VERİ SETİ (%5 Pozitif Sınıf)       │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      XGBoostRiskSiniflandirici (Maliyet Duyarlı Eğitim)                           │
    │  - scale_pos_weight = N_neg / N_pos (Otomatik Dengeleme: ~19x Ağırlık)                           │
    │  - eval_metric = ['logloss', 'aucpr'] & early_stopping_rounds = 20                                │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      EŞİK DEĞERİ OPTİMİZASYONU & TREESHAP AÇIKLAMA MOTORU                         │
    │  - Validation F1-Score Maksimizasyonu ile Optimal Karar Eşiği (tau*) Belirlenir                   │
    │  - Yerel TreeSHAP ile Global Özellik Önemi (Mean |SHAP|) Hesaplanır                               │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 6-PANELLİ RİSK TEŞHİS VE SHAP AÇIKLANABİLİRLİK PANELİ (Day 49)                    │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Gradient Boosted Decision Trees ile tabüler verilerde eksik değerlere dayanıklı, yüksek performanslı ve açıklanabilir risk sınıflandırması yapmak için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Tabüler verilerde derin öğrenme modellerinden çok daha hızlı yakınsar ve Feature Importance ile hangi değişkenin riski artırdığını açıklar.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Görsel veya serbest metin gibi yapısal olmayan (unstructured) verilerde doğrudan kullanılamaz.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- LightGBM, CatBoost, Random Forest veya TabNet.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Gradyan Destekli Ağaçlar** | *Gradient Boosted Decision Trees (GBDT)* | Her yeni karar ağacının önceki ağaçların yaptığı hataların (gradyan ve hessian) üzerine eğitildiği topluluk yöntemi. |
| **Hessian & Gradyan** | *First & Second Order Gradients* | XGBoost'un kayıp fonksiyonunun Taylor açılımındaki 1. (gradyan) ve 2. (hessian) türevlerini kullanarak optimum yaprak ağırlıklarını hesaplaması. |
| **Ölçek Pozitif Ağırlığı** | *`scale_pos_weight` Parameter* | Dengesiz tablosal veri setlerinde azınlık sınıfının kayıp fonksiyonundaki ağırlığını artırarak yakalama oranını yükseltme. |
| **Erken Durdurma (Early Stopping)** | *Early Stopping Rounds* | Doğrulama kaybı belirli bir adım boyunca iyileşmediğinde eğitimi durdurup aşırı öğrenmeyi önleme. |

---

## 2. Matematiksel Formülasyonlar

#### A. Maliyet Duyarlı İkili Lojistik Kayıp (Cost-Sensitive Loss)
$$\mathcal{L}(y, \hat{p}) = - \left[ w \cdot y \ln(\hat{p}) + (1 - y) \ln(1 - \hat{p}) \right], \quad w = \text{scale\_pos\_weight}$$

#### B. TreeSHAP Shapley Değer Katkısı
$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$

---

## 🛠️ Dizin Yapısı

```
day-49-xgboost-tabular-risk-classifier/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # xgboost, scikit-learn, pandas, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca XGBoost eğitim, eşik optimizasyonu ve SHAP betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── risk_veri_ureteci.py         # RiskVeriSimulasyonu (%5 Dengesiz Risk Veri Seti Üreticisi)
│   ├── xgboost_risk_egitici.py      # XGBoostRiskSiniflandirici (scale_pos_weight & TreeSHAP)
│   └── gorsellestirici.py           # 6-Panelli Teşhis Panosu (Risk Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_xgboost_risk.py         # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── xgboost_risk_paneli.png      # 6 panelli yüksek çözünürlüklü teşhis panosu
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

## 📊 Test Performansı ve TreeSHAP Öznitelik Katkıları

| Sıra | Risk Özniteliği | Mean \|SHAP\| Katkısı | Risk Yönü |
|---|---|---|---|
| **1** | `cihaz_degisim_orani` | **$1.1420$** | ⬆️ Yüksek Değişim $\to$ Yüksek Risk |
| **2** | `basarisiz_giris_sayisi` | **$0.8240$** | ⬆️ Tekrarlı Hata $\to$ Şüpheli Girişim |
| **3** | `islem_tutari` | **$0.7150$** | ⬆️ Sıradışı Yüksek Tutar $\to$ Anomali |
| **4** | `gece_islemi_sayisi` | **$0.5310$** | ⬆️ Gece İşlemleri $\to$ Ekstra Risk |
| **5** | `hesap_yasi_gun` | **$0.4890$** | ⬇️ Eski Güvenilir Hesap $\to$ Düşük Risk |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Kaçırılan her riskli işlemin maliyetinin ($C_{\text{FN}} = 1000 \$ $), yanlış alarm inceleme maliyetinden ($C_{\text{FP}} = 20 \$ $) çok daha yüksek olduğu senaryolar için bir **"Cost-Benefit Weighted Threshold Optimizer"** geliştirmek.

**Tamamlanan Çözüm:**
```python
def maliyet_odakli_esik_bul(y_true, y_prob, cost_fn: float = 1000.0, cost_fp: float = 20.0) -> float:
    """Toplam finansal zararı minimize eden karar eşiğini hesaplar."""
    esikler = np.linspace(0.05, 0.95, 91)
    min_maliyet = float("inf")
    en_iyi_esik = 0.50

    for th in esikler:
        pred = (y_prob >= th).astype(int)
        fn = np.sum((y_true == 1) & (pred == 0))
        fp = np.sum((y_true == 0) & (pred == 1))
        toplam_maliyet = (fn * cost_fn) + (fp * cost_fp)

        if toplam_maliyet < min_maliyet:
            min_maliyet = toplam_maliyet
            en_iyi_esik = th

    return float(round(en_iyi_esik, 3))
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Dengesiz veri setlerinde XGBoost modelinde `scale_pos_weight` parametresi kullanıldığında modelin ürettiği ham olasılık değerleri ($\hat{p}$) neden yukarıya doğru kayar (kalibrasyonu bozulur) ve bu durumda neden **Platt Scaling (Isotonic Regression)** veya **Eşik Değeri Mühendisliği (Threshold Tuning)** uygulanmalıdır?

> **Mentor Cevabı:**
> 1. **Olasılık Kayması (Probability Shift):** `scale_pos_weight = 19.0` verildiğinde, kayıp fonksiyonu pozitif sınıfın gradyanlarını 19 kat büyütür. Model matematiksel olarak pozitif sınıfın apriori oranını $\%5$ yerine $\%50$ gibi algılar. Sonuç olarak tahmin edilen ham olasılıklar gerçek olasılıkların üzerinde çıkar.
> 2. **Eşik ve Kalibrasyon Çözümü:** Bu kayma modelin sıralama (ranking) kabiliyetini ve ROC-AUC / PR-AUC skorunu bozmaz. Ancak sabit $0.50$ eşiği yerine **Validation F1 Maksimizasyonu** ile optimize edilen dinamik eşik ($\tau^*$) kullanıldığında veya Isotonic Regression ile olasılıklar yeniden kalibre edildiğinde model mükemmel bir üretim kararlılığına kavuşur.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
