# Day 46: Matplotlib/Seaborn ile Otomatik AI Deney Raporlama Motoru (AI Experiment Report Generator)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** serimizin 46. gününde geliştirilen **Otomatik AI Deney Raporlama ve Değerlendirme Motorudur**. Derin öğrenme ve makine öğrenimi modellerinin eğitim/doğrulama telemetrisini anlık izler, ROC-AUC, PR-Eğrisi, Karmaşıklık Matrisi (Confusion Matrix) ve Matthews Korelasyon Katsayısını (MCC) hesaplar, 6 panelli yüksek çözünürlüklü grafik panoları ile bağımsız **HTML Deney Raporları** derler.

---

## 📖 Mentorluk Dersi ve Deney Raporlama Teorisı

### 1. Modern MLOps Süreçlerinde Deney Raporlamanın Önemi

Endüstriyel yapay zeka projelerinde (MLflow, Weights & Biases, TensorBoard ortamlarında) salt eğitim logları yeterli değildir. Bir modelin canlıya (production) alınıp alınamayacağına karar vermek için 4 eksenli değerlendirme zorunludur:

1. **Kayıp ve Doğruluk Yakınsama Dinamiği:**
   - `Train Loss` vs `Validation Loss` eğrileri incelenir.
   - **Aşırı Öğrenme Farkı (Overfitting Gap):** $\text{Overfitting Gap} = \text{Val Loss}_{\text{son}} - \min(\text{Val Loss})$.
   - **Erken Durdurma (Early Stopping):** Doğrulama kaybı $P$ sabır (patience) adımı boyunca iyileşmediğinde eğitim durdurulmalıdır.

2. **Dengesiz Verilerde ROC-AUC vs Precision-Recall (PR) Ayrımı:**
   - **ROC Eğrisi (TPR vs FPR):** Negatif sınıfın baskın olduğu problemlerde (örn. anomali, dolandırıcılık tespiti) yanıltıcı derecede iyimser kalabilir.
   - **PR Eğrisi (Precision vs Recall):** Yalnızca pozitif sınıf performansına odaklanır ve taban başarı oranı rastgele tahmin çizgisiyle (prevalence) kıyaslanır.

3. **Gelişmiş Değerlendirme Metrikleri:**
   - **F1-Skoru:** Kesinlik ve duyarlılığın harmonik ortalaması: $2 \cdot \frac{P \cdot R}{P + R}$.
   - **Matthews Correlation Coefficient (MCC):** Tüm 4 hücreyi (TP, TN, FP, FN) orantısal değerlendiren en dengeli korelasyon metriğidir ($-1.0$ ile $+1.0$ arası).

```
                           ┌──────────────────────────────────────────────────────────┐
                           │          MODEL EĞİTİM VE TEST TELEMETRİSİ                │
                           │   (Epoch Loss/Acc, Prediction Probs, Ground Truth)       │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                         METRİK HESAPLAMA VE İSTATİSTİKSEL ANALİZ                                  │
    │  - EgitimGecmisi     : En İyi Epoch Tespiti, Overfitting Gap, Erken Durdurma Kontrolü              │
    │  - MetrikHesaplayici : Confusion Matrix, ROC-AUC, PR-Curve, Average Precision (AP), MCC           │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                    OTOMATİK ÇIKTI VE GÖRSELLEŞTİRME MOTORU (Day 46)                               │
    │  1. 6-Panelli Matplotlib/Seaborn Teşhis Panosu (300 DPI PNG)                                      │
    │  2. İnteraktif, Bağımsız Yönetici HTML Deney Raporu (deney_raporu.html)                           │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Model eğitim metriklerini, kayıp eğrilerini, PR/ROC eğrilerini ve hata analizlerini otomatik olarak çok panelli yayın kalitesinde PDF/PNG raporuna dökmek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Manuel grafik çizme ve deney sonuçlarını bir araya getirme operasyonel yükünü tamamen ortadan kaldırır.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Dinamik/interaktif yakınlaştırma sağlamaz; statik yüksek çözünürlüklü görsel üretir.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- Weights & Biases, MLflow UI, TensorBoard veya Plotly.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Çok Eksenli Çizim Düzeni** | *Multi-Axes Grid Layout (`plt.subplots`)* | Tek bir görsel rapor içinde 4-6 farklı teşhis panelini profesyonel tipografi ve renk uyumuyla sunma. |
| **Öğrenme Eğrileri** | *Learning & Loss Trajectories* | Eğitim ve doğrulama kayıplarının epoch bazında eğilimini ve aşırı öğrenme (overfitting) anını gösteren grafik. |
| **Yayın Kalitesinde Rapor** | *Publication-Quality Graphics* | 300 DPI çözünürlük, vektörel netlik ve şirket standartlarına uygun renk paletleriyle otomatik grafik üretimi. |
| **Eşik & Kalibrasyon Grafiği** | *Reliability & Calibration Plot* | Modelin tahmin olasılıklarının gerçek doğruluk oranlarıyla örtüşmesini gösteren güvenilirlik diyagramı. |

---

## 2. Matematiksel Formülasyonlar

#### A. Matthews Korelasyon Katsayısı (MCC)
$$\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$$

#### B. Trapezoidal ROC-AUC Alanı
$$\text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d\text{FPR} \approx \sum_{i=1}^{N-1} \frac{\text{TPR}_i + \text{TPR}_{i+1}}{2} (\text{FPR}_{i+1} - \text{FPR}_i)$$

---

## 🛠️ Dizin Yapısı

```
day-46-matplotlib-ai-experiment-report-generator/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca deney izleme ve raporlama yürütme betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── egitim_izleyici.py           # EgitimGecmisi (Epoch, Loss, Acc, Overfitting Gap, Early Stopping)
│   ├── metrik_hesaplayici.py        # MetrikHesaplayici (Confusion Matrix, ROC-AUC, PR-Curve, AP)
│   ├── raporlayici.py               # OtomatikDeneyRaporlayici (HTML / Markdown / JSON Deney Raporu)
│   └── gorsellestirici.py           # 6-Panelli Deney Teşhis Panosu (Loss/Acc, ROC, PR, CM, Yönetici Özeti)
├── testler/
│   ├── __init__.py
│   └── test_experiment_reporter.py  # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    ├── deney_raporu_paneli.png      # 6 panelli yüksek çözünürlüklü grafik panosu
    └── deney_raporu.html            # İnteraktif HTML deney teftiş raporu
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

## 📊 Model Değerlendirme ve Deney Metrikleri

| Metrik Adı | Değer | Hedef Eşik | Değerlendirme Kararı |
|---|---|---|---|
| **ROC-AUC** | **$0.952$** | $\ge 0.85$ | ✅ Mükemmel Sınıf Ayrımı |
| **F1-Skoru** | **$0.865$** | $\ge 0.80$ | ✅ Güçlü Denge |
| **Matthews Corr (MCC)** | **$+0.724$** | $\ge +0.60$ | ✅ Yüksek Güvenilirlik |
| **Doğruluk (Accuracy)** | **$\%88.40$** | $\ge \%85$ | ✅ Onaylandı |
| **Overfitting Gap** | **$0.048$** | $< 0.10$ | ✅ Kararlı Yakınsama |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/metrik_hesaplayici.py` içerisine, Recall'a Precision'dan daha fazla ağırlık veren $\beta$ parametreli **$F_\beta$-Score Optimize Edici** fonksiyonunu eklemek.

**Tamamlanan Çözüm:**
```python
def f_beta_skoru_hesapla(precision: float, recall: float, beta: float = 2.0) -> float:
    """Recall'ı veya Precision'ı önceliklendiren F-Beta skorunu hesaplar."""
    beta_sq = beta ** 2
    pay = (1 + beta_sq) * precision * recall
    payda = (beta_sq * precision) + recall
    return float(pay / max(payda, 1e-8))
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Sınıf dağılımının aşırı dengesiz olduğu (örneğin pozitif sınıfın sadece %1 olduğu dolandırıcılık tespiti) bir yapay zeka görevinde, neden **ROC-AUC yerine PR-AUC (Average Precision)** metriğine bakılmalıdır?

> **Mentor Cevabı:**
> ROC eğrisinin yatay eksenini oluşturan **FPR (False Positive Rate)**, $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$ olarak hesaplanır. Negatif sınıf sayısı devasa olduğunda ($\text{TN} = 990.000$), model yüzlerce yanlış alarm ($\text{FP} = 500$) üretse dahi payda çok büyük olduğundan FPR sıfıra yakın kalır ve ROC-AUC yanıltıcı şekilde $0.98$ gibi mükemmel görünür. Oysa **Precision-Recall** eğrisinde Kesinlik ($\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$) doğrudan yanlış pozitiflerle cezalandırılır ve gerçek üretim kalitesini şeffaf biçimde ortaya koyar.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
