# Day 24: Model Değerlendirme & Hata Analizi (Model Evaluation & Error Analysis)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; üretim ortamındaki bir derin öğrenme veya makine öğrenmesi modelinin gerçek dünya güvenilirliğini ölçmek için tekil Doğruluk (Accuracy) metriğinin ötesine geçer. **Normalleştirilmiş Karışıklık Matrisi (Confusion Matrix)**, **Çok Sınıflı ROC-AUC**, **Precision-Recall (PR-AUC)**, **Olasılık Kalibrasyonu (ECE / Reliability Diagram)**, **Sıcaklık Ölçekleme (Temperature Scaling)**, **Top-$k$ Doğruluğu** ve **Aşırı Güvenli Hata Denetimini (Overconfident Failure Audit)** içeren 6 panelli endüstri standardı bir teşhis panosu (Diagnostic Dashboard) sunar.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Neden Yalnızca Doğruluk (Accuracy) Yetersizdir?
Güvenlik-kritik (otonom araçlar, medikal görüntüleme, dolandırıcılık tespiti) sistemlerde yanlış pozitif (False Positive) ve yanlış negatif (False Negative) hatalarının maliyeti asimetriktir:
- **Sınıf Dengesizliği (Class Imbalance):** $\%98$ negatif, $\%2$ pozitif olan bir kanser teşhis veri setinde her hastaya "sağlıklı" diyen naif bir model $\%98$ doğruluk alır; ancak tüm hasta bireyleri kaçırarak ölümcül bir başarısızlığa yol açar.
- **Güven Yanılsaması:** Bir model $\%99$ doğruluk elde etse dahi, yanlış tahmin ettiği $\%1$'lik kısımda $\%99.9$ eminse (Overconfident Failure), bu model otonom karar verici sistemlere teslim edilemez.

---

### 2. Çok Sınıflı Metrikler ve Matematiksel Temeller

#### A. Çok Sınıflı ROC-AUC (Receiver Operating Characteristic)
Her bir $c$ sınıfı için pozitif, diğer tüm sınıflar negatif kabul edilerek (**One-vs-Rest**) hesaplanır:

$$\text{TPR (Sensitivity/Recall)} = \frac{TP}{TP + FN}, \quad \text{FPR (1 - Specificity)} = \frac{FP}{FP + TN}$$

$$\text{Macro ROC-AUC} = \frac{1}{C} \sum_{c=1}^C \text{AUC}_c$$

#### B. Precision-Recall (PR-AUC) Eğrisi ve Ortalama Hassasiyet (AP)
Negatif sınıf sayısının ($TN$) devasa olduğu durumlarda ROC-AUC yapay olarak yüksek çıkar. PR eğrisi $TN$'e bağlı olmadığı için dengesiz veri setlerinde gerçek model başarısını gösterir:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{Average Precision (AP)} = \sum_n (R_n - R_{n-1}) P_n$$

#### C. Model Olasılık Kalibrasyonu ve Beklenen Kalibrasyon Hatası (ECE)
Modern derin ağlar (Batch Normalization, Residual Connections vb.), CrossEntropy kaybı nedeniyle aşırı yüksek güven skorları üretir.
- **ECE Formülü:** Olasılık aralığı $[0, 1]$ $M$ adet eşit kutuya (Bin $B_m$) bölünür:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \big| \text{acc}(B_m) - \text{conf}(B_m) \big|$$
- **Brier Skoru:** Çok sınıflı olasılık sapmasını ölçen kuadratik skor ($0$ en iyi):
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{i,k} - y_{i,k})^2$$

#### D. Sıcaklık Ölçekleme (Temperature Scaling - Guo et al., 2017)
Modelin tahmin doğruluğunu ve sınıf sıralamasını değiştirmeden, logitleri optimize edilmiş tek bir skaler $T > 0$ sıcaklık katsayısına bölerek aşırı güveni kalibre eder:

$$p_i = \frac{e^{z_i / T}}{\sum_{j=1}^C e^{z_j / T}}$$

#### E. Top-$k$ Doğruluğu (Top-$k$ Accuracy)
Doğru etiketin modelin en yüksek olasılık atadığı ilk $k$ tahmin kümesi içinde yer alma olasılığı:

$$\text{Top-}k \text{ Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\Big(y_i \in \text{argtopk}(\hat{p}_i, k)\Big)$$

---

## 🛠️ Dizin Yapısı

```
day-24-model-evaluation-and-error-analysis/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # scikit-learn, scipy, torch, matplotlib vb.
├── ana_akis.py                      # Uçtan uca değerlendirme ve hata denetimi akışı
├── README.md                        # Detaylı teorik ve mentorluk dokümantasyonu
├── src/
│   ├── __init__.py
│   ├── metrik_hesaplayici.py        # ROC-AUC, PR-AUC, Top-k ve metrik motoru
│   ├── kalibrasyon_analizcisi.py    # ECE, Brier Skoru ve Sıcaklık Ölçekleme
│   ├── hata_denetcisi.py            # Aşırı güvenli hata ve karışan çiftler denetimi
│   └── gorsellestirici.py           # 6 panelli teşhis panosu (Dashboard) çizici
├── testler/
│   ├── __init__.py
│   └── test_degerlendirme.py        # 7 adet kapsamlı birim test
└── ciktilar/
    └── model_degerlendirme_paneli.png # 6 panelli kapsamlı teşhis panosu
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

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Dengesiz (imbalanced) sınıflara sahip görüntü veri setlerinde genel Doğruluk (Accuracy) metriği neden yanıltıcıdır ve Precision-Recall AUC (PR-AUC) ne zaman ROC-AUC'ye tercih edilmelidir?

> **Mentor Cevabı:**
> 1. **Accuracy Paradoksu:** Eğer veri setinin %98'i sağlıklı, %2'si defolu dokumaysa, her şeye "sağlıklı" diyen naif bir model %98 doğruluk verir ama tüm defoları kaçırır ($0\%$ Recall).
> 2. **PR-AUC vs ROC-AUC:** ROC-AUC negatif sınıfın çok büyük olduğu durumlarda False Positive Rate ($FPR = \frac{FP}{FP+TN}$) paydasındaki büyük $TN$ nedeniyle aşırı iyimser görünür. PR-AUC ise sadece pozitif tahminlere ($Precision = \frac{TP}{TP+FP}$) ve yakalanan pozitiflere ($Recall = \frac{TP}{TP+FN}$) odaklandığı için dengesiz veri setlerinde gerçek model başarısını doğru yansıtır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
