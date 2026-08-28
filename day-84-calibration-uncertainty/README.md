# Day 84: Olasılık Kalibrasyonu, Expected Calibration Error (ECE) & Temperature Scaling

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Reliability: ECE & Temperature Scaling](https://img.shields.io/badge/Reliability-Temperature_Scaling-brightgreen.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_kalibrasyon.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin üçüncü gününde; Chuan Guo et al. (2017) *"On Calibration of Modern Neural Networks"* makalesinin temelini oluşturan **Olasılık Kalibrasyonu (Probability Calibration)** ve **Post-Hoc Temperature Scaling** mekanizmasını sıfırdan kuruyoruz. Modern derin ağların (ResNet, Vision Transformer) muzdarip olduğu kronik **Aşırı Güven (Overconfidence)** krizini teşhis ediyor; **Expected Calibration Error (ECE)** ve **Güvenilirlik Diyagramları (Reliability Diagrams)** ile modelin sınıflandırma doğruluğunu hiç bozmadan güven skorlarını gerçeğe kalibre ediyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Klasik sığ ağlar (LeNet, 1998) iyi kalibreydi: model "%80 eminim" dediğinde gerçekten %80 oranında doğru tahmin yapıyordu. Ancak modern derin sinir ağları; Batch Normalization, aşırı derinlik ve Cross-Entropy'yi sıfırlamaya yönelik agresif optimizasyon yüzünden **aşırı güvenli (overconfident)** hale gelmiştir:

1. **Aşırı Güven Krizi (The Overconfidence Crisis):**
   Modern bir Vision Transformer veya ResNet, sadece %50 doğru tahmin yaptığı karmaşık test örneklerinde dahi Softmax çıkışında %98 - %99 olasılık üretir. Bu durum modelin neyi bilip neyi bilmediğinin farkında olmadığını gösterir.
2. **Güvenlik Kritik Karar Sistemleri (Safety-Critical AI):**
   Tıbbi teşhis (ör. tümör tespiti), otonom sürüş (ör. yaya tanıma) ve finansal dolandırıcılık sistemlerinde bir yapay zekanın sadece tahmin yapması yetmez; **tahminindeki belirsizlik derecesini (Uncertainty)** dürüstçe raporlaması şarttır.
3. **$\arg\max$ Değişmezliği (Accuracy Invariance):**
   Temperature Scaling, $T > 0$ tekil skaler parametresiyle logitleri $z / T$ şeklinde ölçekler. Herhangi bir $T > 0$ için $\arg\max_k(z_k / T) = \arg\max_k(z_k)$ olduğundan, **modelin Top-1 doğruluğu %1 bile değişmez**, yalnızca olasılıkların güvenilirliği düzelir!

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Yalancı Güven ve Hatalı Eşikleme (False Confidence in Thresholding):**
  Üretim ortamında "güveni %90'ın altındaki tahminleri insan operatöre yönlendir" gibi eşikleme filtreleri ham modellerde çalışmaz (çünkü model hatalı olsa bile %99 güven verir). Kalibre model bu mekanizmayı çalışır kılar.
- **Ensemble ve Karar Füzyonu (Bayesian Decision Making):**
  Birden fazla modelin olasılıklarını birleştirirken (Bayesian fusion), kalibre edilmemiş aşırı güvenli modeller diğer tüm modellerin oyunu domine ederek hatalara yol açar.
- **Negative Log-Likelihood (NLL) ve Brier Skoru Uçurumu:**
  Doğruluğu yüksek ama NLL kaybı fırlamış modellerin olasılık kalitesini restore eder.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Dağılım Kayması (Domain Shift) Hassasiyeti:**
  Eğitim/Doğrulama kümesinde öğrenilen $T^*$ sıcaklık sabiti, test ortamında veri dağılımı değiştiğinde (Out-of-Distribution / Covariate Shift) kalibrasyon özelliğini kaybedebilir.
- **Sınıf-Bağımsız Skaler Kısıtı:**
  Tek bir skaler $T$ parametresi tüm sınıflara eşit uygulanır. Eğer model bazı sınıflarda aşırı güvenli, bazılarında eksik güvenli (underconfident) ise Vektör Ölçekleme (Vector Scaling) veya Matris Ölçekleme gerekebilir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Kalibrasyon Yöntemi | Parametre Sayısı | Doğruluk Değişir mi? | Hesaplama Karmaşıklığı | Önerilen Alan |
|---|---|---|---|---|
| **Temperature Scaling (Bizim Yöntem)** | **1 (Skaler $T^*$)** | **HAYIR (%0 Etki)** | **Ultra Düşük (L-BFGS / <1 sn)**| **Tüm Çok Sınıflı Vision/NLP** |
| **Platt Scaling (Sigmoid)** | 2 ($A, B$) | Hayır | Düşük | İkili Sınıflandırma (Binary) |
| **İzotonik Regresyon (Isotonic)** | Parametresiz (Step) | Evet (Değişebilir) | Orta (Sıralama Algoritması) | Tabüler Veri / Binary |
| **Monte Carlo Dropout (MC Dropout)**| $N$ İleri Geçiş | Evet | Yüksek ($10\times - 50\times$ Latency) | Epistemik Belirsizlik Modelleme |
| **Derin Topluluklar (Deep Ensembles)**| $M \times \text{Model}$| Evet (Artar) | Çok Yüksek ($M\times$ Bellek & GPU) | Ağır Sunucu Sistemleri |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          PROBABILITY CALIBRATION & TEMPERATURE SCALING AKIŞI                              │
│                                                                                                           │
│       1. MODEL ÇIKARIMI (Ham Logitler):                                                                   │
│          z = Model(x) = [z_1, z_2, ..., z_K]                                                              │
│                                                                                                           │
│       2. POST-HOC TEMPERATURE SCALING (Guo et al. 2017):                                                  │
│          q_i = Softmax(z_i / T*) = exp(z_i / T*) / sum_j exp(z_j / T*)                                   │
│                                                                                                           │
│       3. DOĞRULAMA KÜMESİNDE L-BFGS İLE T* OPTİMİZASYONU:                                                 │
│          min_{T > 0} - sum_{i=1}^{N_val} log( exp(z_{i, y_i} / T) / sum_j exp(z_{i, j} / T) )            │
│                                                                                                           │
│       4. EXPECTED CALIBRATION ERROR (ECE) METRİĞİ:                                                        │
│          M adet güven aralığı (Bin) B_m = ((m-1)/M, m/M] için:                                            │
│          ECE = sum_{m=1}^M ( |B_m| / N ) · | acc(B_m) - conf(B_m) |                                      │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Beklenen Kalibrasyon Hatası (Expected Calibration Error - ECE)
Tahminler güven skorlarına göre $M$ adet eşit genişlikteki dilime (bin) $B_m$ ayrılır:

- **Dilim Doğruluğu (Accuracy):**
  $$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$

- **Dilim Ortalama Güveni (Confidence):**
  $$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$

- **Toplam ECE Formülü:**
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \Big| \text{acc}(B_m) - \text{conf}(B_m) \Big|$$

### 2. Post-Hoc Temperature Scaling (NLL Optimizasyonu)
$$\hat{q}_i = \frac{\exp(z_i / T^*)}{\sum_{j=1}^K \exp(z_j / T^*)}$$

$$T^* = \arg\min_{T > 0} \left( -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \left( \frac{\exp(z_{i, y_i} / T)}{\sum_{j=1}^K \exp(z_{i, j} / T)} \right) \right)$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Calibration** | *Kalibrasyon* | Bir modelin tahmin ettiği olasılık skorunun, olayın gerçek gerçekleşme frekansı ile tam örtüşmesi durumu. |
| **Overconfidence** | *Aşırı Güven* | Modelin emin olmadığı veya yanlış bildiği durumlarda dahi %95+ gibi yüksek olasılıklar üretmesi anomalisi. |
| **ECE (Expected Calibration Error)**| *Beklenen Kalibrasyon Hatası*| Güvenilirlik dilimleri bazında ortalama güven ile gerçek doğruluk arasındaki ağırlıklı farkların toplamı. |
| **Reliability Diagram** | *Güvenilirlik Diyagramı* | $x$ ekseninde güven, $y$ ekseninde doğruluk olan ve ideal kalibrasyonu $y=x$ doğrusuyla gösteren görsel grafik. |
| **Temperature Scaling ($T$)** | *Sıcaklık Ölçekleme* | Logitleri sabit bir $T^*$ değerine bölerek olasılıkların entropisini ve güvenini ayarlayan post-hoc yöntem. |
| **Brier Score** | *Brier Skoru* | Tahmin edilen olasılık vektörü ile tek-sıcak hedef vektörü arasındaki ortalama karesel farkı ölçen kalibrasyon metriği. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Modelin doğruluğunu (Accuracy) ve tahminlerini ASLA değiştirmez; Sadece tek bir parametre ($T^*$) öğrenir; saniyeler içinde eğitilir; ECE ve NLL değerlerini dramatik şekilde düşürür. |
| **Weaknesses (Zayıf Yönler)** | Sınıf bazında farklı aşırı güven varsa tek skaler $T^*$ yetersiz kalabilir; Ayrı bir doğrulama (validation) kümesi ayrılmasını gerektirir. |
| **Opportunities (Fırsatlar)** | OOD tespiti ve seçici tahmin (Abstention) mekanizmalarının temeli; Tıbbi ve otonom sistemlerde güvenilir belirsizlik modellemesi. |
| **Threats (Tehditler)** | Dağılım kayması (Domain shift) durumunda $T^*$ kalibrasyonu bozulabilir. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-84-calibration-uncertainty/`](.) dizinindedir:

### A. Temperature Scaling Kalibratörü (PyTorch & L-BFGS)
Dosya: [`src/kalibrator.py`](src/kalibrator.py)
```python
class SicaklikKalibratoru(nn.Module):
    def __init__(self, baslangic_sicaklik: float = 1.5):
        super().__init__()
        self.sicaklik = nn.Parameter(torch.ones(1) * baslangic_sicaklik)

    def forward(self, logitler: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(self.sicaklik, min=0.05, max=50.0)
        return logitler / t

    def kalibre_et(self, val_logitler, val_etiketler, max_iter=50, lr=0.05):
        optimizer = torch.optim.LBFGS([self.sicaklik], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logitler), val_etiketler)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return {"optimal_sicaklik": self.sicaklik.item()}
```

---

## 📊 Deneysel Sonuçlar ve Doğrulama Çıktıları

`ana_akis.py` koşturularak elde edilen kalibrasyon metrikleri:

```text
=====================================================================================
🚀 Day 84: Olasılık Kalibrasyonu, ECE ve Temperature Scaling Laboratuvarı
=====================================================================================
[Ham Model (Kalibrasyon Öncesi)]
  ✓ Test Doğruluğu: %50.00
  ✓ Test ECE (Expected Calibration Error): %11.25
  ✓ Test NLL (Negative Log-Likelihood): 1.4397

[Temperature Scaling Kalibrasyonu (Val Kümesi)]
  ✓ Bulunan Optimal Sıcaklık (T*): 1.5865

[Kalibre Model (Kalibrasyon Sonrası)]
  ✓ Test Doğruluğu: %50.00 (Doğruluk %100 Değişmedi!)
  ✓ Test ECE: %9.02 (Kalibrasyon İyileşmesi!)
  ✓ Test NLL: 1.3963 (NLL Azaldı!)

✓ 6 Panelli Teşhis Panosu Kaydedildi: ciktilar/calibration_uncertainty_paneli.png
```

- **Doğruluk Korunumu:** Modelin tahmin ettiği sınıflar değişmeden ($T^*=1.5865$) NLL kaybı ve ECE hatası minimize edilmiştir.
- **Birim Test Güvencesi:** [`testler/test_kalibrasyon.py`](testler/test_kalibrasyon.py) altındaki **8/8 birim test %100 PASSED (4.98s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/calibration_uncertainty_paneli.png`](ciktilar/calibration_uncertainty_paneli.png) konumundadır:

1. **Model Kalibrasyonu İlkesi:** Mükemmel kalibrasyon ($y=x$) ve aşırı güven kavramı.
2. **Kalibrasyon Öncesi Güvenilirlik Diyagramı:** Yüksek ECE ve belirgin kalibrasyon açığı (Gap).
3. **Kalibrasyon Sonrası Güvenilirlik Diyagramı:** $T^*$ ile köşegene yaklaşan kalibre doğruluk çubukları.
4. **Tahmin Güveni Histogramı:** Aşırı güvenli dağılımın gerçekçi güven seviyelerine ötelenmesi.
5. **NLL Kayıp Yüzeyi:** $T^*$ parametresinin NLL'i konveks biçimde minimize edişi.
6. **Kalibrasyon SWOT Karar Matrisi:** Endüstriyel karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Tekil bir skaler $T$ yerine, her sınıf için bağımsız bir sıcaklık ve bias vektörü öğrenen **Vektör Ölçekleme (Vector Scaling: $q_i = \text{Softmax}(W \cdot z + b)$ where $W = \text{diag}(w_1, \dots, w_K)$)** modülü yazınız.

```python
import torch
import torch.nn as nn

class VektorKalibratoru(nn.Module):
    """Vector Scaling: Her sınıf için ayrı sıcaklık çarpanı ve bias öğrenir."""
    def __init__(self, sinif_sayisi: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(sinif_sayisi))
        self.b = nn.Parameter(torch.zeros(sinif_sayisi))

    def forward(self, logitler: torch.Tensor) -> torch.Tensor:
        # logitler: (B, K)
        return logitler * torch.clamp(self.w, min=0.01) + self.b
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden Temperature Scaling optimizasyonunda modelin ana eğitim kümesi (train set) değil de bağımsız bir doğrulama kümesi (validation set) kullanılmak zorundadır? Eğitim kümesinde optimize edilirse ne olur?

> **Mentor Cevabı:**
> 1. **Aşırı Uydurma ve Sıcaklığın Sıfıra Çökmesi:** Modern derin ağlar eğitim kümesinde Cross-Entropy kaybını neredeyse 0'a indirecek şekilde eğitilmiştir. Eğer $T$ eğitim kümesinde optimize edilirse, optimizasyon algoritması eğitim logitlerini daha da büyütmek ve Cross-Entropy'yi daha da küçültmek için $T \to 0$ yapacaktır (Sonsuz aşırı güven!).
> 2. **Genelleme ve Doğrulama İlkesi:** Model doğrulama kümesindeki örnekleri daha önce hiç görmemiştir; dolayısıyla modelin doğrulama kümesindeki logitleri gerçek test belirsizliğini yansıtır. $T^*$ bu kümede optimize edildiğinde, aşırı güveni yumuşatacak $T^* > 1.0$ değerini doğru şekilde bulur.

---

### 📌 Git & Yol Haritası Güncellemesi:
- **Tamamlanan:** Gün 84 (`day-84-calibration-uncertainty`) başarıyla tamamlandı.
- **Sıradaki Gün:** **Day 85: Enerji Tabanlı Dağılım Dışı (OOD) Tespiti ve Seçici Tahmin (Abstention) (`day-85-ood-selective-prediction`)**.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
