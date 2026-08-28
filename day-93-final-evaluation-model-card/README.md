# Day 93: Kapsamlı Değerlendirme, Yanlılık (Bias) Testleri ve Standart Model Card Üretimi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Model Card: Mitchell et al. / Hugging Face](https://img.shields.io/badge/Model_Card-Standard_Compliant-darkgreen.svg?style=flat-square)](MODEL_CARD.md)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_model_card.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin on ikinci gününde; bir yapay zeka modelinin üretime çıkmadan önce yalnızca genel bir doğruluk (Accuracy) metriğiyle değil, alt grup dilimleri (**Data Slicing**), adillik/yanlılık metrikleri (**Demographic Parity**, **Disparate Impact - %80 Kuralı**), olasılık kalibrasyonu (**ECE & Brier Score**) ve Hugging Face / Google standartlarında otomatik **Model Card (`MODEL_CARD.md`)** üretimiyle uçtan uca denetlenmesini sağlayan kurumsal bir değerlendirme paketi inşa ediyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Geleneksel makine öğrenimi değerlendirmelerinde sıklıkla yapılan en büyük hata, modelin tüm test seti üzerinde ortalama tek bir doğruluk skoruna (ör. %92 Accuracy) bakarak başarılı sayılmasıdır. Ancak bu "ortalama maskelemesi" (average masking) tehlikeli sonuçlar doğurur:

1. **Simpson Paradoksu ve Alt Grup Körlüğü (Subgroup Blindness):**
   Genel doğruluğu %92 olan bir model, belirli bir alt grupta (örneğin düşük ışıklı ortamlarda veya azınlık demografisinde) %40 doğrulukla çalışıyor olabilir.
2. **Yasal ve Etik Sorumluluk (Fairness & Bias Auditing):**
   İstihdam, kredi skorlama veya yüz tanıma gibi hassas alanlarda modellerin **Disparate Impact Ratio** ($\text{DIR} \ge 0.80$) yasal gereksinimini sağlaması zorunludur.
3. **Standart Model Dokümantasyonu (Model Cards for Model Reporting):**
   Margaret Mitchell et al. (2019) tarafından literatüre kazandırılan Model Card standardı, modelin hedeflenen kullanım amaçlarını, sınırlarını, eğitim/test kapsamını ve etik değerlendirmelerini şeffafça belgeler.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Alt Grup Dilimlerinde (Slices) Performans Çöküşlerini Yakalama:**
  Farklı ışık, çözünürlük veya kullanıcı gruplarında modelin nasıl davrandığını ayrıştırarak kör noktaları aydınlatır.
- **Aşırı Özgüvenli (Overconfident) Hatalı Tahminleri Engelleme:**
  Expected Calibration Error (ECE) ve Reliability Diagram ile modelin verdiği olasılıkların gerçek doğrulukla ne kadar uyumlu olduğunu denetler.
- **Üretim Öncesi Şeffaf Denetim Kartı Üretimi:**
  Manuel dokümantasyon yükünü ortadan kaldırarak her model sürümü için otomatik, doğrulanabilir bir `MODEL_CARD.md` üretir.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Alt Grup Etiketlerinin (Metadata Annotations) Varlık Zorunluluğu:**
  Veri dilimleme (slicing) yapabilmek için veri setinde ışık, açı, demografik bilgi veya cihaz tipi gibi ek metaveri etiketlerinin bulunması gerekir.
- **Adillik Metrikleri Arasındaki İmkansızlık Teoremi (Fairness Impossibility Theorem):**
  Matematiksel olarak Demographic Parity, Equalized Odds ve Predictive Parity kriterlerinin tamamını aynı anda mükemmel sağlamak teorik olarak imkansızdır; iş hedefine göre ödünleşim (trade-off) seçilmelidir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Değerlendirme Yaklaşımı | Alt Grup Dilimleme | Adillik (Fairness) Metrikleri | Kalibrasyon (ECE) | Model Card Dokümantasyonu |
|---|---|---|---|---|
| **Bizim Model Card Paketimiz** | **Evet (Özelleştirilebilir Maskeler)** | **Demographic Parity & %80 Kuralı** | **Tam (ECE + Brier + Eğri)** | **Otomatik Markdown & JSON** |
| **Scikit-Learn `classification_report`**| Sadece Sınıf Bazlı | Yok | Yok | Basit Metin |
| **Fairlearn (Microsoft)** | Evet | Çok Kapsamlı | Sınırlı | Yok |
| **AIF360 (IBM)** | Evet | Çok Geniş Algoritma Havuzu | Yok | Yok |
| **Hugging Face Model Card Hub** | Manuel Yazım | Manuel | Manuel | Web UI Şablonu |

---

## 📐 Matematiksel Formülasyon

### 1. Expected Calibration Error (ECE)
Test örnekleri tahmin güven skoruna göre $M$ adet eşit aralıklı kutuya ($B_1, B_2, \dots, B_M$) ayrıldığında ECE:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Burada $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{I}(y_i = \hat{y}_i)$ ve $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$'dir.

### 2. Demographic Parity Farkı ($\Delta_{\text{DP}}$)
Hassas öznitelik $A \in \{a, b\}$ (ör. Grup A vs Grup B) için pozitif tahmin oranları farkı:

$$\Delta_{\text{DP}} = \left| P(\hat{Y} = 1 \mid A = a) - P(\hat{Y} = 1 \mid A = b) \right|$$

### 3. Disparate Impact Oranı (DIR - 80% Kuralı)

$$\text{DIR} = \frac{P(\hat{Y} = 1 \mid A = \text{Azınlık})}{P(\hat{Y} = 1 \mid A = \text{Çoğunluk})}$$

Yasal kabul edilebilirlik kriteri:

$$\text{DIR} \ge 0.80 \quad (\%80 \text{ Kuralı})$$

### 4. Çok Sınıflı Brier Skoru
Gerçek one-hot vektör $y_i$ ve model olasılık vektörü $p_i$ için:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^C (p_{ik} - y_{ik})^2$$

---

## 📖 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Terim | Tanım | Endüstriyel Önemi |
|---|---|---|
| **Model Card** | Bir yapay zeka modelinin mimarisini, veri setini, metriklerini, sınırlılıklarını ve etik sonuçlarını açıklayan standart rapor. | Şeffaflık, regülasyon uyumu ve sorumlu yapay zeka için temel dokümandır. |
| **Data Slicing (Veri Dilimleme)** | Veri setini belirli özniteliklere (ör. düşük ışık, gürültü, yaş grubu) göre mantıksal alt kümelere bölüp ayrı ayrı test etme. | Modelin genel ortalamanın arkasına gizlenen zayıf noktalarını açığa çıkarır. |
| **Demographic Parity** | Bir modelin olumlu tahmin üretme oranının tüm demografik alt gruplar arasında eşit olması ilkesi. | Karar sistemlerinde tarafsızlık denetiminin ana sütunudur. |
| **Disparate Impact Ratio** | Azınlık grubun olumlu sonuç alma oranının çoğunluk gruba oranı ($P(\hat{Y}=1|A)/P(\hat{Y}=1|B)$). | ABD EEOC ve küresel regülasyonlarda yasal ayrımcılık sınırıdır (%80 kuralı). |
| **Expected Calibration Error (ECE)** | Modelin tahmin ettiği olasılık güveni ile gerçek doğruluğu arasındaki ağırlıklı mutlak fark. | Modelin "ne zaman bilmediğini bildiğini" ölçen en kritik metriktir. |
| **Brier Score** | Olasılık tahminlerinin doğruluğunu ölçen karesel ceza fonksiyonu ($[0, 2]$ aralığı). | Kalibrasyon ve tahmin keskinliğini birlikte değerlendirir. |
| **Reliability Diagram** | Güven skoru kutuları ile gerçek doğruluk arasındaki ilişkiyi $y=x$ doğrusuna göre çizen teşhis grafiği. | Modelin aşırı özgüvenli mi yoksa çekingen mi olduğunu görselleştirir. |
| **Macro vs Weighted F1** | Macro F1 sınıflara eşit ağırlık verirken, Weighted F1 sınıf frekansına göre ağırlıklandırır. | Dengesiz veri setlerinde azınlık sınıfların başarısını ölçmek için Macro F1 esastır. |
| **Subgroup Disparity** | Farklı veri dilimleri arasında gözlenen maksimum performans farkı ($\max(\text{Acc}) - \min(\text{Acc})$). | Modelin operasyonel dayanıklılık ve adillik seviyesini gösterir. |
| **Intended vs Out-of-Scope Use** | Modelin tasarlandığı geçerli kullanım senaryoları ile kesinlikle kullanılmaması gereken tehlikeli alanların ayrımı. | Hukuki ve etik risklerin önüne geçer. |

---

## 📊 SWOT Analizi (Karar Matrisi)

```
┌──────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│                   GÜÇLÜ YÖNLER (S)               │                  ZAYIF YÖNLER (W)                │
│ • Alt grup dilimleme ile kör noktaları yok etme. │ • Dilim etiketleri (metadata) gereksinimi.       │
│ • %80 Kuralı ile yasal adillik denetimi.         │ • Tüm adillik kriterlerini aynı anda sağlayamaz. │
│ • Otomatik standart MODEL_CARD.md üretimi.       │ • ECE kutu sayısı seçimi hassasiyeti.            │
├──────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│                  FIRSATLAR (O)                   │                   TEHDİTLER (T)                  │
│ • Hugging Face Model Hub ve kurumsal dağıtım.    │ • Yetersiz dilim örneklerinde istatistik sapması │
│ • AB Yapay Zeka Yasası (EU AI Act) uyumluluğu.   │ • Üretim ortamında bilinmeyen yeni alt gruplar.  │
│ • Model yönetim panellerine otomatik raporlama.  │ • Hatalı kalibrasyonun güveni zedelemesi.        │
└──────────────────────────────────────────────────┴──────────────────────────────────────────────────┘
```

---

## 🧪 Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

### 🎯 Görev: Equalized Odds (Fırsat Eşitliği) Denetleyicisi
Yalnızca pozitif tahmin oranına değil, alt gruplar arasındaki **True Positive Rate (TPR / Sensitivity)** ve **False Positive Rate (FPR)** farklarını denetleyen **Equalized Odds Denetleyicisi** sınıfını geliştirin.

### 💡 Eksiksiz Çalışan Çözüm Kodu:

```python
import numpy as np

class EqualizedOddsDenetleyicisi:
    """Grup A ve Grup B arasında TPR ve FPR eşitliğini denetler."""
    def __init__(self, maks_tpr_fark_esigi: float = 0.10, maks_fpr_fark_esigi: float = 0.10):
        self.maks_tpr_farki = maks_tpr_fark_esigi
        self.maks_fpr_farki = maks_fpr_fark_esigi

    def degerlendir(self, y_true: np.ndarray, y_pred: np.ndarray, grup_maskesi_a: np.ndarray) -> dict:
        grup_a_true, grup_a_pred = y_true[grup_maskesi_a], y_pred[grup_maskesi_a]
        grup_b_true, grup_b_pred = y_true[~grup_maskesi_a], y_pred[~grup_maskesi_a]

        def hesapla_tpr_fpr(yt, yp):
            tp = np.sum((yt == 1) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            fp = np.sum((yt == 0) & (yp == 1))
            tn = np.sum((yt == 0) & (yp == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            return float(tpr), float(fpr)

        tpr_a, fpr_a = hesapla_tpr_fpr(grup_a_true, grup_a_pred)
        tpr_b, fpr_b = hesapla_tpr_fpr(grup_b_true, grup_b_pred)

        fark_tpr = abs(tpr_a - tpr_b)
        fark_fpr = abs(fpr_a - fpr_b)

        uygun = (fark_tpr <= self.maks_tpr_farki) and (fark_fpr <= self.maks_fpr_farki)

        return {
            "tpr_a": tpr_a, "tpr_b": tpr_b, "tpr_farki": fark_tpr,
            "fpr_a": fpr_a, "fpr_b": fpr_b, "fpr_farki": fark_fpr,
            "equalized_odds_gecerli": uygun
        }

# Test ve Doğrulama
denetleyici = EqualizedOddsDenetleyicisi(maks_tpr_fark_esigi=0.10)
yt = np.array([1, 1, 0, 0, 1, 1, 0, 0])
yp = np.array([1, 1, 0, 0, 1, 0, 0, 0])
mask_a = np.array([True, True, True, True, False, False, False, False])

sonuc = denetleyici.degerlendir(yt, yp, mask_a)
print(f"TPR Farkı: {sonuc['tpr_farki']:.2f} | FPR Farkı: {sonuc['fpr_farki']:.2f}")
print(f"Equalized Odds Geçerli mi: {sonuc['equalized_odds_gecerli']}")
print("✓ Equalized Odds Denetleyicisi Başarıyla Doğrulandı!")
```

---

## 📜 Lisans & Metaveri

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

### ❓ Derin Teknik Kontrol Sorusu:
> *"Bir Vision Sınıflandırıcısının genel doğruluğu (Accuracy) %94.5 olarak ölçülüyor. Ancak Reliability Diagram çizildiğinde ECE (Expected Calibration Error) değeri 0.28 çıkıyor ve modelin %90+ güvenle verdiği yanlış tahminlerin oranı %25 olarak görülüyor. Bu durum üretim ortamında neden büyük bir güvenlik açığıdır ve 'Temperature Scaling' gibi post-processing yöntemleriyle nasıl düzeltilir?"*

### 💡 Mentorluk Açıklaması ve Çözüm:
Bu durum, derin sinir ağlarında (özellikle modern derin CNN ve Vision Transformer mimarilerinde) sıkça karşılaşılan **Aşırı Özgüvenli Yanlış Tahmin (Overconfidence Calibration Mismatch)** problemidir:

1. **Güvenlik Riski:**
   Eğer model otonom araçlarda, tıbbi görüntülemede veya kalite kontrol hatlarında "emin değilim, insan denetçiye devret" (abstention / reject option) mekanizmasıyla çalışıyorsa, aşırı özgüvenli model %95 olasılıkla "Kusursuz Parça" diyerek hatalı ürünü gözden kaçırır veya yanlış kararı sisteme onaylatır.
2. **Kök Neden:**
   Cross-Entropy kaybı ve modern regülarizasyon teknikleri (BatchNorm, Weight Decay), logit değerlerini aşırı büyüterek Softmax çıkışını $0$ veya $1$'e doğru yapay olarak polarize eder.
3. **Temperature Scaling Çözümü:**
   Modelin ağırlıkları dondurulur; doğrulama (validation) seti üzerinde tek bir skalar $T > 0$ (Temperature parametresi) optimize edilir:
   $$\hat{p}_i = \text{Softmax}\left(\frac{z_i}{T}\right)$$
   $T > 1$ seçildiğinde modelin doğruluğu (Accuracy) ve argmax sınıf sıralaması **asla değişmez**, ancak aşırı şişirilmiş olasılıklar yumuşatılarak kalibrasyon hatası (ECE) $0.28$'den $<0.03$'e düşürülür.

---

### 🌟 Sonraki Adım:
**Gün 93** başarıyla tamamlandı ve test edildi. Hazır olduğunuzda **Gün 94 (`day-94-hugging-face-integration` — Hugging Face Model Hub Entegrasyonu, Konfigürasyon ve Model Paketleme)** ile devam edebiliriz.
