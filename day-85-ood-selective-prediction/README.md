# Day 85: Enerji Tabanlı Dağılım Dışı (OOD) Tespiti ve Seçici Tahmin (Abstention)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Safety: Energy OOD & Abstention](https://img.shields.io/badge/Safety-Energy_OOD_Abstention-brightgreen.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_ood.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin dördüncü gününde; Weitang Liu et al. (NeurIPS 2020) *"Energy-based Out-of-distribution Detection"* ve Yonatan Geifman & Ran El-Yaniv (2017) *"Selective Classification for Deep Neural Networks"* makaleleri ışığında **Enerji Tabanlı Dağılım Dışı (OOD) Tespiti** ve **Seçici Çekimserlik (Abstention)** sistemini sıfırdan kuruyoruz. Modelin eğitim dağılımına ait olmayan (OOD) yabancı nesneleri ve gürültüleri tespit ediyor, belirsiz durumlarda tahmin yapmayı reddederek **"Bilmiyorum, İnsan Uzmana Devret" (Fail-Safe Abstention)** kuralını hayata geçiriyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Standart derin öğrenme sınıflandırıcıları kapalı dünya varsayımı (closed-world assumption) altında çalışır: yani testte gelen her girdinin eğitimdeki $K$ sınıftan birine ait olduğunu varsayar. Ancak gerçek üretim ortamında (açık dünya - open-world) modele daha önce hiç görmediği yabancı nesneler, gürültülü sensör verileri veya saldırı görselleri gelir.

Bu sistem şu bilimsel ilkelerle çalışır:

1. **Softmax Normalizasyonunun Bozulması (The Softmax Fallacy):**
   Klasik Maksimum Softmax Olasılığı (MSP - Hendrycks & Gimpel 2017), $\sum_j \exp(z_j) = 1$ normalizasyon paydası yüzünden tüm logitler küçük olsa dahi (ör. $z = [0.1, 0.2, 0.1]$) yapay olarak yüksek olasılıklar (%40-%90) üretebilir.
2. **Serbest Enerji (Free Energy) İlkesi:**
   Gibbs dağılımı altında girdi yoğunluğu $p(x) \propto e^{-E(x)/T}$ formundadır. Serbest enerji skoru $S_{\text{energy}}(x) = T \cdot \log \sum_{k=1}^K \exp(z_k / T)$ logitlerin toplam büyüklüğünü doğrudan yansıtır. Dağılım İçi (ID) verilerde yüksek enerji skoru üretilirken, yabancı (OOD) verilerde enerji tabana çöker.
3. **Seçici Tahmin ve Çekimserlik (Selective Prediction / Abstention):**
   Model güvenliği için bir $\gamma$ eşiği belirlenir. $S(x) \ge \gamma$ ise model otomatik tahmin yapar; $S(x) < \gamma$ ise model tahmin yapmayı reddederek girdiyi insan denetimine aktarır. Böylece kabul edilen örneklerde doğruluk %100'e yaklaştırılır.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Kritik Hataların (Catastrophic Silent Failures) Önlenmesi:**
  Otonom araçların bilinmeyen bir cismi (ör. devrilmiş buzdolabı) "boş yol" veya medikal teşhis yapay zekasının yabancı bir dokuyu "temiz akciğer" olarak etiketlemesini engeller.
- **Kapsam vs Risk Dengesi (Coverage vs Risk Optimization):**
  İşletmelerin ihtiyaç duyduğu hata toleransına göre otomasyon yüzdesini (Coverage) ayarlama esnekliği sunar.
- **Modeli Yeniden Eğitme Maliyeti Olmadan Güvenlik:**
  Önceden eğitilmiş (pre-trained) herhangi bir modelin logitleri üzerinden $0\text{ ms}$ ek eğitim maliyetiyle uygulanır.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Yakın-OOD (Near-OOD) Zorluğu:**
  Eğer OOD veri kümesi ID sınıflarına semantik olarak çok yakınsa (ör. CIFAR-10 kediye karşı CIFAR-100 leopar), enerji skorları örtüşebilir ve ayrıştırma performansı düşebilir.
- **İnsan İş Gücü Yükü:**
  Çekimser kalınan (reddedilen) örneklerin hacmi arttıkça insan inceleme maliyeti ve gecikme artar.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| OOD / Güvenilirlik Yöntemi | Yeniden Eğitim | Çıkarım Maliyeti | AUROC Başarımı | Sınıf İçi Bilgi |
|---|---|---|---|---|
| **Energy-Based OOD (Bizim Yöntem)** | **GEREKMEZ** | **0 ms (LogSumExp)** | **YÜKSEK (%85 - %95)** | **Logit Yoğunluğu** |
| **Softmax MSP (Hendrycks et al.)**| Gerekmez | 0 ms | Düşük/Orta (%65 - %75) | Normalizasyon kısıtı |
| **Mahalanobis Distance (Lee et al.)**| Kovaryans hesabı | Orta ($O(D^3)$ ters matris)| Çok Yüksek (%90 - %98) | Ara Katman Öznitelikleri |
| **ODIN (Temperature + Perturbation)**| Gerekmez | Yüksek (2x Forward/Backward)| Yüksek (%85 - %92) | Gradyan Pertürbasyonu |
| **Outlier Exposure (OE)** | GEREKİR | 0 ms | En Yüksek (%95+) | Özel OOD Veri Kümesi |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          ENERJİ TABANLI OOD VE SEÇİCİ ÇEKİMSERLİK (ABSTENTION) AKIŞI                      │
│                                                                                                           │
│       Girdi: x                                                                                            │
│          │                                                                                                │
│          ▼                                                                                                │
│       Model Logitleri: z = f(x) = [z_1, z_2, ..., z_K]                                                    │
│          │                                                                                                │
│          ▼                                                                                                │
│       Serbest Enerji Skoru: S_energy(x) = T · log( sum_{k=1}^K exp(z_k / T) )                             │
│          │                                                                                                │
│          ├──> Karar Kriteri: S_energy(x) >= γ ?                                                           │
│          │                                                                                                │
│          ├─────────── [ EVET (ID - Güvenli) ] ───────────┐                                                │
│          │                                               │                                                │
│          ▼                                               ▼                                                │
│       [ RET / ABSTAIN ]                               [ OTOMATİK TAHMİN ]                                 │
│       "Tahmin Reddedildi, İnsan Uzmana Aktar!"        y_hat = argmax_k (z_k)                              │
│       (Fail-Safe Koruma Hattı)                        (Yüksek Güvenilirlik ve Düşük Risk)                 │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Serbest Enerji (Free Energy) Skoru
Helmoltz serbest enerjisi formülasyonu:

$$E(x; T) = -T \cdot \log \sum_{k=1}^K \exp\left(\frac{z_k(x)}{T}\right)$$

OOD tespiti için kullanılan Enerji Karar Skoru (yüksek skor = ID):

$$S_{\text{energy}}(x) = -E(x; T) = T \cdot \text{LogSumExp}\left(\frac{z(x)}{T}\right)$$

### 2. Seçici Tahmin Fonksiyonu (Selective Classifier)
$$\hat{y}_{\text{selective}}(x) = \begin{cases} \arg\max_k z_k(x) & \text{eğer } S_{\text{energy}}(x) \ge \gamma \\ \text{ABSTAIN (UZMANA DEVRET)} & \text{eğer } S_{\text{energy}}(x) < \gamma \end{cases}$$

### 3. Kapsam (Coverage) ve Risk Metrikleri
- **Kapsam (Coverage):**
  $$\text{Coverage}(\gamma) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\big(S(x_i) \ge \gamma\big)$$

- **Seçici Risk (Selective Risk / Error Rate):**
  $$\text{Risk}(\gamma) = \frac{\sum_{i=1}^N \mathbf{1}\big(S(x_i) \ge \gamma \land \hat{y}_i \ne y_i\big)}{\sum_{i=1}^N \mathbf{1}\big(S(x_i) \ge \gamma\big)}$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Out-of-Distribution (OOD)**| *Dağılım Dışı* | Modelin eğitim gördüğü sınıflara ve veri dağılımına ait olmayan yabancı/anormal girdiler. |
| **In-Distribution (ID)** | *Dağılım İçi* | Modelin öğrenmiş olduğu sınıf ve veri dağılımına uygun normal girdiler. |
| **Free Energy** | *Serbest Enerji* | Logitlerin exponansiyel toplamının logaritması üzerinden girdi yoğunluğunu ölçen fizik temelli metrik. |
| **Abstention (Rejection)** | *Çekimserlik / Reddetme* | Modelin hata yapma ihtimali yüksek olduğunda tahmin yapmayı reddedip insan uzmana devretmesi. |
| **AUROC** | *ROC Altındaki Alan* | Tüm olası eşik değerlerinde OOD dedektörünün ID ve OOD'yi ayrıştırma başarı olasılığı (%50 rastgele, %100 mükemmel). |
| **FPR95** | *%95 TPR'deki Yanlış Pozitif Oranı*| Model ID örneklerinin %95'ini doğru kabul ettiğinde, OOD örneklerinin yüzde kaçını yanlışlıkla içeri aldığı metriği. |
| **Coverage** | *Kapsam Oranı* | Sistemin çekimser kalmayıp otomatik olarak tahmin ürettiği örneklerin tüm örneklere oranı. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Modeli yeniden eğitmeye gerek yoktur (Pre-trained ile çalışır); Softmax MSP'ye göre OOD tespitinde belirgin yüksek AUROC üretir; Canlı ortamda kritik yanlış tahminleri neredeyse sıfıra indirir. |
| **Weaknesses (Zayıf Yönler)** | Çekimser kalınan (reddedilen) örnekler insan iş gücü maliyeti yaratır; Yakın-OOD (Near-OOD) sınıflarında ayrıştırma zorlaşabilir. |
| **Opportunities (Fırsatlar)** | Tıbbi tanı ve otonom araçlarda 'güvenli arıza' (fail-safe) kuralı; Bilinmeyen yeni sınıfları otomatik tespit edip etiketleme havuzuna alma. |
| **Threats (Tehditler)** | Eşik çok sıkı seçilirse sistemin otomasyon kapsamı (coverage) çöker. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-85-ood-selective-prediction/`](.) dizinindedir:

### A. Enerji Tabanlı OOD Dedektörü (PyTorch)
Dosya: [`src/enerji_ood.py`](src/enerji_ood.py)
```python
class EnerjiTabanliOODDedektoru:
    def __init__(self, sicaklik: float = 1.0, esik_degeri: float = None):
        self.sicaklik = sicaklik
        self.esik_degeri = esik_degeri

    @classmethod
    def enerji_skoru_hesapla(cls, logitler: torch.Tensor, sicaklik: float = 1.0) -> torch.Tensor:
        # S_energy(x) = T * logsumexp(z / T)
        return sicaklik * torch.logsumexp(logitler / sicaklik, dim=-1)

    def esik_belirle(self, id_logitler: torch.Tensor, hedef_tpr: float = 0.95) -> float:
        skorlar = self.enerji_skoru_hesapla(id_logitler, self.sicaklik).numpy()
        # ID validation örneklerinin %95'ini kabul eden eşik (5. persentil)
        self.esik_degeri = float(np.percentile(skorlar, (1.0 - hedef_tpr) * 100.0))
        return self.esik_degeri
```

---

## 📊 Deneysel Sonuçlar ve Doğrulama Çıktıları

`ana_akis.py` koşturularak elde edilen deneysel karşılaştırma:

```text
=====================================================================================
🚀 Day 85: Enerji Tabanlı OOD Tespiti ve Seçici Tahmin (Abstention) Laboratuvarı
=====================================================================================
[OOD Tespit Başarımı Karşılaştırması]
  ✓ Enerji Skoru  ──> AUROC: %87.89 | FPR95: %21.00 | AUPR: %76.61
  ✓ Softmax (MSP) ──> AUROC: %66.42 | FPR95: %48.50 | AUPR: %54.15
  🚀 Enerji Skoru AUROC Avantajı: +%21.47 Daha Üstün Ayrıştırma!

[Seçici Tahmin (Abstention) Sonuçları]
  ✓ Güvenlik Eşiği (γ): 4.8481
  ✓ Kapsam (Coverage): %96.50 (193/200 örnek kabul edildi)
  ✓ Seçici Tahmin Hata Oranı: %0.00 (Güvenli Alan)

✓ 6 Panelli Teşhis Panosu Kaydedildi: ciktilar/ood_selective_prediction_paneli.png
```

- **+21.47% AUROC Üstünlüğü:** Enerji skoru, Softmax MSP'ye kıyasla dağılım dışı verileri %87.89 AUROC ile başarıyla ayrıştırmıştır.
- **Birim Test Güvencesi:** [`testler/test_ood.py`](testler/test_ood.py) altındaki **8/8 birim test %100 PASSED (3.71s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/ood_selective_prediction_paneli.png`](ciktilar/ood_selective_prediction_paneli.png) konumundadır:

1. **Enerji Tabanlı OOD ve Çekimserlik Akışı:** $S(x) \ge \gamma$ kabul vs $S(x) < \gamma$ çekimserlik akış diyagramı.
2. **ID vs OOD Enerji Skoru Dağılımı:** ID (Yeşil) ve OOD (Kırmızı) histogram ayrımı ve $\gamma$ eşik çizgisi.
3. **OOD Tespit ROC Eğrisi:** Enerji Skoru (%87.89 AUROC) vs Softmax MSP (%66.42 AUROC).
4. **Kapsam (Coverage) vs Doğruluk/Risk Dengesi:** Eşik sıkılaştırıldıkça riskin sıfıra inme eğrisi.
5. **Seçici Çekimserlik Hata Azaltımı:** Filtresiz vs Seçici tahmin hata oranı.
6. **OOD & Selective Prediction SWOT Karar Matrisi:** Endüstriyel karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Logitler yerine derin modelin son katman öznitelik vektörleri ($h \in \mathbb{R}^D$) üzerinden **Mahalanobis Mesafesi Tabanlı OOD Dedektörü (Lee et al. NeurIPS 2018)** yazınız.

```python
import torch
import numpy as np

class MahalanobisOODDedektoru:
    """Lee et al. (2018): Feature-space class-conditional Mahalanobis distance"""
    def __init__(self):
        self.sinif_ortalamalari = {}
        self.ortak_kovaryans_tersi = None

    def fit(self, id_oznitelikler: torch.Tensor, id_etiketler: torch.Tensor):
        siniflar = torch.unique(id_etiketler)
        kovaryans_toplam = torch.zeros(id_oznitelikler.size(1), id_oznitelikler.size(1))
        
        for c in siniflar:
            feat_c = id_oznitelikler[id_etiketler == c]
            mu_c = feat_c.mean(dim=0)
            self.sinif_ortalamalari[c.item()] = mu_c
            fark = feat_c - mu_c
            kovaryans_toplam += torch.mm(fark.t(), fark)
            
        kovaryans = kovaryans_toplam / id_oznitelikler.size(0)
        # Ters kovaryans (Precision Matrix)
        self.ortak_kovaryans_tersi = torch.pinverse(kovaryans)

    def skor_hesapla(self, test_oznitelikler: torch.Tensor) -> torch.Tensor:
        # En yakın sınıf merkezine olan minimum Mahalanobis mesafesi (Negatif skor: ID yüksek)
        skorlar = []
        for x in test_oznitelikler:
            min_dist = float("inf")
            for mu in self.sinif_ortalamalari.values():
                fark = (x - mu).unsqueeze(0)
                dist = torch.mm(torch.mm(fark, self.ortak_kovaryans_tersi), fark.t()).item()
                min_dist = min(min_dist, dist)
            skorlar.append(-min_dist)
        return torch.tensor(skorlar)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden Softmax Maksimum Olasılığı (MSP) dağılım dışı (OOD) girdilerde başarısız olurken, Serbest Enerji Skoru (Free Energy) çok daha yüksek AUROC ile OOD ayrımı yapabilir?

> **Mentor Cevabı:**
> 1. **Normalizasyon Kısıtının Bilgi Silmesi:** Softmax fonksiyonu $\frac{\exp(z_i)}{\sum_j \exp(z_j)}$ paydası nedeniyle göreli olasılık hesaplar. Eğer modele tamamen anlamsız bir gürültü görseli verilirse ve tüm logitler çok küçük çıkarsa (ör. $z = [-8.0, -7.5, -8.1]$), Softmax bu logitleri normalize ederek en yüksek sınıfa %40-%60 gibi yüksek bir olasılık atar. Softmax, logitlerin mutlak genliğini (magnitude) kaybeder.
> 2. **Enerjinin Doğrusal Yoğunluk Uyumu:** Serbest enerji $E(x) = -T \log \sum_k \exp(z_k / T)$ ise hiçbir normalizasyon paydasına sahip değildir; doğrudan logitlerin genlik toplamı ile ölçeklenir. ID örneklerde güçlü aktivasyonlar $E(x)$'i çok negatif (yani $-E(x)$ skorunu çok yüksek) yaparken, anlamsız OOD örneklerde logitler sönük kalır ve enerji skoru tabana çöker. Bu sayede enerji skoru gerçek veri yoğunluğunu ($p(x)$) doğrudan modeller.

---

### 📌 Git & Yol Haritası Güncellemesi:
- **Tamamlanan:** Gün 85 (`day-85-ood-selective-prediction`) başarıyla tamamlandı.
- **Sıradaki Gün:** **Day 86: Görsel Bozulmalar (Bulanıklık/Gürültü) Altında Model Dayanıklılığı & Domain Shift (`day-86-robustness-domain-shift`)**.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
