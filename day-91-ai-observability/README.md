# Day 91: Canlı AI Sistemlerinde Gözlemlenebilirlik: Gecikme, Hacim ve Veri Kayması İzleme

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Observability: Prometheus / Evidently Standard](https://img.shields.io/badge/Observability-KS_Test_%26_PSI_Drift-darkgreen.svg?style=flat-square)](#1-🎯-günün-konusu--teorikmatematiksel-derinlik)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_observability.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin onuncu gününde; üretim ortamındaki (production) yapay zeka modellerinin canlı performansını, servis seviyesi gecikmelerini (P50, P95, P99 SLA), anlık işlem hacmini (Throughput/RPS) ve girdi/çıktı dağılımlarındaki istatistiksel kaymaları (**Kolmogorov-Smirnov Testi**, **Population Stability Index - PSI**, **Wasserstein Distance**) anlık olarak denetleyen tam donanımlı bir **AI Observability (Yapay Zeka Gözlemlenebilirlik) Motoru** inşa ediyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Yapay zeka modelleri canlıya alındıktan sonra performansları statik kalmaz. Geleneksel yazılımların aksine, AI modelleri "sessizce başarısız olur" (silent failure): Kod çökmez, HTTP 200 döner, ancak tahmin kalitesi felaket derecede düşebilir. Bu durumu önlemek için AI Gözlemlenebilirlik sistemi şu 3 temel sütun üzerinde inşa edilir:

1. **Servis Seviyesi ve Gecikme Gözlemlenebilirliği (Service-Level Metrics):**
   Ortalama gecikme yanıltıcıdır. Bir web servisinin SLA garantisi vermesi için $P_{95}$ ve $P_{99}$ gibi kuyruk gecikmelerinin (tail latency) anlık izlenmesi şarttır.
2. **İstatistiksel Veri Kayması Tespiti (Data & Feature Drift):**
   Eğitim verisindeki dağılım ($P_{\text{train}}(X)$) ile canlı ortamdaki dağılım ($P_{\text{prod}}(X)$) zamanla ayrışır (Covariate Shift). Bu kayma **Kolmogorov-Smirnov (KS-Test)** ve **Population Stability Index (PSI)** ile matematiksel olarak yakalanır.
3. **Kavram ve Tahmin Kayması (Concept & Prediction Drift):**
   Modelin tahmin güven skorlarının ($\max P(Y|X)$) veya tahmin ettiği sınıfların dağılımının zaman içindeki erozyonu, etiketler (ground truth) henüz elimizde olmasa dahi modelin yeniden eğitilmesi (re-training trigger) gerektiğini haber verir.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Sessiz Model Çöküşlerini (Silent Model Failures) Anında Yakalama:**
  Kamera sensörü kirlendiğinde veya kullanıcı davranışı değiştiğinde hata logu oluşmaz. Drift dedektörü istatistiksel sapmayı milisaniyeler içinde fark edip alarm üretir.
- **SLA İhlallerini ve Darboğazları Önceden Haber Verme:**
  Kayan pencere histogramları sayesinde kuyruk gecikmesi eşikleri aşıldığında dinamik kaynak artırımı (Auto-scaling) tetiklenebilir.
- **Gereksiz Yeniden Eğitim Maliyetini Engelleme:**
  Modeli her gün körü körüne yeniden eğitmek yerine sadece $\text{PSI} \ge 0.20$ olduğunda tetikleme yaparak bulut maliyetlerini optimize eder.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Gecikmeli Zemin Gerçekliği (Delayed Ground Truth):**
  Canlı ortamda gerçek etiketler ($y_{\text{true}}$) anında gelmez (bazen haftalar sürer). Bu sebeple doğrudan Accuracy/F1 ölçülemez; vekil metrik olarak dağılım kaymalarına güvenilmek zorundadır.
- **Yüksek Boyutluluk Laneti (Curse of Dimensionality):**
  Yüzlerce gizli öznitelik için tek tek KS-test çalıştırmak Çoklu Hipotez Testi (Multiple Hypothesis Testing) sorununa yol açabilir (Bonferroni düzeltmesi gerekebilir).

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Gözlemlenebilirlik Platformu | Desteklenen Metrikler | Drift Algoritmaları | Entegrasyon Kolaylığı | Kaynak Tüketimi |
|---|---|---|---|---|
| **Bizim Geliştirdiğimiz Motor** | **P50/P95/P99, RPS, PSI, KS, EMD** | **KS-Test, PSI, Wasserstein** | **Doğrudan PyTorch / FastAPI** | **Ultra Hafif (In-Memory)** |
| **Prometheus + Grafana** | Sistem/Servis Metrikleri (Sayaç, Histogram) | Yok (Eklenti gerektirir) | Standart Altyapı | Düşük |
| **Evidently AI** | Kapsamlı ML Raporları, Data Drift | KS, PSI, Chi-Square, Jensen-Shannon | Python Kütüphanesi | Orta |
| **Arize AI / WhyLabs** | Kurumsal AI Observability, Embedding Drift | UMAP, PCA, PSI, KS | SaaS / Bulut | Yüksek (Maliyet) |
| **Weights & Biases (W&B Prompts)** | LLM / Deney Takibi | Token/Cost Odaklı | Kolay | Bulut Bağımlı |

---

## 📐 Matematiksel Formülasyon

### 1. Kolmogorov-Smirnov İki Örneklem Testi (KS-Test)
Referans kümülatif dağılım fonksiyonu $F_{\text{ref}}(x)$ ile canlı kümülatif dağılım fonksiyonu $F_{\text{prod}}(x)$ arasındaki maksimum dikey mesafe:

$$D = \sup_x |F_{\text{ref}}(x) - F_{\text{prod}}(x)|$$

İki dağılımın aynı olduğu sıfır hipotezi ($H_0$), $p$-değeri anlamlılık düzeyi $\alpha$'dan (ör. 0.05) küçük olduğunda reddedilir ($p < \alpha \implies \text{Drift Var}$).

### 2. Population Stability Index (PSI)
Sürekli bir özniteliğin referans dağılımdaki oranları $P_k$ ve canlı dağılımdaki oranları $Q_k$ ($k=1 \dots K$ kutuları) olmak üzere:

$$\text{PSI} = \sum_{k=1}^K (Q_k - P_k) \cdot \ln\left(\frac{Q_k}{P_k}\right)$$

- **$\text{PSI} < 0.10$:** Dağılım Kararlı (Değişiklik yok)
- **$0.10 \le \text{PSI} < 0.20$:** Hafif Kayma (Dikkat / İzleme)
- **$\text{PSI} \ge 0.20$:** Kritik Kayma (Alarm / Model Yeniden Eğitilmeli)

### 3. Wasserstein Mesafesi (Earth Mover's Distance - $\mathcal{W}_1$)

$$\mathcal{W}_1(u, v) = \int_{-\infty}^{\infty} |U(x) - V(x)| \, dx$$

Burada $U(x)$ ve $V(x)$ sırasıyla referans ve canlı dağılımların kümülatif dağılım fonksiyonlarıdır.

---

## 📖 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Terim | Tanım | Endüstriyel Önemi |
|---|---|---|
| **AI Observability** | Canlıdaki modelin gecikme, throughput, girdi verisi ve tahmin dağılımlarını gerçek zamanlı izleyip anomali tespit etme pratiği. | Üretim modellerinin güvenilirliğini, SLA uyumunu ve iş değerini korur. |
| **Data Drift (Covariate Shift)** | Girdi verisi dağılımının ($P(X)$) zaman içinde eğitim ortamından farklılaşması. | Modelin doğruluğunu düşüren en yaygın canlı ortam problemidir. |
| **Concept Drift** | Girdi ile çıktı arasındaki ilişkinin ($P(Y|X)$) değişmesi (ör. tüketici alışkanlıklarının değişmesi). | Modelin tamamen geçersiz hale gelmesine yol açar. |
| **Population Stability Index (PSI)** | İki olasılık dağılımı arasındaki ayrışmayı ölçen simetrik göreli entropi metriği. | Bankacılık, kredi skorlama ve ML model izlemede altın standarttır. |
| **Kolmogorov-Smirnov (KS) Test** | İki sürekli dağılımın kümülatif fonksiyonları arasındaki maksimum farkı test eden parametrik olmayan test. | Veri dağılım kaymalarını matematiksel güven aralığıyla doğrular. |
| **Tail Latency (P95 / P99)** | İsteklerin en yavaş %5'lik ve %1'lik diliminin maruz kaldığı gecikme süresi. | Kullanıcı deneyimi ve katı SLA sözleşmeleri için kritik esastır. |
| **Sliding Window Buffer** | Canlı gelen son $N$ adet isteği bellekte tutarak anlık metrik ve drift hesaplayan kayan pencere. | Eski verilerin etkisini azaltıp anlık bozulmaları yakalar. |
| **Silent Model Failure** | Modelin HTTP 200 ile yanıt üretmesine rağmen yanlış veya alakasız tahminler üretmesi durumu. | Gözlemlenebilirlik olmadan fark edilmesi imkansız olan felaket senaryosu. |
| **Throughput (RPS)** | Sistemin saniyede başarıyla işlediği istek sayısı (Requests Per Second). | Sunucu kapasite planlaması ve yük dengeleme için temel metriktir. |
| **Earth Mover's Distance (Wasserstein)** | Bir dağılımı diğerine dönüştürmek için gereken minimum taşıma işini temsil eden metrik. | Aykırı değerlere ve sürekli kaymalara karşı çok dayanıklıdır. |

---

## 📊 SWOT Analizi (Karar Matrisi)

```
┌──────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│                   GÜÇLÜ YÖNLER (S)               │                  ZAYIF YÖNLER (W)                │
│ • Etiket bağımsız (Unlabeled) anlık drift tespiti│ • Çoklu öznitelikte false-positive alarm riski.  │
│ • P50/P95/P99 ile katı SLA denetimi.             │ • Canlı zemin gerçeği (ground truth) yokluğu.   │
│ • Ultra hafif, sıfır harici bağımlılık.          │ • Kayan pencere boyutu seçimi hassasiyeti.       │
├──────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│                  FIRSATLAR (O)                   │                   TEHDİTLER (T)                  │
│ • Otomatik model yeniden eğitim (Continuous Train│ • Ani geçici gürültülerin yanlış alarm üretmesi. │
│ • Prometheus / Grafana / Datadog entegrasyonu.   │ • Yüksek trafikte CPU/Bellek profil yükü.        │
│ • SLA ihlallerinde dinamik auto-scale tetikleme. │ • Dağılım tipi değişiminde parametre uyumsuzluğu.│
└──────────────────────────────────────────────────┴──────────────────────────────────────────────────┘
```

---

## 🧪 Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

### 🎯 Görev: CUSUM (Cumulative Sum) ile Anlık Gecikme Kayması Dedektörü
Ortalama gecikmedeki kalıcı artışları tekil istek bazında biriktirerek anında alarm üreten **CUSUM (Cumulative Sum Control Chart)** algoritmasını uygulayın.

### 💡 Eksiksiz Çalışan Çözüm Kodu:

```python
import numpy as np

class CUSUMGecikmeDedektoru:
    """Gecikmedeki ani ve kalıcı artışları tespit eden CUSUM kontrol algoritması."""
    def __init__(self, hedef_ortalama_ms: float = 15.0, tolerans_k: float = 5.0, esik_h: float = 20.0):
        self.hedef_ortalama_ms = hedef_ortalama_ms
        self.tolerans_k = tolerans_k
        self.esik_h = esik_h
        self.s_pozitif = 0.0

    def guncelle(self, yeni_gecikme_ms: float) -> bool:
        """Yeni bir gecikme kaydı ekler. Alarm verirse True döner."""
        sapma = yeni_gecikme_ms - self.hedef_ortalama_ms - self.tolerans_k
        self.s_pozitif = max(0.0, self.s_pozitif + sapma)
        if self.s_pozitif > self.esik_h:
            return True  # ALARM: Kalıcı gecikme artışı var!
        return False

# Test ve Doğrulama
dedektor = CUSUMGecikmeDedektoru(hedef_ortalama_ms=15.0, tolerans_k=3.0, esik_h=15.0)
normal_gecikmeler = [14.0, 16.0, 15.5, 14.8, 15.2]
for g in normal_gecikmeler:
    assert dedektor.guncelle(g) is False

bozuk_gecikmeler = [28.0, 32.0, 35.0, 40.0]
alarmlar = [dedektor.guncelle(g) for g in bozuk_gecikmeler]
assert any(alarmlar) is True
print("✓ CUSUM Dedektörü Başarıyla Doğrulandı!")
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
> *"Canlı ortamda KS-Testi özniteliklerin %80'inde $p < 0.001$ vererek veri kayması (Drift) olduğunu söylüyor. Ancak Population Stability Index (PSI) değerleri tüm özniteliklerde $0.04$ civarında (yani kararlı) çıkıyor. Bu iki metrik neden çelişir ve kıdemli bir MLOps mühendisi olarak hangi kararı almalısınız?"*

### 💡 Mentorluk Açıklaması ve Çözüm:
Bu durum, **Örneklem Büyüklüğünün (Sample Size) İstatistiksel Güç Üzerindeki Etkisinden** kaynaklanan klasik bir olgudur:

1. **KS-Test'in Hassasiyeti:**
   KS-Testi parametrik olmayan çok güçlü bir hipotez testidir. Örneklem boyutu $N$ çok büyüdüğünde (ör. $N > 100.000$), modelin kararlarını ve tahmin kalitesini pratikte zerre kadar etkilemeyecek mikroskobik dağılım farklarında bile $p$-değerini sıfıra yaklaştırır ($p \to 0$).
2. **PSI'nin Pratik Büyüklük Odaklılığı:**
   PSI, frekans kutuları (bins) üzerindeki göreli entropi farkını ölçtüğü için örneklem sayısından bağımsız olarak dağılımın "şekilsel büyüklüğünü" değerlendirir. $\text{PSI} = 0.04$, pratik olarak dağılımın mükemmel derecede kararlı olduğunu gösterir.
3. **Mühendislik Kararı:**
   Bu senaryoda **yanlış alarm (false alarm)** verilmemelidir. Model hemen yeniden eğitime sokulmaz. KS-Test $p$-değeri yerine $D$-istatistiğinin mutlak büyüklüğü ve PSI skoru ($< 0.10$) baz alınarak sistemin sağlıklı olduğu kabul edilir; loglara bilgilendirme (INFO) notu düşülür.
