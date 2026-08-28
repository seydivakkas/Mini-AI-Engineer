# Day 43: Veri Kayması (Data Drift) Tespiti, Kolmogorov-Smirnov Testi ve Wasserstein Mesafesi (NumPy Data Drift Detector)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 43. gününde geliştirilen **Endüstriyel Veri Kayması (Data Drift) ve Dağılım Bozulması Teftiş Motorudur**. Üretim ortamında çalışan makine öğrenimi modellerinin eğitim verisi ($P_{\text{ref}}$) ile canlı üretim trafiği ($Q_{\text{prod}}$) arasındaki istatistiksel sapmaları parametrik olmayan **2-Örneklemli Kolmogorov-Smirnov (KS) Testi**, **1D Wasserstein (Earth Mover's) Mesafesi** ve **Nüfus Kararlılık İndeksi (PSI)** ile ölçerek otomatik **Yeniden Eğitim (Retraining)** alarmları tetikler.

---

## 📖 Mentorluk Dersi ve İstatistiksel MLOps Teorisı

### 1. Üretimde Veri Kayması (Covariate Shift) Neden Oluşur?

Bir makine öğrenimi modeli eğitildikten sonra performansı zamanla düşer (Model Decay). Bunun başlıca sebebi **Kovaryans Kaymasıdır (Covariate Shift)**: $P_{\text{train}}(X) \ne P_{\text{prod}}(X)$.
- **Fiziksel Nedenler:** Sensör kalibrasyon kayıpları, mevsimsel sıcaklık/nem değişimleri, donanım yıpranması.
- **Davranışsal Nedenler:** Tüketici alışkanlıklarındaki ani değişimler, enflasyon/fiyat dalgalanmaları, pazar dinamikleri.

```
                           ┌──────────────────────────────────────────────────────────┐
                           │      CANLI ÜRETİM TRAFİĞİ GİRDİLERİ (Q_prod)             │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      KSVeWassersteinHesaplayici (İstatistiksel Ölçüm Motoru)                      │
    │  - 2-Örneklemli Kolmogorov-Smirnov: D_KS = sup_x |F_ref(x) - F_prod(x)|, p-değeri (alpha=0.05)   │
    │  - 1D Wasserstein Mesafesi (EMD)  : W_1(P, Q) = int |F_ref(x) - F_prod(x)| dx                   │
    │  - Nüfus Kararlılık İndeksi (PSI) : sum (Actual% - Expected%) * ln(Actual% / Expected%)          │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                    VeriKaymasiDedektoru (Çok Boyutlu MLOps Karar Motoru)                          │
    └──────┬────────────────────────────────────┬───────────────────────────────────────────────┬───────┘
           │                                    │                                               │
           ▼                                    ▼                                               ▼
┌──────────────────────────┐ ┌──────────────────────────────────────┐ ┌─────────────────────────────────┐
│ 1. STABİL NORMAL         │ │ 2. ORTA DÜZEY UYARI                  │ │ 3. KRİTİK DRIFT ALARMI          │
│ - p >= 0.05 & PSI < 0.1  │ │ - p < 0.05 veya 0.1 <= PSI < 0.2     │ │ - p < 0.01 veya PSI >= 0.20     │
│ - MLOps: EYLEM GEREKMEZ  │ │ - MLOps: SIKILAŞTIRILMIŞ İZLEME      │ │ - MLOps: YENİDEN EĞİTİMİ TETİKLE│
└──────────────────────────┘ └──────────────────────────────────────┘ └─────────────────────────────────┘
```

---

### 2. İstatistiksel Testlerin Matematiksel Formülasyonları

#### A. 2-Örneklemli Kolmogorov-Smirnov (KS) Testi
İki bağımsız sürekli dağılımın kümülatif olasılık fonksiyonları (CDF) arasındaki en büyük dikey mesafeyi hesaplar:
$$D_{\text{KS}} = \sup_{x} \left| F_{\text{ref}}(x) - F_{\text{prod}}(x) \right|$$
Asimptotik $p$-değeri ile $H_0$ (iki örneklem aynı dağılımdan gelmektedir) hipotezi test edilir. $p < 0.05$ ise $H_0$ reddedilir ve veri kayması onaylanır.

#### B. 1D Wasserstein Mesafesi (Earth Mover's Distance - $W_1$)
Bir olasılık dağılımını diğerine dönüştürmek için gereken asgari taşıma işini (optimal transport work) ölçer:
$$W_1(P, Q) = \int_{-\infty}^{\infty} \left| F_{\text{ref}}(x) - F_{\text{prod}}(x) \right| dx$$
Sıralı örneklemler için $O(N \log N)$ karmaşıklığında doğrudan hesaplanır:
$$W_1(u, v) = \frac{1}{N} \sum_{i=1}^N |u_{(i)} - v_{(i)}|$$

#### C. Nüfus Kararlılık İndeksi (Population Stability Index - PSI)
Referans dağılımın $K=10$ yüzdelik diliminde (deciles) beklenen oran ($E_k$) ile canlı dağılımdaki gerçek oran ($A_k$) arasındaki bağıl entropiyi ölçer:
$$\text{PSI} = \sum_{k=1}^K (A_k - E_k) \times \ln\left(\frac{A_k + \epsilon}{E_k + \epsilon}\right)$$
- $\text{PSI} < 0.10$: Dağılım Kararlı (No Drift).
- $0.10 \le \text{PSI} < 0.20$: Orta Düzey Değişim (Moderate Warning).
- $\text{PSI} \ge 0.20$: Belirgin ve Kritik Veri Kayması (Critical Retraining Trigger).

---

## 🛠️ Dizin Yapısı

```
day-43-numpy-data-drift-detector/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # 3 üretim senaryosunu teftiş eden ana simülasyon
├── README.md                        # 220+ satır teorik ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── dagilim_olcer.py             # KS, Wasserstein ve PSI hesaplama çekirdeği
│   ├── kayma_tespitci.py            # Çok öznitelikli MLOps veri kayması dedektörü
│   └── gorsellestirici.py           # 6 panelli teşhis panosu (Drift Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_data_drift.py           # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── veri_kaymasi_paneli.png      # 6 panelli yüksek çözünürlüklü teşhis panosu
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

## 📊 3 Üretim Senaryosu ve MLOps Alarmları

| Senaryo | Dağılım Davranışı | Ölçülen Değerler (Sıcaklık) | MLOps Kararı & Aksiyon |
|---|---|---|---|
| **1. Stabil Trafik** | Normal $\mathcal{N}(45, 2.5^2)$ | $D_{\text{KS}}=0.038, p=0.74, \text{PSI}=0.012$ | **DAGILIMLAR_KARARLI_NORMAL** (Eylem Gerekmez) |
| **2. Kademeli Kayma** | Hafif Artış $\mathcal{N}(46.2, 2.6^2)$ | $D_{\text{KS}}=0.210, p=0.0001, \text{PSI}=0.145$ | **ORTA_DUZEY_KAYMA_UYARISI** (Sıkı Takip) |
| **3. Şiddetli Bozulma** | Radikal Artış $\mathcal{N}(54.5, 4.0^2)$ | $D_{\text{KS}}=0.912, p=0.0000, \text{PSI}=3.850$ | **KRITIK_VERI_KAYMASI_ALARM** (Yeniden Eğitimi Tetikle) |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/kayma_tespitci.py` içerisine MLOps izleme platformları (Prometheus / Grafana) için metrik ihraç eden bir **"Prometheus Drift Metric Exporter"** fonksiyonu eklemek.

**Tamamlanan Çözüm:**
```python
def prometheus_metrikleri_uret(genel_rapor: dict) -> str:
    lines = [
        "# HELP mlops_data_drift_ratio Oran of drifted features",
        "# TYPE mlops_data_drift_ratio gauge",
        f"mlops_data_drift_ratio {genel_rapor['kayma_orani'] / 100.0}",
        "# HELP mlops_critical_drift_alarm Binary alarm status (0 or 1)",
        "# TYPE mlops_critical_drift_alarm gauge",
        f"mlops_critical_drift_alarm {1 if genel_rapor['alarm_verildi'] else 0}"
    ]
    for oz, stat in genel_rapor["oznitelikler"].items():
        lines.append(f'mlops_feature_psi_score{{feature="{oz}"}} {stat["psi_skoru"]}')
        lines.append(f'mlops_feature_ks_stat{{feature="{oz}"}} {stat["ks_istatistigi"]}')
    return "\n".join(lines)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden sadece p-değerine bakarak model yeniden eğitim (retraining) kararı vermek tehlikelidir ve $p$-değeri ile birlikte **Wasserstein Mesafesi ($W_1$)** ve **PSI (Population Stability Index)** gibi büyüklük/etki metriklerine bakmak neden zorunludur?

> **Mentor Cevabı:**
> 1. **Büyük Örneklem Paradoksu ($p$-value Hacking):** KS testi gibi hipotez testlerinde örneklem boyutu büyüdükçe ($N > 100.000$), modelin tahmin doğruluğunu zerre kadar etkilemeyecek minik bir dağılım farkı dahi matematiksel olarak $p < 0.0001$ çıkar. Bu durum modellerin gereksiz yere her gün pahalı GPU kaynaklarıyla yeniden eğitilmesine yol açar.
> 2. **Etki Büyüklüğü (Effect Size):** Wasserstein Mesafesi ve PSI, farkın istatistiksel olasılığını değil **gerçek fiziksel sapma büyüklüğünü** ölçer. PSI $< 0.10$ olduğu sürece $p$-değeri küçük çıksa bile veri kaymasının model tahminlerine etkisi önemsiz kabul edilir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
