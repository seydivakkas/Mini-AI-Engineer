# Day 54: Dijital Adli Bilişim, Error Level Analysis (ELA) ve Görsel Manipülasyon Tespiti (Digital Image Forensics & Forgery Inspector)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![Pillow](https://img.shields.io/badge/Pillow-10.0+-yellowgreen.svg?style=flat-square)](https://python-pillow.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 54. gününde geliştirilen **Dijital Medya Adli Bilişimi (Digital Image Forensics), Error Level Analysis (ELA) ve Görsel Sahtecilik/Tahrifat Tespit Motorudur**. Sigorta dolandırıcılığı, sahte fatura/belge tespiti, dijital kanıt doğrulama ve medya manipülasyonu analizlerinde JPEG Ayrık Kosinüs Dönüşümü (DCT) sıkıştırma hatalarını ve sensör gürültü süreksizliklerini (PRNU Residuals) inceleyerek kopyala-yapıştır (splicing/copy-move) alanlarını mikron piksel seviyesinde açığa çıkarır.

---

## 📖 Mentorluk Dersi ve Dijital Adli Bilişim Teorisı

### 1. Error Level Analysis (ELA) Çalışma İlkesi

JPEG sıkıştırma algoritması, görüntüyü $8 \times 8$ piksel bloklarına bölerek **Ayrık Kosinüs Dönüşümü (Discrete Cosine Transform - DCT)** ve kayıplı kuantizasyon (Quantization Matrix) uygular:
1. **Sıkıştırma Dengesi (Compression Equilibrium):** Bir fotoğraf ilk kez JPEG olarak kaydedildiğinde, her $8 \times 8$ bloğu o kalite seviyesinde (örneğin $\%85$) belirli bir hata dengesine ulaşır.
2. **Manipülasyon ve Ekleme (Splicing):** Eğer bu görselin üzerine başka bir görselden kırpılmış bir nesne (sahte imza, değiştirilmiş fatura tutarı, sahte damga veya fotomontaj yüz) yapıştırılır ve görsel yeniden kaydedilirse:
   - Orijinal arka plan pikselleri zaten daha önce sıkıştırıldığı için çok az yeni hata üretir.
   - Yeni eklenen veya modifiye edilen pikseller ise ilk kez o sıkıştırma matrisinden geçtiği için **dramatik derecede yüksek hata (yüksek frekans parlama)** üretir.
3. **Bellekte Yeniden Sıkıştırma (In-Memory Recompression):** Görsel sabit bir kalite faktörüyle (örneğin $Q=90$) bellekte (`io.BytesIO`) yeniden sıkıştırılır. Orijinal aday görsel ile aralarındaki mutlak fark $D(x,y)$ kontrast çarpanıyla ($\alpha = 15.0$) büyütüldüğünde sahte bölgeler parıldayan parlak lekelere dönüşür.

```
                           ┌──────────────────────────────────────────────────────────┐
                           │            ŞÜPHELİ ADAY GÖRSEL / BELGE                   │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      ErrorLevelAnalizoru (Bellek İçi JPEG Yeniden Sıkıştırma)                     │
    │  - Q=90 Kalitesinde Sıkıştırma -> Orijinal ile Mutlak Fark (ELA = min(255, 15 * |I - I_resaved|))│
    │  - İstatistiki Z-Skor Anomali Haritası Çıkarılır                                                  │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      GurultuAdliAnalizoru (Sensör Gürültüsü & PRNU Kalıntısı)                     │
    │  - Medyan Filtreleme ile Yüksek Frekans Sensör Gürültüsü (R = |I - Median(I)|) Ayrıştırılır        │
    │  - Blok Bazlı Lokal Varyans ve Değişim Katsayısı (CV) ile Gürültü Tutarsızlığı Ölçülür            │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      AdliTeftisMotoru (Çoklu Metrik Füzyonu & Karar Motoru)                       │
    │  - Şüpheli Alan Konturları Çıkarılır, Kırmızı Bounding Box ile Konumlandırılır                    │
    │  - Sahtecilik Güven Skoru (%): AUTHENTIC (<25), SUSPICIOUS (25-60), TAMPERED (>=60)               │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 6-PANELLİ ADLİ BİLİŞİM VE ELA TEŞHİS PANOSU (Day 54)                              │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Hata Seviyesi Analizi (ELA)** | *Error Level Analysis* | Görselin belirli bir kalitede yeniden JPEG kaydedilmesiyle oluşan hata matrisini inceleyerek montaj ve ekleme yapılan bölgeleri tespit etme. |
| **Fourier Frekans Analizi (FFT)** | *Fast Fourier Transform Analysis* | Görselin uzamsal frekans spektrumunu çıkararak yapay zeka (GAN/Difüzyon) üretim izlerini ve ızgara anomalilerini saptama. |
| **JPEG Sıkıştırma Artefaktları** | *JPEG Compression Artifacts* | $8 \times 8$ bloklar halindeki DCT dönüşüm izlerinin manipüle edilmiş alanlarda süreksizlik göstermesi. |
| **Gürültü Kalıntısı (Noise Residual)** | *Noise Residual Extraction* | Yüksek geçiren filtreleme ile görseldeki sensör gürültüsü örüntüsünün tutarlılığını analiz etme. |

---

## 2. Matematiksel Formülasyonlar

#### A. Error Level Analysis (ELA) Hata Formülasyonu
$$D(x,y) = |I_{\text{aday}}(x,y) - I_{\text{resaved}}(x,y, Q=90)|$$
$$\text{ELA}(x,y) = \min\left(255, \quad \alpha \cdot D(x,y)\right) \quad (\alpha = 15.0)$$

#### B. Sensör Gürültü Kalıntısı (Sensor Noise Residual)
$$R(x,y) = |I(x,y) - \text{MedianFilter}(I(x,y), k=3)|$$

#### C. Lokal Gürültü Değişim Katsayısı (Noise Inconsistency CV)
$$CV = \frac{\sigma_{\text{varyans}}}{\mu_{\text{varyans}} + \epsilon} \quad (\text{Orijinal: } CV < 0.45, \quad \text{Manipüle: } CV > 0.70)$$

#### D. Bileşik Sahtecilik Güven Skoru (Forgery Confidence Score)
$$S_{\text{forgery}} = \min\left(100, \quad w_1 \cdot \text{AlanOranı} + w_2 \cdot (CV - 0.45) \cdot 60 + w_3 \cdot N_{\text{bölge}} \cdot 8\right)$$

---

## 🛠️ Dizin Yapısı

```
day-54-image-forensics-inspector/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # pillow, opencv-python, numpy, scipy, scikit-image, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca adli tahrifat simülasyonu ve ELA analiz betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── ela_analizoru.py             # ErrorLevelAnalizoru (JPEG bellek sıkıştırması ve ELA ısı haritası)
│   ├── gurultu_adli_analizor.py     # GurultuAdliAnalizoru (PRNU gürültü kalıntısı ve lokal varyans CV)
│   ├── adli_teftis_motoru.py        # AdliTeftisMotoru (Çoklu metrik füzyonu, sahtecilik güven skoru)
│   └── gorsellestirici.py           # 6-Panelli Adli Bilişim Teşhis Panosu (Forensics & ELA Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_image_forensics.py      # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── adli_teftis_paneli.png       # 6 panelli yüksek çözünürlüklü teşhis panosu
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

## 📊 Adli Risk ve Sahtecilik Karar Matrisi

| Manipülasyon Skoru | Karar Kodu | Risk Seviyesi | Adli Bulgular | Önerilen Aksiyon |
|---|---|---|---|---|
| **$0 - 24\%$** | `ORİJİNAL` | **LOW** | Homojen JPEG hata dengesi, sürekli sensör gürültüsü. | Doğrulanmış Belge / Onay |
| **$25 - 59\%$** | `ŞÜPHELİ` | **WARNING** | Hafif ELA varyans kayması, bölgesel gürültü yumuşaması. | Manuel Uzman İncelemesi |
| **$\ge 60\%$** | `MANİPÜLE EDİLMİŞ` | **CRITICAL_REJECT** | Yüksek ELA parlama tepesi, belirgin kopyala-yapıştır (splicing). | Sahtecilik Raporu / Red |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Klasördeki yüzlerce aday belgeyi hızlıca tarayarak yalnızca şüpheli ($S \ge 25$) veya manipüle edilmiş ($S \ge 60$) dosyaları tespit edip JSON adli raporu üreten **"Batch Forensic ELA Scanner"** geliştirmek.

**Tamamlanan Çözüm:**
```python
def toplu_adli_tarama(gorsel_listesi: list) -> list:
    """Tüm aday görselleri ELA ve gürültü analizinden geçirerek adli rapor listesi döner."""
    motor = AdliTeftisMotoru(ela_kalite=90, z_esigi=2.2)
    rapor = []
    for idx, img in enumerate(gorsel_listesi):
        res = motor.teftis_et(img)
        rapor.append({
            "dosya_id": idx + 1,
            "skor": res["manipulasyon_skoru"],
            "karar": res["karar"],
            "supheli_alan_adedi": len(res["supheli_bolgeler"])
        })
    return rapor
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Bir dolandırıcı, görsel üzerinde Photoshop ile tahrifat yaptıktan sonra ELA tespitinden kaçmak için tüm görseli tekrar $\%70$ gibi düşük bir kalitede JPEG olarak yeniden kaydederse (re-saving attack), ELA ve sensör gürültüsü analizi bu hileyi nasıl çözer?

> **Mentor Cevabı:**
> 1. **Çoklu Sıkıştırma İzi (Double Compression Artifacts):** Bir görsel iki kez farklı JPEG kalitelerinde sıkıştırıldığında DCT katsayıları histogramında periyodik "çukur ve tepe" (periodicity / grid artifacts) bozulmaları oluşur.
> 2. **Gürültü Süreksizliği (Noise Discontinuity):** ELA farkı genel olarak homojenleşse bile, harici eklenen nesnenin pikselleri orijinal kamera sensörünün PRNU (Photo-Response Non-Uniformity) gürültü parmak izine sahip olamaz. `GurultuAdliAnalizoru`'ndaki medyan filtre kalıntısı ve lokal varyans ($CV$) analizimiz, eklenen bölgedeki gürültü yokluğunu veya gürültü frekans uyumsuzluğunu anında yakalar.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
