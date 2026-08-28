# Day 07: İstatistiksel ve Makine Öğrenmesi Tabanlı Aykırı Değer Tespiti (Outlier Detection)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-f89939.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; bilgisayarlı görü ve yapay zeka sistemlerinde kamera sensör verilerindeki, öznitelik vektörlerindeki ve üretim bandı telemetrisindeki anormal durumları saptamak için **Z-Skoru, Modifiye Z-Skoru (MAD), Tukey IQR, İzolasyon Ormanı (Isolation Forest) ve Lokal Aykırı Faktör (Local Outlier Factor - LOF)** yöntemlerini tek bir çatıda birleştiren ve karşılaştıran kapsamlı bir aykırı değer tespit laboratuvarıdır.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Bilgisayarlı görü ve akıllı fabrika hatlarında kameralar ve optik sensörler sürekli veri üretir:
- Bir sensör aşırı ısınabilir veya ani bir voltaj dalgalanması yaşayabilir.
- Kamera lensine anlık bir toz zerresi veya yağ damlası düşerek piksel parlaklığını uçurabilir.
- Ya da gerçekten **üretim bandında kusurlu bir parça** (defective product) geçiyor olabilir!

Bu noktaları tespit etmek iki açıdan hayatidir:
1. **Model Eğitimi Öncesi (Veri Temizleme):** Yanlış ölçülmüş uç değerler eğitim setinde kalırsa, Sinir Ağlarının veya Regresyon modellerinin gradyanlarını patlatır.
2. **Canlı İzleme (Real-time Anomaly Detection):** Model çalışırken gelen anormal bir parçayı anında yakalayıp bandı durdurmak gerekir.

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Z-Skoru Filtreleme** | *Z-Score Outlier Filtering* | Noktanın ortalamadan kaç standart sapma uzakta olduğunu hesaplayarak $|Z| > 3$ olanları ayıklayan yöntem. |
| **Medyan Mutlak Sapma (MAD)** | *Median Absolute Deviation* | Aykırı değerlerden etkilenmeyen medyan tabanlı sağlam (robust) dağılım genişliği ölçüsü. |
| **Tukey Çitleri** | *Tukey's Fences (IQR Rule)* | $[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]$ aralığının dışında kalan noktaları aykırı değer kabul eden kural. |
| **Isolation Forest** | *Isolation Forest Algorithm* | Aykırı değerlerin karar ağaçlarında rastgele bölmelerle normal noktalara göre çok daha sığ derinliklerde izole edilmesi prensibine dayalı denetimsiz algoritma. |
| **Bulaşma Oranı** | *Contamination Rate* | Veri setinde bulunması beklenen aykırı değer yüzdesini belirten hiperparametre. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. Klasik Z-Skoru vs. Modifiye Z-Skoru (MAD)
- **Klasik Z-Skoru:**
  Verinin ortalamasını ($\mu$) ve standart sapmasını ($\sigma$) kullanarak her noktanın dağılım merkezinden kaç standart sapma uzakta olduğunu ölçer:
  $$Z_i = \frac{x_i - \mu}{\sigma}$$
  *Eşik:* Genellikle $|Z_i| > 3.0$ olan noktalar aykırı kabul edilir.

- **⚠️ Klasik Z'nin Ölümcül Tuzağı (Maskeleme Etkisi - Masking Effect):**
  Eğer veride devasa bir aykırı değer varsa (ör. $[1, 2, 2, 3, 10000]$), bu değer hem $\mu$'yü aşırı yukarı çeker hem de standart sapmayı ($\sigma$) şişirir. Sonuçta o devasa aykırı değerin kendi Z-skoru küçülür ve kendisini gizler!
  
- **Modifiye Z-Skoru (Boris Iglewicz & David Hoaglin):**
  Ortalama yerine **Medyan ($\tilde{x}$)**, standart sapma yerine **Medyan Mutlak Sapma (MAD)** kullanılır:
  $$MAD = \text{median}(|x_i - \tilde{x}|)$$
  $$M_i = \frac{0.6745 \cdot (x_i - \tilde{x})}{MAD + \epsilon}$$
  *(0.6745 katsayısı, normal dağılımda MAD'in standart sapmaya eşitlenmesi için gereken teorik çarpanıdır).*
  *Eşik:* $|M_i| > 3.5$ olan noktalar kesin aykırıdır.

---

#### B. Tukey Çeyrekler Açıklığı (IQR Yöntemi)
Verinin parametrik (normal) bir dağılıma sahip olmasını şart koşmaz:
- $Q_1$: Verinin ilk %25'lik dilimi
- $Q_3$: Verinin ilk %75'lik dilimi
- $IQR = Q_3 - Q_1$
- $\text{Alt Sınır} = Q_1 - 1.5 \times IQR$
- $\text{Üst Sınır} = Q_3 + 1.5 \times IQR$

---

#### C. İzolasyon Ormanı (Isolation Forest)
Fei Tony Liu ve ekibi (2008) tarafından geliştirilen ağaç tabanlı devrimsel bir algoritmadır:
- Normal makine öğrenimi modelleri "normal noktaların nasıl kümelendiğini" öğrenmeye çalışır. İzolasyon Ormanı ise tam tersini yapar: **Aykırıları izole etmeye çalışır!**
- Çok boyutlu uzayda rastgele bir öznitelik ve rastgele bir kesim değeri seçilerek uzay ikiye bölünür.
- **Mantık:** Aykırı noktalar hem sayıca azdır hem de uzayda diğerlerinden uzaktadır. Bu yüzden çok az sayıda rastgele kesimle (ağacın köküne çok yakın, kısa yol uzunluğunda $h(x)$) hemen tek başına bir yaprakta izole olurlar! Normal noktaları izole etmek için ise ağacın derinliklerine inmek gerekir.
- Anomali Skoru: $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$. Skor $1$'e yakınsa nokta kesin anomalidir.

---

#### D. Lokal Aykırı Faktör (Local Outlier Factor - LOF)
Yoğunluk tabanlı (k-NN) bir algoritmadır (Breunig vd., 2000):
- **Küresel Aykırı (Global Outlier):** Tüm uzaydan çok uzakta olan nokta. (Z-Score ve IQR bunu kolayca bulur).
- **Yerel Aykırı (Local Outlier):** Bir veri kümesinde iki farklı üretim hattı olduğunu düşünün: Biri çok sıkışık/yoğun bir küme, diğeri daha seyrek bir küme. Yoğun kümenin hemen birkaç milim dışında duran bir nokta, global sınırlara göre normal görünebilir; fakat **kendi yerel komşularının yoğunluğuna göre anormal derecede seyrektir!**
- LOF, bir noktanın yerel yoğunluğunu $k$-komşularının yerel yoğunluğu ile oranlar. $LOF \approx 1$ ise nokta küme içindedir; $LOF > 1.5$ ise nokta yerel bir anomalidir.

---

### 3. Yöntemlerin Karşılaştırma Matrisi

| Algoritma | Dağılım Varsayımı | Çok Boyut Desteği | Yerel Yoğunluk Hassasiyeti | Hesaplama Hızı |
| :--- | :--- | :--- | :--- | :--- |
| **Z-Skoru** | Normal Dağılım Şart | Zayıf (Eksen Eksen) | Yok (Küresel) | Çok Hızlı ($\mathcal{O}(N)$) |
| **Modifiye Z (MAD)** | Dağılım Bağımsız | Zayıf | Yok | Çok Hızlı ($\mathcal{O}(N \log N)$) |
| **IQR (Tukey)** | Dağılım Bağımsız | Zayıf | Yok | Çok Hızlı ($\mathcal{O}(N \log N)$) |
| **İzolasyon Ormanı** | Dağılım Bağımsız | **Mükemmel ($D > 100$)** | Orta | Hızlı ($\mathcal{O}(N \cdot T)$) |
| **LOF** | Dağılım Bağımsız | Orta ($D < 20$) | **Mükemmel (Yerel Anomali)** | Orta ($\mathcal{O}(N^2)$) |

---

## 📌 Mimari Tasarım ve Akış Şeması

```
                    Ham Sensör / Öznitelik Verisi (N x D)
                                       │
                                       ▼
                       ┌──────────────────────────────┐
                       │   AykiriDegerKarsilastirici  │
                       └───────────────┬──────────────┘
                                       │
        ┌──────────────────┬───────────┴───────────┬──────────────────┐
        ▼                  ▼                       ▼                  ▼
  [Z-Skoru / MAD]     [Tukey IQR]         [İzolasyon Ormanı]        [LOF]
  İstatistiksel       Çeyreklikler        Rastgele Kesimli     k-Komşuluk
  Sapma Sınırı        Sınırı              Karar Ağaçları       Yoğunluk Oranı
        │                  │                       │                  │
        └──────────────────┼───────────────────────┴──────────────────┘
                           ▼
              ┌─────────────────────────┐
              │   Topluluk Mutabakatı   │
              │   (Ensemble Consensus)  │
              │  4/4 Kesin, 3/4 Şüphe   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │       ciktilar/         │
              │ aykiri_deger_karsilastir│
              │          ma.png         │
              └─────────────────────────┘
```

---

## 💻 Konsol Çalıştırma Çıktısı

```text
==========================================================================
>>> AŞAMA 1: Sentetik Endüstriyel Sensör Verisi
==========================================================================
Toplam Örnek Sayısı          : 525
Normal Kümeler (A ve B)       : 500 nokta
Enjekte Edilen Küresel Aykırı : 15 nokta
Enjekte Edilen Yerel Aykırı   : 10 nokta
Öznitelikler                  : [Sensör Sıcaklığı (°C), Titreşim Şiddeti (mm/s)]

==========================================================================
>>> AŞAMA 2: 4 Temel Algoritmanın Çalıştırılması ve Karşılaştırılması
==========================================================================
Yöntem Adı                 | Aykırı Sayısı   | Yüzde Oranı
--------------------------------------------------------------------------
Z-Skoru                    | 13              | %2.48
IQR (Tukey)                | 13              | %2.48
İzolasyon Ormanı           | 27              | %5.14
Lokal Aykırı Faktör (LOF)  | 27              | %5.14
--------------------------------------------------------------------------

==========================================================================
>>> AŞAMA 3: Topluluk Mutabakatı (Ensemble Consensus) Analizi
==========================================================================
  * 4/4 Oy Birliği (Kesin Anomali)     :  13 adet (% 2.48)
  * 3/4 Oy Çokluğu (Yüksek Şüphe)      :   0 adet (% 0.00)
  * 2/4 Kısmi Ayrışma                  :   8 adet (% 1.52)
  * 1/4 Tek Yöntem Uyarısı             :  12 adet (% 2.29)
  * 0/4 Temiz / Normal                 : 492 adet (%93.71)

==========================================================================
>>> AŞAMA 4: 2x2 Karşılaştırmalı Görselleştirme Çıktısı
==========================================================================
[V] Karşılaştırma grafiği başarıyla kaydedildi: aykiri_deger_karsilastirma.png
[V] Tam Dosya Yolu: day-07-outlier-detection/ciktilar/aykiri_deger_karsilastirma.png

[V] Day 7: İstatistiksel ve ML Tabanlı Aykırı Değer Tespiti tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Dinamik Ağırlıklı Topluluk Anomali Skoru (Weighted Ensemble Anomaly Scorer)**

Endüstride tek bir algoritmanın kararına güvenilmez. Farklı algoritmaların anomali olasılık skorları ağırlıklı olarak birleştirilir.

### Görev Tanımı:
[`src/karsilastirma_ve_gorsellestirme.py`](./src/karsilastirma_ve_gorsellestirme.py) içerisine şu fonksiyonu eklemeni bekliyorum:

```python
def agirlikli_anomali_skoru(
    self,
    izolasyon_skorlari: np.ndarray,
    lof_skorlari: np.ndarray,
    agirlik_izolasyon: float = 0.6,
    agirlik_lof: float = 0.4
) -> np.ndarray:
```

### Beklenen Kurallar:
1. İzolasyon ormanının decision function skorlarını $[-1, 1]$ aralığından $[0, 1]$ anomali olasılığına normalize etmeli (küçük değerler yüksek anomali olasılığına dönüşmelidir).
2. LOF'un negatif outlier factor skorlarını $[0, 1]$ anomali olasılığına normalize etmeli.
3. İki skoru belirtilen ağırlıklarla çarparak her nokta için $[0.0, 1.0]$ arasında tek bir **Nihai Anomali Skoru** üretmelidir.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> İki farklı yoğunlukta kümeden oluşan bir veri setinde (örneğin A Kümesi: 1000 noktalı çok sıkışık bir küme, B Kümesi: 100 noktalı seyrek ve geniş bir küme), **İzolasyon Ormanı** ile **Local Outlier Factor (LOF)** algoritmaları karşılaştırıldığında:  
> B kümesinin dış sınırında duran bir nokta için bu iki algoritma neden **tamamen zıt** kararlar verebilir? Hangisi bu noktayı kesin aykırı sayar, hangisi normal kabul eder? Neden?

---

## 📂 Dizin Yapısı

```
day-07-outlier-detection/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (numpy, pandas, scikit-learn, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve grafik üretim akışı
├── ciktilar/                   # Üretilen 2x2 karşılaştırma grafiği
│   └── aykiri_deger_karsilastirma.png
├── src/
│   ├── __init__.py
│   ├── istatistiksel_tespit.py # Z-Skoru, Modifiye Z (MAD) ve IQR sınıfları
│   ├── makine_ogrenmesi_tespiti.py # İzolasyon Ormanı ve LOF sınıfları
│   └── karsilastirma_ve_gorsellestirme.py # Karşılaştırıcı ve 2x2 çizim motoru
└── testler/
    └── test_aykiri_deger.py    # 7 adet birim testi (7 passed)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışı Çalıştırma
```bash
python ana_akis.py
```

### 3. Testleri Koşma
```bash
python -m pytest testler/test_aykiri_deger.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
