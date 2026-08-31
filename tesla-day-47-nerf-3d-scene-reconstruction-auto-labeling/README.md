# 🚗 Tesla FSD Otonom Sürüş | Gün 47: NeRF (Neural Radiance Fields) ve 3D Otomatik Etiketleme (Auto-Labeling)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NeRF](https://img.shields.io/badge/Graphics-Neural%20Radiance%20Fields-red.svg?style=flat-square)](https://www.tesla.com/)
[![AutoLabel](https://img.shields.io/badge/Dataset-100%25%20Automated%203D%20Ground%20Truth-blue.svg?style=flat-square)](https://www.sae.org/)
[![PSNR](https://img.shields.io/badge/Quality-34.8%20dB%20PSNR-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"47. günümüze hoş geldin stajyer!  
> Otonom sürüşte derin öğrenme modellerini eğitmek için milyonlarca saatlik etiketli 3D veriye (Ground Truth) ihtiyaç vardır. Ancak binlerce insan etiketçiye video karelerinde elle 3D kutu çizdirmek hem aylar sürer, hem milyonlarca dolara mal olur, hem de insan hatası içerir.  
> Tesla bu devasa veri darboğazını **NeRF (Neural Radiance Fields)** tabanlı **Otomatik Etiketleme Motoru (Auto-Labeling Pipeline)** ile aştı:  
> 1. **Hacimsel Işın İzleme (Volume Rendering):** Kamera piksellerinden uzaya fırlatılan ışınlar boyunca yoğunluk ($\sigma$) ve renk ($c$) integrali alınarak 3D sahne sürekli bir fonksiyon olarak öğrenilir.  
> 2. **Kümülatif Geçirgenlik (Transmittance $T(t)$):** Işının katı bir nesneye çarptığında opaklaşması modellenerek milimetre hassasiyetinde yüzey derinliği ($D$) çözülür.  
> 3. **Zaman Dizisi Rekonstrüksiyonu:** Aynı sokaktan farklı zamanlarda geçen Tesla filosu araçlarının klipleri üst üste bindirilerek statik dünya ve dinamik araçlar kusursuzca ayrıştırılır.  
> 4. **Sıfır İnsan Müdahalesi:** $34.8\text{ dB}$ PSNR kalitesinde 3D Zemin Gerçeği etiketleri otomatik olarak NPU/GPU kümesinde üretilir.  
> Bugün Tesla'nın veri fabrikasının kalbini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. NeRF Hacimsel Renk ve Derinlik İntegrali

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

$$T(t) = \exp\left( -\int_{t_n}^t \sigma(\mathbf{r}(s)) \, ds \right)$$

### 2. Ayrıklaştırılmış Alpha Compositing ve Ağırlıklar

$$\alpha_i = 1 - \exp(-\sigma_i \cdot \delta_i), \quad T_i = \prod_{j=1}^{i-1} (1 - \alpha_j), \quad w_i = T_i \cdot \alpha_i$$

$$D(\mathbf{r}) = \sum_{i=1}^N w_i \cdot t_i$$

### 3. Rekonstrüksiyon Kalite Metriği (PSNR)

$$\text{MSE} = \frac{1}{N} \sum \| C_{\text{pred}} - C_{\text{gt}} \|^2, \quad \text{PSNR} = 10 \cdot \log_{10}\left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Milyonlarca saatlik FSD sürüş verisini insan etiketçilere ihtiyaç duymadan, süper bilgisayar kümesinde (Dojo) otomatik olarak 3D etiketlemek ve zemin gerçeği üretmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Etiketleme Maliyeti ve Zamanı:** İnsan etiketleme maliyetini $\%99$ düşürdü ve veri üretim hızını 1000 kat artırdı.
- **Geometrik Kusursuzluk:** İnsan gözünün kestiremediği uzak nesne derinliklerini fotogrametrik NeRF integrali ile milimetrik doğrulukla çıkardı.
- **Sentetik Simülasyon Verisi:** Öğrenilen NeRF sahnesinde hava durumunu (güneşli -> yağmurlu) ve kamera açılarını değiştirerek sonsuz sentetik eğitim karesi üretti.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Eğitim Süresi:** Bir sahnenin NeRF ile tam çözülmesi yüksek GPU saati gerektirir (Instant-NGP / 3D Gaussian Splatting ile hızlandırılır).
- **Yansıtıcı Yüzeyler:** Islak zemin ve ayna gibi aşırı parlayan yüzeylerde hacimsel yoğunluk hesabında belirsizlik oluşabilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Manuel İnsan Etiketleme:** Aşırı pahalı, yavaş ve insan sübjektifliğine açıktır.
- **LiDAR Tabanlı Zemin Gerçeği:** Pahalıdır ve filodaki milyonlarca araçta bulunmaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **NeRF** | 3D sahneleri koordinat bazlı sürekli sinir ağlarıyla temsil eden hacimsel render tekniği. |
| **Volume Rendering** | Bir ışın boyunca renk ve yoğunluk integrali alarak 2D piksel rengi ve derinliği hesaplama süreci. |
| **Transmittance ($T(t)$)** | Işının belirli bir $t$ mesafesine kadar başka bir nesneye çarpmadan ilerleme olasılığı. |
| **Volume Density ($\sigma$)** | 3D uzaydaki bir noktanın ışığı ne kadar bloke ettiğini belirten hacimsel yoğunluk katsayısı. |
| **Auto-Labeling Pipeline** | Filo araçlarından toplanan video kliplerinden insan müdahalesiz 3D zemin gerçeği üreten hat. |
| **Novel View Synthesis** | Kameraların çekmediği yeni açılardan ve konumlardan gerçekçi sentetik sahneler türetme. |
| **Alpha Compositing** | Katmanların geçirgenliklerine göre önden arkaya doğru ağırlıklı birleştirilmesi. |
| **Stratified Sampling** | Işın üzerinde düzenli aralıklara rastgele pertürbasyon ekleyerek aliasing etkisini önleyen örnekleme. |
| **PSNR (Peak Signal-to-Noise)**| Render edilen görüntü ile orijinal kamera karesi arasındaki rekonstrüksiyon kalitesi logaritmik metriği. |
| **Dojo Supercomputer** | Tesla'nın video ve NeRF otomatik etiketleme yüklerini çalıştırmak için geliştirdiği süper bilgisayar. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • İnsan etiketleme maliyetinde %99 düşüş              | • Yoğun GPU kümesi ve hesaplama gücü ihtiyacı         |
| • 34.8 dB yüksek rekonstrüksiyon kalitesi             | • Cam ve ayna gibi yansıtıcı yüzeylerde artefaktlar   |
| • Sonsuz sentetik köşe durum (Corner-Case) üretimi    |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • 3D Gaussian Splatting entegrasyonu ile              | • Aşırı dinamik ve kaotik kavşaklarda hareketli       |
|   gerçek zamanlı (Real-Time) render hızına ulaşma     |   nesne ayrıştırma zorluğu                            |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla NeRF ve Otomatik Etiketleme Akışı

```
[ Filo Video Klipleri (8 Kamera Dizisi) ]
                  |
                  v
[ Kamera Pozları ve Işın Demetleri r(t) ]
                  |
                  v
[ Hacimsel Işın İzleme (Volume Rendering) ]
  - Yoğunluk sigma(t) & Renk c(t)
  - Transmittance T(t) İntegrali
                  |
                  v
[ 3D Sahne Rekonstrüksiyonu & Yoğun Nokta Bulutu ]
                  |
                  v
[ Otomatik 3D Zemin Gerçeği Bounding Box & Voksel Etiketleri ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana NeRF otomatik etiketleme simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
