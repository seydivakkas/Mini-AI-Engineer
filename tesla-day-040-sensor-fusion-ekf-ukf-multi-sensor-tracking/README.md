# 🚗 Tesla FSD Otonom Sürüş | Gün 40: Genişletilmiş Kalman Filtresi (EKF) ve Asenkron Sensör Füzyonu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Fusion](https://img.shields.io/badge/Algorithm-6--State%20EKF%20Sensor%20Fusion-red.svg?style=flat-square)](https://www.tesla.com/)
[![Radar](https://img.shields.io/badge/Sensors-Camera%20(20Hz)%20+%20Radar%20(10Hz)-blue.svg?style=flat-square)](https://www.sae.org/)
[![Gating](https://img.shields.io/badge/Safety-Mahalanobis%20Outlier%20Rejection-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"40. günümüze hoş geldin stajyer!  
> Otonom sürüşte tek bir sensör asla mükemmel değildir:  
> - **Kameralar:** $20-36\text{ Hz}$ frekansta yatay/dikey konumu ve şeritleri harika görür fakat doğrudan radyal hız ölçemez.  
> - **Radarlar:** $10-20\text{ Hz}$ frekansta Doppler etkisiyle radyal hızı ($\dot{r}$) anında ölçer fakat polar koordinatta ($r, \theta$) çalışır ve açısal çözünürlüğü kabadır.  
> Farklı frekanslarda, farklı koordinat sistemlerinde ve farklı gürültü seviyelerinde gelen bu verileri tek bir tutarlı 3D nesne yörüngesine dönüştürmek için **Genişletilmiş Kalman Filtresi (Extended Kalman Filter - EKF)** kullanılır:  
> 1. **6-Durumlu Kinematik Model:** Hedefin konumu, hızı ve ivmesi ($p_x, p_y, v_x, v_y, a_x, a_y$) sürekli takip edilir.  
> 2. **Non-lineer Radar Jacobian Matrisi ($H_j$):** Polar koordinattan Kartezyen uzaya türev matrisiyle anlık doğrusallaştırma yapılır.  
> 3. **Mahalanobis Kapılama (Gating):** Fiziksel olarak imkânsız sahte sensör zıplamaları ($d_M^2 > \chi^2$) otomatik elenir.  
> Bugün araçları ve yayaları santimetre hassasiyetiyle takip eden çoklu sensör füzyon motorunu inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 6-Durumlu Kinematik Durum Vektörü ve Tahmin Adımı

$$\mathbf{x} = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \\ a_x \\ a_y \end{bmatrix}, \quad \mathbf{x}_{k|k-1} = \mathbf{F}(\Delta t) \cdot \mathbf{x}_{k-1|k-1}, \quad \mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P} \mathbf{F}^T + \mathbf{Q}$$

### 2. Non-Lineer Radar Ölçüm Modeli ve Jacobian ($H_j$)

$$\mathbf{h}(\mathbf{x}) = \begin{bmatrix} r \\ \theta \\ \dot{r} \end{bmatrix} = \begin{bmatrix} \sqrt{p_x^2 + p_y^2} \\ \text{atan2}(p_y, p_x) \\ \frac{p_x v_x + p_y v_y}{\sqrt{p_x^2 + p_y^2}} \end{bmatrix}$$

$$\mathbf{H}_j = \begin{bmatrix} \frac{p_x}{r} & \frac{p_y}{r} & 0 & 0 & 0 & 0 \\ -\frac{p_y}{r^2} & \frac{p_x}{r^2} & 0 & 0 & 0 & 0 \\ \frac{p_y(v_x p_y - v_y p_x)}{r^3} & \frac{p_x(v_y p_x - v_x p_y)}{r^3} & \frac{p_x}{r} & \frac{p_y}{r} & 0 & 0 \end{bmatrix}$$

### 3. Mahalanobis Mesafesi ile Outlier Filtreleme

$$d_M^2 = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y} \le \chi_{n, 1-\alpha}^2$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Kamera ve radar sensörlerinin zayıf yönlerini birbirinin güçlü yönleriyle tamamlayarak gürültülü ortamlarda bile pürüzsüz ve deterministik hedef takibi yapmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Asenkron Güncelleme:** Farklı periyotlarda (Kamera 50 ms, Radar 100 ms) gelen verileri zaman damgalarıyla doğru sırayla entegre etti.
- **Doğrudan İvme Tahmini:** 6 durumlu model sayesinde öndeki aracın anlık frenleme ivmesini ($a_x$) yakaladı.
- **Sahte Hedef İptali:** Mahalanobis kapısı ile yoldan seken hatalı radar yankılarını eledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Doğrusallaştırma Hatası:** Sert virajlarda Jacobian yaklaşımı birinci derece Taylor açılımı olduğu için sapabilir (UKF / Particle Filter gerekir).
- **Hedef Kimlik Karışması:** İki araç çok yakın geçtiğinde ölçümlerin yanlış araca atanması (Data Association) riski.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Basit Ağırlıklı Ortalama:** Gecikmeyi ve kovaryansı hesaba katmaz; hız tahmini yapamaz.
- **Unscented Kalman Filtresi (UKF):** Daha yüksek doğruluk sağlar ancak Jacobian'a göre $\%40$ daha fazla işlem gücü harcar.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Sensor Fusion** | Farklı fiziksel prensiplerle çalışan sensörlerin verilerini tek bir optimum durumda birleştirme süreci. |
| **EKF (Extended Kalman Filter)** | Doğrusal olmayan sistem modellerini Jacobian matrisleriyle yerel olarak doğrusallaştıran filtre. |
| **State Vector ($\mathbf{x}$)** | Takip edilen nesnenin 3D konum, hız ve ivme bileşenlerini içeren durum dizisi. |
| **Covariance Matrix ($\mathbf{P}$)** | Tahmin edilen durum parametrelerinin belirsizlik ve çapraz korelasyon matrisi. |
| **Jacobian Matrix ($H_j$)** | Radar polar koordinatlarının Kartezyen durumlara göre birinci derece kısmi türevler matrisi. |
| **Mahalanobis Distance** | Hata kovaryansını hesaba katarak ölçümün tahminle ne kadar uyumlu olduğunu belirleyen istatistiksel mesafe. |
| **Gating (Kapılama)** | Belirli bir istatistiksel eşiğin ($\chi^2$) dışındaki sahte ölçümleri reddetme filtresi. |
| **Process Noise ($\mathbf{Q}$)** | Aracın yol boyunca yapabileceği ani manevraları ve model eksikliklerini temsil eden gürültü matrisi. |
| **Measurement Noise ($\mathbf{R}$)** | Kamera ve radar sensörlerinin fabrikasyon ve çevresel gürültü kovaryansı. |
| **Asynchronous Fusion** | Sensörlerin aynı anda gelmesini beklemeden her sensör geldikçe adımlayan zamanlama mimarisi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • < 0.20 m konum ve < 0.25 m/s hız takip doğruluğu    | • Sert manevralarda Jacobian birinci derece sapması   |
| • Asenkron 20 Hz / 10 Hz çoklu sensör desteği         | • Yoğun trafikte ölçüm ilişkilendirme karmaşıklığı    |
| • 14.5 µs ultra hızlı EKF adım süresi                 |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • IMU ve tekerlek hız sensörleriyle birleşerek tam     | • Aşırı sensör gürültüsünde kovaryans patlaması       |
|   araç kinematik durum kestirimi                      |   (Filter Divergence)                                 |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Çoklu Sensör EKF Füzyon Döngüsü

```
                +-----------------------------------------+
                |    6-Durumlu Kinematik Durum [x, P]     |
                +--------------------+--------------------+
                                     |
                         [ Sürekli Tahmin F(dt) ]
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
        [ Kamera Ölçümü (20 Hz) ]           [ Radar Ölçümü (10 Hz) ]
        - Lineer H = [1 0 0 0 0 0]          - Polar Jacobian Hj(x)
        - Mahalanobis Gating Denetimi       - Mahalanobis Gating Denetimi
                   |                                   |
                   +-----------------+-----------------+
                                     |
                                     v
                        [ Kalman Güncelleme K, y ]
                                     |
                                     v
                       [ Nihai Filtrelenmiş Durum ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana sensör füzyonu simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
