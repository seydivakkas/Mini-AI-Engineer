# 🚗 Tesla FSD Otonom Sürüş | Gün 58: Model Predictive Control (MPC) ile Kinematik Bisiklet Modeli Kontrolü

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![MPC](https://img.shields.io/badge/Control-Model%20Predictive%20Control%20%28MPC%29-red.svg?style=flat-square)](https://www.tesla.com/)
[![State Space](https://img.shields.io/badge/State%20Space-Kinematic%20Bicycle%20%5Bey%2C%20epsi%2C%20ev%5D-blue.svg?style=flat-square)](https://www.sae.org/)
[![Optimization](https://img.shields.io/badge/Optimization-Discrete%20Riccati%20%26%20QP-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"58. günümüze hoş geldin stajyer!  
> Geleneksel PID kontrolcüler geçmişteki hataya bakarak reaksiyon gösterir (Reaktif Kontrol). Ancak $120\text{ km/h}$ hızla otoyolda giden bir Tesla viraja girmeden önce direksiyonu kırmaya başlamalıdır!  
> İşte burada otonom araç kontrolünün kralı olan **Model Predictive Control (MPC - Model Tahminli Kontrol)** devreye girer:  
> 1. **Geleceği Öngörme (Prediction Horizon $N$):** Araç önündeki $2.0$ saniyelik ($N = 20$ adım) yolu kinematik araç modeliyle simüle eder.  
> 2. **Karesel Maliyet Matrisleri ($Q, R$):** Şerit merkezinden sapma ($e_y$), açısal hata ($e_\psi$) ve hız hatası ($e_v$) ile direksiyon eforu ($r_\delta$) arasında optimal bir denge kurar.  
> 3. **Ayrık Riccati Optimizasyonu (Discrete Riccati Equation):** Matris cebri ile optimal geri besleme kazancını ($K$) gerçek zamanlı olarak hesaplar.  
> 4. **Aktüatör Doyumu (Actuator Saturation):** Ön tekerleklerin maksimum dönüş açısını ($\pm 31.5^\circ$) ve ivme sınırlarını kısıtlayarak fiziksel olarak kusursuz bir kapalı çevrim takip sağlar.  
> Bugün Tesla'nın şeritte milimetrik kalmasını sağlayan MPC kontrolcüsünü inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Durum ve Kontrol Vektörleri

$$\mathbf{e}(k) = \begin{bmatrix} e_y(k) \\ e_\psi(k) \\ e_v(k) \end{bmatrix}, \quad \mathbf{u}(k) = \begin{bmatrix} a(k) \\ \delta(k) \end{bmatrix}$$

### 2. Doğrusallaştırılmış Durum-Uzay Ayrık Geçiş Modeli

$$\mathbf{e}(k+1) = A \mathbf{e}(k) + B \mathbf{u}(k)$$

$$A = \begin{bmatrix} 1 & v \Delta t & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad B = \begin{bmatrix} 0 & 0 \\ 0 & \frac{v}{L} \Delta t \\ \Delta t & 0 \end{bmatrix}$$

### 3. MPC / LQR Karesel Maliyet Fonksiyonu ve Ayrık Riccati Denklemi (DARE)

$$J = \sum_{k=0}^{\infty} \left( \mathbf{e}(k)^T Q \mathbf{e}(k) + \mathbf{u}(k)^T R \mathbf{u}(k) \right)$$

$$P = Q + A^T P A - A^T P B \left( R + B^T P B \right)^{-1} B^T P A$$

$$K = \left( R + B^T P B \right)^{-1} B^T P A, \quad \mathbf{u}^*(k) = -K \mathbf{e}(k)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Gelecekteki yol eğriliğini ve hız profilini önceden tahmin ederek aktüatör gecikmelerini telafi etmek ve şerit takip hatasını 5 cm altına indirmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **PID Aşım ve Salınımları (Overshoot & Oscillation):** PID kontrolcülerin viraj çıkışında yaptığı yalpalamaları tamamen yok etti.
- **Çok Eksenli Eşzamanlı Kontrol:** Hem boyuna (gaz/fren) hem de yanal (direksiyon) kontrolü tek bir maliyet matrisinde birleştirdi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Doğrusallaştırma Hatası:** Çok keskin drift manevralarında lastik sürtünme sınırları (Pacejka Magic Formula) non-lineer MPC gerektirir.
- **İşlem Yükü:** Çevrim başına matris tersi ve Riccati iterasyonları RTOS mikroişlemcide deterministik zamanlama gerektirir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Stanley Controller:** Basit ve etkilidir ancak boyuna hız optimizasyonunu ve gelecekteki viraj profilini doğrudan optimize edemez.
- **Pure Pursuit:** Yalnızca tek bir bakış noktasına (Lookahead Point) bakar; aktüatör kısıtlarını dikkate almaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Model Predictive Control (MPC)** | Matematiksel modelle geleceği öngörüp her adımda kısıtlı optimizasyon çözen kontrolcü. |
| **Prediction Horizon ($N$)** | Kontrolcünün gelecekte kaç adım sonrasını hesaba kattığı planlama ufku. |
| **State Feedback Gain ($K$)** | Durum hatası vektörünü aktüatör komutuna çeviren optimal çarpan matrisi. |
| **Cross-Track Error ($e_y$)**| Aracın istenen yörünge merkezine olan dik yanal uzaklığı. |
| **Heading Error ($e_\psi$)** | Aracın burnunun yörünge teğetine olan açısal farkı. |
| **DARE (Discrete Algebraic Riccati Eq)** | Karesel kontrol probleminde sonsuz ufuk kararlı $P$ matrisini veren matris denklemi. |
| **Actuator Saturation** | Direksiyon motorunun fiziksel dönüş açısı ve ivme kısıtlarına ulaşması durumu. |
| **Cost Matrix $Q$** | Durum hatalarına (yanal sapma, açı, hız) verilen ceza ağırlıkları matrisi. |
| **Cost Matrix $R$** | Direksiyon çevirme ve fren/gaz basma eforuna verilen ceza matrisi. |
| **Closed-Loop Simulation** | Kontrolcünün ürettiği komutların araç dinamiğine beslenip durumun güncellendiği döngü. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Gelecekteki virajları önceden sezerek yumuşak dönüş | • Doğrusal model yüksek yanal ivmelerde kayma içerir  |
| • 40 µs ultra hızlı ayrık Riccati çözümü             | • Q ve R matris ağırlıklarının ince ayar gereksinimi  |
| • Aktüatör doyum sınırlarına %100 uyum                |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Otoyol yüksek hız Navigate on Autopilot ve rüzgarlı | • Buzlu/ıslak zeminde lastik tutunma katsayısının      |
|   köprü geçişlerinde sıfır yanal salınım              |   aniden düşmesi (Friction coefficient drop)          |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla MPC Kapalı Çevrim Akış Şeması

```
[ Algılanan Hatalar: e_y (Yanal), e_psi (Açı), e_v (Hız) ]
                            |
                            v
     [ Ayrık Riccati Denklemi (DARE) & Optimal Kazanç K ]
                            |
                            v
       [ Optimal Komutlar: u* = -K * e (Gaz/Fren & Steer) ]
                            |
                            v
     [ Aktüatör Doyum Filtresi: [-31.5°, +31.5°], [-4, +2.5] m/s² ]
                            |
                            v
       [ Kinematik Bisiklet Modeli Durum Güncellemesi ]
                            |
                            v
       [ e_y < 5 cm Hata ile Kusursuz Kapalı Çevrim Takip ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana MPC simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
