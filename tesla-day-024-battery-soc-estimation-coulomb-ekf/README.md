# 🚗 Tesla Batarya Yönetim Sistemi | Gün 24: Şarj Durumu (SoC) Kestirimi & Genişletilmiş Kalman Filtresi (EKF)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-Extended%20Kalman%20Filter%20(EKF)-blue.svg?style=flat-square)](https://www.tesla.com/)
[![State Estimation](https://img.shields.io/badge/State-3--State%20[SoC,%20V_RC1,%20V_RC2]-orange.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/Accuracy-Sub--1%25%20SoC%20Error%20ASIL--D-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"24. günümüze hoş geldin stajyer!  
> Tesla göstergesindeki batarya yüzdesi ($\%82$) nereden gelir? Doğrudan bir 'SoC sensörü' yoktur.  
> Geçmişte mühendisler sadece akımı toplayan **Coulomb Counting** ($\Delta SoC = \frac{\int I dt}{Q}$) yöntemini kullandı. Fakat akım sensöründeki en ufak bir DC ofset ($+1.5\text{ A}$ kayma) birkaç saat içinde menzil göstergesini $\%30$ saptırır ve araç yolun ortasında aniden durabilir!  
> Çözüm: **Genişletilmiş Kalman Filtresi (Extended Kalman Filter - EKF)** algoritmasıdır:  
> 1. **3-Boyutlu Durum Vektörü:** EKF sadece SoC'yi değil, aynı zamanda 2-RC eşdeğer devrenin $V_{RC1}$ ve $V_{RC2}$ polarizasyon voltajlarını da eşzamanlı kestirir.  
> 2. **Jacobian Doğrusallaştırma:** Lineer olmayan $OCV(SoC)$ eğrisinin o anki teğet türevi ($C_k = \left[\frac{\partial OCV}{\partial SoC}, -1, -1\right]$) her adımda dinamik hesaplanır.  
> 3. **Ölçüm Düzeltmesi (Innovation):** Modelin tahmin ettiği terminal voltajı ile sensörün okuduğu voltaj karşılaştırılır; artık hata ($y = V_{\text{meas}} - V_{\text{pred}}$) Kalman Kazancı ($K$) ile çarpılarak hem SoC hem de RC voltajları anında düzeltilir.  
> 4. **Hatalı Başlangıç Toleransı:** BMS resetlense ve başlangıç SoC'si $\%50$ zannedilse bile EKF 20 saniye içinde gerçek değer olan $\%85$'e kusursuz şekilde yakınsar!  
> Bugün Tesla BMS'in beyni olan EKF algoritmasını inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 3-Durumlu EKF Durum Uzayı Modeli
Durum vektörü $\mathbf{x}_k = [SoC_k, V_{RC1, k}, V_{RC2, k}]^T$ ve giriş $u_k = I_k$ olmak üzere:

$$\mathbf{x}_{k+1} = \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k I_k + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q})$$

$$V_{t, k} = OCV(SoC_k) - I_k R_0 - V_{RC1, k} - V_{RC2, k} + v_k, \quad v_k \sim \mathcal{N}(0, R)$$

$$\mathbf{A}_k = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \exp(-\Delta t / \tau_1) & 0 \\ 0 & 0 & \exp(-\Delta t / \tau_2) \end{bmatrix}, \quad \mathbf{B}_k = \begin{bmatrix} -\frac{\Delta t}{Q_{\text{nom}}} \\ R_1 (1 - \exp(-\Delta t / \tau_1)) \\ R_2 (1 - \exp(-\Delta t / \tau_2)) \end{bmatrix}$$

### 2. Zaman Güncellemesi (Tahmin - Time Update)

$$\hat{\mathbf{x}}_k^- = \mathbf{A}_{k-1} \hat{\mathbf{x}}_{k-1}^+ + \mathbf{B}_{k-1} I_{k-1}$$

$$\mathbf{P}_k^- = \mathbf{A}_{k-1} \mathbf{P}_{k-1}^+ \mathbf{A}_{k-1}^T + \mathbf{Q}$$

### 3. Ölçüm Güncellemesi (Düzeltme - Measurement Update)

$$\mathbf{C}_k = \left. \frac{\partial V_t}{\partial \mathbf{x}} \right|_{\hat{\mathbf{x}}_k^-} = \begin{bmatrix} \left.\frac{\partial OCV}{\partial SoC}\right|_{\hat{SoC}_k^-} & -1 & -1 \end{bmatrix}$$

$$\mathbf{K}_k = \mathbf{P}_k^- \mathbf{C}_k^T \left(\mathbf{C}_k \mathbf{P}_k^- \mathbf{C}_k^T + R\right)^{-1}$$

$$\hat{\mathbf{x}}_k^+ = \hat{\mathbf{x}}_k^- + \mathbf{K}_k \left(V_{t, \text{meas}} - V_{t, \text{pred}}\right)$$

$$\mathbf{P}_k^+ = (\mathbf{I} - \mathbf{K}_k \mathbf{C}_k) \mathbf{P}_k^-$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Akım sensörlerindeki DC ofset/bias kaymalarını ve başlangıç durumu belirsizliklerini voltaj geri beslemesiyle yok etmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Sınırsız Sürüklenme Önendi:** Coulomb Counting'in saatler içinde $\%30+$ kaymasını $\%0.5$'in altına indirdi.
- **Kendi Kendini Düzeltme:** BMS yeniden başladığında yanlış olan başlangıç tahminini saniyeler içinde zemin gerçeğine kilitledi.
- **Belirsizlik Ölçümü:** $\mathbf{P}$ kovaryans matrisi sayesinde aracın menzil tahminine $\pm 3\sigma$ güvenilirlik sınırı tanımlandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **LFP Plato Bölgesi:** LFP kimyasında $\frac{\partial OCV}{\partial SoC} \approx 0$ olduğunda voltaj geri beslemesi zayıflar; EKF geçici olarak Coulomb Counting ağırlıklı çalışır.
- **Parametre Değişimi:** Hücre yaşlandıkça $R_0$ ve $Q_{\text{nom}}$ değiştiğinden Dual-EKF (Eşzamanlı SoC + SoH Kestirimi) gerekebilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **UKF (Unscented Kalman Filter):** Türev almaz, sigma noktaları kullanır; doğruluk biraz daha yüksektir fakat matris işlem maliyeti fazladır.
- **Partikül Filtresi (Particle Filter):** Yüksek hesaplama gücü ister; gömülü MCU'larda 1 kHz hızda çalışamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **EKF (Extended Kalman Filter)** | Lineer olmayan dinamik sistemlerin durumlarını Jacobian doğrusallaştırmasıyla kestiren optimal filtre. |
| **Coulomb Counting** | Akımın zamana göre integralini alarak şarj durumunu hesaplayan açık döngü yöntem. |
| **Sensor Bias / Drift** | Akım ölçüm şöntünde veya Hall sensöründe sıcaklık ve yaşlanmayla oluşan sabit ofset hatası. |
| **State Vector ($\mathbf{x}$)** | Sistemin o anki fiziksel durumunu tanımlayan vektör ($[SoC, V_{RC1}, V_{RC2}]$). |
| **Covariance Matrix ($\mathbf{P}$)** | Durum kestirimlerindeki belirsizliği ve değişkenler arası korelasyonu gösteren $3 \times 3$ matris. |
| **Process Noise ($\mathbf{Q}$)** | Model denklemindeki eksiklikleri ve bilinmeyen bozucuları temsil eden gürültü kovaryansı. |
| **Measurement Noise ($R$)** | Voltaj sensörü ve analog-dijital dönüştürücü (ADC) gürültü varyansı. |
| **Kalman Gain ($\mathbf{K}$)** | Model tahmini ile sensör ölçümü arasındaki güven ağırlığını belirleyen kazanç vektörü. |
| **Innovation / Residual ($y$)** | Ölçülen gerçek voltaj ile modelin tahmin ettiği voltaj arasındaki fark. |
| **3-Sigma Bounds ($\pm 3\sigma$)** | İstatistiksel olarak $\%99.73$ olasılıkla gerçek değerin içinde bulunduğu güven sınırları. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %0.5'in altında ultra hassas SoC kestirim garantisi | • LFP düz platosunda Jacobian türevinin sıfıra inmesi |
| • Akım sensörü bias kaymalarına karşı tam bağışıklık  | • Matris tersi alma (Matrix Inversion) ek yükü        |
| • Yanlış başlangıç durumunu 20 sn içinde düzeltme     |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Dual-EKF ile hücre yaşlanmasını (SoH) eşzamanlı     | • Yanlış Q ve R gürültü ayarında filtrenin ıraksaması |
|   tahmin edebilme yeteneği                            |   (Filter Divergence) riski                           |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & EKF Algoritma Döngüsü

```
                  +----------------------------------------------+
                  |         Girdi: Akım I[k] ve Gerilim V_t[k]   |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |             1. TAHMİN ADIMI                  |
                  |    - SoC_pred = SoC - (I * dt) / Q           |
                  |    - V_rc1_pred = exp(-dt/τ1) * V_rc1 + ...  |
                  |    - P_minus = A * P * A^T + Q               |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |             2. DÜZELTME ADIMI                |
                  |    - V_pred = OCV(SoC_pred) - I*R0 - V_rc1...|
                  |    - İnovasyon: y = V_meas - V_pred          |
                  |    - Jacobian: C = [dOCV/dSoC, -1, -1]       |
                  |    - Kalman Kazancı: K = P * C^T * S^(-1)    |
                  |    - x_plus = x_minus + K * y                |
                  |    - P_plus = (I - K * C) * P_minus          |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |        Çıktı: Kestirilen SoC ve ±3σ Sınırı   |
                  +----------------------------------------------+
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana EKF akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
