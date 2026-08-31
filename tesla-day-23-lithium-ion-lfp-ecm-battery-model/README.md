# 🚗 Tesla Batarya Yönetim Sistemi | Gün 23: Lityum İyon / LFP Hücre Kimyası ve 2-RC ECM Modeli

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Physics](https://img.shields.io/badge/Model-2--RC%20Thevenin%20ECM-blue.svg?style=flat-square)](https://www.tesla.com/)
[![Chemistry](https://img.shields.io/badge/Chemistry-LFP%20%2F%20NMC%20%2F%20NCA-orange.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/Thermal-Arrhenius%20Joule%20Loss-red.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"23. günümüze ve FAZ 3'e (Tesla Batarya Yönetim Sistemi & Motor Kontrolü) hoş geldin stajyer!  
> Bir Tesla'nın tabanındaki 400V veya 800V batarya paketi, binlerce silindirik (2170/4680) veya prizmatik LFP hücresinden oluşur.  
> Bu hücrelerin şarj durumunu (SoC), kalan menzilini ve anlık güç sınırlarını hesaplamak için fiziksel elektrokimyasal hücreyi mikrosaniyeler içinde çözen bir matematiksel modele ihtiyacımız vardır: **Eşdeğer Devre Modeli (Equivalent Circuit Model - ECM)**.  
> 1. **LFP vs NMC Kimyası:** LFP (Lityum Demir Fosfat) kimyası $\%20$ ile $\%80$ arasında neredeyse tamamen düz bir $3.28\text{V}$ voltaj platosuna sahiptir; OCV'ye bakarak SoC kestirmek zordur. NMC/NCA ise $3.0\text{V}$ ile $4.2\text{V}$ arasında belirgin bir eğime sahiptir.  
> 2. **2-RC Dual Polarization Thevenin Devresi:**  
>    - $R_0$: Saf ohmik iç direnç (akım geçtiği anda anlık voltaj çökmesi yaratır).  
>    - $R_1 \parallel C_1$: Hızlı polarizasyon ve çift katman yük transferi ($\tau_1 \approx 1-3\text{ s}$).  
>    - $R_2 \parallel C_2$: Katot/anot içindeki yavaş lityum iyon difüzyon dinamiği ($\tau_2 \approx 20-100\text{ s}$).  
> 3. **Arrhenius Sıcaklık Bağımlılığı:** Soğuk havada (Örn: $-10^\circ\text{C}$) lityum iyon hareketliliği yavaşlar ve iç direnç katlanarak artar. Bu yüzden Supercharger şarjı öncesinde bataryayı $45^\circ\text{C}$'ye ön ısıtmak (Preconditioning) şarttır!  
> Bugün Tesla BMS algoritmasının çekirdeği olan 2-RC ECM çözücüsünü kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 2-RC Eşdeğer Devre Modeli (ECM) Durum Uzayı Denklemleri
Zaman domeninde sürekli durum diferansiyel denklemleri:

$$\frac{d V_{RC1}(t)}{dt} = -\frac{V_{RC1}(t)}{R_1 C_1} + \frac{I(t)}{C_1}$$

$$\frac{d V_{RC2}(t)}{dt} = -\frac{V_{RC2}(t)}{R_2 C_2} + \frac{I(t)}{C_2}$$

$$V_t(t) = OCV(SoC(t)) - I(t) R_0(T) - V_{RC1}(t) - V_{RC2}(t)$$

### 2. Ayrık Zamanlı (Discrete-Time) Kesin Çözüm
$\Delta t$ örnekleme adımında ($\tau_1 = R_1 C_1, \tau_2 = R_2 C_2$):

$$V_{RC1}[k+1] = \exp\left(-\frac{\Delta t}{\tau_1}\right) V_{RC1}[k] + R_1 \left(1 - \exp\left(-\frac{\Delta t}{\tau_1}\right)\right) I[k]$$

$$V_{RC2}[k+1] = \exp\left(-\frac{\Delta t}{\tau_2}\right) V_{RC2}[k] + R_2 \left(1 - \exp\left(-\frac{\Delta t}{\tau_2}\right)\right) I[k]$$

### 3. Sıcaklığa Bağlı İç Direnç (Arrhenius Denklemi)
Sıcaklık $T$ (Kelvin) ve referans sıcaklık $T_{\text{ref}} = 298.15\text{ K}$ ($25^\circ\text{C}$) olmak üzere:

$$R_0(T) = R_{0,\text{ref}} \cdot \exp\left(\frac{E_a}{R_{\text{gas}}} \left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Mikrodenetleyicilerde mikrosaniyeler içinde çözülemeyecek kadar ağır olan kısmi türevli diferansiyel elektrokimyasal modeller (Newman P2D) yerine gerçek zamanlı çalışabilen 2-RC Thevenin ECM kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Gerçek Zamanlı Çözüm:** Karmaşık difüzyon fiziğini 2 adet RC dalı ve Arrhenius bağıntısıyla basitleştirerek 1 kHz RTOS döngüsünde çözülmesini sağladı.
- **Voltaj Çökmesi (Sag) Tahmini:** Hızlanma ve rejenerasyon anlarında terminal voltajındaki anlık ve gecikmeli dalgalanmaları hassas modelledi.
- **Isıl Kayıp Hesabı:** $P_{\text{loss}} = I^2 R_0 + I V_{RC1} + I V_{RC2}$ ile batarya soğutma ihtiyacı anlık hesaplandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Histerezis Etkisi:** LFP hücrelerinde şarj ve deşarj OCV eğrileri arasındaki voltaj farkı (Hysteresis) basit 1-RC modellerde ihmal edilirse $\%5$'e varan SoC hatası oluşabilir.
- **Yaşlanma (Degradation):** Hücre yaşlandıkça SEI tabakası kalınlaşır; $R_0$ ve kapasite $Q$ zamanla güncellenmelidir (SoH kestirimi gerektirir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Newman P2D (Pseudo-2-Dimensional) Modeli:** Çok hassastır fakat binlerce diferansiyel denklem içerdiği için gömülü ECU'larda gerçek zamanlı çalışamaz.
- **Yapay Sinir Ağı (NN/LSTM ECM):** Veri tabanlıdır fakat dağılım dışı (OOD) sıcaklıklarda fiziksel garantiler sunamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **ECM (Equivalent Circuit Model)** | Batarya hücresinin elektriksel ve dinamik davranışını taklit eden direnç-kapasitör devresi. |
| **OCV (Open Circuit Voltage)** | Hücre dinlenme durumundayken (akım sıfırken) elektrotlar arasındaki açık devre gerilimi. |
| **SoC (State of Charge)** | Bataryanın o anki doluluk oranını gösteren yüzde değeri ($\%0 - \%100$). |
| **LFP (LiFePO4)** | Lityum Demir Fosfat katotlu, uzun ömürlü, termal olarak güvenli fakat düz OCV platosuna sahip batarya kimyası. |
| **NMC (NiMnCo)** | Yüksek enerji yoğunluklu ve belirgin eğimli OCV karakteristiğine sahip batarya kimyası. |
| **Ohmic Resistance ($R_0$)** | Elektrolit ve akım toplayıcı folyolardan kaynaklanan anlık omik iç direnç. |
| **Charge Transfer Resistance ($R_1$)** | Elektrot-elektrolit arayüzündeki yük transfer reaksiyonu direnci. |
| **Double Layer Capacitance ($C_1$)** | Elektrot yüzeyinde oluşan mikroskobik çift katmanlı elektrostatik kapasitans. |
| **Diffusion RC Branch ($R_2 \parallel C_2$)** | Katı hal lityum iyon difüzyonunun yarattığı yavaş gerilim gevşemesi (relaxation). |
| **Arrhenius Equation** | Sıcaklık azaldıkça kimyasal reaksiyon hızının ve iç direncin katlanarak değişimini modelleyen yasa. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 1 kHz RTOS döngüsünde 3.2 µs ultra hızlı çözüm      | • LFP düz platosunda OCV tek başına SoC veremez       |
| • Sıcaklık ve difüzyon dinamiklerini içeren 2-RC yapı | • Histerezis etkisi için ilave durum değişkeni ister  |
| • Gerçek zamanlı EKF algoritmasına doğrudan uyum      |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Genişletilmiş Kalman Filtresi (EKF) ile birleşerek  | • Aşırı soğuk havada (-20°C) lityum kaplama (plating) |
|   %1'in altında SoC kestirim doğruluğu                |   ve model sapması riski                              |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & 2-RC Devre Şeması

```
   (+) o-------[ R0 (Ohmic) ]-------+---[ R1 ]---+-------+---[ R2 ]---+-------o Terminal (+)
                                    |            |       |            |
                                    +---[ C1 ]---+       +---[ C2 ]---+
                                       (Polarization)       (Diffusion)
                                           (V_RC1)              (V_RC2)
                                                                 |
                                                                 v
                                                     [ - OCV(SoC) + ]
                                                                 |
   (-) o---------------------------------------------------------+------------o Terminal (-)
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana ECM simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
