# 🚗 Tesla Güç Aktarma Mimarisi | Gün 29: Uzay Vektör PWM (SVPWM) ve İnvertör Sürücüleri

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Modulation](https://img.shields.io/badge/Modulation-Space%20Vector%20PWM%20(SVPWM)-blue.svg?style=flat-square)](https://www.tesla.com/)
[![DC Bus](https://img.shields.io/badge/DC%20Utilization-+15.47%25%20Voltage%20Boost-green.svg?style=flat-square)](https://www.sae.org/)
[![Hardware](https://img.shields.io/badge/Hardware-SiC%20MOSFET%20Dead--Time%201.5%C2%B5s-orange.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"29. günümüze hoş geldin stajyer!  
> Dün FOC kontrolcüsünün $V_\alpha, V_\beta$ gerilim komutlarını nasıl ürettiğini gördük. Peki $400\text{V}$ DC batarya gerilimini 3 fazlı motora aktaran 6 adet SiC (Silisyum Karbür) MOSFET anahtarını en yüksek verimle nasıl açıp kapatırız?  
> Klasik Sinüzoidal SPWM yerine **Uzay Vektör Darbe Genişlik Modülasyonu (SVPWM - Space Vector PWM)** kullanıyoruz!  
> 1. **%15.47 DC Gerilim Kazancı:** Klasik SPWM ile elde edilebilecek maksimum faz gerilimi $V_{dc} / 2 = 200\text{V}$ iken, SVPWM 3. harmonik enjeksiyonu sayesinde $V_{dc} / \sqrt{3} = 230.94\text{V}$ üretir. Bu da motorun $\%15.5$ daha fazla tork ve hız üretmesi demektir!  
> 2. **6 Sektör ve 8 Anahtarlama Vektörü:** $360^\circ$ uzay düzlemi $60^\circ$'lik 6 sektöre bölünür ($S_1 - S_6$). Her sektörde 2 aktif vektör ($T_1, T_2$) ve sıfır vektörleri ($T_0$) kombine edilir.  
> 3. **7-Segment Simetrik Anahtarlama:** Anahtarlar $V_0 \to V_1 \to V_2 \to V_7 \to V_2 \to V_1 \to V_0$ sırasında simetrik açılıp kapanarak akım dalgalanmasını (Current Ripple / THD) ve anahtarlama kayıplarını minimuma indirir.  
> 4. **1.5 µs Ölü Zaman (Dead-Time):** Aynı bacak üzerindeki üst ve alt MOSFET aynı anda iletime geçerse batarya kısa devre olur (Shoot-through). Yazılımsal ve donanımsal olarak $1.5\ \mu\text{s}$ ölü zaman eklenir!  
> Bugün invertörün en kritik modülasyon motorunu inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Sektör Tespiti ve Vektör Süreleri
Referans gerilim vektörü $V_{\text{ref}} = \sqrt{V_\alpha^2 + V_\beta^2}$ ve açısı $\theta = \text{atan2}(V_\beta, V_\alpha)$ olmak üzere ($k \in [1, 6]$ sektör numarası, $T_s$ PWM periyodu):

$$T_1 = \frac{\sqrt{3} T_s |V_{\text{ref}}|}{V_{dc}} \sin\left(\frac{k\pi}{3} - \theta_{\text{local}}\right)$$

$$T_2 = \frac{\sqrt{3} T_s |V_{\text{ref}}|}{V_{dc}} \sin\left(\theta_{\text{local}}\right)$$

$$T_0 = T_s - T_1 - T_2$$

### 2. Maksimum Doğrusal Gerilim ve Kazanç

$$V_{\text{max, SPWM}} = \frac{V_{dc}}{2} = 200.0\text{ V}$$

$$V_{\text{max, SVPWM}} = \frac{V_{dc}}{\sqrt{3}} = 230.94\text{ V}$$

$$\text{Kazanç} = \left(\frac{2}{\sqrt{3}} - 1\right) \times 100\% = +15.47\%$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Batarya DC bara geriliminden maksimum düzeyde faydalanmak, invertör anahtarlama harmoniklerini (THD) düşürmek ve motor verimini maksimize etmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **%15.5 Daha Yüksek Tepe Gerilimi:** Aynı $400\text{V}$ bataryadan daha yüksek tork ve son hız elde edildi.
- **Düşük Harmonik ve Sessiz Sürüş:** 7-segment simetrik modülasyon stator akımındaki harmonik dalgalanmaları ve motor uğultusunu yok etti.
- **Kısa Devre Koruması (Dead-time):** $1.5\ \mu\text{s}$ ölü zaman ile H-köprüsü bacaklarında shoot-through arızaları engellendi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Aşırı Modülasyon (Overmodulation):** Heksagon sınırının dışına çıkıldığında gerilim doğrusal olmayan alt harmonikler üretir.
- **Ölü Zaman Bozulması (Dead-time Distortion):** Düşük akımlarda sıfır geçiş noktalarında gerilimde hafif çentikler oluşur.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Sinüzoidal SPWM:** Çok basittir ancak DC baranın sadece $\%86.6$'sını kullanabilir.
- **Discontinuous PWM (DPWM):** Her periyotta bir fazı tamamen sabit tutar; anahtarlama kaybını azaltır fakat harmonikleri artırır.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **SVPWM (Space Vector PWM)** | Gerilim vektörlerini 6 sektörlü uzay düzleminde sentezleyerek anahtarlama sürelerini belirleyen modülasyon. |
| **VSI (Voltage Source Inverter)** | DC bara gerilimini 6 anahtarla 3 fazlı AC gerilime dönüştüren 2-seviyeli invertör köprüsü. |
| **Active Vectors ($V_1 - V_6$)** | 3 fazın farklı kombinasyonlarda iletimde olduğu $60^\circ$ açılı 6 adet gerilim uzay vektörü. |
| **Zero Vectors ($V_0, V_7$)** | 3 fazın hepsinin GND ($000$) veya hepsinin bara ($111$) ile bağlandığı sıfır gerilim vektörleri. |
| **Duty Cycle ($d_a, d_b, d_c$)** | Bir PWM periyodunda üst anahtarın açık kalma süresinin toplam periyoda oranı ($0.0 - 1.0$). |
| **7-Segment Symmetric PWM** | Sıfır ve aktif vektörleri periyot ortasına göre ayna simetrisiyle dizen düşük harmonikli dizi. |
| **Dead-Time (Ölü Zaman)** | Bir bacağın üst anahtarı kapanırken alt anahtar açılmadan önce verilen mikro saniyelik güvenlik gecikmesi. |
| **Shoot-Through** | Aynı invertör bacağındaki üst ve alt anahtarın aynı anda iletime geçerek bataryayı kısa devre etmesi arızası. |
| **THD (Total Harmonic Distortion)** | Motor akımındaki istenmeyen harmonik frekansların toplam temel bileşene oranı. |
| **Saddle-Shape Waveform** | 3. harmonik eklenmiş SVPWM faz görev çevrimlerinin aldığı karakteristik eyer/M şekli. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %15.47 daha yüksek çıkış gerilimi ve tork kapasitesi| • Klasik SPWM'e göre trigonometrik hesaplama yükü     |
| • 7-segment simetri ile minimum akım dalgalanması     | • Ölü zamanın düşük akımlarda yarattığı harmonikler   |
| • 1.85 µs ultra hızlı mikrodenetleyici çözümü         |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • SiC (Silisyum Karbür) anahtarlarla 20-50 kHz PWM    | • Yetersiz ölü zaman sebebiyle SiC MOSFET'lerin       |
|   hızına çıkarak invertör boyutunu küçültme           |   patlaması (Shoot-through) riski                     |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & 6-Sektör Uzay Vektör Şeması

```
                                 V2 (010)
                                    ^
                           S2       |       S1
                     +--------------+--------------+
                    /                               \
         V3 (011)  <                                 >  V1 (100)
                    \                               /
                     +--------------+--------------+
                           S3       |       S6
                                    v
                                 V4 (001)

             [ V0 = 000 (Sıfır) ]  ve  [ V7 = 111 (Sıfır) ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana SVPWM simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
