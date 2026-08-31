# 🚗 Tesla FSD Otonom Sürüş | Gün 41: IMU ve Tekerlek Kilometre Sayacı (Wheel Odometry) Füzyonu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![IMU](https://img.shields.io/badge/Hardware-6--DOF%20IMU%20%28100Hz%29-red.svg?style=flat-square)](https://www.tesla.com/)
[![Odometry](https://img.shields.io/badge/Odometry-Differential%20Wheel%20Speeds-blue.svg?style=flat-square)](https://www.sae.org/)
[![DeadReckoning](https://img.shields.io/badge/Algorithm-Drift--Free%20ESKF-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"41. günümüze hoş geldin stajyer!  
> Tünellere girdiğinde, yeraltı otoparklarında veya kameraların geçici olarak körleştiği yoğun güneş parlamalarında Tesla nasıl yönünü ve konumunu milimetrik olarak korur?  
> Cevap: **Dead Reckoning (Ölü Hesaplama)** ve **Ataletsel Navigasyon (INS)** mimarisidir:  
> 1. **6-Eksenli IMU (Ataletsel Ölçüm Birimi):** $100\text{ Hz}$ hızla ivmeleri ($a_x, a_y, a_z$) ve açısal hızları ($\omega_x, \omega_y, \omega_z$) ölçer. Ancak jiroskoplar zamanla biriken **donanımsal sürüklenmeye (Drift / Bias $b_\omega$)** sahiptir.  
> 2. **Diferansiyel Tekerlek Hız Sensörleri:** Sol ve sağ tekerleklerin dönüş hızı ($v_L, v_R$) doğrudan bağıl hızı ve aracın iz genişliği ($W_{\text{track}} = 1.62\text{ m}$) üzerinden dönüş hızını ($\omega_{\text{odom}}$) verir.  
> 3. **Hata-Durumu Kalman Filtresi (ESKF):** Tekerlek odometrisi ile jiroskopun kaymasını eşzamanlı kestirip nötrleyerek saf IMU'daki metrelerce sürüklenmeyi $\%95$'ten fazla azaltır.  
> Bugün otonom aracın iç pusulasını ve hareket motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Diferansiyel Tekerlek Hız Kinematiği

$$v = \frac{v_R + v_L}{2}, \quad \omega_{\text{wheel}} = \frac{v_R - v_L}{W_{\text{track}}}$$

### 2. Dead Reckoning Durum Türevleri

$$\dot{X} = v \cdot \cos(\psi), \quad \dot{Y} = v \cdot \sin(\psi), \quad \dot{\psi} = \omega_{\text{gyro}} - b_\omega$$

### 3. Jiroskop Bias ve Sürüklenme Telafisi

$$z_{\text{diff}} = \omega_{\text{gyro}} - \omega_{\text{wheel}} \implies \hat{b}_\omega \to b_{\text{hardware}}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
GPS sinyallerinin kesildiği tünellerde ve kameraların kirlendiği durumlarda aracın kendi iç sensörleriyle kesintisiz konumunu takip edebilmesi için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Jiroskop Sürüklenmesi (Drift):** Tekerlek hızlarıyla sürekli jiroskop sıfır kayması ($b_\omega$) hesaplanarak kümülatif sapma engellendi.
- **Tekerlek Kayması (Slip) Tespiti:** IMU ivmesi ile tekerlek hız türevi karşılaştırılarak patinaj ve kızaklama anları yakalandı.
- **Yüksek Frekanslı Kontrol:** 100 Hz RTOS döngüsü ile direksiyon kontrolcüsüne anlık araç yönelimi sağlandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Lastik Yarıçapı Değişimi:** Lastik basıncı düştüğünde veya aşındığında tekerlek odometrisi ölçek hatası (Scale Factor Error) yapabilir.
- **Uzun Süreli Açık Döngü:** Kilometrelerce süren tünellerde mutlak bir referans (Görsel SLAM / Harita) olmadan hata yavaşça birikir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Saf GPS:** Tünellerde ve gökdelenler arasında tamamen kesilir.
- **Görsel Odometri (Visual Odometry):** Zengin görsel ipucu gerektirir ve işlemciyi çok daha fazla yorar.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Dead Reckoning** | Bilinen bir başlangıç noktasından hız ve yön entegrasyonuyla anlık konumu tahmin etme yöntemi. |
| **IMU (Inertial Measurement Unit)**| 3-eksen ivmeölçer ve 3-eksen jiroskoptan oluşan ataletsel algılayıcı. |
| **Wheel Odometry** | Tekerleklerdeki manyetik enkoder veya ABS puls sensörlerinden alınan hız ve mesafe verisi. |
| **Gyro Bias ($b_\omega$)** | Jiroskopun araç dururken bile ürettiği sıfır noktası donanımsal voltaj kayması ($rad/s$). |
| **Track Width ($W_{\text{track}}$)** | Sol ve sağ tekerlek temas merkezleri arasındaki iz genişliği (Tesla Model 3: ~1.62 m). |
| **Yaw Rate ($\dot{\psi}$)** | Aracın düşey eksen etrafındaki anlık dönüş açısal hızı ($rad/s$). |
| **Error-State EKF (ESKF)** | Doğrudan durum yerine durum hatalarını ve sensör sapmalarını takip eden stabil Kalman filtresi. |
| **Wheel Slip Ratio** | Tekerlek yüzey hızı ile aracın gerçek yer hızı arasındaki bağıl kayma oranı. |
| **INS (Inertial Navigation)** | Ataletsel ivme ve dönüşlerin entegrasyonuyla çalışan otonom seyrüsefer sistemi. |
| **Gravity Compensation** | İvmeölçer verisinden $9.81\text{ m/s}^2$'lik yerçekimi bileşeninin araç eğimine göre çıkarılması. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tünel ve kapalı alanlarda %100 kesintisiz çalışma   | • Lastik aşınması ve basınç kaynaklı ölçek hatası     |
| • 100 Hz ultra hızlı (8.5 µs) RTOS adım süresi        | • Buzlu zeminde aşırı tekerlek patinajı (Slip)        |
| • Jiroskop bias'ını otomatik sıfırlayan ESKF          |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Görsel Odometri (VO) ve HD Harita ile birleşerek    | • Çok uzun süreli referanssız sürüşte biriken         |
|   santimetre altı küresel yerelleştirme               |   kümülatif yönelim hatası                            |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ IMU ve Tekerlek Odometrisi Füzyon Akışı

```
[ 6-DOF IMU (100 Hz) ]                                [ Tekerlek Hız Sensörleri (ABS) ]
  - İvme ax                                             - Sol Tekerlek v_L
  - Jiroskop Yaw Rate                                   - Sağ Tekerlek v_R
          |                                                     |
          v                                                     v
[ Kinematik Tahmin F(dt) ]                            [ Diferansiyel Hız & Yaw Odom ]
  - X += v*cos(psi)*dt                                  - v = (v_R + v_L) / 2
  - Y += v*sin(psi)*dt                                  - yaw = (v_R - v_L) / W_track
          \                                                     /
           \                                                   /
            v                                                 v
                     +---------------------------------------+
                     |    Hata-Durumu Kalman Filtresi (ESKF)  |
                     |    - Gyro Bias Kestirimi (b_omega)     |
                     |    - Sürüklenmesiz Araç Durumu         |
                     +---------------------------------------+
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Dead Reckoning simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
