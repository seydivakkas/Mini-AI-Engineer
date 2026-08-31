# 🚗 Tesla FSD Otonom Sürüş | Gün 76: HVAC Dokunmatik Kontrol Arayüzü ve Step Motor PID Sürücüleri

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![HVAC](https://img.shields.io/badge/Thermal-Tesla%20Hidden%20Air%20Vent-red.svg?style=flat-square)](https://www.tesla.com/)
[![PID](https://img.shields.io/badge/Control-Anti--Windup%20PID-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller)
[![Stepper](https://img.shields.io/badge/Actuator-Microstepping%201.8%C2%B0%2FStep-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"76. günümüze hoş geldin stajyer!  
> Tesla Model 3 veya Model Y'nin içine bindiğinizde ön konsolda geleneksel plastik klima ızgaraları ve çevirmeli düğmeler göremezsiniz!  
> Tesla, otomotivde çığır açan **Patentli Gizli Havalandırma (Hidden HVAC Air Vent) ve Akışkanlar Mekaniği (Fluidic Coanda Effect)** mimarisini kullanır:  
> 1. **Gizli Çift Hava Jeti:** Ön panel boyunca uzanan ince yarıkta iki ayrı hava akımı (dikey ve yatay) birbirine çarptırılarak havanın yönü değiştirilir.  
> 2. **Dokunmatik Ekran Kontrolü:** Sürücü ekranda hava akımını parmaklarıyla böler veya yönlendirir.  
> 3. **Step Motor Flap Sürücüleri:** Dokunmatik koordinatlar mikrosaniyelik hesaplamayla step motor adımlarına ($1.8^\circ/\text{step}$) dönüştürülür.  
> 4. **Anti-Windup Termal PID:** Kabin sıcaklığını $35^\circ\text{C}$'den $21.5^\circ\text{C}$'ye aşırı aşma (Overshoot) yapmadan indiren akıllı kapalı döngü kontrolcü çalışır.  
> Bugün Tesla kabin konforunun kalbi olan HVAC PID ve Step Motor sürücü motorunu inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Kabin Sıcaklığı Kapalı Döngü PID Kontrol Kanunu

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}, \quad e(t) = T_{\text{cabin}}(t) - T_{\text{target}}$$

### 2. Kabin Termal Dinamik Diferansiyel Denklemi

$$\frac{dT_{\text{cabin}}}{dt} = -\alpha \cdot \left(\frac{u(t)}{100}\right) + \beta \cdot \left(T_{\text{ambient}} - T_{\text{cabin}}(t)\right)$$

### 3. Flap Açısı $\to$ Step Motor Darbe Dönüşümü

$$\text{Pulses} = \text{round}\left( \frac{\theta_{\text{target}}}{\Delta \theta_{\text{step}}} \right), \quad \Delta \theta_{\text{step}} = 1.8^\circ$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Minimalist Tesla iç mekan tasarımını korumak, fiziksel kırılgan klima ızgaralarını ortadan kaldırmak ve kabin sıcaklığını yazılımsal olarak kişiye özel hassas yönetmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Hava Dağıtım Karmaşası:** Çift hava jetinin akışkan çarpışmasıyla menfezi fiziksel olarak hareket ettirmeden havanın yukarı/aşağı/bölünmüş yönlendirilmesini sağladı.
- **Sıcaklık Dalgalanmaları:** Anti-windup özellikli PID kontrol döngüsü ile aşırı soğutma veya dalgalanma (Hunting) olmadan kararlı $21.5^\circ\text{C}$ rejimi sağladı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Step Motor Mekanik Aşınması:** Çok sık flap kalibrasyonu motor dişlilerinde zamanla mikro boşluklara (Backlash) yol açabilir (Periyodik home kalibrasyonu gerektirir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Manuel Plastik Izgaralar:** Ucuzdur ancak yazılımla kontrol edilemez ve otonom profil eşleştirmesi yapamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Hidden HVAC** | Ön konsol yarığında gizlenmiş, görünür hava ızgarası bulunmayan iklimlendirme sistemi. |
| **Coanda Effect** | Bir akışkan jetinin (havanın) yakındaki kavisli bir yüzeye yapışarak akma eğilimi. |
| **Step Motor** | Tam bir dönüşü eşit sayıda küçük adımlara ($1.8^\circ$) bölen fırçasız DC motor. |
| **PID Kontrolcü** | Oransal, İntegral ve Türevsel bileşenlerle hata payını sıfırlayan kontrol algoritması. |
| **Anti-Windup** | İntegral birikiminin sınırlandırılarak sistemin aşırı aşma (Overshoot) yapmasını önleyen koruma. |
| **Microstepping** | Step motor sargı akımlarını sinüzoidal sürerek $1.8^\circ$'lik adımı 16 veya 32 mikro adıma bölme. |
| **Thermal Mass** | Kabin içi hava, koltuklar ve camların ısıyı tutma ve geçirme kapasitesi. |
| **Dual Fluidic Jet** | İki hava akımının çarpışma açısıyla yönlendirildiği ızgarasız hava üfleme teknolojisi. |
| **Settling Time** | Sıcaklığın hedef değerin $\%2$ tolerans bandına oturması için geçen süre. |
| **Cabin Overheat Protection** | Park halindeki Tesla'nın sıcaklığı $40^\circ\text{C}$'yi geçince klimayı otomatik açan güvenlik modu. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 ızgarasız minimalist ve fütüristik tasarım     | • Dokunmatik ekrana bağımlılık (Fiziksel tuş olmaması)|
| • Coanda jetleri ile hava akımını bölme yeteneği      | • Yazılım donması durumunda iklimlendirme ayarının    |
| • 1.2 µs ultra hızlı RTOS PID kontrol döngüsü         |   ekrandan yapılamaması (Korumalı D-Bus gerektirir)   |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Koltuktaki yolcu algılama sensörlerine göre hava    | • Aşırı tozlu ortamlarda dar hava yarıklarının        |
|   akımını otomatik kişiselleştirme                    |   tıkanma riski (HEPA filtre koruması zorunludur)     |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Hidden HVAC & PID Kontrol Mimarisi

```
[ Dokunmatik Ekran Dokunma Koordinatları (x, y) ]
                        |
                        v
     [ Hava Jeti Açı Hesaplayıcı (Pitch & Yaw) ]
                        |
                        | Flap Açısı -> Step Motor Darbeleri (1.8°/Step)
                        v
     [ Step Motor Sürücüleri (Mikro-Adımlama) ]
                        |
                        v
   [ Çift Jet Coanda Hava Çarpıştırma Menfezi ]
                        |
                        v
[ Kabin İçi Hava Sıcaklığı T(t) ] <--- [ PID Kontrolcü (Kp, Ki, Kd) ]
                                                ^
                                                | Hata e(t) = T - T_hedef
                                        [ Sıcaklık Sensörleri ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana HVAC PID simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
