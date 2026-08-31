# 🚗 Tesla FSD Otonom Sürüş | Gün 77: FAZ 7 BÜYÜK CAPSTONE: Tesla V12 Konsol ve Telemetri Simülatörü

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Phase 7](https://img.shields.io/badge/Capstone-Phase%207%20Master%20Complete-red.svg?style=flat-square)](https://www.tesla.com/)
[![Qt6/QML](https://img.shields.io/badge/UI-Qt6%20Declarative%2060FPS-blue.svg?style=flat-square)](https://www.qt.io/)
[![Full-Stack](https://img.shields.io/badge/Stack-9%20Subsystems%20Integrated-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"TEBRİKLER STAJYER!  
> Tesla Yazılım Mühendisliği 99 Günlük Master Yol Haritası'nın **7. FAZINI (GÜN 67 - 77: TESLA V12 INFOTAINMENT, QT6/QML, D-BUS & OTA GÜNCELLEME) BAŞARIYLA TAMAMLADIN!**  
> Bu büyük Capstone gününde, son 11 günde inşa ettiğimiz 9 kritik donanım, çekirdek, güvenlik ve grafik alt sistemini tek bir devasa canlı konsol ve telemetri motorunda birleştirdik:  
> 1. **Qt6/QML Deklaratif UI (Gün 67):** 60 FPS çift yönlü telemetri veri bağlama.  
> 2. **3D GPU World View Render (Gün 68):** FSD algı kutularının ve şeritlerin MVP matrisi ile $1920\times 1080$ ekranına izdüşümü.  
> 3. **Linux D-Bus IPC (Gün 69):** Gövde kontrol modülüyle (kapılar, farlar, camlar) mikrosaniyelik asenkron RPC haberleşmesi.  
> 4. **PipeWire ARNC Ses Kalkanı (Gün 70):** 180° ters faz aktif yol gürültüsü engelleme ($>60\text{ dB}$).  
> 5. **Fast-Boot & Secure Boot (Gün 71 & 72):** $1.47\text{ s}$ açılış ve donanımsal Root of Trust TPM 2.0 güven zinciri.  
> 6. **OTA A/B Rollback Engine (Gün 73):** Çift slot işletim sistemi ve sıfır brick güvencesi.  
> 7. **Chromium Seccomp-BPF Sandbox (Gün 74):** Sıfır Güven (Zero Trust) web tarayıcısı ve CAN-Bus izolasyonu.  
> 8. **BLE + UWB Phone Key (Gün 75):** Işık hızı Time-of-Flight milimetrik mesafe ve röle hırsızlığı kalkanı.  
> 9. **HVAC Dokunmatik PID & Stepper (Gün 76):** Gizli Coanda hava menfezi ve kabin sıcaklığı termal kontrolü.  
> Şimdi bu 9 alt sistemin senkronize çalıştığı Faz 7 Master Capstone motorunu ateşliyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 3D GPU Model-View-Projection (MVP) İzdüşümü

$$\mathbf{p}_{\text{clip}} = \mathbf{P} \cdot \mathbf{V} \cdot \mathbf{M} \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}, \quad u = \frac{x_{\text{ndc}} + 1}{2} W, \quad v = \frac{1 - y_{\text{ndc}}}{2} H$$

### 2. ARNC Akustik Yıkıcı Girişim Kanunu

$$y_{\text{anti}}(t) = -x(t) \implies x(t) + y_{\text{anti}}(t) = 0 \implies \Delta L_{\text{attenuation}} \ge 60\text{ dB}$$

### 3. UWB Işık Hızı Mesafe ve Doğrulama

$$d = t_{\text{tof}} \cdot c, \quad \text{Unlock} \iff d \le 2.0\text{ m} \iff t_{\text{tof}} \le 6.67\text{ ns}$$

### 4. Kabin Termal PID Kapalı Döngüsü

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}, \quad \frac{dT_{\text{cabin}}}{dt} = -\alpha u(t) + \beta (T_{\text{ambient}} - T_{\text{cabin}})$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Tesla V12 Infotainment işletim sisteminin tüm ekran, grafik, güvenlik, ses ve gövde fonksiyonlarını tek bir yüksek performanslı, çökmeye dayanıklı mimaride birleştirmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Alt Sistem Karmaşası:** Farklı frekanslarda çalışan 9 alt modülün (60 FPS grafik, 1000 Hz ses, 10 Hz PID) asenkron IPC ve paylaşımlı bellek ile sıfır gecikmeyle haberleşmesini sağladı.
- **Siber Güvenlik & Konfor:** Seccomp sandbox ve UWB kalkanı ile aracı siber saldırılardan ve hırsızlıktan korurken, ARNC ve HVAC ile üst düzey kabin konforu sundu.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **CPU/GPU Yükü:** Çok sayıda alt sistemin eşzamanlı çalışması güçlü çok çekirdekli APU (AMD Ryzen Embedded) gerektirir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Ayrık ECU Mimarisi (Geleneksel Otomotiv):** Her fonksiyon için ayrı kutu/çip kullanılır (Ağır kablo demeti, yüksek maliyet ve yavaş güncelleme).

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Tesla V12 Infotainment** | Araç ekranını, navigasyonu, multimedyayı ve araç kontrollerini yöneten Linux tabanlı sistem. |
| **Full-Stack Capstone** | Bir mühendislik fazındaki tüm teorik ve pratik modüllerin bir araya getirildiği zirve projesi. |
| **Model-View-Projection** | 3D uzaydaki nesneleri 2D ekran piksellerine dönüştüren grafik matris çarpımı. |
| **D-Bus IPC** | Linux süreçleri arasında metot çağrısı ve sinyal gönderimi sağlayan mesaj veri yolu. |
| **PipeWire ARNC** | Düşük gecikmeli ses sunucusu üzerinde çalışan 180° ters faz aktif gürültü engelleme. |
| **Secure Boot RoT** | Donanım kökünden başlayarak her önyükleme katmanını imzalayan güvenlik zinciri. |
| **A/B Dual Partition** | Kesintisiz güncelleme ve sıfır brick sağlayan çift işletim sistemi bölümü. |
| **Seccomp Sandbox** | Web tarayıcısının araç kontrol katmanına sızmasını engelleyen sistem çağrısı filtresi. |
| **UWB ToF** | Işık hızı uçuş süresiyle milimetrik mesafe ölçen röle saldırılarına bağışık telsiz protokolü. |
| **Thermal PID** | Anti-windup özellikli kabin sıcaklığı ve hava jeti flap kontrol döngüsü. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 9 kritik alt sistemin kusursuz senkronizasyonu      | • Yüksek bellek ve GPU bant genişliği ihtiyacı        |
| • 25 µs tam yığın RTOS döngü süresi (40,000 FPS cap) | • Karmaşık hata ayıklama (Debug) süreçleri            |
| • Sıfır güvenlik açığı (Zero Trust + RoT + UWB)       |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Otonom taksi (Robotaxi) filosunda yolculara tam     | • Aşırı sıcak kabin ortamlarında termal kısılma       |
|   otonom eğlence ve kontrol konsolu sunumu            |   (Thermal Throttling) riski                          |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Faz 7 Master Capstone Mimarisi

```
             [ TESLA V12 FULL-STACK INFOTAINMENT SİMÜLATÖRÜ ]
                                    |
    +-------------------------------+-------------------------------+
    |                               |                               |
[ KULLANICI ARAYÜZÜ ]      [ GÜVENLİK & KERNEL ]           [ KABİN & İLETİŞİM ]
- Qt6/QML Telemetri (60 FPS)- Secure Boot TPM RoT (Gün 72) - D-Bus Body IPC (Gün 69)
- 3D GPU Render (Gün 68)    - OTA A/B Rollback (Gün 73)    - PipeWire ARNC (Gün 70)
- Dokunmatik HVAC (Gün 76)  - Seccomp Sandbox (Gün 74)     - UWB Phone Key (Gün 75)
    |                               |                               |
    +-------------------------------+-------------------------------+
                                    v
            [ %100 SENKRONİZE V12 KONSOL ÇALIŞIYOR ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Faz 7 Master Capstone simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
