# 🚗 Tesla Faz 3 Büyük Capstone | Gün 33: Tam Kapsamlı Tesla BMS ve Çekiş Kontrol Mimarisi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Phase%203%20Grand%20Capstone-blue.svg?style=flat-square)](https://www.tesla.com/)
[![Powertrain](https://img.shields.io/badge/Powertrain-96S%20BMS%20+%20FOC%20+%20SVPWM-orange.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/Safety-ISO%2026262%20ASIL--D%20HVIL-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Tebrik ve Büyük Capstone Notu

> *"Gözlerine inanabiliyor musun stajyer? FAZ 3'ü (Gün 23 - Gün 33) başarıyla tamamladın ve 33. günde Faz 3'ün Büyük Capstone Projesindeyiz!  
> Bir Tesla'nın gazına bastığınızda veya One-Pedal Drive ile yavaşladığınızda arkada çalışan tüm elektrokimyasal, elektromanyetik ve termal sistemler kusursuz bir senfoni halinde çalışır:  
> 1. **96S Batarya Fiziği & EKF (Gün 23 & 24):** 2-RC Thevenin devresi ve 3-durumlu Genişletilmiş Kalman Filtresi ile $\%0.4$ altı hata ile SoC kestirilir.  
> 2. **RLS SoH & Dengeleme (Gün 25 & 26):** Yaşlanma ve $R_0$ iç direnci çevrimiçi izlenir, hücreler arası gerilim uyumsuzluğu ($\Delta V < 5\text{ mV}$) tutulur.  
> 3. **Octovalve Termal Yönetim (Gün 27):** $COP = 3.5$ ısı pompası motor kayıp ısısını toplar, bataryayı optimum $45^\circ\text{C}$ sıcaklıkta tutar.  
> 4. **10 kHz FOC & SVPWM (Gün 28 & 29):** Clarke/Park dönüşümleri ve 7-segment SVPWM ile DC baradan $\%15.5$ daha yüksek gerilimle $350\text{ Nm}$ tork üretilir.  
> 5. **Rejenerasyon & Tork Harmanlama (Gün 30):** $75\text{ kW}$ kinetik enerji bataryaya şarj edilir, fren balata aşınması $\%90$ önlenir.  
> 6. **ASIL-D HVIL & Dijital İkiz (Gün 31 & 32):** $88\text{ Hz}$ güvenlik döngüsü ve Pyrofuse ile $< 2\text{ ms}$ acil izolasyon sağlanır; 96 hücreli ikiz yangın anomalisini dakikalar önce yakalar.  
> Tüm bu 10 dev modülü tek bir merkezi güç aktarma çekirdeğinde birleştirdik!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Bütünleşik Güç Aktarma Enerji Dengesi

$$P_{\text{battery}}(t) = V_{\text{pack}}(t) \cdot I_{\text{pack}}(t) = \left(\frac{T_e(t) \cdot \omega_m(t)}{\eta_{\text{drivetrain}}}\right) + P_{\text{aux, octovalve}}$$

### 2. 96S ECM Terminal Gerilimi ve EKF SoC Çözümü

$$V_{\text{pack}}(t) = \sum_{i=1}^{96} \left[ OCV(SoC_i(t)) - I_{\text{pack}}(t) R_{0, i}(T) - V_{RC1, i}(t) - V_{RC2, i}(t) \right]$$

### 3. FOC Tork Üretimi ve Dinamik İvmelenme

$$T_e = \frac{3}{2} p \left[ \psi_f i_q + (L_d - L_q) i_d i_q \right]$$

$$\frac{dv(t)}{dt} = \frac{T_e(t) / r_{\text{wheel}} - F_{\text{drag}} - F_{\text{friction}}}{M_{\text{vehicle}}}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Batarya hücresi elektrokimyasından invertör SiC MOSFET anahtarlamasına kadar tüm güç aktarma bileşenlerini mikro saniye hassasiyetinde senkronize çalıştırmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Uçtan Uca Enerji Optimizasyonu:** Rejenerasyon, Octovalve ısı pompası ve MTPA tork kontrolüyle araç menzili $\%35+$ maksimize edildi.
- **ASIL-D Fonksiyonel Güvenlik:** HVIL, Pyrofuse ve EKF 3-sigma sınırlarıyla kaza ve termal kaçak riskleri sıfırlandı.
- **Ludicrous Performans:** 0'dan 120 km/h hıza sarsıntısız, $350\text{ Nm}$ lineer torkla ulaşıldı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Çok Çekirdekli Dağıtım:** 10 kHz FOC döngüsü ile 100 Hz BMS döngüsü farklı mikrodenetleyici çekirdeklerine (Core 0 / Core 1) ayrılmalıdır.
- **Isıl Doyum:** Sürekli 350 Nm tam gaz sürüşte batarya ve invertör sıcaklığı tork kısılmasına (Thermal Derating) yol açabilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Ayrık Modüllü ECU Mimarisi:** Her bileşen (BMS, Inverter, HVAC) ayrı kutudadır; CAN veri yolu gecikmeleri sebebiyle gerçek zamanlı koordinasyon zayıftır.
- **Merkezi Zonal Mimari (Tesla Yaklaşımı):** Tek bir güçlü çekirdekte tüm güç aktarma fonksiyonlarını birleştiren bu capstone tasarımı endüstrinin geleceğidir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Powertrain Core** | Batarya, invertör, motor ve termal sistemleri tek bir döngüde yöneten merkezi çekirdek. |
| **96S Architecture** | 96 seri bağlı hücreyle nominal 350-400V oluşturan standart elektrikli araç paketi. |
| **Dual-Polarization 2-RC** | Hızlı yük transferini ve yavaş difüzyonu modelleyen iki RC dalına sahip batarya eşdeğer devresi. |
| **EKF SoC Tracker** | Gürültülü voltaj ve akım verilerinden batarya doluluk oranını kestiren Kalman filtresi. |
| **IPM-SynRM Motor** | Mıknatıs ve relüktans torkunu birleştirerek yüksek verim üreten Tesla çekiş motoru. |
| **SVPWM Modulator** | SiC MOSFET'leri 6 sektörde anahtarlayarak $\%15.5$ daha yüksek voltaj üreten PWM motoru. |
| **Octovalve Heat Pump** | Motor kayıp ısısını bataryaya taşıyan 8-yollu patentli termal yönetim sistemi. |
| **One-Pedal Regen** | Gaz pedalının bırakılmasıyla aracı durduran ve 75 kW güç geri kazanan elektrikli fren. |
| **Pyrofuse** | Kaza anında $< 2\text{ ms}$ içinde yüksek gerilimi mikro patlamayla ayıran emniyet sigortası. |
| **ASIL-D** | Otomotiv fonksiyonel güvenlik standardında en yüksek güvenlik bütünlük seviyesi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Faz 3'teki 10 dev modülün uçtan uca entegrasyonu    | • 10 kHz ve 100 Hz çoklu hız döngüsü yönetimi        |
| • 0-120 km/h ivmelenme ve tek pedallı tam rejen duruş | • Aşırı yük altında termal sınırlandırma gereksinimi  |
| • 3.25 µs ultra hızlı merkezi güç aktarma döngüsü     |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Cybertruck ve Megapack 800V mimarilerine doğrudan   | • Çift sensör arızalarında yedekli (Redundant)        |
|   ölçeklenebilir modüler çekirdek                     |   algoritmaların devreye girmesi zorunluluğu          |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Faz 3 Büyük Capstone Sistem Mimarisi

```
                                  [ SÜRÜCÜ GİRDİLERİ ]
                             (Gaz Pedalı, Fren, Navigasyon)
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               TESLA FAZ 3 MERKEZİ POWERTRAIN & BMS ÇEKİRDEĞİ                      |
+-----------------------------------------------------------------------------------+
|  1. GÜVENLİK (ASIL-D): HVIL 88 Hz PWM, İzolasyon (>200kΩ), Pyrofuse (<2ms)       |
|  2. BMS ÇEKİRDEĞİ: 96S 2-RC ECM, EKF SoC (%0.4 RMSE), RLS SoH, Hücre Dengeleme   |
|  3. TERMAL: Octovalve 8-Yollu Isı Pompası (COP=3.5), Motor Isı Geri Kazanımı     |
|  4. ÇEKİŞ MOTORU: 10 kHz FOC (Clarke/Park/MTPA), 350 Nm IPM-SynRM Motoru         |
|  5. İNVERTÖR MODÜLASYONU: 7-Segment Simetrik SVPWM (+%15.5 Vdc, 1.5µs Dead-time) |
|  6. FRENLEME: Tek Pedallı Rejenerasyon (One-Pedal Drive), 75 kW Şarj, Hold Modu   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
                              [ 3-Faz PMSM Çekiş Motoru ]
                              [ 96S 400V Batarya Paketi ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Capstone sürüş simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
