# 🚗 Tesla FSD Otonom Sürüş | Gün 79: CCS / NACS (J3400) Şarj Protokolü ve ISO 15118 Tak-Çalıştır (Plug & Charge) Şifreleme

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NACS](https://img.shields.io/badge/Standard-SAE%20J3400%20(NACS)-red.svg?style=flat-square)](https://www.sae.org/)
[![ISO 15118](https://img.shields.io/badge/Protocol-ISO%2015118--20%20Plug%26Charge-blue.svg?style=flat-square)](https://www.iso.org/)
[![PLC](https://img.shields.io/badge/Carrier-HomePlug%20GreenPHY%20PLC-green.svg?style=flat-square)](https://www.homeplug.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"79. günümüze hoş geldin stajyer!  
> Geleneksel benzin istasyonlarında kart okutmak, fiş almak veya mobil uygulamalarla boğuşmak tam bir zaman kaybıdır!  
> Tesla bu süreci otomotiv dünyasının standardı haline gelen **NACS (SAE J3400) ve ISO 15118 Tak-Çalıştır (Plug & Charge)** mimarisiyle sıfır sürtünmeli hale getirdi:  
> 1. **Control Pilot (CP) Donanımsal Durum Geçişi:** Soket takıldığı anda gerilim $12\text{V}$'tan (State A) $9\text{V}$'a (State B) düşer ve $\%5$ PWM darbesi başlar.  
> 2. **HomePlug GreenPHY PLC:** Güç hattı üzerinden yüksek hızlı Ethernet/IP haberleşmesi kurulur.  
> 3. **TLS 1.3 ve Sözleşme Sertifikası:** Araç ve istasyon arasında şifreli tünel açılarak araç VIN numarası ve ödeme sözleşmesi $2\text{ saniye}$ içinde doğrulanır.  
> 4. **Otomatik Kontaktör Kapanması ($6\text{V}$ / State C):** Doğrulama tamamlandığı an kontaktörler kapanır ve 500 kW güç akışı başlar; sürücünün hiçbir tuşa basmasına gerek kalmaz!  
> Bugün tüm otomotiv devlerinin (Ford, GM, Rivian, Mercedes) benimsediği NACS ISO 15118 şarj protokol motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Control Pilot (CP) Gerilim Bölücü Modeli

$$V_{\text{cp}} = V_{\text{source}} \cdot \frac{R_{\text{vehicle}}}{R_{\text{source}} + R_{\text{vehicle}}}, \quad V_{\text{source}} = 12\text{V}, \ R_{\text{source}} = 1000\ \Omega$$

- **State A (Boşta):** $R_{\text{vehicle}} = \infty \implies V_{\text{cp}} = 12.0\text{V}$
- **State B (Takıldı):** $R_{\text{vehicle}} = 2740\ \Omega \implies V_{\text{cp}} = 9.0\text{V}$
- **State C (Şarj Aktif):** $R_{\text{vehicle}} = 882\ \Omega \implies V_{\text{cp}} = 6.0\text{V}$

### 2. ISO 15118-20 Şifreli Doğrulama Kanunu

$$\text{SessionAuth} = \text{VerifyX509}\left( \text{ContractCert}, \ \mathbf{K}_{\text{TeslaOEM}} \right) \land \text{CheckVIN}(\text{VehicleVIN})$$

### 3. V2G Enerji Akış Gücü

$$P_{\text{charging}} = \frac{V_{\text{req}} \cdot I_{\text{req}}}{1000} \quad [\text{kW}]$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Sürücünün kart okutma, ekran menüleri veya mobil uygulama açma zorunluluğunu ortadan kaldırarak şarj sürecini saniyeler içinde tam otonom başlatmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Ödeme Dolandırıcılığı ve Klonlama:** TLS 1.3 ve X.509 dijital sertifikaları ile kredi kartı skimming ve sahte araç kimliklerini %100 engelledi.
- **Kuzey Amerika Şarj Standardı:** Hantal CCS1 soketleri yerine hafif, kompakt ve tek portlu NACS (J3400) standardını yerleştirdi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **PLC Sinyal Gürültüsü:** Yüksek akımlı DC anahtarlama sırasında güç hattında oluşan elektromanyetik parazit (EMI) PLC paket kaybına yol açabilir (Güçlü filtreleme gerektirir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **RFID Kart Okuyucu:** Kart çalınabilir veya unutulabilir; akıllı şebeke V2G entegrasyonu sunamaz.
- **Manuel Mobil Uygulama:** Hücresel çekim olmayan yeraltı otoparklarında çalışmaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **NACS (SAE J3400)** | Tesla'nın AC ve DC şarjı aynı kompakt konnektörde birleştiren Kuzey Amerika şarj standardı. |
| **ISO 15118** | Elektrikli araçlar ile şarj istasyonları arasındaki çift yönlü dijital iletişim ve güvenlik standardı. |
| **Plug & Charge (PnC)** | Kablo takıldığı anda kimlik doğrulama ve faturalandırmanın otomatik gerçekleştiği teknoloji. |
| **Control Pilot (CP)** | Şarj durumunu analog voltaj (12V/9V/6V) ve dijital PWM ile belirleyen sinyal pini. |
| **GreenPHY PLC** | Şarj güç kabloları üzerinden yüksek frekanslı veri taşıyan Powerline Communication yongası. |
| **TLS 1.3** | Araç ile istasyon arasındaki iletişimi uçtan uca şifreleyen modern kriptografik katman. |
| **V2G (Vehicle-to-Grid)** | Aracın gerektiğinde şebekeye geri elektrik vermesini sağlayan çift yönlü protokol. |
| **CurrentDemandReq** | Aracın batarya yönetim sisteminden şarj istasyonuna anlık gerilim ve akım talep mesajı. |
| **Contract Certificate** | Araç sahibinin faturalandırma hesabını doğrulayan X.509 dijital sertifikası. |
| **SECC / EVCC** | Şarj İstasyonu İletişim Kontrolcüsü (SECC) ve Elektrikli Araç İletişim Kontrolcüsü (EVCC). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Sıfır kullanıcı eforu ile 2 saniyede Tak-Çalıştır   | • PLC yongalarının yüksek frekanslı DC gürültüsünden  |
| • TLS 1.3 ve X.509 ile banka seviyesinde şifreleme    |   etkilenme riski                                     |
| • 2 µs ultra hızlı mesajlaşma döngüsü                 |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tüm elektrikli araç markalarının NACS'a geçmesiyle  | • Sertifika iptal listelerinin (CRL/OCSP) internet    |
|   küresel Supercharger gelirlerinin katlanması        |   kesintilerinde gecikmesi                            |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla NACS & ISO 15118 İletişim Şeması

```
[ Araç Şarj Girişi (EVCC) ]                        [ Tesla Supercharger (SECC) ]
            |                                                     |
            | 1. Soket Takıldı -> CP Gerilimi: 12V -> 9V (State B)|
            |---------------------------------------------------->|
            |                                                     |
            | 2. HomePlug GreenPHY PLC Eşleşmesi (IPv6 Link-Local)|
            |<===================================================>|
            |                                                     |
            | 3. TLS 1.3 El Sıkışması & Sözleşme Sertifikası Doğrulama
            |<--------------------------------------------------->|
            |                                                     |
            | 4. Onaylandı! CP Gerilimi: 9V -> 6V (State C Aktif) |
            |---------------------------------------------------->|
            |                                                     |
            | 5. V2G CurrentDemandReq (400V / 500A -> 200 kW)     |
            |<===================================================>|
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana NACS ISO 15118 simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
