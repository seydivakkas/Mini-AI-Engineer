# 🚗 Tesla Teşhis ve Araç Ağları | Gün 19: UDS (ISO 14229) & OBD-II Hata Kodu Okuma

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Protocol](https://img.shields.io/badge/Diagnostics-ISO%2014229%20%2F%20UDS-red.svg?style=flat-square)](https://www.iso.org/)
[![DTC Standard](https://img.shields.io/badge/OBD--II-ISO%2015031--6%20%2F%20SAE%20J2012-orange.svg?style=flat-square)](https://www.sae.org/)
[![Transport](https://img.shields.io/badge/Transport-ISO%2015765--2%20(ISO--TP)%20%2F%20DoIP-blue.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"19. günümüze hoş geldin stajyer!  
> Bir Tesla'nın batarya yönetim sistemi (BMS), motor invertörü veya otopilot bilgisayarı bir anormallik sezdiğinde ne olur?  
> Gösterge panelinde anında sarı bir uyarı ışığı yanar veya servis randevusu önerilir. Peki servis teknisyeni veya uzaktan Tesla Fleet Cloud bu arızayı nasıl okur ve ECU parametrelerini nasıl yeniden yapılandırır?  
> Cevap: **UDS (Unified Diagnostic Services - ISO 14229)** ve **OBD-II DTC** standardıdır!  
> 1. **UDS Servis Mimarisi (Client-Server):** Teşhis cihazı (Tester) bir istek gönderir (Örn: `0x22` ReadDataByIdentifier), hedef ECU bu isteği işleyip pozitif yanıt (`SID + 0x40`) veya negatif hata kodu (NRC - Negative Response Code: `0x7F SID NRC`) ile yanıtlar.  
> 2. **3-Baytlık DTC Kod Yapısı (ISO 15031-6):** Örneğin `P0A1F-00` kodunda en yüksek 2 bit arıza kategorisini ($00 = \text{Powertrain/P}, 01 = \text{Chassis/C}, 10 = \text{Body/B}, 11 = \text{Network/U}$) belirlerken son bayt arıza tipini (Fault Type Byte) tanımlar.  
> 3. **Seed-Key Güvenlik Mekanizması (0x27 SecurityAccess):** Kritik ECU parametrelerinin yetkisiz kişilerce değiştirilmesini önlemek için dinamik kriptografik Challenge-Response el sıkışması zorunludur.  
> 4. **DoIP (Diagnostics over IP):** Tesla araçlarında servis teşhisi klasik 500k CAN yerine Gigabit Ethernet omurgası üzerinden DoIP ile milisaniyeler mertebesinde tamamlanır.  
> Bugün tüm bu teşhis katmanını eksiksiz olarak inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 3-Baytlık DTC Ayrıştırma Matematiği (ISO 15031-6 / ISO 14229)
3 baytlık ham DTC dizisi $B_1, B_2, B_3$ olmak üzere:

$$\text{Kategori Kodu} = (B_1 \gg 6) \ \& \ 0\text{x}03 \implies \{0: \text{'P'}, 1: \text{'C'}, 2: \text{'B'}, 3: \text{'U'}\}$$

$$D_1 = (B_1 \gg 4) \ \& \ 0\text{x}03, \quad D_2 = B_1 \ \& \ 0\text{x}0\text{F}$$

$$D_3 = (B_2 \gg 4) \ \& \ 0\text{x}0\text{F}, \quad D_4 = B_2 \ \& \ 0\text{x}0\text{F}, \quad \text{FTB} = B_3$$

$$\text{Formatlı DTC} = \text{Kategori} + D_1 + \text{Hex}(D_2) + \text{Hex}(D_3) + \text{Hex}(D_4) + \text{"-"} + \text{Hex}(B_3)$$

### 2. Challenge-Response Seed-Key Güvenlik Doğrulaması
ECU tarafından rastgele üretilen $S \in \{0, 1\}^{32}$ (Seed) ve paylaşılan gizli anahtar $K_{\text{secret}}$ ile:

$$K_{\text{response}} = \text{Truncate}_{32}\Big(\text{SHA256}(S \parallel K_{\text{secret}})\Big)$$

İstemcinin gönderdiği anahtar $K_{\text{client}}$ ile $K_{\text{response}}$ eşleşirse güvenlik kilidi (`0x27 0x02`) açılır.

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Araçtaki onlarca ECU'nun durumunu izlemek, hata geçmişini raporlamak, sensör kalibrasyonlarını yapmak ve yazılım güncellemelerini başlatmak için evrensel otomotiv standardı olan UDS (ISO 14229) kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Evrensel Teşhis Dili:** Her sensör ve motor için tescilli (proprietary) protokoller yerine standart servis kimlikleri (`0x22`, `0x19`, `0x27`, `0x2E`) sağlandı.
- **Detaylı Arıza İzolasyonu:** 3-baytlık DTC ve 8-bit durum maskesi ile arızanın anlık mı, onaylanmış mı yoksa geçmiş mi olduğu ayırt edildi.
- **Siber Güvenlik:** Kriptografik Seed-Key ile motor haritası veya akü limitleri yetkisiz müdahalelere karşı kilitlendi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Bant Genişliği Bağımlılığı:** Klasik CAN üzerinde ISO-TP akış kontrolü nedeniyle büyük veri transferleri (örneğin firmware yükleme) dakikalar sürebilir. Bu sebeple Tesla DoIP (Ethernet) omurgasını tercih eder.
- **Zaman Aşımı Riskleri:** $P_2$ ve $P_2^*$ sunucu zamanlama limitleri aşılırsa teşhis oturumu anında düşer.

### 4. Alternatifler Nelerdir? (Alternatives)
- **OBD-II Standart Servisleri (SAE J1979):** Yalnızca emisyon ve temel motor parametrelerini okur; derin ECU konfigürasyonu ve güvenlik kilidi sunmaz.
- **REST / gRPC (DoIP üstü):** Yeni nesil merkezi araç bilgisayarlarında UDS'in yerini almaya başlayan modern API protokolleri.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **UDS (ISO 14229)** | Otomotiv elektronik kontrol üniteleri için birleşik teşhis hizmetleri standardı. |
| **DTC (Diagnostic Trouble Code)** | Araç alt sistemlerindeki arızaları tanımlayan 3-baytlık standart hata kodu. |
| **DID (Data Identifier)** | ECU içerisindeki sensör ve durum parametrelerine erişim sağlayan 16-bitlik veri kimliği (Örn: `0xF190` VIN). |
| **NRC (Negative Response Code)** | İsteğin reddedilme nedenini bildiren UDS hata kodu (Örn: `0x33` SecurityAccessDenied). |
| **Seed-Key** | ECU güvenlik kilidini açmak için kullanılan Challenge-Response kriptografik el sıkışma yöntemi. |
| **ISO-TP (ISO 15765-2)** | CAN veri yolunda 8 bayttan büyük teşhis paketlerinin parçalanıp birleştirilmesini sağlayan taşıma katmanı. |
| **DoIP (ISO 13400)** | UDS teşhis mesajlarının TCP/IP ve Ethernet üzerinden yüksek hızda iletilmesi standardı. |
| **Diagnostic Session** | ECU'nun Default, Extended veya Programming modlarında çalışmasını sağlayan oturum yönetimi. |
| **Status Mask** | DTC'nin onaylanma, aktiflik ve uyarı ışığı durumunu gösteren 8-bitlik durum bayrağı. |
| **TesterPresent (`0x3E`)** | Teşhis oturumunun ve güvenlik kilidinin zaman aşımına uğramasını önleyen periyodik kalp atışı (heartbeat). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tüm otomotiv endüstrisinde evrensel standart        | • Klasik CAN üzerinde ISO-TP çoklu çerçeve ek yükü   |
| • Seed-Key ile güçlü siber güvenlik kilit mekanizması | • Yanlış oturum zamanlamasında oturumun düşmesi       |
| • DoIP ile Gigabit Ethernet omurgasında ultra hız     |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tesla Fleet Cloud üzerinden uzaktan OTA teşhis ve   | • Güvenlik tohumunun (Seed) zayıf entropi ile         |
|   arıza tespiti entegrasyonu                          |   üretilmesi durumunda kırılma riski                  |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & Teşhis Akışı

```
+-------------------+           UDS Request (0x22 0xF190)          +----------------------+
|                   | -------------------------------------------> |                      |
|   Tesla Service   |                                              |   Tesla BMS Core     |
|   Tool / Tester   | <------------------------------------------- |   ECU (Server)       |
|   (Client)        |           Positive Resp (0x62 0xF190 VIN)    |                      |
+-------------------+                                              +----------------------+
          |                                                                   |
          | --- 0x27 0x01 (Request Seed) -----------------------------------> |
          | <--- 0x67 0x01 [Seed: 0xA5 0x5A 0x3C 0xC3] ---------------------- |
          |                                                                   |
          | --- 0x27 0x02 [Key = SHA256(Seed || Secret)] -------------------> |
          | <--- 0x67 0x02 (Security Unlocked E_OK) ------------------------- |
          |                                                                   |
          | --- 0x2E 0x0103 [Write Autopilot Config] ------------------------> |
          | <--- 0x6E 0x0103 (Write OK) ------------------------------------- |
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana teşhis akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
