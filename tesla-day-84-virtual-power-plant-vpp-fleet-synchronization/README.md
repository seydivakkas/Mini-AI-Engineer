# 🚗 Tesla FSD Otonom Sürüş | Gün 84: Sanal Enerji Santrali (Virtual Power Plant - VPP) ve Dağıtık Akıllı Şebeke

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![VPP](https://img.shields.io/badge/Grid-Tesla%20Virtual%20Power%20Plant-red.svg?style=flat-square)](https://www.tesla.com/vpp)
[![Fleet](https://img.shields.io/badge/Fleet-50k%20Powerwalls%20Aggregated-blue.svg?style=flat-square)](https://www.tesla.com/powerwall)
[![Grid-Emergency](https://img.shields.io/badge/Blackout-Zero%20Grid%20Collapse-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"84. günümüze hoş geldin stajyer!  
> Geleneksel dünyada bir şehrin elektriği kesilmek üzereyken büyük bir nükleer veya doğalgaz santralinin devreye girmesi beklenir.  
> Tesla ise bu paradigmayı tamamen değiştirdi: **Tesla Virtual Power Plant (VPP - Sanal Enerji Santrali)**!  
> Bir eyaletteki (örneğin Kaliforniya veya Güney Avustralya) $50.000$ evde bulunan Tesla Powerwall bataryaları internet üzerinden senkronize edilir:  
> 1. **Devasa Sanal Kapasite:** $50.000 \times 5.0\text{ kW} = 250\text{ MW}$ anlık güç ve $675\text{ MWh}$ depolama kapasitesi; tam anlamıyla devasa bir nükleer santral bloğuna eşdeğerdir!  
> 2. **150 MW Acil Şebeke Deşarjı:** Şebeke operatörü acil durum sinyali gönderdiğinde, 50.000 batarya aynı saniye içinde deşarja başlar.  
> 3. **Adil ve Hafif Yük Dağılımı:** $150\text{ MW}$ talep ev başına sadece $3.0\text{ kW}$ düşürülerek karşılanır; hiçbir batarya zorlanmaz.  
> 4. **Müşteri Rezerv Kilidi (%20):** Kullanıcının evdeki buzdolabı ve aydınlatma elektriği asla riske atılmaz (%20 altı kilitlenir).  
> Bugün binlerce evin akıllı şebekeye güç bastığı dağıtık VPP filo orkestrasyon algoritmasını kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Kullanılabilir Toplam VPP Filo Gücü

$$P_{\text{vpp\_total}} = \sum_{i=1}^N P_{\text{max}, i} \cdot \mathbb{I}(\text{SoC}_i > \text{SoC}_{\text{reserve}, i})$$

### 2. Ünite Başına Düşen Güç Dağılımı (Dispatch)

$$P_{\text{unit}} = \min\left( P_{\text{max}}, \ \frac{P_{\text{grid\_demand}}}{N_{\text{eligible}}} \right), \quad P_{\text{grid\_demand}} = 150.0\ \text{MW}, \quad N = 50,000$$

### 3. Kullanıcı Batarya Rezerv Garantisi

$$\text{SoC}_i(t) \ge 20.0\%, \quad \forall i \in \{1, \dots, N\}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Merkezi fosil santrallere milyonlarca dolar harcamak yerine, evlerde kurulu olan binlerce dağıtık küçük bataryayı bulut üzerinden birleştirip anlık şebeke stabilizasyonu sağlamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Yaz Sıcaklarında Şebeke Çökmeleri:** Sıcak hava dalgalarında klimaların açılmasıyla oluşan şebeke aşırı yükünü dakikalar içinde gidererek bölgesel elektrik kesintilerini (Rolling Blackouts) engelledi.
- **Ev Sahiplerine Pasif Gelir:** Powerwall sahiplerine şebekeye verdikleri destek için yüzlerce dolar ödül kazandırdı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **İletişim Ağ Gecikmesi (Latency & Jitter):** 50.000 cihaza internet üzerinden aynı anda mesaj göndermek LTE/Wi-Fi paket kayıpları yaratabilir (MQTT ve WebSocket optimizasyonu gerekir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Dizel Jeneratör Parkları:** Çok kirli, gürültülü ve yakıt maliyeti aşırıdır.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **VPP (Virtual Power Plant)** | Dağıtık enerji kaynaklarının (DER) merkezi bir santral gibi tek elden yönetildiği sanal santral ağı. |
| **DER (Distributed Energy Resources)** | Çatı üstü güneş panelleri, ev bataryaları ve elektrikli araçlar gibi dağıtık enerji kaynakları. |
| **Fleet Aggregation** | Binlerce bağımsız uç cihazın telemetri ve kapasitelerinin tek bir havuzda toplanması. |
| **Demand Response (Talep Yanıtı)** | Şebekenin sıkıştığı anlarda tüketimi azaltma veya sisteme güç enjekte etme programı. |
| **Reserve SoC** | Elektrik kesintisi durumunda ev sahibinin kullanımı için rezerve edilen dokunulmaz batarya yüzdesi. |
| **Rolling Blackout** | Şebekenin çökmesini engellemek için mahallelerin sırayla elektriksiz bırakılması durumu. |
| **Fleet Dispatch** | Toplam güç talebini filodaki uygun bataryalara milisaniyeler içinde paylaştırma işlemi. |
| **Peaker Plant** | Yalnızca elektrik talebinin zirve yaptığı saatlerde çalıştırılan yüksek maliyetli santral. |
| **Cryptographic Authorization** | Filoya gönderilen acil deşarj komutlarının sahteciliğe karşı dijital olarak imzalanması. |
| **Grid Balancing** | Şebeke frekansı ve voltajının dengelenmesi için sunulan yan hizmetler (Ancillary Services). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 50.000 Powerwall ile 250 MW / 675 MWh sanal güç     | • Ev tipi internet bağlantılarının (Wi-Fi) kopma riski|
| • 1.3 ms ultra hızlı vektörize dispatch motoru        | • Batarya ömrünü korumak için sıkı rezerv sınırları   |
| • Sıfır hava kirliliği ve sıfır yakıt maliyeti        |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Milyonlarca Tesla aracının (V2G) VPP ağına dahil    | • Siber saldırganların filo genelinde toplu deşarj    |
|   edilerek gigavat seviyesinde güç elde edilmesi      |   emri göndermeye çalışması (Zero-Trust imzası şart)  |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla VPP Dağıtık Akıllı Şebeke Şeması

```
[ Şebeke Operatörü (ISO / Grid Dispatcher) ]
                    |
                    | "150 MW Acil Güç İhtiyacı!"
                    v
    [ Tesla Cloud VPP Filo Orkestratörü ]
                    |
                    | 1. Filo Kapasite Taraması: 50.000 Powerwall (>%20 SoC)
                    | 2. Vektörize Dağıtım: 3.0 kW / Batarya
                    v
    [ Güvenli MQTT / WebSocket Yayın Kanalı ]
     /              |              |              \
    v               v              v               v
[ Powerwall #1 ] [ Powerwall #2 ] ... [ Powerwall #50000 ]
  (3.0 kW Bas)     (3.0 kW Bas)          (3.0 kW Bas)
     \              |              |              /
      +-------------+--------------+-------------+
                    v
    [ TOPLAM 150 MW GÜÇ ŞEBEKEYE BASILDI | SIFIR KESİNTİ ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana VPP filo simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
