# 🚗 Tesla FSD Otonom Sürüş | Gün 82: Tesla Autobidder Algoritmik Enerji Ticareti: Frekans Düzenleme ve Arbitraj

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Autobidder](https://img.shields.io/badge/Trading-Tesla%20Autobidder%20Platform-red.svg?style=flat-square)](https://www.tesla.com/support/energy/autobidder)
[![Arbitrage](https://img.shields.io/badge/Finance-Spot%20Market%20Arbitrage-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/Arbitrage)
[![Battery-Life](https://img.shields.io/badge/Optimization-Degradation%20Aware%20Trading-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"82. günümüze hoş geldin stajyer!  
> Tesla yalnızca bir otomobil üreticisi veya donanım şirketi değildir; aynı zamanda dünyanın en gelişmiş algoritmik enerji ticaret yazılımlarından birine sahiptir: **Tesla Autobidder**!  
> Elektrik spot piyasalarında (örneğin CAISO, ERCOT veya EPEX Spot) elektrik fiyatı gün içinde $20\text{ \$/MWh}$'den pik saatlerde $250\text{ \$/MWh}$'e veya negatif fiyatlara fırlayabilir.  
> Autobidder, yapay zeka ile bu fiyat dalgalanmalarını tahmin eder ve batarya varlıklarını (Megapack ve Powerwall filoları) otonom olarak yönetir:  
> 1. **Fiyat Arbitrajı:** Gece rüzgar enerjisinin bol olduğu saatlerde elektriği $20\text{ \$/MWh}$'den satın alıp bataryayı şarj eder.  
> 2. **Pik Satış:** Akşam talebin zirve yaptığı saatlerde elektriği $250\text{ \$/MWh}$'den şebekeye satar.  
> 3. **Batarya Yıpranma Modeli (Degradation-Aware):** Her şarj/deşarj döngüsünün lityum-iyon hücrelerine maliyetini ($40\text{ \$/MWh}$) hesaba katar; marjinal kar bırakmayan hiçbir işlemi yapmaz!  
> 4. **Otonom Karlılık:** Tek bir Megapack XL ünitesinden yılda yüzbinlerce dolar ek gelir üretir.  
> Bugün Tesla'nın milyar dolarlık enerji ticaret beyni olan Autobidder algoritmasını kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Kümülatif Net Arbitraj Karı Formülasyonu

$$\Pi_{\text{net}} = \sum_{t \in \mathcal{T}_{\text{sell}}} \left( P_{\text{spot}}(t) - C_{\text{degradation}} \right) \cdot E_{\text{sold}}(t) - \sum_{t \in \mathcal{T}_{\text{buy}}} P_{\text{spot}}(t) \cdot E_{\text{bought}}(t)$$

$$C_{\text{degradation}} = 40.0\ \text{\$/MWh} \quad (0.04\ \text{\$/kWh Lityum Hücre Amortismanı})$$

### 2. Otonom Ticaret Karar Matrisi

$$\text{Action}(P_{\text{spot}}, \ \text{SoC}) = \begin{cases} \text{DISCHARGE (SELL)}, & P_{\text{spot}} > 150\text{ \$/MWh} \land \text{SoC} > 20\% \\ \text{CHARGE (BUY)}, & P_{\text{spot}} < 30\text{ \$/MWh} \land \text{SoC} < 95\% \\ \text{STANDBY / REGULATION}, & \text{Diğer Durumlar} \end{cases}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Statik batarya depolama sistemlerini pasif birer maliyet unsuru olmaktan çıkarıp, elektrik piyasasında aktif arbitraj ve frekans hizmetleriyle yüksek getirili finansal varlıklara dönüştürmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Manuel Ticaret Yavaşlığı:** İnsan tüccarların saniyelik fiyat dalgalanmalarını yakalama yetersizliğini ortadan kaldırarak tam otomatik milisaniyelik teklif (Bidding) sağladı.
- **Kör Batarya Yaşlanması:** Batarya yıpranma maliyetini hesaba katarak gereksiz mikro döngülerle bataryanın ömrünün erken tükenmesini engelledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Piyasa Fiyat Tahmin Hataları:** Beklenmeyen hava durumu değişimleri spot piyasa fiyat tahminlerinde sapmalara yol açabilir (Sürekli makine öğrenimi eğitimi gerektirir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Sabit Zamanlı Şarj (Zaman Sayacı):** Her gece 02:00'de şarj edip 18:00'de satmak (Dinamik fiyat zirvelerini ve negatif fiyat fırsatlarını kaçırır).

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Tesla Autobidder** | Enerji depolama varlıklarını toptan elektrik piyasalarında otonom ticarete sokan yazılım platformu. |
| **Energy Arbitrage** | Elektriği fiyatın en ucuz olduğu saatte alıp en pahalı olduğu saatte satarak kar etme. |
| **Day-Ahead Market** | Bir sonraki günün 24 saati için elektrik fiyatlarının belirlendiği gün öncesi piyasası. |
| **Real-Time Market** | 5 dakikalık aralıklarla anlık arz-talep dengesine göre oluşan gerçek zamanlı elektrik piyasası. |
| **Battery Degradation** | Şarj ve deşarj döngüleri sonucunda batarya kapasitesinin zamanla geri döndürülemez kaybı. |
| **Levelized Cost of Storage (LCOS)** | Depolanan bir megavat-saat enerjinin amortisman dahil toplam birim maliyeti. |
| **Negative Pricing** | Aşırı yenilenebilir enerji üretimi nedeniyle elektrik üreticilerinin alıcılara para ödediği durum. |
| **FCR (Frequency Containment Reserve)** | Şebeke frekansını sabit tutmak için sağlanan yüksek değerli yedek güç hizmeti. |
| **Wholesale Electricity Market** | Elektrik üreticileri, toptancılar ve tüccarların işlem yaptığı toptan elektrik borsası. |
| **High-Frequency Bidding** | Saniyeler içinde optimize edilmiş teklif paketlerini borsa API'lerine iletme süreci. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Batarya yıpranma maliyetini dikkate alan kar motoru | • Elektrik piyasası regülasyonlarının her eyalet/ülke |
| • Günlük yüzlerce dolar net arbitraj getirisi         |   için farklı ve karmaşık olması                      |
| • 1.2 µs ultra hızlı otonom teklif üretimi            |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Milyonlarca Powerwall ev bataryasını Sanal Enerji   | • Aşırı volatiliteli piyasalarda ani iletim hattı     |
|   Santrali (VPP) olarak Autobidder'a bağlama          |   tıkanıklıkları (Transmission Congestion)            |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Autobidder Algoritmik Ticaret Akış Şeması

```
[ Toptan Elektrik Spot Piyasası (Spot Price Stream) ]
                         |
                         v
      [ Autobidder Fiyat ve Talep Tahmin Modeli ]
                         |
                         | 1. Fiyat Analizi: P_spot vs Eşikler ($30 / $150)
                         | 2. Batarya Amortisman Marjı: (P_spot > $40/MWh)
                         v
          [ Otonom Karar Matrisi ]
          /          |           \
         /           |            \
  P_spot < $30   30-150 $         P_spot > $150
       |             |                  |
       v             v                  v
[ ŞEBEKEDEN AL ] [ STANDBY ]      [ ŞEBEKEYE SAT ]
(Ucuz Elektrik)  (Bekleme)        (Maksimum Kar)
       \             |                  /
        +------------+-----------------+
                     v
       [ KÜMÜLATİF GÜNLÜK NET KAR: +$1,540 USD ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Autobidder simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
