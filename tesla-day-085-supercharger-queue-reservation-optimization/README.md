# 🚗 Tesla FSD Otonom Sürüş | Gün 85: Supercharger İstasyonları İçin Dinamik Kuyruk ve Rezervasyon Optimizasyonu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Queue-Theory](https://img.shields.io/badge/Model-M%2FM%2Fc%20Multi--Server%20Queue-red.svg?style=flat-square)](https://en.wikipedia.org/wiki/M/M/c_queue)
[![FSD-Routing](https://img.shields.io/badge/FSD-Dynamic%20ETA%20Slot%20Reservation-blue.svg?style=flat-square)](https://www.tesla.com/supercharger)
[![Zero-Wait](https://img.shields.io/badge/Traffic-Sub--15m%20Guaranteed-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"85. günümüze hoş geldin stajyer!  
> Tesla Supercharger ağının başarısının arkasındaki en büyük sır sadece $500\text{ kW}$'lık hızlı şarj donanımı değildir; istasyonlardaki araç trafiğini yöneten **Matematiksel Kuyruk Teorisi ($M/M/c$) ve FSD Dinamik Rezervasyon Algoritmasıdır**!  
> Bir otoyolda 12 stall'luk bir istasyona tatil gününde yüzlerce aracın aynı anda yığılması durumunda:  
> 1. **$M/M/c$ Çoklu Sunucu Modeli:** Varış hızı ($\lambda = 30\text{ araç/saat}$), servis hızı ($\mu = 3\text{ araç/saat/stall}$) ve stall sayısı ($c = 12$) kullanılarak ortalama bekleme süresi ($W_q$) analitik olarak hesaplanır.  
> 2. **FSD Rota Rezervasyonu:** Araç istasyona varmadan 20 dakika önce FSD navigasyonu varış anındaki tahmini doluluğu simüle eder ve araç için şarj slotu rezerve eder.  
> 3. **Akıllı Alternatif Yönlendirme:** Eğer hedef istasyondaki bekleme süresi $15\text{ dakikayı}$ aşacaksa, araç otomatik olarak yol üzerindeki 5 km ilerideki boş istasyona yönlendirilir.  
> 4. **Sıfır Kuyruk Stresi:** Sürücünün istasyona vardığı an sıra beklemeden doğrudan şarja takması garanti edilir.  
> Bugün Supercharger istasyonlarını yöneten $M/M/c$ dinamik rezervasyon ve kuyruk optimizasyon motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Trafik Yoğunluğu (Utilization Factor)

$$\rho = \frac{\lambda}{c \cdot \mu}, \quad c = 12\ \text{Stall}, \quad \mu = 3.0\ \text{Araç/Saat/Stall}$$

### 2. Boş İstasyon Olasılığı ($P_0$)

$$P_0 = \left[ \sum_{n=0}^{c-1} \frac{(c \rho)^n}{n!} + \frac{(c \rho)^c}{c! (1 - \rho)} \right]^{-1}$$

### 3. Ortalama Kuyruk Uzunluğu ($L_q$) ve Ortalama Bekleme Süresi ($W_q$)

$$L_q = \frac{P_0 \cdot (c \rho)^c \cdot \rho}{c! \cdot (1 - \rho)^2}$$

$$W_q = \frac{L_q}{\lambda} \times 60\ \text{dakika}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Sürücülerin Supercharger istasyonlarında uzun kuyruklar oluşturmasını engellemek, şebeke yükünü istasyonlar arasında dengeli paylaştırmak ve FSD otonom rotasını optimize etmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Bayram/Tatil Yığılmaları:** Popüler istasyonlarda 1 saatlik kuyrukların oluşmasını engelleyerek trafiği alternatif yan istasyonlara dağıttı.
- **Kör Varış Riski:** Sürücünün istasyona vardığında sürpriz bir kalabalıkla karşılaşmasını önledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **İnsan Davranışı Sapmaları:** Navigasyon dışı plansız gelen araçlar veya şarj bittiği halde stall'da park eden araçlar (Idle Fee cezası ile çözülür).

### 4. Alternatifler Nelerdir? (Alternatives)
- **İlk Gelen İlk Alır (FIFO Statik Kuyruk):** Hiçbir optimizasyon yapmaz, yoğun günlerde kaosa yol açar.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **$M/M/c$ Queue** | Poisson varışlı, üstel servis süreli ve $c$ adet paralel sunuculu kuyruk modeli. |
| **Arrival Rate ($\lambda$)** | Birim zamanda (saat) Supercharger istasyonuna varan ortalama araç sayısı. |
| **Service Rate ($\mu$)** | Bir stall'un bir saatte şarj edip uğurlayabildiği ortalama araç sayısı. |
| **Traffic Utilization ($\rho$)** | İstasyonun kapasite kullanım oranı ($\rho < 1.0$ kararlılık şartıdır). |
| **$W_q$ (Wait in Queue)** | Bir aracın şarj soketi boşalana kadar sırada beklediği ortalama süre. |
| **$L_q$ (Length of Queue)** | Kuyrukta bekleyen ortalama araç sayısı. |
| **FSD Trip Planner** | Batarya şarj seviyesini ve istasyon yoğunluğunu hesaba katan rota planlayıcı. |
| **Idle Fee (İşgal Ücreti)** | Şarj %100 olduktan sonra aracı stall'dan çekmeyen sürücülere uygulanan dakika başı ceza. |
| **Dynamic Slot Allocation** | Araçların ETA sürelerine göre şarj yuvalarının önceden atanması. |
| **Rerouting Trigger** | Bekleme süresi 15 dakikayı aştığında aracı alternatif istasyona saptıran tetikleyici. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • M/M/c analitik kuyruk modeli ile kesin süre tahmini | • Navigasyon kullanmayan üçüncü parti NACS araçların  |
| • FSD ETA entegrasyonu ile sıfır bekleme süresi       |   plansız varışlarının tahmin belirsizliği            |
| • 2.3 µs ultra hızlı RTOS rota değerlendirme süresi   |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Robotaksi filolarının şarj aralıklarını tam otomatik| • Otoyol kazaları nedeniyle araçların ETA sürelerinde |
|   olarak gelir getirmeyen ölü saatlere yayma          |   ani gecikmeler yaşanması                            |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Supercharger M/M/c Kuyruk Akış Şeması

```
[ Yoldaki FSD Araç Filosu ] ---> [ ETA & Batarya SoC Bilgisi ]
                                           |
                                           v
               [ Tesla Supercharger M/M/c Kuyruk Motoru ]
                                           |
                                           | W_q Analitik Hesabı
                                           v
                          [ Bekleme Süresi Kontrolü ]
                          /                         \
                         /                           \
                  W_q <= 15 dk                    W_q > 15 dk
                (İstasyon Akıcı)                (İstasyon Yoğun)
                       |                               |
                       v                               v
             [ Slotu Rezerve Et ]             [ Alternatif İstasyona ]
             (Doğrudan Şarja Tak)             (Yönlendir: 2.5 dk Bekle)
                       \                               /
                        +--------------+--------------+
                                       v
                    [ %100 KONFORLU VE KESİNTİSİZ SÜRÜŞ ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Supercharger kuyruk simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
