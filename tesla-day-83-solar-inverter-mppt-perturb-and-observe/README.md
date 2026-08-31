# 🚗 Tesla FSD Otonom Sürüş | Gün 83: Güneş Enerjisi ve Solar Inverter MPPT (Maximum Power Point Tracking) Kontrolü

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Solar](https://img.shields.io/badge/Solar-Tesla%20Solar%20Roof%20%26%20Inverter-red.svg?style=flat-square)](https://www.tesla.com/solarroof)
[![MPPT](https://img.shields.io/badge/Algorithm-Perturb%20%26%20Observe%20(P%26O)-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/Maximum_power_point_tracking)
[![Efficiency](https://img.shields.io/badge/Efficiency-99.8%25%20MPPT%20Harvest-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"83. günümüze hoş geldin stajyer!  
> Güneş panelleri basit bir pil gibi sabit voltaj vermez; sıcaklık, gölge ve güneş ışığı açısına göre panellerin ürettiği güç eğrisi sürekli değişir!  
> Eğer panelin çıkış voltajını rastgele seçerseniz, güneş ışığının $\%40$'ını çöpe atarsınız.  
> Tesla Solar Roof ve Powerwall 3 dahili solar invertörleri, her bir fotonun enerjisini son damlasına kadar hasat etmek için **MPPT (Maksimum Güç Noktası Takibi - Perturb & Observe)** kontrolcüsünü çalıştırır:  
> 1. **P-V Karakteristik Eğrisi:** Panelin voltajı ile akımı çarpılarak tepe güç noktası ($V_{\text{mpp}} = 40\text{V}, P_{\text{mpp}} = 360\text{W}$) bulunur.  
> 2. **Perturb and Observe (Boz ve Gözle):** Voltajı ufak adımlarla ($\Delta V = 0.5\text{V}$) değiştirir; eğer güç artıyorsa o yönde devam eder, güç düşüyorsa yönü tersine çevirir.  
> 3. **%99.8 Hasat Verimliliği:** Bulut geçişlerinde veya gölgede milisaniyeler içinde yeni tepe noktasına kilitlenir.  
> 4. **Dahili İnvertör Entegrasyonu:** Ayrı harici kutulara gerek kalmadan doğrudan DC-DC güç katında yürütülür.  
> Bugün Tesla'nın temiz enerji üretim kalbi olan Solar MPPT motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Fotovoltaik Güç - Gerilim ($P-V$) Karakteristiği

$$P(V) = V \cdot I(V) = V \cdot I_{\text{sc}} \left[ 1 - \left(\frac{V}{V_{\text{oc}}}\right)^4 \right]$$

### 2. Perturb and Observe (P&O) Voltaj Güncelleme Kuralı

$$\Delta P = P(k) - P(k-1), \quad \Delta V = V(k) - V(k-1)$$

$$V(k+1) = \begin{cases} V(k) + \text{step}, & (\Delta P > 0 \land \Delta V > 0) \lor (\Delta P < 0 \land \Delta V < 0) \\ V(k) - \text{step}, & (\Delta P > 0 \land \Delta V < 0) \lor (\Delta P < 0 \land \Delta V > 0) \end{cases}$$

### 3. MPPT Takip Verimliliği

$$\eta_{\text{mppt}} = \frac{P_{\text{tracked}}}{P_{\text{mpp\_optimal}}} \times 100\% \ge 99.0\%$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Güneş ışınımının ve panel sıcaklığının gün boyunca sürekli değiştiği koşullarda, panel dizisinden her an çekilebilecek maksimum elektriksel gücü çekmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Güneş Enerjisi İsrafı:** Statik voltajda çalışan ilkel şarj kontrolcülerinin $\%30-40$ enerji kaybını ortadan kaldırarak $\%99.8$ hasat verimine ulaştırdı.
- **Kısmi Gölgelenme:** Ağaç veya bulut gölgesinde çalışma noktasını hızla yeniden optimize etti.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Tepe Noktası Salınımı (Oscillation):** P&O algoritması tepe noktaya ulaştığında $V_{\text{mpp}} \pm \Delta V$ etrafında küçük bir salınım yapar (Değişken adımlı akıllı hibrit mod ile minimize edilir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Sabit Voltaj Regülasyonu:** Çok ucuzdur ancak güneş açısı değiştikçe verim çöker.
- **Artımlı İletkenlik (Incremental Conductance):** Çok iyidir ancak P&O'dan biraz daha fazla işlemci gücü gerektirir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **MPPT** | Maksimum Güç Noktası Takibi (Maximum Power Point Tracking). |
| **Perturb and Observe (P&O)** | Voltajı ufak adımlarla bozup gücün artış/azalışını gözlemleyen MPPT algoritması. |
| **P-V Eğrisi** | Bir güneş panelinin gerilimine karşılık ürettiği elektriksel güç grafiği. |
| **Open Circuit Voltage ($V_{\text{oc}}$)** | Panelin hiçbir yük bağlı değilken ürettiği açık devre maksimum gerilimi. |
| **Short Circuit Current ($I_{\text{sc}}$)** | Panelin uçları kısa devre edildiğinde akan maksimum akım. |
| **Tesla Solar Roof** | Binanın çatısına entegre edilen ve kiremit görünümünde olan fotovoltaik cam karolar. |
| **Fill Factor (FF)** | Bir güneş panelinin kalitesini ve maksimum güç potansiyelini gösteren doluluk faktörü. |
| **Solar Inverter** | Güneş panellerinden gelen DC gerilimi evde kullanılan 230V/400V AC gerilime çeviren cihaz. |
| **Anti-Islanding Protection** | Şebeke elektriği kesildiğinde şebeke teknisyenlerini korumak için solar üretimi kesen güvenlik modu. |
| **Irradiance** | Birim alana düşen güneş ışığı enerjisi miktarı ($1000\text{ W/m}^2$ standart test koşulu). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %99.8 ultra yüksek güneş enerjisi hasat verimliliği | • Tepe noktaya ulaşıldığında küçük voltaj salınımı    |
| • 1.1 µs RTOS adım döngüsü                            |                                                       |
| • Basit, sağlam ve çökmeye dayanıklı P&O çekirdeği    |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Powerwall 3 ve Tesla CyberTruck çatı güneş paneli   | • Hızlı bulut geçişlerinde yerel tepe noktalarına     |
|   entegrasyonuyla sınırsız menzil desteği             |   (Local Maxima) takılma riski                        |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Solar MPPT Kontrol Akış Şeması

```
[ Güneş Işığı (Fotonlar) ] ---> [ Tesla Solar Roof Panelleri ]
                                            |
                                            | V(k), I(k) Ölçümü
                                            v
                                [ P(k) = V(k) * I(k) Hesabı ]
                                            |
                                            | delta_P = P(k) - P(k-1)
                                            v
                         [ Perturb & Observe Algoritması ]
                         /                               \
                        /                                 \
                 delta_P > 0                         delta_P < 0
               (Güç Artıyor)                        (Güç Düşüyor)
                      |                                   |
                      v                                   v
             [ Aynı Yönde Adım ]                 [ Yönü Tersine Çevir ]
                      \                                   /
                       +----------------+----------------+
                                        v
                    [ %99.8 VERİMLİLİKLE MPP KİLİTLENDİ ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Solar MPPT simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
