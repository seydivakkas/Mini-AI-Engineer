# 🚗 Tesla FSD Otonom Sürüş | Gün 87: Güç Dönüştürücü Simülasyonu: LLC Rezonans Dönüştürücü ve SiC MOSFET Güç Kaybı

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![LLC-Converter](https://img.shields.io/badge/Topology-LLC%20Resonant%20Converter-red.svg?style=flat-square)](https://en.wikipedia.org/wiki/Resonant_converter)
[![SiC-MOSFET](https://img.shields.io/badge/Semiconductor-Silicon%20Carbide%20(SiC)-blue.svg?style=flat-square)](https://www.tesla.com)
[![Efficiency](https://img.shields.io/badge/Efficiency-98.7%25%20Ultra--High-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"87. günümüze hoş geldin stajyer!  
> Tesla Supercharger V4 kabinlerinin ve araç içi 800V Onboard Charger (OBC) ünitelerinin kompakt ve hafif olabilmesinin arkasında güç elektroniği devrimi yatar: **LLC Rezonans Dönüştürücü ve Silisyum Karbür (SiC) MOSFET'ler**!  
> Geleneksel silisyum (Si) transistörler yüksek frekanslarda anahtarlama yaparken aşırı ısınır ve güç kaybeder.  
> Tesla bu kaybı sıfırlamak için **ZVS (Zero Voltage Switching - Sıfır Gerilimde Anahtarlama)** topolojisini kullanır:  
> 1. **265 kHz Yüksek Rezonans:** $L_r = 15\ \mu\text{H}$ ve $C_r = 24\ \text{nF}$ LC tank devresi ile transformatör ve bobin boyutları $\%70$ küçülür.  
> 2. **ZVS Yumuşak Anahtarlama:** MOSFET tam sıfır volttayken açılır; böylece anahtarlama enerjisi ($E_{\text{sw}}$) $\%85$ oranında buharlaşır!  
> 3. **SiC Düşük İletim Direnci ($15\text{ m}\Omega$):** Yüksek jonksiyon sıcaklıklarında ($125^\circ\text{C}$) bile minimum $I^2 R$ ısınması üretir.  
> 4. **%98.7 Rekor Verimlilik:** Yüzlerce kilovatlık güç neredeyse sıfır ısı kaybıyla doğrudan bataryaya aktarılır.  
> Bugün Tesla'nın güç elektroniği mühendisliği şaheseri olan LLC rezonans güç katı simülasyonunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. LLC Rezonans Frekansı

$$f_r = \frac{1}{2 \pi \sqrt{L_r \cdot C_r}} = \frac{1}{2 \pi \sqrt{15 \times 10^{-6} \times 24 \times 10^{-9}}} \approx 265.26\ \text{kHz}$$

### 2. Sıcaklığa Bağlı SiC MOSFET İletim Kaybı (H-Köprüsü)

$$R_{\text{ds(on)}}(T_j) = R_{\text{ds(on)}}(25^\circ\text{C}) \cdot \left[ 1 + 0.005 \cdot (T_j - 25) \right]$$

$$P_{\text{conduction}} = 4 \cdot \left( I_{\text{rms}}^2 \cdot R_{\text{ds(on)}}(T_j) \right)$$

### 3. ZVS (Sıfır Gerilimde Anahtarlama) ile Anahtarlama Kaybı

$$P_{\text{switching}} = 4 \cdot \left( E_{\text{sw\_zvs}} \cdot f_{\text{sw}} \right), \quad E_{\text{sw\_zvs}} = 0.15 \cdot E_{\text{sw\_nominal}}$$

$$\eta_{\text{converter}} = \frac{P_{\text{out}}}{P_{\text{out}} + P_{\text{conduction}} + P_{\text{switching}} + P_{\text{magnetic}}} \times 100\% \ge 98.5\%$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
$800\text{V}$ yüksek voltajlı batarya mimarisinde ve $500\text{ kW}$ Supercharger güç kabinlerinde minimum termal ısı yayılımı ve maksimum güç yoğunluğu sağlamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Ağır ve Hantal Transformatörler:** 265 kHz çalışma frekansı sayesinde transformatörlerin fiziksel ağırlığını $\%70$ azalttı.
- **Termal Aşırı Isınma:** ZVS yumuşak anahtarlama ile anahtarlama kayıplarını $\%85$ düşürerek soğutma ihtiyacını dramatik seviyede azalttı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Geniş Yük Aralıklarında Frekans Kontrolü:** Çok hafif yüklerde rezonans frekansının üzerine çıkıldığında ZVS kaybı yaşanabilir (Burst mode kontrolü gerektirir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Sert Anahtarlamalı Tam Köprü (Hard-Switching Full Bridge):** Çok yüksek anahtarlama kayıplarına sahiptir ve verim $\%93$'ün üzerine çıkamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **LLC Converter** | İki bobin ($L_r, L_m$) ve bir kondansatörden ($C_r$) oluşan yüksek verimli rezonant DC-DC dönüştürücü. |
| **ZVS (Zero Voltage Switching)** | Transistörün üzerindeki gerilim tam sıfır volt iken anahtarlanması tekniği. |
| **Silicon Carbide (SiC)** | Geleneksel silisyuma göre çok daha yüksek gerilim, sıcaklık ve frekansa dayanan geniş bant aralıklı yarı iletken. |
| **$R_{\text{ds(on)}}$** | MOSFET iletimde iken Drain ile Source terminalleri arasındaki omik direnç. |
| **Resonant Tank** | Enerjiyi indüktif ve kapasitif olarak rezonans halinde depolayıp aktaran LC devresi. |
| **Conduction Loss** | Akımın yarı iletkenin iç direnci üzerinden geçerken oluşturduğu $I^2 R$ Joule kaybı. |
| **Switching Loss** | Transistörün açılma ve kapanma geçiş anlarında voltaj-akım çakışmasından doğan kayıp. |
| **Magnetizing Inductance ($L_m$)** | Transformatörün primer sargısının mıknatıslanma endüktansı. |
| **Onboard Charger (OBC)** | Araç içinde bulunan ve şebeke AC elektriğini batarya DC gerilimine çeviren dönüştürücü. |
| **Power Density** | Birim hacim başına üretilen elektriksel güç miktarı ($\text{kW/Litre}$). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %98.7 ultra yüksek DC-DC dönüştürücü verimliliği    | • SiC MOSFET yarı iletken üretim maliyetinin          |
| • 265 kHz rezonans ile minyatür transformatör boyutu  |   geleneksel silisyuma göre daha pahalı olması        |
| • ZVS ile %85 anahtarlama kaybı tasarrufu             |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Gelecek nesil GaN (Galyum Nitrür) ile 1 MHz         | • Yüksek frekanslı anahtarlamada elektromanyetik      |
|   seviyelerine çıkarak güç yoğunluğunu 2 katına çıkarma|   parazit (EMI) gürültüsünün artması                  |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla LLC Rezonant Dönüştürücü Şeması

```
[ 800V DC Giriş Gerilimi ]
            |
            v
[ 4x SiC MOSFET H-Köprüsü ] ---> [ 265 kHz ZVS Anahtarlama ]
                                             |
                                             v
                           [ L_r (15uH) + C_r (24nF) Rezonans Tankı ]
                                             |
                                             v
                                  [ Yüksek Frekans Trafosu ]
                                             |
                                             v
                                 [ SiC Doğrultucu Köprüsü ]
                                             |
                                             v
                          [ %98.7 VERİMLE 800V BATARYAYA ŞARJ ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana LLC dönüştürücü simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
