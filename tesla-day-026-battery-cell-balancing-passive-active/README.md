# 🚗 Tesla Batarya Yönetim Sistemi | Gün 26: Hücre Dengeleme Algoritmaları (Pasif ve Aktif Dengeleme)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Balancing](https://img.shields.io/badge/Balancing-Passive%20Bleed%20%26%20Active%20Inductive-blue.svg?style=flat-square)](https://www.tesla.com/)
[![Efficiency](https://img.shields.io/badge/Active%20Efficiency-88%25%20Charge%20Shuttle-green.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/Thermal-Max%2055%C2%B0C%20Cutoff-red.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"26. günümüze hoş geldin stajyer!  
> Bir elektrikli aracın 96S veya 192S batarya paketinde binlerce hücre seri bağlanmıştır.  
> Üretim toleransları, kimyasal safsızlıklar ve paket içindeki sıcaklık gradyanları yüzünden hücrelerin voltajları ve kapasiteleri zamanla birbirinden ayrışır (**Hücre Uyumsuzluğu - Cell Imbalance**).  
> Burada **Fıçı Yasası (Barrel Effect)** geçerlidir:  
> - Şarj olurken voltajı en yüksek olan hücre ilk önce $4.20\text{V}$ sınırına ulaşır ve BMS tüm paketin şarjını durdurur; diğer hücreler $\%85$'te kalır.  
> - Deşarj olurken voltajı en düşük olan hücre ilk önce $3.00\text{V}$ sınırına iner ve araç gücü keser.  
> Sonuç: Paket kapasitesinin $\%15-20$'si kullanılamaz hale gelir!  
> İki temel dengeleme mimarisi vardır:  
> 1. **Pasif Dengeleme (Passive Bleeding):** Yüksek voltajlı hücreye paralel bir MOSFET anahtarla $33\ \Omega$ direnç bağlanır; fazla enerji $P = V^2 / R$ Joule ısısına dönüştürülüp yakılır ($100-150\text{ mA}$). Ucuzdur fakat ısı yayar.  
> 2. **Aktif Dengeleme (Active Inductive Shuttling):** Enerji yakılmaz; yüksek hücreden endüktif anahtarlamalı konvertörle çekilip en zayıf hücreye $\%88$ verimle transfer edilir ($2.0\text{ A}$).  
> Bugün her iki stratejiyi ve termal güvenlik kesme mantığını kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Fıçı Yasası ve Kullanılabilir Paket Enerjisi

$$Q_{\text{usable}} = \min_{i \in [1, N]} \left(Q_i \cdot SoC_i\right) + \min_{i \in [1, N]} \left(Q_i \cdot (1 - SoC_i)\right)$$

$$\Delta V_{\text{imbalance}} = \max_{i}(V_i) - \min_{i}(V_i)$$

### 2. Pasif Direnç Dengeleme (Passive Bleeding)
Direnç $R_{\text{bleed}} = 33\ \Omega$ üzerinden çekilen anlık akım ve üretilen Joule ısısı:

$$I_{\text{bleed}} = \frac{V_{\text{cell}}}{R_{\text{bleed}}}, \quad P_{\text{bleed}} = \frac{V_{\text{cell}}^2}{R_{\text{bleed}}}$$

$$\Delta SoC_i = -\frac{I_{\text{bleed}} \cdot \Delta t}{Q_{\text{nominal}} \cdot 3600}$$

### 3. Aktif Çift Yönlü Endüktif Enerji Transferi
Kaynak hücreden ($H$) hedef hücreye ($L$) verim $\eta \approx 0.88$ ile aktarılan akım:

$$I_{\text{extracted}} = I_{\text{active}}, \quad I_{\text{delivered}} = \eta \cdot I_{\text{active}}$$

$$P_{\text{loss}} = V_H \cdot I_{\text{active}} \cdot (1 - \eta)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Batarya paketinin kullanılabilir menzilini maksimize etmek, zayıf hücrelerin erken aşırı şarj/aşırı deşarj ile hasar görmesini engellemek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Kullanılamayan Kapasiteyi Kurtardı:** Hücreler arası $80\text{ mV}$ uyumsuzluğu $< 5\text{ mV}$ eşiğine indirerek paketin tam kapasite şarj olmasını sağladı.
- **Hücre Ömrünü Eşitledi:** Zayıf hücrelerin aşırı voltaj stresine maruz kalıp erken yaşlanmasını engelledi.
- **Termal Güvenlik:** Pasif dirençler aşırı ısındığında ($T > 55^\circ\text{C}$) dengelemeyi durduran güvenlik kesicisi eklendi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Pasif Hız Sınırı:** $120\text{ mA}$ düşük akım sebebiyle büyük kapasiteli ($75\text{ Ah}$) hücrelerde dengeleme saatler sürer (Genellikle araç park halindeyken yapılır).
- **Aktif Donanım Maliyeti:** Aktif dengeleme her hücre için trafo/endüktans ve sürücü MOSFET'ler gerektirdiğinden kart maliyetini artırır.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Uçtan Uca Şarj Dengelemesi (Top Balancing):** Yalnızca şarj sonu $\%100$ SoC'ye yaklaşıldığında devreye giren basit pasif strateji.
- **Kapasitif Enerji Transferi:** Anahtarlamalı kapasitörler (Switched Capacitor) ile komşu hücreler arasında şarj paylaşımı.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Cell Balancing** | Seri bağlı batarya hücrelerinin voltaj ve şarj durumlarını eşit seviyeye getirme işlemi. |
| **Cell Imbalance ($\Delta V$)** | Paketteki en yüksek voltajlı hücre ile en düşük voltajlı hücre arasındaki gerilim farkı ($mV$). |
| **Passive Balancing** | Fazla enerjiyi dirençler üzerinde ısıya dönüştürerek yok eden basit ve ucuz yöntem. |
| **Active Balancing** | Yüksek enerjili hücreden düşük enerjili hücreye DC-DC konvertörle enerji aktaran verimli yöntem. |
| **Bleed Resistor ($R_{\text{bleed}}$)** | Pasif dengelemede hücre voltajını düşürmek için paralel bağlanan akım sınırlayıcı direnç. |
| **Barrel Effect (Fıçı Yasası)** | Bir batarya paketinin toplam kapasitesinin en zayıf hücre tarafından sınırlandırılması kuralı. |
| **Top Balancing** | Hücrelerin şarjın en tepe noktasında ($4.20\text{V}$) dengelenmesi stratejisi. |
| **Rest Balancing** | Araç park halinde ve akım sıfırken OCV voltajlarına göre yapılan dengeleme. |
| **Inductive Charge Shuttling** | Enerjiyi manyetik alanda depolayarak bir hücreden diğerine taşıyan aktif dengeleme devresi. |
| **Thermal Cutoff** | Dengeleme dirençlerinin kartı aşırı ısıtmasını önleyen yazılımsal ve donanımsal güvenlik eşiği. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Aktif dengelemede 4.4x daha hızlı ve 8.7x az ısı    | • Pasif dengelemede tüm fazla enerjinin ısıya gitmesi |
| • 1.8 µs ultra hızlı RTOS karar döngüsü               | • Aktif dengelemenin PCB komponent maliyet yüksekliği |
| • 55°C termal koruma kesicisiyle yangın güvenliği     |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Megapack ve ticari depolamada aktif dengeleme ile   | • Arızalı bir dengeleme MOSFET'inin takılı kalarak    |
|   yıllık MWh düzeyinde enerji tasarrufu               |   hücreyi 0V'a kadar tüketmesi (Under-voltage) riski  |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & Dengeleme Devre Şeması

```
   [ Hücre 1 (Yüksek V) ] ----+----[ MOSFET 1 ]----[ R_bleed ]----+
                              |                                  |
                              +---[ Aktif Endüktif Konvertör ]---+
                                              |
                                              v  (%88 Enerji Aktarımı)
   [ Hücre 2 (Düşük V) ]  <-------------------+
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana dengeleme simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
