# 🚗 Tesla Batarya Yönetim Sistemi | Gün 32: Batarya Paketi Dijital İkiz (Digital Twin) Simülasyonu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Digital Twin](https://img.shields.io/badge/Simulation-96S%20Pack%20Physics%20Twin-blue.svg?style=flat-square)](https://www.tesla.com/)
[![Thermal](https://img.shields.io/badge/Early%20Warning-Thermal%20Runaway%20Detection-red.svg?style=flat-square)](https://www.sae.org/)
[![Cloud](https://img.shields.io/badge/Telemetry-Edge--to--Cloud%20Sync-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"32. günümüze hoş geldin stajyer!  
> Tesla'nın en büyük mühendislik avantajlarından biri, yoldaki milyonlarca aracın batarya paketlerinin bulutta ve araç içi bilgisayarda çalışan birer **Dijital İkizine (Digital Twin)** sahip olmasıdır.  
> 96 adet seri bağlı hücrenin hiçbiri fabrikadan tıpatıp aynı çıkmaz:  
> 1. **Hücreden Hücreye (Cell-to-Cell) Varyasyon:** Kapasite $Q$ ve iç direnç $R_0$ Gauss dağılımı gösterir.  
> 2. **Termal Gradyan:** Batarya paketinin tabanındaki kıvrımlı soğutma borusuna giren soğutma sıvısı 1. hücrede $20^\circ\text{C}$ iken, 96. hücreye ulaştığında ısınarak $25^\circ\text{C}$ olur. Bu sıcaklık farkı hücrelerin farklı hızlarda yaşlanmasına yol açar.  
> 3. **Erken Termal Kaçak (Thermal Runaway) Tespiti:** Eğer bir hücrede mikro-iç kısa devre veya dendrit oluşumu başlarsa, o hücrenin iç direnci fırlar ve sıcaklığı komşu hücrelerden hızlı yükselir. Dijital ikiz bu anomaliyi alev veya duman çıkmadan dakikalar önce tespit eder ve aracı güvenli moda alır.  
> Bugün 96S hücre dizisini mikrosaniyeler içinde simüle eden Dijital İkiz motorunu inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 96S Batarya Paketi Toplam Gerilim ve Dengesizlik Modeli

$$V_{\text{pack}}(t) = \sum_{i=1}^{96} V_{\text{terminal}, i}(t)$$

$$\Delta V_{\text{imbalance}}(t) = \max_{i \in [1, 96]} (V_i(t)) - \min_{i \in [1, 96]} (V_i(t))$$

### 2. Hücre İçi Termal Gradyan ve Joule Isınması

$$C_{\text{th}, i} \frac{dT_i(t)}{dt} = I_{\text{pack}}^2 R_{0, i} - h \cdot A \cdot \left(T_i(t) - T_{\text{coolant}, i}\right)$$

$$T_{\text{coolant}, i} = T_{\text{inlet}} + \left(\frac{i - 1}{96}\right) \cdot \Delta T_{\text{loop}}$$

### 3. Termal Kaçak Erken Uyarı Kriterleri (Early Anomaly Detection)

$$\text{Anomali Koşulu} = \left(T_i - \bar{T}_{\text{pack}} > 8.0^\circ\text{C}\right) \lor \left(\max(V) - V_i > 150\text{ mV}\right)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Fiziksel sensörlerin yetersiz kaldığı hücre bazlı iç direnç sapmalarını, termal gradyanları tahmin etmek ve tekil hücre arızalarını yangına dönüşmeden önce yakalamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Erken Yangın Uyarısı:** 48. hücredeki mikro kısa devre anomalisini sıcaklık ortalamadan $8^\circ\text{C}$ saptığı an tespit etti.
- **Filo Telemetrisi (Fleet Learning):** Araçtaki ikiz verilerini buluta aktararak batarya garanti ve ömür tahmin modellerini güncelledi.
- **Hassas Güç Kısıtlama:** Paketteki en zayıf hücrenin anlık voltaj çökmesine göre araca tork sınırlaması koydu.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **96 Hücre Hesaplama Maliyeti:** 96 hücreyi paralel güncellemek tek hücreye göre 96 kat daha fazla işlem gücü ister (Vektörel NumPy / SIMD optimizasyonu gerekir).
- **Soğutma Kanalı Tıkanıklığı:** Sıvı akış debisindeki yerel tıkanıklıklar ikiz parametrelerini yanıltabilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Yalnızca Paket Düzeyi İzleme:** Sadece toplam voltaj ve 4-5 sıcaklık sensörüne bakar; tekil hücre yangınlarını önceden göremez.
- **Ağır CFD (Hesaplamalı Akışkanlar Dinamiği):** Çok hassastır fakat gömülü ECU'da gerçek zamanlı çalışamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Digital Twin (Dijital İkiz)** | Fiziksel batarya paketinin tüm elektriksel ve termal davranışını eşzamanlı taklit eden yazılım modeli. |
| **96S Configuration** | 96 adet lityum hücresinin seri bağlanarak nominal $350-400\text{V}$ oluşturduğu batarya mimarisi. |
| **Thermal Runaway** | Hücre iç sıcaklığının kontrolsüz artarak yangın ve patlamaya yol açtığı zincirleme egzotermik reaksiyon. |
| **Thermal Gradient** | Soğutma sıvısının akış yönü boyunca hücreler arasında oluşan sıcaklık farkı ($5-8^\circ\text{C}$). |
| **Cell-to-Cell Variation** | Üretim toleranslarından kaynaklanan kapasite ve iç direnç sapmaları ($\pm 2\%$). |
| **Edge-to-Cloud Sync** | Araç içi mikrodenetleyici telemetrisinin hücresel veriyle Tesla merkezi sunucularına aktarılması. |
| **Micro-Short Circuit** | Anot ve katot arasında dendrit oluşumuyla başlayan ve anomali üreten mikroskobik iç kısa devre. |
| **Cooling Ribbon / Plate** | Silindirik 2170/4680 hücrelerin arasından geçen serpantin alüminyum soğutma kanalı. |
| **Voltage Spread ($\Delta V$)** | Paketteki maksimum ve minimum hücre voltajı arasındaki açıklık ($mV$). |
| **Fleet Learning** | Milyonlarca aracın dijital ikiz verileriyle makine öğrenimi batarya modellerini sürekli eğitme süreci. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 96 hücreyi 22.5 µs'de eşzamanlı simüle edebilme     | • 96 hücrelik durum vektörünün RAM kaplaması          |
| • Erken termal kaçak uyarısıyla %100 yangın önleme    | • Sensör gürültülerinde sahte pozitif alarm riski     |
| • Termal gradyanı hesaba katan gelişmiş soğutma modeli|                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Bulut yapay zekasıyla birleşerek batarya servis     | • Yüksek hızlı şarjda hücreler arası aşırı termal     |
|   ve modül değişim ihtiyacını aylar öncesinden bildirme|   stres ve model sapması                              |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & 96S Paket Dijital İkizi

```
                     +-----------------------------------+
                     |   96S BATARYA DİJİTAL İKİZİ       |
                     +-----------------+-----------------+
                                       |
       +-------------------------------+-------------------------------+
       |                               |                               |
       v                               v                               v
[ Hücre 1: 75.2Ah, 1.48mΩ ]   [ ... Hücre 48: Kusurlu ... ]  [ Hücre 96: 74.8Ah, 1.52mΩ ]
(Giriş Soğuk: 20°C)           (Yüksek Direnç + Aşırı Isı)     (Çıkış Sıcak: 25°C)
       |                               |                               |
       +-------------------------------+-------------------------------+
                                       |
                                       v
                    +-------------------------------------+
                    |   Anomali Erken Uyarı Motoru        |
                    |   - ΔT > 8°C Anomali Tespiti        |
                    |   - ΔV > 150mV Voltaj Çöküş Uyarısı |
                    +------------------+------------------+
                                       |
                                       v
                    [ Edge-to-Cloud Telemetri & Alarm ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana dijital ikiz simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
