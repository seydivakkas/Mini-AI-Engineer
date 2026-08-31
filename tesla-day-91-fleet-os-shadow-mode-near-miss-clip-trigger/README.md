# 🚗 Tesla FSD Otonom Sürüş | Gün 91: Tesla Filo İşletim Sistemi (Fleet OS): Milyonlarca Araçtan Telemetri Toplama ve Gölge Öğrenme

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Fleet-OS](https://img.shields.io/badge/Fleet-Tesla%20Fleet%20OS%20Data%20Engine-red.svg?style=flat-square)](https://www.tesla.com)
[![Shadow-Mode](https://img.shields.io/badge/Learning-Shadow%20Mode%20Discrepancy-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/Tesla_Autopilot)
[![Edge-Filter](https://img.shields.io/badge/Edge-15s%20H.265%20Clip%20Trigger-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"91. günümüze hoş geldin stajyer!  
> Tesla'nın otonom sürüş yarışında rakiplerine (Waymo, Cruise) karşı en büyük avantajı milyonlarca araçtan oluşan küresel filosudur.  
> Ancak filodan sürekli 8-kameralı video yüklemeye çalışırsanız petabaytlarca anlamsız otoyol videosu buluta dolar ve hücresel ağlar çöker!  
> Tesla bu veri seçilimini **Fleet OS Gölge Mod (Shadow Mode) ve Akıllı Kenar Tetikleyicileri (Edge Triggers)** ile çözer:  
> 1. **Gölge Mod (Shadow Mode):** FSD arka planda sessizce çalışır ve rotayı tahmin eder; eğer insan sürücünün yaptığı eylem ile FSD'nin tahmini arasında sapma ($> 2.0\text{ m/s}^2$) oluşursa anında tetiklenir.  
> 2. **Sert Fren & Acil Kaçış:** Araç $0.8\text{ g}$ üzerinde sert fren yaptığında veya direksiyon $200^\circ/\text{s}$ üzerinde döndüğünde kaza tehlikesi algılanır.  
> 3. **15 Saniyelik Akıllı Klip:** Olay anının 10 saniye öncesi ve 5 saniye sonrası (8 kamera + tam CAN-Bus telemetrisi) H.265 formatında paketlenir.  
> 4. **Dojo Otomatik Etiketleme:** Araç gece evde Wi-Fi ağına bağlandığında bu kritik klipler Dojo'ya aktarılır ve yapay zekanın "en zorlandığı senaryolar" otomatik eğitilir!  
> Bugün Tesla'nın veri motorunun kalbi olan Filo OS tetikleyicisini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Sert Frenleme (Hard Braking) ve Acil Direksiyon Eşikleri

$$\text{Trigger}_{\text{brake}} = \mathbb{I}(g_{\text{deceleration}} > 0.8\text{ g})$$

$$\text{Trigger}_{\text{steer}} = \mathbb{I}(|\dot{\delta}_{\text{wheel}}| > 200.0^\circ/\text{s})$$

### 2. Gölge Mod (Shadow Mode) İnsan vs FSD İvme Sapması

$$\Delta a_{\text{shadow}} = |a_{\text{human}} - a_{\text{fsd}}| > 2.0\ \text{m/s}^2$$

### 3. Akıllı Klip Zaman Penceresi

$$T_{\text{clip}} = [t_{\text{event}} - 10.0\text{s}, \ t_{\text{event}} + 5.0\text{s}] \implies \text{Toplam } 15\ \text{Saniye}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Sadece yapay zekanın hata yaptığı veya beklenmedik zorlu durumlarla (köpek çıkması, yol çalışması, ani fren) karşılaştığı kritik video anlarını seçip Dojo süperbilgisayarına beslemek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Veri Kirliliği ve Ağ Maliyeti:** Milyonlarca saatlik sıkıcı düz yol videosunu eleyerek veri hacmini $\%99.5$ azalttı.
- **Uç Durum (Corner-Case) Keşfi:** İnsan mühendislerin aklına gelmeyecek nadir sürüş senaryolarını filodan otomatik topladı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Yerel Depolama (Ring Buffer):** Araç Wi-Fi bulana kadar klipleri USB veya araç içi flash bellekte saklamak zorundadır.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Sürekli Video Kaydı:** Her şeyi kaydeder; depolama ve hücresel veri maliyeti milyarlarca dolar tutar.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Shadow Mode** | Otonom sürüş yazılımının aracı fiziksel olarak kontrol etmeden arka planda sessizce simülasyon yapması. |
| **Data Engine** | Filodan veri toplama, otomatik etiketleme, model eğitme ve OTA dağıtımını birleştiren kapalı döngü sistem. |
| **Near-Miss Event** | Kazanın eşiğinden dönülen, ani fren veya direksiyon gerektiren tehlikeli sürüş anı. |
| **Corner Case** | Yapay zekanın eğitim setinde çok nadir rastlanan uç sınır senaryoları. |
| **Edge Trigger** | Araç içi bilgisayarda belirli fiziksel eşikler aşıldığında video kaydını başlatan yerel yazılım kuralı. |
| **Map-Reduce Filter** | Binlerce telemetri akışından sadece kritik olay paketlerini ayrıştıran dağıtık filtreleme. |
| **CAN-Bus Telemetry** | Direksiyon açısı, gaz pedalı, fren basıncı ve tekerlek hızlarını içeren araç içi veri akışı. |
| **H.265 (HEVC)** | Yüksek çözünürlüklü 8-kamera video verilerini düşük boyutta sıkıştıran video kodeki. |
| **Autolabeler** | Toplanan ham video kliplerini Dojo üzerinde 3D uzayda otomatik etiketleyen yapay zeka sistemi. |
| **VIN (Vehicle Identification Number)** | Her Tesla aracının global düzeyde benzersiz şasi kimlik numarası. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Milyonlarca araçtan hedefe yönelik köşe durum hasadı| • Wi-Fi olmayan araçlarda kritik kliplerin günlerce   |
| • 15s H.265 paketleme ile %99.5 veri bant tasarrufu   |   araç içi hafızada bekleme riski                     |
| • 0.65 µs ultra hızlı RTOS olay değerlendirmesi       |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Robotaksi filosu devreye girdikçe veri motorunun    | • Kullanıcı gizlilik politikaları (GDPR) ve kamera    |
|   öğrenme hızının geometrik olarak katlanması         |   yüz/plaka anonimleştirme zorunlulukları             |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Fleet OS Veri Motoru Akış Şeması

```
[ Tesla Filo Araçları (Milyonlarca Araç) ]
                    |
                    | Gölge Mod Telemetrisi (CAN-Bus + IMU)
                    v
    [ Edge Tetikleyici Değerlendirme Çekirdeği ]
    /               |                         \
   /                |                          \
Fren > 0.8g    Direksiyon > 200°/s    İnsan vs FSD Sapma > 2m/s²
   |                |                          |
   +----------------+--------------------------+
                    v
       [ 15 Saniyelik Video & CAN Paketi ]
       (10s Öncesi + 5s Sonrası, 8 Kamera)
                    |
                    v
      [ Evde Wi-Fi Bağlandığında Buluta ]
                    |
                    v
      [ DOJO OTOMATİK ETİKETLEME VE EĞİTİM ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Filo OS tetikleyici simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
