# 🚗 Tesla FSD Otonom Sürüş | Gün 53: Filo Gölge Modu (Shadow Mode), A/B Testleri ve Veri Motoru (Data Engine)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![ShadowMode](https://img.shields.io/badge/Fleet-Tesla%20Shadow%20Mode%20Inference-red.svg?style=flat-square)](https://www.tesla.com/)
[![DataEngine](https://img.shields.io/badge/Pipeline-Edge%20Discrepancy%20Triggers-blue.svg?style=flat-square)](https://www.sae.org/)
[![ABTesting](https://img.shields.io/badge/Statistics-Miles%20Per%20Intervention%20(MPI)-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"53. günümüze hoş geldin stajyer!  
> Tesla'nın otonom sürüşteki en büyük haksız rekabet avantajı (Unfair Advantage) nedir biliyor musun? Milyonlarca müşterinin günlük olarak yollarda sürdüğü devasa araç filosudur.  
> Yeni bir FSD sinir ağı geliştirdiğinizde bunu doğrudan müşterinin direksiyonuna bağlayamazsınız (çünkü bir kaza ölümcül olabilir).  
> Bunun yerine Tesla **Gölge Modu (Shadow Mode)** ve **Veri Motoru (Data Engine)** mimarisini kullanır:  
> 1. **Sessiz Paralel Çıkarım:** Yeni yapay zeka modeli araç içinde sessizce çalışır, direksiyonu ve freni fiziksel olarak hareket ettirmez; sadece 'Ben olsaydım ne yapardım?' diye tahmin üretir.  
> 2. **Uyuşmazlık Tetikleyicisi (Discrepancy Trigger):** İnsan sürücü aniden direksiyonu kırarsa veya frene basarsa ($|\delta_{\text{human}} - \delta_{\text{shadow}}| > 5^\circ$), model ile insan arasındaki karar çatışması anında yakalanır.  
> 3. **Uç Klip Paketleme:** Uyuşmazlık anının $[-10\text{s}, +5\text{s}]$ arasındaki 8 kamera videosu ve IMU telemetrisi paketlenip araç Wi-Fi'a bağlandığında Dojo bulutuna yüklenir.  
> 4. **A/B İstatistiği (MPI):** Milyonlarca mil boyunca modelin insanla uyuşma oranı ve Müdahale Başına Mil (MPI) değeri $p < 0.001$ istatistiksel anlamlılıkla test edilir.  
> Bugün Tesla'nın otonom filoyu eğiten gizli veri döngüsünü kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Uyuşmazlık Uç Tetikleme Koşulları

$$\text{Trigger}_{\text{steer}} = \mathbb{I}\left( |\delta_{\text{human}} - \delta_{\text{shadow}}| > \Delta\theta_{\text{thresh}} \right), \quad \Delta\theta_{\text{thresh}} = 5.0^\circ$$

$$\text{Trigger}_{\text{accel}} = \mathbb{I}\left( |a_{\text{human}} - a_{\text{shadow}}| > \Delta a_{\text{thresh}} \right), \quad \Delta a_{\text{thresh}} = 1.5\text{ m/s}^2$$

### 2. Müdahale Başına Mil (Miles Per Intervention - MPI)

$$\text{Rate} = \frac{N_{\text{interventions}}}{M_{\text{total\_miles}}}, \quad \text{MPI} = \frac{1}{\text{Rate}} = \frac{M_{\text{total\_miles}}}{N_{\text{interventions}}}$$

### 3. A/B Hipotez Z-Testi ve $p$-Değeri

$$p_{\text{pool}} = \frac{N_A + N_B}{M_A + M_B}, \quad \text{SE} = \sqrt{p_{\text{pool}}(1 - p_{\text{pool}})\left( \frac{1}{M_A} + \frac{1}{M_B} \right)}$$

$$Z = \frac{\text{Rate}_A - \text{Rate}_B}{\text{SE}}, \quad p = 2 \left( 1 - \Phi(|Z|) \right)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Yeni FSD modellerini gerçek dünya trafiğinde sıfır güvenlik riskiyle test etmek ve modelin yetersiz kaldığı en zorlu köşe durumları (Corner-Cases) filodan otomatik toplamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Nadir Köşe Durum Toplama:** Yolda devrilen bir buzdolabı veya kaçan bir köpek gibi laboratuvarda akla gelmeyecek durumlar insan sürücünün kaçınma manevrasıyla anında buluta yüklendi.
- **Sıfır Güvenlik Riski:** Aday yazılım sürüş kontrolünü yönetmediği için milyonlarca milde güvenli A/B validasyonu yapıldı.
- **Veri Motoru Kapatma Döngüsü:** Modelin hata yaptığı durumlar otomatik etiketlenip (NeRF) yeniden eğitime sokuldu ve model her hafta daha akıllı hale geldi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Hücresel Veri Maliyeti:** Milyonlarca araçtan sürekli gigabaytlarca ham video yüklenemez (Yalnızca Wi-Fi ve kritik tetiklemeler yüklenir).
- **Kötü İnsan Sürücüler:** Bazen insan sürücü gereksiz yere yoldan sapabilir veya hatalı manevra yapabilir (Veri temizleme filtreleri gerekir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Kapalı Test Pisti Denemeleri:** Yalnızca birkaç yüz senaryoyu test edebilir; gerçek dünyanın sonsuz çeşitliliğini yakalayamaz.
- **Saf Simülasyon (SIL):** Gerçek kamera sensör gürültüsünü, kirli lensleri ve güneş parlamalarını tam olarak yansıtamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Shadow Mode** | Yeni yapay zeka modelinin arka planda sessizce çıkarım yapıp insan kararlarıyla kıyaslandığı filo çalışma modu. |
| **Data Engine** | Hatalı senaryoları toplayan, etiketleyen, eğiten ve araca geri yükleyen Tesla'nın kapalı döngü veri sistemi. |
| **Discrepancy Trigger**| İnsan ile modelin eylemleri eşik değerlerin üzerine çıktığında tetiklenen uç olay mekanizması. |
| **MPI (Miles Per Intervention)**| Aracın insan müdahalesine ihtiyaç duymadan katettiği ortalama mil cinsinden güvenlik metriği. |
| **Ring Buffer** | Bellekte son 10 saniyelik video ve telemetriyi sürekli tutan dairesel bellek tamponu. |
| **Edge Snapshot** | Tetikleme anında dairesel tampondan çıkarılıp buluta gönderilmek üzere sıkıştırılan veri paketi. |
| **A/B Fleet Testing** | Filoyu iki gruba ayırarak yeni model ile eski modelin gerçek yol başarı oranlarını istatistiksel kıyaslama. |
| **Z-Score Significance**| İki model arasındaki güvenlik farkının rastlantısal olmadığını kanıtlayan istatistiksel güven skoru. |
| **Corner-Case Discovery**| Rutin trafikte nadir görülen, yapay zekayı zorlayan ekstrem yol durumlarının keşfi. |
| **OTA (Over-The-Air)** | İyileştirilen yeni model ağırlıklarının araç filosuna kablosuz olarak yüklenmesi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Milyonlarca araçlık filo ile sıfır riskli test      | • Hücresel LTE/5G veri yükleme bant genişliği sınırı  |
| • Otomatik köşe durum yakalama ve veri motoru         | • Kötü insan sürücü manevralarını filtreleme zorluğu  |
| • 8.5 µs ultra hızlı RTOS uyuşmazlık denetimi         |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Robotaksi filolarında kesintisiz güvenilirlik takibi| • Veri gizliliği (GDPR) kuralları gereği plaka ve yüz |
|   ve anında uç kaza analizi                           |   bulanıklaştırma (Blurring) zorunluluğu              |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Gölge Modu ve Veri Motoru Mimarisi

```
[ İnsan Sürüş Eylemleri ] <---+   +---> [ HW3/HW4 Gölge Model Çıkarımı ]
                              |   |
                              v   v
            [ Uyuşmazlık Tetikleyicisi (Discrepancy Engine) ]
            - Direksiyon Farkı > 5.0°
            - Fren Farkı > 1.5 m/s²
                              | (Tetiklendi!)
                              v
            [ Ring Buffer Klip Paketleme: [-10s, +5s] ]
                              | (Wi-Fi Bağlantısı)
                              v
        [ Tesla Bulut Veri Motoru (Data Engine & Dojo) ]
                              | (Yeniden Eğitim)
                              v
            [ OTA ile Filoya Yeni Model Yükleme ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Gölge Modu simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
