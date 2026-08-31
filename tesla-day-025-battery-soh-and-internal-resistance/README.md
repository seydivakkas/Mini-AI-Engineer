# 🚗 Tesla Batarya Yönetim Sistemi | Gün 25: Sağlık Durumu (SoH) & Çevrimiçi İç Direnç İzleme Algoritmaları

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Degradation](https://img.shields.io/badge/Model-SEI%20Layer%20Aging-blue.svg?style=flat-square)](https://www.tesla.com/)
[![Algorithm](https://img.shields.io/badge/Algorithm-Recursive%20Least%20Squares%20(RLS)-orange.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/Standard-ISO%2026262%20EOL%20Diagnosis-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"25. günümüze hoş geldin stajyer!  
> Bir Tesla'nın bataryası ilk günkü performansını sonsuza kadar koruyamaz. Her şarj ve deşarj döngüsünde anot yüzeyinde **SEI (Solid Electrolyte Interphase)** katmanı mikron düzeyinde kalınlaşır, lityum iyonları hapsolur ve iki temel yaşlanma olayı gerçekleşir:  
> 1. **Kapasite Kaybı (Capacity Fade - $SoH_C$):** Bataryanın saklayabildiği toplam amper-saat (Ah) azalır; maksimum menzil düşer.  
> 2. **İç Direnç Artışı (Resistance Growth - $SoH_R$):** $R_0$ iç direnci arttığı için ani gaz pedalına basıldığında voltaj daha fazla çöker ve Joule ısınması ($I^2 R$) artar.  
> Otomotiv standartlarında $\%80$ kapasiteye düşen veya iç direnci $2$ katına çıkan bataryalar **EOL (End of Life - Ömür Sonu)** kabul edilir ve garanti kapsamında yenilenir ya da ikinci ömür (Megapack/Powerwall) depolamasına aktarılır.  
> Sürüş esnasında laboratuvar testleri yapamayacağımız için **RLS (Recursive Least Squares - Özyinelemeli En Küçük Kareler)** algoritmasıyla mikrosaniyeler içinde iç direnci anlık takip ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Kapasite ve Direnç Tabanlı Sağlık Durumu (SoH)

$$SoH_C = \frac{Q_{\text{current}}}{Q_{\text{fresh}}} \times 100\%$$

$$SoH_R = \frac{R_{0, \text{EOL}} - R_{0, \text{current}}}{R_{0, \text{EOL}} - R_{0, \text{fresh}}} \times 100\%, \quad \text{burada } R_{0, \text{EOL}} = 2.0 \times R_{0, \text{fresh}}$$

### 2. Unutma Faktörlü Özyinelemeli En Küçük Kareler (RLS with Forgetting Factor $\lambda$)
Parametre vektörü $\theta = [R_0]$, regresör $\phi_k = \Delta I_k$ ve ölçüm $y_k = \Delta V_k$ olmak üzere:

$$K_k = \frac{P_{k-1} \phi_k}{\lambda + \phi_k^2 P_{k-1}}$$

$$\hat{\theta}_k = \hat{\theta}_{k-1} + K_k \left(y_k - \phi_k \hat{\theta}_{k-1}\right)$$

$$P_k = \frac{1}{\lambda} \left(P_{k-1} - K_k \phi_k P_{k-1}\right)$$

### 3. SEI Katmanı Döngüsel Yaşlanma Modeli (Cycle Aging)
Sıcaklık $T$ (Kelvin), $N$ döngü sayısı ve $DoD$ deşarj derinliği olmak üzere:

$$Q_{\text{loss}} = B \cdot \exp\left(-\frac{E_a}{R_{\text{gas}} T}\right) \cdot (DoD)^\beta \cdot N^z, \quad (z \approx 0.5)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Bataryanın kalan garanti ömrünü, maksimum güvenli şarj/deşarj akım sınırlarını (SOP - State of Power) belirlemek ve termal kaçak riskini önlemek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Çevrimiçi Parametre Tespiti:** Aracı servise çekip saatlerce deşarj etmeden, normal sürüş dinamiklerindeki $\Delta V / \Delta I$ darbeleriyle iç direnç kestirildi.
- **Kişiselleştirilmiş Menzil Göstergesi:** Kullanıcının sürüş alışkanlıklarına ve bataryanın yaşlanma durumuna göre kalan menzil milimetrik güncellendi.
- **Supercharger Hız Kısma:** Yüksek iç dirençli yaşlı hücrelerde aşırı ısınmayı önlemek için şarj akımı otomatik sınırlandırıldı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Kısa Süreli Darbeler:** Sabit hızda (Cruise Control) giderken akım değişimi ($\Delta I \approx 0$) olmadığı için RLS parametre güncellemesi duraklar.
- **Sıcaklık Ayrıştırması:** Sıcaklıktan kaynaklanan geçici iç direnç düşüşü ile kalıcı yaşlanma direnç artışı doğru kompanse edilmelidir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Laboratuvar EIS (Elektrokimyasal Empedans Spektroskopisi):** Çok hassastır ancak araç üzerinde pahalı AC sinyal jeneratörleri gerektirir.
- **Dual-Kalman Filtresi (Dual-EKF):** SoC ve SoH'ı iki paralel Kalman filtresiyle kestirir; RLS'e göre biraz daha fazla işlem gücü gerektirir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **SoH (State of Health)** | Bataryanın fabrikadan çıktığı ilk güne göre sağlık ve performans yüzdesi. |
| **Capacity Fade** | Şarj edilebilir lityum miktarının azalması sonucu bataryanın nominal kapasitesinin düşmesi. |
| **Resistance Growth** | Elektrot ve elektrolit arayüzlerinde oluşan bariyerler nedeniyle iç direncin büyümesi. |
| **EOL (End of Life)** | Otomotivde bataryanın $\%80$ kapasiteye düşmesi veya direncin $2$ katına çıkması eşiği. |
| **SEI (Solid Electrolyte Interphase)** | Anot grafit yüzeyinde ilk şarjdan itibaren oluşan pasifleştirici koruyucu katman. |
| **RLS (Recursive Least Squares)** | Zamanla değişen parametreleri her yeni sensör verisiyle anında güncelleyen algoritma. |
| **Forgetting Factor ($\lambda$)** | RLS algoritmasında eski verilerin ağırlığını azaltarak yeni dinamiklere uyum sağlayan katsayı ($0.95 - 0.999$). |
| **Calendar Aging** | Bataryanın hiç kullanılmadan, zaman ve saklama sıcaklığına bağlı olarak kendiliğinden yaşlanması. |
| **Cycle Aging** | Şarj ve deşarj akımları altında elektrokimyasal stresle gerçekleşen döngüsel yaşlanma. |
| **SOP (State of Power)** | Bataryanın o anki sıcaklık, SoC ve SoH durumuna göre güvenle çekilebilecek anlık pik gücü ($kW$). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Sürüş esnasında 2.1 µs hızında çevrimiçi R0 takibi  | • Düşük dinamikli sabit sürüşte parametre uyarımı yok |
| • SEI büyümesini ve Arrhenius sıcaklık etkisini içerir| • Kapasite kalibrasyonu için tam döngü şarj ister    |
| • EOL tespitinde çift kriter (Kapasite + Direnç)      |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • İkinci ömür (Second-life) batarya geri dönüşümünde  | • Soğuk havada geçici direnç artışının kalıcı SoH     |
|   otomatik ekspertiz raporu üretimi                   |   hatası gibi algılanması riski                       |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & RLS Parametre Döngüsü

```
     Sensör Girişleri:
     - Akım Değişimi (ΔI) --------+
     - Gerilim Değişimi (ΔV) -----|
                                  v
                  +-------------------------------+
                  |  RLS Çevrimiçi Kestirici       |
                  |  - Regresör: phi = ΔI         |
                  |  - Hata: e = ΔV - phi * R0    |
                  |  - Kazanç: K = P*phi / (λ...) |
                  |  - Güncelleme: R0 += K * e    |
                  +-------------------------------+
                                  |
                                  v
                  +-------------------------------+
                  |  BMS SoH Teşhis Motoru        |
                  |  - SoH_R = (2*R_fresh - R) /..|
                  |  - SoH_C = Q_curr / Q_fresh   |
                  |  - EOL Alarm Kontrolü (%80)   |
                  +-------------------------------+
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana SoH akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
