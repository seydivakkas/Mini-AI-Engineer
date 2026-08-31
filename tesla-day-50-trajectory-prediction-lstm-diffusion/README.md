# 🚗 Tesla FSD Otonom Sürüş | Gün 50: Dinamik Nesne Yörünge Tahmini (LSTM, GRU ve Difüzyon Modelleri)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Trajectory](https://img.shields.io/badge/Prediction-5s%20Future%20Trajectory%20Horizon-red.svg?style=flat-square)](https://www.tesla.com/)
[![Diffusion](https://img.shields.io/badge/Model-Conditional%20Diffusion%20%26%20LSTM-blue.svg?style=flat-square)](https://www.sae.org/)
[![TTC](https://img.shields.io/badge/Safety-Time--to--Collision%20(TTC)-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"50. günümüze hoş geldin stajyer!  
> Bir otonom aracın sadece etrafındaki araçların 'nerede olduğunu' bilmesi yetersizdir; o araçların 'gelecekte nereye gideceğini' tahmin etmesi gerekir. Çünkü 100 km/h hızla giden bir araç 1 saniyede 28 metre yol alır.  
> Ancak insanların sürüş kararları tekil (deterministik) değildir: Önünüzdeki araç şeridinde kalabilir, aniden sola kırıp önünüze geçebilir (Cut-In) veya aniden frene basabilir.  
> Tesla bu çoklu geleceği **Çoklu Modal Yörünge Tahmini (Multi-Modal Trajectory Prediction)** ile modeller:  
> 1. **5 Saniyelik Gelecek Ufku ($H = 50$ adım, $dt = 0.1\text{ sn}$):** Çevredeki her aktörün sonraki 5 saniyedeki 2D yol koordinatları ($x(t), y(t)$) üretilir.  
> 2. **Çoklu Modal Olasılıklar:** Ağ tek bir çizgi yerine 3 olası senaryo (Şeritte Kalma %70, Sola Geçiş %20, Ani Fren %10) tahmin eder.  
> 3. **Koşullu Difüzyon (Conditional Diffusion):** Trafik akışının karmaşık dağılımını gürültü giderme adımlarıyla (Denoising) yüksek çeşitlilikte modeller.  
> 4. **TTC (Time-to-Collision):** Göreli hız ve mesafe türetilerek en erken çarpışma riski milisaniyeler içinde planlayıcıya iletilir.  
> Bugün otonom sürüşün geleceği öngören beynini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Çoklu Modalite Karışım Dağılımı

$$P(Y \mid X) = \sum_{k=1}^K P(k \mid X) \cdot \mathcal{N}\left( Y \mid \mu_k(X), \, \Sigma_k(X) \right)$$

### 2. Koşullu Difüzyon Ters Örnekleme Adımı (Denoising)

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) \right) + \sigma_t \mathbf{z}$$

### 3. Kazanan Hepsini Alır Kaybı (Winner-Takes-All Loss)

$$\mathcal{L}_{\text{WTA}} = \min_{k} \left\| Y_{\text{gt}} - \hat{Y}_k \right\|^2 - \lambda \log P(k^* \mid X)$$

### 4. Çarpışmaya Kalan Süre (Time-to-Collision - TTC)

$$\text{TTC} = \frac{d_{\text{rel}}}{v_{\text{rel}}} = \frac{y_{\text{target}} - y_{\text{ego}}}{v_{\text{ego}} - v_{\text{target}}}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Dinamik trafikte araçların birbirlerinin niyetlerini (şerit değiştirme, dönüş, fren) önceden tahmin ederek FSD planlayıcısının ani ve sert manevralar yapmasını önlemek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Hayalet Frenleme (Phantom Braking):** Yan şeritteki aracın sadece hafif yalpalamasını önümüze kırıyor sanıp yapılan gereksiz sert frenlemeleri çoklu modalite ile engelledi.
- **Kavşak Yol Hakkı:** Kontrolsüz kavşaklarda diğer araçların yavaşlama eğilimini analiz ederek güvenli geçiş kararı üretti.
- **5 Saniye Önceden Savunma:** Olası kaza senaryolarını 5 saniye önceden simüle ederek aracı defansif sürüş pozisyonuna aldı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Agresif İnsan Sürücüler:** Sinyalsiz ve son milisaniyede yapılan aşırı agresif makas hareketleri difüzyon olasılık kuyruğunda kalabilir.
- **Hesaplama Yükü:** Çevredeki 50 aracın her biri için 50 adımlık difüzyon çalıştırmak NPU'da yüksek matris gücü gerektirir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Sabit Hız / Sabit İvme Modelleri (CV/CA):** Araçların dönüş yapacağını veya şerit değiştireceğini öngöremez.
- **Saf Deterministik Regresyon:** Tek bir ortalama çizgi çizer, ikiye ayrılan yolda refüje doğru hayali bir çizgi üretebilir (Mode Collapse).

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Trajectory Prediction** | Çevredeki araç ve yayaların gelecekteki 2D/3D konumlarını zaman serisi olarak tahmin etme. |
| **Multi-Modality** | Geleceğin tek bir kesin yol yerine birden fazla olası davranış modundan (düz, sol, fren) oluşması. |
| **Prediction Horizon** | Yörüngenin ne kadar ileriye tahmin edildiğini belirten zaman aralığı (Tesla FSD: 5.0 saniye). |
| **Time-to-Collision (TTC)**| Mevcut hız ve yörünge korunduğunda iki aracın çarpışmasına kalan saniye cinsinden süre. |
| **Cut-In** | Yan şeritteki bir aracın aniden ego aracın önüne geçiş yapması durumu. |
| **Conditional Diffusion** | Geçmiş gözlem ve harita koşullarına bağlı olarak gelecek yörünge örnekleyen difüzyon modeli. |
| **Mode Collapse** | Çoklu modalite yerine modelin farklı yolların ortalamasını alıp geçersiz bir çizgi üretmesi hatası. |
| **Winner-Takes-All Loss** | Gerçekleşen yola en yakın tahmin edilen modun gradyanını güncelleyen kayıp fonksiyonu. |
| **LSTM / GRU** | Zamansal bağımlılıkları ve geçmiş araç hızlanma profillerini hafızasında tutan tekrarlayan sinir ağı. |
| **Defensive Driving Planner**| En riskli yörünge moduna karşı minimum güvenlik tamponu oluşturan hareket planlayıcısı. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 5 saniyelik geniş gelecek ufku ile defansif sürüş   | • 50+ dinamik aktörde NPU hesaplama yükü artışı       |
| • Çoklu modalite sayesinde Mode Collapse önleme       | • Kural tanımayan aşırı agresif sürücü belirsizliği   |
| • 18 µs ultra hızlı RTOS çözüm süresi                 |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • FSD v12 Uçtan Uca Dünya Modelleri (World Models)   | • Görüşün tamamen kapalı olduğu kör virajlardan       |
|   ile video simülasyonu seviyesine yükseltme          |   aniden fırlayan kontrolsüz yayalar                  |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Çoklu Modal Yörünge Mimarisi

```
[ Geçmiş Araç Takibi (t-20:t) ] ===> [ LSTM / GRU / Diffusion Omurgası ]
                                                    |
             +--------------------+-----------------+--------------------+
             |                    |                                      |
             v                    v                                      v
    [ Şeritte Kalma (%70) ]   [ Sola Geçiş (%20) ]                   [ Ani Fren (%10) ]
    - Sabit Hız Cruise        - Sigmoid Yanal Kayma                  - 5 m/s^2 Deselerasyon
             \                    |                                      /
              \                   |                                     /
               v                  v                                    v
                  [ TTC Çarpışma Riski ve Minimum Yaklaşma Kontrolü ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Yörünge Tahmini simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
