# 🚗 Tesla FSD Otonom Sürüş | Gün 59: Dinamik Engelden Kaçınma ve Sürekli Eğrilik (Clothoid/Spline) Yörünge Optimizasyonu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Clothoid](https://img.shields.io/badge/Geometry-Clothoid%20%28Euler%20Spiral%29-red.svg?style=flat-square)](https://www.tesla.com/)
[![Continuity](https://img.shields.io/badge/Continuity-C%C2%B2%20Smooth%20Curvature-blue.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/Safety-Dynamic%20Obstacle%20Avoidance-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"59. günümüze hoş geldin stajyer!  
> Otoyolda 100 km/h hızla seyrederken şeridimizde aniden duran bir arıza aracı veya enkaz gördüğümüzde ne yaparız? Aniden direksiyonu kırmak aracı savurur ve ESP sistemini tetikler.  
> Karayolları mühendisliğinde ve Tesla FSD yörünge üretiminde kullanılan en temel geometrik eğri **Clothoid (Euler Spirali)**'dir:  
> 1. **Doğrusal Değişen Eğrilik ($\kappa(s) = \kappa_0 + c \cdot s$):** Eğrilik adım adım ve doğrusal değiştiği için direksiyon simidi sabit bir açısal hızla ($\dot{\delta}$) çevrilir.  
> 2. **$C^2$ Geometrik Süreklilik:** Dairesel yaylar ile düz çizgileri birbirine bağlarken sıfır eğrilik sıçraması (Zero Curvature Jump) sağlar.  
> 3. **Aktüatör Koruma Kalkanı:** Direksiyon motorunun maksimum dönüş hızını ($|\dot{\delta}| \le 0.6\text{ rad/s}$) aşmayacak eğrilik türevi garantisi verir.  
> 4. **Dinamik Engelden Kaçınma:** 4 kademeli S-Clothoid zinciri ile engelin yanından en az 1.5 metre güvenlik payıyla akıcı bir şekilde sıyrılır.  
> Bugün Tesla FSD'nin engellerin etrafından bir sanat eseri gibi akmasını sağlayan Clothoid matematiğini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Clothoid Eğrilik Denklemi ve Eğrilik Değişim Oranı

$$\kappa(s) = \kappa_0 + c \cdot s, \quad c = \frac{d\kappa}{ds}$$

### 2. Fresnel İntegralleri ile Durum Uzayı Pozisyon Hesabı

$$\theta(s) = \theta_0 + \kappa_0 s + \frac{1}{2} c s^2$$

$$x(s) = x_0 + \int_0^s \cos\left( \theta(\tau) \right) d\tau$$

$$y(s) = y_0 + \int_0^s \sin\left( \theta(\tau) \right) d\tau$$

### 3. Direksiyon Simidi Dönüş Hızı Sınırı (Steering Rate Limit)

$$\dot{\delta}(t) = L \cdot \frac{d\kappa}{dt} = L \cdot v \cdot \frac{d\kappa}{ds} \le \dot{\delta}_{\max} \quad (0.60\text{ rad/s})$$

### 4. Minimum Engel Güvenlik Mesafesi

$$d_{\min} = \min_{t} \|\mathbf{p}_{\text{vehicle}}(t) - \mathbf{p}_{\text{obstacle}}\| \ge 1.5\text{ Metre}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Direksiyon açısının zamanda ani sıçramalar yapmasını engelleyerek, aktüatör motorunun fiziksel hız limitlerine uygun $C^2$ sürekli kaçınma yörüngeleri üretmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Direksiyon Motoru Şokları:** Dairesel yayların başlangıcındaki sonsuz direksiyon ivmesi talebini sıfırladı.
- **Yanal Savrulma (Lateral Slip):** Doğrusal artan eğrilik ile lastik yanal kuvvetlerinin yumuşakça oluşmasını sağladı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Fresnel Sayısal İntegrasyonu:** Kapalı formda çözümü olmadığı için Taylor serisi veya sayısal integral (Simpson/Trapezoid) gerektirir.
- **Dar Şehir İçi Köşeleri:** Çok dar 90 derece kavşaklarda Clothoid boyunun sığmaması durumunda Hibrit A* arama gerekir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Polinomial Splines (Cubic/Quintic):** Hesaplaması kolaydır ancak eğriliğin doğrusal değişeceğini garanti etmez.
- **Dubins Eğrileri:** Yalnızca dairesel yay ve doğruları birleştirir; eğrilik süreksizliği ($C^1$) içerir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Clothoid (Euler Spiral)** | Eğriliği yay uzunluğu boyunca doğrusal olarak değişen ($d\kappa/ds = \text{sabit}$) geçiş eğrisi. |
| **Fresnel Integral** | Clothoid koordinatlarını hesaplamak için kullanılan özel trigonometrik integral formülasyonu. |
| **Curvature Rate ($c$)** | Yol eğriliğinin birim mesafedeki artış veya azalış hızı ($d\kappa/ds$). |
| **$C^2$ Continuity** | Konum ($C^0$), teğet/açı ($C^1$) ve eğriliğin ($C^2$) yol boyunca pürüzsüz ve kesintisiz olması. |
| **Steering Slew Rate ($\dot{\delta}$)** | Direksiyon motorunun bir saniyede dönebileceği maksimum radyan cinsinden açısal hız. |
| **Evasive Maneuver** | Önümüzde aniden beliren engele karşı geliştirilen acil şerit değiştirme kaçış yolu. |
| **Clearance Margin** | Aracın gövdesi ile engel arasındaki minimum tampon güvenlik mesafesi. |
| **Transition Curve** | Düz yoldan dairesel viraja geçerken yanal ivmeyi sıfırdan nominale yumuşatan geçiş bölgesi. |
| **Wheelbase ($L$)** | Tesla Model 3'te ön ve arka tekerlek merkezleri arası 2.875 metre boyuna mesafe. |
| **Peak Curvature ($\kappa_{\text{peak}}$)**| Kaçınma manevrası sırasında ulaşılan maksimum viraj eğriliği ($1/R$). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 C² süreklilik ile sıfır direksiyon şoku        | • Kapalı form cebirsel formülü yoktur                 |
| • Karayolu geometrisi standartlarıyla %100 uyum       | • Sayısal Fresnel integrasyonu gerektirir             |
| • 35 µs ultra hızlı RTOS yörünge üretimi              |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Otoyol enkaz kaçınma ve yol yapım dubaları          | • Yan şeritte çok yakın seyreden diğer araçların      |
|   etrafından milimetrik süzülme yeteneği              |   kaçış koridorunu kapatması                          |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Clothoid Dinamik Kaçınma Akış Şeması

```
[ Algılanan Engel: x=35m, y=0m | Araç Hızı: 20 m/s (72 km/h) ]
                                |
                                v
      [ 4 Kademeli Clothoid Dilim Sentezi (Toplam 60m) ]
      - Dilim 1: 0 -> +kappa_peak   (Giriş Geçişi)
      - Dilim 2: +kappa_peak -> 0   (Tepe Noktası)
      - Dilim 3: 0 -> -kappa_peak   (Karşı Viraj)
      - Dilim 4: -kappa_peak -> 0   (Şeride Oturma)
                                |
                                v
        [ Aktüatör Hız Güvenlik Doğrulaması: |dkappa/dt| <= 0.6 ]
                                |
                                v
      [ Minimum Engele Yaklaşım: 2.1m (Güvenlik Payı >= 1.5m) ]
                                |
                                v
        [ %100 ÇARPIŞMASIZ SÜREKLİ EĞRİLİKLİ KAÇINMA YÖRÜNGESİ ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Clothoid Kaçınma simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
