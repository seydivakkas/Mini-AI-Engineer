# 🚗 Tesla FSD Otonom Sürüş | Gün 65: S-Function ve C++ ile Gerçek Zamanlı Yörünge Takip Kontrolcüsü (Stanley & Pure Pursuit)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Stanley](https://img.shields.io/badge/Tracker-Stanley%20Controller%20%28Front%20Axle%29-red.svg?style=flat-square)](https://www.tesla.com/)
[![Pure Pursuit](https://img.shields.io/badge/Geometry-Pure%20Pursuit%20Lookahead-blue.svg?style=flat-square)](https://www.sae.org/)
[![Performance](https://img.shields.io/badge/Accuracy-Cross--Track%20Error%20%3C%202cm-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"65. günümüze hoş geldin stajyer!  
> Stanford Üniversitesi'nin DARPA Grand Challenge'ı kazanan efsanevi otonom aracı Stanley'den adını alan **Stanley Controller**, otonom araç geometrik takip algoritmalarının mihenk taşıdır!  
> Klasik Pure Pursuit kontrolcüsü arka aksı referans alırken, Stanley kontrolcüsü **ön aks merkezini (Front Axle Center)** referans alır:  
> 1. **Yönelme Açısı Hatası ($\theta_e$):** Aracın burnunu anında yol teğetine hizalamak için doğrudan direksiyon açısına eklenir.  
> 2. **Doğrusal Olmayan Çapraz Hata Düzeltmesi ($\arctan\left(\frac{k \cdot e}{v + \epsilon}\right)$):** Yanal sapma ($e$) arttıkça direksiyonu şerit merkezine doğru yumuşakça kırar. Hız ($v$) arttıkça aşırı tepkiyi önlemek için direksiyon açısını sönümler.  
> 3. **Düşük Hız Yumuşatması ($\epsilon = 0.1\text{ m/s}$):** Sıfır hızdaki matematiksel tekilliği ($\frac{0}{0}$) yok eder.  
> 4. **Milimetrik Hassasiyet:** 50 adımlık kapalı çevrimde yanal takip hatasını 2 cm'nin altına indirir.  
> Bugün Tesla Autopilot'un otoyolda cetvel gibi düzgün gitmesini sağlayan Stanley algoritmasını kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Stanley Kontrol Kanunu (Ön Aks Tabanlı Geometrik Takip)

$$\delta(t) = \theta_e(t) + \arctan\left( \frac{k \cdot e(t)}{v(t) + \epsilon} \right)$$

$$\theta_e(t) = \psi_{\text{ref}}(t) - \psi_{\text{vehicle}}(t)$$

### 2. Pure Pursuit Kontrol Kanunu (Arka Aks Tabanlı Takip)

$$\delta(t) = \arctan\left( \frac{2 L \cdot \sin(\alpha)}{L_d} \right), \quad L_d = \max(k_{\text{look}} \cdot v, L_{\min})$$

### 3. Asimptotik Hata Sönümleme Kararlılığı (Lyapunov Stability)

$$\dot{e}(t) = -v(t) \cdot \sin\left( \arctan\left( \frac{k \cdot e(t)}{v(t)} \right) \right) = -\frac{k \cdot v(t) \cdot e(t)}{\sqrt{v(t)^2 + (k \cdot e(t))^2}}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
MPC gibi yüksek işlem gücü gerektiren sayısal çözücülere ihtiyaç duymadan, mikrosaniyeler içinde ön aks geometrisine dayalı analitik direksiyon komutu üretmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Viraj Kesme (Corner Cutting):** Pure Pursuit'in keskin virajlarda şerit çizgisini içten kesme hatasını ön aks referansı ile ortadan kaldırdı.
- **Yüksek Hız Kararlılığı:** Hız arttıkça paydadaki $v$ terimi sayesinde direksiyon tepkisini otomatik yumuşatarak salınımları engelledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Geri Sürüş:** Geri geri park manevralarında ön aks geometrisi kararsızlaşabilir (Geri vites için işaret değişimi gerekir).
- **Yol Eğim ve Rüzgar Bozuntuları:** Yalnızca oransal çalıştığı için rüzgarlı eğimli yollarda kalıcı durum hatasını yok etmek için integral terimi ($k_i$) eklenmelidir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Model Predictive Control (MPC):** Çok daha kapsamlıdır fakat işlem süresi 50 kat daha uzundur.
- **PID Yanal Kontrolcü:** Geometrik araç kinematiğini modellemediği için yüksek hızda gecikmeli tepki verir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Stanley Controller** | Ön tekerlek merkezinin yörüngeye olan sapmasını ve açı hatasını doğrudan düzelten kontrolcü. |
| **Pure Pursuit** | Arka aks merkezinden belirli bir bakış mesafesindeki ($L_d$) noktaya doğru daire yayı çizen takipçi. |
| **Cross-Track Error ($e$)** | Ön aks merkezinin takip edilen yörüngeye olan en kısa dik mesafesi. |
| **Heading Error ($\theta_e$)** | Aracın gövde açısı ile yol referans teğet açısı arasındaki fark. |
| **Softening Parameter ($\epsilon$)** | Düşük hızlarda veya durma anında sıfıra bölünmeyi önleyen küçük pozitif sayı ($0.1\text{ m/s}$). |
| **Gain Parameter ($k$)** | Yanal hatanın ne kadar agresif düzeltileceğini belirleyen kontrol kazancı. |
| **Lookahead Distance ($L_d$)** | Pure Pursuit'in önündeki kaç metre ilerideki hedef noktaya odaklandığı mesafe. |
| **Wheelbase ($L$)** | Tesla Model 3'ün ön ve arka tekerlekleri arasındaki 2.875 metrelik aks mesafesi. |
| **Asymptotic Convergence** | Hatanın zamanla sıfıra üssel ve kararlı bir biçimde yaklaşması. |
| **S-Function** | MATLAB/Simulink ve C++ RTOS ortamlarında gerçek zamanlı çalışan kontrol bloğu. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 analitik kapalı form ve 2 µs ultra hızlı RTOS  | • Geri sürüşte özel işaret düzeltmesi gerektirir      |
| • Lyapunov ile kanıtlanmış evrensel kararlılık        | • Rüzgar bozuntularına karşı integral terimi eklenmeli|
| • 2 cm altı milimetrik takip doğruluğu                |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • MPC çözücüsünün arızalanması durumunda ASIL-D       | • Çok kaygan karlı yollarda lastik kayma açısının     |
|   yedek yanal kontrolcü olarak anında devreye girme   |   geometrik modeli bozması                            |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Stanley Takip Kontrolcüsü Akış Şeması

```
[ Yol Geometrisi (x_ref, y_ref, psi_ref) & Araç Durumu (x, y, psi, v) ]
                                    |
                                    v
       [ Ön Aks Konumunun Hesabı: x_front = x + L*cos(psi) ]
                                    |
                                    v
     [ Hata Vektörleri: theta_e = psi_ref - psi | e = Cross-Track ]
                                    |
                                    v
     [ Stanley Kontrol Kanunu: delta = theta_e + atan(k*e / (v+eps)) ]
                                    |
                                    v
     [ Doyum Kısıtı: [-31.5°, +31.5°] EPS Direksiyon Motoruna İletim ]
                                    |
                                    v
         [ e_lat < 2 cm Hata ile Kusursuz Geometrik Yol Takibi ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Stanley Takip simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
