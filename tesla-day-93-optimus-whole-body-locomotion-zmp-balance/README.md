# 🤖 Tesla FSD Otonom Sürüş | Gün 93: Optimus Bütünsel Denge (Whole-Body Locomotion) ve Sıfır An Moment Noktası (ZMP)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![ZMP-Locomotion](https://img.shields.io/badge/Robotics-Zero%20Moment%20Point%20(ZMP)-red.svg?style=flat-square)](https://en.wikipedia.org/wiki/Zero_moment_point)
[![LIPM-Model](https://img.shields.io/badge/Physics-Linear%20Inverted%20Pendulum-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/Inverted_pendulum)
[![Push-Recovery](https://img.shields.io/badge/Safety-Capture%20Point%20Stepping-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"93. günümüze hoş geldin stajyer!  
> İki ayaklı insansı bir robot için en büyük fiziksel zorluk yürümektir; çünkü iki ayak üzerinde durmak ve yürümek sürekli kontrollü bir düşme eylemidir!  
> Tesla Optimus bu problemi klasik robotik fiziği ile modern yapay zekayı birleştiren **Sıfır An Moment Noktası (Zero Moment Point - ZMP) ve Doğrusal Ters Sarkaç Modeli (LIPM)** ile çözer:  
> 1. **Doğrusal Ters Sarkaç (LIPM):** 56 kg'lık robotun tüm gövde kütlesini tek bir dinamik noktada ($z_{\text{com}} = 0.85\text{ m}$) modeller.  
> 2. **Sıfır An Moment Noktası (ZMP):** Yere temas eden tabanın yatay dönme momentinin sıfır olduğu noktayı hesaplar. ZMP ayak tabanı (Destek Poligonu) içinde kaldığı sürece robot asla devrilmez!  
> 3. **Denge Kurtarma (Push Recovery):** Biri robotu aniden ittiğinde önce bilek torkuyla direnir; eğer itme kuvveti büyükse ($> 40\text{ Ns}$) anında **Capture Point** noktasına yeni bir adım atarak düşmeyi engeller.  
> 4. **1000 Hz Bütünsel Dengeleme:** Ayaklar, dizler, kalça ve gövde torkları 1 ms içinde senkronize çalışır.  
> Bugün Tesla Optimus'un iki ayak üzerinde dimdik durmasını ve yürümesini sağlayan bütünsel denge motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Doğrusal Ters Sarkaç Modeli (LIPM) ve ZMP Formülasyonu

$$x_{\text{zmp}} = x_{\text{com}} - \frac{z_{\text{com}}}{g} \ddot{x}_{\text{com}}$$

$$y_{\text{zmp}} = y_{\text{com}} - \frac{z_{\text{com}}}{g} \ddot{y}_{\text{com}}$$

### 2. Destek Poligonu (Support Polygon) Kararlılık Kriteri

$$x_{\text{min}} \le x_{\text{zmp}} \le x_{\text{max}} \quad \land \quad y_{\text{min}} \le y_{\text{zmp}} \le y_{\text{max}}$$

### 3. Yakalama Noktası (Capture Point) ve Doğal Frekans

$$\omega_0 = \sqrt{\frac{g}{z_{\text{com}}}} \approx \sqrt{\frac{9.81}{0.85}} = 3.397\ \text{rad/s}$$

$$x_{\text{cp}} = x_{\text{com}} + \frac{\dot{x}_{\text{com}}}{\omega_0}, \quad y_{\text{cp}} = y_{\text{com}} + \frac{\dot{y}_{\text{com}}}{\omega_0}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
İki ayaklı robotun yürürken, yük taşırken veya dış darbeler aldığında dengesini kaybetmeden kararlı kalmasını ve düşmesini engellemek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Dengesizlik ve Yere Düşme:** ZMP'yi destek poligonu içinde tutarak robotun kontrolsüz devrilmesini tamamen önledi.
- **İtme Sonrası Toparlanma:** Capture Point hesabı ile adımlama hedefini analitik olarak belirledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Düz Zemin Varsayımı:** LIPM modeli sabit $z_{\text{com}}$ yüksekliği varsayar; merdiven veya engebeli arazilerde tam 3D dinamiğe ihtiyaç duyar.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Statik Denge (CoM Taban İçi):** Sadece çok yavaş hareketlerde çalışır, insansı dinamik yürüyüşü imkansız kılar.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **ZMP (Zero Moment Point)** | Yatay düzlemdeki zemin tepki kuvvetlerinin net momentinin sıfır olduğu temas noktası. |
| **LIPM (Linear Inverted Pendulum)** | Robotun kütle merkezini sabit bir yükseklikte hareket eden ters sarkaç olarak modelleyen dinamik yaklaşım. |
| **CoM (Center of Mass)** | Robotun toplam 56 kg kütlesinin ağırlık merkezi. |
| **Support Polygon** | Robotun yere basan ayağının veya iki ayağının oluşturduğu geometrik temas alanı. |
| **Capture Point** | Robotun düşmeden durabilmesi için bir sonraki adımı basması gereken analitik nokta. |
| **Push Recovery** | Robot dışarıdan bir itme aldığında düşmemek için uyguladığı bilek, kalça ve adımlama stratejisi. |
| **Ankle Strategy** | Küçük itmelerde ayağı yerden kaldırmadan yalnızca bilek torkuyla dengeyi koruma. |
| **Stepping Strategy** | Büyük darbelerde dengeyi kurtarmak için Capture Point'e doğru hızlı bir adım atma. |
| **Whole-Body Control (WBC)** | Tüm eklemlerin (kollar, bacaklar, gövde) aynı anda dengeye katkı sağladığı hiyerarşik kontrol. |
| **Stance Phase** | Yürüyüş döngüsünde ayağın yere basıp gövdeyi taşıdığı basış evresi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Analitik ZMP hesabı ile garantili dinamik denge     | • Engebeli ve kaygan zeminlerde LIPM kısıtı           |
| • Capture Point ile büyük itmelerde sıfır devrilme    | • Ayak taban sensörü gürültülerine hassasiyet         |
| • 1.8 µs ultra hızlı RTOS döngüsü                     |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Merdiven tırmanma ve fabrika içinde kutu taşırken   | • Beklenmedik zemin çökmesi veya kayma durumunda      |
|   ağırlık merkezini dinamik olarak yeniden dengeleme  |   sürtünme konisinin anlık aşılması                   |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Optimus ZMP ve Denge Akış Şeması

```
[ IMU & Ayak Tabanı Tork Sensörleri ]
                 |
                 v
     [ Kütle Merkezi (CoM) Kestirimi ]
                 |
                 v
     [ LIPM ZMP Hesabı: x_zmp, y_zmp ]
                 |
        +--------+--------+
        |                 |
(ZMP Destek İçinde)   (Dış İtme / Sapma)
        |                 |
        v                 v
[ Bilek & Kalça Torku ] [ Capture Point Hesabı ]
        |                 |
        v                 v
[ 1000 Hz Sabit Yürüyüş ] [ ACİL KURTARMA ADIMI ATMA ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Optimus ZMP denge simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
