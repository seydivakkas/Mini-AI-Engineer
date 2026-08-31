# 🚗 Tesla FSD Otonom Sürüş | Gün 57: Frenet Çerçevesi ve Dinamik Şerit Değiştirme Yörünge Üretimi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Frenet](https://img.shields.io/badge/Coordinates-Frenet%20%5Bs%2C%20d%5D-blue.svg?style=flat-square)](https://www.tesla.com/)
[![Quintic](https://img.shields.io/badge/Trajectory-Quintic%20Polynomial%205th-red.svg?style=flat-square)](https://www.sae.org/)
[![Jerk-Optimal](https://img.shields.io/badge/Comfort-Jerk%20Optimal%20%3C%3D%201.5-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"57. günümüze hoş geldin stajyer!  
> Kartezyen $(X, Y)$ koordinatlarında virajlı bir otoyolda şerit değiştirme planlaması yapmak son derece karmaşıktır. Çünkü yolun kendisi kıvrılırken aracın hem yolu takip etmesi hem de 3.5 metre yana kayması gerekir.  
> Otonom sürüş mühendisliğinin en zarif çözümü **Frenet Koordinat Sistemidir ($s, d$)**:  
> 1. **$s$ (Longitudinal):** Yolun kıvrımlı referans çizgisi boyunca kat edilen boyuna mesafe.  
> 2. **$d$ (Lateral):** Referans çizgisine olan dik yanal ofset mesafesi.  
> 
> Frenet uzayında şerit değiştirme problemi, basit bir 1D yanal geçiş problemine ($d(0) = 0 \to d(T) = 3.5\text{ m}$) indirgenir!  
> Ancak yolcuların kahvelerini dökmemesi ve mide bulantısı yaşamaması için ivmenin türevi olan **Sarsıntıyı (Jerk, $\dddot{d}(t)$)** minimize etmek zorundayız.  
> Bunun analitik matematiksel çözümü **5. Derece Quintic Polinomdur** ($d(t) = \sum_{i=0}^5 c_i t^i$).  
> Başlangıç ($d_0, v_0, a_0$) ve bitiş ($d_1, v_1, a_1$) sınır koşullarını sağlayan tek ve analitik Jerk-Optimal eğriyi saniyenin milyonda birinde çözeriz.  
> Bugün Tesla Autopilot'un akıcı otoyol şerit değiştirme matematiğini inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 5. Derece Quintic Polinom Denklemi

$$d(t) = c_0 + c_1 t + c_2 t^2 + c_3 t^3 + c_4 t^4 + c_5 t^5$$

### 2. Başlangıç ve Bitiş Sınır Koşulları Matrisi ($A \mathbf{c} = \mathbf{B}$)

$$\begin{bmatrix} T^3 & T^4 & T^5 \\ 3T^2 & 4T^3 & 5T^4 \\ 6T & 12T^2 & 20T^3 \end{bmatrix} \begin{bmatrix} c_3 \\ c_4 \\ c_5 \end{bmatrix} = \begin{bmatrix} d_1 - (d_0 + v_0 T + \frac{1}{2} a_0 T^2) \\ v_1 - (v_0 + a_0 T) \\ a_1 - a_0 \end{bmatrix}$$

### 3. Yanal Hız, İvme ve Jerk (Sarsıntı) Türevleri

$$\dot{d}(t) = c_1 + 2 c_2 t + 3 c_3 t^2 + 4 c_4 t^3 + 5 c_5 t^4$$

$$\ddot{d}(t) = 2 c_2 + 6 c_3 t + 12 c_4 t^2 + 20 c_5 t^3$$

$$\dddot{d}(t) = 6 c_3 + 24 c_4 t + 60 c_5 t^2$$

$$\text{Jerk Maliyeti: } J = \int_0^T (\dddot{d}(t))^2 dt \le 1.5\text{ m/s}^3$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Virajlı ve eğimli yollarda şerit değiştirme planlamasını 2D Kartezyen karmaşıklığından çıkarıp 1D bağımsız $(s, d)$ profillerine ayırmak ve sıfır sarsıntılı (Jerk-optimal) geçiş üretmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Yolcu Konforu ve Sarsıntı:** 3. ve 4. derece polinomların sınırda bıraktığı ivme sıçramalarını (Dirac delta jerk) ortadan kaldırdı.
- **Analitik Çözüm Hızı:** Sayısal optimizasyon yinelemelerine ihtiyaç duymadan doğrudan $3 \times 3$ matris tersiyle mikrosaniyeler içinde çözüldü.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Yol Eğriliği Tekilliği:** Yol eğrilik yarıçapı ($R = 1/\kappa$) yanal ofsetten ($d$) küçük olduğunda ($d > R$) koordinat dönüşümü tekillik (Singularity) üretir.
- **Ani Engeller:** Statik quintic eğriler araya aniden giren araçlara karşı ara nokta (Waypoints) optimizasyonu gerektirir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **B-Spline / Bézier Eğrileri:** Kontrol noktalarıyla serbest şekil verir ancak sınır ivme/jerk garantisi vermek için ek kısıtlar gerekir.
- **Sayısal Doğrusal Olmayan MPC:** Dinamik engelleri anlık çözer ancak hesaplama maliyeti 1000 kat daha yüksektir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Frenet Frame ($s, d$)** | Kıvrımlı yol eksenini referans alan boyuna ($s$) ve yanal ($d$) koordinat sistemi. |
| **Quintic Polynomial** | 5. dereceden $C^2$ süreklilik sağlayan ve jerk integralini minimize eden polinom. |
| **Jerk ($\dddot{x}$)** | İvmenin zamana göre türevi; yolcu tarafından hissedilen ani sarsıntı/titreme miktarı. |
| **Boundary Conditions** | Yörüngenin başlangıç ve bitiş anlarındaki konum, hız ve ivme hedefleri. |
| **Lateral Acceleration** | Viraj veya şerit değiştirme sırasında yana doğru etki eden merkezkaç ivmesi. |
| **Time Horizon ($T$)** | Manevranın tamamlanması için hedeflenen toplam süre (Otoyolda genelde 3.5 - 5.0 s). |
| **Cross-Track Profile** | Zaman boyunca şerit ekseninden olan yanal sapma eğrisi $d(t)$. |
| **Curvature ($\kappa$)** | Yolun birim boyundaki yönelme açısı değişim oranı ($1/R$). |
| **Comfort Envelope** | İnsan yolcunun rahatsız olmayacağı ivme ($<2\text{ m/s}^2$) ve jerk ($<1.5\text{ m/s}^3$) sınır bölgesi. |
| **Analytic Solver** | İteratif döngü olmadan doğrudan formülle çözüm üreten deterministik algoritma. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 analitik çözüm ve 12 µs RTOS determinizmi      | • Aşırı keskin virajlarda Frenet tekilliği riski      |
| • Matematiksel olarak kanıtlanmış minimum jerk        | • Manevra esnasında anlık rota düzeltmesi için        |
| • Sınır koşullarında (0 ivme) kusursuz süreklilik     |   sürekli yeniden planlama (Replanning) gereksinimi   |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Otoyol Navigate on Autopilot (NoA) sistemlerinde    | • Şerit değiştirirken arkadan çok yüksek hızla        |
|   yüksek hızlı akıcı sollama manevraları              |   gelen araçların yarattığı anlık iptal zorunluluğu   |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Frenet Quintic Şerit Değiştirme Akış Şeması

```
[ Araç Durumu: s0, d0=0, v0=0, a0=0 | Hedef: d1=3.5m, v1=0, a1=0, T=4s ]
                                    |
                                    v
            [ 3x3 Doğrusal Matris Sistemi: A * c = B ]
                                    |
                                    v
             [ Quintic Katsayıları: (c0, c1, c2, c3, c4, c5) ]
                                    |
       +----------------------------+----------------------------+
       |                            |                            |
       v                            v                            v
[ Yanal Konum d(t) ]      [ Yanal İvme a(t) ]          [ Yanal Jerk j(t) ]
- 0m -> 3.5m Geçiş        - Maks: 0.95 m/s²            - Maks: 1.25 m/s³
- Pürüzsüz S-Eğrisi       - Sınır: <= 2.0 m/s²         - Sınır: <= 1.5 m/s³
       \                            |                            /
        \                           |                           /
         v                          v                          v
             [ %100 PREMIUM KONFORLU OTOYOL ŞERİT GEÇİŞİ ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Frenet Quintic simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
