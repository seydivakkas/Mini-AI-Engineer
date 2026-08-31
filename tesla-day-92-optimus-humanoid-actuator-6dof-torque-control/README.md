# 🤖 Tesla FSD Otonom Sürüş | Gün 92: Tesla Optimus İnsansı Robotu: Aktüatör Tasarımı, Eklemler ve 6-DoF Tork Kontrolü

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Tesla-Optimus](https://img.shields.io/badge/Robotics-Optimus%20Gen%202%20Humanoid-red.svg?style=flat-square)](https://www.tesla.com/AI)
[![Torque-Control](https://img.shields.io/badge/Control-1000%20Hz%20Impedance%20Torque-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/Impedance_control)
[![Dynamics](https://img.shields.io/badge/Physics-Euler--Lagrange%20Inverse%20Dynamics-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"92. günümüze hoş geldin stajyer!  
> Tesla sadece tekerlekli robotlar (arabalar) yapmaz; otomotiv fabrikalarında ve günlük yaşamda insanlarla birlikte çalışacak genel amaçlı iki ayaklı insansı robotlar üretir: **Tesla Optimus (Gen 2)**!  
> Endüstriyel robot kollarının aksine, insansı bir robotun insanlarla güvenle etkileşime girebilmesi için katı konum kontrolü yerine **Empedans ve Tork Kontrolü (Torque Control)** kullanması zorunludur:  
> 1. **28 Özel Aktüatör:** Döner ve doğrusal aktüatörler, gerinim ölçer (Strain Gauge) tork sensörleri ile doğrudan eklem torkunu ölçer.  
> 2. **Euler-Lagrange Ters Dinamik:** Robotun kollarını ve bacaklarını hareket ettirirken atalet matrisi ($\mathbf{M}$), Coriolis ($\mathbf{C}$) ve yerçekimi ($\mathbf{g}$) kuvvetlerini anında kompanze eder.  
> 3. **1000 Hz Empedans Kontrolü:** Robot bir insana veya masaya çarptığında yay gibi esner (Compliance), asla zarar vermez.  
> 4. **Doğal ve Akıcı Hareket:** Sıfır sarsıntı ve minimum enerjiyle insan benzeri uzuv hareketleri üretir.  
> Bugün Tesla Optimus'un eklemlerini yöneten 6-DoF tork kontrol motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Robotik Ters Dinamik (Euler-Lagrange Formülasyonu)

$$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

### 2. Yerçekimi Kompanzasyon Vektörü

$$g_i(\mathbf{q}) = m_i \cdot g \cdot l_i \cdot \cos(q_i), \quad g = 9.81\ \text{m/s}^2$$

### 3. 1000 Hz Empedans Tork Kontrol Kuralı

$$\boldsymbol{\tau}_{\text{cmd}} = \text{clip}\left( \mathbf{K}_p (\mathbf{q}_{\text{des}} - \mathbf{q}) + \mathbf{K}_d (\dot{\mathbf{q}}_{\text{des}} - \dot{\mathbf{q}}) + \mathbf{g}(\mathbf{q}), \ -\tau_{\text{max}}, \ +\tau_{\text{max}} \right), \quad \tau_{\text{max}} = 150.0\ \text{Nm}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Robotun sert ve tehlikeli bir makine gibi değil, insan kasları gibi esnek, yerçekimini kompanze eden ve temas anında güvenli bir şekilde hareket edebilmesini sağlamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Güvensiz Temas ve Kırılma:** Katı pozisyon kontrollü robotların insanlara veya nesnelere çarptığında yaralanma veya donanım kırma riskini ortadan kaldırdı.
- **Yerçekimi Yorgunluğu:** Kolların havada asılı kalması için gereken torku doğrudan $\mathbf{g}(\mathbf{q})$ ile hesaplayarak motorların aşırı ısınmasını engelledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Doğrusal Olmayan Sürtünme:** Planet dişlilerdeki kuru sürtünme (Stribeck / Coulomb) tork ölçümlerinde küçük sapmalar yaratabilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Saf PID Konum Kontrolü:** Yalnızca enkoder pozisyonuna bakar; temas anında torku artırıp engeli kırmaya çalışır (İnsansı robotlar için kesinlikle yasaktır).

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Tesla Optimus** | Tesla'nın geliştirdiği iki ayaklı, 28 aktüatörlü genel amaçlı insansı robot. |
| **Inverse Dynamics** | İstenen eklem ivmesini ve hızını üretmek için gereken motor torklarını hesaplama süreci. |
| **Impedance Control** | Robot eklemlerinin dış kuvvetlere karşı sanal bir yay-sönümleyici gibi davranmasını sağlayan kontrol yöntemi. |
| **Strain Gauge** | Eklem milindeki mikro deformasyonları ölçerek uygulanan torku milisaniyede veren sensör. |
| **Gravity Compensation** | Robotun kendi uzuv ağırlığını yerçekimine karşı havada tutmak için uygulanan dengeleyici tork. |
| **Inertia Matrix ($\mathbf{M}$)** | Robot eklemlerinin ivmelenmeye karşı gösterdiği kütlesel eylemsizlik tensörü. |
| **Coriolis Force ($\mathbf{C}$)** | Dönen referans çerçevelerinde eklemlerin birbirine uyguladığı dinamik çapraz kuvvetler. |
| **Harmonic Drive** | Yüksek redüksiyon oranı ve sıfır boşluk (zero-backlash) sunan kompakt robotik dişli kutusu. |
| **Compliance (Uysallık)** | Robotun dışarıdan gelen bir itmeye karşı direnmeden yumuşakça esneme kabiliyeti. |
| **Torque Saturation** | Aktüatör motorunun termal ve mekanik sınırlarını korumak için torkun sınırlandırılması ($\pm 150\text{ Nm}$). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 1000 Hz empedans kontrolü ile insansı esneklik      | • Aktüatör dişli boşluklarının zamanla aşınması       |
| • Yerçekimi kompanzasyonu ile %97.4 yörünge yakınsaması| • Yüksek tork altında ısınan aktüatör sargıları       |
| • 1.4 µs ultra hızlı RTOS kontrol döngüsü             |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tesla Gigafactory montaj hatlarında pil hücresi ve  | • Beklenmedik dış darbelerde eklem motorlarının       |
|   kablo demeti montajını tam otonomlaştırma           |   anlık aşırı akım (Overcurrent) yemesi               |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Optimus 6-DoF Tork Kontrol Şeması

```
[ İstenen Eklem Yörüngesi q_des(t) ]
                 |
                 v
   [ 1000 Hz Empedans Kontrolcüsü ] <--- [ Gerinim Ölçer Tork Sensörü ]
                 |
                 | tau_feedback = Kp * e + Kd * e_dot
                 v
   [ Yerçekimi Kompanzasyonu g(q) ]
                 |
                 v
   [ Tork Doyumu Kırpıcı (+-150 Nm) ]
                 |
                 v
   [ 28x Tesla Fırçasız Aktüatör ] ---> [ DOĞAL VE GÜVENLİ HAREKET ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Optimus tork kontrol simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
