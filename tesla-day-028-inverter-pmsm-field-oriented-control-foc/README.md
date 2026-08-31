# 🚗 Tesla Güç Aktarma Mimarisi | Gün 28: İnvertör & PMSM Motor Kontrolü (Field Oriented Control - FOC)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Control](https://img.shields.io/badge/Control-Field%20Oriented%20Control%20(FOC)-blue.svg?style=flat-square)](https://www.tesla.com/)
[![Motor](https://img.shields.io/badge/Motor-IPM--SynRM%20PMSM%20350Nm-orange.svg?style=flat-square)](https://www.sae.org/)
[![Frequency](https://img.shields.io/badge/Loop-10%20kHz%20(100%C2%B5s)%20Current%20Loop-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"28. günümüze hoş geldin stajyer!  
> Tesla Model 3 veya Model S Plaid'in gaz pedalına bastığınızda o meşhur koltuğa yapıştıran anlık torku üreten nedir?  
> Aracın arka aksındaki **IPM-SynRM (İç Sabit Mıknatıslı Senkron Relüktans)** motorunu süren SiC (Silisyum Karbür) invertörün içindeki **Field Oriented Control (FOC - Alan Yönlendirmeli Kontrol)** algoritmasıdır!  
> 1. **Clarke Dönüşümü ($abc \to \alpha\beta$):** 3 fazlı $120^\circ$ açılı $i_a, i_b, i_c$ alternatif akımları, sabit $90^\circ$ iki eksenli $\alpha\beta$ çerçevesine indirgenir.  
> 2. **Park Dönüşümü ($\alpha\beta \to dq$):** Rotorun dönme açısı $\theta_e$ ile koordinat sistemi döndürülür; böylece AC sinüzoidal akımlar DC büyüklüklere dönüştürülür:  
>    - $i_d$ (Direct): Manyetik akıyı kontrol eder (Temel hızda $i_d = 0$, yüksek hızda alan zayıflatma için $i_d < 0$).  
>    - $i_q$ (Quadrature): Torku kontrol eder. Tork doğrudan $i_q$ akımıyla orantılıdır ($T_e \propto i_q$).  
> 3. **10 kHz Akım Çevrimi:** Her $100\ \mu\text{s}$'de bir motor akımları okunur, PI denetleyiciler gerilim komutlarını üretir ve SVPWM (Uzay Vektör Darbe Genişlik Modülasyonu) ile 6 MOSFET/IGBT kapısına tetikleme sinyali yollanır.  
> Bugün Tesla'nın çekiş motor kontrol çekirdeğini inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Clarke Dönüşümü (Genlik Değişmez - Amplitude Invariant)

$$\begin{bmatrix} i_\alpha \\ i_\beta \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ \frac{1}{\sqrt{3}} & \frac{2}{\sqrt{3}} & 0 \end{bmatrix} \begin{bmatrix} i_a \\ i_b \\ i_c \end{bmatrix}$$

### 2. Park Dönüşümü (Dönen dq Senkron Çerçevesi)

$$\begin{bmatrix} i_d \\ i_q \end{bmatrix} = \begin{bmatrix} \cos(\theta_e) & \sin(\theta_e) \\ -\sin(\theta_e) & \cos(\theta_e) \end{bmatrix} \begin{bmatrix} i_\alpha \\ i_\beta \end{bmatrix}$$

### 3. Elektromanyetik Tork Denklemi (IPM-SynRM Motoru)
Kutup çifti sayısı $p$, mıknatıs akısı $\psi_f$, $L_d$ ve $L_q$ endüktansları olmak üzere:

$$T_e = \frac{3}{2} p \left[ \psi_f i_q + (L_d - L_q) i_d i_q \right]$$

- İlk terim: **Sabit Mıknatıs Torku (PM Torque)**
- İkinci terim: **Relüktans Torku (Reluctance Torque)** ($L_q > L_d$ olduğu için $i_d < 0$ yapıldığında ilave pozitif tork üretir!).

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Karmaşık AC motor kontrolünü tıpkı bir doğru akım (DC) motoru gibi tork ve akı kanallarını birbirinden bağımsız ve doğrusal kontrol edebilmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Maksimum Tork/Amper (MTPA):** En düşük akımla en yüksek torku üreterek batarya verimini ve menzili artırdı.
- **Sıfır Tork Dalgalanması (Ripple):** Düşük hızlarda ve kalkışlarda motorda sarsıntıyı ve gürültüyü tamamen yok etti.
- **Geniş Hız Aralığı:** Alan zayıflatma (Field Weakening) ile motorun $18,000\text{ RPM}$ gibi çok yüksek hızlara çıkmasını sağladı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Açı Sensörü Hassasiyeti:** Rotor elektriksel açısı $\theta_e$ resolver veya optik enkoderden en ufak bir faz kaymasıyla gelirse tork üretimi çöker.
- **10 kHz Hesaplama Yükü:** Çevrim süresi $< 100\ \mu\text{s}$ olmak zorundadır (Özel DSP/FPU donanımı gerektirir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Doğrudan Tork Kontrolü (DTC - Direct Torque Control):** Histerezis karşılaştırıcı kullanır; çok hızlıdır ancak yüksek akım dalgalanması (Ripple) ve gürültü yaratır.
- **Skaler V/f Kontrolü:** Sanayi tipi motorlarda kullanılır; dinamik tork tepkisi otomotiv için yetersizdir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **FOC (Field Oriented Control)** | Stator akım vektörünü rotor manyetik alanına kilitleyerek torku ve akıyı ayrık kontrol eden yöntem. |
| **Clarke Transform** | 3-fazlı $120^\circ$ stator akımlarını 2-fazlı $90^\circ$ sabit $\alpha\beta$ koordinatlarına dönüştüren işlem. |
| **Park Transform** | Sabit $\alpha\beta$ eksenlerini rotorla birlikte dönen $dq$ koordinat eksenlerine dönüştüren işlem. |
| **Direct Current ($i_d$)** | Rotor manyetik akı ekseni doğrultusundaki akım (Manyetik alanı güçlendirir veya zayıflatır). |
| **Quadrature Current ($i_q$)** | Rotor manyetik kutbuna $90^\circ$ dik olan ve doğrudan mekanik tork üreten akım bileşeni. |
| **IPM-SynRM** | Rotorun içine gömülü sabit mıknatıslar ve manyetik bariyerler içeren yüksek verimli hibrit relüktans motoru. |
| **MTPA (Maximum Torque Per Ampere)** | Verilen bir stator akımı büyüklüğü için maksimum elektromanyetik torku üreten optimum $i_d/i_q$ oranı. |
| **Field Weakening (Alan Zayıflatma)** | Yüksek hızlarda zıt elektromotor kuvveti (BEMF) yenmek için $i_d < 0$ basarak net akıyı zayıflatma tekniği. |
| **Rotor Angle ($\theta_e$)** | Rotorun elektriksel açı pozisyonu ($\theta_e = p \cdot \theta_{\text{mech}}$). |
| **Anti-Windup** | PI denetleyicinin çıkış doyumuna ulaştığında integral toplamının şişmesini engelleyen güvenlik devresi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %97+ motor verimi ve sıfır tork dalgalanması        | • Hassas rotor açı pozisyonu (Resolver) bağımlılığı   |
| • 2.45 µs ultra hızlı DSP/MCU icra süresi             | • Motor sıcaklığı değiştikçe Rs ve Ld/Lq kayması      |
| • Ludicrous modda 350 Nm anlık tork tepkisi           |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Sensörsüz (Sensorless Sliding Mode Observer) FOC    | • Resolver sinyal kablosunda gürültü oluşması halinde |
|   ile fiziksel resolver maliyetini sıfırlama          |   motorun kontrolsüz frenleme yapma riski             |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & FOC Kontrol Blok Şeması

```
   Hedef Tork (T_ref) ----> [ MTPA / Tork Haritası ] ---> i_q_ref
                                                      ---> i_d_ref (0 veya Alan Zayıflatma)
                                                                |
                                                                v
   (i_d_ref - i_d) ----> [ PI Denetleyici ] ----> v_d ----+
   (i_q_ref - i_q) ----> [ PI Denetleyici ] ----> v_q ----+
                                                          |
                                                          v
                                              [ Ters Park (dq -> αβ) ]
                                                          |
                                                          v
                                              [ Ters Clarke (αβ -> abc) ]
                                                          |
                                                          v
                                              [ 6-MOSFET / IGBT İnvertör ]
                                                          |
                                                          v
                                              [ 3-Faz PMSM Çekiş Motoru ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana FOC motor kontrol simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
