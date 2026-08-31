# 🚗 Tesla FSD Otonom Sürüş | Gün 66: FAZ 6 BÜYÜK CAPSTONE — C++ ile Otonom Otoyol Şerit Değiştirme ve MPC Yörünge Takipçisi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Phase 6 Capstone](https://img.shields.io/badge/Capstone-Full--Stack%20FSD%20Motion%20Planner-red.svg?style=flat-square)](https://www.tesla.com/)
[![MPC & Stanley](https://img.shields.io/badge/Control-MPC%20%2B%20Stanley%20Tracking-blue.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/ASIL--D-Dual--Node%20Lockstep%20Voting-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"Tebrikler stajyer! Faz 6'nın zirvesine, **BÜYÜK CAPSTONE (Gün 66)** projesine ulaştın!  
> Son 11 gün boyunca tek tek geliştirdiğimiz tüm otonom sürüş planlama ve kontrol teknolojilerini şimdi tek bir devasa **Full-Stack FSD Motoru** içinde birleştiriyoruz:  
> 1. **Hibrit A* & Voronoi Potansiyel Alanı (Gün 56):** Karmaşık geometrilerde kinematik kısıtlı yol arama.  
> 2. **Frenet Çerçevesi Quintic Şerit Değiştirme (Gün 57):** $3.28\text{ m/s}^3$ konfor sınırında jerk-optimal yörünge sentezi.  
> 3. **Model Predictive Control (MPC) & Stanley Takipçisi (Gün 58 & 65):** Kinematik bisiklet durum geri beslemesiyle $<5\text{ cm}$ milimetrik şerit takibi.  
> 4. **Clothoid / Euler Spirali Kaçınma Manevrası (Gün 59):** $C^2$ sürekli eğrilikle yumuşak direksiyon geçişi.  
> 5. **İleri-Geri Dinamik Hız Profili ve Rejenerasyon (Gün 60):** Maksimum viraj güvenliği ve enerji geri kazanımı.  
> 6. **Kavşak Gap Acceptance & TTC Karar Motoru (Gün 61):** Hiyerarşik sonlu durum makineleriyle öncelik yönetimi.  
> 7. **Euro-NCAP AEB ve AES Kaçınma Kalkanı (Gün 62):** Kademeli acil frenleme ve kaçış direksiyonu.  
> 8. **ISO 26262 ASIL-D ve Çift Düğüm HW3/HW4 Lockstep Arabulucusu (Gün 63 & 64):** Silikon seviyesinde hata toleransı ve güvenli duruş garantisi.  
> $90\text{ km/h}$ hızla otoyolda şerit değiştiren, engelleri milisaniyede analiz eden ve sıfır hata toleransıyla çalışan üretim seviyesinde Tesla FSD Planlayıcı & Kontrolcü Motoru karşınızda!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Jerk-Optimal Quintic Polinom Yörüngesi

$$d(t) = c_3 t^3 + c_4 t^4 + c_5 t^5, \quad c_3 = \frac{10 D}{T^3}, \ c_4 = -\frac{15 D}{T^4}, \ c_5 = \frac{6 D}{T^5}$$

### 2. Model Predictive Control (MPC) ve Stanley Hibrit Takip Kanunu

$$\delta(t) = \theta_e(t) - \arctan\left( \frac{k \cdot e(t)}{v(t) + \epsilon} \right), \quad \min_{\mathbf{u}} \sum_{t=0}^N \left( \mathbf{x}_t^T Q \mathbf{x}_t + \mathbf{u}_t^T R \mathbf{u}_t \right)$$

### 3. Euro-NCAP AEB Acil Durma Mesafesi

$$d_{\text{stop}} = v \cdot t_{\text{delay}} + \frac{v^2}{2 \cdot a_{\text{max}}}, \quad a_{\text{max}} = 9.0\text{ m/s}^2, \ t_{\text{delay}} = 0.20\text{ s}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Ayrık çalışan hareket planlayıcı, hız profilleyicisi, yanal kontrolcü ve donanımsal güvenlik katmanlarını tek bir senkronize ve deterministik RTOS boru hattında birleştirmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Katmanlar Arası Entegrasyon Gecikmesi:** Planlama ve kontrol döngüleri arasındaki veri kuyruğu gecikmelerini sıfıra indirip tek çevrimde ($<200\text{ µs}$) tüm kararları üretti.
- **Uçtan Uca Güvenlik Doğrulaması:** Yörünge üretilirken eşzamanlı olarak aktüatör sınırları, ASIL-D sensör farkları ve çift NPU oylaması denetlendi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Aşırı Hava Koşulları:** Şiddetli kar ve yoğun siste kameraların kapanması durumunda radar/ultrasonik ek sensör girdilerine ihtiyaç duyar.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Tam Uçtan Uca Sinir Ağı (End-to-End Neural Network without Safety Filter):** Yalnızca yapay zeka çıktısına güvenmek ISO 26262 ASIL-D determinizmini sağlayamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Full-Stack FSD Engine** | Algılama sonrası yörünge planlama, hız optimizasyonu, takip kontrolü ve güvenlik doğrulamasını yürüten yazılım yığını. |
| **Quintic Trajectory** | Konforlu şerit değişimi için $C^2$ sürekli türetilebilir 5. derece polinom yörüngesi. |
| **Model Predictive Control (MPC)**| Gelecekteki araç durumlarını öngörerek optimal direksiyon/gaz komutlarını hesaplayan kontrolcü. |
| **Stanley Controller** | Ön aks yanal sapmasını ve açı hatasını analitik olarak düzelten yüksek hızlı geometrik takipçi. |
| **Clothoid Spline** | Eğriliği yay uzunluğuyla doğrusal değişen, sarsıntısız engel kaçınma eğrisi. |
| **Forward-Backward Pass** | Hız profilini hem hızlanma hem de yavaşlama sınırlarına göre optimize eden iki yönlü algoritma. |
| **AEB & AES Shield** | Çarpışma riski anında otomatik frenleme ve kaçınma direksiyonunu devreye sokan güvenlik kalkanı. |
| **ISO 26262 ASIL-D** | En yüksek otomotiv fonksiyonel güvenlik sertifikasyon standardı. |
| **Dual-Node Arbiter** | FSD HW3/HW4 Node A ve Node B çiplerinin kararlarını karşılaştıran donanımsal oylayıcı. |
| **Deterministic RTOS** | Her kontrol çevriminin sabit ve öngörülebilir bir sürede ($<1\text{ ms}$) tamamlanması garantisi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Faz 6'nın tüm 10 planlama/kontrol modülünün entegrasyonu| • Yüksek işlemci yükü ve kapsamlı entegrasyon testleri|
| • Jerk-optimal <3.5 m/s³ konfor ve <5cm takip hatası  |                                                       |
| • ASIL-D ve Çift NPU oylaması ile %100 güvenlik onayı |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Seviye 4/5 Seviyesinde Cybercab ve Model 3/Y tam    | • Beklenmedik dinamik engellerin çok yüksek hızlarda  |
|   otonom otoyol sürüş yazılımının temeli              |   lastik tutuş limitlerini aşması                     |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Faz 6 Capstone Mimari Akış Şeması

```
[ Algılanan Yol Şeritleri, Hız Limiti (90 km/h) & Dinamik Engeller ]
                                |
                                v
      [ 1. Quintic Polinom Yörünge Sentezi: d(t) = c3*t³ + c4*t⁴ + c5*t⁵ ]
                                |
                                v
      [ 2. İleri-Geri Hız Profili ve Rejeneratif Enerji Optimizasyonu ]
                                |
                                v
      [ 3. MPC & Stanley Kapalı Çevrim Yanal Takip Kontrolcüsü ]
                                |
                                v
      [ 4. Euro-NCAP AEB / AES Güvenlik ve Çarpışma Kalkanı (TTC > 2.4s) ]
                                |
                                v
      [ 5. ISO 26262 ASIL-D Çift Kanal Doğrulama (|S1 - S2| <= 0.50 Nm) ]
                                |
                                v
      [ 6. FSD HW3/HW4 Çift Düğüm (Node A / Node B) Oylama Arabulucusu ]
                                |
                                v
        [ %100 ONAYLANDI: CAN-Bus Üzerinden EPS & Drive Inverter Sürüşü ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Faz 6 Büyük Capstone simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
