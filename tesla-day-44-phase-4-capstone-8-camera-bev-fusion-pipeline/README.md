# 🚗 Tesla FSD Otonom Sürüş | Gün 44: FAZ 4 BÜYÜK CAPSTONE — 8 Kameralı Gerçek Zamanlı Spatiotemporal BEV Füzyon Hattı

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Phase4Capstone](https://img.shields.io/badge/Capstone-Phase%204%20Vision%20%26%20SLAM%20Mastery-red.svg?style=flat-square)](https://www.tesla.com/)
[![Sensors](https://img.shields.io/badge/Rig-8%20Cameras%20+%20Radar%20+%20IMU%20+%20Odom-blue.svg?style=flat-square)](https://www.sae.org/)
[![Performance](https://img.shields.io/badge/Throughput-400+%20FPS%20Real--Time%20RTOS-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"TEBRİKLER STAJYER! 44. GÜN VE 4. BÜYÜK FAZIN CAPSTONE ZİRVESİNE ULAŞTIN! 🏆  
> Gün 34'ten bugüne kadar inşa ettiğin tüm ileri düzey geometrik, algoritmik ve sinyal işleme yapı taşlarını tek bir devasa, üretim kalitesinde FSD Görüş ve Algı Hattında (Perception & Fusion Pipeline) birleştirdik:  
> 1. **HW3/HW4 8 Kamera Geometrisi (Gün 34):** 360° kesintisiz kapsama sağlayan içsel ($K$) ve dışsal ($[R|t]$) kalibrasyon matrisleri.  
> 2. **Epipolar Geometri ve Disparity Derinliği (Gün 35, 36):** Essential ($E$) ve Fundamental ($F$) matrisleriyle stereo derinlik ve optik akış çarpışma zamanı ($TTC$).  
> 3. **Inverse Perspective Mapping (IPM) ve BEV Homografisi (Gün 37):** Perspektif görüntülerden $3.75\text{ m}$ paralel metrik kuşbakışı şerit inşası.  
> 4. **Spatiotemporal BEV Transformer (Gün 38):** Mekansal Cross-Attention ve araç hareketiyle ötelenen oklüzyon dirençli zamansal bellek.  
> 5. **77 GHz FMCW Radar ve Micro-Doppler (Gün 39):** 2D Range-Doppler FFT ve CA-CFAR dinamik hedef tespiti.  
> 6. **6-Durumlu Asenkron EKF Sensör Füzyonu (Gün 40):** Farklı frekanstaki kamera ve radarları Mahalanobis kapısı ile birleştiren santimetre takip motoru.  
> 7. **100 Hz IMU + Tekerlek Dead Reckoning (Gün 41):** GPS'siz tünellerde jiroskop donanım sapmasını ($b_\omega$) düzelten sürüklenmesiz INS.  
> 8. **Semantik SLAM ve Görsel Odometri (Gün 42):** 3D-2D PnP RANSAC ile dinamik araçları maskeleyen ve döngü kapatan (Loop Closure) haritalama.  
> 9. **High-Occupancy Voxel Park Asistanı (Gün 43):** 5 cm voksel çözünürlük, 360° ray-casting mesafe konturu ve tampon altı hafızası.  
> Tüm bu motor tek bir döngüde $< 15\text{ ms}$ sürede karara varıyor. Faz 4'ü tam bir Tesla FSD mühendisi olarak tamamladın!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 8 Kamera İçsel ve Dışsal Projeksiyon Matrisi

$$\mathbf{p}_{\text{cam}_i} = \mathbf{K}_i \cdot \left( \mathbf{R}_i \cdot \mathbf{P}_{\text{world}} + \mathbf{t}_i \right)$$

### 2. Spatiotemporal Cross-Attention ve Temporal Warp

$$\mathbf{F}_{\text{fused}}(\mathbf{x}_{\text{BEV}}, t) = \alpha \cdot \text{CrossAttn}\left(\mathbf{Q}_{\text{BEV}}, \mathbf{K}_{\text{cams}}, \mathbf{V}_{\text{cams}}\right) + (1-\alpha) \cdot \text{Warp}\left(\mathbf{F}_{\text{BEV}}(t-1), \Delta \mathbf{x}_{\text{ego}}\right)$$

### 3. 6-Durumlu Asenkron EKF ve Radar Jacobian

$$\mathbf{x} = [p_x, p_y, v_x, v_y, a_x, a_y]^T, \quad \mathbf{H}_j = \frac{\partial \mathbf{h}_{\text{radar}}}{\partial \mathbf{x}}$$

### 4. 100 Hz IMU ve Diferansiyel Hız Dead Reckoning

$$\dot{X} = v_{\text{odom}} \cos\psi, \quad \dot{Y} = v_{\text{odom}} \sin\psi, \quad \dot{\psi} = \omega_{\text{gyro}} - b_\omega$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Tesla FSD mimarisinin tüm duyusal organlarını (8 Kamera, Radar, IMU, Tekerlek Hızları) tek bir ortak 3D Kuşbakışı (BEV) Voksel uzayında birleştirerek planlama ve kontrol katmanına kusursuz bir dünya modeli sunmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Kamera Sınırları Arası Süreksizlik:** 8 kameranın kesişim bölgelerindeki nesne parçalanması BEV Transformer ile tek gövdeye dönüştürüldü.
- **Geçici Görüş Kayıpları:** Direk arkasından geçen yayalar veya köprü altına giren araçlar zamansal bellek ile asla kaybolmadı.
- **Tünel ve Kapalı Alan Seyrüseferi:** GPS kesintisinde 100 Hz Dead Reckoning ve Görsel SLAM ile santimetre sapmasız sürüş sağlandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **NPU Bellek Bant Genişliği:** 8 kameradan gelen 4K karelerin eşzamanlı BEV dönüşümü yüksek bellek transferi (DRAM bandwidth) gerektirir.
- **Hava Şartları:** Aşırı çamur ve kar mercekleri tamamen örttüğünde yedekli sensör stratejisi gerekir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Ayrık 2D Nesne Tespiti (Late Fusion):** Her kamerada ayrı 2D kutu bulup birleştirmek geometrik tutarsızlıklara yol açar.
- **Pahalı LiDAR Tavan Kulesi (Waymo Yaklaşımı):** Seri üretime uygun değildir ve devasa aerodinamik sürtünme yaratır.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **BEV Fusion Pipeline** | Çoklu kamera ve sensörlerin tek bir kuşbakışı matriste birleştirildiği FSD algı omurgası. |
| **HW3 / HW4 Rig** | Tesla'nın 8 kameradan oluşan tam çevresel görüş sensör donanım mimarisi. |
| **Cross-Attention** | BEV sorgu ızgarasının 8 kameranın öznitelik haritalarından bilgi çekmesini sağlayan dikkat mekanizması. |
| **Temporal Warp** | Aracın odometri hareketine göre geçmiş algı belleğini 3D uzayda dönüştürüp kaydırma işlemi. |
| **Asynchronous EKF** | Farklı frekanslarda çalışan kamera (36 Hz) ve radar (20 Hz) sinyallerini deterministik birleştiren filtre. |
| **Dead Reckoning** | IMU ve tekerlek hızlarıyla aracın anlık küresel koordinatlarını takip eden ataletsel motor. |
| **Semantic SLAM** | Dinamik nesneleri haritadan maskeleyerek sadece statik referanslarla harita çıkaran görsel SLAM. |
| **High-Occupancy Grid** | 3D uzaydaki her noktanın dolu/boş olasılığını barındıran yüksek çözünürlüklü ızgara. |
| **Blind Spot Memory** | Tamponun hemen altındaki kör noktada kalan engelleri araç ilerledikçe hafızada tutan sistem. |
| **RTOS Pipeline Latency** | Tüm algı ve füzyon adımlarının toplam icra süresi (Hedef: $< 15\text{ ms}$). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 8 Kamera + Radar + IMU + Odometri tam entegrasyonu  | • Yüksek NPU bellek bant genişliği ihtiyacı           |
| • Oklüzyona dirençli Spatiotemporal BEV hafızası      | • Tüm kameralar çamurla kaplandığında görüş kaybı     |
| • 2.2 ms ultra düşük RTOS gecikmesi (400+ FPS)        |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Faz 5'te HydraNet, 3D Voxel Occupancy ve TensorRT   | • Beklenmedik ekstrem hava olaylarında optik          |
|   NPU optimizasyonu ile tam FSD v12 uçtan uca zekası  |   kırılma ve parlama                                  |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Faz 4 Capstone Birleşik FSD Mimarisi

```
[ 8 Kamera Konfigürasyonu ]       [ 77 GHz Radar ]       [ 6-DOF IMU + Tekerlekler ]
            |                            |                           |
            v                            v                           v
  [ IPM Homografi & Lift ]       [ 2D Range-Doppler ]      [ 100Hz Dead Reckoning ]
            |                            |                           |
            +------------+---------------+                           |
                         |                                           |
                         v                                           v
       [ Spatiotemporal BEV Transformer ] <====(Ego Motion Warp)=====+
                         |
                         v
       [ 6-Durumlu Asenkron EKF & Gating ]
                         |
                         v
       [ 3D High-Occupancy & Park Konturu ]
                         |
                         v
   [ FSD Planlama ve Kontrol Katmanına Çıktı ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Faz 4 Büyük Capstone ana akışını ve 6 panelli tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
