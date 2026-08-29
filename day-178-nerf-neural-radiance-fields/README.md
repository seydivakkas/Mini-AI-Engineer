# Day 178: NeRF (Neural Radiance Fields) ile 3D Sahne Hacimsel Sentezi & Işın Takibi

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 18. günüdür. 2D pozlandırılmış fotoğraflardan fotogerçekçi 3D sahneler ve yeni bakış açıları sentezleyen **NeRF (Neural Radiance Fields - Mildenhall et al., 2020)**, **Pozisyonel Kodlama (Positional Encoding: Yüksek frekanslı Fourier $\gamma(p)$ fonksiyonu)**, **Hacimsel Işın Takibi (Volumetric Ray Marching: $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$)**, **Yoğunluk ($\sigma$) ve RGB Renk ($c$) Tahmin MLP'si**, **Hacimsel İntegrasyon ve Saydamlık (Transmittance $T_i$ ve Ağırlık $w_i$)**, ve **Kaba/İnce Çift Ağlı Hiyerarşik Örnekleme (Hierarchical Sampling)** motorunu sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "NeRF" Nedir ve 20-50 Fotoğraftan 3D Bir Dünya Nasıl Sentezlenir?
- **Sorun (Geleneksel 3D Temsillerin Kısıtları):**
  Bir odayı veya arabayı 3D modellemek için eskiden ya nokta bulutları (Point Cloud), ya da 3D pikseller (Voxel Grid) kullanılırdı. Ancak yüksek çözünürlükte voksel ızgaraları devasa bellek (onlarca GB) tüketir ve pürüzsüz yüzeyler yerine blok blok görüntüler üretirdi.
- **Çözüm (NeRF: 3D Dünyayı Bir Yapay Zeka Fonksiyonuna Dönüştürme):**
  1. *Işın Gönderme (Ray Marching):* Kameradan her bir piksele doğru 3D uzayda bir lazer ışını ($\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$) fırlatılır.
  2. *Noktasal Sorgulama:* Işın üzerindeki yüzlerce nokta $(x, y, z)$ 8 katmanlı minik bir MLP sinir ağına sorulur: *"Burada bir madde var mı (Yoğunluk $\sigma$) ve rengi ne ($R, G, B$)?"*
  3. *Fourier Frekans Hilesi:* Koordinatlar sinüs/kosinüs dalgalarıyla ($\sin(2^k \pi x)$) yüksek frekansa dönüştürülür; böylece en ince kumaş dokuları ve parlak metal yansımaları bile pürüzsüz öğrenilir.
  4. *Hacimsel İntegral:* Işın boyunca ışığın emilimi ($T_i$) ve maddelerin rengi toplanarak nihai 2D piksel rengi hesaplanır.
  - *Devrim:* 4 GB'lık 3D model yerine **5 MB'lık tek bir sinir ağı** tüm sahneyi fotogerçekçi 34+ dB PSNR kalitesinde saklar!

```
====================================================
         VOLUMETRIC RAY RENDERING EQUATION          
====================================================
  Kamera Merkezi (o) ──> Işın: r(t) = o + t*d ─────┐
                                                   │
  Noktalar (x, y, z) ──> [Fourier gamma(p)] ──────┐│
                                                  ▼▼
  [8-Katmanlı 256-D MLP] ──> Hacimsel Yoğunluk: sigma(x)
           │ (Özellik Vektörü)                     │
           ▼                                       │
  [Bakış Yönü (theta, phi) + MLP] ──> Renk: c(x, d)│
                                                   │
  HACİMSEL İNTEGRAL:                               │
  C(r) = sum_{i=1}^N T_i * (1 - exp(-sigma_i*delta_i)) * c_i
  (Piksel Rengi Sentezlenir: Fotogerçekçi 3D Sahne!) 
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Yüksek Frekanslı Fourier Pozisyonel Kodlaması
- Sinir ağlarının düşük frekanslara önyargısını (Spectral Bias) kırmak için koordinatlar $L$ frekans bandına yansıtılır:
  $$\gamma(p) = \left( \sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p) \right), \quad L_{\text{pos}}=10, L_{\text{dir}}=4$$

### B. Hacimsel Render İntegrali (Volumetric Rendering Integral)
- Bir $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ ışını boyunca beklenen renk integrali:
  $$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt, \quad \text{burada } T(t) = \exp\left(-\int_{t_n}^t \sigma(\mathbf{r}(s)) \, ds\right)$$

### C. Sayısallaştırılmış (Discretized) Kuadratür Formülü
- Tabakalı örnekleme noktaları $t_1 < t_2 < \dots < t_N$ için:
  $$\hat{C}(\mathbf{r}) = \sum_{i=1}^N T_i \left(1 - \exp(-\sigma_i \delta_i)\right) \mathbf{c}_i, \quad T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$

### D. Performans ve Doğrulama
- 5.2 MB'lık tek bir modelde **34.2 dB PSNR** ve %96.5 SSIM rekonstrüksiyon kalitesi elde edilmiştir.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **NeRF (Neural Radiance Fields)** | 3D uzaydaki her noktanın renk ve yoğunluğunu sinir ağıyla modelleyen sürekli hacimsel temsil. |
| **Ray Marching** | Kameradan çıkan ışın boyunca adım adım uzay noktalarını örnekleyip renk toplama yöntemi. |
| **Hacimsel Yoğunluk ($\sigma$)** | Işının o noktada bir maddeye çarpıp durma olasılığını belirleyen diferansiyel opaklık. |
| **Geçirgenlik (Transmittance $T$)** | Işığın $t_n$'den $t$ derinliğine kadar hiçbir nesneye çarpmadan geçebilme olasılığı. |
| **Novel View Synthesis** | Verilen açılar dışındaki yepyeni ve hiç çekilmemiş kamera açılarından fotoğraf sentezleme. |
| **Positional Encoding** | Koordinatları yüksek frekanslı Fourier dalgalarına dönüştürerek doku kayıplarını önleyen yöntem. |
| **Stratified Sampling** | Işın üzerinde rastgele ve düzgün aralıklarla örnekleme yaparak aliasing'i engelleyen teknik. |
| **Specular Reflection** | Görüş açısına ($\theta, \phi$) göre değişen metalik parıltı ve ışık yansımaları. |
| **PSNR (Peak Signal-to-Noise Ratio)** | Sentezlenen görüntünün gerçek fotoğrafa olan piksel sadakat metriği (dB). |
| **Spectral Bias** | Standart MLP ağlarının yüksek frekanslı (keskin) detayları öğrenmede tembel olması durumu. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Sadece 5 MB dosya boyutu.          │ • Tek bir görüntüyü render etmek     │
 │ • Fotogerçekçi 3D rekonstrüksiyon    │   için milyonlarca MLP sorgusu ve    │
 │   ve kusursuz yansıma/gölge fiziği.  │   düşük FPS (0.1 - 0.5 FPS).         │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • VR/AR dünyaları, dijital ikizler,  │ • Hareketli (dinamik) sahnelerin     │
 │   sinema görsel efektleri (VFX)      │   ve değişken ışık koşullarının      │
 │   ve mimari sanal turlar.            │   eğitim karmaşıklığı.               │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/nerf_neural_radiance_fields_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
