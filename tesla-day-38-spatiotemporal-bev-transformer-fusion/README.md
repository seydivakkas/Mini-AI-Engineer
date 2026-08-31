# 🚗 Tesla FSD Otonom Sürüş | Gün 38: Mekansal-Zamansal (Spatiotemporal) Öznitelik Füzyonu ve Transformer BEV Dönüşümü

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Transformer](https://img.shields.io/badge/Model-BEVFormer%20Spatiotemporal-red.svg?style=flat-square)](https://www.tesla.com/)
[![Attention](https://img.shields.io/badge/Attention-Spatial%20Cross%20+%20Temporal%20Self-blue.svg?style=flat-square)](https://www.sae.org/)
[![Memory](https://img.shields.io/badge/Feature-Occlusion%20Resistance%20Memory-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"38. günümüze hoş geldin stajyer!  
> Tesla AI Day sunumlarında Andrej Karpathy ve Ashok Elluswamy'nin anlattığı en devrimsel derin öğrenme mimarisi **Spatiotemporal BEV Transformer (BEVFormer)** modelidir:  
> Bir Tesla kavşağa yaklaştığında, sağ taraftaki bir yaya büyük bir kamyonun arkasına geçip 2 saniyeliğine kameraların görüşünden tamamen kaybolabilir (Oklüzyon). İnsan bir sürücü o yayanın yok olmadığını, kamyonun arkasında yürümeye devam ettiğini bilir. Saf bir 2D tespit ağı ise nesneyi anında unutur ve yaya tekrar belirdiğinde acil fren yapıp panikler!  
> Tesla FSD bunu iki kritik Transformer dikkat mekanizmasıyla çözer:  
> 1. **Mekansal Çapraz Dikkat (Spatial Cross-Attention):** 3D uzaydaki BEV ızgara sorguları ($Q_{\text{BEV}}$), 8 kameranın 2D öznitelik haritalarına 3D ışınlar fırlatarak (Ray Casting) pikselleri çeker.  
> 2. **Zamansal Öz-Dikkat (Temporal Self-Attention):** Araç hareket ettikçe ($dx, dy, d\psi$), geçmiş zamandaki ($t-1, t-2$) BEV hafızasını araç dinamiğine göre öteleyerek (Ego-Motion Warp) güncel algıyla birleştirir. Kameralar engeli görmese bile zamansal bellek engelin varlığını korur.  
> Bugün Tesla'nın zamansal hafızaya sahip BEV Transformer füzyon çekirdeğini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Mekansal Çapraz Dikkat (Spatial Cross-Attention)

$$\text{SCA}(\mathbf{Q}_p, \mathbf{F}_{\text{cam}}) = \sum_{i=1}^{N_{\text{cam}}} \sum_{j=1}^{N_{\text{points}}} \mathcal{W}_{ij} \cdot \mathbf{V}\left(\mathbf{F}_{\text{cam}}^{(i)}, \mathcal{P}(\mathbf{p}, z_j)\right)$$

### 2. Ego-Motion Kompanzasyonlu Zamansal Öz-Dikkat

$$\mathbf{B}_{t-1}^{\text{warp}}(x, y) = \mathbf{B}_{t-1}\left( \mathbf{R}_{\Delta \psi} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} \Delta X \\ \Delta Y \end{bmatrix} \right)$$

$$\mathbf{B}_t = \text{TSA}\left(\mathbf{Q}_{\text{BEV}}, \mathbf{B}_{t-1}^{\text{warp}}\right) = \alpha \cdot \mathbf{B}_{\text{spatial}} + (1 - \alpha) \cdot \mathbf{B}_{t-1}^{\text{warp}}$$

### 3. Voksel Doluluk Olasılığı (Occupancy Sigmoid Activation)

$$P(\text{Occupied}_{x, y}) = \sigma\left( \frac{1}{C} \sum_{c=1}^C \mathbf{B}_t(x, y, c) \right)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
8 kameranın anlık 2D piksel verilerini zaman boyutuyla (Time Horizon) birleştirerek oklüzyon, sis veya geçici kör noktalarda bile nesnelerin konumunu hafızada tutmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Oklüzyon Kaybı (Flickering / Dropped Tracks):** Ağaç veya kamyon arkasına geçen nesnelerin takip kimliğinin (ID) kaybolmasını önledi.
- **Kamera Sınır Geçişleri:** Bir araç sol çamurluk kamerasından sol sütun kamerasına geçerken yaşanan kesintileri BEV uzayında pürüzsüzleştirdi.
- **Doğrudan 3D Hız Vektörü:** Zamansal farktan ($B_t - B_{t-1}$) nesnelerin 3D hız vektörlerini ek sensör olmadan doğrudan çıkardı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **NPU Bellek Bant Genişliği:** Zamansal tensör kuyrukları GPU/NPU SRAM belleğinde önemli yer tutar (Kanal budama / INT8 gerekir).
- **Hatalı Odometri:** Tekerlek patinaj yaparsa Ego-Motion Warp açısı sapabilir ve geçmiş bellek yanlış hizalanabilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Kamera Bazlı Tekil 2D İzleme (SORT/DeepSORT):** 2D kutuları Kalman filtresiyle bağlar; 3D uzayda oklüzyonları çözemez.
- **Saf RNN / LSTM Hücreleri:** Transformer Attention mekanizmasına göre uzun süreli uzamsal ilişkileri korumakta yetersiz kalır.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **BEVFormer** | 8 kamerayı spatial ve temporal attention ile BEV ızgarasına dönüştüren Tesla FSD Transformer mimarisi. |
| **Spatial Cross-Attention** | 3D BEV sorgularının 8 kameranın 2D öznitelik tensörlerinden bilgi çekmesini sağlayan dikkat katmanı. |
| **Temporal Self-Attention** | Önceki zaman adımlarına ait BEV haritasını mevcut anla kaynaştıran zamansal dikkat motoru. |
| **BEV Queries ($Q$)** | $50 \times 50$ (veya $100 \times 100$) ızgara üzerindeki her hücreyi temsil eden öğrenilebilir gömme vektörleri. |
| **Ego-Motion Warp** | Aracın yer değiştirmesini ve dönüş açısını hesaba katarak eski BEV tensörünü geometrik öteleme işlemi. |
| **Ray Casting (Işın Fırlatma)** | BEV ızgarasından yukarı doğru farklı Z yüksekliklerinde noktalar alıp kamera piksellerine izdüşürme tekniği. |
| **Occlusion (Oklüzyon)** | Bir nesnenin başka bir engel arkasında kalarak kameraların görüş alanından geçici olarak çıkması durumu. |
| **Persistent Tracking** | Görsel temas kopsa dahi nesnenin koordinat ve hızını bellekte canlı tutma yeteneği. |
| **Occupancy Probability** | BEV ızgarasındaki her hücrenin dolu veya boş olma olasılığı ($P \in [0, 1]$). |
| **Affine Coordinate Shift** | Tensör matrislerinin 2D öteleme ve rotasyonla kaydırılması matematiksel işlemi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Oklüzyon anında %95+ doğrulukla nesne belleği       | • Bellek tensörlerinin NPU RAM tüketimi               |
| • 8 kamerayı tek bir global BEV uzayında birleştirme  | • Odometri kaymalarında zamansal hizalama hatası      |
| • 85 µs ultra hızlı Transformer karar çevrimi         |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tesla HW4 FSD çipinde özel NPU çekirdekleriyle      | • Dinamik olarak yön değiştiren oklüzyon altındaki    |
|   FP8/INT8 hassasiyetinde sıfır gecikmeli yürütme     |   yayaların hareket belirsizliği                      |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Spatiotemporal BEV Transformer Mimarisi

```
[ 8 Kamera 2D Öznitelikleri ] --------> [ Spatial Cross-Attention ]
                                                   |
                                                   v
[ Geçmiş BEV Tensörü (t-1) ] --(Ego Warp)--> [ Temporal Self-Attention ]
                                                   |
                                                   v
                                      [ Güncel BEV Tensörü (t) ]
                                                   |
                                                   v
                                    [ 3D Doluluk & Engel Haritası ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana BEV Transformer simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
