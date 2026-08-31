# 🚗 Tesla FSD Otonom Sürüş | Gün 46: 3D Occupancy Network (Hacimsel Voksel ve Akış Hızı)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Occupancy](https://img.shields.io/badge/Architecture-Tesla%203D%20Occupancy%20Network-red.svg?style=flat-square)](https://www.tesla.com/)
[![VoxelFlow](https://img.shields.io/badge/Velocity-3D%20Voxel%20Flow%20Field-blue.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/Obstacles-Arbitrary--Shaped%20Object%20Shield-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"46. günümüze hoş geldin stajyer!  
> Geleneksel otonom sürüş sistemleri dünyayı 3D Sınırlayıcı Kutularla (3D Bounding Boxes: 'Araba', 'Kamyon', 'Yaya') anlamaya çalışır. Ancak gerçek dünya düzenli geometrik kutulara sığmaz:  
> - Otoyola devrilmiş bir tır,  
> - Kamyondan yola saçılmış inşaat demirleri ve kaya parçaları,  
> - Fırtınada yola devrilen ağaç dalları.  
> Bir yapay zekaya 'Bu nesnenin sınıfı ne?' diye sorduğunuzda eğer eğitim verisinde yoksa onu yok sayabilir ve ölümcül kazalara yol açabilir.  
> Tesla bu problemi **3D Occupancy Network** ile çözdü:  
> 1. **3D Voksel Izgarası ($100 \times 100 \times 16$):** Aracın etrafındaki 3D uzay küçük küplere (Voxel) bölünür. Sistem nesnenin ne olduğunu bilmek zorunda değildir; o vokselin **DOLU** mu yoksa **BOŞ** mu olduğunu ($P_{\text{occ}}$) kesin olarak kestirir.  
> 2. **3D Voxel Flow (Hacimsel Hız Alanı):** Her bir dolu vokselin uzaydaki anlık 3D hız vektörü ($\vec{v} = [v_x, v_y, v_z]^T$) hesaplanır.  
> 3. **Kutulanamaz Engel (Arbitrary-Shape) Güvencesi:** Ne olduğu bilinmeyen her türlü cisim için fiziksel frenleme alanı garantilenir.  
> Bugün Tesla AI Day'in en çığır açıcı vizyonunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Voksel Doluluk Olasılığı (Sigmoid)

$$P_{\text{occ}}(x, y, z) = \sigma(z_{\text{logit}}) = \frac{1}{1 + e^{-z_{\text{logit}}}}$$

### 2. 3D Voxel Flow ve Süreklilik Denklemi

$$\vec{\mathbf{v}}(x, y, z) = \begin{bmatrix} v_x(x,y,z) \\ v_y(x,y,z) \\ v_z(x,y,z) \end{bmatrix}, \quad \frac{\partial P_{\text{occ}}}{\partial t} + \nabla \cdot (P_{\text{occ}} \vec{\mathbf{v}}) = 0$$

### 3. Deformable 3D Voxel-to-Camera Projeksiyonu

$$\mathbf{p}_{\text{cam}_i} = \mathbf{K}_i \cdot \left( \mathbf{R}_i \cdot \mathbf{P}_{\text{voxel}} + \mathbf{t}_i \right)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Sınıflandırılamayan ve standart 3D kutulara uymayan bilinmeyen tehlikeli yol engellerine karşı mutlak geometrik çarpışma önleme sağlamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Sınıflandırma Körlüğü (Out-of-Distribution Objects):** Eğitilmemiş nesnelerin yok sayılmasını önleyerek saf doluluk bilgisi sağladı.
- **Doğrudan 3D Hız Kestirimi:** Her vokselin optik akış benzeri 3D hareket vektörüyle dinamik engeller anında izlendi.
- **Kaldırım ve Zemin Sürekliliği:** Yol sınırlarını ve çukurları yüksek çözünürlükle haritaladı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Yüksek Bellek Tüketimi:** $100 \times 100 \times 16 = 160,000$ voksel NPU SRAM üzerinde yoğun bellek alanı kaplar.
- **Oklüzyon Arkası Boşluk:** Büyük bir kamyonun arkasındaki alan kamerayla görülemediğinden belirsizlik taşır.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Sadece 3D Bounding Box (Kutu Tespiti):** Düşük bellek tüketir ancak kutulanamayan nesnelerde tamamen kördür.
- **LiDAR Voxelization:** Mükemmel geometrik doğruluk sunar ancak çok pahalıdır.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **3D Occupancy Network** | 3D uzayı voksellere bölerek her vokselin doluluğunu tahmin eden hacimsel yapay zeka ağı. |
| **Voxel (Volumetric Pixel)** | 3 boyutlu uzaydaki en küçük hacim elemanı / 3D piksel. |
| **3D Voxel Flow** | Her bir vokselin 3D uzaydaki yer değiştirme hız vektörü ($\vec{v} = [v_x, v_y, v_z]^T$). |
| **Arbitrary-Shaped Obstacle**| Standart prizma kutulara sığmayan düzensiz şekilli engeller (dökülen yük, ağaç gövdesi). |
| **Sigmoid Activation** | Ham nöron logit değerlerini $[0, 1]$ aralığında doluluk olasılığına dönüştüren fonksiyon. |
| **Deformable Attention** | 3D voksel sorgularının 8 kamera görüntüsü üzerinde esnek referans noktalarından öznitelik toplaması. |
| **Occupancy Grid Map** | Çevrenin boş, dolu ve bilinmeyen hücreler halinde temsil edildiği olasılıksal harita. |
| **Static vs Dynamic Voxels** | Voxel Flow hızına göre hareketsiz (bina, yol) veya hareketli (araç, yaya) ayrımı. |
| **NPU SRAM Bandwidth** | 3D tensörlerin işlemci çekirdeklerine taşınma hızı ve bellek bant genişliği. |
| **Semantic Voxel** | Her bir dolu voksele atanan semantik sınıf etiketi (Asfalt, Araç, Yaya, Bariyer). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Ne olduğu bilinmeyen her nesneye karşı tam koruma   | • 40k - 160k voksel için yüksek NPU bellek yükü       |
| • 3D Voxel Flow ile anlık hız alanı kestirimi         | • Görüş hattı arkasındaki oklüzyon belirsizliği       |
| • 450 µs ultra hızlı RTOS çözümleme performansı       |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • NeRF otomatik etiketleme ile milyonlarca saatlik    | • Yoğun yağmurda su serpintilerinin dolu voksel       |
|   gerçek sürüş verisinden zemin gerçeği üretimi       |   olarak algılanması (False Positive)                 |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla 3D Occupancy Network Mimarisi

```
[ 8 Kamera Görüntüleri ] ===> [ 2D Çok Ölçekli Öznitelikler ]
                                            |
                                            v
[ 3D Voxel Sorguları (Query Grid) ] ===> [ Deformable Cross-Attention ]
                                            |
                                            v
                                 [ 3D Hacimsel Voksel Alanı ]
                                            |
                         +------------------+------------------+
                         |                                     |
                         v                                     v
             [ Doluluk Olasılığı P_occ ]             [ 3D Voxel Flow Hız Vektörü ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana 3D Occupancy simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
