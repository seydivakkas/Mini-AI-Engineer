# Day 25: Nesne Tespiti Temelleri & Bounding Box Regresyonu (Object Detection Basics)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; modern nesne tespiti (Object Detection) mimarilerinin (YOLO, SSD, Faster R-CNN, RetinaNet) temel yapı taşlarını sıfırdan inşa eder. **Bounding Box Koordinat Dönüşümleri (VOC / COCO / YOLO)**, **Intersection over Union (IoU / GIoU / DIoU)**, **Klasik NMS vs Soft-NMS Algoritmaları**, **Çok Ölçekli Anchor Box Üretimi** ve **Bounding Box Delta Regresyon Hedeflerini** kapsayan 4 panelli endüstri standardı bir teşhis panosu (Diagnostic Dashboard) sunar.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Görsel Sınıflandırmadan Nesne Tespitine Geçiş
Sınıflandırma problemi *"Görselde ne var?"* ($y \in \{1, \dots, C\}$) sorusuna yanıt verirken; Nesne Tespiti problemi *"Hangi nesneler görselin neresinde kaç adet bulunuyor?"* sorusunu eşzamanlı olarak çözer:
- **Çıktı Vektörü:** Her tespit edilen nesne için $\mathbf{y} = [p_{\text{obj}}, x, y, w, h, c_1, c_2, \dots, c_C]^T$

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Sınırlayıcı Kutu (Bounding Box)** | *Bounding Box Representation* | Nesnenin konumunu belirten $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ (Pascal VOC) veya $[x_c, y_c, w, h]$ (YOLO) formatındaki koordinat tensörü. |
| **Kesişim / Birleşim (IoU)** | *Intersection over Union* | Tahmin edilen kutu ile gerçek etiket kutusunun kesişim alanının birleşim alanına oranı ($IoU = \frac{A \cap B}{A \cup B}$). |
| **Maksimum Olmayan Bastırma** | *Non-Maximum Suppression (NMS)* | Aynı nesne için üretilen yüksek örtüşmeli çoklu kutular arasından en yüksek güven skorlu olanı seçip diğerlerini eleyen algoritma. |
| **Ortalama Kesinlik (mAP)** | *Mean Average Precision (mAP@0.5)* | Farklı IoU eşiklerinde tüm sınıflar için hesaplanan kesinlik-duyarlılık eğrisi altındaki alanların ortalaması. |

---

## 2. Bounding Box Formatları

```
1. Pascal VOC (xyxy)  : [x_min, y_min, x_max, y_max] -> Piksel mutlak koordinatları
2. MS COCO    (xywh)  : [x_min, y_min, width, height] -> Sol-üst köşe + Genişlik/Yükseklik
3. YOLO       (cxcywh): [cx, cy, width, height]       -> Merkez koordinatları (Normalize 0..1)
```

---

### 3. Çakışma Metrikleri (IoU Ailesi)

#### A. Standart Intersection over Union (IoU)
İki kutu arasındaki kesişim alanının birleşim alanına oranıdır:

$$\text{IoU} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1) + \text{Area}(B_2) - \text{Area}(B_1 \cap B_2)}$$

#### B. Generalized IoU (GIoU — Rezatofighi et al., 2019)
Kutular tamamen ayrık olduğunda ($\text{IoU} = 0$), standart IoU gradyan üretemez. GIoU, iki kutuyu da çevreleyen en küçük kutu $C$ üzerinden gradyan akışını sağlar:

$$\text{GIoU} = \text{IoU} - \frac{\text{Area}(C) - \text{Area}(B_1 \cup B_2)}{\text{Area}(C)} \quad \in [-1, 1]$$

#### C. Distance-IoU (DIoU — Zheng et al., 2020)
Kutuların merkez noktaları arasındaki Öklid mesafesi $\rho(b_1, b_2)$ ve çevreleyen kutunun köşegen uzunluğu $c$ ile normalize edilir:

$$\text{DIoU} = \text{IoU} - \frac{\rho^2(b_1, b_2)}{c^2}$$

---

### 4. Non-Maximum Suppression (NMS) ve Soft-NMS

1. **Klasik NMS (Açgözlü / Greedy):**
   - Tespit adaylarını güven skorlarına göre azalan sırada sırala.
   - En yüksek skorlu $M$ kutusunu seç.
   - $M$ ile $\text{IoU}(M, b_i) > \theta_{\text{nms}}$ olan diğer tüm çakışan kutuları tamamen ele.
2. **Soft-NMS (Bodla et al., 2017):**
   - Birbiriyle çakışan iki farklı nesne olduğunda klasik NMS arkadaki nesneyi yanlışlıkla yok edebilir.
   - Soft-NMS kutuyu silmek yerine güven skorunu IoU oranına bağlı olarak Gaussian fonksiyonu ile sönümler:
     $$s_i \leftarrow s_i \cdot \exp\left( - \frac{\text{IoU}(M, b_i)^2}{\sigma} \right)$$

---

### 5. Anchor Box Üretimi ve Bounding Box Regresyon Hedefleri

Grid hücreleri ($H \times W$) üzerinde farklı ölçek ve en-boy oranlarında ($1:1, 1:2, 2:1$) üretilen öncül kutulardır (Prior/Anchor Boxes).

#### Delta Regresyon Formülasyonu:
Bir anchor kutusu $A = (x_a, y_a, w_a, h_a)$ ile hedef Ground Truth $G = (x^*, y^*, w^*, h^*)$ arasındaki öğrenme hedefleri:

$$t_x = \frac{x^* - x_a}{w_a}, \quad t_y = \frac{y^* - y_a}{h_a}$$

$$t_w = \ln\left(\frac{w^*}{w_a}\right), \quad t_h = \ln\left(\frac{h^*}{h_a}\right)$$

---

## 🛠️ Dizin Yapısı

```
day-25-object-detection-basics/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, torch, matplotlib, pytest
├── ana_akis.py                      # Uçtan uca nesne tespiti, NMS ve anchor akışı
├── README.md                        # Detaylı teorik ve mentorluk dokümantasyonu
├── src/
│   ├── __init__.py
│   ├── kutu_donusturucu.py          # VOC, COCO, YOLO format ve normalizasyon
│   ├── iou_hesaplayici.py           # IoU, GIoU ve DIoU matris hesaplama
│   ├── nms_filtresi.py              # Klasik NMS, Sınıfa Duyarlı NMS ve Soft-NMS
│   ├── anchor_ureteci.py            # Anchor grid üretimi, GT matching & regresyon
│   └── gorsellestirici.py           # 4 panelli nesne tespiti panosu çizici
├── testler/
│   ├── __init__.py
│   └── test_nesne_tespiti.py        # 8 adet kapsamlı birim test
└── ciktilar/
    └── nesne_tespiti_paneli.png     # 4 panelli teşhis panosu görseli
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışın Çalıştırılması
```bash
python ana_akis.py
```

### 3. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Nesne tespitinde standart Non-Maximum Suppression (Hard NMS) algoritmasının kalabalık sahnelerde (örneğin yan yana duran iki insan) yaşadığı "False Negative" sorunu nedir ve Soft-NMS bu sorunu nasıl çözer?

> **Mentor Cevabı:**
> 1. **Hard NMS Problemi:** Standart NMS, en yüksek skorlu kutuyla IoU değeri eşiği ($\text{IoU} \ge N_t$) aşan tüm komşu kutuları tamamen siler ($s_i = 0$). Kalabalık sahnelerde üst üste binen gerçek iki nesneden arkada olanın kutusu yanlışlıkla yok edilir.
> 2. **Soft-NMS Çözümü:** Soft-NMS kutuyu tamamen silmek yerine, yüksek IoU örtüşmesine sahip kutuların güven skorunu sürekli bir fonksiyonla (Lineer veya Gauss bozulması: $s_i \leftarrow s_i \cdot \exp(-\frac{\text{IoU}^2}{\sigma})$) kademeli olarak düşürür. Böylece gerçek tespitler sıralamada gerilese bile tamamen kaybolmaz.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
