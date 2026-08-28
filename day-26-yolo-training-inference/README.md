# Day 26: YOLO ile Nesne Tespiti Eğitimi & Çıkarımı (YOLO Training & Inference)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8%20%2F%20YOLO11-00ADEF.svg?style=flat-square&logo=yolo)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; endüstri standardı modern nesne tespiti mimarisi olan **Ultralytics YOLO (YOLOv8 / YOLO11)** ile sıfırdan özel veri seti (Custom Dataset) hazırlama, eğitim döngüsü yürütme, **mAP@0.5** ve **COCO mAP@0.5:0.95** metriklerini bağımsız doğrulama ve yüksek hızlı görsel çıkarımı (Inference) işlemlerini kapsayan 4 panelli bir teşhis panosu (Diagnostic Dashboard) sunar.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. YOLO Mimari Evrimi ve Tek Aşamalı (Single-Stage) Yaklaşım
İki aşamalı dedektörler (Faster R-CNN: Önce RPN ile Region Proposal üret, sonra sınıflandır) yüksek doğruluk sunarken; **YOLO (You Only Look Once)** tüm görseli tek bir ileri beslemede (single forward pass) ızgaralara bölerek eşzamanlı lokalizasyon ve sınıflandırma yapar.

```
+-------------------------------------------------------------------------------+
| GİRİŞ GÖRSELİ (512x512x3)                                                    |
+---------------------------------------+---------------------------------------+
                                        │
                                        ▼
+-------------------------------------------------------------------------------+
| OMURGA (BACKBONE): CSPDarknet / C2f / C3k2 (Özellik Çıkarımı)                 |
+---------------------------------------+---------------------------------------+
                                        │
                                        ▼
+-------------------------------------------------------------------------------+
| BOYUN (NECK): PANet / BiFPN (Çok Ölçekli Özellik Piramidi Birleştirme)        |
+---------------------------------------+---------------------------------------+
                                        │
                                        ▼
+-------------------------------------------------------------------------------+
| AYRIŞTIRILMIŞ BAŞLIK (DECOUPLED HEAD):                                        |
|   ├── Sınıflandırma Dalı (Classification Head: BCE Loss)                      |
|   └── Regresyon Dalı (Bounding Box Head: CIoU Loss + Distribution Focal Loss) |
+-------------------------------------------------------------------------------+
```

---

### 2. Modern YOLOv8/YOLO11 İnovasyonları

#### A. Anchor-Free Mimari
Önceden tanımlanmış sabit anchor kutularına olan bağımlılığı kaldırarak, her özellik haritası pikselini doğrudan bir nesne merkezi adayı olarak değerlendirir.

#### B. Distribution Focal Loss (DFL — Li et al., 2020)
Bounding box kenarlarını tek bir sürekli sayı ($x, y$) olarak tahmin etmek yerine, belirsizliği modelleyen bir olasılık dağılımı (Probability Distribution) üzerinden öğrenir:

$$\text{DFL}(S_i, S_{i+1}) = - \Big( (y_{i+1} - y) \log(S_i) + (y - y_i) \log(S_{i+1}) \Big)$$

---

### 3. Değerlendirme Metrikleri: mAP@0.5 vs mAP@0.5:0.95

1. **Ortalama Hassasiyet (Average Precision - AP):**
   Precision-Recall eğrisi altındaki alanın integrali:
   $$\text{AP} = \int_{0}^1 P(R) \, dR = \sum_{k=1}^N (R_k - R_{k-1}) P_{\text{interp}}(R_k)$$

2. **mAP@0.5 (PASCAL VOC Standardı):**
   Yalnızca $\text{IoU} \ge 0.50$ eşiğinde hesaplanan sınıflar arası ortalama AP.

3. **mAP@0.5:0.95 (MS COCO Standardı):**
   Kutunun hassas yerleşim kalitesini ölçmek için 10 farklı IoU eşiğinin ($0.50, 0.55, 0.60, \dots, 0.95$) ortalamasıdır:
   $$\text{mAP}@[0.5:0.95] = \frac{1}{10} \sum_{\text{IoU}=0.50}^{0.95} \text{mAP}_{\text{IoU}}$$

---

## 🛠️ Dizin Yapısı

```
day-26-yolo-training-inference/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # ultralytics, torch, opencv, pyyaml vb.
├── ana_akis.py                      # Uçtan uca veri üretimi, eğitim ve çıkarım akışı
├── README.md                        # Detaylı teorik ve mentorluk dokümantasyonu
├── veri_seti/
│   ├── data.yaml                    # YOLO veri seti konfigürasyonu
│   ├── images/ (train, val)         # Sentetik eğitim ve doğrulama görselleri
│   └── labels/ (train, val)         # YOLO formatı etiketler (.txt)
├── src/
│   ├── __init__.py
│   ├── sentetik_veri_ureteci.py     # Sentetik endüstriyel parça veri üreticisi
│   ├── map_hesaplayici.py           # Bağımsız mAP@0.5 ve COCO mAP metrik motoru
│   ├── yolo_yoneticisi.py           # Ultralytics YOLO eğitim ve çıkarım sarmalayıcısı
│   └── gorsellestirici.py           # 4 panelli teşhis panosu (Dashboard) çizici
├── testler/
│   ├── __init__.py
│   └── test_yolo.py                 # 5 adet kapsamlı birim test
└── ciktilar/
    └── yolo_egitim_ve_cikarim_paneli.png # 4 panelli teşhis panosu görseli
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

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
