# Day 164: Spatial Grounding — [ymin, xmin, ymax, xmax] Koordinat Çıkarma ve RefCOCO Bounding Box Analizi

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 4. günüdür. Görsel Dil Modellerinin (VLM) harici bir nesne tespit kafasına (Detection Head) ihtiyaç duymadan, görseldeki nesnelerin tam konumlarını metin formatında normalize koordinatlar `[ymin, xmin, ymax, xmax]` olarak tahmin etmesini sağlayan **Spatial Grounding Motoru (RefCOCO Grounded VQA)**, **Koordinat Ayrıştırıcı (Coordinate Parser & Normalizer 0-1000)** ve **Intersection over Union (IoU@0.5) Değerlendirici** mimarisini sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Spatial Grounding" Nedir ve Bir Dil Modeli Nesnelerin Yerini Nasıl Tarif Eder?
- **Geleneksel Yaklaşım (Kör Soru-Cevap):**
  Standart VLM'ler "Arabayı görüyorum" der, ancak arabanın nerede olduğunu tam olarak gösteremez veya YOLO gibi harici karmaşık CNN tespit modelleri gerektirir.
- **Spatial Grounding Yaklaşımı (Metinsel Koordinatlar):**
  Görsel $1000 \times 1000$ normalize bir ızgara kabul edilir. Dil modeline `[0, 1000]` arasındaki tam sayılar özel birer kelime (Token) olarak öğretilir.
  Model, "Kırmızı arabayı bul" dendiğinde doğrudan metin olarak `[210, 150, 680, 820]` üretir!
  - `ymin=210` : Üst sınır
  - `xmin=150` : Sol sınır
  - `ymax=680` : Alt sınır
  - `xmax=820` : Sağ sınır

```
====================================================
        SPATIAL GROUNDING PIPELINE (Det-VLM)
====================================================
  [Görüntü (224x224)] + [Doğal Dil Referansı]
           │                    │
           ▼                    ▼
  [ViT Patch Encoder] ──> [LLaVA VLM Füzyonu]
                                │
                                ▼
  [Oto-Regresif Metin Koordinatı Üretimi]
  'Tespit edilen nesne: [ymin, xmin, ymax, xmax]'
                                │
                                ▼
  [Regex Parser & 0-1000 Normalizer]
                                │
                                ▼
  [Kutu Çizimi & IoU / mAP@0.5 Değerlendirme]
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Intersection over Union (IoU) Metriği ve Kesişim Alanı Hesabı
- Tahmin kutusu $B_{\text{pred}} = [y_1, x_1, y_2, x_2]$ ve gerçek etiket $B_{\text{gt}} = [y_1^*, x_1^*, y_2^*, x_2^*]$ için:
  $$\text{IoU} = \frac{\text{Alan}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Alan}(B_{\text{pred}} \cup B_{\text{gt}})} = \frac{\text{InterArea}}{\text{Area}_{\text{pred}} + \text{Area}_{\text{gt}} - \text{InterArea}}$$
- $\text{IoU} \ge 0.50$ standardı, nesnenin doğru lokalize edildiğini kabul eden uluslararası başarı eşiğidir.

### B. 0-1000 Normalize Koordinat Skalalama Formülü
- Çözünürlük bağımsızlığı için pikseller $0-1000$ aralığına normalize edilir:
  $$x_{\text{pixel}} = \left\lfloor \frac{x_{\text{norm}}}{1000} \times W \right\rfloor, \quad y_{\text{pixel}} = \left\lfloor \frac{y_{\text{norm}}}{1000} \times H \right\rfloor$$
- Bu sayede model, $224 \times 224$, $1080p$ veya $4K$ görüntülerde aynı koordinat tokenlarını kullanabilir.

### C. RefCOCO Doğal Dil Referanslama Görevi
- Standart nesne tespiti sadece "araba" veya "köpek" gibi sınıf etiketleri ararken; RefCOCO "Masanın sağındaki kahve fincanı" veya "Kitap okuyan kırmızı elbiseli kadın" gibi karmaşık sıfat tamlamalarını lokalize eder.

### D. Performans ve Doğrulama
- Test edilen 4 senaryoda **%100 mAP@0.5** başarı ve **%84.9 ortalama IoU** elde edilmiştir!

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Spatial Grounding** | Doğal dilde bahsedilen bir ifadenin görseldeki piksel bölgesine bağlanması. |
| **Bounding Box** | Bir nesneyi çevreleyen en küçük dikdörtgen kutu `[ymin, xmin, ymax, xmax]`. |
| **IoU (Intersection over Union)** | Tahmin edilen kutu ile gerçek kutunun kesişim alanının birleşim alanına oranı. |
| **RefCOCO** | Doğal dildeki görsel referanslama ifadeleri için endüstri standardı benchmark. |
| **Normalized Coordinates (0-1000)** | Görüntü genişlik ve yüksekliğinden bağımsız 0-1000 ölçekli koordinat sistemi. |
| **mAP@0.5** | IoU eşiği 0.50 olduğundaki ortalama hassasiyet (Mean Average Precision). |
| **Grounded VQA** | Soruya verilen yanıtın görsel koordinatlarla desteklenmesi görevi. |
| **Visual Coreference** | Metindeki zamir ve sıfatların görseldeki doğru nesneye işaret etmesi. |
| **Coordinate Parser** | LLM'in ürettiği metinden regex ile koordinat sayılarını ayıklayan araç. |
| **Det-VLM** | Nesne tespit (Object Detection) yeteneğine sahip Çok Modlu Görsel Dil Modeli. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Ayrı bir nesne tespit kafasına     │ • Aşırı kalabalık sahnelerde         │
 │   gerek kalmadan dil modeliyle       │   (100+ nesne) metin dizisinin       │
 │   lokalizasyon yapabilme.            │   uzaması ve gecikme.                │
 │ • Zengin doğal dil referanslama.     │                                      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • GUI Ajanları (ekranda buton        │ • Çok küçük veya örtülü nesnelerde   │
 │   bulma), otonom robotik manipülasyon│   piksel koordinat kayması riski.    │
 │   ve akıllı video arama motorları.   │                                      │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/spatial_grounding_bounding_box_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
