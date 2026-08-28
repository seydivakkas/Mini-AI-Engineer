# Day 28: İleri Düzey Bölütleme & Mask R-CNN / SegFormer (Advanced Segmentation)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-EE4C2C.svg?style=flat-square)](https://pytorch.org/vision/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; bilgisayarla görmenin en ileri seviye piksel anlama görevleri olan **Örnek Tabanlı Bölütleme (Instance Segmentation)**, **Anlamsal Bölütleme (Semantic Segmentation)** ve ikisini birleştiren **Panoptik Bölütleme (Panoptic Segmentation)** paradigmalarını sıfırdan ele alır. **Mask R-CNN Mimarisi (RoIAlign & FCN Mask Head)**, **SegFormer Vision Transformer (Mix Transformer Encoder + All-MLP Decoder)** ve **Panoptik Kalite ($PQ = SQ \times RQ$)** metriklerini kapsayan 6 panelli endüstri standardı bir teşhis panosu (Diagnostic Dashboard) sunar.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Üç Temel Bölütleme Paradigması: Stuff vs Things

Görsel dünyayı oluşturan pikseller iki ana grupta toplanır:
1. **Sayılabilir Nesneler ("Things"):** Araçlar, insanlar, hayvanlar, engeller (Sınırlayıcı kutusu ve tekil kimliği olan nesneler).
2. **Biçimsiz Arka Plan ("Stuff"):** Gökyüzü, yol, deniz, çimen, bina duvarı (Sınırları net olmayan, dokusal sürekli alanlar).

| Bölütleme Türü | Kapsam | Nesne Örnek Ayrımı? | Tipik Çıktı |
|---|---|---|---|
| **Anlamsal (Semantic)** | Stuff + Things | ❌ Hayır (Tüm arabalar aynı etiket: 2) | $(H, W)$ sınıf haritası |
| **Örnek Tabanlı (Instance)** | Yalnızca Things | ✅ Evet (Araba #1, Araba #2 ayrı ayrı) | N adet ikili maske + Bounding Box |
| **Panoptik (Panoptic)** | Stuff + Things | ✅ Evet (Things tekil ID + Stuff genel ID) | $(H, W)$ Panoptik Harita ($c \times 1000 + \text{id}$) |

```
+-----------------------------------------------------------------------------------+
|                            BÖLÜTLEME PARADİGMALARI                                |
+-----------------------------------------------------------------------------------+
| 1. Semantik: [Gökyüzü: 0] [Yol: 1] [Araba: 2 (Tümü)]                              |
| 2. Instance: [Araba 1: Mask_A] [Araba 2: Mask_B] [Yaya 1: Mask_C]                 |
| 3. Panoptic: [Gökyüzü: 0] [Yol: 1000] [Araba 1: 2001] [Araba 2: 2002]             |
+-----------------------------------------------------------------------------------+
```

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **U-Net Mimarisi** | *U-Net Architecture* | Daralan kodlayıcı (encoder) ve genişleyen kod çözücü (decoder) kollarından oluşan 'U' biçimli bölütleme ağı. |
| **Atlama Bağlantıları** | *Skip Connections* | Kodlayıcıdaki yüksek çözünürlüklü uzamsal detayları doğrudan kod çözücü katmanlarına aktararak ince sınırların kaybolmasını önleyen bağlantılar. |
| **Focal Loss** | *Focal Loss* | Kolay sınıflandırılan arka plan piksellerinin gradyan ağırlığını düşürüp ($1-p_t)^\gamma$ zor ön plan piksellerine odaklanan kayıp fonksiyonu. |
| **Sınıf Dengesizliği** | *Extreme Class Imbalance* | Arka plan piksellerinin hedef nesne piksellerine kıyasla %95+ oranında baskın olması problemi. |

---

## 2. Mask R-CNN ve RoIAlign Devrimi (He et al., ICCV 2017)

Faster R-CNN mimarisinin üzerine her aday bölge için piksel düzeyinde ikili maske üreten bir **FCN Maske Dalı (Mask Branch)** eklenerek oluşturulmuştur.

```
                    Giriş Görseli -> ResNet + FPN Omurga -> RPN (Aday Bölgeler)
                                                                 │
                                                                 ▼
                                                    ┌──────────────────────────┐
                                                    │   RoIAlign (14x14 feat)  │
                                                    └────────────┬─────────────┘
                                                                 │
                                ┌────────────────────────────────┴────────────────────────────────┐
                                ▼                                                                 ▼
                ┌───────────────────────────────┐                                 ┌───────────────────────────────┐
                │     Sınıflandırma & Kutu      │                                 │       FCN Maske Başlığı       │
                │  - Sınıf: Lojit (K)           │                                 │  - 4 x Conv(256) -> Deconv    │
                │  - Kutu: Regresyon (4K)       │                                 │  - Çıktı: K x 28 x 28 Maske   │
                └───────────────────────────────┘                                 └───────────────────────────────┘
```

#### A. RoIPool vs RoIAlign (Hizalama Problemi)
- **RoIPool (Faster R-CNN):** $16 \times$ küçültülmüş özellik haritasında koordinatları tam sayıya yuvarlar ($\lfloor x / 16 \rfloor$). Bu kuantizasyon, kaba kutu tespiti için tolere edilebilirken; piksel düzeyindeki maske sınırlarında devasa kaymalara sebep olur.
- **RoIAlign:** Hiçbir kuantizasyon yapmaz. Her bir RoI bölmesi (bin) içinde 4 düzenli örnekleme noktası belirler ve **Çift Doğrusal İnterpolasyon (Bilinear Interpolation)** ile sürekli koordinatlardan hassas özellik çeker.

#### B. Çok Görevli Kayıp (Multi-Task Loss)
$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}$$

- $\mathcal{L}_{\text{mask}}$: Sınıflandırma kararından bağımsızlaştırılmıştır (**Decoupled**). Model $K$ adet bağımsız ikili maske üretir; kayıp yalnızca Ground Truth sınıfına karşılık gelen $k$. maske kanalında İkili Çapraz Entropi (Binary Cross-Entropy) olarak hesaplanır:
  $$\mathcal{L}_{\text{mask}} = - \frac{1}{M^2} \sum_{i, j} \Big( y_{ij} \log \sigma(z_{ij}^k) + (1 - y_{ij}) \log(1 - \sigma(z_{ij}^k)) \Big)$$

---

### 3. Vision Transformers ile Bölütleme: SegFormer (Xie et al., NeurIPS 2021)

Klasik CNN tabanlı bölütleme ağları sınırlı reseptif alan (Receptive Field) nedeniyle geniş bağlamı kaçırırken; **SegFormer** Vision Transformer mimarisini verimlilikle buluşturur:

1. **Mix Transformer (MiT) Hiyerarşik Encoder:**
   - 4 kademeli ($H/4, H/8, H/16, H/32$) çok ölçekli özellik haritaları üretir.
   - **Overlapping Patch Merging:** Yama kenarlarındaki uzamsal sürekliliği korumak için örtüşen evrişimler ($7 \times 7$ stride 4 ve $3 \times 3$ stride 2) kullanır.
2. **Mekansal İndirgemeli Dikkat (Spatial Reduction Attention - SRA):**
   - Self-Attention'ın $O(N^2)$ işlem yükünü, Key/Value dizilerini $R$ oranıyla alt-örnekleyerek $O(N^2 / R)$ seviyesine indirir.
3. **Konumsal Kodlamasız Yapı (Mix-FFN):**
   - Sabit konumsal gömmeler yerine $3 \times 3$ Derinlemesine Evrişim (Depthwise Conv) kullanarak modelin keyfi çözünürlüklerdeki test görsellerine interpolasyonsuz adapte olmasını sağlar.
4. **All-MLP Hafif Decoder:**
   - Karmaşık ve ağır deşifre blokları yerine, 4 kademenin çıktılarını basit doğrusal katmanlarla (Linear MLP) birleştirip $1 \times 1$ Conv ile sınıflandırır.

---

### 4. İleri Düzey Değerlendirme Metrikleri

#### A. Panoptik Kalite (Panoptic Quality - PQ)
$$PQ = \underbrace{\frac{\sum_{(p, g) \in TP} \text{IoU}(p, g)}{|TP|}}_{\text{Segmentation Quality (SQ)}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2} |FP| + \frac{1}{2} |FN|}}_{\text{Recognition Quality (RQ)}}$$

- **SQ (Bölütleme Kalitesi):** Doğru tespit edilen segmentlerin ortalama Maske IoU değeridir.
- **RQ (Tanıma Kalitesi):** Segmentlerin varlığını doğru tespit etme F1-skorudur.
- **Tekillik Garantisi:** Eşleşme kuralı $\text{IoU}(p, g) > 0.5$ olduğunda, hiçbir tahmin aynı anda iki gerçek nesneyle eşleşemez.

#### B. Örnek Tabanlı Maske Ortalama Hassasiyeti ($AP^{\text{mask}}$)
$$AP_{50}^{\text{mask}} \implies \text{IoU} \ge 0.50, \quad AP_{75}^{\text{mask}} \implies \text{IoU} \ge 0.75$$

---

## 📊 Deneysel Sonuçlar ve Metrik Tablosu

Sentetik Otonom Sürüş Sahnesi (Gökyüzü, Yol, Araçlar, Yayalar, Engeller) üzerinde elde edilen ileri düzey değerlendirme çıktıları:

```
================================================================================
>>> İleri Düzey Bölütleme & Panoptik Kalite Değerlendirmesi
================================================================================
[+] Sahne Çözünürlüğü         : 256x256x3
[+] Tespit Edilen Nesne Sayısı: 6 Örnek (Araç, Yaya, Engel)
[+] Multi-Task Loss Toplamı   : 1.4820 (L_cls: 0.612, L_box: 0.380, L_mask: 0.490)
[+] SegFormer Parametre Sayısı: 3,712,453
```

### Kapsamlı Metrik Çizelgesi

| Metrik Adı | Skor (%) | Açıklama |
|---|---|---|
| **Panoptik Kalite (PQ)** | **%88.42** | Genel sahne anlama başarısı ($SQ \times RQ$) |
| **Bölütleme Kalitesi (SQ)**| **%94.18** | Eşleşen nesnelerin sınır örtüşme hassasiyeti |
| **Tanıma Kalitesi (RQ)** | **%93.88** | Nesne örneklerinin eksiksiz bulunma F1 skoru |
| **Örnek Maske AP@50** | **%100.0** | $\text{IoU} \ge 0.50$ eşiğinde maske hassasiyeti |
| **Örnek Maske AP@75** | **%83.33** | $\text{IoU} \ge 0.75$ yüksek hassasiyetli maske AP |

---

## 🛠️ Dizin Yapısı

```
day-28-advanced-segmentation/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, torchvision, opencv, seaborn vb.
├── ana_akis.py                      # Uçtan uca sahne üretimi, Mask R-CNN, SegFormer ve PQ akışı
├── README.md                        # Detaylı teorik ve mentorluk dokümantasyonu
├── src/
│   ├── __init__.py
│   ├── bolutleme_turleri.py         # Semantic, Instance, Panoptic veri yapıları & dönüştürücü
│   ├── mask_rcnn_modulu.py          # RoIAlign, FCN Maske Başlığı ve Multi-Task Loss
│   ├── segformer_mimari.py          # SegFormer Vision Transformer (MiT Encoder + All-MLP Decoder)
│   ├── panoptik_ve_mask_metrikleri.py # Mask IoU, Instance AP ve Panoptic Quality (PQ=SQxRQ) motoru
│   ├── sentetik_sahne_ureteci.py    # Örtüşen nesneli zengin otonom sahne simülatörü
│   └── gorsellestirici.py           # 6 panelli ileri düzey teşhis panosu (Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_advanced_segmentation.py# 6 adet kapsamlı birim test
└── ciktilar/
    └── ileri_bolutleme_teshis_paneli.png # 6 panelli teşhis panosu görseli
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

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/bolutleme_turleri.py` içine panoptik harita üzerinde örtüşen nesnelerin alanına göre küçük nesneleri (ör. Yaya veya Engel) büyük nesnelerin (ör. Kamyon veya Arka Plan) üzerine öncelikli çizen akıllı bir `alan_duyarli_birlestir` metodu ekleyin.

**Çözüm:**
```python
def alan_duyarli_birlestir(semantik_harita, maskeler, siniflar):
    # Maske alanlarını küçükten büyüğe sırala (Küçük nesneler öne çizilsin)
    alanlar = [np.sum(m > 0.5) for m in maskeler]
    sirali_indeksler = np.argsort(alanlar)[::-1]  # Büyükler alta, küçükler üste
    return PanoptikDonusturucu.birlestir_panoptik(
        semantik_harita, maskeler, siniflar, ornek_skorlari=[-a for a in alanlar]
    )
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden nesne tespiti modellerinde (Faster R-CNN) kullanılan standart `RoIPool` katmanı, örnek tabanlı bölütleme (Mask R-CNN) görevinde kullanılamaz? RoIAlign bu kuantizasyon hatasını nasıl çözer ve SegFormer mimarisi neden konumsal gömme (Positional Embedding) kullanmadan farklı çözünürlüklerde çalışabilir?

> **Cevap:**
> 1. **RoIPool Kuantizasyon Hatası vs RoIAlign Çözümü:** 
>    - $16 \times$ alt-örneklenmiş bir özellik haritasında $x = 25$ pikselindeki bir sınır $\lfloor 25 / 16 \rfloor = 1$ olarak yuvarlanır. Orijinal görüntüye geri dönüldüğünde $1 \times 16 = 16$ pikseline denk gelir; bu da **$9$ piksellik bir konum kaymasına** yol açar. Kaba kutu tespiti için bu hata önemsiz olsa da, piksel düzeyindeki maske sınırlarını tamamen parçalar.
>    - **RoIAlign**, koordinatları hiçbir zaman yuvarlamaz. Sürekli kayan noktalı koordinatlar üzerinde 4 düzenli örnekleme noktası tanımlar ve **Çift Doğrusal İnterpolasyon (Bilinear Interpolation)** ile özellik haritasından pürüzsüz değerler alarak mükemmel piksel hizalaması sağlar.
> 2. **SegFormer ve Konumsal Kodlamasız Yapı:**
>    - Standart Vision Transformer'lar (ViT) sabit boyutlu öğrenilmiş 1B veya 2B konumsal gömmeler (Positional Embeddings) kullanır. Bu durum, eğitim çözünürlüğünden farklı boyutta bir test görseli geldiğinde konumsal gömmelerin interpolasyonla bozulmasına yol açar.
>    - **SegFormer**, konumsal kodlama yerine Feed-Forward bloğu içerisine **$3 \times 3$ Derinlemesine Evrişim (Depthwise Conv / Mix-FFN)** entegre eder. Evrişim çekirdekleri doğası gereği komşu pikseller arasındaki bağıl mesafeyi (Relative Position) kodladığı için model herhangi bir çözünürlükteki girdiyi sıfır bozulmayla işleyebilir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
