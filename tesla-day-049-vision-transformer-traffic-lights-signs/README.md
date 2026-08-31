# 🚗 Tesla FSD Otonom Sürüş | Gün 49: Vision Transformer (ViT) ile Yüksek Hızlı Trafik İşareti, Işık ve Nesne Tespiti

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![ViT](https://img.shields.io/badge/Transformer-Vision%20Transformer%20(ViT)-red.svg?style=flat-square)](https://www.tesla.com/)
[![SelfAttention](https://img.shields.io/badge/Attention-Multi--Head%20Self--Attention-blue.svg?style=flat-square)](https://www.sae.org/)
[![OCR](https://img.shields.io/badge/Detection-Traffic%20Lights%20%26%20Signs-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"49. günümüze hoş geldin stajyer!  
> Geleneksel CNN (Evrişimli Sinir Ağları) lokal çekirdeklerle (Kernel $3 \times 3$) çalıştığı için 150 metre uzaktaki küçük bir trafik ışığını veya hız levhasını arka plandaki binalardan, parlak neon tabelalardan ve ağaç yapraklarından ayırt etmekte zorlanırdı.  
> Tesla FSD algı hattında kritik trafik elemanlarını sınıflandırmak için **Vision Transformer (ViT)** mimarisini kullanır:  
> 1. **Yama Gömme (Patch Embedding):** Kamera görüntüsü $16 \times 16$ veya $8 \times 8$ piksellik küçük yamalara bölünerek 1D belirteç (token) dizisi haline getirilir.  
> 2. **Global Öz-Dikkat (Multi-Head Self-Attention):** Tüm yamalar aynı anda birbirleriyle iletişim kurar. Trafik ışığının durumu, direğin altındaki şerit ve yol çizgileriyle global bağlamda ilişkilendirilir.  
> 3. **Çoklu Görev Başlıkları:** Işığın rengi (Kırmızı, Sarı, Yeşil, Sol Ok), flaş durumu, dijital geri sayım sayacının süresi ($8.5\text{ sn}$) ve hız levhası OCR'ı tek bir çıkarımda eşzamanlı çözülür.  
> Bugün Tesla'nın kavşak gözünü kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Yama Gömme ve Pozisyonel Kodlama

$$N = \frac{H \cdot W}{P^2}, \quad \mathbf{z}_0 = \left[ \mathbf{x}_{\text{class}}; \, \mathbf{x}_p^1 \mathbf{E}; \, \dots; \, \mathbf{x}_p^N \mathbf{E} \right] + \mathbf{E}_{\text{pos}}$$

### 2. Ölçekli Noktasal Çarpım Çok Başlıklı Öz-Dikkat (MHSA)

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q \cdot K^T}{\sqrt{d_k}} \right) \cdot V$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O$$

### 3. Çoklu Görev Kayıp Fonksiyonu

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}}(\hat{y}_{\text{light}}, y_{\text{light}}) + \lambda_1 \mathcal{L}_{\text{Huber}}(\hat{t}_{\text{count}}, t_{\text{count}}) + \lambda_2 \mathcal{L}_{\text{CrossEntropy}}(\hat{y}_{\text{sign}}, y_{\text{sign}})$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Uzak mesafedeki küçük trafik ışığı ve levhalarının yarattığı yerel çözünürlük kaybını küresel öz-dikkat (Global Self-Attention) ile gidermek ve bağlamsal doğruluğu artırmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Yanlış Işık Eşleştirmesi:** Yan şeridin veya ters yönün trafik ışığına odaklanma hatasını (False Positive) şerit-ışık geometrik dikkatiyle engelledi.
- **Geri Sayım Sayacı Regresyonu:** Işığın üzerindeki dijital saniyeleri doğrudan okuyarak aracın kavşağa ne zaman varacağını hesaplayan hız optimizasyonunu mümkün kıldı.
- **Düşük Işık ve Parıltı:** Güneş ışığı parlamasında (Sun Glare) CNN'lerin kör olduğu noktalarda semantik global bağlamla ışığı tanıdı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Karesel Karmaşıklık:** Standart öz-dikkat yama sayısının karesi $\mathcal{O}(N^2)$ ile ölçeklenir (Pencere tabanlı Swin Transformer ile optimize edilir).
- **Bellek Bant Genişliği:** NPU üzerinde ağırlık matrisi erişimi yüksek bellek bant genişliği gerektirir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Saf CNN (ResNet/YOLO):** Uzak ışıklarda bağlam kaybı yaşar ve küçük nesnelerde güven skoru düşüktür.
- **Optik Karakter Tanıma (OCR Tesseract):** Çok yavaş ve hareketli araç kameralarında kullanılamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Vision Transformer (ViT)**| Görüntüleri piksel yamaları halinde transformer bloklarına besleyen dikkat tabanlı derin öğrenme mimarisi. |
| **Patch Embedding** | 2D görüntü bloğunu lineer projeksiyon katmanıyla 1D vektör temsiline dönüştürme işlemi. |
| **Multi-Head Self-Attention**| Farklı temsil alt uzaylarında görüntü yamalarının birbirine olan dikkat ağırlıklarını hesaplama mekanizması. |
| **Countdown Regression** | Trafik ışığı üzerindeki dijital geri sayım süresini saniye cinsinden tahmin eden regresyon başlığı. |
| **Flashing Yellow** | Dikkatli geçilmesi gerektiğini belirten sarı ikaz ışığı modu. |
| **Speed Limit OCR** | Levhadaki hız sınırını (örn. 50, 70, 90 km/h) doğrudan okuyan optik karakter tanıma modeli. |
| **Positional Encoding** | Görüntü yamalarının 2D uzaydaki konum sırasını modele bildiren konumsal vektörler. |
| **Classification Token (CLS)**| Tüm görselin genel sınıf temsilini toplayan öğrenilebilir özel token. |
| **Attention Map** | Modelin görsel üzerinde hangi piksellere/yamalara yoğunlaştığını gösteren 2D dikkat ısı haritası. |
| **NPU HW3/HW4 Core** | ViT matris çarpımlarını mikrosaniyeler seviyesinde koşturan Tesla NPU çekirdeği. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Küresel bağlam (Global Context) ile yüksek doğruluk | • Yama sayısı arttıkça O(N^2) bellek artışı           |
| • 150m uzaklıktaki küçük ışıklarda %98+ tespit        | • Düşük boyutlu NPU'larda FP16/INT8 kuantizasyon      |
| • Geri sayım ve levha OCR'ının tek ağda birleşmesi    |   optimizasyonu zorunluluğu                           |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Swin/FlashAttention ile RTOS süresini 10 µs         | • Şiddetli kar ve çamurda levha yüzeyinin tamamen     |
|   seviyesine indirme potansiyeli                      |   kapanması                                           |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla ViT Trafik Algılayıcı Mimarisi

```
[ Ön Kamera Karesi (256x256) ]
              |
              v
    [ Yama Gömme (Patch Embedding: 16x16 Yamalar) ]
              |
              v
    [ Çok Başlıklı Öz-Dikkat (MHSA Transformer Blokları) ]
              |
      +-------+-------+
      |               |
      v               v
[ Trafik Işığı Başlığı ]     [ Trafik Levhası Başlığı ]
- Kırmızı (%96 Güven)        - Hız Sınırı (70 km/h)
- 8.5s Geri Sayım            - DUR / YOL VER
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana ViT simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
