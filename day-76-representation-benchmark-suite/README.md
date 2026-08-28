# Day 76: Temsil Kalitesi Değerlendirmesi — Linear Probing ve k-NN Sınıflandırma Protokolü

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Benchmark: Passed](https://img.shields.io/badge/Representation_Benchmark-100%25_LinearProbe-brightgreen.svg?style=flat-square)](#deneysel-benchmark-sonuçları)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_benchmark_suite.py)

Self-Supervised Learning (SimCLR, DINO, MAE) ve Metrik Öğrenimi (SupCon, Triplet Margin) ile eğitilmiş bir omurga modelinin (Backbone) öğrendiği öznitelik kalitesini, omurga ağırlıklarını dondurarak (**Frozen Backbone**) nesnel olarak test eden **Linear Probing**, **Sıcaklık Ölçekli k-NN** ve **Few-Shot Veri Verimliliği** Değerlendirme Paketi.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Denetimsiz (Self-Supervised) veya metrik öğrenimle eğitilen bir modelin başarısı geleneksel Cross-Entropy gibi doğrudan bir sınıflandırma kaybıyla ölçülemez. Modelin içgörü yeteneğini ölçmek için şu iki temel bilimsel gerekçeyle bu değerlendirme paketi kullanılır:

1. **Omurga Değişmezliği (Backbone Freezing & Representation Purity):**
   Tüm omurga parametreleri $f(\cdot)$ dondurulur (`requires_grad = False`). Amaç, sınıflandırma yeteneğinin omurganın sonradan eğitilmesinden mi yoksa öğrendiği temsil uzayının doğasından mı kaynaklandığını kesin olarak ayrıştırmaktır.
2. **Doğrusal Ayrışabilirlik Hipotezi (Linear Separability Hypothesis):**
   İyi bir temsil öğrenici, karmaşık görsel girdileri öyle bir uzaya taşır ki sınıflar hiper-düzlemlerle (Hyperplanes) kolayca ayrılabilir hale gelir. Tek katmanlı bir lineer model ($W \cdot h + b$) bu uzayda yüksek başarı gösteriyorsa, temsil kalitesi kusursuzdur.
3. **Sıfır Eğitimli Manifold Doğrulaması (Zero-Training k-NN):**
   Hiçbir optimizasyon veya öğrenme oranı araması yapmadan, yalnızca $L_2$ normalize kosinüs mesafesiyle komşuluk oylaması yaparak modelin yerel manifold geometrisini doğrudan doğrular.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Aşırı Uydurma (Overfitting) Yanılsamasını Önler:**
  Uçtan uca (End-to-End) ince ayar (Fine-Tuning) yapıldığında, güçlü bir sınıflandırma kafası kötü bir omurgayı bile telafi edebilir ve yanıltıcı yüksek skorlar verebilir. Linear Probing omurgayı dondurarak gerçek öznitelik kalitesini çıplak bırakır.
- **Hiperparametre Bağımlılığını Sıfırlar (k-NN Protokolü):**
  Linear probing bile öğrenme oranı, ağırlık cezası (weight decay) ve epoch sayısına bağlıdır. Sıcaklık ağırlıklı k-NN, 0 epoch ile tamamen deterministik ve saf bir temsil kalitesi metrik puanı üretir.
- **Az Verili Senaryolarda (Few-Shot) Dayanıklılığı Test Eder:**
  Veri kümesinin sadece %1'i veya %10'u etiketli olduğunda modelin ne kadar transfer öğrenebildiğini ölçerek etiketleme maliyeti darboğazını çözer.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Doğrusal Olmayan (Non-Linear) Yapıları Cezalandırabilir:**
  Temsil uzayı sınıfları çok iyi ayırmış ancak spiral veya küresel (non-linear) bir manifold üzerine yerleştirmişse, Linear Probe düşük puan verebilir.
- **k-NN Büyük Veri Kümelerinde Bellek Yoğundur:**
  Doğrulama kümesindeki her sorgu için eğitim kümesinin tamamıyla ($N_{\text{val}} \times N_{\text{train}}$) mesafe matrisi hesaplamak GPU VRAM gerektirir.
- **Boyut Çöküşünü (Dimensional Collapse) Tek Başına Tespit Edemeyebilir:**
  Tüm sınıflar tek bir doğru üzerinde sıralanırsa Linear Probe yüksek çıkabilir; bu yüzden paketimiz ek olarak **SVD Efektif Boyut** ve **İzotropi İndeksi** metriklerini hesaplar.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Değerlendirme Yaklaşımı | Eğitim Gereksinimi | Ölçtüğü Nitelik | Hesaplama Maliyeti | Güvenilirlik |
|---|---|---|---|---|
| **Linear Probing (Bizim Yaklaşımımız)** | Düşük (Tek Katman FC) | Doğrusal Ayrışabilirlik | Hızlı (1-2 dakika) | ⭐⭐⭐⭐⭐ (Altın Standart) |
| **Non-Parametric k-NN (Bizim Yaklaşımımız)** | **Sıfır (0 Epoch)** | Yerel Manifold Topolojisi | Anlık (Saniyeler) | ⭐⭐⭐⭐⭐ (Objektif) |
| **End-to-End Fine-Tuning** | Yüksek (Tüm Ağ) | Görev Adaptasyonu | Çok Yavaş | ⭐⭐⭐ (Omurga kalitesini gizler) |
| **Attentive Probing (Transformer)** | Orta (Cross-Attention) | Karmaşık Temsil İlişkileri | Orta | ⭐⭐⭐⭐ |
| **Zero-Shot Classifier (CLIP)** | Sıfır (Metin Projeksiyonu) | Çok Modlu Semantik Hizalama | Çok Hızlı | ⭐⭐⭐⭐⭐ (Sadece VLM'ler) |

---

## 📐 Matematiksel Formülasyon

### 1. Dondurulmuş Temsil Çıkarımı
Girdi görseli $x \in \mathbb{R}^{C \times H \times W}$ omurga ağı $f_\theta$ tarafından temsil vektörüne dönüştürülür ve birim hiperküreye normalize edilir:

$$h = f_\theta(x) \in \mathbb{R}^D, \quad e = \frac{h}{\|h\|_2} \in \mathbb{S}^{D-1}$$

### 2. Linear Probing Sınıflandırıcısı
Sadece $W \in \mathbb{R}^{C \times D}$ ve $b \in \mathbb{R}^C$ eğitilir:

$$\hat{y} = \text{Softmax}(W e + b), \quad \mathcal{L}_{\text{CE}} = - \sum_{c=1}^C y_c \log \hat{y}_c$$

### 3. Sıcaklık Ölçekli Ağırlıklı k-NN (DINO / MoCo Protokolü)
Doğrulama sorgusu $e_q$ ile eğitim gömülmeleri $e_i$ arasındaki kosinüs benzerliği $s_{q, i} = e_q^\top e_i$ hesaplanır. En yakın $k$ komşu için sıcaklık ağırlığı $\tau$:

$$w_{q, i} = \exp\left( \frac{s_{q, i}}{\tau} \right)$$

$$P(y = c \mid e_q) = \frac{\sum_{i \in \text{TopK}, y_i = c} w_{q, i}}{\sum_{i \in \text{TopK}} w_{q, i}}$$

$$\hat{y}_q = \arg\max_{c} P(y = c \mid e_q)$$

### 4. Temsil Geometrisi: SVD Efektif Boyut & İzotropi
Temsil matrisinin $X \in \mathbb{R}^{N \times D}$ tekil değerleri (Singular Values) $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_D$ hesaplanır. Normalize varyans $p_k = \frac{\sigma_k^2}{\sum \sigma_j^2}$ üzerinden Shannon entropisi:

$$H(X) = - \sum_{k=1}^D p_k \ln p_k \implies \text{Efektif Boyut} = \exp(H(X))$$

$$\text{İzotropi İndeksi} = \frac{\sigma_{\min}}{\sigma_{\max}}$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Linear Probing** | *Linear Probing* | Dondurulmuş omurga üzerine tek katmanlı lineer sınıflandırıcı eğiterek temsil kalitesini ölçme protokolü. |
| **Frozen Backbone** | *Frozen Feature Extractor* | Gradyan hesaplaması kapatılmış (`requires_grad = False`) sabit öznitelik çıkarıcı model. |
| **Weighted k-NN** | *Temperature-Weighted k-NN* | En yakın komşuların oylarını kosinüs benzerliği ve sıcaklık sabiti ($\tau$) ile ağırlıklandıran sınıflandırıcı. |
| **Few-Shot Probing** | *Few-Shot Linear Probing* | Eğitim etiketlerinin yalnızca %1'i veya %10'u kullanılarak yapılan veri verimliliği testi. |
| **Representation Collapse** | *Temsili Çöküş* | Tüm girdi örneklerinin temsil uzayında tek bir noktaya veya tek bir boyuta sıkışması hatası. |
| **SVD Effective Rank** | *Singular Value Effective Dimension* | Temsil uzayının gerçekten kaç bağımsız boyutu aktif olarak kullandığını ölçen entropi tabanlı metrik. |
| **Isotropy Index** | *İzotropi İndeksi* | Temsillerin uzayda her yöne eşit dağılıp dağılmadığını (en küçük tekil değer / en büyük tekil değer) ölçen oran. |
| **Silhouette Score** | *Silhouette Manifold Skoru* | Sınıf içi sıkılığı ($a$) ve en yakın komşu sınıf mesafesini ($b$) $\frac{b - a}{\max(a, b)}$ ile $[-1, +1]$ aralığında ölçen küme ayrışma metriği. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Foundation ve Self-Supervised modellerin (DINO, SimCLR, MAE) resmi literatür standardı; Hızlı, objektif ve tekrarlanabilir kalite raporu. |
| **Weaknesses (Zayıf Yönler)** | Doğrusal olmayan karmaşık ilişkileri göremeyebilir; k-NN doğrulama seti çok büyüdüğünde $O(N^2)$ bellek ihtiyacı. |
| **Opportunities (Fırsatlar)** | MLOps CI/CD süreçlerinde model sürümleme öncesi otomatik kalite kapısı (Quality Gate); Foundation model seçim kıyaslamaları. |
| **Threats (Tehditler)** | Temsil uzayında yüksek performans gösteren bir omurganın hedef donanımda aşırı bellek tüketmesi. |

---

## 💻 Deneysel Benchmark Sonuçları

`ana_akis.py` çalıştırılarak Rastgele Başlatılmış Omurga (Random Baseline) ile Kontrastif Eğitilmiş Omurga (Trained Model) kıyaslanmıştır:

```text
================================================================================
      Değerlendirme Protokolü       |   Rastgele Model   |  Eğitilmiş Model  
================================================================================
Linear Probe (%100 Etiket)          | %     100.00      | %     100.00     
Linear Probe Few-Shot (%10 Etiket)  | %     34.00       | %     100.00     
Linear Probe Few-Shot (%2 Etiket)   | %     15.33       | %      82.67     
Non-Parametric k-NN (k=1)           | %     100.00      | %     100.00     
Non-Parametric k-NN (k=5)           | %     100.00      | %     100.00     
Non-Parametric k-NN (k=20)          | %     100.00      | %     100.00     
--------------------------------------------------------------------------------
Silhouette Küme Skoru (-1 ila +1)   |       0.989        |       0.995       
İzotropi İndeksi (min/max s.v.)     |       0.000        |       0.000       
Efektif Boyut (SVD Entropisi)       |        2.5         |        4.9        
Sınıf Ayrışma Marjini               |       0.080        |       1.141       
================================================================================
```

### 🔑 Temel Çıkarımlar
1. **Few-Shot Üstünlüğü:** Rastgele omurga %2 etiketle yalnızca %15.33 (şans faktörü) doğruluk verirken, eğitilmiş temsil omurgası **%82.67 doğruluk** üretmiştir.
2. **Sınıf Ayrışma Marjini:** Eğitilmiş modelde sınıflar arası kosinüs marjini $+0.080$'dan **$+1.141$ seviyesine yükselmiştir**.
3. **Efektif Boyut:** Modelin temsil çeşitliliği ve aktif boyut kullanımı $2.5$'tan **$4.9$'a çıkmıştır**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/temsil_benchmark_paneli.png`](ciktilar/temsil_benchmark_paneli.png) konumundadır:

1. **Değerlendirme Protokolleri Metodolojisi:** Linear Probe, k-NN ve Few-Shot mimari akış şeması.
2. **Linear Probe vs k-NN Kıyaslaması:** Eğitilmiş omurga ile rastgele omurganın Top-1 doğruluk sütunları.
3. **Few-Shot Veri Verimliliği Eğrisi:** %1'den %100'e kadar etiket oranı arttıkça doğruluk değişimi.
4. **Temsil Uzayı Manifold Ayrışması (PCA):** 6 sınıfın 2D hiperdüzlemdeki doğrusal ayrışma sınırları.
5. **Manifold Kalite İndeksleri:** Silhouette, İzotropi, Efektif Boyut ve Kosinüs Marjini metrik çubukları.
6. **SWOT Karar Matrisi:** Mühendislik karar kriterleri ve endüstriyel kullanım kılavuzu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Büyük ölçekli temsil değerlendirmelerinde ($N_{\text{val}} > 100.000$) GPU belleğini tüketmemek (Out-of-Memory) için k-NN sorgularını minibatch'ler halinde çözen ve Top-$K$ oylarını biriktiren bellek-verimli (Chunked/Batched) bir k-NN değerlendirici yazınız.

```python
import torch
import torch.nn.functional as F

class BellekVerimliKNNDegerlendirici:
    """Büyük veri kümeleri için belleği koruyan Chunked k-NN Sınıflandırıcısı."""
    def __init__(self, sicaklik: float = 0.07, chunk_boyutu: int = 1024):
        self.sicaklik = sicaklik
        self.chunk_boyutu = chunk_boyutu

    @torch.no_grad()
    def degerlendir_chunked(self, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, k: int = 5, sinif_sayisi: int = 10) -> float:
        x_train = F.normalize(x_train, p=2, dim=1)
        x_val = F.normalize(x_val, p=2, dim=1)
        dogru = 0
        N_val = x_val.size(0)

        for i in range(0, N_val, self.chunk_boyutu):
            val_chunk = x_val[i : i + self.chunk_boyutu]
            y_chunk = y_val[i : i + self.chunk_boyutu]
            
            # (Chunk_size, N_train) benzerlik matrisi
            sims = torch.matmul(val_chunk, x_train.T)
            topk_sims, topk_idx = torch.topk(sims, k=k, dim=1)
            topk_labels = y_train[topk_idx]
            weights = torch.exp(topk_sims / self.sicaklik)

            oylar = torch.zeros(val_chunk.size(0), sinif_sayisi, device=x_val.device)
            for c in range(sinif_sayisi):
                maske = (topk_labels == c).float()
                oylar[:, c] = (maske * weights).sum(dim=1)

            tahminler = torch.argmax(oylar, dim=1)
            dogru += (tahminler == y_chunk).sum().item()

        return (dogru / max(1, N_val)) * 100.0
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Bir Self-Supervised modelin başarısını ölçerken neden doğrudan tüm ağı ince ayar (Fine-Tuning) yapmak yerine Linear Probing ve k-NN protokolleri tercih edilir?

> **Mentor Cevabı:**
> 1. **Temsil Kalitesini İzole Etme (Isolation of Representation):** Uçtan uca fine-tuning yapıldığında omurga ağının tüm ağırlıkları değişir. Bu durumda modelin başarısı, önceden öğrendiği özniteliklerden mi yoksa fine-tuning sırasında optimize edilen gradyanlardan mı kaynaklandı bilinemez. Linear Probing omurgayı dondurarak temsilin saf ayrışma gücünü ölçer.
> 2. **Sıfır Aşırı Uydurma Riski:** Küçük veri setlerinde fine-tuning tüm ağı kolayca overfit edebilir. Dondurulmuş omurga üzerinde sadece tek bir lineer katman eğitildiğinde parametre sayısı çok az olduğundan aşırı uydurma riski minimumdur.
> 3. **k-NN ile Hiperparametre Tarafsızlığı:** Linear Probing'de dahi optimizer seçimi, learning rate ve batch size sonuçları etkileyebilir. Sıcaklık ölçekli k-NN hiçbir eğitim aşaması içermediği için donanım ve kütüphanelerden bağımsız, %100 objektif bir manifold benchmark'ıdır.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
