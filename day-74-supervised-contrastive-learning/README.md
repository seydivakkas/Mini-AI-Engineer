# Day 74: Etiketli Veride Supervised Contrastive (SupCon) Kaybı ile Sınıf Ayrıştırma

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passed-brightgreen?style=flat-square)

## 🎯 Proje Özeti & Mühendislik Hedefi

Geleneksel derin öğrenmede sınıflandırma görevleri onlarca yıldır standart **Cross-Entropy (Çapraz Entropi)** kaybı ile eğitilmektedir. Ancak Cross-Entropy; etiket gürültüsüne (label noise) karşı aşırı hassastır, temsil uzayında geometrik bir marjin oluşturamaz ve dağılım dışı (OOD) örneklerde yanıltıcı yüksek güven skoru (overconfidence) üretir. 

Öte yandan, Day 73'te uyguladığımız **SimCLR (Self-Supervised)**, etiket kullanmadığı için aynı sınıfa ait iki farklı görüntüyü (örneğin iki farklı köpek resmini) minibatch içinde negatif çift varsayıp uzayda birbirinden iter (**False Negative Çıkmazı**).

**Supervised Contrastive Learning (SupCon - Khosla et al., NeurIPS 2020)**; sınıf etiketlerini kontrastif öğrenme paradigmasıyla harmanlayarak, minibatch içindeki **aynı sınıfa ait tüm örnekleri ve bunların tüm artırılmış görünümlerini pozitif kümede ($\mathcal{P}(i)$)** toplar. Böylece temsil uzayında sınıflar arası devasa bir geometrik ayrışma ve aşırı dayanıklı (robust) bir öznitelik uzayı inşa eder.

---

## 🔬 Teorik & Matematiksel Derinlik

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CROSS-ENTROPY vs SIMCLR vs SUPCON GEOMETRİSİ                                 │
│                                                                                                           │
│   1. CROSS-ENTROPY                  2. SIMCLR (Self-Supervised)         3. SUPCON (Supervised Contrastive)│
│   ────────────────                  ───────────────────────────         ──────────────────────────────────│
│   Tekil örnekleri bağımsız          Aynı kaynaktan gelenleri çeker;     AYNI SINIFTAN GELEN TÜM           │
│   lineer hiper-düzlemlerle böler.   AYNI SINIFTAN FARKLI GÖRSELLERİ     ÖRNEKLERİ VE GÖRÜNÜMLERİ ÇEKER;   │
│   Geometrik kümeleme yapmaz.        DA NEGATİF SAYIP İTER (False Neg).  FARKLI SINIFLARI UZAYDA İTER!     │
│                                                                                                           │
│         ▲                                   ▲                                     ▲                       │
│      o  │  x                             o1---o1'                              o1---o2                    │
│    o    │    x                                                                  \   /                     │
│   ──────┼──────                          x1---x1'                                 o1'                     │
│      o  │  x                                                                                              │
│         │                                o2---o2' (o1 ve o2 itilir!)           x1---x2                    │
│                                                                                                           │
│   [Doğrusal Karar Sınırı]           [Örnek Başına Bireysel Çekim]       [Mükemmel Sınıf Kümeleri & Marjin]│
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. Supervised Contrastive (SupCon) Kayıp Formülasyonu
$N$ adet örnekten oluşan bir minibatch'ten $2N$ adet artırılmış görünüm ($I \equiv \{1, \dots, 2N\}$) üretilir. Her $i \in I$ için pozitif örnekler kümesi:

$$\mathcal{P}(i) \equiv \big\{ p \in A(i) : y_p = y_i \big\}, \quad \text{burada } A(i) \equiv I \setminus \{i\}$$

SupCon kayıp fonksiyonu:

$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|\mathcal{P}(i)|} \sum_{p \in \mathcal{P}(i)} \log \frac{\exp\left(\frac{z_i \cdot z_p}{\tau}\right)}{\sum_{a \in A(i)} \exp\left(\frac{z_i \cdot z_a}{\tau}\right)}$$

- $|\mathcal{P}(i)|$: $i$ örneği ile aynı sınıfa ait olan (diğer görünümler ve diğer örnekler dahil) toplam pozitif eleman sayısı.
- $z_i = \frac{g(f(x_i))}{\|g(f(x_i))\|_2}$: L2 normalize edilmiş projeksiyon vektörü.
- $\tau$: Sıcaklık parametresi (**Temperature**).

---

### 2. İki Aşamalı Eğitim Protokolü (Two-Stage Pipeline)

1. **Aşama 1 (Stage 1 - SupCon Temsil Ön Eğitimi):**
   - Temel Kodlayıcı $f(\cdot)$ ve Non-lineer Projeksiyon Kafası $g(\cdot)$, $\mathcal{L}_{\text{SupCon}}$ kaybı ile uçtan uca eğitilir.
   - Sınıf içi temsiller birim hiperküre üzerinde sıkıca kümelenir, sınıflar arası açısal mesafe maksimize edilir.
2. **Aşama 2 (Stage 2 - Linear Probing):**
   - Projeksiyon kafası $g(\cdot)$ atılır.
   - Kodlayıcı $f(\cdot)$ dondurulur (**Freeze**).
   - $h = f(x)$ temsilleri üzerine tek bir Doğrusal Katman ($c(h) = W h + b$) eklenir ve standart Cross-Entropy ile yalnızca birkaç epoch eğitilir.

---

## 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- **Geometrik Temsil Kalitesi:** Cross-Entropy yalnızca doğrusal sınıf sınırları bulmaya çalışırken, SupCon temsil uzayının kendisini metrik olarak yapılandırır. Sınıf içi varyansı minimize ederken sınıflar arası ayrışma marjinini maksimize eder.
- **Doğal Düzenlileştirme (Regularization):** Stokastik çoklu görünümler sayesinde model aşırı öğrenmeye (overfitting) karşı bağışıklık kazanır.

### 2. Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- **False Negative Çıkmazı:** SimCLR'ın aynı sınıfa ait farklı görsellere uyguladığı yanlış itme kuvvetini ortadan kaldırır.
- **Etiket Gürültüsü (Label Noise) Dayanıklılığı:** Cross-Entropy yanlış etiketlenmiş bir örnekte aşırı gradyan patlaması yaşarken, SupCon log-oran normalizasyonu ve $|\mathcal{P}(i)|$ paydası sayesinde gürültüyü doğal olarak sönümler.
- **Hiperparametre Kararlılığı:** Öğrenme oranı ve ağırlık cezası değişimlerine karşı Cross-Entropy'ye oranla çok daha kararlıdır.

### 3. Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- **İki Aşamalı Süreç (Two-Stage Overhead):** Doğrudan tek aşamada sınıflandırıcı üretemez. Önce temsil eğitimi (Stage 1), ardından sınıflandırıcı eğitimi (Stage 2) gerekir.
- **Batch İçi Sınıf Çeşitliliği Zorunluluğu:** Eğer minibatch içinde bir sınıftan yalnızca 1 örnek düşerse ($|\mathcal{P}(i)| = 0$), o sınıf kontrastif çift oluşturamaz ve gradyan üretemez. Bu nedenle dengeli/büyük batch gerektirir.

### 4. Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- **Cross-Entropy:** Tek aşamalıdır ve hızlıdır, ancak temsil geometrisi zayıftır ve gürültüye duyarlıdır.
- **ArcFace / CosFace (Additive Angular Margin):** Ağırlık matrisi merkezleri ile açısal marjin koyar; yüz tanımada çok popülerdir ancak çoklu veri artırma çiftlerinden SupCon kadar faydalanamaz.
- **Triplet Loss:** Örnekleri üçlü (Anchor, Positive, Negative) seçer; madencilik (mining) algoritmaları karmaşık ve hesaplama maliyeti yüksektir.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **SupCon** | *Supervised Contrastive Learning* | Sınıf etiketlerini kontrastif InfoNCE kaybına entegre ederek aynı sınıftan gelen tüm örnekleri pozitif kümede toplayan temsil öğrenimi yöntemi. |
| **False Negative Dilemma** | *False Negative Issue in SSL* | Kendi kendine denetimli modellerin (SimCLR), aynı sınıfa ait farklı örnekleri negatif sanarak birbirinden uzaklaştırması problemi. |
| **Positive Set ($\mathcal{P}(i)$)** | *Intra-class Positive Set* | Minibatch içinde $i$ örneği ile aynı sınıf etiketine sahip olan tüm diğer görünümlerin ve örneklerin indeks kümesi. |
| **Two-Stage Training** | *Two-Stage Training Protocol* | Önce omurga ağın kontrastif kayıpla eğitildiği, ardından dondurularak üzerine sınıflandırıcı eğitildiği iki fazlı mimari. |
| **Linear Probing** | *Frozen Backbone Linear Probing* | Eğitilmiş öznitelik çıkarıcının ağırlıklarını dondurup, üzerine tek bir lineer katman eğiterek temsil kalitesini ölçme yöntemi. |
| **Intra-Class Cosine Similarity** | *Intra-Class Cosine Similarity* | Aynı sınıfa ait örneklerin temsil vektörleri arasındaki kosinüs benzerliği ($+1.0$'a yaklaşması hedeflenir). |
| **Inter-Class Cosine Similarity** | *Inter-Class Cosine Similarity* | Farklı sınıflara ait örneklerin temsil vektörleri arasındaki kosinüs benzerliği ($0.0$ veya negatif olması hedeflenir). |
| **Separation Margin** | *Representation Separation Margin* | Sınıf içi kosinüs benzerliği ile sınıflar arası kosinüs benzerliği arasındaki fark ($\text{Margin} = \text{Sim}_{\text{intra}} - \text{Sim}_{\text{inter}}$). |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Cross-Entropy'ye göre etiket gürültüsüne (label noise) karşı aşırı dayanıklı; SimCLR'daki 'False Negative' sorununu tamamen çözer; Temsil uzayında sınıflar arasında $+1.26$ seviyesinde devasa geometrik marjin açar. |
| **Weaknesses (Zayıf Yönler)** | İki aşamalı eğitim gerektirir (Stage 1: SupCon + Stage 2: Linear Probing); Batch içinde her sınıftan birden fazla örnek bulunması zorunludur. |
| **Opportunities (Fırsatlar)** | Az örnekli öğrenme (Few-shot learning), Dağılım Dışı (OOD) tespiti, tıbbi görüntüleme ve yüz tanıma sistemlerinde sınıf ayrıştırma. |
| **Threats (Tehditler)** | Çok küçük batch boyutu kullanıldığında sınıfların tek düşmesi sonucu pozitif eşleşme bulunamaması. |

---

## 📈 Deneysel Benchmark & SupCon Eğitim Sonuçları

Sentetik 5 sınıflı görsel kümesi üzerinde $N=400$ eğitim, $N=100$ doğrulama örneği ile Stage 1 (8 Epoch) ve Stage 2 (5 Epoch) koşturulmuştur:

### Aşama 1: SupCon Temsil Ön Eğitimi
| Epoch | SupCon Kaybı | Sınıf İçi Kosinüs (Pos) | Sınıflar Arası Kosinüs (Neg) | Ayrışma Marjini | Durum |
|---|---|---|---|---|---|
| **1** | $4.3273$ | $+0.9856$ | $-0.2588$ | $+1.2444$ | Hızlı Kümelenme |
| **2** | $4.3646$ | $+0.9972$ | $-0.2723$ | $+1.2695$ | Kararlı İtme |
| **3** | $4.3519$ | $+0.9986$ | $-0.2709$ | $+1.2695$ | Yüksek Ayrışma |
| **4** | $4.3313$ | $+0.9981$ | $-0.2594$ | $+1.2576$ | Kararlı |
| **5** | $4.3001$ | $+0.9991$ | $-0.2557$ | $+1.2547$ | Sıkı Kümeler |
| **6** | $4.4542$ | $+0.9897$ | $-0.3019$ | $+1.2916$ | Maksimum Marjin |
| **7** | $4.3089$ | $+0.9993$ | $-0.2599$ | $+1.2592$ | Optimal Temsil |
| **8** | **$4.3371$** | **$+0.9988$** | **$-0.2680$** | **$+1.2669$** | **Aşama 1 Tamamlandı** |

### Aşama 2: Dondurulmuş Omurga Üzerinde Linear Probing (Sınıflandırma)
| Epoch | Cross-Entropy Kaybı | Doğrulama Doğruluğu (Val Acc) | Durum |
|---|---|---|---|
| **1** | $0.5324$ | **%100.00** | Anında Yakınsama |
| **2** | $0.0030$ | **%100.00** | Kayıp Sıfırlandı |
| **3** | $0.0003$ | **%100.00** | Kusursuz Ayrışma |
| **4** | $0.0001$ | **%100.00** | Kararlı Tepe |
| **5** | **$0.0001$** | **%100.00** | **Linear Probing Başarılı** |

- **Geometrik Ayrışma:** Sınıf içi kosinüs benzerliği $+0.9988$, sınıflar arası kosinüs benzerliği $-0.2680$ seviyesine oturmuş, **$+1.2669$ net ayrışma marjini** elde edilmiştir.
- **PCA Temsil Gücü:** İlk 2 temel bileşen toplam varyansın **%67.25**'ini açıklamakta ve 5 sınıf 2D uzayda birbirine temas etmeyen 5 izole ada oluşturmaktadır.
- **Birim Test Başarımı:** **$8 / 8$ PASSED (%100 Başarı, 10.20s)**

---

## 🖼️ Görsel Çıktı: 6 Panelli Teşhis Panosu

Laboratuvar çıktısı [`ciktilar/supcon_egitim_paneli.png`](file:///c:/Users/seydieryilmaz/Desktop/Github%20Mini%20AI%20Engineer/day-74-supervised-contrastive-learning/ciktilar/supcon_egitim_paneli.png) dosyasında üretilmiştir:
1. **SupCon Artırma Çiftleri**: Girdi görüntülerinin eşzamanlı üretilen $v_1$ ve $v_2$ görünümleri.
2. **Stage 1 SupCon Kayıp Trajektorisi**: SupCon kaybının yakınsama grafiği.
3. **Öğrenilen Temsil Uzayı (PCA İzdüşümü)**: 5 sınıfın mükemmel ayrışmış 2D kümeleri.
4. **Stage 2 Linear Probing Doğruluğu**: Dondurulmuş omurga üzerinde %100 doğrulama başarımı.
5. **Sınıf İçi vs Sınıflar Arası Kosinüs Benzerliği**: $+0.999$ pozitif vs $-0.268$ negatif kosinüs gelişimi.
6. **SupCon SWOT Karar Matrisi**: Mimari avantaj, dezavantaj ve endüstriyel fırsat haritası.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** SupCon eğitiminde etiket gürültüsünü (Label Noise) simüle eden ve etiketlerin rastgele %20'si bozulduğunda bile modelin sınıf ayrışma marjinini koruyan bir **Gürültülü Etiket Enjeksiyon Modülü (Noisy Label Injector)** yazınız.

**Eksiksiz Çözüm:**
```python
import torch

class GurultuluEtiketEnjektoru:
    """Etiketlerin belirli bir yüzdesini rastgele başka sınıflara çevirir."""
    def __init__(self, gurultu_orani: float = 0.20, sinif_sayisi: int = 5, tohum: int = 42):
        self.gurultu_orani = gurultu_orani
        self.sinif_sayisi = sinif_sayisi
        self.rng = torch.Generator().manual_seed(tohum)

    def enjekte_et(self, etiketler: torch.Tensor) -> torch.Tensor:
        gurultulu = etiketler.clone()
        N = etiketler.size(0)
        bozulacak_maske = torch.rand(N, generator=self.rng) < self.gurultu_orani
        
        rastgele_etiketler = torch.randint(0, self.sinif_sayisi, (N,), generator=self.rng)
        gurultulu[bozulacak_maske] = rastgele_etiketler[bozulacak_maske]
        return gurultulu
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Cross-Entropy kaybı bir sınıflandırma modelini eğitirken eğitim setinde %100 doğruluğa ulaştığı halde, SupCon ile eğitilip ardından linear probing uygulanan bir model dağılım dışı (OOD) testlerde ve etiket bozulmalarında neden dramatik şekilde daha üstün performans gösterir?

> **Mentor Cevabı:**
> 1. **Karar Sınırı Geometrisi:** Cross-Entropy, sınıfları yalnızca hiper-düzlemlerle (linear hyperplanes) ayırmaya odaklanır. Örnek doğru tarafında kaldığı sürece, sınıf merkezine ne kadar uzak olduğu veya diğer sınıfa ne kadar yakın olduğu Cross-Entropy için önemsizdir. Bu durum sınır bölgelerinde "kırılgan" temsiller yaratır.
> 2. **SupCon'un Açısal Kümeleme Gücü:** SupCon ise açısal mesafeleri doğrudan optimize eder; aynı sınıfa ait tüm örnekleri birim hiperküre üzerinde birbirine çekerken, diğer sınıfları zıt yönlere fırlatır. Sonuç olarak sınıflar arasında çok geniş bir "boş marjin" oluşur.
> 3. **OOD ve Gürültü Dayanıklılığı:** Dağılım dışı (OOD) veya bozuk bir girdi geldiğinde, Cross-Entropy rastgele bir hiper-düzlem tarafına düşüp %99 güvenle yanlış sınıf tahmin edebilirken; SupCon temsil uzayında bu girdi hiçbir sınıf kümesine yakın düşmez ve model anomaliyi hemen tespit eder.

---

## 📜 Lisans & Telif Hakkı

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
