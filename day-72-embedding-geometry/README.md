# Day 72: t-SNE, UMAP Boyut İndirgeme, Temsil Uzayı Geometrisi & İzotropi Analizi

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passed-brightgreen?style=flat-square)

## 🎯 Proje Özeti & Mühendislik Hedefi

Derin öğrenme modelleri (CNN'ler, Vision Transformer'lar, LLM'ler ve Kontrastif Temsil Modelleri), yüksek boyutlu uzaylarda ($d = 64, 512, 2048$) sürekli vektör temsilleri (**embeddings**) öğrenir. Ancak model eğitiminde iki büyük tehlike mevcuttur:
1. **Boyutsal Çöküş (Dimensional Collapse / Anisotropy Cone):** Yüksek boyutlu uzaydaki tüm vektörlerin dar bir koni içerisine sıkışarak boyutların büyük çoğunluğunun atıl kalması.
2. **Temsil Ayrışma Yetersizliği:** Sınıf içi ve sınıflar arası kosinüs mesafelerinin birbirine karışması.

Bu projede; yüksek boyutlu temsil uzaylarını 2D manifoldlara aktaran **PCA, t-SNE ve UMAP** algoritmalarını karşılaştırmalı olarak koşturan, **SVD Tekil Değer Spektrumu**, **Entropi Tabanlı İzotropi İndeksi** ve **Kosinüs Ayrışma Marjini** ile temsil kalitesini matematiksel olarak denetleyen kurumsal bir **Temsil Geometrisi Analiz Platformu** geliştirilmiştir.

---

## 🔬 Teorik & Matematiksel Derinlik

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           BOYUT İNDİRGEME VE TEMSİL GEOMETRİSİ KARŞILAŞTIRMASI                            │
│                                                                                                           │
│  [1. PCA (Lineer Projeksiyon)]:                                                                           │
│  • Maksimum varyans eksenlerini bulur. Global mesafeleri korur, non-lineer manifoldları katlar/büker.     │
│                                                                                                           │
│  [2. t-SNE (Non-Lineer Manifold İzdüşümü)]:                                                               │
│  • Yüksek boyutta Gauss, düşük boyutta Student-t dağılımı (Crowding Problem çözümü).                     │
│  • KL-Diverjansı ile YEREL komşulukları mükemmel ayırır, ancak KÜRESEL mesafeleri kaybeder.                │
│                                                                                                           │
│  [3. UMAP (Riemann Geometrisi & Bulanık Kümeler)]:                                                        │
│  • Fuzzy Simplicial Sets + Çapraz Entropi minimizasyonu.                                                  │
│  • Hem YEREL kümeleri hem de KÜRESEL topolojik ilişkileri korur; t-SNE'den kat kat hızlıdır.              │
│                                                                                                           │
│  [4. SVD İzotropi & Boyutsal Çöküş (Dimensional Collapse)]:                                              │
│  • İdeal Durum: Enerji tüm boyutlara eşit yayılır (İzotropi -> 1.0).                                      │
│  • Çöküş Durumu: İlk 2-3 eksen varyansın %95+'ini yutar (Anizotropik Koni, İzotropi -> 0.0).            │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. t-SNE Matematiksel Mekanizması & Student-t Dağılımı
Yüksek boyuttaki noktalar arasındaki benzerlik Gauss dağılımı ile modellenir:

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$

Düşük boyutlu 2D uzaydaki yığılma problemini (**Crowding Problem**) çözmek için ağır kuyruklu Student-t dağılımı kullanılır:

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \|y_k - y_l\|^2)^{-1}}$$

Optimizasyon hedefi Kullback-Leibler (KL) Diverjansıdır:

$$\mathcal{L}_{\text{t-SNE}} = D_{\text{KL}}(P \parallel Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

---

### 2. UMAP Matematiksel Mekanizması & Riemann Geometrisi
UMAP, verinin yerel olarak bağlantılı bir Riemann manifoldu üzerinde yattığını varsayar. Bulanık Kümeler arasındaki Çapraz Entropiyi (**Fuzzy Set Cross-Entropy**) minimize eder:

$$\mathcal{L}_{\text{UMAP}} = \sum_{e \in E} \left[ \mu(e) \log \left(\frac{\mu(e)}{\nu(e)}\right) + (1 - \mu(e)) \log \left(\frac{1 - \mu(e)}{1 - \nu(e)}\right) \right]$$

Bu sayede sadece komşuları değil, farklı kümelerin birbirine olan küresel göreli mesafelerini de korur.

---

### 3. İzotropi İndeksi & Boyutsal Çöküş Formülasyonu
Merkezlenmiş temsil matrisi $\bar{X} = X - \mu \in \mathbb{R}^{N \times d}$ için Tekil Değer Ayrışımı (SVD) yapılır:

$$\bar{X} = U \Sigma V^\top, \quad \Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_d)$$

Varyans enerji olasılık dağılımı $p_i$ ve Spektral Entropi $\mathcal{H}(\sigma)$:

$$p_i = \frac{\sigma_i^2}{\sum_{j=1}^d \sigma_j^2}, \quad \mathcal{H}(\sigma) = -\sum_{i=1}^d p_i \ln(p_i + 10^{-12})$$

Normalize İzotropi Skoru:

$$\mathcal{I}(X) = \frac{\exp(\mathcal{H}(\sigma))}{d} \in (0, 1]$$

- $\mathcal{I}(X) \approx 1.0$: Mükemmel izotropik hiperküre (Tüm boyutlar aktif ve bilgi taşıyor).
- $\mathcal{I}(X) \to 0$: Boyutsal çöküş (Anisotropy Cone - Vektörler sadece 1-2 boyutta hapsolmuş).

---

## 🛠️ Neden Bu Yöntem Seçildi? (Mühendislik Gerekçesi & Kaçınılan Tuzaklar)

1. **t-SNE Yanılsaması (Distance Illusion):** t-SNE'de iki küme arasındaki mesafe tamamen rastlantısaldır; küresel topoloji korunmaz. UMAP bu açığı kapatarak SimCLR/SupCon gibi kontrastif modellerde küme mesafelerini doğru analiz etmemizi sağlar.
2. **Kosinüs Benzerliği Bozulması:** Bir temsil uzayı anizotropik ise (dar bir koniye çökmüşse), tamamen ilgisiz iki vektörün kosinüs benzerliği $+0.95$ çıkar. İzotropi analizi, kontrastif eğitime geçmeden önce bu bozulmayı önceden yakalar.

---

## 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Yüksek boyutlu temsil vektörlerinin geometrisini, izotropi skorunu ve boyutsal çöküşünü (Dimensional Collapse) SVD, t-SNE ve UMAP ile denetlemek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Modelin tüm temsilleri tek bir dar koniye veya alt uzaya sıkıştırması (çöküş) tehlikesini erkenden teşhis eder.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- t-SNE ve UMAP non-lineer izdüşümler olduğu için global mesafeleri bazen yanıltıcı gösterebilir; SVD spektrumu ile desteklenmelidir.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- PCA, PaCMAP, Isomap veya Kernel PCA.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Embedding Space** | *Representation / Latent Space* | Modelin girdileri haritalandırdığı çok boyutlu sürekli vektör uzayı. |
| **Isotropy** | *Representation Isotropy* | Temsil vektörlerinin tüm uzay yönlerinde homojen ve dengeli dağılması durumu. |
| **Dimensional Collapse** | *Representation / Dimensional Collapse* | Yüksek boyutlu uzayda eğitilen modelin sadece 1-2 boyutu kullanıp diğer boyutları sıfırlaması hatası. |
| **Anisotropy Cone** | *Anisotropic Representation Cone* | Tüm embedding vektörlerinin dar bir koni içinde toplanıp kosinüs mesafelerini anlamsızlaştırması. |
| **t-SNE** | *t-Distributed Stochastic Neighbor Embedding* | Olasılıksal yerel komşulukları 2D/3D'ye aktaran non-lineer boyut indirgeme algoritması. |
| **UMAP** | *Uniform Manifold Approximation & Projection* | Riemann geometrisi ve topoloji tabanlı, hem yerel hem küresel yapıyı koruyan hızlı algoritma. |
| **Perplexity** | *t-SNE Perplexity Parameter* | t-SNE'de yerel komşu sayısı dengesini ayarlayan hiperparametre (genelde 5-50 arası). |
| **Crowding Problem** | *High-to-Low Dimension Crowding Problem* | Yüksek boyuttaki hacmin düşük boyuta aktarılırken noktaların birbirinin üstüne yığılması sorunu. |
| **Cosine Separation Margin** | *Intra vs Inter Cosine Margin* | Sınıf içi ortalama kosinüs benzerliği ile sınıflar arası ortalama benzerlik arasındaki fark ($\bar{S}_{\text{intra}} - \bar{S}_{\text{inter}}$). |
| **Singular Value Spectrum** | *SVD Singular Value Spectrum* | Temsil matrisinin tekil değerlerinin büyükten küçüğe dizilimi (Varyans çöküşünün birincil göstergesi). |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | UMAP ile hem yerel kümelerin hem küresel mesafelerin korunması; SVD İzotropi Analizi ile çöküşün önceden tespiti; Kosinüs marjini ile objektif ayrışabilirlik ölçümü. |
| **Weaknesses (Zayıf Yönler)** | t-SNE'nin $O(N^2)$ hesaplama karmaşıklığı ve küresel mesafeleri çarpıtması. |
| **Opportunities (Fırsatlar)** | SimCLR, SupCon ve Vision Transformer eğitimleri öncesinde temsil kalitesi denetim standardı oluşturulması. |
| **Threats (Tehditler)** | Yanlış perplexity veya neighbor seçimi sonucu yapay kümelenme yanılsaması (hallucinated clustering). |

---

## 📈 Deneysel Benchmark & Geometrik Analiz Sonuçları

Platformumuzda $N=600, d=64, C=5$ boyutlu **Sağlıklı Temsil** ile **Boyutsal Çöküşe Uğramış Temsil** karşılaştırılmıştır:

| Metrik Adı | Sağlıklı Temsil | Çökmüş Temsil (Collapsed) | Endüstriyel Yorum |
|---|---|---|---|
| **İzotropi İndeksi ($\exp(H) / d$)** | **$0.3615$** | **$0.0316$** | Çökmüş uzayda izotropi %91 oranında yok olmuştur. |
| **Min/Max Tekil Değer Oranı** | **$0.123382$** | **$0.001989$** | Çökmüş uzayda son boyutların varyansı sıfıra yaklaşmıştır. |
| **Efektif Boyut ($\exp(H)$)** | **$23.14$ / 64** | **$2.02$ / 64** | 64 boyutun yalnızca 2 tanesi aktif olarak kullanılıyor! |
| **İlk 3 Eksen Varyans Payı** | **%48.1** | **%99.9** | **Çöküş Teşhisi:** İlk 3 eksen tüm varyansı yutmuştur. |
| **Sınıf İçi Kosinüs Benzerliği** | **$+0.6085$** | N/A | Sınıf üyeleri birbirine güçlü şekilde çekilmiştir. |
| **Sınıflar Arası Kosinüs** | **$-0.0116$** | N/A | Farklı sınıflar ortogonal konuma itilmiştir. |
| **Kosinüs Ayrışma Marjini** | **$+0.6201$** | N/A | **Mükemmel Temsil Ayrışması ($\Delta_{\text{sep}} > 0.5$).** |
| **t-SNE KL Diverjansı** | **$0.9164$** | N/A | Düşük boyutlu izdüşüm yakınsamıştır. |
| **Birim Test Başarımı** | **$8 / 8$ PASSED** | **%100 Başarı (27.1s)** | Tüm algoritmalar ve teşhis motoru doğrulandı. |

---

## 🖼️ Görsel Çıktı: 6 Panelli Teşhis Panosu

Laboratuvar çıktısı [`ciktilar/temsil_geometrisi_paneli.png`](file:///c:/Users/seydieryilmaz/Desktop/Github%20Mini%20AI%20Engineer/day-72-embedding-geometry/ciktilar/temsil_geometrisi_paneli.png) dosyasında üretilmiştir:
1. **PCA 2D İzdüşümü**: Lineer varyans eksenlerinde sınıf ayrışması (%35.5 varyans).
2. **t-SNE 2D İzdüşümü**: Yerel manifold kümelerinin ayrışması ($KL = 0.916$).
3. **UMAP 2D İzdüşümü**: Riemann geometrisi ve kosinüs metriği ile yerel + küresel topoloji.
4. **SVD Spektrumu & Decay**: Sağlıklı vs Çökmüş temsilin tekil değer sönümlenme eğrisi (Log Scale).
5. **Kosinüs Benzerlik Dağılımı**: Sınıf içi ($+0.608$) ve sınıflar arası ($-0.012$) dağılım eğrileri.
6. **SWOT Karar Matrisi**: Geometri analizlerinin endüstriyel sentezi.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Verilen bir temsil matrisinde ($N \times d$) **Anizotropik Koni Etkisini** gidermek için ortalama vektörü çıkarıp (**Mean Centering**) ve ilk temel bileşeni çıkararak (**PCA Subtraction / Whitening**) izotropiyi artıran bir fonksiyon yazınız.

**Eksiksiz Çözüm:**
```python
import numpy as np

def izotropi_iyilestir_ve_beyazlat(X: np.ndarray, cikarilacak_bilesen_sayisi: int = 1) -> np.ndarray:
    """Temsil uzayındaki anizotropik koniyi ortalama çıkarma ve SVD projeksiyonu ile temizler."""
    # 1. Ortalama Çıkarma (Mean Centering)
    X_merkezli = X - np.mean(X, axis=0, keepdims=True)
    
    # 2. SVD ile Dominant Anizotropik Eksenleri Bul
    U, S, Vt = np.linalg.svd(X_merkezli, full_matrices=False)
    
    # 3. İlk k dominant ekseni çıkar (Anizotropi temizleme)
    dominant_yonler = Vt[:cikarilacak_bilesen_sayisi, :] # (k, d)
    projeksiyon = np.dot(X_merkezli, dominant_yonler.T) @ dominant_yonler
    X_temiz = X_merkezli - projeksiyon
    
    # 4. L2 Normalizasyon
    X_norm = X_temiz / (np.linalg.norm(X_temiz, axis=1, keepdims=True) + 1e-12)
    return X_norm
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Yüksek boyutlu bir derin öğrenme modelinde (örneğin Vision Transformer veya SimCLR) t-SNE görselleştirmesinde kümelerin birbirinden çok uzak görünmesi, sınıfların gerçek yüksek boyutlu uzayda da birbirine bu kadar uzak olduğunu kanıtlar mı? Neden UMAP bu analizde t-SNE'ye tercih edilir?

> **Mentor Cevabı:**
> 1. **t-SNE'nin Küresel Mesafe Yanılsaması:** Hayır, kesinlikle kanıtlamaz! t-SNE algoritması maliyet fonksiyonu olarak yalnızca yerel komşulukları modelleyen KL-Diverjansını optimize eder. Uzak noktalar için gradyan hızla sıfıra yaklaşır. Bu nedenle t-SNE projeksiyonunda iki küme arasındaki 2D mesafe tamamen rastlantısaldır ve küresel topolojik mesafeyi yansıtmaz.
> 2. **UMAP'in Üstünlüğü:** UMAP, Riemann manifoldu ve bulanık kümeler (Fuzzy Simplicial Sets) teorisi üzerine inşa edilmiştir. UMAP'in optimizasyonunda yer alan çapraz entropi fonksiyonu hem $e \in E$ (bağlantılı/komşu noktalar) hem de $e \notin E$ (ayrık/uzak noktalar) için negatif gradyan üretir. Bu sayede UMAP, yerel kümeleri ayrıştırırken aynı zamanda kümelerin birbirine olan **küresel göreli mesafelerini** de matematiksel olarak korur.

---

## 📜 Lisans & Telif Hakkı

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
