# Day 73: Sıfırdan SimCLR Temsil Öğrenimi, Artırma Çiftleri, NT-Xent (InfoNCE) Kaybı

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passed-brightgreen?style=flat-square)

## 🎯 Proje Özeti & Mühendislik Hedefi

Geleneksel denetimli öğrenme (Supervised Learning), milyonlarca insan tarafından etiketlenmiş veri gerektirir; bu durum tıp, uydu görüntüleme ve endüstriyel kalite kontrol gibi alanlarda aşırı maliyetlidir. **SimCLR (Simple Framework for Contrastive Learning of Visual Representations - Chen et al., 2020)**, tek bir insan etiketi dahi kullanmadan, yalnızca stokastik veri artırma çiftleri (**Augmentation Pairs**) ve **NT-Xent (InfoNCE)** kaybı ile görsel temsilleri öğrenir.

Bu projede; stokastik çift görünüm artırma boru hattını, temel kodlayıcı omurgayı ($f$), temsil kalitesini %10 artıran **Non-lineer Projeksiyon Kafasını ($g$)**, tensörel olarak optimize edilmiş **NT-Xent Kaybını** ve **Hizalama & Düzgünlük (Alignment & Uniformity)** metriklerini sıfırdan inşa ediyoruz.

---

## 🔬 Teorik & Matematiksel Derinlik

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                SimCLR KONTRASTİF ÖĞRENİM MİMARİSİ                                         │
│                                                                                                           │
│             ┌───────────────┐                                                                             │
│             │  Girdi x (PIL)│                                                                             │
│             └───────┬───────┘                                                                             │
│                     │                                                                                     │
│          ┌──────────┴──────────┐                                                                          │
│          ▼                     ▼                                                                          │
│    t ~ T (Crop+Color)    t' ~ T (Crop+Blur)                                                               │
│    ┌────────────┐        ┌────────────┐                                                                   │
│    │ Görünüm x_i│        │ Görünüm x_j│  --> POZİTİF ÇİFT (Aynı kaynaktan)                                │
│    └──────┬─────┘        └─────┬──────┘                                                                   │
│           ▼                    ▼                                                                          │
│    ┌────────────┐        ┌────────────┐                                                                   │
│    │  f(x_i)=h_i│        │  f(x_j)=h_j│  --> Temel Temsil (Downstream görevlerde kullanılan)              │
│    └──────┬─────┘        └─────┬──────┘                                                                   │
│           ▼                    ▼                                                                          │
│    ┌────────────┐        ┌────────────┐                                                                   │
│    │  g(h_i)=z_i│        │  g(h_j)=z_j│  --> Non-lineer MLP Projeksiyonu (L2 Normalize)                   │
│    └──────┬─────┘        └─────┬──────┘                                                                   │
│           └──────────┬─────────┘                                                                          │
│                      ▼                                                                                    │
│            NT-Xent (InfoNCE) Kaybı                                                                        │
│    (z_i ve z_j'yi ÇEK; batch'teki diğer 2(N-1) negatif örneği İT)                                         │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. NT-Xent (Normalized Temperature-scaled Cross Entropy) Kaybı
$N$ boyutlu bir minibatch'ten $2N$ adet artırılmış görünüm üretilir. Pozitif bir çift $(i, j)$ için kayıp:

$$\ell_{i, j} = -\log \frac{\exp\left(\frac{\text{sim}(z_i, z_j)}{\tau}\right)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp\left(\frac{\text{sim}(z_i, z_k)}{\tau}\right)}$$

Burada $\text{sim}(u, v) = \frac{u^\top v}{\|u\|_2 \|v\|_2}$ kosinüs benzerliği, $\tau$ ise sıcaklık parametresidir (**Temperature**). Minibatch üzerindeki toplam SimCLR kaybı:

$$\mathcal{L}_{\text{SimCLR}} = \frac{1}{2N} \sum_{k=1}^N \Big[ \ell_{2k-1, 2k} + \ell_{2k, 2k-1} \Big]$$

---

### 2. Sıcaklık Parametresi ($\tau$) ve Zor Negatif Örnekler (Hard Negatives)
Sıcaklık parametresi $\tau$, modelin negatif örneklere uyguladığı ceza sertliğini kontrol eder:
- $\tau \to 0$ olduğunda: Kayıp fonksiyonu yalnızca en zor negatif örneğe ($\max_{k} \text{sim}(z_i, z_k)$) odaklanır (Hard Negative Mining).
- $\tau \to \infty$ olduğunda: Tüm negatif örneklere eşit ağırlık verilir ve temsil ayrışma kabiliyeti çöker.
- Optimal değer: $\tau \in [0.1, 0.5]$ aralığıdır.

---

### 3. Hizalama (Alignment) ve Düzgünlük (Uniformity) Teorisi (Wang & Isola, 2020)
İyi bir temsil uzayının sahip olması gereken iki temel matematiksel özellik:

$$\mathcal{L}_{\text{align}}(f; \alpha) \triangleq \mathbb{E}_{(x, x^+)} \Big[ \|f(x) - f(x^+)\|_2^\alpha \Big], \quad \alpha > 0$$

$$\mathcal{L}_{\text{uniform}}(f; t) \triangleq \log \mathbb{E}_{x, y \stackrel{i.i.d.}{\sim} p_{\text{data}}} \Big[ \exp\left(-t \|f(x) - f(y)\|_2^2\right) \Big], \quad t > 0$$

- **Alignment:** Pozitif çiftlerin uzayda aynı noktaya çekilme başarısı ($\|z_1 - z_2\|^2 \to 0$).
- **Uniformity:** Vektörlerin birim hiperküre üzerine maksimum entropi ile homojen yayılması (Boyutsal çöküşü engeller).

---

## 🛠️ Neden Bu Yöntem Seçildi? (Mühendislik Gerekçesi & Kaçınılan Tuzaklar)

1. **Neden Non-Lineer Projeksiyon Kafası $g(\cdot)$ Kullanılır?**
   - $g(h) = W^{(2)} \text{ReLU}(W^{(1)} h)$ katmanı, artırmalara karşı değişmezlik (invariance) sağlarken bazı yararlı bilgileri (renk, konum, nesne yönelimi) atar.
   - Eğer $z$ doğrudan sınıflandırıcıya verilirse doğruluk düşer. Ancak $g(\cdot)$'den önceki $h$ temsili hem genel invariant özellikleri hem de downstream görevler için zengin detayları korur! SimCLR makalesinde $h$ temsili, $z$'ye göre **+%10 daha yüksek lineer probing doğruluğu** vermiştir.
2. **Kritik Artırma Bileşimi:** Yalnızca Random Crop veya yalnızca Color Jitter tek başına yetersizdir. Model görüntülerin renk histogramını ezberleyerek kestirme yol (shortcut learning) bulur. **Random Crop + Color Jitter bileşimi**, kestirme yolları imkansız kılar.

---

## 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Tek bir insan etiketi kullanmadan, stokastik artırma çiftleri ve NT-Xent (InfoNCE) kaybı ile zengin görsel temsiller öğrenmek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Etiketleme maliyetini ortadan kaldırır; downstream görevler için transfer edilebilir güçlü öznitelik uzayı kurar.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Büyük batch boyutu ($N=4096$) gerektirir ve aynı sınıftan gelen farklı örnekleri de negatif sanıp itebilir (False Negative sorunu).

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- MoCo v2 (Memory Queue), BYOL (Negative-free), SwAV veya Supervised Contrastive (SupCon).

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Contrastive Learning** | *Contrastive Representation Learning* | Pozitif örnek çiftlerini birbirine çekerken negatif örnekleri uzayda birbirinden iten temsil öğrenimi yaklaşımı. |
| **Self-Supervised Learning** | *Self-Supervised Learning (SSL)* | İnsan etiketine ihtiyaç duymadan, verinin kendi içsel yapısından gözetim sinyali üreten makine öğrenimi paradigması. |
| **Augmentation Pair** | *Correlated Stochastic Views* | Aynı kaynak görüntüye iki farklı stokastik veri artırma uygulanarak üretilen pozitif görüntü çifti. |
| **NT-Xent** | *Normalized Temperature-scaled Cross Entropy* | InfoNCE kaybının L2 normalize edilmiş ve sıcaklık ölçekli kosinüs benzerliği kullanan SimCLR türevi. |
| **InfoNCE** | *Information Noise-Contrastive Estimation* | Mutual Information (Karşılıklı Bilgi) alt sınırını maksimize eden olasılıksal kontrastif kayıp fonksiyonu. |
| **Projection Head** | *Non-linear Projection MLP* | Temel temsil vektörünü $h$ kontrastif kayıp uzayı $z$'ye eşleyen 2 katmanlı non-lineer MLP bloğu. |
| **Base Encoder** | *Feature Extractor Backbone* | Görüntüyü yüksek boyutlu öznitelik temsiline dönüştüren ana omurga ağ ($f(x) = h$). |
| **Temperature ($\tau$)** | *Softmax Temperature Parameter* | Softmax olasılık dağılımının keskinliğini ve zor negatif örneklere verilen cezayı ayarlayan katsayı. |
| **Alignment** | *Representation Alignment* | Aynı sınıftan / pozitif çiftlerden gelen temsil vektörlerinin uzaydaki yakınlık derecesi. |
| **Uniformity** | *Hyperspherical Uniformity* | Temsil vektörlerinin birim hiperküre yüzeyinde maksimum bilgi taşıyacak şekilde homojen yayılma derecesi. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Tek bir etiket olmadan ImageNet seviyesinde temsil gücü; Non-lineer Projeksiyon Kafası ile downstream doğruluğunda +%10 sıçrama; NT-Xent kaybı ile pozitifleri çekerken negatifleri homojen itme. |
| **Weaknesses (Zayıf Yönler)** | Büyük batch boyutu gereksinimi ($N = 4096$) - Çok GPU belleği ister; yanlış artırma politikası seçildiğinde temsil kalitesi çöker. |
| **Opportunities (Fırsatlar)** | Tıp ve uydu görüntüleme gibi etiket maliyeti astronomik alanlarda devrim; Linear Probing ile sadece %1 etiketle tam denetimli modeli yakalama. |
| **Threats (Tehditler)** | Negatif çiftlerin içinde aslında aynı sınıftan olan örneklerin bulunması (False Negatives); sıcaklık katsayısı $\tau$ çok küçük seçilirse gradyan patlaması. |

---

## 📈 Deneysel Benchmark & SimCLR Eğitim Sonuçları

Sentetik 5 sınıflı görsel kümesi üzerinde $N=500$ örnek ve batch boyutu $64$ ile 8 epoch SimCLR eğitimi koşturulmuştur:

| Epoch | NT-Xent Kaybı | Alignment Hatası ($\|z_1-z_2\|^2$) | Pozitif Çift Kosinüs | Negatif Çift Kosinüs | Ayrışma Marjini | Durum |
|---|---|---|---|---|---|---|
| **1** | $3.5657$ | $0.0811$ | $+0.9594$ | $-0.0094$ | $+0.9689$ | Isınma |
| **2** | $3.5148$ | $0.0034$ | $+0.9983$ | $-0.0125$ | $+1.0108$ | Yakınsama |
| **3** | $3.5251$ | $0.0012$ | $+0.9994$ | $-0.0089$ | $+1.0083$ | Stabil |
| **4** | $3.5134$ | $0.0006$ | $+0.9997$ | $-0.0115$ | $+1.0112$ | Yüksek Hizalama |
| **5** | $3.5295$ | $0.0019$ | $+0.9991$ | $-0.0084$ | $+1.0075$ | Stabil |
| **6** | $3.5269$ | $0.0006$ | $+0.9997$ | $-0.0092$ | $+1.0089$ | Homojen İtme |
| **7** | $3.5185$ | $0.0007$ | $+0.9996$ | $-0.0114$ | $+1.0110$ | Yüksek Ayrışma |
| **8** | **$3.5321$** | **$0.0018$** | **$+0.9991$** | **$-0.0074$** | **$+1.0065$** | **Ön Eğitim Tamamlandı** |

- **Pozitif Hizalama (Alignment):** Pozitif çiftler arasındaki kosinüs benzerliği $+0.959 \to +0.999$ seviyesine yükselmiş, hizalama hatası neredeyse sıfırlanmıştır ($0.0018$).
- **Negatif İtme (Uniformity):** Negatif çiftler arasındaki kosinüs benzerliği $-0.0074$ (tam ortogonal) seviyesinde tutularak boyutsal çöküş %100 engellenmiştir.
- **PCA Temsil Ayrışması:** Öğrenilen $h$ temsillerinin ilk 2 temel bileşeni toplam varyansın **%73.19**'unu açıklamaktadır.
- **Birim Test Başarımı:** **$8 / 8$ PASSED (%100 Başarı, 6.82s)**

---

## 🖼️ Görsel Çıktı: 6 Panelli Teşhis Panosu

Laboratuvar çıktısı [`ciktilar/simclr_egitim_paneli.png`](file:///c:/Users/seydieryilmaz/Desktop/Github%20Mini%20AI%20Engineer/day-73-simclr-from-scratch/ciktilar/simclr_egitim_paneli.png) dosyasında üretilmiştir:
1. **Stokastik Artırma Çiftleri**: Girdi görüntüsünün eşzamanlı üretilen $v_1$ ve $v_2$ görünümleri.
2. **NT-Xent Kayıp Trajektorisi**: InfoNCE kaybı ve Alignment hatasının yakınsama grafiği.
3. **Etiketsiz Öğrenilen Temsil Uzayı**: Modelin hiçbir insan etiketi görmeden sınıfları uzayda ayrıştırdığını gösteren $h$ projeksiyonu.
4. **Sıcaklık Katsayısı ($\tau$) Sertlik Analizi**: Farklı $\tau \in \{0.1, 0.2, 0.5, 1.0\}$ değerlerinde negatif örneklere verilen ceza eğrisi.
5. **Kosinüs Benzerliği ve Marjin Gelişimi**: Pozitif ($+0.999$) vs Negatif ($-0.007$) kosinüs eğrileri ve $+1.007$ marjin.
6. **SWOT Karar Matrisi**: SimCLR mimarisinin endüstriyel avantaj ve dezavantajları.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** SimCLR'da büyük batch boyutu gereksinimini hafifletmek amacıyla, önceki batch'lerden gelen negatif projeksiyon vektörlerini saklayan bir **Bellek Kuyruğu (Memory Queue / MoCo-style FIFO Buffer)** mekanizması yazınız.

**Eksiksiz Çözüm:**
```python
import torch
import torch.nn.functional as F

class SimCLRBellekKuyrugu:
    """Önceki batch'lerin negatif z vektörlerini FIFO kuyruğunda saklar."""
    def __init__(self, kuyruk_boyutu: int = 1024, ozellik_boyutu: int = 64, cihaz: str = "cpu"):
        self.kuyruk_boyutu = kuyruk_boyutu
        self.kuyruk = F.normalize(torch.randn(kuyruk_boyutu, ozellik_boyutu, device=cihaz), p=2, dim=1)
        self.isaretci = 0

    @torch.no_grad()
    def kuyruga_ekle(self, z: torch.Tensor):
        batch_boyutu = z.size(0)
        z = z.detach()
        if self.isaretci + batch_boyutu <= self.kuyruk_boyutu:
            self.kuyruk[self.isaretci:self.isaretci + batch_boyutu] = z
            self.isaretci = (self.isaretci + batch_boyutu) % self.kuyruk_boyutu
        else:
            kalan = self.kuyruk_boyutu - self.isaretci
            self.kuyruk[self.isaretci:] = z[:kalan]
            self.kuyruk[:batch_boyutu - kalan] = z[kalan:]
            self.isaretci = batch_boyutu - kalan

    def negatifleri_al(self) -> torch.Tensor:
        return self.kuyruk.clone()
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** SimCLR eğitiminde neden temel kodlayıcının çıktısı $h = f(x)$ doğrudan NT-Xent kaybına sokulmaz da araya non-lineer bir Projeksiyon Kafası $z = g(h)$ eklenir? Model eğitildikten sonra downstream görevlerde (örn. sınıflandırma) neden $z$ yerine $h$ kullanılır?

> **Mentor Cevabı:**
> 1. **Projeksiyon Kafasının Amacı (Bilgi Kaybı Filtresi):** Kontrastif NT-Xent kaybı, iki artırılmış görünümü birbirine eşitlerken ($z_i \approx z_j$), artırma operasyonunun getirdiği farklılıkları (nesnenin rengi, ölçeği, yönelimi, arka planı) yok etmeye zorlar. Bu bilgi kaybı kontrastif uzayda gereklidir ancak downstream görevlerde (örneğin segmentasyon veya ince taneli sınıflandırma) nesnenin rengi ve yönelimi çok değerlidir.
> 2. **Downstream Görevlerde $h$'ın Seçilmesi:** $g(h) = W^{(2)} \text{ReLU}(W^{(1)} h)$ non-lineer bir projeksiyon olduğu için, renk/konum gibi bilgileri $z$ seviyesinde atar; fakat bu bilgiler $g$'den önceki $h$ temsil vektöründe eksiksiz olarak muhafaza edilir.
> 3. **Kanıt:** Chen et al. (2020) makalesinde, lineer sınıflandırıcı $z$ üzerine kurulduğunda doğruluğun %10'dan fazla düştüğü, $h$ üzerine kurulduğunda ise tam denetimli ResNet-50 seviyesini yakaladığı deneysel olarak kanıtlanmıştır.

---

## 📜 Lisans & Telif Hakkı

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
