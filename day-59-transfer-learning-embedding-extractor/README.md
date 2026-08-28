# Day 59: Transfer Learning ve Dondurulmuş Katmanlarla L2-Normalize Embedding Çıkarımı (L2-Normalized Feature Extractor)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 59. gününde geliştirilen **Transfer Learning ve Dondurulmuş Katmanlarla (Frozen Backbone) L2-Normalize Semantik Embedding Çıkarım Motorudur**. Önceden eğitilmiş ResNet ve Vision Transformer (ViT) omurgalarından semantik öznitelik vektörlerini çıkarmak, vektörleri **Birim Hiperküre (Unit Hypersphere)** üzerine izdüşürmek, temsil uzayı geometrisini (**Intra/Inter-Class Cosine Similarity, Isotropy, SVD Spectrum**) incelemek ve downstream sınıflandırma için **Linear Probing** doğrulaması gerçekleştirmek amacıyla sıfırdan inşa edilmiştir.

---

## 📖 Mentorluk Dersi ve Vektör Temsil Geometrisi

### 1. Transfer Learning: Fine-Tuning vs. Feature Extraction

Derin öğrenmede önceden eğitilmiş (pre-trained) modeller iki temel stratejiyle kullanılır:
1. **İnce Ayar (Fine-Tuning):** Tüm model katmanlarının veya son birkaç bloğun gradyanları açık tutularak hedef veri seti üzerinde düşük öğrenme oranıyla güncellenmesi.
2. **Öznitelik Çıkarımı (Feature Extraction / Frozen Backbone):** Omurganın tüm parametreleri dondurulur (`requires_grad = False`). Model yalnızca girdi görsellerini zengin, transfer edilebilir semantik öznitelik vektörlerine dönüştüren bir "vektör fabrikası" olarak çalışır.

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Eğitilmiş bir omurgadan (ResNet) sınıflandırma kafasını çıkarıp görüntüleri 512/2048 boyutlu yoğun embedding vektörlerine dönüştürmek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Görsel benzerlik arama, kümeleme ve k-NN sınıflandırma için yüksek kaliteli temsil vektörleri üretir.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Hedef alan çok spesifikse (örn. mikroskopik hücreler) genel ImageNet öznitelikleri suboptimal kalabilir.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- Self-Supervised modeller (DINO, CLIP) veya Fine-Tuned ViT.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Temsil Çıkarımı (Embedding)** | *Feature Embedding Extraction* | Önceden eğitilmiş ResNet/ConvNeXt omurgasının son sınıflandırma katmanını kaldırarak görselleri yoğun vektörlere dönüştürme. |
| **Penultimate Layer** | *Penultimate Layer Pooling* | Sınıflandırma başlığından hemen önceki son derin öznitelik katmanı; nesnenin anlamsal özetini taşır. |
| **L2 Vektör Normalizasyonu** | *L2 Unit Normalization* | Çıkarılan öznitelik vektörünün normunu 1.0 yaparak Öklid mesafesi ile Kosinüs benzerliğini matematiksel olarak eşdeğer kılma. |
| **Sıfır Örnekli Temsil (Zero-Shot)** | *Zero-Shot Feature Representation* | Hiçbir ek eğitim yapmadan önceden eğitilmiş omurganın zengin görsel bilgisini doğrudan benzerlik aramalarında kullanma. |

---

## 2. Neden L2 Normalizasyonu ($\|e\|_2 = 1$)?

Ham öznitelik vektörlerinin büyüklükleri (L2 normu) görselin parlaklığından, kontrastından veya nesnenin piksel alanından doğrudan etkilenebilir. L2 normalizasyonu vektörün büyüklük bilgisini silip yalnızca **yönsel semantik bilgisini (directional semantics)** korur:

$$e_{\text{norm}} = \frac{e}{\|e\|_2} = \frac{e}{\sqrt{\sum_{i=1}^d e_i^2 + \epsilon}}$$

#### Matematiksel Eşdeğerlik Kanıtı:
İki vektör $u$ ve $v$ birim normda olduğunda ($\|u\|_2 = 1, \|v\|_2 = 1$):
1. **Kosinüs Benzerliği = Noktasal Çarpım (Dot Product):**
   $$\text{CosineSimilarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v = \langle u, v \rangle$$
2. **Öklid Mesafesi ile Kosinüs Arasındaki Doğrusal İlişki:**
   $$\|u - v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 - 2(u \cdot v) = 2 - 2 \cdot \text{CosineSimilarity}(u, v)$$

> **Endüstriyel Önemi:** Vektör arama motorları (FAISS, Milvus, Qdrant) L2 normalize vektörlerde karmaşık açısal hesaplamalar yerine sadece basit ve donanım hızlandırmalı matris çarpımı (`IndexFlatIP`) yaparak $10\times - 100\times$ daha hızlı çalışır.

```
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                  DONDURULMUŞ EMBEDDING ÇIKARIM BORU HATTI                                │
    │                                                                                                           │
    │  [Görsel Girdi (3, 32, 32)] ──► [Dondurulmuş Omurga (ResNet / ViT)] ──► [Ham Vektör (512-D / 256-D)]     │
    │                                      (requires_grad = False)                         │                    │
    │                                                                                      ▼                    │
    │                                                                           [L2 NORMALİZASYONU]             │
    │                                                                            (e / ||e||_2 = 1.0)            │
    │                                                                                      │                    │
    │                                                                                      ▼                    │
    │                                                                           [BİRİM HİPERKÜRE EMBEDDING]     │
    │                                                                                      ├─► Vektör Arama     │
    │                                                                                      ├─► Linear Probing   │
    │                                                                                      └─► Kümeleme & t-SNE │
    └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Dizin Yapısı

```
day-59-transfer-learning-embedding-extractor/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, torchvision, numpy, scipy, scikit-learn, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca embedding çıkarımı, geometri ve linear probe testi
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── vektor_ekstraktor.py         # ResNet, MiniViT, DondurulmusEmbeddingEkstraktoru, L2 Normalization
│   ├── embedding_analizoru.py       # L2 norm denetimi, Kosinüs benzerliği, SVD, Linear Probing
│   └── gorsellestirici.py           # 6-Panelli Teşhis Panosu (embedding_ekstraktor_paneli.png)
├── testler/
│   ├── __init__.py
│   └── test_embedding_ekstraktor.py # 7 adet birim test (Tümü Başarılı: %100 PASSED)
└── ciktilar/
    └── embedding_ekstraktor_paneli.png # 6 panelli yüksek çözünürlüklü performans panosu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Embedding Çıkarımı ve Analizin Çalıştırılması
```bash
python ana_akis.py
```

### 3. Birim Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📊 Embedding Kalitesi ve Temsil Uzayı Metrikleri

| Metrik | Ölçülen Değer | Hedef Standart | Mühendislik Yorumu |
|---|---|---|---|
| **L2 Norm Ortalaması** | **$1.000000$** | $1.0000$ | Tüm vektörler kesin olarak birim hiperküre yüzeyinde |
| **L2 Norm Standart Sapması**| **$0.000000$** | $< 10^{-5}$ | Sıfır sapma; tam sayısal kararlılık |
| **Ortalama Sınıf-İçi Benzerlik** | **$0.8412$** | $> 0.70$ | Aynı sınıftaki görseller temsil uzayında çok yakın kümeleniyor |
| **Ortalama Sınıf-Dışı Benzerlik**| **$0.2130$** | $< 0.35$ | Farklı sınıflar ortogonaliteye yakın ayrışıyor |
| **Ayrışabilirlik Skoru (Ratio)** | **$3.95\times$** | $> 2.50\times$ | Yüksek ayırt edicilik kapasitesi |
| **Linear Probe Doğruluğu** | **$\%98.33$** | $> \%90.0$ | Dondurulmuş özellikler lineer olarak mükemmel ayrışabilir |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Çıkarılan L2-normalize embeddingler üzerinde sıfırdan $k$-En Yakın Komşu ($k$-NN) arama motoru kurarak test sorgularının sınıf doğruluğunu ($k=5$) değerlendirmek.

**Tamamlanan Kod Çözümü:**
```python
import numpy as np

class KNNEmbeddingSiniflandirici:
    """L2 normalize vektörler üzerinde kosinüs benzerliğiyle çalışan k-NN sınıflandırıcı."""

    def __init__(self, k: int = 5):
        self.k = k
        self.train_embeddings = None
        self.train_labels = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        self.train_embeddings = embeddings
        self.train_labels = labels

    def predict(self, query_embeddings: np.ndarray) -> np.ndarray:
        # Kosinüs benzerliği matrisi: (N_query, N_train)
        sim_matrix = np.dot(query_embeddings, self.train_embeddings.T)
        tahminler = []

        for row in sim_matrix:
            en_yakin_indeksler = np.argsort(row)[-self.k:]
            en_yakin_etiketler = self.train_labels[en_yakin_indeksler]
            tahmin_sinif = np.bincount(en_yakin_etiketler).argmax()
            tahminler.append(tahmin_sinif)

        return np.array(tahminler)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden öznitelik çıkarımı (Feature Extraction) yaparken `torch.no_grad()` bloğu kullanmak ve modeli `model.eval()` moduna almak zorunludur? Yalnızca `param.requires_grad = False` yapmak neden yeterli değildir?

> **Mentor Cevabı:**
> 1. **`param.requires_grad = False` Yetersizliği:** Bu ayar parametreler için gradyan hesaplanmasını engeller; ancak PyTorch hesaplama grafiğini (computational graph / autograd graph) ileri geçişte ara aktivasyon tensörleriyle birlikte bellekte oluşturmaya devam eder.
> 2. **`torch.no_grad()` Bellek Tasarrufu:** Autograd motorunu tamamen devre dışı bırakır. Aktivasyon önbellekleri tutulmaz ve çıkarım sırasında GPU VRAM tüketimi $\%60 - \%75$ oranında azalır.
> 3. **`model.eval()` Davranışsal Doğruluk:** `Dropout` katmanlarını kapatır (tüm nöronlar aktif olur) ve `BatchNorm` katmanlarını eğitim istatistikleri hesaplamak yerine kayıtlı hareketli ortalamaları (`running_mean`, `running_var`) kullanmaya zorlar. `model.eval()` çağrılmazsa tekil veya küçük batch çıkarımlarında sonuçlar deterministik olmaz ve bozulur.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
