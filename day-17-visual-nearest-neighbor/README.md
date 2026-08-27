# Day 17: Vektör Benzerliği Tabanlı Görsel Arama (Visual Nearest Neighbor - Image Search Engine)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.24+-orange.svg?style=flat-square)](https://scikit-image.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; e-ticarette tersine görsel arama (Reverse Image Search - Google Lens, Pinterest Visual Search, Trendyol "Fotoğrafla Ara"), telif hakkı koruma sistemlerinde görsel kopya tespiti (Image Deduplication) ve dijital varlık yönetiminde (DAM) kullanılan **Vektör Benzerliği Tabanlı Görsel Arama Motorunu (Visual Nearest Neighbor)** çok modaliteli hibrit gömme (Renk + Doku + Şekil), L2/Cosine mesafe indeksleme ve Top-$K$ en yakın komşu mimarisiyle hayata geçirir.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Kullanıcı bir mobilya mağazasında veya caddede beğendiği bir vazonun fotoğrafını çekip sisteme yüklediğinde:
- Arama motoru piksel matrislerini karşılaştıramaz (açı, ışık ve arka plan farklıdır).
- Metin araması yetersizdir çünkü kullanıcı nesnenin tam adını veya modelini bilmeyebilir.

**Çözüm:** Görselleri yüksek boyutlu bir öznitelik uzayına (**Embedding Space**) yansıtmak ve vektör benzerliği araması ($k$-Nearest Neighbors - $k$-NN) ile bu uzayda sorgu vektörüne en yakın katalog ürünlerini milisaniyeler içinde çekmektir.

---

### 2. Matematiksel Temeller ve Algoritmik Mantık

#### A. Çok Modaliteli Hibrit Vektör Temsili (Hybrid Image Embedding)
Tek bir öznitelik türü görsel aramada yanıltıcıdır:
- Yalnızca renk kullanırsanız kırmızı bir elma ile kırmızı bir araba eşleşir.
- Yalnızca HOG (şekil) kullanırsanız mavi bir vazo ile kırmızı bir vazo ayırt edilemez.

Bu nedenle sistemimiz 3 kritik görsel modaliteyi birleştirir:
1. **Renk Bilgisi (Color Distribution):** HSV uzayında $8 \times 8$ Hue-Saturation 2D Histogramı ($64$ boyut).
2. **Mikro Doku Bilgisi (Texture):** Uniform LBP ($P=8, R=1$) Histogramı ($10$ boyut).
3. **Geometrik Şekil ve Kenar (Shape & Edges):** HOG Gradyan Yönelim Vektörü ($1568$ boyut).

Toplam Vektör Boyutu: $D = 64 + 10 + 1568 = \mathbf{1642\text{ Boyut}}$.

#### B. Alt-Vektör Normalizasyonunun Hayati Önemi (Modalite Dengesi)
Eğer 1568 boyutlu HOG ile 64 boyutlu renk histogramı doğrudan birleştirilirse, HOG'un vektör normu renk histogramını matematiksel olarak ezer (renk bilgisi %95 oranında kaybolur).  
Bunu engellemek için **her modalite önce kendi içinde $L_2$ normalize edilir**, belirlenen katsayılarla ağırlıklandırılır ve birleştirildikten sonra nihai vektör tekrar birim küreye izdüşürülür:

$$v_{\text{renk}} = \frac{h_{\text{renk}}}{\|h_{\text{renk}}\|_2}, \quad v_{\text{doku}} = \frac{h_{\text{lbp}}}{\|h_{\text{lbp}}\|_2}, \quad v_{\text{sekil}} = \frac{h_{\text{hog}}}{\|h_{\text{hog}}\|_2}$$

$$v_{\text{hibrit}} = [w_r \cdot v_{\text{renk}} \,\|\, w_d \cdot v_{\text{doku}} \,\|\, w_s \cdot v_{\text{sekil}}], \quad \hat{v} = \frac{v_{\text{hibrit}}}{\|v_{\text{hibrit}}\|_2}$$

#### C. Mesafe ve Benzerlik Metrikleri

1. **Öklid (L2) Mesafesi:**
   $$d_{L2}(u, v) = \sqrt{\sum_{i=1}^D (u_i - v_i)^2} = \|u - v\|_2$$
2. **Kosinüs Benzerliği (Cosine Similarity):**
   $$S_{\cos}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
   Vektörler birim küreye normalize edildiği için ($\|u\|_2 = 1, \|v\|_2 = 1$):
   $$S_{\cos}(u, v) = u \cdot v \quad (\text{Skaler / Nokta Çarpım})$$
   $$d_{\cos}(u, v) = 1 - S_{\cos}(u, v)$$
3. **L2 ile Kosinüs Mesafesi Arasındaki Eşdeğerlik İspatı:**
   Birim vektörler için:
   $$\|u - v\|_2^2 = (u - v) \cdot (u - v) = \|u\|_2^2 + \|v\|_2^2 - 2 (u \cdot v) = 1 + 1 - 2 S_{\cos}(u, v) = 2(1 - S_{\cos}(u, v))$$
   $$\mathbf{\|u - v\|_2^2 = 2 \cdot d_{\cos}(u, v)}$$
   Bu eşitlik, birim vektörlerde **L2 sıralaması ile Kosinüs sıralamasının matematiksel olarak birebir aynı olacağını** kanıtlar!

#### D. $k$-NN Arama Mekanizması ve Boyutun Laneti (Curse of Dimensionality)
- Düşük boyutlu uzaylarda ($D < 20$) KD-Tree ve BallTree logaritmik $\mathcal{O}(\log N)$ arama sunar.
- Ancak görsel özniteliklerin yüksek boyutunda ($D = 1642$) ağaç yapıları çöker ve kaba kuvvet (Brute-Force / Flat Scan) aramasına geriler.
- Kaba kuvvet araması $\mathcal{O}(N \cdot D)$ karmaşıklığına sahiptir ve matris çarpımı (BLAS / SIMD / GPU) ile binlerce görsel için birkaç milisaniyede tamamlanır.

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Vektörleri Normalize Etmeden Birleştirmek:**
   Farklı boyut ve genlikteki özniteliklerin doğrudan birleştirilmesi, büyük boyutlu özniteliğin (HOG) diğer tüm bilgileri yok etmesine neden olur.
2. **Arka Plan Gürültüsü Baskısı:**
   Görselin arka planı nesneden çok daha fazla yer kaplıyorsa, sistem nesneyi değil arka planı eşleştirmeye başlar. (Bu nedenle Day 15'teki GrabCut ile ön plan segmentasyonu, arama motorunun başarısını dramatik şekilde artırır!).

---

## 📌 Mimari Tasarım ve Akış Şeması

```
[Katalog Görselleri] (N Adet)                   [Sorgu Görseli] (1 Adet)
        │                                                │
        ▼                                                ▼
┌───────────────────────────────┐              ┌───────────────────────────────┐
│     GorselVektorCikarici      │              │     GorselVektorCikarici      │
│  - HSV 2D Renk Hist (64-B)    │              │  - HSV 2D Renk Hist (64-B)    │
│  - Uniform LBP Doku (10-B)    │              │  - Uniform LBP Doku (10-B)    │
│  - HOG Şekil Gradyan (1568-B) │              │  - HOG Şekil Gradyan (1568-B) │
│  - Alt-Normalizasyon + L2     │              │  - Alt-Normalizasyon + L2     │
└───────────────┬───────────────┘              └───────────────┬───────────────┘
                │                                              │
                ▼                                              ▼
        Katalog Matrisi (N, 1642)                     Sorgu Vektörü (1642,)
                │                                              │
                └──────────────────────┬───────────────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │       GorselAramaMotoru       │
                       │    Vektörize Nokta Çarpımı    │
                       │   S_cos = Matrix @ Vektör     │
                       │    Mesafeye Göre Sıralama     │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     Top-K Arama Sonuçları     │
                       │   (#1, #2, #3, #4, #5 Eşleşme)│
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │      AramaGorsellestirici     │
                       │     Renkli Çerçeveli Panel    │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                        [ciktilar/gorsel_arama_raporu.png]
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

1. **[`src/vektor_cikarici.py`](./src/vektor_cikarici.py):**
   - `GorselVektorCikarici`: Renk, doku ve HOG gradyanlarını bağımsız olarak normalize edip ağırlıklı birleştiren hibrit gömme motoru.
2. **[`src/knn_arama_motoru.py`](./src/knn_arama_motoru.py):**
   - `GorselAramaMotoru`: Katalog matrisini tutan, L2, Kosinüs ve Manhattan metrikleriyle Top-$K$ en yakın aramayı çalıştıran sınıf.
   - `AramaSonucu`: Eşleşme sırası, katalog etiketi, mesafe ve benzerlik yüzdesini taşıyan veri yapısı.
3. **[`src/gorsellestirici.py`](./src/gorsellestirici.py):**
   - `AramaGorsellestirici`: Sorgu görseli ve bulunan Top-$K$ katalog eşleşmesini benzerlik derecesine göre yeşil/turuncu/kırmızı çerçevelerle birleştiren raporlama aracı.
4. **[`ana_akis.py`](./ana_akis.py):**
   - 4 kategoride (vazo, kumaş, yıldız, ahşap) 16 ürünlük sentetik kataloğu indeksleyen ve seramik vazo sorgusu için Top-5 araması yapan yürütücü betik.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
==============================================================================
>>> AŞAMA 1: Sentetik Katalog ve Çok-Kategorili Görsellerin Üretilmesi
==============================================================================
[+] Katalogdaki Toplam Ürün Adedi: 16 adet
[+] Ürün Kategorileri            : ahsap, kumas, vazo, yildiz

==============================================================================
>>> AŞAMA 2: Çok Modaliteli Hibrit Vektörlerin İndekslenmesi
==============================================================================
[V] Başarıyla İndekslenen Görsel : 16 adet
[V] Hibrit Vektör Boyutu (D)    : 1642 boyut (Renk 64 + LBP 10 + HOG 1568)
[V] Toplam İndeksleme Süresi    : 77.62 ms (4.85 ms/görsel)

==============================================================================
>>> AŞAMA 3: Görsel Sorgu ve k-NN (Kosinüs Benzerliği) ile Top-5 Arama
==============================================================================
[+] Sorgu Tamamlanma Süresi     : 7.054 ms (Mikrosaniyeler mertebesinde!)

--- TOP-5 EŞLEŞME TABLOSU (Kosinüs Benzerliği) ---
Sıra   | Katalog Etiketi           | Cosine Mesafe  | Benzerlik (%)  | Kategori Eşleşti mi?
------------------------------------------------------------------------------
#1     | vazo_koyu_seramik         | 0.4013         | %59.9          | EVET (Doğru)
#2     | vazo_kirmizi_klasik       | 0.4095         | %59.0          | EVET (Doğru)
#3     | vazo_altin_bantli         | 0.4310         | %56.9          | EVET (Doğru)
#4     | vazo_terracotta_ince      | 0.4390         | %56.1          | EVET (Doğru)
#5     | ahsap_ceviz_kaplama       | 0.6931         | %30.7          | HAYIR
------------------------------------------------------------------------------

[+] L2 Metriği Kontrolü: 1. Sıradaki Eşleşme -> vazo_koyu_seramik (L2 Mesafe: 0.8959)

==============================================================================
>>> AŞAMA 4: Görsel Arama Rapor Çizelgesinin Kaydedilmesi
==============================================================================
[V] Görsel arama raporu kaydedildi: gorsel_arama_raporu.png
[V] Kayıt Konumu: day-17-visual-nearest-neighbor/ciktilar/gorsel_arama_raporu.png

[V] Day 17: Vektör Benzerliği Tabanlı Görsel Arama başarıyla tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Benzerlik Eşiği Tabanlı Filtreleme ve Arama Hassasiyet Skoru (Top-K Precision@K)**

Görsel arama motorlarında kullanıcının aradığı nesne katalogda hiç olmayabilir. Bu durumda alakasız ürünlerin listelenmesini önlemek için bir **benzerlik eşiği (threshold)** koymamız ve kategori doğruluğunu (**Precision@K**) ölçmemiz gerekir.

### Görev Tanımı:
[`src/knn_arama_motoru.py`](./src/knn_arama_motoru.py) sınıfına şu iki metodu eklemeni bekliyorum:

```python
def esik_ile_filtrele(
    self,
    sorgu_gorseli_bgr: np.ndarray,
    min_benzerlik_yuzdesi: float = 50.0,
    metrik: str = "cosine"
) -> List[AramaSonucu]:
    """Yalnızca belirlenen benzerlik yüzdesinin üzerindeki katalog ürünlerini döndürür."""

@staticmethod
def hassasiyet_hesapla(
    sonuclar: List[AramaSonucu],
    hedef_kategori_oneki: str
) -> float:
    """Top-K sonuçlar içerisindeki doğru kategori oranını (Precision@K) hesaplar (0.0 - 1.0)."""
```

### Beklenen Kurallar:
1. `esik_ile_filtrele`, benzerlik oranı `min_benzerlik_yuzdesi` değerinin altında kalan tüm sonuçları budamalıdır.
2. `hassasiyet_hesapla`, dönen sonuçların kaç tanesinin etiketi `hedef_kategori_oneki` (örneğin `"vazo"`) ile başlıyorsa bunu toplam sonuç adedine bölerek kesinlik oranını vermelidir.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> İki öznitelik vektörü $L_2$ normalize edildiğinde ($\|u\|_2 = 1, \|v\|_2 = 1$), Öklid mesafesi ile Kosinüs mesafesi arasında kurduğumuz matematiksel bağıntı:  
> $$\|u - v\|_2^2 = 2 \cdot (1 - S_{\cos}(u, v))$$  
> Bu bağıntıya göre:  
> 1. Bir sorgu vektörü için katalogdaki $N$ adet ürün **Öklid mesafesine ($d_{L2}$)** göre sıralandığında elde edilen sıralama ile **Kosinüs mesafesine ($d_{\cos}$)** göre sıralandığında elde edilen sıralamanın **birebir aynı olması matematiksel olarak garanti midir? Neden?**  
> 2. Peki vektörler $L_2$ normalize **edilmeseydi**, bu iki metriğin sıralaması neden birbirinden tamamen farklılaşırdı?

---

## 📂 Dizin Yapısı

```
day-17-visual-nearest-neighbor/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (opencv, numpy, scikit-learn, scikit-image, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve görsel arama üretim akışı
├── ciktilar/                   # Üretilen görsel arama paneli
│   └── gorsel_arama_raporu.png
├── src/
│   ├── __init__.py
│   ├── vektor_cikarici.py      # Çok modaliteli hibrit öznitelik çıkarıcı
│   ├── knn_arama_motoru.py     # k-NN görsel arama ve indeksleme motoru
│   └── gorsellestirici.py      # Görsel arama rapor çizelge motoru
└── testler/
    └── test_gorsel_arama.py    # 7 adet birim testi (7 passed in 1.31s)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışı Çalıştırma
```bash
python ana_akis.py
```

### 3. Testleri Koşma
```bash
python -m pytest testler/test_gorsel_arama.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
