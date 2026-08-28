# Day 18: Etiketsiz Görsellerin Otomatik Kümelenmesi (Image Clustering)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.24+-orange.svg?style=flat-square)](https://scikit-image.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; e-ticaret platformlarında etiketlenmemiş ürün kataloglarının otomatik kategorizasyonu, dijital varlık yönetiminde (DAM) benzer görsellerin gruplanması ve veri kalitesi denetiminde gürültülü/bozuk görsellerin elenmesi (Anomaly Detection) için **K-Means**, **DBSCAN** ve **Hiyerarşik (Agglomerative) Kümeleme** algoritmalarını ve **Silhouette Analizini** üretim kalitesinde sunar.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Modern yapay zeka sistemlerinde verilerin %90'ından fazlası **etiketlenmemiştir (unlabeled)**:
- Bir pazar yerine (ör. Trendyol, Amazon) satıcılar tarafından her gün on binlerce yeni ürün görseli yüklenir; bunları manuel etiketlemek aylar sürer ve yüksek maliyetlidir.
- Ürün kataloglarında benzer varyantların (farklı açılardan çekilmiş aynı vazo, aynı kumaşın renkleri) tespit edilip gruplanması gerekir.
- Bozuk, rastgele yüklenmiş veya spam görsellerin veri setinden izole edilmesi zorunludur.

**Çözüm:** Görselleri çok modaliteli embedding uzayına yansıtıp denetimsiz kümeleme (Unsupervised Clustering) algoritmalarıyla doğal öbeklerine ayırmak ve Silhouette analiziyle optimal küme sayısını ($K$) matematiksel olarak belirlemektir.

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **DBSCAN** | *Density-Based Spatial Clustering* | Yoğunluk temelli çalışan, küme sayısını önceden bilmeyi gerektirmeyen ve rastgele şekilli kümeler ile aykırı değerleri tespit eden algoritma. |
| **Silüet Skoru** | *Silhouette Coefficient* | Küme içi benzerlik ile komşu kümelere olan uzaklığı kıyaslayarak kümeleme kalitesini $[-1, 1]$ aralığında ölçen metrik. |
| **Çekirdek Noktası (Core Point)** | *DBSCAN Core Point* | $\epsilon$ yarıçaplı komşuluğunda en az `MinPts` kadar komşusu bulunan yoğun bölge noktası. |
| **Gürültü Noktası (Noise)** | *DBSCAN Noise Point* | Hiçbir yoğun kümeye dahil edilemeyen ve aykırı değer kabul edilen izole nokta. |

---

## 2. Matematiksel Temeller ve Algoritmik Mantık

#### A. K-Means Kümeleme (Lloyd Algoritması & WCSS)
K-Means, veri noktalarını önceden tanımlanmış $K$ adet kümeye ayırırken **Küme İçi Kareler Toplamını (Within-Cluster Sum of Squares - WCSS)** yani **ataleti (inertia)** minimize eder:

$$J = \sum_{k=1}^K \sum_{x_i \in S_k} \|x_i - \mu_k\|^2$$

Burada $\mu_k$, $S_k$ kümesinin ağırlık merkezidir (centroid):
$$\mu_k = \frac{1}{|S_k|} \sum_{x_i \in S_k} x_i$$

- **Lloyd Döngüsü:**
  1. *Atama Adımı:* Her nokta kendisine en yakın centroid'e atanır:
     $$c_i = \arg\min_k \|x_i - \mu_k\|_2$$
  2. *Güncelleme Adımı:* Centroid'ler kümeye atanan noktaların ortalaması olarak güncellenir.
- **K-Means++ Başlatma:** Rastgele merkez seçimi yerine ilk merkezi rastgele seçip sonraki merkezleri mevcut merkezlere olan uzaklığın karesi ($D(x)^2$) ile orantılı olasılıkla seçerek yerel minimumlara takılmayı dramatik biçimde azaltır.

---

#### B. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
K-Means küresel (spherical) kümeler varsayar ve her noktayı zorla bir kümeye atar. **DBSCAN** ise yoğunluk tabanlıdır ve iki temel parametreyle çalışır:
1. $\epsilon$ (Epsilon): Komşuluk yarıçapı.
2. $minPts$: Bir noktanın çekirdek (core point) sayılması için $\epsilon$ yarıçapı içinde bulunması gereken minimum komşu sayısı.

- **Nokta Sınıflandırması:**
  - **Çekirdek Nokta (Core Point):** $|\mathcal{N}_\epsilon(p)| \ge minPts$
  - **Sınır Noktası (Border Point):** $|\mathcal{N}_\epsilon(p)| < minPts$, ancak bir çekirdek noktanın $\epsilon$ komşuluğunda yer alır.
  - **Gürültü / Aykırı Nokta (Noise/Outlier):** Ne çekirdek ne de sınır noktasıdır (Etiket: $-1$).
- **Görsel Alanındaki Gücü:** Bozuk veya aykırı görselleri kümelere dahil etmez, doğrudan "gürültü" olarak dışlar.

---

#### C. Hiyerarşik (Agglomerative) Kümeleme
Aşağıdan yukarıya (bottom-up) çalışan hiyerarşik bir yaklaşımdır. Başlangıçta her görsel kendi başına 1 kümedir. Her adımda en yakın iki küme birleştirilir.
- **Bağlantı Kriterleri (Linkage Criteria):**
  - **Average Linkage:** İki küme arasındaki ortalama mesafe:
    $$D(A, B) = \frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} d(a, b)$$
  - **Ward Linkage:** Kümelerin birleşmesiyle WCSS'teki artışı minimize eder.

---

#### D. Kümeleme Kalitesi ve Silhouette Skoru
Etiket bulunmayan bir ortamda kümelemenin kalitesini nasıl doğrularız?

Her $i$ örneği için:
1. **$a(i)$ (Küme İçi Uyum):** $i$'nin kendi kümesindeki diğer tüm noktalara olan ortalama mesafesi.
2. **$b(i)$ (En Yakın Komşu Küme Ayrışması):** $i$'nin kendisine en yakın diğer kümedeki tüm noktalara olan ortalama mesafesi.

**Silhouette Katsayısı:**
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad -1 \le s(i) \le 1$$

| $s(i)$ Değeri | Anlamı |
|---|---|
| $\approx +1.0$ | Örnek kendi kümesine çok yakın, komşu kümelerden çok uzaktır (Mükemmel Kümeleme). |
| $\approx 0.0$ | Örnek iki kümenin sınır çizgisindedir (Belirsizlik). |
| $\approx -1.0$ | Örnek yanlış kümeye atanmıştır (Komşu kümeye daha yakındır). |

---

### 3. Yüksek Boyutlu Görsel Embedding'lerinde Kritik Darboğazlar
1. **Boyutun Laneti (Curse of Dimensionality):** Yüksek boyutta (ör. $D > 100$) Öklid mesafesi noktalar arasında homojenleşir (en uzak ve en yakın mesafe birbirine yaklaşır). Bu nedenle görsel embedding'lerinde **Kosinüs Mesafesi ($1 - \cos(\theta)$)** veya **L2 normalize edilmiş vektörler üzerinde Öklid** kullanılmalıdır.
2. **Modalite Ağırlık Dengesizliği:** Renk histogramı (64D) ile doku (10D) doğrudan birleştirilirse, yüksek boyutlu olan modalite kümeleme merkezlerini tekeline alır. Her modalite alt-vektör düzeyinde normalize edilmelidir.

---

## 🛠️ Dizin Yapısı

```
day-18-image-clustering/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # Bağımlılıklar (scikit-learn, opencv, matplotlib, numpy, pytest)
├── ana_akis.py                      # Uçtan uca sentetik veri üretimi, kümeleme ve kıyaslama
├── README.md                        # Detaylı mentorluk ve teknik dokümantasyon
├── src/
│   ├── __init__.py                  # Modül dışa aktarımları
│   ├── vektor_cikarici.py           # Renk + Doku + Şekil hibrit L2 normalize embedding çıkarıcı
│   ├── kumeleme_motoru.py           # K-Means, DBSCAN, Agglomerative ve metrik hesaplayıcı
│   └── gorsellestirici.py           # PCA 2B, Silhouette analizi ve galeri rapor motoru
├── testler/
│   ├── __init__.py
│   └── test_kumeleme.py             # 8 adet kapsamlı birim test
└── ciktilar/
    └── kumeleme_raporu.png          # 4 panelli yüksek çözünürlüklü teşhis çizelgesi
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

Konsolda K-Means optimal K taraması, DBSCAN gürültü ayrıştırması ve kıyaslama tablosu listelenecek, `ciktilar/kumeleme_raporu.png` dosyası otomatik üretilecektir.

### 3. Testlerin Çalıştırılması
```bash
pytest testler -v
```

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/kumeleme_motoru.py` içerisine DBSCAN için otomatik $\epsilon$ (epsilon) kestirimi yapan bir `k_mesafe_grafigi_hesapla(X, k=4)` metodu ekleyin.
- **Detay:** Her veri noktasının en yakın $k$. komşusuna olan mesafesini bulun, mesafeleri küçükten büyüğe sıralayın ve eğrinin dirsek (elbow) noktasını otomatik saptayarak önerilen $\epsilon$ değerini döndürün.

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Görüntü özniteliklerini kümelemede K-Means algoritması yerine DBSCAN ne zaman tercih edilmelidir ve K-Means'in küresel (spherical) küme varsayımı hangi durumlarda başarısız olur?

> **Mentor Cevabı:**
> 1. **Dışsal / Karmaşık Şekilli Kümeler (Non-convex Clusters):** K-Means her kümenin Öklid uzayında küresel (convex/spherical) ve benzer varyansa sahip olduğunu varsayar. Eğer görüntü öznitelik uzayı hilal, halka veya manifold şeklindeyse K-Means kümeleri yanlış böler; yoğunluk tabanlı DBSCAN ise keyfi geometrik şekilleri başarıyla tespit eder.
> 2. **Gürültü ve Aykırı Değer İzolasyonu:** K-Means tüm verileri mutlaka bir merkeze atamak zorundadır, bu da aykırı görüntülerin küme merkezlerini saptırmasına yol açar. DBSCAN ise yoğunluk eşiğini (`min_samples`) geçemeyen aykırı noktaları doğrudan gürültü (`-1`) olarak izole eder.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
