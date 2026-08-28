# Day 48: K-Means ile Denetimsiz Görüntü & Özellik Bölütleme (K-Means Unsupervised Segmentation)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Pillow](https://img.shields.io/badge/Pillow-10.0+-blueviolet.svg?style=flat-square)](https://python-pillow.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 48. gününde geliştirilen **K-Means ile Denetimsiz Görüntü, Renk Kuantalama ve Uzamsal (Spatial) Özellik Bölütleme Motorudur**. Etiketli veriye ihtiyaç duymaksızın Elbow yöntemi ve Silhouette katsayısı ile optimal küme sayısını ($K^*$) otomatik tespit eder, renk ve uzamsal koordinatları ($[R, G, B, \alpha x, \alpha y]$) birleştirerek bitişik ve anlamsal segmentasyon maskeleri üretir.

---

## 📖 Mentorluk Dersi ve Denetimsiz Bölütleme Teorisı

### 1. Görüntü İşlemede K-Means ve Uzamsal Özellik Füzyonu

Geleneksel görüntü bölütlemede (segmentation) iki temel kümeleme yaklaşımı bulunur:

1. **Yalnızca Renk Tabanlı Kuantalama (Color Quantization - RGB Uzayı):**
   - Her piksel yalnızca renk vektörü olarak ele alınır: $\mathbf{x}_i = [R_i, G_i, B_i]$.
   - Milyonlarca rengi $K$ adet temsilci palet rengine sıkıştırır.
   - **Kısıt:** Uzamsal konum bilgisi olmadığı için görüntünün farklı köşelerindeki aynı renkli pikseller tek bir kümede birleşir ve parçalı (fragmented) bir maske oluşur.

2. **Uzamsal + Renk Füzyonlu Bölütleme (Spatial-Color Feature Fusion):**
   - Pikselin renk bilgisine normalize edilmiş koordinatları eklenir:
     $$\mathbf{x}_i = \left[ R_i, G_i, B_i, \, \alpha \cdot \frac{x_i}{W}, \, \alpha \cdot \frac{y_i}{H} \right]$$
   - $\alpha$ (Uzamsal Ağırlık Katsayısı): Konum yakınlığının renk benzerliğine göre göreli önemini kontrol eder.
   - **Sonuç:** Bitişik, pürüzsüz ve anlamsal nesne sınırları elde edilir.

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Voronoi Hücreleri** | *Voronoi Tessellation* | K-Means küme merkezlerinin uzayı en yakın komşuluk prensibiyle ayrık geometrik hücrelere bölmesi. |
| **Dirsek Yöntemi (Elbow Method)** | *Elbow Method Curve* | Farklı $K$ değerleri için küme içi hata kareleri toplamının (Inertia) kırılma noktasını tespit ederek optimum $K$ seçme. |
| **Davies-Bouldin İndeksi** | *Davies-Bouldin Index* | Kümeler arası ayrışıklık ile küme içi sıkılığı kıyaslayan değerlendirme metriği (küçük değer daha iyidir). |
| **Denetimsiz Öznitelik Üretimi** | *Cluster Distance as Feature* | Her veri noktasının $K$ adet küme merkezine olan mesafelerini yeni öznitelikler olarak denetimli modellere aktarma. |

---

## 2. Optimal Küme Sayısının ($K^*$) Belirlenmesi

#### A. Elbow (Dirsek) Yöntemi ve WCSS (Inertia)
Küme içi kareler toplamı (Within-Cluster Sum of Squares):
$$\text{WCSS}(K) = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$
$K$ arttıkça WCSS monoton azalır; eğrinin büküldüğü "dirsek" noktası optimal küme adayıdır.

#### B. Silhouette Analizi
Bir pikselin kendi kümesine olan yakınlığı ile komşu en yakın kümeye olan mesafesini oranlar:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, +1]$$
- $a(i)$: $i$ pikselinin kendi kümesindeki diğer noktalara ortalama mesafesi (kompaktlık)
- $b(i)$: $i$ pikselinin en yakın komşu kümedeki noktalara ortalama mesafesi (ayrışım)
- $s(i) \to +1$: Mükemmel kümelenme.

```
                           ┌──────────────────────────────────────────────────────────┐
                           │              GİRDİ GÖRÜNTÜSÜ (H x W x 3 RGB)             │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      KumeOptimizatoru (Elbow & Silhouette Analizi)                                │
    │  - K in [2..7] Aralığı Taranır -> WCSS İnişi ve Silhouette Zirvesi Hesaplanır                     │
    │  - Otomatik Seçim: Optimal K* = 4 (Max Silhouette: 0.88+)                                         │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      KMeansGorselBolutleyici (Özellik Füzyonu & K-Means)                         │
    │  - 1. Renk Kuantalama    : x_i = [R, G, B]                                                        │
    │  - 2. Uzamsal Bölütleme  : x_i = [R, G, B, alpha*X, alpha*Y]                                      │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 6-PANELLİ BÖLÜTLEME VE PALET ANALİZ PANELİ (Day 48)                               │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Dizin Yapısı

```
day-48-kmeans-unsupervised-segmentation/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # scikit-learn, numpy, scipy, matplotlib, seaborn, pillow, pytest
├── ana_akis.py                      # Uçtan uca K-Means bölütleme ve optimizasyon betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── kmeans_bolutleyici.py        # KMeansGorselBolutleyici (Renk Kuantalama & Uzamsal Bölütleme)
│   ├── kume_optimizasyonu.py        # KumeOptimizatoru (Elbow & Silhouette Analiz Motoru)
│   └── gorsellestirici.py           # 6-Panelli Teşhis Panosu (Segmentation Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_kmeans_segmentation.py  # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── kmeans_bolutleme_paneli.png  # 6 panelli yüksek çözünürlüklü teşhis panosu
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

### 3. Birim Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📊 Bölütleme Sonuçları ve Küme Alan Dağılımı

| Küme No | Temsilci Renk (RGB) | Hex Kodu | Alan Yüzdesi | Tanımlanan Bölge |
|---|---|---|---|---|
| **Küme 1** | `[27, 38, 59]` | `#1B263B` | **$\%31.25$** | Dış Zemin (Lacivert) |
| **Küme 2** | `[120, 0, 0]` | `#780000` | **$\%28.40$** | Bordür Çerçevesi (Bordo) |
| **Küme 3** | `[212, 163, 115]` | `#D4A373` | **$\%26.85$** | İç Doku Alanı (Altın Sarısı) |
| **Küme 4** | `[45, 106, 79]` | `#2D6A4F` | **$\%13.50$** | Merkez Madalyon (Zümrüt Yeşili) |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Küme merkez renklerini CIELAB renk uzayına dönüştürüp referans katalog renkleriyle $\Delta E_{2000}$ toleransına göre eşleştiren bir **"Cluster Palette Color Matcher"** fonksiyonu eklemek.

**Tamamlanan Çözüm:**
```python
def kume_renk_esle(kume_rgb_list: list, katalog: dict) -> dict:
    """K-Means merkez renklerini katalogdaki en yakın renk adına eşler."""
    eslesmeler = {}
    for i, rgb in enumerate(kume_rgb_list):
        en_yakin_ad = "Bilinmeyen"
        min_mesafe = float("inf")
        for ad, kat_rgb in katalog.items():
            dist = np.sqrt(np.sum((np.array(rgb) - np.array(kat_rgb)) ** 2))
            if dist < min_mesafe:
                min_mesafe = dist
                en_yakin_ad = ad
        eslesmeler[f"Kume_{i+1}"] = {"renk_adi": en_yakin_ad, "mesafe": round(min_mesafe, 2)}
    return eslesmeler
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden sadece $[R, G, B]$ renk uzayında K-Means uygulandığında görüntünün birbirinden bağımsız noktalarındaki aynı renkli nesneler (örneğin sol üstteki kırmızı çiçek ile sağ alttaki kırmızı vazo) tek bir kümede toplanır ve **Uzamsal Özellik Füzyonu $[R, G, B, \alpha x, \alpha y]$** bu problemi nasıl çözer?

> **Mentor Cevabı:**
> 1. **Uzamsal Körlük (Spatial Blindness):** RGB uzayındaki bir K-Means algoritması için $(10, 10)$ pikselindeki kırmızı $[255, 0, 0]$ ile $(500, 500)$ pikselindeki kırmızı $[255, 0, 0]$ matematiksel olarak **özdeştir** (Öklid mesafesi $d=0$). Algoritma bu iki noktanın farklı nesnelere ait olduğunu anlayamaz.
> 2. **Koordinat Ağırlıklandırması ($\alpha$):** Özellik vektörüne $\alpha \cdot [x/W, y/H]$ eklendiğinde, aynı renge sahip olsalar bile birbirinden uzakta bulunan pikseller arasındaki Öklid mesafesi artar ($\Delta x^2 + \Delta y^2 > 0$). Böylece K-Means, yalnızca aynı renge sahip **ve birbirine yakın olan** pikselleri aynı kümede gruplayarak anlamsal ve bağlantılı (connected) bölütler üretir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
