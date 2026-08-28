# Day 11: Görsellerden Baskın Renk Paletinin Çıkarılması (Dominant Color Extractor & K-Means)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-f89939.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; e-ticaret katalog aramalarında, tekstil/halı üretiminde ve kullanıcı arayüzü (UI/UX) tasarımında görsellerin karakteristik renk kimliğini saptamak için **K-Means denetimsiz makine öğrenmesi algoritması ile piksel uzayını kümeleyen, her rengin yüzdesel ağırlığını, RGB ve HEX kodlarını çıkaran ve görüntüyü $K$ renge indirgeyen (Color Quantization)** üretim seviyesinde bir renk analiz motorudur.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Modern bir e-ticaret platformunda (örneğin halı, mobilya veya moda) kullanıcılar genellikle şu aramaları yapar:
> *"Bana 'Hardal Sarısı ve Gece Mavisi' tonlarında klasik bir halı öner!"*

Fotoğraftaki milyonlarca pikselin tek tek ortalamasını almak felaketle sonuçlanır: Sarı, mavi, kırmızı ve krem renkleri karıştırırsanız ortaya çamurlu anlamsız bir kahverengi çıkar!
Görseli tanımlayan şey ortalama renk değil; **görüntüyü oluşturan baskın renk kümelerinin (clusters) merkezleri ve yüzdesel oranlarıdır.**

**Kullanım Alanları:**
- **Renk Bazlı Arama ve Öneri Motorları:** İki ürünün renk paleti benzerliğini vektörel olarak kıyaslamak.
- **Otomatik UI Tema Üretimi:** Spotify veya Netflix gibi albüm/film kapağının ana renklerine göre arka plan rengini dinamik değiştirmek.
- **Görüntü Sıkıştırma ve Kuantizasyon:** 16.7 milyon olası rengi sadece $K=5$ renge indirgeyerek bant genişliğinden tasarruf etmek.

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **K-Means Kümeleme** | *K-Means Clustering* | Piksel RGB/Lab vektörlerini en yakın $K$ adet merkez etrafında toplayarak baskın renk kümelerini bulan denetimsiz algoritma. |
| **Küme Merkezi (Centroid)** | *Cluster Centroid* | Bir renk kümesine atanan tüm piksellerin ortalama renk vektörü. |
| **Renk Kuantizasyonu** | *Color Quantization* | Görüntüdeki milyonlarca farklı rengi görsel kaliteyi koruyarak $K$ adet temsilci renge indirgeme işlemi. |
| **Eylemsizlik (Inertia)** | *Cluster Inertia (WCSS)* | Her pikselin atandığı küme merkezine olan karesel mesafelerinin toplamı; $K$ seçiminde dirsek noktasını belirler. |
| **Uzamsal Alt-Örnekleme** | *Spatial Subsampling* | K-Means hesaplama maliyetini düşürmek için görseli orantılı küçülterek kümeleme hızını onlarca kat artırma tekniği. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. Renk Uzayında K-Means Kümeleme (Lloyd's Algoritması)
Bir görüntüdeki her piksel $p_i = (R_i, G_i, B_i)$, 3 boyutlu uzayda bir veri noktasıdır.
K-Means, küme içi kareler toplamını (**WCSS / Within-Cluster Sum of Squares / Inertia**) minimize eden $K$ adet küme merkezini ($\mu_1, \mu_2, \dots, \mu_K$) bulur:

$$J = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|_2^2$$

1. **Başlatma (Initialization):** $k$-means++ yöntemiyle uzayda birbirine en uzak $K$ başlangıç noktası seçilir.
2. **Atama Adımı:** Her piksel kendisine en yakın merkeze atanır:
   $$c_i = \arg\min_k \|x_i - \mu_k\|$$
3. **Güncelleme Adımı:** Küme merkezleri, o kümeye düşen piksellerin ağırlıklı ortalamasıyla güncellenir:
   $$\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$$
4. Merkezler değişmeyene (yakınsayana) kadar adımlar tekrarlanır.

#### B. Yüzdesel Ağırlık Dağılımı ve HEX Dönüşümü
Her kümeye düşen piksel sayısı $N_k$ hesaplanır ve toplam piksel sayısına ($N_{\text{toplam}}$) bölünerek yüzde elde edilir:

$$\text{Yüzde}_k = \frac{N_k}{N_{\text{toplam}}} \times 100$$

Her RGB merkez vektörü standart web ve grafik formatına çevrilir:
$$\text{HEX} = \text{"\#\{R:02X\}\{G:02X\}\{B:02X\}"}$$

#### C. Renk Kuantizasyonu (Color Quantization)
Görüntüdeki her piksel $x_i$, ait olduğu kümenin merkezi $\mu_{c_i}$ ile değiştirilir:
$$I_{\text{quant}}(x, y) = \mu_{c(x, y)}$$
Böylece binlerce farklı ton içeren zengin bir fotoğraf, tam $K$ adet saf renkten oluşan grafiksel bir poster görünümüne kavuşur.

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Büyük Çözünürlüklü Görüntülerde Bellek ve CPU Şişmesi:**
   $4000 \times 3000$ (12 Megapiksel) bir fotoğrafta $12.000.000$ adet 3B nokta vardır. Bu veriyi doğrudan K-Means'e sokmak dakikalarca bekletebilir ve bellek tüketimini patlatır.
   - *Endüstriyel Çözüm:* K-Means küme merkezlerini öğrenirken veriyi alt örneklemek (örneğin rastgele $20.000$ piksel ile fit etmek) yeterlidir. Renk merkezleri öğrendikten sonra tüm pikseller saniyeler içinde etiketlenir.
2. **Arka Planın Paleti Domine Etmesi:**
   Beyaz bir zemin üzerine konulmuş küçük bir kırmızı ayakkabı fotoğrafında beyaz renk %85 çıkacaktır. Ürün odaklı palet çıkarırken arka planı (segmentasyon maskesiyle) çıkarmak gerekebilir.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
                      Girdi Görüntüsü (H x W x 3, BGR)
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │     BGR -> RGB Dönüşümü     │
                      │    (N x 3 Piksel Matrisi)   │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    BaskinRenkCikarici       │
                      │ (K-Means Kümeleme, K=5)     │
                      └──────────────┬──────────────┘
                                     │
        ┌────────────────────────────┴────────────────────────────┐
        ▼                                                         ▼
[Baskın Renk Merkezleri]                                  [Piksel Etiketleri]
- RGB: (200, 60, 40)                                      - Her pikselin kümesi
- HEX: #C83C28                                            - Kuantize Görüntü
- Yüzde: %40.31                                             Oluşturma
        │                                                         │
        └────────────────────────────┬────────────────────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │    PaletGorsellestirici     │
                      │   - Orijinal Görsel         │
                      │   - Kuantize Görsel         │
                      │   - Orantısal Renk Şeridi   │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      [ciktilar/baskin_renk_paleti.png]
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

1. **[`src/renk_kumeleyici.py`](./src/renk_kumeleyici.py):**
   - `RenkBilgisi`: Renk RGB, HEX kodu, yüzdesi ve piksel adedini tutan dondurulmuş veri sınıfı.
   - `BaskinRenkCikarici`: K-Means kümelemesi, hızlı alt örnekleme, azalan sırada palet çıkarma ve görüntü kuantizasyon motoru.
2. **[`src/palet_gorsellestirici.py`](./src/palet_gorsellestirici.py):**
   - `PaletGorsellestirici`: Orijinal görseli, kuantize görseli ve yüzdesel orantılı yatay renk şeridini (Color Swatch Bar) metin kontrastına duyarlı olarak diske çizer.
3. **[`ana_akis.py`](./ana_akis.py):**
   - Gece mavisi, hardal sarısı, kiremit kırmızısı, adaçayı yeşili ve fildişi renklerinden oluşan sentetik halı motifini analiz eden konsol yürütücüsü.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
==========================================================================
>>> AŞAMA 1: Sentetik Tekstil Görselinin Üretimi
==========================================================================
[+] Çözünürlük         : 300 x 300 (90,000 piksel)
[+] Kanal Sayısı       : 3 (BGR)

==========================================================================
>>> AŞAMA 2: K-Means (K=5) ile Baskın Renklerin Kümelenmesi
==========================================================================
Sıra  | HEX Kodu   | RGB Değeri         | Yüzde (%)  | Piksel Adedi
--------------------------------------------------------------------------
#1    | #C83C28    | (200, 60, 40)      | %40.31     | 36,282
#2    | #141E50    | (20, 30, 80)       | %28.68     | 25,815
#3    | #DCB41E    | (220, 180, 30)     | %19.22     | 17,294
#4    | #508C3C    | (80, 140, 60)      | %9.92      | 8,926
#5    | #F5F0DC    | (245, 240, 220)    | %1.87      | 1,683
--------------------------------------------------------------------------
[V] Toplam Ağırlık Tutarlılığı: %100.00

==========================================================================
>>> AŞAMA 3: Görüntü Renk Kuantizasyonu (Color Quantization)
==========================================================================
[+] Orijinal Benzersiz Renk Adedi : 5
[+] Kuantize Benzersiz Renk Adedi : 5 (Tam 5 baskın renge indirgendi!)

==========================================================================
>>> AŞAMA 4: Görsel Rapor ve Renk Şeridinin Kaydedilmesi
==========================================================================
[V] Renk paleti çizelgesi başarıyla kaydedildi: baskin_renk_paleti.png
[V] Kayıt Konumu: day-11-dominant-color-extractor/ciktilar/baskin_renk_paleti.png

[V] Day 11: Baskın Renk Paletinin Çıkarılması başarıyla tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Optimum K Belirleyici (Dirsek Yöntemi / Elbow Method)**

Her görüntü için $K=5$ ideal olmayabilir; minimalist bir logoda $K=2$ yeterliyken, karmaşık bir manzarada $K=8$ gerekebilir.

### Görev Tanımı:
[`src/renk_kumeleyici.py`](./src/renk_kumeleyici.py) sınıfına şu metodu eklemeni bekliyorum:

```python
def optimum_k_analizi(
    self,
    gorsel_bgr: np.ndarray,
    min_k: int = 2,
    max_k: int = 8
) -> List[Tuple[int, float]]:
```

### Beklenen Kurallar:
1. $K \in [min\_k, max\_k]$ aralığındaki her değer için K-Means modelini eğitmeli.
2. Modelin `inertia_` (WCSS / Küme içi hata kareleri toplamı) değerini kaydetmeli.
3. Çıktı olarak `[(k, wcss_skoru), ...]` çiftlerini döndürmeli (Böylece hata kırılma noktası - dirsek / elbow tespit edilebilir).

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> Beyaz bir arka plan üzerinde fotoğraflanmış renkli bir mobilya görüntüsünde, doğrudan tüm görüntü pikselleri K-Means'e verildiğinde **beyaz arka plan rengi** paletin %70'ini kaplayarak ilk sıraya oturur.  
> Bu sorunu çözmek ve **yalnızca ürünün kendi renk paletini** elde etmek için endüstriyel boru hatlarında nasıl bir ön işleme (preprocessing) veya filtreleme stratejisi uygulanmalıdır?

---

## 📂 Dizin Yapısı

```
day-11-dominant-color-extractor/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (opencv-python, numpy, scikit-learn, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve görselleştirme üretim akışı
├── ciktilar/                   # Üretilen renk paleti çizelgesi
│   └── baskin_renk_paleti.png
├── src/
│   ├── __init__.py
│   ├── renk_kumeleyici.py      # BaskinRenkCikarici ve RenkBilgisi sınıfları
│   └── palet_gorsellestirici.py # Orantısal renk şeridi ve kuantizasyon paneli
└── testler/
    └── test_baskin_renk.py     # 7 adet birim testi (7 passed in 4.86s)
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
python -m pytest testler/test_baskin_renk.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
