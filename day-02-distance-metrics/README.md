# Day 02: Vektörel ve Piksel Düzeyinde Mesafe ve Benzerlik Metrikleri

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; bilgisayarlı görüde, semantik görsel aramada, k-En Yakın Komşu (k-NN) algoritmalarında ve derin öğrenme embedding eşleştirmelerinde kullanılan temel matematiksel mesafe ve benzerlik metriklerini (Öklid, Manhattan, Kosinüs, Chebyshev, Minkowski) sıfırdan vektörize NumPy operasyonları ile uygular ve görsel katalog aramasını simüle eder.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Bir derin öğrenme modeli (ör. ResNet, CLIP, DINOv2) bir görseli aldığında, onu $128, 512$ veya $2048$ boyutlu bir **öznitelik vektörüne (embedding)** dönüştürür.
Peki iki görselin "aynı halı deseni", "aynı araba" veya "aynı insan yüzü" olup olmadığına nasıl karar verilir?

İşte burada devreye **vektör uzayındaki mesafe ve açı hesaplamaları** girer. Yanlış metrik seçimi veya normalizasyon eksikliği; sistemin aynı nesneyi sadece ışık farkı yüzünden bambaşka nesneler sanmasına yol açar.

---

### 2. Matematiksel ve Algoritmik Mantık

#### A. Öklid Mesafesi ($L_2$ Normu)
İki nokta arasındaki en kısa doğrusal fiziksel mesafedir. Vektör elemanlarının kareler farkının toplamının kareköküdür:

$$d_2(u, v) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2} = \|u - v\|_2$$

#### B. Manhattan Mesafesi ($L_1$ Normu / Şehir Bloku)
Izgara benzeri bir şehirde sokakları takip eder gibi sadece eksenler doğrultusunda ilerlenen mesafedir. Koordinat farklarının mutlak değerlerinin toplamıdır:

$$d_1(u, v) = \sum_{i=1}^n |u_i - v_i| = \|u - v\|_1$$
- **Avantajı:** $L_2$ normu farkların karesini aldığı için aşırı büyük aykırı değerlere (outliers) karşı çok duyarlıdır; $L_1$ normu ise aykırı değerlere karşı çok daha dayanıklıdır (robust).

#### C. Kosinüs Benzerliği ve Mesafesi (Açısal Benzerlik)
Vektörlerin uzunluğuna (magnitude) bakmaksızın, çok boyutlu uzayda aralarındaki açının kosinüsünü ($\cos \theta$) hesaplar:

$$S_{\cos}(u, v) = \frac{u \cdot v}{\|u\|_2 \cdot \|v\|_2 + \epsilon} = \frac{\sum u_i v_i}{\sqrt{\sum u_i^2} \sqrt{\sum v_i^2} + \epsilon}$$

- Sonuç $[-1.0, 1.0]$ aralığındadır. $1.0$ tam özdeş yönü, $0.0$ birbirine dik/ilişkisiz olmayı, $-1.0$ ise tam zıtlığı ifade eder.
- **Kosinüs Mesafesi:** $D_{\cos}(u, v) = 1.0 - S_{\cos}(u, v)$ (Aralık: $[0.0, 2.0]$).

#### D. Chebyshev Mesafesi ($L_\infty$ Normu / Satranç Tahtası)
Satrançtaki Şah'ın iki kare arasındaki en kısa hamle sayısı mantığıdır. Vektörün herhangi bir koordinatındaki **en büyük mutlak farkı** alır:

$$d_\infty(u, v) = \max_{i} |u_i - v_i| = \|u - v\|_\infty$$

#### E. Genelleştirilmiş Minkowski Mesafesi ($L_p$ Normu)
Yukarıdaki tüm metriklerin üst kümesidir:
$$d_p(u, v) = \left(\sum_{i=1}^n |u_i - v_i|^p\right)^{1/p}$$
- $p = 1 \implies$ Manhattan
- $p = 2 \implies$ Öklid
- $p \to \infty \implies$ Chebyshev

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Büyüklük Yanlılığı (Magnitude Bias):**
   Bir fotoğraf düşünün: Orijinal hali ve aynı fotoğrafın %50 daha karanlık hali. İki görselin pikselleri veya derin öznitelikleri tamamen **aynı yöne** bakar; sadece ikinci vektörün uzunluğu daha küçüktür. 
   - $L_2$ (Öklid) mesafesi bu iki görseli **"birbirinden çok uzak ve alakasız"** zanneder!
   - Kosinüs benzerliği ise aradaki açıyı ölçtüğü için **skoru $1.0$ (kusursuz eşleşme)** verir.
   - **Kural:** Embedding'ler $L_2$ ile normalize edilmedikçe ($\|v\|_2 = 1$), Öklid mesafesi doğrudan kullanılmamalıdır.
2. **Boyut Laneti (Curse of Dimensionality):**
   Boyut sayısı $D$ arttıkça ($D = 1000+$), uzaydaki tüm rastgele noktaların birbirine olan Öklid mesafeleri birbirine eşitlenmeye başlar. Yüksek boyutlu uzaylarda Manhattan veya Kosinüs metriği her zaman Öklid'den daha iyi ayrıştırma sağlar.
3. **Python Döngüleri ile k-NN Yapmak (Performans İntiharı):**
   100.000 adet katalog embedding'i ile sorgu vektörünü tek tek `for` döngüsüyle karşılaştırmak yerine, NumPy'ın **yayınlama (broadcasting)** ve **matris çarpımı (`np.dot`)** yetenekleri kullanılarak tüm veri tabanı tek hamlede C seviyesinde taranmalıdır.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
       Sorgu Vektörü (Q)           Katalog Matrisi (N x D)
              │                               │
              └───────────────┬───────────────┘
                              ▼
                      ┌───────────────┐
                      │  MesafeOlcer  │
                      └───────┬───────┘
                              │
       ┌──────────────┬───────┴───────┬──────────────┐
       ▼              ▼               ▼              ▼
  [Öklid (L2)] [Manhattan (L1)] [Kosinüs Benzerliği] [Chebyshev (L-Sonsuz)]
  Geometrik    Izgara / Şehir   Açısal Yönelim       Maksimum Boyut
  Uzaklık      Bloku Mesafesi   (Işık Bağımsız)      Farkı
```

---

## 💻 Konsol Çalıştırma Çıktısı

```text
======================================================================
>>> AŞAMA 1: Temel Vektörler Üzerinde Tüm Metriklerin Karşılaştırılması
======================================================================
Vektör 1: [0.25 0.7  0.1  0.95 0.4 ]
Vektör 2: [0.3  0.65 0.15 0.85 0.5 ]
----------------------------------------------------------------------
Metrik Adı                | Değer        | Ölçek Tipi      | Yorum
----------------------------------------------------------------------
Öklid (L2)                | 0.16583      | mesafe          | Sıfıra yakınsa benzer
Manhattan (L1)            | 0.35000      | mesafe          | Sıfıra yakınsa benzer
Chebyshev (L-Sonsuz)      | 0.10000      | mesafe          | Sıfıra yakınsa benzer
Minkowski (p=3)           | 0.13342      | mesafe          | Sıfıra yakınsa benzer
Kosinüs Benzerliği        | 0.99192      | benzerlik       | 1.0'a yakınsa benzer
Kosinüs Mesafesi          | 0.00808      | mesafe          | Sıfıra yakınsa benzer

======================================================================
>>> AŞAMA 2: Büyüklük Yanlılığı (Magnitude Bias) Deneyi: Öklid vs. Kosinüs
======================================================================
Vektör A (Düşük Parlaklık) : [1. 2. 3.]
Vektör B (Yüksek Parlaklık): [10. 20. 30.]
[!] Öklid Mesafesi (L2)    : 33.6749  (Büyüklük farkından ötürü çok uzak görünüyor!)
[V] Kosinüs Benzerliği     : 1.0000  (Yönler özdeş olduğu için KUSURSUZ eşleşme!)

======================================================================
>>> AŞAMA 3: İki Görsel Arasında Piksel Düzeyinde Uzamsal Fark Haritası
======================================================================
[+] Fark Haritası Boyutu   : (64, 64)
[+] Min Piksel Farkı       : 0.00
[+] Max Piksel Farkı       : 294.15 (Kusurlu bölgedeki Öklid renk farkı)
[+] Kusurlu Piksel Adedi   : 400 piksel (Beklenen: 20x20 = 400)

======================================================================
>>> AŞAMA 4: Görsel Benzerlik Motoru ile Katalog Araması (Top-3 Retrieval)
======================================================================
Sorgu Vektörü kataloga soruluyor...

  #1 Eşleşme: Hali_Vintage_Usak          -> Skor: 0.9991 (kosinus)
  #2 Eşleşme: Hali_Klasik_Hereke         -> Skor: 0.9972 (kosinus)
  #3 Eşleşme: Hali_Ipek_Kayseri          -> Skor: 0.9811 (kosinus)
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Canberra Mesafesi (Canberra Distance) Metriğini Eklemek**

Endüstride ve özellikle histogram karşılaştırmalarında Manhattan mesafesi, büyük değerli boyutların baskısı altında ezilebilir. Bu sorunu çözmek için sıfıra yakın değerlerde bile hassas kalan **Canberra Mesafesi** kullanılır:

$$d_{\text{Canberra}}(u, v) = \sum_{i=1}^n \frac{|u_i - v_i|}{|u_i| + |v_i| + \epsilon}$$

### Görev Tanımı:
[`src/mesafe_olcer.py`](./src/mesafe_olcer.py) içerisine şu fonksiyonu eklemeni ve birim testini yazmanı bekliyorum:

```python
@classmethod
def canberra_mesafesi(
    cls,
    vektor_a: np.ndarray,
    vektor_b: np.ndarray,
    epsilon: float = 1e-9
) -> float:
```

### Kurallar:
1. $u_i = 0$ ve $v_i = 0$ olduğunda payda sıfır olacağından $\epsilon$ ile sıfıra bölme engellenmelidir.
2. Vektörün her boyutu için fark paya, mutlak değerler toplamı paydaya bölünmelidir.
3. Sonuç tüm boyutlar üzerinden toplanmalıdır.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> İki adet $D$ boyutlu vektörün **$L_2$ Normu (Öklid Uzunluğu) $1.0$ yapıldığında** (yani $L_2$ normalizasyonu uygulandığında), **Öklid Mesafesinin karesi ($d_2^2$)** ile **Kosinüs Benzerliği ($S_{\cos}$)** arasında nasıl doğrudan bir matematiksel eşitlik kurulur?  
> *(İpucu: $\|u - v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 - 2(u \cdot v)$ açılımını düşün).*

---

## 📂 Dizin Yapısı

```
day-02-distance-metrics/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar
├── ana_akis.py                 # Konsol laboratuvar akışı
├── src/
│   ├── __init__.py
│   ├── mesafe_olcer.py         # MesafeOlcer sınıfı
│   └── gorsel_eslestirici.py   # Top-K katalog arama motoru
└── testler/
    └── test_mesafeler.py       # 8 adet birim testi (8 passed)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Laboratuvar Akışını Çalıştırma
```bash
python ana_akis.py
```

### 3. Testleri Koşma
```bash
python -m pytest testler/test_mesafeler.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
