# Day 15: Ön Plan ve Arka Plan Segmentasyonu (GrabCut Background Remover)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; e-ticaret stüdyo fotoğrafçılığında (ürün dekupe), fotoğraf düzenleme araçlarında (remove.bg benzeri) ve sinema/video kurgusunda yeşil perde (green screen) olmadan karmaşık ve homojen olmayan zeminlerden nesneleri ayırmak için **Gauss Karışım Modelleri (GMM), Markov Rasgele Alanları (MRF) ve Min-Cut / Max-Flow Çizge Kesme (Graph Cut)** algoritmasını birleştiren, interaktif fırça iyileştirmesi ve alfa kompozisyonu sunan üretim seviyesinde bir arka plan temizleme motorudur.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Görüntü işlemede nesne ayırmak için basit renk eşikleme (Thresholding) veya renk maskeleme yetersizdir:
- Bir e-ticaret satıcısı salon halısını ya da seramik vazoyu çekip platforma yüklediğinde, zemin ahşap parke, gölgeli duvar veya desenli örtü olabilir.
- Ön plan ile arka plan benzer renk tonlarını paylaşabilir.
- Klasik kenar bulucular kesintili hatlar üretirken, piksel bazlı sınıflandırıcılar gürültülü delikler bırakır.

**GrabCut (Rother, Kolmogorov, Blake - SIGGRAPH 2004):**
Kullanıcının nesneyi kabaca bir dikdörtgen kutu içine almasıyla çalışan; renk olasılıklarını **Gauss Karışım Modelleri (GMM)** ile, mekansal sınır sürekliliğini ise **Çizge Kesme (Graph Cut)** ile küresel olarak optimize eden çığır açıcı bir algoritmadır.

---

### 2. Matematiksel ve Algoritmik Mantık

#### A. Gauss Karışım Modelleri (Gaussian Mixture Models - GMM)
Ön plan ($F$) ve arka plan ($B$) piksellerinin RGB renk dağılımları tek bir ortalama ile değil, her biri $K=5$ bileşenli tam kovaryanslı Gauss karışımları ile modellenir:

$$p(z | \alpha) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(z; \mu_k, \Sigma_k)$$

- $\alpha \in \{0, 1\}$: 0 arka plan, 1 ön plan.
- $z = (R, G, B)$: Piksel renk vektörü.
- $\pi_k, \mu_k, \Sigma_k$: Karışım ağırlığı, ortalama vektörü ve $3 \times 3$ kovaryans matrisi.

#### B. Gibbs Enerji Fonksiyonu ($E = U + V$)
GrabCut, piksel etiketleme problemini şu toplam enerji fonksiyonunu minimize eden bir çizge problemine dönüştürür:

$$E(\underline{\alpha}, \mathbf{k}, \underline{\theta}, \mathbf{z}) = U(\underline{\alpha}, \mathbf{k}, \underline{\theta}, \mathbf{z}) + V(\underline{\alpha}, \mathbf{z})$$

1. **Veri Terimi (Regional / Data Term - $U$):**
   Bir pikselin ön plan ya da arka plan GMM modeline uyma negatif log-olasılığıdır:
   $$U = -\sum_n \log p(z_n | \alpha_n, k_n, \theta)$$
   Piksel arka plana benziyorsa ön plan etiketi alması yüksek ceza puanı doğurur.

2. **Düzgünlük Terimi (Boundary / Smoothness Term - $V$):**
   Komşu pikseller arasındaki etiket süreksizliklerini cezalandırır. Ancak iki komşu piksel arasında güçlü bir renk farkı (kontrast/gradyan) varsa bu ceza düşer (çünkü sınır oradadır!):
   $$V = \gamma \sum_{(m, n) \in \mathcal{C}} [\alpha_m \neq \alpha_n] \cdot \exp\left(-\beta \|z_m - z_n\|^2\right)$$
   Burada $\beta = \frac{1}{2 \langle \|z_m - z_n\|^2 \rangle}$ yerel renk kontrastına göre normalize edilir.

#### C. Min-Cut / Max-Flow Çizge Kesme (Boykov-Kolmogorov)
Tüm görüntü pikselleri bir çizgenin düğümleri (nodes), $s$ (Kaynak - Ön Plan) ve $t$ (Hedef - Arka Plan) ise terminalleridir.
Düğümler arası kenar ağırlıkları $V$ düzgünlük terimiyle, terminallere olan kenar ağırlıkları ise $U$ veri terimiyle kurulur.
Polinomial zamanda çözülen **Minimum Çizge Kesimi (Min-Cut)**, küresel minimum enerjili kesin ve kesintisiz nesne sınırını bulur.

#### D. OpenCV 4-Durumlu Maske Hiyerarşisi
- `cv2.GC_BGD` (0): Kesin Arka Plan (Kutunun dışı).
- `cv2.GC_FGD` (1): Kesin Ön Plan (Kullanıcı fırçasıyla onaylanan).
- `cv2.GC_PR_BGD` (2): Olası Arka Plan (Algoritmanın arka plan sandığı).
- `cv2.GC_PR_FGD` (3): Olası Ön Plan (Algoritmanın ön plan sandığı).

Nihai İkili Maske:
$$\text{Maske} = (\text{maske} == \text{GC\_FGD}) \mid (\text{maske} == \text{GC\_PR\_FGD})$$

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Kutunun Dışında Kalan Nesne Parçaları:**
   GrabCut kuralı gereği, başlangıç dikdörtgeninin dışında kalan pikseller `GC_BGD` (Kesin Arka Plan) damgası yer. Bir piksel bir kez kesin arka plan yapıldığında, sonraki iterasyonlarda **asla ön plan olamaz!** Bu yüzden başlangıç kutusu nesneyi tamamen içine alacak kadar cömert seçilmelidir.
2. **Ön Plan ve Arka Planın Aynı Renk Olması:**
   Nesne ile arka plan birebir aynı renkteyse GMM ayrım yapamaz; düzgünlük terimi $V$ ve kullanıcının `GC_INIT_WITH_MASK` ile atacağı fırça darbeleri (Strokes) hayati önem taşır.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
                       Karmaşık Zeminli Ürün Sahnesi
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Kullanıcı Sınırlayıcı    │
                      │     Kutusu (Bounding Box)   │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │   GrabCut (Faz 1 - Kutu)    │
                      │   - Arka Plan: GC_BGD (0)   │
                      │   - Kutu İçi: GC_PR_FGD (3) │
                      │   - 5x İterasyon (GMM + Cut)│
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │   GrabCut (Faz 2 - Maske)   │
                      │  - İnteraktif Fırça İzleri  │
                      │    (GC_FGD / GC_BGD)        │
                      │  - 3x İterasyon İyileştirme │
                      └──────────────┬──────────────┘
                                     │
        ┌────────────────────────────┴────────────────────────────┐
        ▼                                                         ▼
[Şeffaf Ürün İzolasyonu]                                  [Yeni Arka Plan Kompoziti]
- 4 Kanallı BGRA Görüntü                                  - Stüdyo Degrade Arka Plan
- Arka Planı Şeffaf PNG                                   - Feathered Alpha Blending
        │                                                         │
        └────────────────────────────┬────────────────────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │    GrabCutGorsellestirici   │
                      │  (4 Panelli Karşılaştırma)  │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                   [ciktilar/grabcut_segmentasyon_paneli.png]
```

---

## 🛠️ Kod Bileşenleri ve Modüler Yapı

1. **[`src/grabcut_ayristirici.py`](./src/grabcut_ayristirici.py):**
   - `GrabCutAyristirici`: Dikdörtgen tabanlı başlatma, interaktif fırça maskesiyle iyileştirme, 4 kanallı şeffaf BGRA üretimi ve alfa karıştırmalı (feathering) arka plan kompozisyonu.
2. **[`src/gorsellestirici.py`](./src/gorsellestirici.py):**
   - `GrabCutGorsellestirici`: Orijinal kutulu görseli, 4-durumlu renkli enerji maskesini, izole ön planı ve stüdyo kompozitini 4 panelli Matplotlib çizelgesi olarak kaydeder.
3. **[`ana_akis.py`](./ana_akis.py):**
   - Karmaşık dokulu parke zemin üzerindeki seramik vazoyu modelleyen, 2 fazlı GrabCut segmentasyonunu yürüten ve çıktıları kaydeden konsol betiği.

---

## 💻 Konsol Çalıştırma Çıktısı

```text
============================================================================
>>> AŞAMA 1: Karmaşık Sahnenin ve Başlangıç Kutusunun Hazırlanması
============================================================================
[+] Görüntü Çözünürlüğü       : 360 x 360 (129,600 piksel)
[+] Başlangıç Dikdörtgeni     : (x=85, y=75, w=190, h=230)
[+] Kutu Alanı                : 43,700 piksel (%33.7)

============================================================================
>>> AŞAMA 2: GrabCut (GMM + Graph Cut) 1. Faz: Dikdörtgen Başlatma
============================================================================
[V] Faz 1 Tamamlandı (5 İterasyon): 0.253 saniye
[V] Ayrıştırılan Ön Plan Pikselleri: 23,546 adet (%18.2)

============================================================================
>>> AŞAMA 3: GrabCut 2. Faz: İnteraktif Fırça İpuçlarıyla Kenar İyileştirme
============================================================================
[V] Faz 2 Tamamlandı (İnteraktif Maske İyileştirme)
[V] İyileştirilmiş Ön Plan Pikselleri: 23,546 adet

============================================================================
>>> AŞAMA 4: Şeffaf PNG (RGBA) ve Yeni Stüdyo Arka Plan Kompoziti
============================================================================
[V] Şeffaf 4 kanallı (BGRA) ürün görseli kaydedildi: izole_nesne_seffaf.png

============================================================================
>>> AŞAMA 5: 4 Panelli Analiz Çizelgesinin Kaydedilmesi
============================================================================
[V] GrabCut analiz paneli başarıyla kaydedildi: grabcut_segmentasyon_paneli.png
[V] Kayıt Konumu: day-15-grabcut-background-remover/ciktilar/grabcut_segmentasyon_paneli.png

[V] Day 15: GrabCut Ön Plan ve Arka Plan Segmentasyonu başarıyla tamamlandı.
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Otomatik Başlangıç Kutusu Üretici (Saliency / Contrast Bounding Box)**

GrabCut'ın en büyük kısıtı kullanıcının kutuyu elle çizmesidir. Bunu otomatikleştirmek için renk kontrastı veya kenar yoğunluğu üzerinden nesnenin tahmini sınırlayıcı kutusunu (`bounding box`) bulan algoritmayı eklemeni bekliyorum.

### Görev Tanımı:
[`src/grabcut_ayristirici.py`](./src/grabcut_ayristirici.py) sınıfına şu metodu eklemelisin:

```python
@classmethod
def otomatik_kutu_kestirimi(
    cls,
    gorsel_bgr: np.ndarray,
    kenar_marj_yuzdesi: float = 0.08
) -> Tuple[int, int, int, int]:
```

### Beklenen Kurallar:
1. Görüntünün kenarlarından (border) arka plan renk modelini örneklemeli.
2. Merkezdeki renk sapmalarını (Öklid mesafesi) bir dikkat (saliency) haritasına dönüştürmeli.
3. Otsu eşikleme ve en büyük konturun `cv2.boundingRect` kutusunu bulup hafif bir güvenlik marjı (`kenar_marj_yuzdesi`) ekleyerek `(x, y, w, h)` tuple olarak döndürmelidir.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> GrabCut'ın düzgünlük enerjisindeki (smoothness term) ceza formülü:  
> $$V = \gamma \sum_{(m, n)} [\alpha_m \neq \alpha_n] \cdot \exp\left(-\beta \|z_m - z_n\|^2\right)$$  
> Buradaki **$\beta$ parametresinin formülü neden $\beta = \frac{1}{2 \langle \|z_m - z_n\|^2 \rangle}$ olarak hesaplanır?**  
> Bu normalizasyon yapılmasaydı, çok düşük kontrastlı (soluk/sisli) bir görüntü ile çok yüksek kontrastlı bir görüntüde Graph Cut kesim sınırları nasıl davranırdı?

---

## 📂 Dizin Yapısı

```
day-15-grabcut-background-remover/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (opencv-python, numpy, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve görsel üretim akışı
├── ciktilar/                   # Üretilen şeffaf PNG ve analiz çizelgesi
│   ├── grabcut_segmentasyon_paneli.png
│   └── izole_nesne_seffaf.png
├── src/
│   ├── __init__.py
│   ├── grabcut_ayristirici.py  # GrabCut, GMM, şeffaf PNG ve arka plan kompoziti
│   └── gorsellestirici.py      # 4 panelli Matplotlib çizelge motoru
└── testler/
    └── test_grabcut.py         # 7 adet birim testi (7 passed in 0.99s)
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
python -m pytest testler/test_grabcut.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
