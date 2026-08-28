# Day 19: Geleneksel Makine Öğrenmesi ile Görsel Sınıflandırma (Classical Image Classifier)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.24+-orange.svg?style=flat-square)](https://scikit-image.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; kısıtlı kaynaklı IoT ve Edge AI cihazlarında GPU gerektirmeyen, veri miktarının az olduğu (Low-Data Regime) senaryolarda derin öğrenmeye kıyasla çok daha hızlı eğitilip yüksek doğruluk sağlayan **Geleneksel Makine Öğrenmesi ile Görsel Sınıflandırma (HOG + LBP + Renk Momentleri + SVM / Random Forest)** mimarisini üretim standartlarında sunar.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Derin Öğrenme (CNN / ViT) modelleri milyonlarca parametreye ve devasa GPU hesaplama gücüne ihtiyaç duyar. Ancak endüstride:
- **Uç Cihazlar (Edge / Microcontrollers / Raspberry Pi / Akıllı Kameralar):** Bellek ve enerji kısıtları nedeniyle ağır derin ağları gerçek zamanlı çalıştıramaz.
- **Az Verili Senaryolar (Few-Shot / Small Datasets):** Sınıf başına yalnızca 20-50 görselin bulunduğu durumlarda derin öğrenme aşırı öğrenir (overfitting).
- **Yorumlanabilirlik (Explainable AI - XAI):** Medikal ve savunma sanayinde kararın hangi kenar veya doku özniteliğinden kaynaklandığının açıklanabilir olması zorunludur.

**Çözüm:** Alan bilgisiyle tasarlanmış (Hand-crafted) güçlü öznitelik çıkarıcılar (**HOG**, **LBP**, **Renk İstatistikleri**) ile **Support Vector Machines (SVM)** ve **Random Forest** sınıflandırıcılarını birleştirmektir.

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **HOG** | *Histogram of Oriented Gradients* | Yerel bölgelerdeki kenar ve gradyan yönelimlerinin histogramını çıkararak nesne şekil ve konturlarını yakalayan öznitelik tanımlayıcısı. |
| **Destek Vektör Makinesi (SVM)** | *Support Vector Machine* | Sınıflar arasındaki geometrik marjini (margin) maksimize eden optimum hiper düzlemi bulan güçlü doğrusal/doğrusal olmayan sınıflandırıcı. |
| **RBF Çekirdeği (Kernel Trick)** | *Radial Basis Function Kernel* | Veriyi açıkça hesaplamadan sonsuz boyutlu uzaya haritalayarak doğrusal olmayan sınırları ayıran çekirdek fonksiyonu. |
| **Öznitelik Normalizasyonu** | *Feature Scaling (StandardScaler)* | SVM'in marjin hesabında büyük değerli özniteliklerin baskın gelmesini önlemek için uygulanan standartlaştırma. |

---

## 2. Matematiksel Temeller ve Algoritmik Mantık

#### A. HOG (Histogram of Oriented Gradients)
Nesnelerin şekil ve kenar yapılarını gradyan yönelimlerinin dağılımıyla yakalar:
1. **Gradyan Hesabı:**
   $$g_x = I(x+1, y) - I(x-1, y), \quad g_y = I(x, y+1) - I(x, y-1)$$
   $$m(x, y) = \sqrt{g_x^2 + g_y^2} \quad (\text{Büyüklük}), \quad \theta(x, y) = \arctan\left(\frac{g_y}{g_x}\right) \quad (\text{Yönelim})$$
2. **Hücre Histogramları:** $16 \times 16$ piksellik hücrelerde 8 açılı yönelim histogramı oluşturulur.
3. **Blok Normalizasyonu ($L_2$-Hys):** Aydınlatma ve gölge değişimlerine karşı $2 \times 2$ bloklar halinde normalize edilir:
   $$v_{\text{norm}} = \frac{v}{\sqrt{\|v\|_2^2 + \epsilon^2}}, \quad v_{\text{clipping}} = \min(v_{\text{norm}}, 0.2)$$

---

#### B. LBP (Local Binary Patterns)
Mikro doku sürekliliğini yakalar:
- Merkez piksel $g_c$, $P$ adet komşusu $g_p$ ile karşılaştırılır:
  $$LBP_{P, R} = \sum_{p=0}^{P-1} s(g_p - g_c) 2^p, \quad s(x) = \begin{cases} 1, & x \ge 0 \\ 0, & x < 0 \end{cases}$$
- **Uniform LBP:** İkili dizilimde 0'dan 1'e veya 1'den 0'a geçiş sayısı $\le 2$ olan desenleri gruplayarak boyut patlamasını önler ($P=8 \implies 10$ boyut).

---

#### C. Support Vector Machines (SVM) & Kernel Trick
Sınıflar arasındaki marjini (ayrım sınırını) maksimize eden hiper-düzlemi bulur:

$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{Kısıt:} \quad y_i (w^T \phi(x_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

- **RBF (Radyal Tabanlı Fonksiyon) Çekirdeği:** Doğrusal olmayan sınırları sonsuz boyutlu Hilbert uzayına projekte eder:
  $$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$
  - $C$ (Ceza Parametresi): Yüksek $C$ düşük eğitim hatasını hedefler (aşırı uyum riski), düşük $C$ daha geniş marjini hedefler.
  - $\gamma$ (Gamma): Tek bir örneğin etki yarıçapı.

---

#### D. Random Forest (Topluluk Öğrenmesi / Ensemble)
- Çok sayıda bağımsız Karar Ağacının (**Decision Trees**) Bagging (Bootstrap Aggregation) ve Rastgele Alt-Uzay (Random Subspace) yöntemiyle eğitilip çoğunluk oyuyla (Voting) karar vermesidir.
- **Gini Saflık Bozulması (Gini Impurity):**
  $$G = 1 - \sum_{k=1}^K p_k^2$$
- Model hangi özniteliklerin (HOG kenarları mı, LBP dokusu mu, Renk mi) sınıflandırmada en çok bilgi kazancı sağladığını **Öznitelik Önemi (Feature Importance)** olarak sunar.

---

### 3. Kritik Mühendislik Tuzakları
1. **Veri Sızıntısı (Data Leakage):** Normalizasyon ve ölçekleme (`StandardScaler`), tüm veri setine değil; **yalnızca eğitim kümesine fit edilip** test kümesine uygulanmalıdır. Projemizde bu kural `sklearn.pipeline.Pipeline` ile garanti altına alınmıştır.
2. **Sınıf Dengesizliği:** Gerçek dünya verilerinde sınıflar eşit sayıda olmadığında `class_weight='balanced'` kullanılmazsa model çoğunluk sınıfına yanlılık (bias) gösterir.

---

## 🛠️ Dizin Yapısı

```
day-19-classical-image-classifier/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # Bağımlılıklar
├── ana_akis.py                      # Uçtan uca veri üretimi, eğitim, k-fold CV ve kıyaslama
├── README.md                        # Detaylı dokümantasyon
├── src/
│   ├── __init__.py
│   ├── oznitelik_cikarici.py        # HOG (72D) + LBP (10D) + Renk (12D) = 94D öznitelik motoru
│   ├── siniflandirici.py            # Pipeline(StandardScaler -> SVM / Random Forest) yöneticisi
│   └── degerlendirici.py            # Confusion Matrix, F1, Feature Importance görselleştirici
├── testler/
│   ├── __init__.py
│   └── test_siniflandirici.py       # 7 adet kapsamlı birim test
└── ciktilar/
    └── siniflandirma_raporu.png     # 4 panelli yüksek çözünürlüklü teşhis çizelgesi
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

### 3. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Görüntü sınıflandırmada el yapımı öznitelikler (Handcrafted Features: HOG, LBP, Renk Momentleri) ile eğitilen klasik modeller (SVM/Random Forest) ile derin öğrenme (CNN) modelleri arasındaki temel fark ve trade-off nedir?

> **Mentor Cevabı:**
> 1. **Öznitelik Öğrenimi vs Manuel Mühendislik:** Klasik modellerde uzmanın hangi özniteliklerin (kenar yönelimleri için HOG, doku için LBP) ayırt edici olduğunu önceden bilmesi gerekir. CNN'ler ise hiyerarşik filtrelerle (kenar -> doku -> parça -> nesne) veriden uçtan uca öznitelik öğrenir.
> 2. **Veri Hacmi ve Çıkarım Maliyeti:** Küçük veri setlerinde ($< 500$ örnek) ve düşük işlem gücüne sahip mikrodenetleyicilerde HOG + Linear SVM aşırı öğrenmeye (overfitting) karşı daha dirençlidir ve mikro-saniye mertebesinde CPU çıkarımı sağlar.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
