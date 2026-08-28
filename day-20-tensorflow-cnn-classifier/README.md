# Day 20: TensorFlow/Keras ile Derin Öğrenme Görsel Sınıflandırma (CNN Classifier)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.15+-D00000.svg?style=flat-square&logo=keras)](https://keras.io/)
[![PyTorch/TensorFlow](https://img.shields.io/badge/backend-multi--backend-FF6F00.svg?style=flat-square)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; derin öğrenmenin temel taşı olan **Evrişimli Sinir Ağlarını (Convolutional Neural Networks - CNN)** kullanarak sıfırdan çok sınıflı görsel sınıflandırma mimarisini kurar. `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Flatten`, `Dense` ve `Dropout` katmanlarını modern Keras mimarisiyle uçtan uca hayata geçirir.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Geleneksel Tam Bağlantılı (Fully Connected / Dense) Yapay Sinir Ağları görsellerde neden başarısız olur?
1. **Parametre Patlaması:** $256 \times 256 \times 3$ boyutunda bir görsel düzleştirildiğinde (Flatten) $196.608$ girdi nöronu oluşur. İlk gizli katmanda $1024$ nöron olursa yalnızca ilk katmanda **201 milyon ağırlık** gerekir. Bu bellek taşmasına ve aşırı ezberlemeye (overfitting) yol açar.
2. **Uzamsal Hiyerarşinin Kaybı:** Flatten işlemi piksellerin yan yana ve alt alta olma ilişkisini (2B geometriyi) yok eder.
3. **Öteleme Değişmezliğinin (Translation Invariance) Olmayışı:** Dense ağlar bir kedinin görselin sol üst köşesinde olmasıyla sağ alt köşesinde olmasını iki tamamen farklı durum sanır.

**Çözüm (CNN):** Evrişim işlemi, pikseller arasındaki yerel uzamsal korelasyonu küçük çekirdeklerle (**Kernels/Filters**) tarar ve **ağırlık paylaşımı (weight sharing)** sayesinde parametre sayısını binlerce kat azaltır.

---

### 2. Matematiksel Temeller ve Katman Dinamikleri

#### A. 2B Evrişim (2D Convolution) İşlemi
Girdi görseli $I$ ile $K \times K$ boyutundaki evrişim filtresi $W$ arasındaki işlem:

$$S(i, j) = (I * W)(i, j) + b = \sum_{m} \sum_{n} I(i-m, j-n) W(m, n) + b$$

- **Çıktı Boyutu Formülü (Feature Map Size):**
  $$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
  - $W$: Girdi genişliği / yüksekliği
  - $K$: Çekirdek boyutu (ör. $3 \times 3$)
  - $P$: Dolgu (Padding) miktarı (`same` padding için $P = \frac{K-1}{2}$)
  - $S$: Adım (Stride) miktarı

#### B. Batch Normalization (BN)
Eğitim sırasında önceki katmanların ağırlıkları değiştikçe sonraki katmanların girdi dağılımı sürekli kayar (**Internal Covariate Shift**). Batch Normalization, mini-batch bazında tensörü normalize eder:

$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

- $\gamma$ (ölçek) ve $\beta$ (kaydırma) öğrenilebilir parametrelerdir. BN, daha yüksek öğrenme oranları (`learning_rate`) kullanılmasına olanak tanır ve eğitimi stabilize eder.

#### C. MaxPooling (Maksimum Havuzlama)
Evrişim çıktısını $2 \times 2$ pencerelerde en büyük değeri alarak alt-örnekler (downsampling).
- Uzamsal boyutu yarıya indirir ($H/2, W/2$).
- Küçük yerel kaymalara ve gürültüye karşı öteleme değişmezliği (**translation invariance**) sağlar.
- Reseptif alanı (Receptive Field) genişleterek sonraki katmanların görselin daha büyük bir bölgesini görmesini sağlar.

#### D. Dropout (Aşırı Öğrenmeyi Kırma)
Eğitim sırasında her ileri geçişte (forward pass) nöronların rastgele bir kısmını ($p$ olasılıkla, ör. $\%40$) sıfırlar.
- Nöronların birbirine bağımlı olarak aynı özellikleri ezberlemesini (**co-adaptation**) engeller.
- Matematiksel olarak binlerce farklı alt-ağın ortalamasını alan bir topluluk (ensemble) etkisi yaratır.

---

### 3. Kritik Mühendislik Tuzakları
1. **Piksel Normalizasyonu Unutulması:** Piksel değerleri $[0, 255]$ yerine mutlaka $[0.0, 1.0]$ veya standartlaştırılmış $[-1.0, 1.0]$ aralığına çekilmelidir; aksi halde aktivasyonlar doyuma (saturation) ulaşır ve gradyanlar sıfırlanır.
2. **Aşırı Flatten Boyutu:** Son Conv katmanından sonra doğrudan Flatten uygulamak yerine uzamsal boyutun yeterince küçüldüğünden emin olunmalı ya da GlobalAveragePooling2D tercih edilmelidir.

---

## 🛠️ Dizin Yapısı

```
day-20-tensorflow-cnn-classifier/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # Bağımlılıklar (keras, scikit-learn, opencv, matplotlib vb.)
├── ana_akis.py                      # Uçtan uca veri üretimi, model derleme, eğitim ve değerlendirme
├── README.md                        # Detaylı mentorluk ve teorik dokümantasyon
├── src/
│   ├── __init__.py
│   ├── model_mimari.py              # Conv2D + BatchNorm + MaxPool + Dense + Dropout mimarisi
│   ├── veri_hazirlayici.py          # Görsel normalizasyonu, veri üretimi ve stratified bölümleme
│   ├── egitici.py                   # EarlyStopping, ReduceLROnPlateau ve metrik değerlendirme
│   └── gorsellestirici.py           # Loss/Acc eğrileri, Confusion Matrix ve test tahmin çizelgesi
├── testler/
│   ├── __init__.py
│   └── test_cnn.py                  # 6 adet kapsamlı birim test
└── ciktilar/
    └── cnn_egitim_raporu.png        # 4 panelli yüksek çözünürlüklü teşhis çizelgesi
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

Konsolda katman katman mimari özeti, eğitim epoch'ları ve sınıf bazında test doğrulukları listelenecek; `ciktilar/cnn_egitim_raporu.png` dosyası otomatik üretilecektir.

### 3. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
