# Day 22: Veri Çoğaltma (Data Augmentation) & Veri Hikayeciliği

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-EE4C2C.svg?style=flat-square)](https://pytorch.org/vision/)
[![Albumentations](https://img.shields.io/badge/Albumentations-1.4+-00A86B.svg?style=flat-square)](https://albumentations.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; derin sinir ağlarının veri kıtlığında ezberlemesini (**Overfitting**) engelleyen ve modelleri gerçek dünya gürültülerine, kamera açılarına ve aydınlatma değişimlerine karşı dirençli kılan modern **Veri Çoğaltma (Data Augmentation)** tekniklerini uçtan uca inceler. `Albumentations`, `torchvision.transforms`, `MixUp` ve `CutMix` stratejilerini deneysel bir veri hikayesi (Data Storyteller) çerçevesinde karşılaştırır.

---

## 📖 Mentorluk Dersi ve Veri Hikayeciliği (Data Storytelling)

### 1. Veri Kıtlığı ve Aşırı Öğrenme Hikayesi
Derin yapay sinir ağları milyonlarca serbest parametreye sahiptir ve doğası gereği "tembel öğrencilerdir":
- Veri seti sınırlı olduğunda, nesnelerin gerçek biçimsel özelliklerini öğrenmek yerine arka plandaki önemsiz dokuları, sabit aydınlatma açılarını veya piksel gürültülerini ezberlerler.
- **Sonuç:** Eğitim kümesinde $\%100$ başarı elde eden bir model, sahada hafif gölgeli veya farklı açıyla çekilmiş bir fotoğrafta dramatik bir çöküş yaşar (**Domain Shift / Brittleness**).
- **Veri Çoğaltmanın Rolü:** Görsellere etiketini değiştirmeyecek biçimde geometrik ve fotometrik dönüşümler uygulayarak modele **Öteleme, Dönme, Ölçek ve Aydınlatma Değişmezliği (Invariances)** kazandırır.

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Döndürme, kırpma, renk titremesi ve gürültü ekleme gibi geometrik/fotometrik dönüşümlerle eğitim verisinin dağılımını sentetik olarak zenginleştirmek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Aşırı öğrenmeyi (overfitting) engeller, modelin kamera açısı ve aydınlatma değişimlerine karşı dayanıklılığını (invariance) artırır.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Aşırı agresif artırmalar yapıldığında sınıfın ayırt edici öznitelikleri bozulabilir (örneğin rakam 6'nın dönerek 9 olması).

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- Albumentations, AutoAugment, RandAugment veya Mixup/CutMix.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Veri Artırma (Augmentation)** | *Data Augmentation* | Mevcut görsellere dönme, kırpma, renk değişimi gibi dönüşümler uygulayarak eğitim verisini sanal olarak zenginleştirme tekniği. |
| **Aşırı Öğrenme (Overfitting)** | *Overfitting Prevention* | Modelin eğitim verisini ezberlemesini engelleyip yeni ve görülmemiş verilere genelleme yapmasını sağlama. |
| **Rastgele Kırpma & Çevirme** | *Random Crop & Flip* | Görselin belirli bölgelerini rastgele kesme ve yatayda aynalama yaparak konumsal değişmezlik kazandıran dönüşümler. |
| **Renk Titreşimi (Color Jitter)** | *Color Jittering* | Görselin parlaklık, kontrast, doygunluk ve ton değerlerini rastgele değiştirerek ışık değişimlerine dayanıklılık kazandırma. |

---

## 2. Dönüştürme Yöntemleri ve Matematiksel Dinamikler

#### A. Geleneksel Geometrik ve Fotometrik Dönüşümler (Albumentations & torchvision)
1. **Rastgele Aynalama & Döndürme (Random Flip & Rotation):** Görselin yönelimine olan aşırı bağımlılığı kırar.
2. **Renk Titreşimi (ColorJitter):** Parlaklık, kontrast, doygunluk ve ton değerlerini rastgele kaydırarak aydınlatma değişimlerine bağışıklık sağlar.
3. **CoarseDropout / RandomErasing (Bölgesel Silme):** Görselin rastgele bölgelerini maskeleyerek modelin kısmi kapanmalarda (**Occlusion**) dahi nesneyi tanımasını zorunlu kılar.

#### B. MixUp (Doğrusal Giriş & Etiket İnterpolasyonu - Zhang et al., 2018)
MixUp, eğitim kümesindeki iki rastgele örneği ve bunların one-hot etiketlerini $\lambda \sim \text{Beta}(\alpha, \alpha)$ katsayısıyla harmanlar:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$

$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

- **Neden İşe Yarar?** Standart sınıflandırıcılar sınıflar arasında sert ve dik karar sınırları oluşturur. MixUp karar sınırlarını pürüzsüzleştirir ve modelin aşırı emin (overconfident) yanlış tahminler yapmasını engeller (**Model Calibration**).

#### C. CutMix (Bölgesel Kes-Yapıştır - Yun et al., 2019)
CutMix, MixUp'ın pikselleri şeffaf biçimde karıştırması yerine, görsel $B$'den kesilen bir dikdörtgeni görsel $A$'nın içine yerleştirir:

$$\tilde{x} = \mathbf{M} \odot x_i + (\mathbf{1} - \mathbf{M}) \odot x_j$$

$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j, \quad \lambda = 1 - \frac{W_{\text{box}} \times H_{\text{box}}}{W \times H}$$

- Model hem kesilen nesnenin bölgesel özelliklerini hem de arka plandaki ana nesneyi aynı anda ayırt etmeyi öğrenir.

#### D. Çift Hedefli Kayıp Fonksiyonu (Soft-Target Cross-Entropy)
MixUp ve CutMix ile harmanlanmış etiketler için kayıp:

$$\mathcal{L}(\hat{y}, \tilde{y}) = \lambda \cdot \mathcal{L}_{\text{CE}}(\hat{y}, y_a) + (1 - \lambda) \cdot \mathcal{L}_{\text{CE}}(\hat{y}, y_b)$$

---

### 3. Kritik Mühendislik Tuzakları
1. **Semantik Anlamı Bozan Dönüşümler:** Örneğin OCR/Plaka tanıma problemlerinde `6` rakamını dikey ters çevirmek (`VerticalFlip`) onu `9` yapar; bu durum modele yanlış etiket öğretir (**Label Corruption**).
2. **Doğrulama ve Test Kümelerine Çoğaltma Uygulanması:** Veri çoğaltma **yalnızca eğitim kümesine** uygulanmalıdır. Validation ve Test kümeleri deterministik kalmalıdır (sadece standart Resize ve Normalization).

---

## 🛠️ Dizin Yapısı

```
day-22-data-augmentation/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # Albumentations, PyTorch, Torchvision vb.
├── ana_akis.py                      # Uçtan uca dönüşüm, ablation deneyleri ve hikaye özeti
├── README.md                        # Detaylı teorik ve veri hikayeciliği dokümantasyonu
├── src/
│   ├── __init__.py
│   ├── albumentations_donusturucu.py # Hızlı C++ tabanlı görsel dönüştürme boru hattı
│   ├── torchvision_donusturucu.py   # PyTorch yerel tensör dönüşüm düzeni
│   ├── mixup_cutmix.py              # MixUp & CutMix algoritmaları ve çift hedefli kayıp
│   ├── karsilastirici.py            # 4 stratejili Ablation deney motoru (Temiz vs Gürültülü)
│   └── gorsellestirici.py           # Galeri çizici ve karşılaştırmalı veri hikayesi grafiği
├── testler/
│   ├── __init__.py
│   └── test_veri_cogaltma.py        # 7 adet kapsamlı birim test
└── ciktilar/
    ├── veri_cogaltma_galerisi.png   # Orijinal vs Albumentations vs MixUp vs CutMix
    └── veri_cogaltma_karsilastirma_raporu.png # Dayanıklılık ve Bozulma Analizi
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

> **Soru:** Görüntü çoğaltmada MixUp ve CutMix tekniklerinin modelin aşırı özgüvenli (over-confident) tahminlerini engellemedeki matematiksel rolü nedir ve etiket düzleştirme (label smoothing) ile ilişkisi nasıldır?

> **Mentor Cevabı:**
> 1. **Doğrusal İnterpolasyon ve Karar Sınırları (Linearity Bias):** Standart cross-entropy kaybı modeli tek bir sınıfa %100 olasılık atamaya zorlar. MixUp ($\tilde{x} = \lambda x_i + (1-\lambda) x_j, \tilde{y} = \lambda y_i + (1-\lambda) y_j$) ve CutMix, etiketleri de orantılı olarak karıştırarak modelin sınıflar arasında doğrusal ve yumuşak geçişler öğrenmesini sağlar.
> 2. **Kalıplara Aşırı Uyumun Engellenmesi:** CutMix belirli bir bölgeyi kesip başka görüntüyle doldurduğu için modelin tek bir ayırt edici piksele (örneğin sadece göze veya logoya) bağımlı kalmasını önler ve küresel temsilleri öğrenmeye zorlar.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
