# Day 23: Transfer Öğrenme ve İnce Ayar (Transfer Learning & Fine-Tuning)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-EE4C2C.svg?style=flat-square)](https://pytorch.org/vision/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; sıfırdan derin öğrenme modeli eğitmenin getirdiği devasa veri ve hesaplama maliyetlerini ortadan kaldıran **Transfer Öğrenme (Transfer Learning)** ve **İnce Ayar (Fine-Tuning)** stratejilerini inceler. **ResNet18** ve **EfficientNet-B0** omurgaları üzerinde Öznitelik Çıkarma (Feature Extraction), Kısmi Kilit Açma (Fine-Tuning) ve Ayrıştırılmış Öğrenme Oranları (**Discriminative Learning Rates**) mekanizmalarını deneysel olarak karşılaştırır.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Problem
Gerçek dünya projelerinde on binlerce etiketli görsel toplamak ve sıfırdan 50+ katmanlı bir evrişimli sinir ağını eğitmek haftalar sürebilir ve aşırı ezberlemeye (**Overfitting**) açıktır.
- **Transfer Öğrenme Yaklaşımı:** ImageNet (1.2 milyon görsel, 1000 sınıf) üzerinde önceden eğitilmiş (**Pretrained**) bir omurga modelin ilk katmanları temel kenarları, köşeleri ve dokuları; derin katmanları ise karmaşık geometrik desenleri ve nesne parçalarını zaten öğrenmiştir.
- Bu zengin görsel temsil yeteneği, özel hedef görevimize yalnızca birkaç yüz örnekle transfer edilir (**Few-Shot Adaptation**).

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Transfer Öğrenme** | *Transfer Learning* | ImageNet gibi devasa veri setlerinde önceden eğitilmiş omurga (backbone) ağların öznitelik temsil gücünü yeni bir alt göreve aktarma. |
| **Katman Dondurma (Freezing)** | *Layer Freezing* | Önceden eğitilmiş ağırlıkların gradyan hesaplamasını kapatıp (`requires_grad=False`) sadece yeni sınıflandırıcı başlığı eğitme stratejisi. |
| **İnce Ayar (Fine-Tuning)** | *Fine-Tuning* | Modelin tüm katmanlarını veya üst katmanlarını çok düşük bir öğrenme oranıyla yeni veri setine göre hassas biçimde yeniden eğitme. |
| **Sınıflandırıcı Başlık** | *Classifier Head* | Öznitelik haritasının sonundaki global pooling ardından gelen ve hedef sınıf sayısı kadar çıktı üreten tam bağlantılı katman. |

---

## 2. İki Temel Transfer Öğrenme Stratejisi

```
+-----------------------------------------------------------------------------------+
|                        TRANSFER ÖĞRENME STRATEJİLERİ                              |
+-----------------------------------------------------------------------------------+
| 1. Öznitelik Çıkarma (Feature Extraction / Frozen Backbone):                     |
|    - Omurga Katmanları: DONDURULMUŞ (requires_grad = False)                      |
|    - Sınıflandırıcı Başlık: EĞİTİLEBİLİR (requires_grad = True)                   |
|    - Avantaj: Düşük bellek/hesaplama maliyeti, sıfır aşırı ezberleme riski.       |
|                                                                                   |
| 2. İnce Ayar (Fine-Tuning / Discriminative Learning Rates):                       |
|    - Omurga Katmanları: KISMEN AÇIK (Örn. Son Residual Blok / requires_grad=True)|
|    - Ayrıştırılmış LR: η_omurga = 1e-4 (Düşük) vs η_başlık = 1e-3 (Yüksek)        |
|    - Avantaj: Hedef göreve özel yüksek temsiliyet ve maksimum test doğruluğu.      |
+-----------------------------------------------------------------------------------+
```

#### A. Ayrıştırılmış Öğrenme Oranları (Discriminative Layer Training)
Omurganın önceden öğrendiği genel özellikleri bozmamak (**Catastrophic Forgetting**) için omurga parametreleri çok küçük bir öğrenme oranıyla, sıfırdan eklenen sınıflandırıcı başlık ise daha büyük bir öğrenme oranıyla güncellenir:

$$\theta_{\text{omurga}}^{(t+1)} = \theta_{\text{omurga}}^{(t)} - \eta_{\text{düşük}} \cdot \nabla \mathcal{L}$$

$$\theta_{\text{başlık}}^{(t+1)} = \theta_{\text{başlık}}^{(t)} - \eta_{\text{yüksek}} \cdot \nabla \mathcal{L}, \quad \left(\eta_{\text{düşük}} \approx \frac{1}{10} \eta_{\text{yüksek}}\right)$$

---

### 3. Omurga Mimarileri: ResNet vs EfficientNet

#### A. ResNet (Residual Networks - He et al., 2015)
Klasik derin ağlarda katman sayısı arttıkça gradyan kaybolması (Vanishing Gradient) nedeniyle başarım düşer. ResNet **Artık Bağlantılar (Skip / Residual Connections)** ile bu sorunu çözmüştür:

$$y = \mathcal{F}(x, \{W_i\}) + x$$

Gradyan geri yayılırken $+ x$ terimi sayesinde türev doğrudan önceki katmanlara $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right)$ olarak kesintisiz akar.

#### B. EfficientNet (Compound Scaling - Tan & Le, 2019)
Ağı yalnızca derinleştirmek yerine; Derinlik ($d$), Genişlik ($w$) ve Girdi Çözünürlüğünü ($r$) sabit bir bileşik ölçekleme katsayısı $\phi$ ile dengeli şekilde büyütür:

$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$

$$\text{Kısıt: } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha \ge 1, \beta \ge 1, \gamma \ge 1$$

---

### 4. Kritik Mühendislik Tuzakları
1. **Felaketsel Unutma (Catastrophic Forgetting):** Başlık rastgele ağırlıklarla başlatıldığında ilk epoch'larda devasa gradyanlar üretir. Omurga dondurulmadan yüksek LR ile eğitilirse önceden öğrenilmiş ImageNet ağırlıkları silinir. Çözüm: Önce 3-5 epoch başlık eğitilmeli (Warmup), ardından omurganın kilidi açılmalıdır.
2. **Normalizasyon Uyuşmazlığı:** Pretrained modeller mutlaka ImageNet ortalama ve standart sapması ile beslenmelidir:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## 🛠️ Dizin Yapısı

```
day-23-transfer-learning/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # PyTorch, Torchvision, scikit-learn, OpenCV vb.
├── ana_akis.py                      # Uçtan uca kıyaslama ve deney yürütme betiği
├── README.md                        # Detaylı teorik ve mentorluk dokümantasyonu
├── src/
│   ├── __init__.py
│   ├── model_secici.py              # ResNet18 & EfficientNet-B0 omurga ve başlık kurucu
│   ├── veri_hazirlayici.py          # ImageNet uyumlu Dataset ve DataLoader yönetimi
│   ├── egitici.py                   # Ayrıştırılmış LR destekli TransferEgitici
│   ├── karsilastirici.py            # 4 modlu karşılaştırmalı deney motoru
│   └── gorsellestirici.py           # 4 panelli analiz ve parametre verimliliği raporu
├── testler/
│   ├── __init__.py
│   └── test_transfer_learning.py    # 7 adet kapsamlı birim test
└── ciktilar/
    └── transfer_ogrenme_raporu.png  # Öğrenme eğrileri, parametre ve doğruluk grafiği
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

> **Soru:** Transfer öğreniminde omurga (backbone) katmanlarını ince ayar (Fine-Tuning) yaparken neden sınıflandırma başlığından (Classification Head) $10\times$ veya $100\times$ daha düşük bir öğrenme oranı (`lr_backbone = 1e-5`, `lr_head = 1e-3`) kullanılmalıdır?

> **Mentor Cevabı:**
> 1. **Önceden Eğitilmiş Ağırlıkların Yıkımı (Catastrophic Forgetting):** Rastgele ilklendirilen yeni sınıflandırma başlığı başlangıçta çok büyük gradyanlar üretir. Eğer omurga katmanları da bu büyük gradyanlarla güncellenirse, ImageNet üzerinde milyonlarca görüntüyle öğrenilen genel görsel öznitelikler (kenarlar, dokular) birkaç adımda bozulur.
> 2. **Ayrıştırılmış Öğrenme Oranı (Discriminative Fine-Tuning):** Başlığın hızlıca hedef alanın sınıflarına adapte olması sağlanırken, omurga katmanlarının sadece çok ince nüanslarla ayarlanması modelin aşırı öğrenmesini önler ve genelleme performansını artırır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
