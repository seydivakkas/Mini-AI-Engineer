# Day 21: PyTorch ile Derin Öğrenme Görsel Sınıflandırma (PyTorch CNN Classifier)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![CUDA Ready](https://img.shields.io/badge/CUDA-Ready-76B900.svg?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; derin öğrenme araştırmalarının ve endüstri standardı yapay zeka üretim hatlarının merkezinde yer alan **PyTorch** çatısını kullanarak sıfırdan çok sınıflı bir Görsel Sınıflandırıcı (CNN) inşa eder. `nn.Module`, `Dataset`, `DataLoader`, `AdamW`, `CosineAnnealingLR`, `Gradient Clipping`, `EarlyStopping` ve **Grad-CAM** açıklanabilirlik kancalarını (Hooks) uçtan uca hayata geçirir.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve PyTorch Mimarisi
PyTorch, dinamik hesaplama grafiği (**Dynamic Computational Graph / Eager Execution**) ve Pythonik nesne yönelimli yapısıyla modern yapay zeka araştırmalarının ve üretim sistemlerinin omurgasını oluşturur.
- **`torch.nn.Module`:** Katmanların, ağırlıkların ve ileri geçiş (`forward`) mantığının kapsüllendiği temel yapı taşıdır.
- **`torch.utils.data.Dataset & DataLoader`:** Bellek taşmalarını önlemek için veriyi diskten veya bellekten mini-batch'ler halinde, çok çekirdekli asenkron kuyruklarla besler.

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **`nn.Module`** | *PyTorch Base Module* | Tüm sinir ağı katmanlarının, parametrelerinin ve ileri yayılım mantığının (`forward`) tanımlandığı temel PyTorch sınıfı. |
| **Hesaplama Grafı (Autograd)** | *Dynamic Computational Graph* | Geriye doğru otomatik türev almak (`loss.backward()`) için tensor operasyonlarının dinamik olarak kaydedildiği yönlü çizge. |
| **Ağırlık Güncelleme** | *Optimizer Step (`optimizer.step()`)* | Hesaplanan gradyanları ve öğrenme oranını kullanarak model parametrelerini güncelleyen optimizasyon adımı. |
| **Cihaz Yönetimi** | *Device Placement (`.to(device)`)* | Tensörlerin ve model ağırlıklarının CPU veya GPU (CUDA) belleğine transfer edilmesi. |

---

## 2. Tensör Düzenleri: PyTorch ($NCHW$) vs TensorFlow ($NHWC$)
Görsel tensörlerinin bellekteki ardışık yerleşimi:
- **PyTorch Formatı ($N, C, H, W$):** Batch, Channels, Height, Width. GPU donanımları (özellikle NVIDIA cuDNN kütüphaneleri) kanal bazında evrişimi $NCHW$ bellek formatında çok daha yüksek paralel önbellek verimiyle işler.
- **TensorFlow Formatı ($N, H, W, C$):** Batch, Height, Width, Channels.
- **Dönüşüm Kuralı:** NumPy $[H, W, C]$ görseli PyTorch'a aktarılırken `np.transpose(img, (2, 0, 1))` işlemi ile $[C, H, W]$ formatına dönüştürülmelidir.

---

### 3. Matematiksel Temeller ve Optimizasyon Dinamikleri

#### A. Ağırlık İlklendirmesi (Kaiming / He Normal Initialization)
ReLU aktivasyonu negatif değerleri sıfırladığı için sinir ağının her katmanında sinyalin varyansı yarıya iner. Standart Xavier/Glorot ilklendirmesi yerine **He/Kaiming** ilklendirmesi kullanılır:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right)$$

Bu yöntem, katmanlar ne kadar derin olursa olsun ileri ve geri geçişte gradyanların patlamasını veya sönümlenmesini matematiksel olarak engeller.

#### B. AdamW (Decoupled Weight Decay)
Klasik Adam optimizasyonunda L2 regülarizasyonu gradyan momentlerine karışarak ağırlıkların etkin bir şekilde küçültülmesini engeller. **AdamW (Loshchilov & Hutter, 2019)** ağırlık çürümesini doğrudan ağırlık güncelleme adımında uygular:

$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

#### C. Cosine Annealing Öğrenme Oranı Zamanlayıcısı
Öğrenme oranını eğitim boyunca bir kosinüs eğrisi boyunca kademeli olarak düşürür:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\max}}\pi\right)\right)$$

#### D. Grad-CAM (Gradient-Weighted Class Activation Mapping)
Modelin bir $c$ sınıfı tahmininde son evrişim katmanının $k$. kanalındaki aktivasyon haritası $A^k$ için kanal önem katsayısı $\alpha_k^c$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

- Pozitif katkı sağlayan pikseller $\text{ReLU}$ ile süzülür ve orijinal görselin üzerine ısı haritası olarak bindirilir.

---

### 4. Kritik Mühendislik Tuzakları
1. **`model.train()` vs `model.eval()` Ayrımı:**
   - `model.train()` modunda `BatchNorm` mini-batch istatistiklerini hesaplar ve `Dropout` nöronları rastgele sıfırlar.
   - `model.eval()` modunda `BatchNorm` kayıtlı hareketli ortalamaları (`running_mean`, `running_var`) kullanır ve `Dropout` tamamen kapatılır. Unutulursa test tahminleri tamamen tutarsızlaşır.
2. **`optimizer.zero_grad()` Unutulması:**
   - PyTorch'ta gradyanlar varsayılan olarak birikir (`accumulate`). Her iterasyon başında sıfırlanmazsa önceki batch'lerin gradyanları üst üste eklenir ve optimizasyon yönü sapar.
3. **`with torch.no_grad():` Eksikliği:**
   - Doğrulama veya test sırasında `torch.no_grad()` kullanılmazsa hesaplama grafiği (Autograd Graph) bellekte tutulmaya devam eder ve VRAM/RAM sızıntısına (**OOM**) yol açar.

---

## 🛠️ Dizin Yapısı

```
day-21-pytorch-cnn-classifier/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # PyTorch, Torchvision, scikit-learn, OpenCV vb.
├── ana_akis.py                      # Uçtan uca eğitim, test ve Grad-CAM akışı
├── README.md                        # Detaylı teorik ve mentorluk dokümantasyonu
├── src/
│   ├── __init__.py
│   ├── model_mimari.py              # PyTorchVisionCNN & Kaiming ilklendirmesi
│   ├── veri_hazirlayici.py          # SentetikGorselDataset & VeriYoneticisi (DataLoader)
│   ├── egitici.py                   # PyTorchEgitici (Training Loop, EarlyStopping, CosineLR)
│   ├── gorsellestirici.py           # 4 panelli teşhis çizelgesi ve tahmin galerisi
│   └── grad_cam.py                  # PyTorch Kanca (Hook) tabanlı Grad-CAM XAI modülü
├── testler/
│   ├── __init__.py
│   └── test_pytorch_cnn.py          # 7 adet kapsamlı birim test
└── ciktilar/
    ├── pytorch_cnn_raporu.png       # Kayıp/Doğruluk/Karışıklık Matrisi teşhisi
    └── grad_cam_aciklanabilirlik.png # Grad-CAM ısı haritası ve görsel bindirme
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

> **Soru:** PyTorch eğitim döngüsünde `optimizer.zero_grad()`, `loss.backward()` ve `optimizer.step()` sıralaması neden kritiktir ve `model.eval()` ile `torch.no_grad()` birlikte neden kullanılmalıdır?

> **Mentor Cevabı:**
> 1. **Gradyan Birikmesi (Gradient Accumulation):** PyTorch'ta gradyanlar varsayılan olarak her backward çağrısında toplanır (accumulate edilir). Bu nedenle her adım başında `optimizer.zero_grad()` ile sıfırlanmalıdır. `loss.backward()` ters yayılım ile gradyanları hesaplar, `optimizer.step()` ise ağırlıkları günceller.
> 2. **`model.eval()` vs `torch.no_grad()`:** `model.eval()` Dropout ve BatchNorm gibi katmanların davranışını çıkarım moduna geçirir (BatchNorm hareketli ortalamaları kullanır, Dropout kapanır). `torch.no_grad()` ise Autograd hesaplama grafiği oluşturmayı durdurarak bellek tüketimini azaltır ve hız kazandırır. Tam çıkarım güvenliği için ikisi birlikte kullanılmalıdır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
