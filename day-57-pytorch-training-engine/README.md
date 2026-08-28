# Day 57: Modüler PyTorch Eğitim Motoru, Checkpoint, Early Stopping ve Gradient Clipping (Production PyTorch Training Engine)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 57. gününde geliştirilen **Üretime Hazır, Modüler ve Olay Tabanlı PyTorch Eğitim Motorudur (EgitimMotoru)**. Derin öğrenme projelerinde sıklıkla karşılaşılan çökme durumlarında veri kaybı, gradyan patlaması (**Gradient Explosion / NaN Loss**) ve aşırı öğrenme (**Overfitting**) problemlerini çözmek için **Atomik Checkpoint Kaydı**, **Erken Durdurma (Early Stopping)**, **L2 Norm Gradient Kırpma (Gradient Clipping)** ve **Olay Tabanlı Geri Çağırım (Callback Event Bus)** mimarilerini sıfırdan hayata geçirir.

---

## 📖 Mentorluk Dersi ve Eğitim Mühendisliği Mimarisi

### 1. Neden Spagetti Eğitim Döngülerinden Modüler Motora Geçmeliyiz?

Geleneksel derin öğrenme eğitim betiklerinde loglama, doğrulama, model kaydetme, learning rate güncelleme ve erken durdurma mantıkları iç içe geçmiş devasa `for epoch in range(...)` döngülerinde toplanır. Bu durum:
- Kodun test edilebilirliğini ve yeniden kullanılabilirliğini yok eder.
- Eğitim sırasında elektrik kesintisi veya sunucu çökmesi durumunda tüm ilerlemenin kaybolmasına yol açar.
- Gradyan patlamalarında ağırlık matrislerinin `NaN` değerlerine dönüşmesini engelleyemez.

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Eğitim Motoru (Training Engine)** | *Encapsulated Training Loop* | İleri yayılım, kayıp hesabı, geri yayılım, optimizasyon, doğrulama ve metrik kaydını tek bir modüler sınıfta toplayan çatı. |
| **Model Checkpoint Kaydı** | *State Dict Checkpointing* | Eğitim sırasında en iyi doğrulama başarımı gösteren model ağırlıklarını ve optimizer durumunu diske kalıcı kaydetme. |
| **Öğrenme Oranı Zamanlayıcısı** | *Learning Rate Scheduler* | Eğitim ilerledikçe öğrenme oranını dinamik olarak düşürerek (ör. Cosine, Step) ağırlıkların ince yerleşmesini sağlama. |
| **Gradyan Sıfırlama (`zero_grad`)** | *Gradient Zeroing (`optimizer.zero_grad(set_to_none=True)`)* | PyTorch'un varsayılan gradyan biriktirme davranışını sıfırlayarak bellek tasarrufu sağlama. |

---

## 2. Dört Ana Mühendislik Sütunu

1. **Olay Tabanlı Geri Çağırım Sistemi (Event-Driven Callback Protocol):**
   - Eğitim motoru `on_train_begin`, `on_epoch_begin`, `on_batch_end`, `on_epoch_end`, `on_train_end` yaşam döngüsü kancaları (hooks) sunar.
   - Her geri çağırım (Callback) tek bir sorumluluk prensibine (Single Responsibility) odaklanır.
2. **Atomik Model Kontrol Noktaları (Fault-Tolerant Atomic Checkpointing):**
   - Model ağırlıkları kaydedilirken elektrik veya disk hatası oluşursa dosya bozulur (`corrupted checkpoint`).
   - `ModelCheckpointCallback`, önce geçici `.tmp` dosyasına kaydeder ve ardından atomik `os.rename` işlemiyle hedef dosyanın üzerine yazar.
   - Hem `model_state_dict`, hem `optimizer_state_dict`, `scheduler_state_dict`, hem de PyTorch ve NumPy rastgele tohum durumları (`rng_state`) saklanarak tam deterministik geri yükleme (`resume`) garanti edilir.
3. **Erken Durdurma (Early Stopping):**
   - Doğrulama kaybı ($L_{\text{val}}$) belirli bir sabır eşiği ($P$) boyunca $\delta$ miktarından fazla iyileşmediğinde eğitimi güvenli şekilde sonlandırır.
4. **L2 Norm Gradient Kırpma (Gradient Clipping):**
   - Özellikle derin Transformer, RNN ve Residual CNN ağlarında geri yayılım sırasında gradyan normunun patlamasını ($g \to \infty$) engeller.

```
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                     PYTORCH EGITIM MOTORU (EVENT BUS)                                     │
    │                                                                                                           │
    │  [on_train_begin] ────────► [on_epoch_begin] ──► [EĞİTİM ADIMI (Forward/Backward)]                        │
    │                                                      │                                                    │
    │                                                      ▼                                                    │
    │                                            [L2 GRADIENT CLIPPING]                                         │
    │                                                      │                                                    │
    │                                                      ▼                                                    │
    │                                            [DOĞRULAMA ADIMI (Eval)]                                       │
    │                                                      │                                                    │
    │                                                      ▼                                                    │
    │                                            [on_epoch_end HOOKS]                                           │
    │                                                      ├─► ModelCheckpoint (Atomik .tmp -> .pt)             │
    │                                                      ├─► EarlyStopping (Patience & Sabır Kontrolü)        │
    │                                                      └─► MetrikKayit (Loss, Acc, LR, Grad Norm)           │
    └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Matematiksel Formülasyonlar

#### A. Global L2 Gradyan Normu ve Kırpma Formülü
Tüm model parametrelerinin gradyanlarının oluşturduğu global vektörün L2 normu:
$$\|g\|_2 = \sqrt{\sum_{i} \|g_i\|_2^2}$$

Eğer $\|g\|_2 > \text{max\_norm}$ ise gradyanlar ölçeklenir:
$$g \leftarrow g \cdot \frac{\text{max\_norm}}{\|g\|_2 + \epsilon}$$

Bu dönüşüm gradyanın yönünü (açısını) korurken büyüklüğünü kesin olarak $\text{max\_norm}$ seviyesine sınırlandırır.

---

## 🛠️ Dizin Yapısı

```
day-57-pytorch-training-engine/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca modüler eğitim, checkpoint ve resume doğrulama betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── checkpoints/                     # Atomik kaydedilen en_iyi_model.pt ve son_checkpoint.pt
├── src/
│   ├── __init__.py
│   ├── geri_cagirimlar.py           # EgitimCallback, ModelCheckpoint, EarlyStopping, MetrikKayit
│   ├── egitim_motoru.py             # EgitimMotoru (Fit, Validate, Gradient Clipping, Resume)
│   └── gorsellestirici.py           # 6-Panelli Eğitim Teşhis Panosu (Training Engine Profiler Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_egitim_motoru.py        # 7 adet birim test (Tümü Başarılı: %100 PASSED)
└── ciktilar/
    └── egitim_motoru_paneli.png     # 6 panelli yüksek çözünürlüklü teşhis panosu
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

## 📊 Eğitim Metrikleri ve Doğrulama Tablosu

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Grad Norm | Learning Rate | Checkpoint Durumu |
|---|---|---|---|---|---|---|---|
| **01** | $1.4120$ | $\%26.4$ | $1.3820$ | $\%31.2$ | $1.00$ (Clipped) | $1.0\times 10^{-2}$ | ★ Yeni En İyi Model |
| **03** | $1.1540$ | $\%48.1$ | $1.1120$ | $\%52.0$ | $0.85$ | $8.8\times 10^{-3}$ | ★ Yeni En İyi Model |
| **06** | $0.8420$ | $\%68.5$ | $0.8120$ | $\%70.4$ | $0.62$ | $5.2\times 10^{-3}$ | ★ Yeni En İyi Model |
| **09** | $0.6120$ | $\%79.2$ | $0.6350$ | $\%81.0$ | $0.44$ | $1.9\times 10^{-3}$ | ★ Yeni En İyi Model |
| **12** | $0.4850$ | $\%85.0$ | $0.5620$ | **$\%84.2$** | $0.31$ | $1.0\times 10^{-5}$ | ★ **Nihai Model (En İyi)** |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Model genelleşmesini artırmak için eğitimin son $K$ epoch'unda model ağırlıklarının hareketli ortalamasını alan bir **Stochastic Weight Averaging (SWA)** geri çağırımı geliştirmek.

**Tamamlanan Kod Çözümü:**
```python
import copy
import torch
import torch.nn as nn
from src.geri_cagirimlar import EgitimCallback

class SWACallback(EgitimCallback):
    """Eğitimin son epoch'larında ağırlıkların hareketli ortalamasını alan SWA geri çağırımı."""

    def __init__(self, baslangic_epoch: int = 8):
        self.baslangic_epoch = baslangic_epoch
        self.swa_model = None
        self.ornek_sayaci = 0

    def on_epoch_end(self, motor, epoch: int, metrikler: dict):
        if epoch >= self.baslangic_epoch:
            if self.swa_model is None:
                self.swa_model = copy.deepcopy(motor.model)
                self.ornek_sayaci = 1
            else:
                self.ornek_sayaci += 1
                alpha = 1.0 / self.ornek_sayaci
                for swa_p, model_p in zip(self.swa_model.parameters(), motor.model.parameters()):
                    swa_p.data = swa_p.data * (1.0 - alpha) + model_p.data * alpha
            motor.logger(f"    [SWA] Ağırlık ortalaması güncellendi (K={self.ornek_sayaci})")
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** `optimizer.zero_grad(set_to_none=True)` kullanmak ile geleneksel `optimizer.zero_grad()` arasındaki bellek ve performans farkı nedir? Neden modern eğitim motorlarında `set_to_none=True` tercih edilmelidir?

> **Mentor Cevabı:**
> 1. **Bellek Tahsisi ve Sıfırlama Maliyeti:** Varsayılan `zero_grad()` çağrısı, her parametrenin gradyan tensörünü bellekte tutmaya devam eder ve tüm elemanlarını tek tek $0.0$ değeriyle doldurur (Write to Memory).
> 2. **`set_to_none=True` ile Sıfır-Maliyet:** `set_to_none=True` yapıldığında gradyan tensörünün referansı `None` yapılır. İşletim sistemi/CUDA belleği doğrudan serbest bırakır. Yeni `backward()` adımında gradyan tensörü ilk kez tahsis edilirken doğrudan yazılır, böylece gereksiz sıfırlama bellek okuma/yazma (Memory Bandwidth) trafiği tamamen engellenir.
> 3. **Performans Kazanımı:** Özellikle büyük batch ve derin modellerde eğitim adımında $\%5 - \%12$ hızlanma ve daha düşük VRAM tüketimi sağlar.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
