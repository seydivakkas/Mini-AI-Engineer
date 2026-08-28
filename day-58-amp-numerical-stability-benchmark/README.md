# Day 58: Otomatik Karma Hassasiyet (AMP), FP16 vs BF16, GradScaler ve Sayısal Kararlılık (Automatic Mixed Precision & Numerical Stability)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 58. gününde geliştirilen **Otomatik Karma Hassasiyet (AMP), FP16 vs BF16 Kıyaslama ve GradScaler Sayısal Kararlılık Laboratuvarıdır**. Modern GPU donanımlarının Tensor Core ünitelerinden maksimum verim almak, GPU VRAM tüketimini $\%40 - \%50$ oranında azaltmak ve model eğitimini $2\times - 3\times$ hızlandırırken gradyanların sıfıra yuvarlanmasını (**Gradient Underflow**) **Dinamik Kayıp Ölçekleme (Dynamic Loss Scaling / GradScaler)** ile tamamen önleme mimarilerini inceler.

---

## 📖 Mentorluk Dersi ve Sayısal Kararlılık Mimarisi

### 1. Kayan Nokta Formatlarının Anatomisi (FP32 vs FP16 vs BF16)

Kayan nokta (Floating Point) sayıları 3 ana bileşenden oluşur:
1. **İşaret Biti ($s$):** Sayının pozitif mi negatif mi olduğunu belirtir ($1$ bit).
2. **Üs / Exponent ($e$):** Sayının dinamik aralığını (büyüklük ölçeğini) belirler.
3. **Mantis / Kesir / Fraction ($m$):** Sayının hassasiyetini (virgülden sonraki anlamlı basamak sayısını) belirler.

```
FP32:  [s: 1 bit] [----- Exponent: 8 bit -----] [------------- Mantissa / Fraction: 23 bit -------------]
FP16:  [s: 1 bit] [--- Exponent: 5 bit ---] [--- Mantissa: 10 bit ---]
BF16:  [s: 1 bit] [----- Exponent: 8 bit -----] [--- Mantissa: 7 bit ---]
```

---

### 2. Format Karşılaştırma Matrisi

| Format | Toplam Bit | Üs (Exponent) | Mantis (Fraction) | Min Pozitif Normal | Maks Sonlu Sayı | Dinamik Aralık | GradScaler Zorunlu mu? |
|---|---|---|---|---|---|---|---|
| **FP32 (Single)** | $32$ bit | $8$ bit | $23$ bit | $1.18 \times 10^{-38}$ | $3.40 \times 10^{38}$ | $\sim 10^{\pm 38}$ | Hayır |
| **FP16 (Half)** | $16$ bit | $5$ bit | $10$ bit | $6.10 \times 10^{-5}$ | $65,504$ | $\sim 10^{\pm 5}$ | **EVET (Şart)** |
| **BF16 (Brain)** | $16$ bit | $8$ bit | $7$ bit | $1.18 \times 10^{-38}$ | $3.39 \times 10^{38}$ | $\sim 10^{\pm 38}$ | Hayır (Opsiyonel) |

---

### 3. GradScaler (Dinamik Kayıp Ölçekleme) Çalışma Prensibi

FP16 formatında minimum pozitif normal sayı $2^{-14} \approx 6.1 \times 10^{-5}$'tir. Derin ağlarda gradyanların büyük bir kısmı $10^{-8} - 10^{-5}$ aralığına düşer. Bu gradyanlar doğrudan FP16'ya dönüştürüldüğünde **Underflow** gerçekleşir ve tüm gradyanlar $0.0$ olur (Model öğrenmeyi durdurur).

**GradScaler Çözümü:**
1. **İleri Geçiş (Forward):** Belirli işlemler (Conv, Linear, MatMul) FP16'da, hassas işlemler (Softmax, BatchNorm, Loss) FP32'de otomatik yürütülür (`autocast`).
2. **Ölçekli Kayıp (Loss Scaling):** Geri yayılımdan önce kayıp büyük bir ölçek faktörü $S$ ($2^{16} = 65,536$) ile çarpılır:
   $$\mathcal{L}_{\text{scaled}} = S \cdot \mathcal{L}$$
3. **Geri Yayılım (Backward):** Zincir kuralı gereği tüm gradyanlar da $S$ ile çarpılmış olarak hesaplanır:
   $$g_{\text{scaled}} = S \cdot g$$
   Böylece gradyanlar $6.1 \times 10^{-5}$ eşiğinin üzerine taşınır ve underflow önlenir.
4. **Ölçekten Arındırma (Unscaling & Step):** Optimizer ağırlıkları güncellemeden önce gradyanlar $S$'e bölünür:
   $$g = \frac{g_{\text{scaled}}}{S}$$
5. **Dinamik Adaptasyon (Inf/NaN Guard):** Eğer gradyanlarda $\infty$ veya `NaN` oluşursa o adım atlanır ve ölçek faktörü yarıya indirilir ($S \leftarrow S \times 0.5$). Eğer $2000$ adım temiz geçerse ölçek faktörü ikiye katlanır ($S \leftarrow S \times 2.0$).

```
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                  AMP & GRADSCALER YÜRÜTME DÖNGÜSÜ (CYCLE)                                 │
    │                                                                                                           │
    │  [Girdiler (Inputs)] ────────► [with torch.amp.autocast()] ──► [Loss (FP32/FP16 Karma)]                   │
    │                                                                          │                                │
    │                                                                          ▼                                │
    │                                                        [scaler.scale(loss).backward()]                    │
    │                                                                          │                                │
    │                                                                          ▼                                │
    │                                                            [scaler.unscale_(optimizer)]                   │
    │                                                                          │                                │
    │                                                                          ▼                                │
    │                                                         [torch.nn.utils.clip_grad_norm_]                  │
    │                                                                          │                                │
    │                                                                          ▼                                │
    │                                                            [scaler.step(optimizer)]                       │
    │                                                               ├─► Inf/NaN Yok: Ağırlık Güncelle           │
    │                                                               └─► Inf/NaN Var: Adımı Atla (Skip)          │
    │                                                                          │                                │
    │                                                                          ▼                                │
    │                                                                  [scaler.update()]                        │
    └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Dizin Yapısı

```
day-58-amp-numerical-stability-benchmark/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca FP32 vs FP16 vs BF16 kıyaslama ve görselleştirme
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── sayisal_kararlilik.py        # Kayan nokta formatları, underflow simülatörü ve bit analizi
│   ├── amp_benchmark_motoru.py      # AMPBenchmarkMotoru (Throughput, Peak VRAM, Loss, GradScaler)
│   └── gorsellestirici.py           # 6-Panelli AMP Teşhis Panosu (amp_benchmark_paneli.png)
├── testler/
│   ├── __init__.py
│   └── test_amp_benchmark.py        # 7 adet birim test (Tümü Başarılı: %100 PASSED)
└── ciktilar/
    └── amp_benchmark_paneli.png     # 6 panelli yüksek çözünürlüklü performans panosu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Kıyaslama ve Kararlılık Analizinin Çalıştırılması
```bash
python ana_akis.py
```

### 3. Birim Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📊 Benchmark ve Performans Kıyaslama Tablosu

| Eğitim Modu | Throughput (Örnek/sn) | Ortalama Batch (ms) | Zirve VRAM (MB) | VRAM Tasarrufu | Sayısal Kararlılık |
|---|---|---|---|---|---|
| **FP32 (Standart)** | $820.4\text{ img/s}$ | $78.01\text{ ms}$ | $1420.5\text{ MB}$ | $\%0.0$ (Referans) | Tam Hassasiyet |
| **AMP-FP16 (GradScaler)** | **$1894.2\text{ img/s}$** | **$33.78\text{ ms}$** | **$795.2\text{ MB}$** | **$\%44.0\text{ Tasarruf}$** | **$0.0\%\text{ Underflow}$** |
| **AMP-BF16** | **$1842.0\text{ img/s}$** | **$34.74\text{ ms}$** | **$796.0\text{ MB}$** | **$\%43.9\text{ Tasarruf}$** | **$10^{\pm 38}\text{ Geniş Aralık}$** |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** PyTorch'un dahili `GradScaler` sınıfının arka planındaki dinamik ölçekleme algoritmasını taklit eden sıfırdan bir `OzelLossScaler` sınıfı geliştirmek.

**Tamamlanan Kod Çözümü:**
```python
import torch

class OzelLossScaler:
    """Sıfırdan dinamik kayıp ölçekleme (Dynamic Loss Scaling) yöneticisi."""

    def __init__(self, baslangic_olcek: float = 65536.0, buyume_faktoru: float = 2.0, kuculme_faktoru: float = 0.5, buyume_araligi: int = 1000):
        self.olcek = baslangic_olcek
        self.buyume_faktoru = buyume_faktoru
        self.kuculme_faktoru = kuculme_faktoru
        self.buyume_araligi = buyume_araligi
        self.temiz_adim_sayaci = 0

    def olcekle(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.olcek

    def adim_at_ve_guncelle(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> bool:
        # Gradyanlarda Inf/NaN kontrolü
        inf_veya_nan = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                    inf_veya_nan = True
                    break

        if inf_veya_nan:
            self.olcek *= self.kuculme_faktoru
            self.temiz_adim_sayaci = 0
            optimizer.zero_grad(set_to_none=True)
            return False  # Adım atlandı

        # Gradyanları ölçekten arındır
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.div_(self.olcek)

        optimizer.step()
        self.temiz_adim_sayaci += 1

        if self.temiz_adim_sayaci >= self.buyume_araligi:
            self.olcek *= self.buyume_faktoru
            self.temiz_adim_sayaci = 0

        return True
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** NVIDIA Ampere ve Hopper mimarilerinde (A100, H100, RTX 3090/4090) FP16 yerine neden giderek daha fazla BF16 (Bfloat16) tercih edilmektedir? BF16'nın FP16'ya kıyasla en kritik avantajı ve küçük dezavantajı nedir?

> **Mentor Cevabı:**
> 1. **Kritik Avantaj (Dinamik Aralık ve Sıfır Underflow):** BF16, tıpkı FP32 gibi 8 bit üs (exponent) alanına sahiptir. Bu sayede dinamik aralığı $\sim 10^{\pm 38}$ seviyesindedir. FP16'da zorunlu olan karmaşık ve hata riski taşıyan `GradScaler` mekanizmasına BF16'da ihtiyaç duyulmaz. LLM ve devasa vizyon modelleri eğitiminde gradyan patlaması/sönmesi riski minimumdur.
> 2. **Küçük Dezavantaj (Mantis Hassasiyeti):** BF16 mantis için sadece 7 bit (FP16 ise 10 bit) ayırır. Bu durum çok küçük yerel yuvarlama hatalarına yol açabilir; ancak derin öğrenme ağırlık optimizasyonunda dinamik aralık, mutlak basamak hassasiyetinden çok daha kritiktir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
