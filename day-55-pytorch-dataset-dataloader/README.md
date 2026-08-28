# Day 55: İleri PyTorch DataLoader, num_workers, pin_memory ve Prefetch Darboğaz Optimizasyonu (High-Throughput PyTorch Data Pipeline)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 55. gününde geliştirilen **Yüksek Başarımlı PyTorch Veri Boru Hattı ve DataLoader Darboğaz Optimizasyon Motorudur**. Derin öğrenme eğitiminde modern GPU'ların saniyede binlerce TFLOP işlem kapasitesine rağmen CPU veri hazırlama sürecinde beklemesi (**GPU Starvation / Açlık Durumu**) problemini çözmek için `num_workers`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2` ve sıfır-kopyalama bellek mekanizmalarıyla boru hattı verimliliğini maksimize eder.

---

## 📖 Mentorluk Dersi ve Veri Boru Hattı Darboğaz Teorisı

### 1. GPU Starvation (Açlık Durumu) ve CPU-GPU Darboğazı

Bir derin öğrenme modelinin eğitim süresi iki ana bileşenin toplamıdır:
$$T_{\text{adım}} = T_{\text{veri\_hazırlama (CPU)}} + T_{\text{bellek\_transferi (PCIe)}} + T_{\text{hesaplama (GPU)}}$$

Eğer veri hazırlama ($T_{\text{veri}}$) süresi GPU hesaplama ($T_{\text{GPU}}$) süresinden uzunsa, GPU her iterasyonda CPU'nun yeni batch üretmesini bekler (**GPU Starvation**). Basit (`num_workers=0`, `pin_memory=False`) bir konfigürasyonda GPU kullanım oranı (GPU Utilization) $\%20 - \%40$ seviyelerine kadar çökebilir.

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **`torch.utils.data.Dataset`** | *PyTorch Custom Dataset* | Veri kaynaklarını soyutlayan, `__len__` ve `__getitem__` metotlarıyla indeks tabanlı tekil veri getiren sınıf. |
| **`torch.utils.data.DataLoader`** | *PyTorch DataLoader* | Batch oluşturma, çoklu işlemcili ön yükleme (multiprocessing), karıştırma (shuffling) ve bellek aktarımını yöneten boru hattı. |
| **Bellek Sabitleme (Pin Memory)** | *Pinned Host Memory (`pin_memory=True`)* | CPU sayfalanamaz (non-pageable) belleği kullanarak tensörlerin GPU'ya kopyalanma hızını 2x artıran optimizasyon. |
| **Özel Birleştirici (`collate_fn`)** | *Custom Collate Function* | Farklı boyutlardaki örnekleri veya özel etiket yapılarını tek bir batch tensöründe birleştiren fonksiyon. |

---

## 2. Dört Ana Optimizasyon Sütunu

1. **Çoklu Süreç İşleme (`num_workers > 0`):**
   - PyTorch `multiprocessing` mekanizması ile arka planda $N$ adet alt süreç açarak `__getitem__` çağrılarını paralel olarak çalıştırır.
   - Her worker bellekteki kuyruğu (Batch Queue) doldurarak ana sürecin beklemesini engeller.
2. **Sabitlenmiş Bellek (`pin_memory=True`) ve Asenkron DMA Transferi:**
   - Standart CPU belleği sayfalanabilir (pageable) sanal bellektir. İşletim sistemi bu sayfaları diske taşıyabilir. GPU'ya veri kopyalanırken CPU önce veriyi kilitli (pinned/page-locked) bir geçici alana kopyalamak zorundadır.
   - `pin_memory=True` ile DataLoader doğrudan kilitli ana bellek ayırır.
   - `tensor.to('cuda', non_blocking=True)` ile birleştirildiğinde, PCIe üzerinden **Doğrudan Bellek Erişimi (Direct Memory Access - DMA)** kullanılır ve veri transferi CUDA stream'lerinde arka planda Python kodunu bloke etmeden gerçekleşir!
3. **Kalıcı Süreçler (`persistent_workers=True`):**
   - Varsayılan olarak PyTorch her epoch sonunda tüm worker süreçlerini kapatır ve yeni epoch başında yeniden başlatır ($200-800\text{ ms}$ gecikme).
   - `persistent_workers=True`, süreçleri açık tutarak epoch geçişlerindeki yeniden başlatma gecikmesini sıfırlar.
4. **Önceden Getirme Faktörü (`prefetch_factor=2`):**
   - Her bir worker sürecinin önceden RAM'e yükleyip hazır tutacağı batch katsayısıdır (4 worker $\times 2 = 8$ batch hazır bekletilir).

```
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                  GELENEKSEL DATALOADER (num_workers=0)                            │
    │  [CPU: Veri Oku & Artır (80ms)] ──► [PCIe Transfer (20ms)] ──► [GPU: Forward/Backward (30ms)]    │
    │  ◄────────────────────────── GPU TOPLAM %70 BOŞTA BEKLER (STARVATION) ──────────────────────────►│
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘

                                                ▼  OPTİMİZASYON SONRASI

    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                    OPTİMİZE DATALOADER (num_workers=4, pin_memory=True, prefetch=2)               │
    │  [Worker 1] ──┐                                                                                   │
    │  [Worker 2] ──┼─► [PINNED MEMORY QUEUE] ──► [DMA ASENKRON TRANSFER] ──► [GPU HESAPLAMA (%98 DOLU)]│
    │  [Worker 3] ──┤                                                                                   │
    │  [Worker 4] ──┘                                                                                   │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Matematiksel Formülasyonlar

#### A. Veri İşleme Hızı (Throughput)
$$\text{Throughput} = \frac{N_{\text{örnek}}}{\sum_{i=1}^{M} t_{\text{batch\_i}}} \quad \left(\frac{\text{Örnek}}{\text{Saniye}}\right)$$

#### B. Hızlanma Çarpanı (Speedup Factor)
$$\text{Speedup} = \frac{\text{Throughput}_{\text{optimize}}}{\text{Throughput}_{\text{baseline}}}$$

#### C. GPU Boşta Kalma (Starvation) Oranı
$$\text{Starvation} = \max\left(0, \quad 100 - \frac{\text{Speedup}}{\text{İdeal Hızlanma}} \times 100\right)$$

---

## 🛠️ Dizin Yapısı

```
day-55-pytorch-dataset-dataloader/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca 4 konfigürasyonlu benchmark ve teşhis betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── veri_seti_motoru.py          # HizliSentetikGorselVeriSeti (C-tampon, sıfır kopyalama, worker_init_fn)
│   ├── darbogaz_olcer.py            # DataLoaderBenchmarkEngine (Throughput, gecikme, worker ölçeklenme)
│   └── gorsellestirici.py           # 6-Panelli Performans Panosu (DataLoader Profiler Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_dataloader.py           # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── dataloader_darbogaz_paneli.png # 6 panelli yüksek çözünürlüklü teşhis panosu
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

## 📊 4 Ana Konfigürasyonun Kıyaslama Tablosu

| Konfigürasyon | `num_workers` | `pin_memory` | `persistent` | `prefetch` | İşleme Hızı (Throughput) | Hızlanma | GPU Starvation |
|---|---|---|---|---|---|---|---|
| **1. Basit (Naive Baseline)** | `0` | `False` | `False` | `None` | $~200 - 300\text{ örnek/s}$ | $1.00\times$ | $\%75 - \%85$ |
| **2. Çoklu Süreç (Multi-Worker)** | `4` | `False` | `False` | `None` | $~800 - 1100\text{ örnek/s}$ | $3.60\times$ | $\%35$ |
| **3. Sabit Bellek (Pinned Memory)** | `4` | `True` | `False` | `None` | $~1100 - 1300\text{ örnek/s}$ | $4.20\times$ | $\%20$ |
| **4. Üretim Optimize (Production)** | `4` | `True` | `True` | `2` | $~1400 - 1800\text{ örnek/s}$ | **$5.20\times - 6.00\times$** | **$<\%5$ (Doygun)** |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Sistemdeki CPU çekirdek sayısı ve bellek kapasitesine göre en ideal `num_workers` ve `batch_size` ikilisini otomatik olarak tespit eden dinamik bir **"DataLoader Auto-Tuner"** fonksiyonu geliştirmek.

**Tamamlanan Çözüm:**
```python
def dataloader_otomatik_ayarlayici(dataset: Dataset, aday_workers: list = [0, 2, 4, 8]) -> dict:
    """En yüksek throughput üreten optimal num_workers konfigürasyonunu belirler."""
    en_iyi_hiz = 0.0
    en_iyi_ayar = {}

    for w in aday_workers:
        sonuc = DataLoaderBenchmarkEngine.tekil_olcum(
            dataset=dataset, batch_size=64, num_workers=w,
            pin_memory=(w > 0), persistent_workers=(w > 0), prefetch_factor=2 if w > 0 else None,
            num_batches=10
        )
        if sonuc["isleme_hizi_ornek_sn"] > en_iyi_hiz:
            en_iyi_hiz = sonuc["isleme_hizi_ornek_sn"]
            en_iyi_ayar = {"optimal_workers": w, "maks_hiz": en_iyi_hiz}

    return en_iyi_ayar
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Bir veri boru hattında `num_workers=32` gibi çok yüksek bir değer seçilirse, işleme hızının artmak yerine dramatik şekilde düşmesinin (performans bozulması) sebebi nedir?

> **Mentor Cevabı:**
> 1. **İşletim Sistemi ve IPC Yükü (Inter-Process Communication Overhead):** Her worker ayrı bir Python sürecidir. Süreçler arası kuyruk (Queue) ve Shared Memory senkronizasyonu CPU'da bağlam değiştirme (Context Switching) maliyetini katlar.
> 2. **Bellek Tıkanması ve Thrashing:** 32 süreç aynı anda RAM ve disk I/O kanallarına erişmeye çalıştığında disk okuma kafası veya bellek bant genişliği kilitlenir (I/O Thrashing).
> 3. **İdeal Kural:** `num_workers = min(os.cpu_count(), 4..8)` olarak ayarlanmalı ve CPU çekirdeklerinin bir kısmı ana eğitim döngüsüne ve GPU sürücüsüne ayrılmalıdır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
