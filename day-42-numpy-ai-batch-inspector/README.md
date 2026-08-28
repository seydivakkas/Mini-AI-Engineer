# Day 42: Üretim Girdi Tensörleri Doğrulama & Batch Boyutu / NaN-Inf-Shape Anomali Tespiti (NumPy AI Batch Inspector)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** serimizin açılış projesidir. Üretim ortamında (Triton Inference Server, TorchServe, FastAPI, TensorRT, ONNX Runtime) çalışan derin öğrenme modellerine beslenen girdi tensörlerinin **Şekil (Shape)**, **Dinamik Batch Boyutu**, **Veri Tipi (Dtype)**, **Sayısal Kararsızlık (NaN/Inf)**, **Değer Aralığı (Out-of-Range)** ve **Bellek Düzeni (C-Contiguity / Strides)** anomalilerini mikro-saniye seviyesinde teftiş eden ve düzelten bir **Yapay Zeka Tensör Doğrulama & Koruma Kalkanı (AI Tensor Guardrail)** motorudur.

---

## 📖 Mentorluk Dersi ve Çekirdek Mühendislik Teorisı

### 1. Üretim Modellerinde Sessiz Bozulmalar ve CUDA Çökmeleri

Derin öğrenme modelleri GPU üzerinde çalışırken CPU seviyesindeki gibi ayrıntılı istisna (exception) fırlatamazlar. Girdi tensöründeki anomaliler iki büyük felakete yol açar:

1. **Sessiz Tahmin Bozulması (Silent Model Degradation):**
   - Normalizasyon hatası nedeniyle $[-1.0, 1.0]$ bekleyen modele $[0, 255]$ ham RGB tensörü girdiğinde, aktivasyon fonksiyonları (Softmax, Sigmoid, GeLU) aşırı doyar (saturation) ve model her girdi için tek bir sınıfa %99 güvenle yanlış tahmin üretir.
   - Tek bir `NaN` veya `Inf` değeri, matris çarpımları boyunca yayılarak (NaN Propagation) tüm batch tahminini `[NaN, NaN, ...]` haline getirir.

2. **Donanım ve CUDA Kernel Çökmeleri:**
   - Beklenmeyen boyutlar veya asenkron CUDA kernel çağrılarında sınır dışı indeksleme `CUDA error: device-side assert triggered` hatasına yol açar. Bu hata Triton veya TorchServe worker sürecini (process) tamamen kilitler ve tüm sunucunun yeniden başlatılmasını gerektirir.
   - İstemciden gelen kontrolsüz büyük bir batch boyutu ($B=128$), GPU üzerinde ani bellek patlamasına (CUDA Out of Memory - OOM) sebep olur.

```
                           ┌──────────────────────────────────────────────────────────┐
                           │   İSTEMCİ / EDGE KAMERA / REST API GİRDİSİ (Ham Dizi)    │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      AIBatchDenetleyici (NumPy Vektörize Teftiş Motoru)                           │
    │  - Şekil Doğrulama: Rank kontrolü, NCHW vs NHWC tespit, Dinamik Batch [1, B_max]                  │
    │  - Sayısal Bütünlük: np.isnan().sum(), np.isinf().sum(), Alt-normal float analizi                 │
    │  - İstatistik & Aralık: Min, Max, Ortalama, Std, Değer Kırpma İhtiyacı                            │
    │  - Bellek & Layout: C-Contiguous, Byte Hizalama, Bellek Ayak İzi (MB)                             │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
    ┌──────────────────────────┐                                  ┌──────────────────────────┐
    │  KRİTİK REDDEDİLENLER     │                                  │  DÜZELTİLEBİLİR UYARI    │
    │  - NaN / Inf Değerleri   │                                  │  - NHWC Kanal Düzeni     │
    │  - Batch Boyutu > B_max  │                                  │  - Aralık Dışı Değerler  │
    │  - Rank Uyuşmazlığı      │                                  │  - float64 Dtype         │
    │  - Bellek Limiti Aşımı   │                                  └────────────┬─────────────┘
    └──────────────────────────┘                                               │
                                                                               ▼
                                                                  ┌──────────────────────────┐
                                                                  │      BatchTemizleyici    │
                                                                  │  - Transpose NHWC->NCHW  │
                                                                  │  - np.clip(min, max)     │
                                                                  │  - np.ascontiguousarray  │
                                                                  │  - Dtype float32 casting │
                                                                  └────────────┬─────────────┘
                                                                               │
                                                                               ▼
                                                                  ┌──────────────────────────┐
                                                                  │ GÜVENLİ TENSÖR GİRİŞİ    │
                                                                  │ (Model Inference Engine) │
                                                                  └──────────────────────────┘
```

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Bellek Adım Boyu (Strides)** | *Array Strides* | NumPy dizisinde bir sonraki satıra veya kanala geçmek için bellekte kaç bayt atlanması gerektiğini tanımlayan demet. |
| **Boyut Yayınlama (Broadcasting)** | *Tensor Broadcasting Rules* | Farklı şekillerdeki tensörlerin bellek kopyalaması yapmadan sanal olarak genişletilerek eleman bazlı işlenmesi. |
| **Sayısal Sağlık Denetimi** | *NaN/Inf Sanitization* | Eğitim veya çıkarım batch'lerinde oluşan tanımsız (`NaN`) veya sonsuz (`Inf`) değerlerin anında tespiti ve maskelenmesi. |
| **Vektörize Batch İşlemleri** | *Vectorized Batch Statistics* | Python döngüsü kullanmadan tüm batch için eksen bazlı ortalama, varyans ve norm hesaplama. |

---

## 2. Vektörize Teftiş Matematik ve Bellek Algoritmaları

- **Bellek Ayak İzi (Memory Footprint):**
  $$\text{Bellek (MB)} = \frac{\prod_{i=1}^{d} S_i \times \text{itemsize}}{1024^2}$$
- **Sayısal Kararsızlık Analizi:**
  $$N_{\text{NaN}} = \sum \mathbb{I}(\text{isnan}(x)), \quad N_{\text{Inf}} = \sum \mathbb{I}(\text{isinf}(x))$$
- **Aralık Dışı Değer Tespiti:**
  $$N_{\text{Aralık Dışı}} = \sum \mathbb{I}(x < v_{\text{min}} \lor x > v_{\text{max}})$$
- **Sabit Sensör Doygunluk Kontrolü:**
  $$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2} < 10^{-6} \implies \text{Kör Kamera / Sabit Sensör Uyarısı}$$

---

## 🛠️ Dizin Yapısı

```
day-42-numpy-ai-batch-inspector/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # 4 farklı üretim senaryosunu teftiş eden ana betik
├── README.md                        # 220+ satır teorik ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── sema.py                      # TensorSemasi ve TensorSekilKurali tanımları
│   ├── denetleyici.py               # AIBatchDenetleyici (Şekil, NaN/Inf, Aralık, Bellek)
│   ├── temizleyici.py               # BatchTemizleyici (Transpose, Clip, Casting, Contiguity)
│   └── gorsellestirici.py           # 6 panelli teşhis panosu (Inspector Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_batch_inspector.py      # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── tensor_denetim_paneli.png    # 6 panelli yüksek çözünürlüklü teftiş panosu
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

## 📊 4 Üretim Senaryosu ve Teftiş Kararları

| Senaryo | Girdi Tensör Özelliği | Teftiş Kararı | Uygulanan Aksiyon |
|---|---|---|---|
| **1. Golden Batch** | `(8, 3, 224, 224)`, `float32`, `[-2.8, +2.8]` | **GECERLI (PASS)** | Doğrudan modele gönderildi. |
| **2. Düzeltilebilir Uyarı** | `(4, 224, 224, 3)` (NHWC), `float64`, Outlier `5.8` | **DUZELTILEBILIR_UYARI** | Transpose NCHW, Clip `[-3, 3]`, Cast `float32`. |
| **3. Sayısal Bozulma** | `(2, 3, 224, 224)`, NaN ve Inf Değerleri | **KRITIK_RED (REJECT)** | CUDA çökmesini önlemek için derhal reddedildi. |
| **4. Batch Aşımı** | `(64, 3, 224, 224)` ($B=64 > 32$) | **KRITIK_RED (REJECT)** | GPU OOM bellek taşmasını önlemek için reddedildi. |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/denetleyici.py` içerisine FastAPI veya Triton sunucularında HTTP 422 Unprocessable Entity cevabına dönüştürülebilecek bir **"FastAPI Guardrail Middleware Wrapper"** fonksiyonu eklemek.

**Tamamlanan Çözüm:**
```python
def fastapi_guardrail_dogrula(tensor: np.ndarray, sema: TensorSemasi) -> dict:
    denetleyici = AIBatchDenetleyici(sema)
    rapor = denetleyici.denetle(tensor)
    if not rapor["guvenli_gecis"]:
        raise ValueError(f"HTTP 422 Unprocessable Entity: {rapor['ihlaller'][0]['mesaj']}")
    if rapor["karar"] == "DUZELTILEBILIR_UYARI":
        temizleyici = BatchTemizleyici(sema)
        temiz_tensor, _ = temizleyici.temizle_ve_uyarla(tensor, rapor)
        return {"status": "SANITIZED", "tensor": temiz_tensor}
    return {"status": "VALID", "tensor": tensor}
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** PyTorch modeline `np.transpose((0, 3, 1, 2))` ile NHWC'den NCHW'ye çevrilmiş bir NumPy dizisini doğrudan `torch.from_numpy()` ile verdiğimizde neden `RuntimeError: default_collate: batch must contain tensors, numpy arrays numbers, dicts or lists` veya strided copy uyarıları alırız ve bunu engellemek için neden `np.ascontiguousarray()` zorunludur?

> **Mentor Cevabı:**
> NumPy'da `transpose` işlemi bellekteki piksellerin fiziksel yerini değiştirmez; sadece dizinin **adım uzunluklarını (strides)** manipüle ederek yeni bir bellek görünümü (view) oluşturur.
> Ancak PyTorch ve CUDA tensör çekirdekleri, matris çarpımlarında (GEMM) ve konvolüsyonlarda ardışık bellek blokları (**C-Contiguous**) üzerinde en yüksek bant genişliğine ulaşır. Eğer tensör non-contiguous ise PyTorch arka planda zorunlu ve pahalı bir `tensor.contiguous()` kopyası yapar veya bazı C++ C-API binding'lerinde doğrudan bellek adresi hatası fırlatır. `np.ascontiguousarray()` çağırmak, diziyi bellekte sıralı tek bir blok haline getirerek GPU transfer süresini (PCIe aktarımını) %40'a varan oranda hızlandırır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
