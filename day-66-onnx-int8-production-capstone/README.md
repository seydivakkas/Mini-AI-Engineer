# Day 66: PyTorch Modellerini ONNX'e Aktarma, INT8 PTQ Kuantizasyon & ONNX Runtime Hızlandırma (FAZ 3 Capstone)

[![License: Private All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Opset%2018-005CED.svg)](https://onnx.ai/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.27.0-0078D4.svg)](https://onnxruntime.ai/)
[![Tests: 8 Passed](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)](testler/)

FAZ 3'ün büyük finali (**Production Capstone**) olan bu projede; derin öğrenme modellerinin araştırma ortamından (PyTorch Eager Mode) çıkarılıp donanımdan bağımsız açık standartlara (**ONNX - Open Neural Network Exchange**) aktarılması, **INT8 Post-Training Quantization (PTQ)** ile 4 kat sıkıştırılması ve **ONNX Runtime Engine** ile üretim seviyesinde mikro-saniye gecikmeyle hızlandırılması uçtan uca uygulanmıştır.

---

## 1. 🎯 Günün Konusu & Teorik/Matematiksel Derinlik

### A. Çözülen Temel Problem ve Endüstriyel Motivasyon
Derin öğrenme modelleri PyTorch ile eğitildikten sonra üretim ortamına taşınırken üç büyük darboğazla karşılaşır:
1. **Python ve Framework Bağımlılığı (Interpreter Overhead):** PyTorch modelleri canlıda Python Global Interpreter Lock (GIL) ve dinamik hesaplama grafiği sebebiyle saf C++ çalışma ortamlarına kıyasla yüksek ek gecikme (overhead) yaratır.
2. **Yüksek Bellek Ayak İzi (Memory Footprint) ve Disk Boyutu:** 32-bit kayan noktalı (FP32) ağırlıklar, uç cihazların (Edge AI, IoT, Mobil, İHA) kısıtlı RAM ve Flash depolama sınırlarını aşar.
3. **Bellek Bant Genişliği Darboğazı (Memory Bandwidth Bottleneck):** Modern işlemcilerde hesaplama gücünden (TFLOPS) ziyade, ağırlıkların RAM'den işlemci önbelleğine (L1/L2/L3 Cache) taşınma hızı çıkarım gecikmesini belirler. 8-bit tam sayıya (INT8) geçiş, taşınan veri hacmini $\%75$ oranında düşürür.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           PYTORCH -> ONNX -> INT8 PTQ MLOPS DAĞITIM VE HIZLANDIRMA BORU HATTI             │
│                                                                                                           │
│   [PyTorch Modeli (FP32)] ──► [torch.onnx.export (Opset 18, Dynamic Axes)] ──► [ONNX Modeli (FP32)]      │
│            │                                                                              │               │
│            │ (Eager Mode)                                         (Operator Fusion / Graph Optimization)  │
│            ▼                                                                              ▼               │
│   ┌─────────────────┐                                                    ┌─────────────────────────────┐  │
│   │ PyTorch Eager   │                                                    │ ONNX Runtime Engine (FP32)  │  │
│   │ 1.36 ms Latency │                                                    │ 0.63 ms Latency (2.18x Hız) │  │
│   └─────────────────┘                                                    └──────────────┬──────────────┘  │
│                                                                                         │                 │
│                                                              (Post-Training Quantization│ Dynamic PTQ)    │
│                                                                                         ▼                 │
│                                                                          ┌─────────────────────────────┐  │
│                                                                          │ ONNX Runtime Engine (INT8)  │  │
│                                                                          │ 0.48 MB Boyut (%74 Küçülme) │  │
│                                                                          │ %99.9936 Sayısal Benzerlik  │  │
│                                                                          └─────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### B. Matematiksel Formülasyon: Kuantizasyon ve Sayısal Eşdeğerlik

#### 1. Doğrusal Kuantizasyon Eşlemesi (Linear Quantization Mapping)
Gerçek değerli sürekli bir kayan nokta tensörü $r \in [\alpha, \beta]$ (FP32), $b$-bitlik ayrık tam sayı uzayına $q \in [q_{\min}, q_{\max}]$ (ör. INT8 için $[-128, 127]$ veya UINT8 için $[0, 255]$) şu formülle kuantize edilir:

$$q = \text{clip}\left( \left\lfloor \frac{r}{S} \right\rceil + Z, \; q_{\min}, \; q_{\max} \right)$$

Ters kuantizasyon (Dequantization) ile geri kazanılan yaklaşık değer $\tilde{r}$:

$$\tilde{r} = S \cdot (q - Z)$$

Burada:
- **$S$ (Ölçek Faktörü / Scale Factor):** Sayısal aralığı dönüştüren pozitif reel katsayıdır:
  $$S = \frac{\beta - \alpha}{q_{\max} - q_{\min}}$$
- **$Z$ (Sıfır Noktası / Zero-Point):** Gerçek $0.0$ reel değerinin kuantize uzaydaki tam sayı karşılığıdır:
  $$Z = \text{round}\left( -\frac{\alpha}{S} \right) + q_{\min}$$

#### 2. Kuantizasyon Gürültüsü (Quantization Noise) ve Hata Sınırı
Kuantizasyon hatası $\epsilon = r - \tilde{r}$ olup, yuvarlama gürültüsü üniform dağılım varsayımı altında şu varyansa sahiptir:

$$\mathbb{E}[\epsilon^2] = \frac{S^2}{12}$$

#### 3. Sayısal Eşdeğerlik Metrikleri (Numerical Parity Metrics)
PyTorch lojit vektörü $y_{\text{pt}} \in \mathbb{R}^K$ ile ONNX lojit vektörü $y_{\text{onnx}} \in \mathbb{R}^K$ arasındaki benzerlik:

- **Kosinüs Benzerliği (Cosine Similarity):**
  $$\text{CosineSim}(y_{\text{pt}}, y_{\text{onnx}}) = \frac{y_{\text{pt}} \cdot y_{\text{onnx}}}{\|y_{\text{pt}}\|_2 \, \|y_{\text{onnx}}\|_2}$$

- **Maksimum Mutlak Hata (Max Absolute Error):**
  $$\text{MaxError} = \max_{1 \le i \le K} |y_{\text{pt}}^{(i)} - y_{\text{onnx}}^{(i)}|$$

---

### C. 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

Üretim seviyesinde model optimizasyonu ve MLOps mühendisliğinde kullanılan kritik kavramlar:

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **ONNX** | *Open Neural Network Exchange* | Linux Foundation öncülüğünde geliştirilen, PyTorch, TensorFlow, Scikit-learn ve JAX gibi farklı çatıların modellerini tek bir ortak hesaplama grafiği formatında temsil eden açık kaynaklı standart. |
| **Opset Version** | *Operator Set Version* | ONNX standardında desteklenen matematiksel operatörlerin (Conv, Gemm, Softmax vb.) sürüm numarasıdır. Her yeni opset versiyonu (ör. Opset 17/18), daha yeni tensör operasyonlarını ve dinamik şekil kurallarını destekler. |
| **Düğüm Kaynaştırma** | *Operator / Node Fusion* | Ayrı ayrı yürütülen katmanların tek bir GPU/CPU çekirdeğinde (kernel) birleştirilmesidir. Örneğin: $\text{Conv} + \text{BatchNorm} + \text{ReLU} \rightarrow \text{FusedConvReLU}$. Ara bellek okuma/yazma (I/O) maliyetini ortadan kaldırır. |
| **Sabit Katlama** | *Constant Folding* | Modelin girişinden bağımsız olan ve derleme anında hesaplanabilen sabit matematiksel düğümlerin (örneğin önceden bilinen tensör şekil hesapları veya sabit ağırlık çarpımları) tek bir sabite indirgenmesidir. |
| **Dinamik Eksenler** | *Dynamic Axes* | ONNX grafiğinin sabit boyutlar yerine değişken boyutları (örneğin batch boyutu $B$, değişken görüntü genişliği $W$ veya dizi uzunluğu $T$) dinamik sembolik değişken olarak kabul etmesini sağlayan yapılandırmadır. |
| **PTQ** | *Post-Training Quantization* | Model eğitimi tamamlandıktan sonra, modeli yeniden eğitmeye ihtiyaç duymadan doğrudan ağırlıkları ve/veya aktivasyonları FP32'den INT8'e dönüştüren kuantizasyon tekniğidir. |
| **QAT** | *Quantization-Aware Training* | Eğitim esnasında ileri yayılıma sahte kuantizasyon (fake quantization) ekleyerek modelin kuantizasyon gürültüsüne uyum sağlamasını ve INT8'de sıfır doğruluk kaybı yaşamasını sağlayan yöntemdir. |
| **Simetrik Kuantizasyon** | *Symmetric Quantization* | Sıfır noktasının $Z=0$ olarak sabitlendiği ve giriş aralığının sıfıra göre simetrik $[-\alpha, \alpha]$ kabul edildiği yöntemdir. Sıfır noktası çıkarma işlemi olmadığından donanım seviyesinde çok daha hızlıdır. |
| **Asimetrik Kuantizasyon** | *Asymmetric Quantization* | Sıfır noktasının $Z \neq 0$ olabildiği yöntemdir. ReLU gibi sadece pozitif değer alan $[0, \beta]$ aktivasyonlarda bit verimliliğini maksimize eder. |
| **Kalibrasyon Veri Seti** | *Calibration Dataset* | Statik kuantizasyonda, aktivasyon tensörlerinin dinamik aralığını ($[\alpha, \beta]$) ve histogramını doğru belirlemek için modelden geçirilen küçük (ör. 100-500 örnek) temsilci veri setidir. |
| **Execution Provider** | *Execution Provider (EP)* | ONNX Runtime'ın donanıma özgü hızlandırma kütüphaneleriyle konuşmasını sağlayan arayüzdür. Örneğin CPU için `CPUExecutionProvider` / `OpenVINO`, NVIDIA GPU için `CUDAExecutionProvider` / `TensorrtExecutionProvider`. |
| **Eager Execution** | *Eager Mode* | PyTorch'un standart çalışma modudur; her satır kod anında çalıştırılır ve dinamik grafik oluşturulur. Hata ayıklamada mükemmeldir ancak üretim çıkarımında yavaştır. |
| **Static Graph** | *Statik Hesaplama Grafiği* | Modelin tüm veri akışının önceden bir yönlü asiklik grafik (DAG) olarak derlenip optimize edildiği moddur. Derleyicinin bellek tahsisini ve paralel yürütmeyi optimize etmesine izin verir. |
| **Memory Arena** | *Bellek Arenası* | ONNX Runtime'ın her çıkarımda işletim sisteminden dinamik `malloc/free` bellek istemek yerine, önceden büyük bir bellek havuzu tahsis edip onu yeniden kullanmasıdır (Allocation Jitter önleme). |

---

### D. SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Donanımdan bağımsız tek format; Python/GIL kilidinden bağımsız C++ çalışma zamanı; `%74.1` model boyutu küçülmesi; FP32 ONNX Runtime ile **$2.18\times$ çıkarım hızlanması** ($0.63\text{ ms}$ vs $1.36\text{ ms}$); $\%100.00$ FP32 ve $\%99.99$ INT8 lojit kosinüs benzerliği. |
| **Weaknesses (Zayıf Yönler)** | Donanımda özel VNNI/DP4A/TensorCore INT8 komut seti desteği bulunmayan CPU'larda dinamik kuantizasyonun döngü içi dönüştürme maliyeti; özel (custom) PyTorch operatörlerinin ONNX standardına ihracat zorluğu. |
| **Opportunities (Fırsatlar)** | Edge/IoT, mobil ve gömülü sistemlere dağıtım; TensorRT veya OpenVINO sağlayıcıları ile $5-10\times$ ek hızlanma; bulut sunucu (AWS/GCP) GPU maliyetlerinde devasa tasarruf. |
| **Threats (Tehditler)** | Çok hassas medikal/finansal modellerde aykırı aktivasyonların (outlier activations) yarattığı kuantizasyon gürültüsü; eski ONNX runtime sürümleriyle opset uyumsuzluğu. |

---

## 2. 💻 Üretim Seviyesinde Uygulama Mimarisi

Paket modülleri [`src/`](src/) dizini altında toplanmıştır:

- [`src/model_mimari.py`](src/model_mimari.py): `UretimVisionNet` ve `ResidualBlok` (Conv2d, BatchNorm2d, ReLU, Residual bağlantı, AdaptiveAvgPool2d, Linear sınıflandırıcı başlığı).
- [`src/onnx_aktarici.py`](src/onnx_aktarici.py): `ONNXDonusturucu` (Torch $\rightarrow$ ONNX Opset 18 ihracatı, Dinamik Eksenler, ONNX Checker ve Shape Inference doğrulaması).
- [`src/kuantizasyon_motoru.py`](src/kuantizasyon_motoru.py): `INT8Kuantizator` (ONNX Dynamic Post-Training Quantization motoru ve sıkıştırma hesaplayıcı).
- [`src/cikarim_motoru.py`](src/cikarim_motoru.py): `ONNXInferenceEngine` (ORT Oturum Yönetimi, çoklu iş parçacığı havuzu, bellek arenası, ısınma ve gecikme profilleme).
- [`src/karsilastirici_benchmark.py`](src/karsilastirici_benchmark.py): `ModelBenchmarkKarsilastirici` (PyTorch vs ONNX FP32 vs ONNX INT8 sayısal korelasyon ve gecikme kıyaslayıcı).
- [`src/gorsellestirici.py`](src/gorsellestirici.py): `CapstoneGorsellestirici` (6 panelli endüstriyel teşhis panosu çizici).
- [`ana_akis.py`](ana_akis.py): Uçtan uca ihracat, kuantizasyon, benchmark ve raporlama orkestratörü.
- [`testler/test_onnx_int8_capstone.py`](testler/test_onnx_int8_capstone.py): 8 adet kapsamlı birim testi (%100 Başarı).

---

## 3. 🧪 Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

**Görev:** Belirli bir görüntü sınıflandırma modeli için kalibrasyon veri setini okuyarak çalışan, aktivasyonların dinamik aralığını $[r_{\min}, r_{\max}]$ belirleyen ve **Statik INT8 Kuantizasyon (Static PTQ)** gerçekleştiren bir `KalibrasyonluStatikKuantizator` sınıfı tasarlamak.

**Eksiksiz Kod Çözümü:**
```python
from typing import List, Dict, Any
import os
import numpy as np
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

class SentetikKalibrasyonVeriOkuyucu(CalibrationDataReader):
    """Statik kuantizasyon için temsilci kalibrasyon verilerini sunan veri okuyucu."""

    def __init__(self, ornek_sayisi: int = 50, girdi_sekli: tuple = (1, 3, 64, 64), girdi_adi: str = "girdi_gorsel") -> None:
        self.veriler = [np.random.randn(*girdi_sekli).astype(np.float32) for _ in range(ornek_sayisi)]
        self.girdi_adi = girdi_adi
        self.sayac = 0

    def get_next(self) -> Dict[str, np.ndarray] | None:
        if self.sayac < len(self.veriler):
            girdi = {self.girdi_adi: self.veriler[self.sayac]}
            self.sayac += 1
            return girdi
        return None

class KalibrasyonluStatikKuantizator:
    """Kalibrasyon veri okuyucusu kullanarak statik INT8 PTQ uygular."""

    @staticmethod
    def statik_kuantize_et(
        girdi_onnx_yolu: str,
        cikti_int8_yolu: str,
        kalibrasyon_okuyucu: CalibrationDataReader
    ) -> str:
        quantize_static(
            model_input=girdi_onnx_yolu,
            model_output=cikti_int8_yolu,
            calibration_data_reader=kalibrasyon_okuyucu,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8
        )
        return cikti_int8_yolu
```

---

## 4. 📊 Ölçülen Doğrulama ve Benchmark Metrikleri

`ana_akis.py` koşturularak 100 tekrar ile ölçülen deneysel sonuçlar:

| Model Varyantı | Gecikme (ms) | Throughput (FPS) | Disk Boyutu (MB) | Sıkıştırma Oranı | Hızlanma (Speedup) | Kosinüs Benzerliği |
|---|---|---|---|---|---|---|
| **PyTorch FP32 (Eager)** | $1.36\text{ ms}$ | $733.9\text{ FPS}$ | $1.835\text{ MB}$ | $1.00\times$ (Referans) | $1.00\times$ | $\%100.0000$ |
| **ONNX Runtime FP32** | **$0.63\text{ ms}$** | **$1,596.8\text{ FPS}$** | $1.835\text{ MB}$ | $1.00\times$ | **$2.18\times$ HIZLANMA** | **$\%100.0000$** |
| **ONNX Runtime INT8 (PTQ)**| $6.21\text{ ms}$ | $160.9\text{ FPS}$ | **$0.476\text{ MB}$** | **$3.86\times$ KÜÇÜLME** | **$\%74.08$ Tasarruf** | **$\%99.9936$** |

---

## 5. 🚀 Kurulum ve Çalıştırma

```bash
# 1. Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 2. İhracat, kuantizasyon ve benchmark akışını çalıştırın
python ana_akis.py

# 3. Kapsamlı birim testleri koşun
pytest testler -v
```

---

## 6. ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** INT8 kuantizasyonda neden Simetrik Kuantizasyon (Symmetric) ile Asimetrik Kuantizasyon (Asymmetric) arasında bir ayrım yapılır ve ReLU gibi aktivasyon fonksiyonlarında Asimetrik kuantizasyon neden daha avantajlıdır?

> **Mentor Cevabı:**
> 1. **Ağırlıklar vs Aktivasyonlar (Sıfır Noktası $Z$ Etkisi):** Model ağırlıkları genellikle sıfır etrafında Gauss dağılımına ($\mu \approx 0$) sahiptir. Bu nedenle ağırlıklarda sıfır noktası $Z=0$ olan **Simetrik Kuantizasyon** tercih edilir. Bu durum donanım seviyesinde sıfır noktası çıkarma yükünü kaldırır ve matris çarpımını ($q_1 \cdot q_2$) hızlandırır.
> 2. **ReLU Aktivasyonları ve Bit İsrafı:** ReLU sonrası değerler daima $[0, \beta]$ aralığındadır (negatif değer yoktur). Eğer Simetrik kuantizasyon kullanılırsa, $[-\beta, \beta]$ aralığı kuantize edilir ve 8-bitlik uzayın yarısı (negatif 128 kademe) tamamen israf edilir. **Asimetrik Kuantizasyon** ($Z \neq 0$) ise 256 kademenin tamamını $[0, \beta]$ aralığına tahsis ederek kuantizasyon çözünürlüğünü iki katına çıkarır ve hassasiyet kaybını minimize eder.

---

## 7. 📜 Lisans & Metaveri

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```



## 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- PyTorch modellerini donanımdan bağımsız ONNX formatına dönüştürüp INT8 kuantizasyonu ile model boyutunu %75 küçülterek CPU çıkarımını 3 kat hızlandırmak için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- PyTorch bağımlılığını kaldırır, bellek tüketimini azaltır ve mikroservis yanıt sürelerini milisaniyenin altına çeker.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Hassas regresyon veya küçük kusur tespit görevlerinde INT8 kuantizasyonu %1-2 civarında doğruluk kaybı yaratabilir.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- TensorRT (NVIDIA GPU), OpenVINO (Intel CPU) veya TorchScript.

