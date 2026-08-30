# 🧠 Day 324: Neuromorphic Hardware Mapping (Intel Loihi 2 & SynSense)

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![Phase 17](https://img.shields.io/badge/Phase-17%3A%20Neuromorphic%20AI%20%26%20BCI-blueviolet?style=flat-square)
![Status: Completed](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

> 🎓 **Stajyer Notu & Mentor Rehberi:** Harika gidiyorsun! Yazılımsal Spiking Sinir Ağlarını (SNN) ve STDP kurallarını öğrendikten sonra şimdi işin en heyecanlı donanım boyutuna adım atıyoruz: **Nöromorfik Çip Eşleme (Neuromorphic Hardware Mapping)**. PyTorch'ta eğittiğimiz bir SNN modelini **Intel Loihi 2** veya **SynSense Speck** gibi fiziksel silikon nöro-çekirdek matrislerine ($M \times N$ Neuro-Core Mesh) nasıl böleceğimizi, ağırlıkları nasıl INT8 sabitleştirilmiş (fixed-point) formata dönüştüreceğimizi ve çekirdekler arası AER paket yönlendirmesini adım adım göreceğiz!

---

## 🎯 1. Günün Konusu & Teorik/Matematiksel Derinlik

### 1.1 Intel Loihi 2 Mimarisi ve Mesh Çip Tasarımı

Standart GPU donanımlarında (ör. NVIDIA H100/A100) devasa bir Von-Neumann bellek-işlemci ayrımı varken; **Intel Loihi 2** çipinde binlerce bağımsız **Neuro-Core (Nöro-Çekirdek)** bulunur. Her çekirdek kendi yerel SRAM belleğinde sinaptik ağırlıklarını saklar ve asenkron **Network-on-Chip (NoC)** yönlendiricisi ile diğer çekirdeklere bağlanır.

```text
┌─────────────────────────────────────────────────────────────┐
│                 Intel Loihi 2 Neuro-Core Mesh               │
│  ┌──────────────┐   (AER Hop)   ┌──────────────┐            │
│  │ Neuro-Core 0 │ ────────────> │ Neuro-Core 1 │            │
│  │ (1024 Neurons│               │ (1024 Neurons│            │
│  │ INT8 Weights)│ <──────────── │ INT8 Weights)│            │
│  └──────────────┘               └──────────────┘            │
│         │                              │                    │
│         ▼                              ▼                    │
│  ┌──────────────┐               ┌──────────────┐            │
│  │ Neuro-Core 2 │ ────────────> │ Neuro-Core 3 │            │
│  └──────────────┘               └──────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

#### Sabitleştirilmiş (Fixed-Point) INT8 Kuantizasyonu
Loihi 2 çekirdekleri kayan noktalı (FP32) sayıları değil, düşük güçlü sabitleştirilmiş **INT8** tam sayıları işler:

$$W_{int8} = \text{clamp}\left( \text{round}(W_{fp32} \times S_W), -128, 127 \right)$$

Ölçekleme Faktörü (Scale Factor):

$$S_W = \frac{127.0}{\max(|W_{fp32}|)}$$

Sinyal-Kuantizasyon Gürültü Oranı (SQNR):

$$\text{SQNR (dB)} = 10 \log_{10} \left( \frac{\sum W_{fp32}^2}{\sum (W_{fp32} - \hat{W}_{fp32})^2 + \epsilon} \right)$$

---

### 1.2 AER (Address Event Representation) Paket Yönlendirmesi

Çekirdekler arasında spike sinyalleri iletilirken 1-bitlik aksiyon potansiyeli, kaynağı ve hedef çekirdeği belirten **AER (Address Event Representation)** veri paketlerine sarılır.

Çekirdekler arası Manhattan yönlendirme mesafesi (**Hop Distance**):

$$\text{Hop}(C_{src}, C_{dst}) = |x_{src} - x_{dst}| + |y_{src} - y_{dst}|$$

Yönlendirme gecikmesi hop mesafesiyle doğru orantılıdır: $t_{route} = \text{Hop} \times \tau_{hop}$.

---

### 1.3 4 Zorunlu Mimari Analiz

#### 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- **Extreme Edge Neuromorphic Deployment:** Biyomedikal implantlar, dronlar ve otonom araçlarda pili yıllarca tükenmeyen milivat (mW) seviyesinde çıkarım yapabilmek için.
- **On-Chip Local SRAM Efficiency:** Donanımda bellek taşıma (DRAM fetch) maliyetini sıfırlayan bellek içi hesaplama (In-Memory Computing).

#### 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- **Von Neumann Dönüşüm Darboğazı (Von Neumann Bottleneck):** İşlemci ve bellek arasındaki yüksek enerjili otobüs trafiğini tamamen ortadan kaldırır.
- **Katman Boyut Aşımı (Layer Oversize Constraint):** Büyük SNN katmanlarını fiziksel donanım çekirdeklerinin limitlerine ($max\_neurons$) göre otomatik bölümler (tiling).

#### ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- **Kuantizasyon Hassasiyet Kaybı:** FP32'den INT8'e dönüşümde SQNR düşük kalırsa model doğruluğu düşebilir.
- **Mesh Yönlendirme Tıkanıklığı (NoC Congestion):** Yanlış haritalamada hop mesafeleri uzarsa AER paket trafiği tıkanabilir.

#### 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- **NVIDIA TensorRT GPU Execution:** Yüksek güç harcayan FP16/INT8 matris hızlandırıcısı.
- **Intel Loihi 2 Mapping (Bizim Yaklaşımımız):** Çekirdek bölünmeli, ultra-düşük güçlü ($0.1 \text{ pJ/SOP}$) asenkron nöromorfik eşleme.
- **SynSense Speck / SpiNNaker:** Diğer nöromorfik donanım işlemci mimarileri.

---

### 1.4 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Terim | Tanım ve Stajyer Açıklaması |
| --- | --- |
| **Loihi 2** | Intel'in 2. nesil 4nm/N3 nöromorfik mikro-işlemci çipi. |
| **Neuro-Core** | Çip üzerinde kendi nöronlarını ve sinaps belleklerini barındıran bağımsız çekirdek. |
| **AER** | Address Event Representation: Spike ileten adrese dayalı donanım paketi. |
| **Manhattan Hop** | Mesh çip ızgarasında iki çekirdek arasındaki adımlı yönlendirme mesafesi. |
| **SQNR** | Signal-to-Quantization-Noise Ratio: Kuantizasyon sinyal-gürültü oranı (dB). |
| **Fixed-Point (INT8)** | Kayan nokta yerine sabit noktalı tam sayı aritmetiği. |
| **Tiling (Bölümleme)** | Büyük nöron katmanlarını küçük çekirdek bloklarına dağıtma. |
| **NoC** | Network-on-Chip: Çip üzerindeki asenkron çekirdekler arası iletişim ağı. |
| **SynSense Speck** | Görmeye odaklı olay tabanlı ultradüşük güçlü nöromorfik çip. |
| **Fan-in / Fan-out** | Donanım çekirdeğine giren ve çıkan maksimum bağlantı kapasitesi. |

---

### 1.5 SWOT Analizi Karar Matrisi

```
               GÜÇLÜ YÖNLER (STRENGTHS)                      ZAYIF YÖNLER (WEAKNESSES)
      ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
      │ • MicroJoule seviyesinde aşırı düşük güç. │  │ • Sabit noktalı INT8 kuantizasyonda      │
      │ • Von-Neumann bellek taşıma maliyeti 0. │  │   potansiyel hassasiyet kaybı.           │
      │ • Ölçeklenebilir mesh çip mimarisi.      │  │ • Çekirdek sayısının kısıtlı olması.     │
      └──────────────────────────────────────────┘  └──────────────────────────────────────────┘
               FIRSATLAR (OPPORTUNITIES)                       TEHDİTLER (THREATS)
      ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
      │ • İmplante edilebilir BCI protezleri.    │  │ • Kuantizasyon uyumsuzluğunda modelin    │
      │ • Uzay araçları ve nano-İHA çıkarımı.    │  │   tamamen bozulması.                     │
      └──────────────────────────────────────────┘  └──────────────────────────────────────────┘
```

---

## 💻 2. Üretim Seviyesinde Uygulama Kodu & Mimarisi

Dizin yapısı:

```text
day-324-neuromorphic-hardware-mapping-loihi/
├── ana_akis.py
├── gereksinimler.txt
├── LICENSE
├── README.md
├── ciktilar/
│   └── loihi_donanim_paneli.png
├── src/
│   ├── __init__.py
│   ├── loihi_mapper.py
│   ├── loihi_gorsellestirici.py
│   └── loihi_profilleyici.py
└── testler/
    └── test_loihi_mapper.py
```

---

## 🧪 3. Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

### 🎯 Görev (Stajyer Görevi)
PyTorch SNN ağırlık matrisi donanım çekirdeğine kuantize edilirken 8-bit yerine 4-bit INT4 formata ($[-8, 7]$) düşürüldüğünde SQNR (Sinyal Gürültü Oranı) değerinin kaç dB düştüğünü hesaplayan bir fonksiyon yazınız.

### 💡 Çözüm Kodu
```python
import numpy as np

def test_int4_vs_int8_sqnr():
    np.random.seed(42)
    w_fp32 = np.random.randn(64, 64).astype(np.float32)

    def compute_sqnr(w_fp32, bits=8):
        max_abs = np.max(np.abs(w_fp32)) + 1e-9
        max_int = (2 ** (bits - 1)) - 1
        scale = max_int / max_abs
        w_quant = np.clip(np.round(w_fp32 * scale), -max_int - 1, max_int)
        w_dequant = w_quant / scale
        signal = np.mean(w_fp32 ** 2)
        noise = np.mean((w_fp32 - w_dequant) ** 2) + 1e-9
        return 10.0 * np.log10(signal / noise)

    sqnr_8bit = compute_sqnr(w_fp32, bits=8)
    sqnr_4bit = compute_sqnr(w_fp32, bits=4)

    print(f"INT8 Kuantizasyon SQNR: {sqnr_8bit:.2f} dB")
    print(f"INT4 Kuantizasyon SQNR: {sqnr_4bit:.2f} dB")
    print(f"Düşüş Miktarı:          {sqnr_8bit - sqnr_4bit:.2f} dB (INT4 gürültüsü arttı!)")

if __name__ == "__main__":
    test_int4_vs_int8_sqnr()
```

---

## 📊 4. Donanım & Performans Benchmark Tablosu

| Metrik | Intel Loihi 2 Neuro-Core | NVIDIA GPU (FP16) | Donanım Kazancı |
| --- | --- | --- | --- |
| **Enerji / Operasyon** | **0.1 pJ / INT8 SOP** | 5.0 pJ / FP16 FLOP | **50.0x Daha Düşük Enerji** |
| **Bellek Mimarisi** | In-Memory Local SRAM | Global VRAM HBM | **Sıfır DRAM Taşıma** |
| **Kuantizasyon Hassasiyeti** | INT8 Fixed-Point | FP16 Floating Point | **Donanım Dostu Tam Sayı** |
| **Yönlendirme İletişimi** | AER Hop Mesh Router | NVLink Bus | **Asenkron Olay Tabanlı** |

---

## 📜 5. Lisans & Metaveri

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 201-Day AI, CV, LLM/RAG, Reasoning & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```

---

## ❓ 6. Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

### ❓ Soru
Bir SNN katmanındaki nöron sayısı bir Neuro-Core'un fiziksel limitinden (ör. 64 nöron) büyükse ne yapılır ve bu durum AER yönlendirme hop mesafesini nasıl etkiler?

### 💬 Mentorluk Yanıtı
Eğer SNN katmanı tek bir çekirdeğin kapasitesini aşıyorsa, **Layer Partitioning (Tiling)** tekniği uygulanır. Katman, matris satırları boyunca bölünerek birden fazla komşu Neuro-Core üzerine dağıtılır (ör. 220 nöron 4 çekirdeğe 64'erli olarak bölünür).

Bu durum çekirdekler arası iletişimi artırır. Presinaptik spike'lar tek bir çekirdeğe değil, AER yönlendiricisi üzerinden **Manhattan Hop** mesafesi kat ederek farklı grid koordinatlarındaki çekirdeklere paketlenip gönderilir. Haritalama algoritmasının amacı, birbiriyle sık iletişim kuran nöron gruplarını grid üzerinde fiziksel olarak en yakın komşu çekirdeklere (Hop = 1) yerleştirerek NoC tıkanıklığını önlemektir.
