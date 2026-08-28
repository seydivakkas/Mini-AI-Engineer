# 🔬 Day 97: MiniViT v1.0 Deterministik Çıkarım ve Donanımdan Bağımsız Doğrulama Testleri

Bu proje, **101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği Master Serisi** kapsamında; MiniViT v1.0 modelinin üretim ortamında bit-seviyesinde deterministik (tam tekrarlanabilir) çıkarım yapmasını sağlamak, CPU ve GPU donanımları arasındaki sayısal pariteyi ($L_\infty$) doğrulamak ve FP32 vs FP16 vs BF16 hassasiyet sapmalarını (precision drift) analiz etmek amacıyla geliştirilmiştir.

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)
![Deterministic](https://img.shields.io/badge/Bit--Level-Deterministic-success?style=flat-square)
![Status](https://img.shields.io/badge/Status-100%25%20Verified-blue?style=flat-square)

---

## 📑 İçindekiler
- [Teorik ve Matematiksel Temeller](#-teorik-ve-matematiksel-temeller)
  - [1. Neden Bu Sistem Kullanılır? (Mühendislik Gerekçesi)](#1-neden-bu-sistem-kullanılır-mühendislik-gerekçesi)
  - [2. Ne Gibi Sorunları Çözer? (Çözülen Darboğazlar)](#2-ne-gibi-sorunları-çözer-çözülen-darboğazlar)
  - [3. Ne Konuda Eksik Kalır? (Limitler & Riskler)](#3-ne-konuda-eksik-kalır-limitler--riskler)
  - [4. Alternatif Yaklaşımlar & Karşılaştırma](#4-alternatif-yaklaşımlar--karşılaştırma)
- [Matematiksel Formülasyon](#-matematiksel-formülasyon)
- [Mimari Yapı & Dosya Düzeni](#-mimari-yapı--dosya-düzeni)
- [Hızlı Başlangıç & Kurulum](#-hızlı-başlangıç--kurulum)
- [Testler ve Doğrulama](#-testler-ve-doğrulama)
- [Lisans](#-lisans)

---

## 🧠 Teorik ve Matematiksel Temeller

Derin öğrenme modelleri üretim ortamına alındığında en büyük operasyonel risklerden biri, aynı model ve aynı girdiyle yapılan çıkarımların zaman içinde veya farklı donanımlarda (CPU sunucusu vs GPU çıkarım düğümü) küçük de olsa sayısal farklılıklar göstermesidir. Bu "belirsizlik" (non-determinism), özellikle finansal tahmin, medikal tanı ve yasal regülasyona tabi yapay zeka sistemlerinde kabul edilemez bir kusurdur.

### 1. Neden Bu Sistem Kullanılır? (Mühendislik Gerekçesi)
- **Hata Ayıklama ve Regresyon Güvencesi:** Bir hata meydana geldiğinde aynı girdiyle hatanın %100 tekrarlanabilmesi (reproducibility) gerekir.
- **Donanımdan Bağımsız Test Edilebilirlik:** CI/CD sunucularında GPU bulunmadığında dahi CPU üzerinde yapılan testlerin GPU çıkarımıyla matematiksel olarak tutarlı ($L_\infty < 10^{-4}$) olduğu kanıtlanır.
- **Kuantizasyon ve Hassasiyet Analizi:** Modelin FP16 veya BF16 formatına çekildiğinde ne kadar sayısal sapmaya (drift) uğradığı Sinyal-Gürültü Oranı (SNR) ile nesnel olarak ölçülür.

### 2. Ne Gibi Sorunları Çözer? (Çözülen Darboğazlar)
- **cuDNN Heuristics Kaynaklı Rastlantısallık:** cuDNN varsayılan olarak en hızlı algoritmayı seçmek için asenkron ve paralel atomik eklemeler kullanır; bu da her çalıştırmada LSB (Least Significant Bit) seviyesinde rastgele kaymalara yol açar. `torch.use_deterministic_algorithms(True)` ve `CUBLAS_WORKSPACE_CONFIG=:4096:8` ile bu durum engellenir.
- **CPU vs GPU Çıkarım Uyuşmazlığı:** CPU (AVX-512/FMA) ve GPU (CUDA Cores) kayan nokta yuvarlama (IEEE 754) farkları denetlenir ve tolerans sınırları içine hapsedilir.

### 3. Ne Konuda Eksik Kalır? (Limitler & Riskler)
- **Çıkarım Performans Maliyeti:** Deterministik algoritmalar paralel optimizasyonları kısıtladığı için çıkarım süresinde yaklaşık %5-%15 oranında ek gecikme yaratabilir.
- **Farklı Donanım Mimarileri (ARM vs x86):** Fused Multiply-Add (FMA) komut setlerinin farklı donanımlarda $10^{-7}$ mertebesinde kaçınılmaz farklar üretmesi.

### 4. Alternatif Yaklaşımlar & Karşılaştırma

| Yaklaşım | Determinizm Düzeyi | Performans | Donanım Bağımsızlık |
|---|---|---|---|
| **Tam Deterministik PyTorch (Bizim)** | **Bit-Level (%100 Özdeş)** | **Yüksek (~2.8 ms)** | **Doğrulanmış Parite ($L_\infty < 10^{-4}$)** |
| **Varsayılan PyTorch İleri Geçiş** | Yaklaşık (~%99.9) | Çok Yüksek | Kontrolsüz |
| **ONNX Runtime Deterministic Mode** | Yüksek | Yüksek | Backend Bağımlı |
| **TensorRT Strict Determinism** | Yüksek | En Yüksek | Yalnızca NVIDIA GPU |

---

## 📐 Matematiksel Formülasyon

### 1. Sayısal Hata Normları ($L_1, L_2, L_\infty$)
Modelin CPU çıktısı $y_{\text{cpu}} \in \mathbb{R}^C$ ve GPU çıktısı $y_{\text{gpu}} \in \mathbb{R}^C$ arasındaki farklar şu normlarla ölçülür:

$$L_1(y_{\text{cpu}}, y_{\text{gpu}}) = \frac{1}{C} \sum_{i=1}^C |y_{\text{cpu}}^{(i)} - y_{\text{gpu}}^{(i)}|$$

$$L_2(y_{\text{cpu}}, y_{\text{gpu}}) = \sqrt{\frac{1}{C} \sum_{i=1}^C (y_{\text{cpu}}^{(i)} - y_{\text{gpu}}^{(i)})^2}$$

$$L_\infty(y_{\text{cpu}}, y_{\text{gpu}}) = \max_{i \in \{1, \dots, C\}} |y_{\text{cpu}}^{(i)} - y_{\text{gpu}}^{(i)}|$$

### 2. Sinyal-Gürültü Oranı (SNR - Signal-to-Noise Ratio)
Hassasiyet sapmasının (ör. FP32 referansına karşı FP16 gürültüsü) kalitesini desibel (dB) cinsinden ölçmek için:

$$\text{SNR}_{\text{dB}} = 10 \log_{10} \left( \frac{\sum_{i=1}^C (y_{\text{fp32}}^{(i)})^2}{\sum_{i=1}^C (y_{\text{fp32}}^{(i)} - y_{\text{fp16}}^{(i)})^2 + \epsilon} \right)$$

---

## 📂 Mimari Yapı & Dosya Düzeni

```
day-97-reproducible-inference/
├── LICENSE
├── gereksinimler.txt
├── README.md
├── ana_akis.py
├── src/
│   ├── __init__.py
│   ├── konfigurasyon.py
│   ├── model.py
│   ├── determinizm_yoneticisi.py
│   ├── capraz_donanim_motoru.py
│   └── gorsellestirici.py
├── testler/
│   ├── __init__.py
│   └── test_determinizm.py
└── ciktilar/
    └── deterministik_cikarim_paneli.png
```

---

## 🚀 Hızlı Başlangıç & Kurulum

```bash
# Gerekli bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Determinizm doğrulamasını ve teşhis panosu üretimini başlatın
python ana_akis.py
```

---

## 🧪 Testler ve Doğrulama

```bash
pytest testler/ -v
```

Çıktı Özeti:
```
testler/test_determinizm.py::test_determinizm_ortami_context_manager PASSED
testler/test_determinizm.py::test_bithash_hesaplayici_aynilik PASSED
testler/test_determinizm.py::test_ardil_cikarim_determinizmi PASSED
testler/test_determinizm.py::test_minivit_ileri_gecis_seki PASSED
testler/test_determinizm.py::test_capraz_donanim_cpu_gpu_parite PASSED
testler/test_determinizm.py::test_hassasiyet_kiyaslayici_fp16_bf16 PASSED
testler/test_determinizm.py::test_farkli_batch_boyutlari_determinizm PASSED
testler/test_determinizm.py::test_gorsellestirici_pano_uretme PASSED

======================= 8 passed in 12.20s =======================
```

---

## 📜 Lisans

```
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
