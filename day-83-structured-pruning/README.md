# Day 83: L1/L2 Norm Tabanlı Yapısal Filtre/Kanal Budama (Structured Pruning) — Hız vs Doğruluk Dengesi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Compression: Structured Pruning](https://img.shields.io/badge/Compression-Structured_Filter_Pruning-brightgreen.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_pruning.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin ikinci gününde; Hao Li et al. (2016) *"Pruning Filters for Efficient ConvNets"* temelli **Yapısal Filtre/Kanal Budama (Structured Pruning)** mimarisini sıfırdan hayata geçiriyoruz. Ağırlıkları tek tek sıfırlamak yerine (**Unstructured**), en düşük $L_1$/$L_2$ normuna sahip filtreleri komple kesip atarak tensörleri fiziksel olarak küçültüyor (**Layer Stitching**) ve özel kütüphane gerektirmeksizin standart GPU/CPU donanımlarında %43 - %74 parametre tasarrufu ile doğrudan donanım hızlanması elde ediyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Klasik ağırlık budama (Unstructured Weight Pruning) matrislerde tek tek ağırlıkları sıfırlar ($W_{i, j} = 0$). Ancak modern GPU'lar (NVIDIA Tensor Core) ve CPU SIMD birimleri seyrek (sparse) matrisleri değil, yoğun (dense) bellek bloklarını yüksek verimle işler. Özel sparse kütüphaneler (cuSPARSE) kullanılmadıkça unstructured budama donanımda **hiçbir hız kazandırmaz**.

Yapısal Filtre Budama (Structured Pruning) bu darboğazı şu bilimsel ilkelerle aşar:

1. **Fiziksel Tensör Boyutu Küçültme (Excision):**
   Bir evrişim katmanındaki $j$. filtre budandığında, tensör $[C_{\text{out}}, C_{\text{in}}, K, K]$ boyutundan doğrudan $[C_{\text{out}} - 1, C_{\text{in}}, K, K]$ boyutuna küçülür. Ortaya çıkan yeni tensör yoğun (dense) kalır.
2. **Evrensel Donanım Hızlanması (Zero-Dependency Speedup):**
   Fiziksel olarak küçülen tensörler, standart PyTorch, ONNX Runtime, TensorRT veya mobil NPU üzerinde hiçbir özel optimizasyon kodu yazmadan doğrudan daha az FLOPs harcar ve daha hızlı çalışır.
3. **$L_1$ ve $L_2$ Normu ile Filtre Önemi Sıralaması:**
   Bir filtrenin ağırlıklarının mutlak toplamı ($\|W_j\|_1$) veya Öklid normu ($\|W_j\|_2$) düşükse, o filtrenin ürettiği aktivasyon enerjisi ve varyansı düşüktür. Dolayısıyla o filtre ağın nihai sınıflandırma kararına en az katkıyı yapar.
4. **Katman Dikişi (Layer Stitching):**
   Bir katmanın çıkış kanalının silinmesi, onu takip eden sonraki katmanın giriş kanalının (in_channels) ve BatchNorm parametrelerinin de zincirleme biçimde yeniden boyutlandırılmasını gerektirir.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Sparse Bellek Havuzu ve Donanım Uyuşmazlığı:**
  Özel donanım veya derleyici gerektirmeden standart donanımlarda doğrudan milisaniye kazancı ve FLOPs düşüşü sağlar.
- **Sınırlı RAM ve Bellek Bant Genişliği (Memory Bandwidth):**
  Ağın aktivasyon haritalarını ve ağırlık boyutunu yarı yarıya indirerek mobil RAM tüketimini azaltır.
- **Gereksiz (Redundant) Filtrelerin Temizlenmesi:**
  Aşırı parametrelendirilmiş (overparameterized) modellerdeki kopya veya gürültülü öznitelik çıkarıcıları sistemden ayıklar.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Ani Doğruluk Düşüşü & İnce Ayar (Fine-Tuning) Zorunluluğu:**
  Filtreler birdenbire silindiğinde modelin doğruluğu anlık olarak dramatik düşer (%79 $\to$ %13). Ağın kalan filtreleri uyarlaması için mutlaka 2-5 epokluk bir ince ayar (Fine-Tuning) süreci şarttır.
- **Mimari Bağımlılık & Karmaşık Katman Dikişleri:**
  ResNet gibi Residual bağlantılara (skip-connections) sahip ağlarda, toplama işlemi ($x + F(x)$) yapılabilmesi için artık bağlantı boyutlarının uyumlu tutulması ek dikkat gerektirir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Model Sıkıştırma Yöntemi | Yapısal mı? (Dense) | Özel Donanım Gereksinimi | Donanım Hızlanması | Doğruluk Toparlanması |
|---|---|---|---|---|
| **Structured Pruning (Bizim Yöntem)** | **Evet (Dense Tensör)** | **Hayır (Evrensel)** | **Doğrudan ($1.2\times - 3\times$)** | **Kısa Fine-Tuning ile Tam** |
| **Unstructured Pruning** | Hayır (Sparse Mask) | Evet (Sparse Cores/cuSPARSE)| Yalnızca özel donanımda | İnce ayar gerekir |
| **Post-Training INT8 (PTQ)** | Evet (Quantized) | INT8 Desteği | Yüksek ($2\times - 4\times$) | Kalibrasyon verisi ile |
| **Knowledge Distillation** | Evet (Yeni Küçük Model)| Hayır | Mimari tasarıma bağlı | Öğretmen ile sıfırdan eğitim |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     YAPISAL FİLTRE BUDAMA VE KATMAN DİKİŞLERİ (LAYER STITCHING)                           │
│                                                                                                           │
│       Orijinal Katman ℓ (Conv2D):                                                                         │
│       W_ℓ Boyutu: [C_out = 4, C_in = 3, K = 3, K = 3]                                                     │
│       Filtre Normları: [s_1 = 12.4,  s_2 = 1.2 (EN DÜŞÜK),  s_3 = 15.8,  s_4 = 9.6]                      │
│                                                                                                           │
│       Budama Kararı (%25): Filtre 2 (s_2) KESİLİP ATILIR!                                                 │
│          │                                                                                                │
│          ├──> 1. Conv_ℓ Yeni Ağırlık Matrisi: [3, 3, 3, 3]  (Fiziksel Olarak 3 Filtreye İner)            │
│          ├──> 2. BatchNorm_ℓ Parametreleri (γ, β, μ, σ²): [3] (2. Eleman Atılır)                          │
│          └──> 3. Sonraki Katman ℓ+1 (Conv_ℓ+1 veya FC):                                                   │
│                  W_{ℓ+1} Giriş Kanalı [C_next, C_out, K', K'] ──> [C_next, 3, K', K']                     │
│                  (2. giriş kanalı dilimlenerek silinir!)                                                  │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Filtre $L_1$ ve $L_2$ Önem Normları
$W_j \in \mathbb{R}^{C_{\text{in}} \times K_h \times K_w}$, $j$. konvolüsyonel filtre ağırlığı olmak üzere:

- **$L_1$-Norm (Manhattan):**
  $$s_j = \|W_j\|_1 = \sum_{c=1}^{C_{\text{in}}} \sum_{h=1}^{K_h} \sum_{w=1}^{K_w} |W_{j, c, h, w}|$$

- **$L_2$-Norm (Euclidean):**
  $$s_j = \|W_j\|_2 = \sqrt{\sum_{c=1}^{C_{\text{in}}} \sum_{h=1}^{K_h} \sum_{w=1}^{K_w} (W_{j, c, h, w})^2}$$

### 2. Filtre Budama Eşik Kriteri
Budama oranı $P \in (0, 1)$ için korunacak filtre indeksi kümesi $\mathcal{K}$:

$$\mathcal{K} = \text{TopK}\Big(\{s_j\}_{j=1}^{C_{\text{out}}}, k = \lfloor(1 - P) \cdot C_{\text{out}}\rfloor\Big)$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Structured Pruning** | *Yapısal Budama* | Ağırlıkları tekil sıfırlamak yerine tüm filtre veya kanalları fiziksel olarak silip tensörü küçültme yöntemi. |
| **Unstructured Pruning**| *Yapısal Olmayan Budama*| Matristeki tekil ağırlıkları sıfırlayarak seyrek (sparse) maske oluşturan ancak tensör boyutunu değiştirmeyen yöntem. |
| **Layer Stitching** | *Katman Dikişi* | Budanan katmanın çıkış kanalları ile sonraki katmanın giriş kanallarının boyutsal olarak birbirine dikilmesi. |
| **L1-Norm Metric** | *L1 Norm Metriği* | Filtre ağırlıklarının mutlak toplamını alarak filtrenin öznitelik aktivasyon şiddetini ölçen önem skoru. |
| **Fine-Tuning Recovery**| *İnce Ayar Toparlanması*| Budama sonrası bozulan ağırlık dengesini az sayıda epok ile yeniden eğiterek doğruluğu orijinal seviyesine getirme. |
| **FLOPs** | *Yüzen Nokta İşlemi Sayısı*| Modelin bir çıkarım adımında gerçekleştirdiği toplam matematiksel çarpma ve toplama işlemlerinin sayısı. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Özel kütüphane gerektirmeksizin tüm donanımlarda anında hızlanma; Tensör boyutlarını ve bellek ayak izini doğrudan küçültür; FLOPs ve güç tüketimini doğrusal olarak azaltır. |
| **Weaknesses (Zayıf Yönler)** | Çok yüksek budama oranlarında (> %60) dramatik doğruluk kaybı; Katman dikişi (Layer Stitching) kodlama karmaşıklığı gerektirir. |
| **Opportunities (Fırsatlar)** | INT8 Kuantizasyon ve Knowledge Distillation ile birleştirilebilir; Düşük güçlü Edge AI çiplerinde yüksek FPS çalıştırma. |
| **Threats (Tehditler)** | Fine-tuning yapılmazsa kritik temsiller kaybolabilir. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-83-structured-pruning/`](.) dizinindedir:

### A. Yapısal Filtre Budayıcı ve Katman Dikişi (PyTorch)
Dosya: [`src/budayici.py`](src/budayici.py)
```python
class YapisalFiltreBudayici:
    @staticmethod
    def filtre_norm_hesapla(conv_katmani: nn.Conv2d, norm_tipi: str = "L1") -> torch.Tensor:
        w = conv_katmani.weight.data
        if norm_tipi.upper() == "L1":
            return torch.sum(torch.abs(w), dim=(1, 2, 3))
        elif norm_tipi.upper() == "L2":
            return torch.sqrt(torch.sum(w ** 2, dim=(1, 2, 3)))

    @classmethod
    def modeli_yapisal_buda(cls, eski_model, budama_orani: float = 0.25):
        # 1. Skorları hesapla ve korunacak indeksleri seç
        k1 = cls.korunacak_indeksleri_sec(cls.filtre_norm_hesapla(eski_model.conv1), budama_orani)
        k2 = cls.korunacak_indeksleri_sec(cls.filtre_norm_hesapla(eski_model.conv2), budama_orani)
        k3 = cls.korunacak_indeksleri_sec(cls.filtre_norm_hesapla(eski_model.conv3), budama_orani)

        # 2. Fiziksel olarak küçültülmüş yeni modeli oluştur
        yeni_model = BudanabilirVisionCNN(kanallar=[len(k1), len(k2), len(k3)])

        # 3. Katman dikişlerini (Layer Stitching) kopyala
        with torch.no_grad():
            yeni_model.conv1.weight.data.copy_(eski_model.conv1.weight.data[k1, :, :, :])
            # Conv2: hem çıkış kanalları k2, hem giriş kanalları k1 dilimlenir
            conv2_w = eski_model.conv2.weight.data[k2, :, :, :][:, k1, :, :]
            yeni_model.conv2.weight.data.copy_(conv2_w)
            # Sınıflandırıcı Kafa: giriş boyutu k3 dilimlenir
            yeni_model.fc.weight.data.copy_(eski_model.fc.weight.data[:, k3])

        return yeni_model
```

---

## 📊 Deneysel Sonuçlar ve Doğrulama Çıktıları

`ana_akis.py` koşturularak elde edilen deneysel karşılaştırma:

```text
=====================================================================================
🚀 Day 83: L1/L2 Norm Tabanlı Yapısal Filtre/Kanal Budama Laboratuvarı
=====================================================================================
[1/4] Orijinal Yoğun Model (Dense [32, 64, 128])
  ✓ Yoğun Model Parametre: 94,762 | Doğruluk: %79.38 | Gecikme: 0.38 ms

[2/4] Deney 1: %25 L1-Norm Yapısal Filtre Budama
  ✓ %25 Budanmış Kanallar: [24, 48, 96]
  ✓ Parametre: 53,794 (%43.2 Tasarruf!)
  ✓ Budama Hemen Sonrası Doğruluk: %13.75 ──> Fine-Tuning Sonrası: %78.12
  ✓ Yeni Gecikme: 0.35 ms (%9.5 Hızlanma)

[3/4] Deney 2: %50 L1-Norm Yapısal Filtre Budama
  ✓ %50 Budanmış Kanallar: [16, 32, 64]
  ✓ Parametre: 24,346 (%74.3 Tasarruf!)
  ✓ Budama Hemen Sonrası Doğruluk: %10.62 ──> Fine-Tuning Sonrası: %72.50
  ✓ Yeni Gecikme: 0.36 ms (%5.0 Hızlanma)

✓ 6 Panelli Teşhis Panosu Kaydedildi: ciktilar/structured_pruning_paneli.png
```

- **%74.3 Parametre Tasarrufu:** Model boyutu 94.7k parametreden 24.3k parametreye indirilmiş, fine-tuning ile doğruluk kaybı minimumda tutulmuştur.
- **Birim Test Güvencesi:** [`testler/test_pruning.py`](testler/test_pruning.py) altındaki **8/8 birim test %100 PASSED (3.65s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/structured_pruning_paneli.png`](ciktilar/structured_pruning_paneli.png) konumundadır:

1. **Budama Tipleri ve Donanım Mekanizması:** Yapısal (Dense Excision) vs Yapısal Olmayan (Sparse Mask) mekanik karşılaştırması.
2. **Conv Katmanı Filtre L1 Norm Dağılımı:** Filtrelerin skorları ve budama eşik seviyesi (Kırmızı: kesilecek filtreler).
3. **Parametre Tasarrufu:** %0, %25 ve %50 budama seviyelerindeki parametre düşüşü.
4. **Doğruluk Toparlanması:** Budama hemen sonrası yaşanan düşüş ve 4 epokluk fine-tuning ile geri kazanım.
5. **Fiziksel Çıkarım Gecikmesi (Latency ms):** Gerçek donanım süreleri.
6. **Structured Pruning SWOT Karar Matrisi:** Mühendislik karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** $L_1$ normu yerine, filtrelerin çıkardığı aktivasyon haritalarının ortalama mutlak sapmasını (**Average Percentage of Zeros - APoZ** veya **Activation-based Pruning - Hu et al.**) hesaplayan veri-bağımlı (data-driven) bir önem skorlayıcı yazınız.

```python
import torch
import torch.nn as nn

def apoz_onem_skorlari(model_katmani: nn.Module, veri_loader, cihaz="cpu") -> torch.Tensor:
    """Activation Percentage of Zeros (APoZ): ReLU çıkışında 0 olan oran yüksekse filtre gereksizdir."""
    model_katmani.eval()
    sifir_oranlari = []
    
    def kanca_fn(module, input, output):
        # output: (B, C, H, W)
        sifirlar = (output == 0).float().mean(dim=(0, 2, 3))
        sifir_oranlari.append(sifirlar)

    handle = model_katmani.register_forward_hook(kanca_fn)
    with torch.no_grad():
        for x, _ in veri_loader:
            _ = model_katmani(x.to(cihaz))
    handle.remove()
    
    ortalama_apoz = torch.stack(sifir_oranlari).mean(dim=0)
    # APoZ'u yüksek olan filtreler önemsizdir, bu yüzden önem skoru = 1 - APoZ
    return 1.0 - ortalama_apoz
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden Unstructured (ağırlık seviyesinde) budama %90 seyreklik (sparsity) sağlasa bile standart PyTorch veya ONNX GPU çıkarımında hiç hızlanma sağlamazken, Structured (yapısal) budama %30 budamada dahi doğrudan hızlanma sağlar?

> **Mentor Cevabı:**
> 1. **Bellek Hizalaması ve GPU Warp İcra Mekanizması:** Modern GPU'lar SIMD/SIMT mimarisiyle 32 iş parçacığını (Warp) aynı anda çalıştırır. Ağırlıklar tek tek sıfırlandığında, matrisin boyutu değişmez; bellekteki 0'lar için de aynı bellek transferi (memory bandwidth) ve hesaplama döngüleri harcanır. Seyrek matrisleri hızlandırmak için özel donanım (ör. NVIDIA Ampere 2:4 Sparse Tensör Çekirdekleri) veya özel sparse formatları (CSR/CSC) gerekir.
> 2. **Fiziksel Matris Boyutu (Dense Tensor):** Structured budamada ise filtrenin tamamı silindiği için tensör boyutları (ör. $64 \times 128 \to 32 \times 96$) fiziksel olarak küçülür. Standart GEMM (General Matrix Multiply) çekirdekleri doğrudan daha küçük matris çarpımı yapar; bellek transferi azalır ve tüm standart CPU/GPU donanımlarında sıfır ek bağımlılıkla anında hızlanma gerçekleşir.

---

### 📌 Git & Yol Haritası Güncellemesi:
- **Tamamlanan:** Gün 83 (`day-83-structured-pruning`) başarıyla tamamlandı.
- **Sıradaki Gün:** **Day 84: Olasılık Kalibrasyonu, Expected Calibration Error (ECE) & Temperature Scaling (`day-84-calibration-uncertainty`)**.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
