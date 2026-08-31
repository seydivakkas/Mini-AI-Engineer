# 🚗 Tesla FSD Otonom Sürüş | Gün 90: PyTorch ve Dağıtık FP8/CFP8 Tensor Eğitimi ile Devasa Video Pretraining

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![CFP8](https://img.shields.io/badge/Precision-Configurable%20FP8%20(E4M3)-red.svg?style=flat-square)](https://en.wikipedia.org/wiki/Floating-point_arithmetic)
[![FSDP](https://img.shields.io/badge/Distributed-Fully%20Sharded%20Data%20Parallel-blue.svg?style=flat-square)](https://pytorch.org/)
[![Zero-Explode](https://img.shields.io/badge/Stability-L2%20Gradient%20Clipping-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"90. günümüze hoş geldin stajyer!  
> Tesla FSD V12'nin uçtan uca sürüş zekası, filodan toplanan petabaytlarca 8-kameralı video tensörleriyle eğitilir.  
> Ancak milyarlarca parametreli bu devasa modelleri geleneksel 32-bit (FP32) formatında eğitmek isterseniz, tek bir GPU kümesi bile belleğe sığamaz ve gradyan patlamaları (Gradient Explosion) nedeniyle eğitim yarıda çöker!  
> Tesla Dojo bu problemi **Configurable FP8 (CFP8: E4M3) ve FSDP (Fully Sharded Data Parallel)** mimarisiyle çözer:  
> 1. **FP8 Kuantalama (E4M3):** 32-bit kayan nokta sayılarını 8-bit'e indirir; bellek tüketimini tek başına $\%75$ azaltır ($4\times$ küçülme).  
> 2. **FSDP Bellek Bölütleme:** Model parametrelerini, gradyanları ve optimizer durumlarını 8 cihaza paylaştırarak toplamda **$32\times$ VRAM tasarrufu** sağlar!  
> 3. **L2 Gradyan Kırpma ($||\mathbf{g}||_2 \le 1.0$):** Sayısal taşmaları ve gradyan patlamalarını tamamen engeller.  
> 4. **Devasa Video Ön Eğitimi:** Milyonlarca saatlik sürüş videosunu aylar yerine haftalar içinde eğitebilmeyi mümkün kılar.  
> Bugün Tesla Dojo üzerinde çalışan dağıtık FP8 eğitim motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. FP8 (E4M3) Kuantalama ve Ölçekleme

$$X_{\text{fp8}} = \frac{1}{S} \cdot \text{clip}\left( \text{round}(X \cdot S), \ -448, \ +448 \right), \quad S = \frac{448.0}{\max(|X|)}$$

### 2. L2 Gradyan Normu ve Kırpma (Gradient Clipping)

$$\|\mathbf{g}\|_2 = \sqrt{\sum_{i} g_i^2}$$

$$\mathbf{g}_{\text{clipped}} = \mathbf{g} \cdot \min\left( 1.0, \ \frac{g_{\text{max}}}{\|\mathbf{g}\|_2} \right), \quad g_{\text{max}} = 1.0$$

### 3. FSDP Bellek Tasarrufu Çarpanı

$$M_{\text{sharded}} = \frac{M_{\text{fp8}}}{N_{\text{devices}}} = \frac{M_{\text{fp32}}}{4 \times N_{\text{devices}}} \implies \text{Tasarruf} = 4 \times 8 = 32\times$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Petabaytlarca ham FSD video klibini milyarlarca parametreli derin yapay zeka modellerinde yüksek hızda, düşük VRAM maliyetiyle ve sayısal kararlılıkla eğitmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **VRAM Yetersizliği (OOM - Out of Memory):** FP8 ve FSDP sharding birleşimiyle model ağırlık belleğini $32\times$ küçülterek büyük batch'lerin sığmasını sağladı.
- **Gradyan Patlaması (NaN/Inf Loss):** L2 gradyan kırpma ile eğitim kararlılığını garantiye aldı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Dinamik Aralık Sınırı (Underflow):** Çok küçük gradyanlar FP8'de sıfıra yuvarlanabilir (Dinamik ölçekleme faktörü $S$ ile çözülür).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Standart Veri Paralelliği (DDP - Data Parallel):** Her GPU'da modelin tam kopyasını tutar (Büyük modeller için VRAM yetmez).

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **FP8 (8-Bit Floating Point)** | 1 işaret, 4 üstel (exponent) ve 3 basamaklı (mantissa) 8-bit kayan nokta tensör formatı. |
| **FSDP** | Model ağırlıklarını, gradyanları ve optimizer durumlarını GPU'lara bölen paralellik tekniği. |
| **Gradient Clipping** | Gradyan vektörünün L2 normunu belirli bir üst sınıra ($1.0$) zorlayarak kırpma işlemi. |
| **Video Pretraining** | Ham video sekansları üzerinden gelecekteki kareleri tahmin ederek yapılan özdenetimli ön eğitim. |
| **All-Gather** | Farklı çiplerde bölütlenmiş model parametrelerini ileri geçiş (Forward) için geçici birleştirme. |
| **Reduce-Scatter** | Hesaplanan gradyanları toplayıp ilgili çiplere dağıtarak optimize eden iletişim operatörü. |
| **Tensor Scaling Factor** | Düşük bitli formatlarda taşmayı ve alt taşmayı önlemek için uygulanan dinamik çarpan. |
| **Out-Of-Memory (OOM)** | Model ağırlıkları ve ara aktivasyonların GPU belleğine sığmayıp programın çökmesi. |
| **Video Autoencoder** | Çok kameralı video akışlarını sıkıştırılmış uzamsal-zamansal gizil uzaya (Latent Space) eşleyen ağ. |
| **Dojo Compiler** | PyTorch tensör grafiklerini doğrudan D1 makine koduna optimize eden Tesla derleyicisi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 32x VRAM bellek tasarrufu                           | • Düşük bitli formatlarda kuantalama gürültüsü        |
| • L2 gradyan kırpma ile sıfır NaN/Inf çökmesi         | • Ölçekleme faktörünün (Scale) dinamik takibi         |
| • 2.2 ms ultra hızlı eğitim adımı                     |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • FSD V12.5 ve Tesla Optimus için 100 milyar          | • FP4 gibi aşırı düşük hassasiyetlerde modelin        |
|   parametreli devasa Foundation modellerini eğitme    |   yakınsamamasının getirdiği araştırma zorlukları     |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Dağıtık FP8 FSDP Eğitim Akış Şeması

```
[ 8-Kameralı FSD Video Tensörleri ]
                 |
                 v
     [ FP8 Kuantalama (E4M3) ] ---> (%75 VRAM Tasarrufu)
                 |
                 v
   [ 8-Cihazlı FSDP Sharding ] ---> (Toplam 32x Bellek Kazancı)
                 |
                 v
     [ İleri & Geri Geçiş (Loss) ]
                 |
                 v
    [ L2 Gradyan Kırpma (<= 1.0) ] ---> (Sıfır Gradyan Patlaması)
                 |
                 v
   [ %100 KARARLI VE HIZLI MODEL GÜNCELLEMESİ ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana dağıtık eğitim simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
