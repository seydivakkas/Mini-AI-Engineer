# 🚗 Tesla FSD Otonom Sürüş | Gün 51: Model Sıkıştırma, Kuantizasyon ve HW3/HW4 FSD NPU Derleme (TensorRT / INT8)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Quantization](https://img.shields.io/badge/NPU-Symmetric%20INT8%20Quantization-red.svg?style=flat-square)](https://www.tesla.com/)
[![Memory](https://img.shields.io/badge/Optimization-75%25%20SRAM%20Footprint%20Reduction-blue.svg?style=flat-square)](https://www.sae.org/)
[![SQNR](https://img.shields.io/badge/Signal-43.2%20dB%20SQNR-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"51. günümüze hoş geldin stajyer!  
> Süper bilgisayarda PyTorch ile FP32 (32-bit Kayan Nokta) formatında eğitilen devasa HydraNet ve Occupancy modellerini doğrudan araç üzerindeki FSD Bilgisayarına (Hardware 3 / Hardware 4) yükleyemezsiniz.  
> Çünkü aracın NPU'su (Neural Processing Unit) sınırlı güç bütçesine (36W-72W) ve SRAM önbelleğine (32MB-64MB) sahiptir. FP32 ağırlıklar belleği anında tüketir ve çıkarım hızını düşürür.  
> Tesla bu donanım darboğazını **Symmetric INT8 Kuantizasyon ve Katman Birleştirme (Layer Fusion)** ile aşar:  
> 1. **FP32 -> INT8 Dönüşümü:** 32-bitlik ağırlıklar 8-bit tam sayılara ($[-128, 127]$) indirgenir.  
> 2. **%75 SRAM Tasarrufu:** Bellek ihtiyacı 4 Byte'tan 1 Byte'a düşer, DRAM bant genişliği tüketimi 4 kat azalır.  
> 3. **43.2 dB SQNR Kalitesi:** Simetrik ölçekleme ($S = \frac{\max(|W|)}{127}$) ile bilgi kaybı yok denecek kadar azdır ($< %0.05$ doğruluk farkı).  
> 4. **Katman Birleştirme (Layer Fusion):** Conv + BatchNorm + ReLU katmanları NPU üzerinde tek bir donanımsal çekirdeğe (Fused Kernel) kaynaştırılarak ara bellek okuma/yazma maliyeti sıfırlanır.  
> Bugün Tesla FSD NPU'sunun silikon seviyesinde hızlandırıcı motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Simetrik INT8 Kuantizasyon ve Ölçekleme

$$S = \frac{\max(|W|)}{127}, \quad q = \text{clip}\left( \left\lfloor \frac{W}{S} \right\rceil, -128, 127 \right)$$

$$\hat{W} = q \cdot S$$

### 2. Sinyal-Kuantizasyon-Gürültü Oranı (SQNR)

$$\text{SQNR} = 10 \cdot \log_{10}\left( \frac{\sum_{i=1}^N W_i^2}{\sum_{i=1}^N (W_i - \hat{W}_i)^2} \right) \quad [\text{dB}]$$

### 3. Katman Birleştirme (Layer Fusion Matematiksel Eşdeğeri)

$$y = \text{ReLU}\left( \gamma \left( \frac{W * x - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta \right) \implies y = \text{ReLU}\left( W_{\text{fused}} * x + b_{\text{fused}} \right)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
HydraNet ve Transformer modellerini Tesla HW3/HW4 NPU üzerinde 36 Watt enerji bütçesiyle 36 FPS gerçek zamanlı hızda koşturmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **SRAM Bellek Taşması:** 32 MB SRAM içine sığmayan devasa modeller %75 sıkıştırılarak yerel NPU önbelleğine sığdırıldı.
- **DRAM Bellek Darboğazı (Memory Bottleneck):** Bellekten veri çekme süresi 4 kat kısaltılarak NPU işlem çekirdeklerinin boşta beklemesi (starvation) önlendi.
- **144 TOPS NPU Verimi:** INT8 matris çarpım üniteleri tam kapasitede paralel çalıştırıldı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Aykırı Değerler (Outliers):** Transformer dikkat matrislerinde çok büyük aykırı değerler varsa düzgün ölçekleme zorlaşabilir (Kanal Bazlı / Per-Channel Quantization gerekir).
- **Gradyan Eğitimi:** INT8 çıkarım içindir; geriye yayılım (Backpropagation) sırasında kuantizasyon farkındalıklı eğitim (QAT - Quantization Aware Training) gereklidir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **FP16 / BF16 (16-bit Float):** INT8'e göre 2 kat daha fazla bellek ve enerji harcar.
- **INT4 Kuantizasyon:** Doğruluk kaybı kritik algılama başlıklarında hissedilebilir seviyeye çıkar.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **INT8 Quantization** | 32-bit kayan nokta tensörleri 8-bit tam sayı aralığına sıkıştıran donanım optimizasyon tekniği. |
| **Scale Factor ($S$)** | Gerçek kayan nokta değerini tam sayıya dönüştürmek için kullanılan bölme katsayısı. |
| **SQNR** | Kuantizasyon sonucu oluşan gürültünün sinyal gücüne oranını belirten desibel (dB) kalitesi. |
| **Layer Fusion** | Ardışık evrişim, normalizasyon ve aktivasyon katmanlarını tek bir NPU çekirdeğinde birleştirme. |
| **FSD HW3/HW4 NPU** | Tesla'nın araç içinde FSD modellerini koşturmak için tasarladığı özel silikon yapay zeka hızlandırıcısı. |
| **Symmetric Quantization**| Sıfır noktasının (Zero Point) tam olarak 0'a denk geldiği ve ek ofset gerektirmeyen kuantizasyon modu. |
| **TensorRT** | NVIDIA/Tesla donanımlarında modelleri graf seviyesinde optimize eden yüksek performanslı çıkarım motoru. |
| **QAT (Quantization Aware Training)**| Model eğitilirken sahte kuantizasyon gürültüsü ekleyerek INT8 kaybını önceden telafi etme yöntemi. |
| **Per-Channel Quantization**| Tensörün her evrişim filtresi için bağımsız bir ölçek katsayısı ($S_c$) hesaplayarak hassasiyeti artırma. |
| **SRAM vs DRAM** | NPU çekirdeğine entegre ultra hızlı önbellek (SRAM) ve ana araç sistem belleği (DRAM). |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %75 SRAM bellek tasarrufu ve 4x bant genişliği      | • Aşırı aykırı değerlerde kanal bazlı ölçekleme şartı |
| • 43.2 dB yüksek sinyal-gürültü kalitesi (SQNR)       | • QAT olmaksızın doğrudan Post-Training Quantization  |
| • Katman birleştirme ile sıfır DRAM ara bellek yükü  |   bazı başlıklarda %0.5 kayıp yapabilir               |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • HW4 NPU'da FP8 ve karma hassasiyet (Mixed Precision)| • Yanlış kalibrasyon veri seti seçiminde dinamik      |
|   desteğiyle 300+ TOPS çıkarım kapasitesi             |   aralık taşması (Clipping Saturation)                |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ FSD NPU INT8 Derleme Akışı

```
[ PyTorch FP32 Eğitilmiş Model (400 MB) ]
                   |
                   v
[ Kalibrasyon ve Dinamik Aralık Tespiti (KL-Divergence) ]
                   |
                   v
[ Simetrik INT8 Kuantizasyon: Scale S = max(|W|)/127 ]
                   |
                   v
[ Katman Birleştirme (Layer Fusion: Conv+BN+ReLU -> Tek Çekirdek) ]
                   |
                   v
[ Tesla FSD NPU İkili Derlemesi (100 MB, 144 TOPS INT8 Hızlandırma) ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Kuantizasyon simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
