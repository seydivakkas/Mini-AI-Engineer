# Day 107: QLoRA (NF4 - NormalFloat4 Kuantizasyon, Double Quantization) & Unsloth Autograd

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; 65B-70B parametreli devasa temel modellerin tek bir tüketici GPU'sunda (24GB VRAM) fine-tuning yapılmasını mümkün kılan **QLoRA (Dettmers et al., 2023)**, **4-bit NormalFloat4 (NF4)**, **Double Quantization (DQ)**, **Paged Optimizers** ve **Unsloth Tarzı Füzyonlu Hızlı Autograd** mimarilerini sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Şeffaf Asetat ve Fotoğraf Sıkıştırma Analojisi"

Büyük bir yapay zeka modelini (örn. 70 Milyar parametreli LLaMA-3-70B) eğitmek için normalde 16 adet 80GB A100 GPU (1.120 GB VRAM) ve yüz binlerce dolar gerekir.

QLoRA bunu şu dahiyane 3 adımla çözer:

1. 📷 **NF4 Kuantizasyon (Fotoğraf Sıkıştırma):** Modelin ağırlıkları rastgele sayılar değil, çan eğrisi (normal dağılım) şeklindedir. NF4, 16 farklı renk tonu belirler. Her ağırlığı 16-bit yerine 4-bitlik renk numarasıyla kaydeder. Model boyutu 4 kat küçülür!
2. 🗜️ **Double Quantization (Ölçekleri de Sıkıştırma):** Fotoğrafı bloklara böldüğümüzde her bloğun ölçek katsayısını tutmak ek yer kaplar. QLoRA bu ölçekleri de 8-bit ile ikinci kez sıkıştırarak parametre başına ek yükü 0.5 bitten 0.127 bite düşürür!
3. 📄 **LoRA (Şeffaf Asetat):** 70 milyar parametrelik dev kitabı dondurup üzerine incecik şeffaf bir asetat kağıdı koyuyoruz ($A$ ve $B$ matrisleri). Yalnızca asetat kağıdına tükenmez kalemle küçük notlar alıyoruz. Kitap dondurulduğu için sıfır gradyan belleği harcar!

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Matematiksel Modelleme
Standart normal dağılım $\mathcal{N}(0, 1)$ kümülatif dağılım fonksiyonu $Q_X$ üzerinden $2^k = 16$ kuantile seviyesi belirlenir:
$$q_i = \frac{1}{2}\left(Q_X\left(\frac{i}{2^k}\right) + Q_X\left(\frac{i+1}{2^k}\right)\right)$$

Model ileri geçişi ($Y$) ve füzyonlu geri yayılımı:
$$Y = X \cdot \hat{W}^T + \frac{\alpha}{r} (X \cdot A^T) \cdot B^T$$
$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \left(\frac{\partial \mathcal{L}}{\partial Y}\right)^T (X A^T), \quad \frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} \left(\left(\frac{\partial \mathcal{L}}{\partial Y}\right) B\right)^T X$$

### 2. Bellek, Hesaplama Karmaşıklığı ve VRAM Ölçeklenmesi Analizi
- **Double Quantization (DQ):** Birincil $c_1$ ölçekleri ($B_1=64$) $c_1 \sim \text{FP32} \implies 32/64 = 0.5 \text{ bpp}$. İkincil blokta ($B_2=256$) 8-bit kuantizasyonla:
$$\text{Ek Yük} = \frac{8}{64} + \frac{32}{64 \times 256} = 0.125 + 0.00195 = \mathbf{0.127 \text{ bpp}}$$

| Model Ölçeği | Full Fine-Tuning (16B/param) | FP16 LoRA (2B/param) | QLoRA (NF4+DQ ~0.55B/param) | Tasarruf Oranı |
|:---|:---|:---|:---|:---|
| **7B Modeli** | 112.0 GB | 16.3 GB | **5.1 GB** | **%95.5 Tasarruf** |
| **13B Modeli** | 208.0 GB | 28.6 GB | **8.3 GB** | **%96.0 Tasarruf** |
| **33B Modeli** | 528.0 GB | 69.6 GB | **19.4 GB** | **%96.3 Tasarruf** |
| **70B Modeli** | 1,120.0 GB | 145.5 GB | **39.7 GB** | **%96.5 Tasarruf!** |

### 3. Donanım, Paged Optimizers & CUDA Unified Memory Etkisi
Eğitim sırasında GPU VRAM ani bir bellek tepe noktasına (OOM spike) ulaştığında, **Paged Optimizers** mekanizması CUDA Unified Memory kullanarak AdamW durumlarını dinamik olarak CPU RAM'e takas eder (page-out) ve GPU çökmesini engeller.

### 4. Endüstriyel Entegrasyon (Unsloth, BitsAndBytes, PEFT, TRL)
- **BitsAndBytes (`bnb`):** 4-bit NF4 ve 8-bit matmul işlemlerini doğrudan CUDA C++ çekirdeklerinde çalıştırır.
- **Unsloth:** Ara bellek tahsislerini (activations) C++/Triton seviyesinde birleştirerek (fused backward) standart Hugging Face PEFT'e göre **2x - 5x daha hızlı** eğitim sağlar.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **QLoRA** | Dondurulmuş 4-bit kuantize ana model üzerine düşük rütbeli adaptörler ekleyerek yapılan ince ayar yöntemi. |
| **NF4 (NormalFloat4)** | Normal dağılımlı tensörler için teorik bilgi entropisini maksimize eden 16-noktalı kuantizasyon türü. |
| **Double Quantization (DQ)** | Kuantizasyon sabitlerinin (scale constants) kendisini de 8-bit kuantize ederek bellek ek yükünü azaltma. |
| **Paged Optimizers** | GPU belleği dolduğunda optimizer durumlarını CPU RAM'e sayfalayan (paging) bellek yönetim mekanizması. |
| **LoRA (Low-Rank Adaptation)** | Ağırlık güncelleme matrisini iki düşük rütbeli matrisin çarpımı ($W + B \cdot A$) olarak modelleme. |
| **Dequantization** | 4-bit NF4 indekslerini matris çarpımı için geçici olarak FP16/BF16 sayılarına dönüştürme işlemi. |
| **Weight Tying** | Giriş embedding katmanı ile çıkış LM başlığı ağırlıklarının aynı tensörü paylaşması. |
| **Quantization Scale ($c_1$)** | Bir blok içindeki mutlak maksimum değeri temsil eden birincil normalizasyon katsayısı. |
| **Bits Per Parameter (bpp)** | Modeldeki her bir parametre başına düşen ortalama bit bellek maliyeti. |
| **Fused Autograd** | İleri ve geri geçiş matris işlemlerini tek bir CUDA çekirdeğinde birleştirerek bellek trafiğini azaltma. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • 70B modeli tek bir 48GB GPU'da     │ • Sürekli on-the-fly dekuantizasyon  │
 │   eğitebilme imkânı (%96.5 VRAM tas.)│   nedeniyle saf FP16'ya göre hafif   │
 │ • Cosine benzerliği > %99.5.         │   çıkarım gecikmesi.                 │
 │ • 0.127 bpp Double Quantization.     │ • Özel CUDA/Triton kerneli gereksin. │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Bireysel geliştiricilerin kendi    │ • INT4/FP4 donanım desteği olmayan   │
 │   evlerindeki RTX 3090/4090 GPU'larla│   eski GPU'larda hesaplama yükü.     │
 │   dev modelleri eğitebilmesi devrimi.│                                      │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/qlora_nf4_unsloth_paneli.png` dosyası üretilir:
1. **Model Ölçeklerine Göre VRAM İhtiyacı (GB: Full FT vs LoRA vs QLoRA)**
2. **16-Noktalı NF4 Normal Dağılım Kuantile Seviyeleri ve Gauss PDF Grafiği**
3. **Double Quantization (DQ) Bellek Sıkıştırması (0.500 bpp -> 0.127 bpp)**
4. **70B Model Fine-Tuning VRAM Kıyaslaması (1,120 GB vs 39.7 GB)**
5. **QLoRA & Unsloth Autograd Matematiksel Formül Kartı**
6. **Stajyer Notu & QLoRA Endüstri Karar Sertifikası**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Ana kıyaslama ve görselleştirme akışını koşturun
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
