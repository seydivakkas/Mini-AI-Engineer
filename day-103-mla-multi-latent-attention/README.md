# Day 103: Multi-Head Latent Attention (MLA - DeepSeek V2/V3) Sıkıştırılmış KV Projeksiyonu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; DeepSeek-V2, DeepSeek-V3 ve DeepSeek-R1 modellerinin bellek devrimini gerçekleştiren **Multi-Head Latent Attention (MLA)**, **Düşük Dereceli Ortak KV Sıkıştırması (Low-Rank Joint KV Compression)**, **Ayrık RoPE (Decoupled RoPE)** ve **Matris Soğurma (Matrix Absorption)** mimarilerini sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Zip Dosyası Analojisi"

Dün öğrendiğimiz GQA, 32 araştırmacıya 8 ansiklopedi vererek masayı rahatlatmıştı. Ancak DeepSeek-V3 gibi devasa bir modelde tam **128 dikkat başlığı** bulunur! 128 başlıkta GQA kullansanız dahi 128k bağlamda yüzlerce gigabayt VRAM tükenir.

DeepSeek mühendisleri şu dâhiyane soruyu sordu:  
*"Her kelimenin 128 kafa için ayrı ayrı dev Key ve Value vektörlerini bellekte tutmak zorunda mıyız? Neden bunları küçük bir ZIP dosyası gibi sıkıştırıp saklamıyoruz?"*

* 📦 **Ortak Latent Sıkıştırması ($c_t^{KV}$):** Gelen token'ın tüm Key ve Value anlamını tek bir küçük latent vektöre ($d_c = 512$) sıkıştırırız. GPU belleğinde 32,768 sayı yerine sadece **512 sayı** saklarız!
* 🗝️ **Ayrık RoPE ($k_t^R$):** Konum bilgisi (Rotary Position Embedding) sıkıştırılamayacağı için sadece 64 sayılık minik bir konumsal anahtarı ($d_R = 64$) ayrı saklarız.
* ⚡ **Matris Soğurma (Matrix Absorption):** Çıkarım (Inference) anında zip dosyasını açmaya (decompress) bile gerek kalmaz! Açma matrisini ($W_{UK}$) sorgu matrisinin içine matematiksel olarak yutarız ($Q' = Q W_{UK}^T$) ve doğrudan sıkıştırılmış latent ile iç çarpım yaparız. Sıfır açma gecikmesi!

Sonuç: **%98.2'ye varan (56.8 kat) KV Cache bellek tasarrufu!**

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Matematiksel Modelleme
Girdi $h_t \in \mathbb{R}^d$ için:
1. **Düşük Dereceli Ortak KV Sıkıştırması:**
   $$c_t^{KV} = h_t W_{DKV} \in \mathbb{R}^{d_c} \quad (W_{DKV} \in \mathbb{R}^{d \times d_c})$$
   Açılım (Up-projection):
   $$K_t^C = c_t^{KV} W_{UK} \in \mathbb{R}^{H \times d_h}, \quad V_t^C = c_t^{KV} W_{UV} \in \mathbb{R}^{H \times d_h}$$

2. **Düşük Dereceli Query Sıkıştırması:**
   $$c_t^Q = h_t W_{DQ} \in \mathbb{R}^{d_q}, \quad Q_t^C = c_t^Q W_{UQ} \in \mathbb{R}^{H \times d_h}$$

3. **Ayrık RoPE (Decoupled Position Embedding):**
   $$K_t^R = \text{RoPE}(h_t W_{KR}) \in \mathbb{R}^{d_R}, \quad Q_t^R = \text{RoPE}(c_t^Q W_{QR}) \in \mathbb{R}^{H \times d_R}$$

4. **Birleşik Dikkat Skoru:**
   $$\text{Skor}_{i, j} = \frac{(Q_{i}^C)^T K_{j}^C + (Q_{i}^R)^T K_{j}^R}{\sqrt{d_h + d_R}}$$

### 2. Bellek (KV Cache) ve Hesaplama Karmaşıklığı Analizi
Token başına önbelleğe alınan eleman sayısı:
$$\text{MHA: } 2 \times H \times d_h = 2 \times 128 \times 128 = \mathbf{32,768 \text{ Eleman}}$$
$$\text{GQA-8: } 2 \times 8 \times 128 = \mathbf{2,048 \text{ Eleman}}$$
$$\text{DeepSeek MLA: } d_c + d_R = 512 + 64 = \mathbf{576 \text{ Eleman}}$$

| Mimari Türü ($L=32, B=16$) | 4096 Token (MB) | 16384 Token (MB) | 32768 Token (GB) | Bellek Tasarrufu |
|:---|:---|:---|:---|:---|
| **MHA (16 Kafa)** | 4096.0 MB | 16384.0 MB | **32.00 GB** | Referans (%0) |
| **GQA (4 Kafa)** | 1024.0 MB | 4096.0 MB | **8.00 GB** | %75.0 Tasarruf |
| **DeepSeek MLA** | **640.0 MB** | **2560.0 MB** | **5.00 GB** | **%84.4 Tasarruf** |

### 3. Donanım & GPU Bellek Bant Genişliği (Memory Bandwidth) Etkisi
Standart LLM mimarilerinde 128k bağlamda KV cache boyutu onlarca GPU'ya yayılmak zorundayken, DeepSeek MLA sayesinde devasa modeller tek bir sunucu düğümünün (8x H800 / H100) HBM belleğinde tam bağlam uzunluğuyla çalışabilir. Matris soğurma sayesinde SRAM/HBM veri transfer yükü minimuma iner.

### 4. Endüstriyel Entegrasyon (DeepSeek-V2, DeepSeek-V3, DeepSeek-R1)
DeepSeek'in açık kaynak dünyasında OpenAI GPT-4o ve Claude 3.5 Sonnet seviyesine çok daha düşük maliyetle ulaşmasının ana temeli MLA mimarisidir:
- **DeepSeek-V2 (236B MoE):** $H=128, d_h=128, d_c=512, d_R=64$.
- **DeepSeek-V3 (671B MoE):** $H=128, d_h=128, d_c=512, d_R=64$.
- **DeepSeek-R1 (Reasoning):** V3 mimarisi üzerinde pekiştirmeli öğrenme (RL).

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Multi-Head Latent Attention (MLA)** | Key ve Value tensörlerini düşük dereceli bir gizli uzayda sıkıştıran DeepSeek dikkat mimarisi. |
| **KV Latent ($c^{KV}$)** | Girdi token'ının tüm Key ve Value bilgilerini özetleyen düşük dereceli sıkıştırılmış vektör ($d_c$). |
| **Decoupled RoPE** | Konumsal bilgiyi içerik latent'ından ayırarak bağımsız bir vektör ($d_R$) olarak işleyen mekanizma. |
| **Matrix Absorption (Matris Soğurma)** | Çıkarım anında Key açılım matrisini Query projeksiyonuna çarparak açma maliyetini sıfırlama tekniği. |
| **Down-Projection ($W_{DKV}$)** | Yüksek boyutlu token temsilini düşük boyutlu latent uzaya indirgeyen matris. |
| **Up-Projection ($W_{UK}, W_{UV}$)** | Düşük boyutlu latent vektörü tam dikkat başlığı boyutuna genişleten matris. |
| **Compression Ratio (Sıkıştırma Oranı)** | Orijinal KV önbellek boyutu ile sıkıştırılmış MLA önbellek boyutu arasındaki oran ($56.8\times$). |
| **Rotary Position Embedding (RoPE)** | Vektörleri karmaşık düzlemde açıyla döndürerek göreli konum bilgisi kazandıran kodlama. |
| **Memory-Bound Decoding** | Otoregresif üretimde donanım hızının GPU bellek bant genişliği ile sınırlandığı durum. |
| **DeepSeek-R1** | MLA ve MoE mimarisi üzerine inşa edilmiş açık kaynaklı akıl yürütme (reasoning) modeli. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Token başına %98'e varan bellek    │ • Eğitim sırasında ekstra projeksiyon│
 │   tasarrufu (56.8x sıkıştırma).      │   matrisleri (FLOPs artışı).         │
 │ • 128 kafa ile tam MHA kapasitesi.  │ • FlashAttention-2'ye doğrudan       │
 │ • Matris soğurma ile sıfır açma yükü.│   beslenemez, özel kernel gerektirir.│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • 128k+ bağlamlı modelleri tek       │ • Özel donanım optimizasyonları      │
 │   sunucuda çalıştırma imkânı.        │   (Triton/CUDA) yazma karmaşıklığı.  │
 │ • DeepSeek-V3 ve R1 ile endüstri     │ • Standart Hugging Face mimarilerine │
 │   standardı haline gelmesi.          │   göre daha karmaşık kod yapısı.     │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/deepseek_mla_teshis_paneli.png` dosyası üretilir:
1. **Bağlam Uzunluğuna Göre KV Cache (MB - Log Scale)**
2. **Token Başına Önbellek Eleman Sayısı (1024 vs 256 vs 160 Eleman)**
3. **DeepSeek MLA Matematiksel Formül Kartı**
4. **32,768 Token Bağlamda Toplam VRAM Tüketimi (32 GB vs 8 GB vs 5 GB)**
5. **DeepSeek MLA VRAM Tasarruf Oranları (%84.4 vs MHA)**
6. **Stajyer Notu & DeepSeek MLA Mimari Karar Sertifikası**

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
