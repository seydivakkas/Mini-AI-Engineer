# Day 104: Rotary Position Embeddings (RoPE), NTK-Aware Scaling & YaRN ile 128k+ Bağlam Uzatma Matematiği

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; 4k bağlamla eğitilmiş büyük dil modellerinin (LLM) bağlam penceresini (Context Window) minimum fine-tuning ile 32k, 64k ve 128k+'ya uzatan **RoPE**, **Linear Position Interpolation (PI)**, **NTK-Aware Scaling** ve **YaRN (Yet another RoPE extensioN)** algoritmalarını ele alır.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Saat Akrebi ve Yelkovanı Analojisi"

Bir dil modelinin kelimelerin sırasını anlamasını sağlayan en popüler yöntem **Rotary Position Embedding (RoPE)**'dur. RoPE, her kelimeyi bir saatin kolları gibi belirli bir açıyla döndürür.

* 🕰️ **Yelkovan (Yüksek Frekans):** Hızlı döner. Birbirine çok yakın kelimeler arasındaki gramer ve yerel anlam ilişkisini kodlar.
* ⌛ **Akrep (Düşük Frekans):** Çok yavaş döner. Kitabın başındaki kelime ile sonundaki kelime arasındaki uzun menzilli temayı kodlar.

### Peki 4000 Kelimelik Modeli 128.000 Kelimeye Nasıl Çıkarırız?
1. **Standart RoPE (Ekstrapolasyon):** 4000'den sonraki kelimelere saatin kadranında hiç görmediği aşırı yüksek açılar veririz. Model ne yapacağını şaşırır, Perplexity (şaşkınlık/hata) tavan yapar ve anlamsız saçmalar.
2. **Linear Position Interpolation (PI):** Saatin hızını genel olarak 32 kat yavaşlatırız ($m' = m / 32$). Uzun metinler sığar fakat hem akrep hem yelkovan yavaşladığı için model yakın kelimeler arasındaki ince gramer farklarını (yelkovanın hassasiyetini) unutur.
3. **NTK-Aware Scaling:** Neural Tangent Kernel teorisiyle taban frekansı ($base$) ölçeklenir. Yelkovan (yerel gramer) orijinal hızını korurken, sadece akrep (uzun menzil) yavaşlatılır.
4. **YaRN (Nihai Çözüm):** Dalga boyuna göre yumuşak bir rampa ($\gamma(r)$) kurar ve 128k'da dikkat skorlarının aşırı düzleşmesini engelleyen bir sıcaklık düzeltmesi ($t = 0.1 \ln(s) + 1$) uygular. Model 128k bağlamda sanki baştan beri 128k ile eğitilmiş gibi net ve kararlı çalışır!

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Matematiksel Modelleme
Standart RoPE $d$ boyutlu vektörü 2D alt-uzaylara bölerek her $i$-inci çifte açısal dönüşüm uygular:
$$\theta_i = \text{base}^{-2i / d}, \quad R_{\theta_i, m} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$
İç çarpımda göreli mesafe özelliği:
$$\langle R_m q, R_n k \rangle = q^T R_{n-m} k = g(q, k, m-n)$$

### 2. Bağlam Uzatma Yöntemleri ve Matematiksel Dönüşümler
- **Linear PI (Chen et al. 2023):** $m' = m / s$ (burada $s = L_{\text{hedef}} / L_{\text{egitim}}$).
- **NTK-Aware Scaling (bloc97):** $\text{base}' = \text{base} \cdot s^{d / (d-2)}$.
- **YaRN (Peng et al. 2023):**
  - Dalga boyu: $\lambda_i = 2\pi / \theta_i$.
  - Rampa faktörü: $\gamma_i = \text{clamp}\left(\frac{L_{\text{train}}/\lambda_i - \beta_{\text{slow}}}{\beta_{\text{fast}} - \beta_{\text{slow}}}, 0, 1\right)$.
  - Hibrit frekans: $\theta_i^{\text{yarn}} = (1 - \gamma_i) \frac{\theta_i}{s} + \gamma_i \theta_i$.
  - Sıcaklık faktörü: $t = 0.1 \ln(s) + 1.0$.

### 3. Perplexity (PPL) ve Kararlılık Analizi ($4k \to 128k$)

| Yöntem | 4k Bağlam | 16k Bağlam | 64k Bağlam | 128k Bağlam | Kararlılık |
|:---|:---|:---|:---|:---|:---|
| **Standart RoPE** | 8.50 | 179.45 | > 500.00 | **> 500.00** | Katastrofik Çöküş |
| **Linear PI** | 8.50 | 13.50 | 18.50 | **21.00** | Yerel Detay Kaybı |
| **NTK-Aware** | 8.50 | 10.90 | 13.30 | **14.50** | İyi Kararlılık |
| **YaRN (Nihai)** | **8.50** | **9.00** | **9.50** | **9.75** | **Kusursuz Kararlılık** |

### 4. Endüstriyel Entegrasyon (LLaMA-3.1 128k, Mistral, Qwen-2.5)
- **Meta LLaMA-3.1 (8B, 70B, 405B):** $\text{base} = 500,000$ ve YaRN/NTK türevi ölçekleme ile 128k bağlam.
- **Mistral Large 2:** 128k bağlam için optimize edilmiş RoPE ölçeklemesi.
- **Qwen-2.5:** 128k bağlam desteği için YaRN tabanlı frekans enterpolasyonu.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Rotary Position Embedding (RoPE)** | Token vektörlerini açısal rotasyon matrisiyle çarparak göreli konumu koruyan kodlama. |
| **Position Interpolation (PI)** | Pozisyon indekslerini hedef bağlam ölçeğine bölerek interpolasyon yapan temel yöntem. |
| **NTK-Aware Scaling** | Yüksek ve düşük frekansları orantılı ölçeklemek için taban frekansı değiştiren teknik. |
| **YaRN (Yet another RoPE)** | Dalga boyu bazlı rampa ve dikkat entropi sıcaklık düzeltmesi kullanan gelişmiş bağlam uzatıcı. |
| **Context Window (Bağlam Penceresi)** | Bir dil modelinin tek seferde işleyebildiği maksimum token sayısı (örn. 128k). |
| **Wavelength ($\lambda_i$)** | Bir RoPE frekansının bir tam tur dönmesi için gereken token mesafesi ($2\pi / \theta_i$). |
| **Attention Temperature Scaling** | Uzun bağlamlarda softmax dağılımının aşırı düzleşmesini engelleyen sıcaklık çarpanı ($t$). |
| **Extrapolation (Ekstrapolasyon)** | Modelin eğitimde görmediği daha büyük pozisyon açılarına tahmin yürütmesi. |
| **Interpolation (İnterpolasyon)** | Yeni pozisyonları modelin eğitimde gördüğü açı aralığına sıkıştırma işlemi. |
| **Catastrophic Forgetting** | Bağlam uzatılırken modelin kısa metinlerdeki dilbilgisi kurallarını unutması riski. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Sıfır veya minimal fine-tuning     │ • Rampa katsayıları ve sıcaklık      │
 │   ile 128k bağlam desteği.           │   hiperparametreleri optimizasyonu.  │
 │ • Yerel detayları ve uzun menzilli   │ • Aşırı uzun dizilerde FlashAttention│
 │   temaları aynı anda korur.          │   kernel modifikasyonu gereği.       │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • 128k bağlam ile tüm kitapları ve   │ • Donanım seviyesinde KV Cache       │
 │   kod repolarını tek seferde işleme. │   bellek sınırına takılma riski.     │
 │ • LLaMA-3.1 ve Qwen-2.5 standardı.   │ • Bağlam uzadıkça artan çıkarım süresi│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/rope_yarn_baglam_uzatma_paneli.png` dosyası üretilir:
1. **128k Bağlamda Perplexity (PPL - Log Scale)**
2. **Göreli Mesafeye Göre Dikkat Benzerliği Bozulması**
3. **YaRN Frekans Rampa Bölgeleri ($\gamma$ Katsayısı)**
4. **128k Bağlamda Yöntem Kararlılık İndeksi (%)**
5. **RoPE ve Bağlam Uzatma Matematiksel Formül Kartı**
6. **Stajyer Notu & 128k Bağlam Karar Sertifikası**

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
