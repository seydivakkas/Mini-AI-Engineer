# Day 105: Sliding Window Attention (SWA - Mistral) & Rolling Buffer Cache

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; Mistral-7B ve Mixtral-8x7B modellerinde kullanılan, milyonlarca token'lık devasa bağlamlarda bile KV Cache bellek tüketimini sabit bir tavanda kilitleyen **Sliding Window Attention (SWA)**, **Rolling Buffer Cache (Dairesel Önbellek)** ve **Etkin Alıcı Alan (Effective Receptive Field)** mimarilerini sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Dönen Bant Analojisi"

Klasik bir Transformer modelinde 100.000 kelimelik bir metin üretirken, 100.000'inci kelime 1'inci kelimeye doğrudan bakmaya çalışır. Bu yüzden GPU belleği sonsuza kadar büyümek zorundadır ($O(S)$ karmaşıklığı).

Mistral mühendisleri şu gerçeği fark etti:  
*"Bir cümlenin dilbilgisi ve anlamı için en önemli kelimeler genellikle son birkaç bin kelimedir ($W = 4096$). Öyleyse her katmanda sadece son $W$ kelimeye baksak ve eski kelimeleri süpermarketin dönen kasası gibi ezerek sabit bir tamponda tutsak ne olur?"*

* 🔄 **Rolling Buffer Cache (Dairesel Kasa Bandı):** Önbellek masamızda sadece 4096 tabaklık yer vardır. 4097'nci kelime geldiğinde 1'inci tabağın üstüne konur ($t \pmod W$). Bellek boyutu asla 1 megabayt bile büyümez! Sabit kalır ($O(W)$).
* 🔭 **Teleskopik Katman İstiflemesi (Receptive Field):** "Eski kelimeleri silersek model geçmişi tamamen unutmaz mı?" **Hayır!** Çünkü:
  - 1. Katman: 0 ile 4096 arasındaki kelimeleri özetler.
  - 2. Katman: 1. katmanın özetlerine bakar ve 8192 kelimelik menzile ulaşır.
  - 32. Katman: Tüm katmanların özetleri birleştiğinde model $32 \times 4096 = \mathbf{131,072 \text{ kelimeyi (128k)}}$ dolaylı olarak eksiksiz anlar!

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Matematiksel Modelleme
Sliding Window Attention, her $i$-inci token'ın yalnızca son $W$ pencere aralığına dikkat etmesini şart koşar:
$$\text{Attention}(Q_i, K_j, V_j) \quad \text{için geçerli aralık: } \max(0, i - W + 1) \le j \le i$$

Bantlı Nedensel Maske (Banded Causal Mask) matrisi:
$$M_{i, j} = \begin{cases} 0, & \text{eğer } j \le i \text{ ve } i - j < W \\ -\infty, & \text{aksi takdirde} \end{cases}$$

### 2. Bellek (Rolling Buffer) ve Hesaplama Karmaşıklığı Analizi
- **Hesaplama Karmaşıklığı (Compute):** Standart $O(S^2)$ karesel patlamasından $O(S \cdot W)$ doğrusal ölçeklenmeye iner.
- **Bellek Karmaşıklığı (KV Cache):** Dizi uzunluğu $S$'den bağımsız olarak $O(W)$ sabitindedir.

| Bağlam Boyutu ($S$) | Full Attention ($O(S)$) | Mistral SWA ($W=512$) | VRAM Tasarruf Oranı |
|:---|:---|:---|:---|
| **512 Token** | 128.0 MB | 128.0 MB | %0.0 |
| **2048 Token** | 512.0 MB | **128.0 MB (Sabit)** | %75.0 Tasarruf |
| **8192 Token** | 2048.0 MB | **128.0 MB (Sabit)** | %93.8 Tasarruf |
| **32768 Token** | 8192.0 MB (8.0 GB) | **128.0 MB (0.125 GB)** | **%98.4 Tasarruf!** |

### 3. Donanım & GPU Bellek Bant Genişliği (Memory Bandwidth) Etkisi
Çıkarım sırasında $S=32k$ olduğunda 8 GB'lık KV cache verisini her adımda GPU belleğinden taşımak yerine, yalnızca 128 MB'lık sabit tampon taşınır. Bu durum bellek veri yolu (Memory Bus) yükünü %98.4 azaltır ve saniyedeki token üretim hızını (Throughput) zirveye taşır.

### 4. Endüstriyel Entegrasyon (Mistral 7B, Mixtral 8x7B)
- **Mistral 7B:** $W = 4096$, $L = 32 \implies \text{Etkin Alıcı Alan} = 32 \times 4096 = \mathbf{131,072 \text{ Token (128k)}}$.
- **Mixtral 8x7B (MoE):** SWA ve Sparse MoE mimarisini birleştirerek hem hesaplama hem bellek tasarrufu sağlar.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Sliding Window Attention (SWA)** | Her token'ın sadece sabit $W$ boyutundaki yerel geçmişe dikkat ettiği pencere mimarisi. |
| **Rolling Buffer Cache** | Modulo $t \pmod W$ indeksleme ile eski token'ların üzerine yazan sabit boyutlu dairesel önbellek. |
| **Banded Causal Mask** | Hem nedenselliği ($j \le i$) hem de pencere sınırını ($i - j < W$) sağlayan bant matrisi. |
| **Effective Receptive Field** | Katmanların istiflenmesiyle modelin dolaylı olarak erişebildiği toplam bağlam uzunluğu ($L \times W$). |
| **Window Size ($W$)** | Bir dikkat katmanının doğrudan erişebildiği ardışık maksimum token sayısı (örn. 4096). |
| **Chunked Pre-fill** | Uzun başlangıç metinlerini $W$ boyutunda parçalara bölerek rolling cache ile kademeli işleme. |
| **Constant Memory Complexity** | Bellek ihtiyacının girdi boyutundan bağımsız olarak $O(W)$ sabitinde kalması. |
| **Information Propagation** | Bilginin alt katmanlardan üst katmanlara doğru pencere adım adım taşınarak yayılması. |
| **Mistral-7B** | SWA ve GQA mimarilerini standartlaştıran öncü açık kaynaklı dil modeli. |
| **FlashAttention SWA** | Sliding Window maskesini GPU SRAM seviyesinde hesaplayarak hızlandıran özel kernel. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Sonsuz bağlamda bile sabit KV cache│ • İlk katmanlar $W$ mesafesinden     │
 │   belleği ($O(W)$).                  │   uzaktaki token'lara doğrudan       │
 │ • $O(S \cdot W)$ doğrusal hesaplama. │   bakamaz.                           │
 │ • 32 katmanda 131k alıcı alan.       │ • Modulo indeksli rolling cache      │
 │ • %98.4 VRAM tasarrufu.              │   yönetim mantığı karmaşıklığı.      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Tek GPU'da yüz binlerce kelimelik  │ • Çok katı iğne-samanlık (NIAH)      │
 │   sohbet geçmişi tutabilme imkânı.   │   görevlerinde ilk katman doğrudan   │
 │ • vLLM ve Mistral ile tam uyumluluk. │   dikkat kaybı yaşayabilir.          │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/swa_rolling_cache_paneli.png` dosyası üretilir:
1. **Bağlam Uzadıkça KV Cache (MB) — Sabit Bellek Tavanı Grafiği**
2. **Katman Sayısına Göre Etkin Alıcı Alan ($L \times W$) Genişlemesi**
3. **Bantlı Nedensel Maske Matrisi Görseli (Heatmap)**
4. **32,768 Token Bağlamda KV Cache VRAM Kıyaslaması (8 GB vs 0.125 GB)**
5. **Mistral SWA ve Rolling Buffer Matematiksel Formül Kartı**
6. **Stajyer Notu & Mistral SWA Mimari Karar Sertifikası**

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
