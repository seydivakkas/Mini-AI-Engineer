# Day 106: Instruction Supervised Fine-Tuning (SFT) & Token Packing (Sıfır Padding Kaybı)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; Instruction SFT eğitiminde GPU hesaplama gücünün %40-%70'ini boşa harcayan padding token israfını sıfırlayan **Token Packing (Multipack / Bin-Packing)**, **Blok-Diyagonal Dikkat Maskelemesi (Block-Diagonal Attention Mask)** ve **Prompt Loss Masking (-100)** mimarilerini sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Kargo Kutusu ve Tetris Analojisi"

Bir e-ticaret deposunda çalıştığınızı düşünün. Müşteriler farklı boyutlarda eşyalar sipariş ediyor: Kimi 1 küçük kitap (50 token), kimi 1 büyük mont (400 token).

1. **Standart Padding (Eski Usul):** Her eşya için standart devasa bir buzdolabı kutusu ($4096$ token) açıyorsunuz. Küçük kitabı kutunun köşesine koyup kalan dev boşluğu strafor köpükle (`<pad>`) dolduruyorsunuz. Kamyon (GPU) strafor köpük taşımaktan helak oluyor ve mazot (elektrik/para) boşa gidiyor!
2. **Token Packing (Yeni Usul - Tetris):** Birden fazla müşterinin siparişlerini aynı büyük kutunun içine Tetris gibi boşluk bırakmadan yan yana diziyorsunuz.
3. **Peki Eşyalar Birbirine Karışmaz mı?**  
   - **Blok-Diyagonal Maske:** Her paketin arasına görünmez bir duvar örülür. A müşterisinin kitabı, B müşterisinin montunu asla göremez (Çapraz dikkat engellenir).
   - **Prompt Maskeleme (`-100`):** Model yalnızca asistanın yazdığı cevapları üretmekten sorumlu tutulur; kullanıcının sorduğu sorularda gradyan hesaplanmaz.

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Matematiksel Modelleme
Instruction SFT eğitiminde kayıp fonksiyonu yalnızca asistan cevapları ($y_{\text{resp}}$) üzerinden hesaplanır:
$$\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{t \in \text{Response}} \log P_\theta(x_t \mid x_{<t})$$

Prompt token'ları için etiket $\text{label}_t = -100$ atanır ve `CrossEntropyLoss(ignore_index=-100)` ile gradyan hesaplaması devre dışı bırakılır.

Paketlenmiş dizide alt-örnekler $k \in \{1, \dots, K\}$ için blok-diyagonal nedensel maske:
$$M_{i, j} = \begin{cases} 0, & \text{eğer } j \le i \text{ ve } \text{ornek}(i) = \text{ornek}(j) \\ -\infty, & \text{aksi takdirde (farklı örnekler)} \end{cases}$$

### 2. Bellek, Hesaplama Karmaşıklığı ve Padding İsrafı Analizi
- **First-Fit Decreasing (FFD) Paketleme:** Sohbetler uzunluklarına göre azalan sıralanır ve en az boşluk bırakacak şekilde $L_{\max}$ torbalarına yerleştirilir.
- **Doluluk Oranı:** FFD algoritması ile torba doluluk oranı **%98.2** seviyesine çıkar; padding kaybı %34-%70 bandından **%1.8'e** düşer.

| Metrik | Standart Padding | Token Packing (FFD) | Kazanç / Tasarruf |
|:---|:---|:---|:---|
| **İşlenen Toplam Token** | 65,896 Token | **44,032 Token** | %33.2 Daha Az İşlem |
| **Boşa Giden Padding** | 22,641 Token | **777 Token** | **%96.6 İsraf Azalması** |
| **Eğitim Adımı Sayısı** | 75 Adım | **43 Adım** | %42.7 Daha Az Forward/Backward |
| **Throughput (Örnek/sn)** | 76.1 ö/s | **359.2 ö/s** | **4.72x Daha Hızlı!** |

### 3. Donanım & GPU Bellek Bant Genişliği (Memory Bandwidth) Etkisi
Standart paddingli eğitimde GPU Tensor Çekirdekleri (Tensor Cores), softmax ve FFN işlemlerinde sıfırlardan oluşan `<pad>` tensörleri üzerinde trilyonlarca anlamsız çarpma-toplama işlemi yapar. Token Packing ile GPU donanımının FLOP/s kullanım oranı (MFU - Model FLOPs Utilization) %15'lerden **%45-%55** bandına fırlar.

### 4. Endüstriyel Entegrasyon (Axolotl, Unsloth, LLaMA-Factory, TRL)
- **Hugging Face TRL (`SFTTrainer`):** `packing=True` ve `dataset_text_field` desteği.
- **FlashAttention-2 `flash_attn_varlen_func`:** `cu_seqlens` (kümülatif dizi uzunlukları) tensörünü alarak blok-diyagonal maskeyi tensör oluşturmadan doğrudan CUDA SRAM seviyesinde hesaplar.
- **Unsloth & Axolotl:** Açık kaynaklı LLM fine-tuning ekosisteminde standart eğitim yaklaşımıdır.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Instruction SFT** | Önceden eğitilmiş bir LLM'e talimat takip etme yeteneği kazandırmak için yapılan denetimli ince ayar. |
| **Token Packing (Multipack)** | Birden fazla değişken uzunluklu metin örneğini tek bir maksimum uzunluk tensöründe birleştirme tekniği. |
| **First-Fit Decreasing (FFD)** | Örnekleri uzunluğa göre azalan sıralayıp ilk sığdığı torbaya yerleştiren optimal paketleme algoritması. |
| **Block-Diagonal Attention Mask** | Paketlenmiş dizide farklı örneklere ait token'ların birbirine dikkat etmesini engelleyen matris. |
| **Cross-Contamination (Çapraz Sızıntı)**| Paketleme sırasında farklı sohbetlerin bağlamlarının birbirine karışarak halüsinasyona yol açması hatası. |
| **Prompt Loss Masking (`-100`)** | Yalnızca asistan cevaplarında kayıp hesaplayıp kullanıcı prompt'larında gradyanı sıfırlayan etiketleme. |
| **Position IDs Reset** | Her alt-örneğin başlangıcında pozisyon indeksini $0$'a sıfırlayarak RoPE/pozisyon doğruluğunu koruma. |
| **Cumulative Sequence Lengths (`cu_seqlens`)**| FlashAttention'a alt-örnek sınırlarını bildiren kümülatif uzunluk dizisi ($[0, N_1, N_1+N_2, \dots]$). |
| **Model FLOPs Utilization (MFU)** | GPU'nun teorik maksimum hesaplama gücünün ne kadarının gerçek faydalı matris çarpımına ayrıldığı oranı. |
| **Ignore Index** | PyTorch `CrossEntropyLoss` fonksiyonunda kayba dahil edilmeyecek özel etiket değeri (`-100`). |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %98+ doluluk oranı ile sıfır       │ • Blok-diyagonal dikkat maskesi      │
 │   padding kaybı.                     │   oluşturma/yönetme zorluğu.         │
 │ • 3x - 5x daha hızlı SFT eğitimi.    │ • Pozisyon ID'lerinin her örnekte    │
 │   Model FLOPs verimliliği (MFU).     │   sıfırlanması zorunluluğu.          │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Tek A100/H100 GPU'da günlerce süren│ • Maskeleme doğru yapılmazsa         │
 │   fine-tuning'i saatler seviyesine   │   örnekler arası bağlam sızıntısı    │
 │   indirme imkânı.                    │   (cross-contamination) tehlikesi.   │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/sft_token_packing_paneli.png` dosyası üretilir:
1. **Toplam İşlenen Token İçinde Padding İsraf Oranı (%) Kıyaslaması**
2. **Blok-Diyagonal Nedensel Dikkat Maskesi Isı Haritası (Örnek Ayrımı)**
3. **SFT Veri Seti Sohbet Uzunluk Dağılımı Histogramı**
4. **SFT Eğitim Throughput (Örnek / Saniye) ve Hızlanma Grafiği (4.72x Hız)**
5. **SFT Loss Masking & Token Packing Matematiksel Formül Kartı**
6. **Stajyer Notu & SFT Token Packing Mimari Karar Sertifikası**

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
