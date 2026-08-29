# Day 102: Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) ile KV Cache Azaltma

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; modern büyük dil modellerinde (LLaMA-3, Mistral-7B, Gemma) çıkarım (inference) anındaki VRAM krizini çözen **Grouped-Query Attention (GQA)** ve **Multi-Query Attention (MQA)** mekanizmalarını teorik, matematiksel ve deneysel olarak ele alır.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Kütüphane Analojisi"

Bir dil modeli metin üretirken (örneğin ChatGPT gibi kelime kelime yazarken), her yeni kelimede geçmişte ne yazdığını hatırlamak zorundadır.

* **Eski Yöntem (MHA - Multi-Head Attention):** Diyelim ki 32 araştırmacı (32 Query Başlığı) bir kütüphanede çalışıyor. Her araştırmacı için kütüphaneci masaya ayrı ayrı 32 adet dev ansiklopedi seti (Key ve Value) getirip koyar. 32 araştırmacı $\times$ 32 ansiklopedi = Masada 1024 dev cilt kitap birikir! Bağlam uzadıkça masanın (GPU VRAM) üzerinde yer kalmaz ve sistem çöker (OOM).
* **Aşırı Tasarruf Yöntemi (MQA - Multi-Query Attention):** Kütüphaneci masaya yalnızca **1 tek** ansiklopedi koyar ve 32 araştırmacının hepsi aynı kitabı sırayla okumaya çalışır. Bellek masrafı miniciktir ama araştırmacılar birbirinin ayağına basar, modelin detayları yakalama kalitesi ve akıl yürütme gücü düşer.
* **Altın Oran (GQA - Grouped-Query Attention):** 32 araştırmacıyı 8 gruba ayırırız (her grupta 4 araştırmacı). Kütüphaneci her grup için sadece 1 ansiklopedi seti koyar (toplam 8 set). Bellek tüketimi anında **4 kat (%75) azalır**, üstelik araştırmacılar birbirini kısıtlamadığı için model kalitesi MHA ile neredeyse birebir aynı kalır! İşte LLaMA-3 ve Mistral'in sırrı budur.

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Matematiksel Modelleme
Standart Multi-Head Attention'da $H_q$ adet Query başlığı, $H_{kv} = H_q$ adet Key ve Value başlığı ile eşleşir:
$$\text{MHA: } H_q = H_{kv} = 32$$

Grouped-Query Attention'da ise $G = H_{kv}$ adet grup oluşturulur ve her grupta $n = H_q / H_{kv}$ adet Query başlığı tek bir Key-Value çiftini paylaşır:
$$\text{GQA: } H_q = 32, \quad H_{kv} = 8 \implies n = 4 \text{ (Her 4 Q başlığına 1 KV başlığı)}$$

İleri geçiş esnasında KV tensörleri broadcast/repeat edilerek standart SDPA fonksiyonuna sokulur:
$$K_{\text{rep}} = \text{repeat\_kv}(K, n), \quad V_{\text{rep}} = \text{repeat\_kv}(V, n)$$
$$\text{Attention}(Q, K_{\text{rep}}, V_{\text{rep}}) = \text{softmax}\left(\frac{Q K_{\text{rep}}^T}{\sqrt{d_k}}\right) V_{\text{rep}}$$

### 2. Bellek (KV Cache) ve Hesaplama Karmaşıklığı Analizi
$L$ katmanlı, $B$ batch boyutlu, $S$ dizi uzunluklu ve $d_h$ kafa boyutuna sahip bir modelde KV Cache bellek tüketimi (FP16):
$$\text{KV Cache (Bayt)} = 2 \times L \times B \times H_{kv} \times S \times d_h \times 2$$

| Metrik | MHA ($H_{kv}=32$) | GQA ($H_{kv}=8$) | MQA ($H_{kv}=1$) |
|:---|:---|:---|:---|
| **4096 Token (MB)** | **4096.0 MB** | **1024.0 MB** | **128.0 MB** |
| **8192 Token (MB)** | **8192.0 MB** | **2048.0 MB** | **256.0 MB** |
| **VRAM Tasarrufu** | Referans (%0) | **%75.0 Tasarruf** | **%96.9 Tasarruf** |
| **Model Kalitesi (PPL)** | En Yüksek | **Kayba Uğramaz** | Hafif Bozulma |

### 3. Donanım & GPU Bellek Bant Genişliği (Memory Bandwidth) Etkisi
Otoregresif token üretiminde (Decode aşaması), GPU hesaplama çekirdeklerinden ziyade **Bellek Bant Genişliği (Memory Bandwidth Bound)** sınırına takılır. Her token üretilirken devasa KV Cache tensörünün HBM bellekten SRAM'e çekilmesi gerekir. GQA, taşınan veri hacmini 4 kat azalttığı için GPU bellek veri yolunu rahatlatır ve çıkarım throughput'unu doğrudan katlar.

### 4. Endüstriyel Entegrasyon (LLaMA-3, Mistral-7B, Gemma)
Modern açık kaynaklı ve kapalı kaynaklı temel modellerin neredeyse tamamı GQA standardına geçmiştir:
- **Mistral 7B & Mixtral 8x7B:** $H_q=32, H_{kv}=8$ (GQA-8).
- **Meta LLaMA-3 (8B ve 70B):** $H_q=32, H_{kv}=8$ (GQA-8).
- **Gemma-2 (9B ve 27B):** $H_q=16, H_{kv}=8$ (GQA-2 / GQA-8).

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Multi-Head Attention (MHA)** | Her dikkat başlığı için bağımsız Q, K ve V projeksiyonları kullanan klasik dikkat yapısı. |
| **Multi-Query Attention (MQA)** | Tüm Q başlıklarının tek bir ortak K ve V başlığını paylaştığı aşırı hafif dikkat mimarisi. |
| **Grouped-Query Attention (GQA)** | Q başlıklarını gruplara ayıran ve her gruba 1 adet KV başlığı tahsis eden dengeli mimari. |
| **KV Cache** | Otoregresif üretimde geçmiş token'ların anahtar ve değerlerini saklayarak tekrar hesaplamayı önleyen önbellek. |
| **Prefill Aşaması** | Sisteme verilen ilk prompt'un tüm token'larının paralel olarak işlendiği ilk adım ($S > 1$). |
| **Decode Aşaması** | Modelin kelime kelime yeni token ürettiği adım adım çalışan aşama ($S = 1$). |
| **Memory Bandwidth Bound** | İşlemcinin hesaplama kapasitesinden ziyade belleğin veri aktarım hızına takıldığı donanım durumu. |
| **Repeat KV (Broadcast)** | $H_{kv}$ boyutundaki tensörü Query başlık sayısına ($H_q$) kopyalamadan genişletme işlemi. |
| **Head Dimension ($d_h$)** | Her bir dikkat başlığının vektör boyutu ($d_{\text{model}} / H_q$). |
| **Perplexity (PPL)** | Bir dil modelinin sonraki kelimeyi ne kadar iyi tahmin ettiğini ölçen belirsizlik metriği (düşük daha iyi). |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • KV Cache VRAM tüketiminde %75      │ • MQA kadar aşırı küçük bellek       │
 │   tasarruf.                          │   sağlamaz (MQA %96 tasarruf eder).  │
 │ • MHA ile birebir model kalitesi.    │ • Repeat KV fonksiyonunun tensör     │
 │ • GPU bellek darboğazını çözer.      │   yeniden şekillendirme maliyeti.    │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • 128k+ bağlam pencerelerini tek     │ • $H_q$ başlık sayısının $H_{kv}$    │
 │   GPU'ya sığdırabilme imkânı.        │   sayısına tam bölünme zorunluluğu.  │
 │ • vLLM ve TensorRT-LLM tam desteği.  │ • Eski MHA kontrol noktalarından     │
 │ • LLaMA-3 ve Mistral ile uyumluluk.  │   GQA'ya geçişte uptraining gereği.  │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/gqa_mqa_kv_cache_paneli.png` dosyası oluşturulur:
1. **Bağlam Uzunluğuna Göre KV Cache (MB)**
2. **P50 Çıkarım Gecikmesi (ms)**
3. **Çıkarım Throughput (Tokens/s)**
4. **MHA vs GQA vs MQA Mimari Şeması**
5. **4096 Bağlamda VRAM Tasarruf Oranları**
6. **Stajyer Notu & GQA Mimari Karar Özeti**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim ve entegrasyon testlerini çalıştırın (8/8 Test)
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
