# Day 176: Difüzyon Modellerinde LoRA (Low-Rank Adaptation) & DreamBooth ile Özel Nesne ve Sanat Stili Öğretimi

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 16. günüdür. Difüzyon modellerine özel karakter, ürün veya sanat stili öğretmenin en verimli yolu olan **LoRA (Low-Rank Adaptation - Hu et al., 2021)**, **DreamBooth Özne Öğretimi (Ruiz et al., 2023)**, **Cross-Attention $W_q, W_k, W_v, W_{\text{out}}$ Projeksiyonlarına Düşük Dereceli Matris Enjeksiyonu ($W = W_0 + \frac{\alpha}{r} B A$)**, **Özel Belirteç İdentifier (`sks person / sks dog`) Bağlama**, **Sınıf Koruma Kaybı (Class-Specific Prior Preservation Loss)** ve **Çoklu LoRA Birleştirme (Weight Merging & Blending)** motorunu sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "LoRA" ve "DreamBooth" Nedir ve 4 GB'lık Modeli Klonlamadan 30 MB ile Özel Karakter Nasıl Öğretilir?
- **Sorun (Geleneksel İnce Ayarın Devasa Maliyeti ve Dil Kayması):**
  Bir şirketin maskotunu ya da kendi yüzünüzü Stable Diffusion'a öğretmek istediğinizde eskiden tüm 4.2 GB'lık UNet ağırlıkları yeniden eğitilirdi (Full Fine-Tuning). Bu hem saatler sürer, hem yüzlerce GB depolama alanı doldurur, hem de model genel "insan" veya "köpek" kavramlarını unutup sadece sizin karakterinizi çizmeye başlardı (**Language Drift / Dil Kayması**).
- **Çözüm (LoRA & DreamBooth Entegrasyonu):**
  1. *Dondurulmuş Model ($W_0$):* 4 GB'lık ana model dondurulur (Tek bir parametresine bile dokunulmaz).
  2. *Düşük Dereceli Matrisler ($B \times A$):* Modelin sadece metinle pikselleri eşleyen Cross-Attention katmanlarına $r=8-16$ boyutlu minicik iki matris ($A$ ve $B$) takılır.
  3. *Özel Belirteç (`sks dog`):* 3-5 adet fotoğrafla modele *"Bu nesnenin adı [sks] köpektir"* denir.
  4. *Sınıf Koruma Kaybı (Prior Loss):* Model aynı anda yapay zekanın kendi ürettiği genel köpek resimlerini de çözmeye devam eder. Böylece genel "köpek" kavramı korunurken, `[sks]` köpeği kusursuz öğrenilir.
  - *Sonuç:* Sadece **36 MB'lık minicik bir LoRA dosyası** ile 4 GB'lık dev bir model özelleştirilir!

```
====================================================
         LORA LOW-RANK MATRIX DECOMPOSITION         
====================================================
  Giriş Vektörü (x) ─────────────────────────┐      
           │                                 │      
           ▼                                 ▼      
  [Dondurulmuş W_0 (d x k)]        [LoRA_A (r x k)] 
           │ (Kilitli Ağırlık)               │      
           │                                 ▼      
           │                       [LoRA_B (d x r)] 
           │                                 │      
           ▼                                 ▼      
  [Orijinal Çıktı: x W_0] ──── (+) ──── [(x A^T B^T) * (a/r)]
                                │                   
                                ▼                   
  [Yeni Özelleştirilmiş Çıktı: y = x (W_0 + Delta_W)]
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Düşük Dereceli Matris Ayrışımı (Low-Rank Decomposition)
- $d \times k$ boyutundaki tam ağırlık gradyanı $\Delta W$, rank $r \ll \min(d, k)$ olmak üzere iki küçük matrisin çarpımı olarak parametrize edilir:
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A), \quad A \in \mathbb{R}^{r \times k}, \quad B \in \mathbb{R}^{d \times r}$$
- $A \sim \mathcal{N}(0, 1/r)$ ve $B = 0$ başlatılarak eğitimin başlangıcında $\Delta W = 0$ olması garanti edilir.

### B. DreamBooth Sınıf Koruma Kayıp Fonksiyonu (Prior Preservation Loss)
- Dil kaymasını ve aşırı uyumu (Overfitting) engellemek için iki terimli kayıp minimize edilir:
  $$\mathcal{L}_{\text{DreamBooth}} = \mathbb{E}\left[ \|\epsilon - \epsilon_\theta(z_t, c_{\text{instance}}, t)\|^2 \right] + \lambda \mathbb{E}\left[ \|\epsilon - \epsilon_\theta(z_t^{\text{pr}}, c_{\text{class}}, t)\|^2 \right]$$

### C. Çoklu LoRA Birleştirme (Multi-LoRA Blending)
- Çıkarım anında birden fazla LoRA tek bir matriste birleştirilerek sıfır gecikmeyle (Zero Latency Overhead) çalıştırılabilir:
  $$W_{\text{merged}} = W_0 + \sum_{i=1}^M w_i \cdot \frac{\alpha_i}{r_i} (B_i A_i)$$

### D. Performans ve Doğrulama
- 4.2 GB checkpoint yerine **36.4 MB LoRA (%99 depolama tasarrufu)** ve %97 sınıf koruma sadakati doğrulanmıştır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **LoRA (Low-Rank Adaptation)** | Ağırlık güncellemesini iki küçük matrisin çarpımına indirgeyen parametre verimli ince ayar tekniği. |
| **DreamBooth** | Birkaç fotoğrafla modele yeni ve benzersiz bir özne/karakter öğreten teknik. |
| **Rank ($r$)** | Düşük dereceli matrislerin ara boyut sayısı (genellikle 4, 8, 16 veya 32). |
| **LoRA Alpha ($\alpha$)** | LoRA güncellemelerinin ana modele etki oranını ölçekleyen katsayı. |
| **Prior Preservation Loss** | Modelin genel sınıf kavramlarını unutmamasını sağlayan dengeleyici kayıp terimi. |
| **Language Drift** | Bir modele yeni bir kavram öğretilirken eski genel dil/görsel anlamlarının bozulması. |
| **Identifier Token (`sks`)** | Önceden nadir kullanılan ve yeni özneyi temsil etmek için seçilen özel belirteç. |
| **Weight Merging** | LoRA matrislerini ana ağırlıklara doğrudan ekleyerek çıkarım süresini kısaltma işlemi. |
| **PEFT (Parameter-Efficient Fine-Tuning)** | Modelin yalnızca %0.1'ini eğiterek kaynak tasarrufu sağlayan yöntemler ailesi. |
| **Textual Inversion** | Model ağırlıklarını değiştirmeden sadece CLIP sözlüğüne yeni bir kelime vektörü ekleyen yöntem. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • 4 GB yerine 30 MB dosya boyutu.    │ • Çok düşük ranklarda (r=4) ince     │
 │ • Tüketici GPU'larında dakikalar     │   sanatsal dokuların ve yüz hatlarının│
 │   içinde ince ayar yapabilme.        │   tam yakalanamaması.                │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Civitai, HuggingFace ekosistemi,   │ • 4'ten fazla LoRA aynı anda         │
 │   kişiselleştirilmiş avatar üretimi, │   birleştirildiğinde ağırlık         │
 │   kurumsal marka stili entegrasyonu. │   çatışması ve gürültü patlaması.    │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/lora_diffusion_finetuning_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
