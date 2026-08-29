# Day 169: Sinirsel Ses Sıkıştırma (EnCodec / SoundStream & RVQ) — Sürekli Sesi Ayrık Tokenlara Bölme

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 9. günüdür. Meta EnCodec, Google SoundStream ve Audio-LLM (Moshi, SpeechGPT, VALL-E) mimarilerinin temel yapı taşı olan **Sürekli Ses Dalgalarını Ayrık Tokenlara Bölme (Discrete Neural Audio Tokens)**, **Artık Vektör Kuantalama (Residual Vector Quantization - RVQ)**, **Kod Defteri (Codebook Perplexity)** ve **Ses Yeniden Yapılandırma (Reconstruction)** motorunu sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Neural Audio Tokenizer" Nedir ve Neden LLM'ler İçin Zorunludur?
- **Sorun (Metin Tokenizer'ı Ses Anlamaz):**
  Metin modelleri (LLM) BPE ile kelime parçalarını token indekslerine (`[1523, 902]`) böler. Ancak insan sesi sürekli bir analog dalgadır (24 kHz = saniyede 24.000 float sayı!). Bu devasa float dizisini LLM'e doğrudan veremeyiz.
- **Çözüm (EnCodec + Residual Vector Quantization):**
  1. *1D Conv Kodlayıcı:* 24.000 sayılık ses dalgasını 75 Hz zaman adımlı gizli vektörlere ($z \in \mathbb{R}^{D}$) sıkıştırır.
  2. *Kademeli RVQ (8 Katman):* İlk katman ($Q_1$) ana sesi kodlar, ikinci katman ($Q_2$) arta kalan hatayı (residual), üçüncü katman ($Q_3$) onun da artığını kodlar.
  3. *Ayrık Kod Defteri İndeksleri:* Her zaman adımı için 8 adet tam sayı token üretilir: `[412, 89, 902, 12, 650, 321, 104, 77]`.
  4. *LLM Uyumluluğu:* Artık LLM sesi tıpkı bir metin dili gibi okuyabilir ve konuşabilir!

```
====================================================
         ENCODEC / SOUNDSTREAM RVQ ARCHITECTURE     
====================================================
  [Ham Ses Dalgası (24 kHz PCM Waveform: 1D Signal)] 
           │                                        
           ▼                                        
  [1D Strided Conv Encoder (4x Zamansal Sıkıştırma)]
           │                                        
           ▼  (Sürekli Gizli Vektör z ∈ R^D)         
  [Residual Vector Quantizer (RVQ - N_q=8 Katman)]  
      ├── Q1: z1 = Codebook1[idx1], Artık r1 = z - z1
      ├── Q2: z2 = Codebook2[idx2], Artık r2 = r1 - z2
      └── ... Q8: z8 = Codebook8[idx8]              
           │                                        
           ▼  (Ayrık Ses Tokenları: [N_q, T])       
  [1D Transposed Conv Decoder] ──> [Yeniden Ses Dalga]
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Kademeli Artık Vektör Kuantalama (RVQ) Formülasyonu
- Başlangıçta artık $r_0 = z$. $i = 1, \dots, N_q$ kuantalayıcı katmanları için:
  $$z_q^{(i)} = \mathcal{C}_i\big[\arg\min_k \|r_{i-1} - e_k^{(i)}\|_2^2\big], \quad r_i = r_{i-1} - z_q^{(i)}$$
- Toplam kuantalanmış vektör tüm katmanların toplamıdır:
  $$\hat{z} = \sum_{i=1}^{N_q} z_q^{(i)}$$

### B. Düz Geçiş Gradyan Tahmini (Straight-Through Estimator - STE)
- $\arg\min$ işlemi türevlenemez olduğundan geri yayılımda (backward pass) gradyan doğrudan kopyalanır:
  $$\hat{z}_{\text{STE}} = z + (\hat{z} - z).\text{detach}()$$

### C. Bit Hızı (Bitrate) Hesabı
- $F_s = 75\text{ Hz}$, $N_q = 8$, $|\mathcal{C}| = 1024 = 2^{10}$ için:
  $$\text{Bitrate} = \frac{75 \times 8 \times 10}{1000} = 6.0\text{ kbps}$$
- Ham 24 kHz 16-bit PCM ses (384 kbps) ile kıyaslandığında **%98.4 sıkıştırma oranı** elde edilir.

### D. Performans ve Doğrulama
- 8 katmanlı RVQ ile kod defteri perplexity'si ve rekonstrüksiyon doğruluğu test edilerek **8/8 test PASSED** ile onaylanmıştır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Neural Audio Codec** | Ses dalgalarını sinir ağlarıyla sıkıştırıp açan yapay zeka kodlayıcısı. |
| **Residual Vector Quantization (RVQ)** | Kademeli kod defterleriyle arta kalan hatayı iteratif kuantalayan yöntem. |
| **Codebook (Kod Defteri)** | Kuantalama için öğrenilmiş $K$ adet temsil vektörünün bulunduğu sözlük. |
| **Codebook Perplexity** | Kod defterindeki vektörlerin ne kadar dengeli ve çeşitli kullanıldığının ölçüsü. |
| **Straight-Through Estimator (STE)** | Ayrık kuantalama adımından geriye gradyan akmasını sağlayan hile. |
| **Signal-to-Noise Ratio (SNR)** | Orijinal sinyal gücünün rekonstrüksiyon gürültüsüne oranı (dB). |
| **EnCodec** | Meta AI tarafından geliştirilen 1D Conv ve RVQ tabanlı açık kaynak ses tokenizer'ı. |
| **SoundStream** | Google tarafından geliştirilen ilk end-to-end sinirsel ses codec mimarisi. |
| **Discrete Audio Token** | Sesin harf/kelime gibi sayısal bir tam sayı kimliğiyle ifade edilmesi. |
| **Transposed 1D Conv** | Sıkıştırılmış gizli vektörü orijinal ses dalgası uzunluğuna genişleten katman. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Ses sinyalini LLM uyumlu ayrık     │ • Yüksek katman sayısı ($N_q$)       │
 │   token formatına dönüştürme.        │   kullanıldığında autoregressive     │
 │ • 6.0 kbps gibi ultra düşük bit hızı.│   üretimde token uzunluğunun artması.│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Speech-to-Speech LLM'ler (Moshi,   │ • Düşük kod defteri boyutlarında     │
 │   VoiceGPT), ses klonlama (VALL-E)   │   arka plan müziği veya fısıltılı    │
 │   ve gerçek zamanlı sesli asistanlar.│   seslerde yapaylaşma (Metallic art).│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/audio_tokenizer_encodec_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
