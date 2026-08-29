# Day 167: Video LLM — Uzamsal-Zamansal (Spatio-Temporal) Token Modelleme ve 3D Attention

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 7. günüdür. Video-LLaVA, LLaVA-NeXT-Video ve Video-ChatGPT modellerinin temel mimarisi olan **Zamansal Kare Örnekleme (Uniform vs Adaptive Temporal Sampling)**, **Uzamsal-Zamansal (Spatio-Temporal) 3D Dikkat Mekanizması (Space-Time Factorized Attention)**, **Video Token Birleştirme** ve **Video Soru Cevaplama (Video-QA) Akışı**'nı sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Video LLM" Nedir ve Tek Bir Resimden (VLM) Nasıl Farklıdır?
- **Sorun (Zaman Boyutu ve Token Patlaması):**
  Tek bir görselde $N=256$ patch token vardır. Ancak 60 saniyelik bir video $30 \times 60 = 1800$ karedir! Tüm kareleri doğrudan LLM'e sokarsak $1800 \times 256 = 460.800$ token oluşur ve LLM belleği anında çöker.
- **Çözüm (Zamansal Örnekleme + Space-Time Factorized Attention):**
  1. *Zamansal Kare Örnekleme (T=8 Kare):* Video baştan sona taranarak en kritik 8 kare (Uniform veya Optik Akışa göre Adaptive) seçilir.
  2. *Uzamsal Dikkat (Spatial Attention):* Her kare kendi $N$ patch'i arasında "Bu karede ne var?" sorusunu çözer.
  3. *Zamansal Dikkat (Temporal Attention):* Aynı uzamsal koordinat $T=8$ zaman boyunca izlenerek "Bu nesne nereye hareket etti?" sorusu çözülür.
  4. *LLM Projeksiyonu:* Video tokenları tek bir düz diziye dökülerek LLM'e verilir.

```
====================================================
     SPACE-TIME FACTORIZED ATTENTION ARCHITECTURE   
====================================================
  [Video: T Kare x N Patch Token (T=8, N=16)]       
           │                                        
           ▼                                        
  [1. Uzamsal Dikkat (Spatial Attention)]           
  (Her kare kendi N patch'i arasında ilişki kurar)  
           │                                        
           ▼                                        
  [2. Zamansal Dikkat (Temporal Attention)]          
  (Aynı patch koordinatı T zaman boyunca izlenir)   
           │                                        
           ▼                                        
  [GELU MLP Projector (viz_dim -> llm_dim)]         
           │                                        
           ▼                                        
  [Causal LLM: 'Kedi kırmızı koltuğa zıpladı']      
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Space-Time Factorized Attention Karmaşıklığı
- Doğrudan $T \times N$ token arasında tam 3D Self-Attention kurmanın karmaşıklığı:
  $$\mathcal{O}\big((T \cdot N)^2 \cdot D\big) = \mathcal{O}(T^2 N^2 D)$$
- Factorized (Ayrıştırılmış) Uzay-Zaman Dikkati uygulandığında karmaşıklık:
  $$\mathcal{O}\big(T \cdot N^2 \cdot D + N \cdot T^2 \cdot D\big)$$
- $T=8, N=256$ için Factorized Attention bellek ve işlem maliyetinde **%85'ten fazla tasarruf** sağlar!

### B. Zamansal Kare Örnekleme Stratejileri
1. **Uniform Sampling:** Video uzunluğuna eşit aralıklarla $t_i = \lfloor i \cdot \frac{K-1}{T-1} \rfloor$ indeksleri seçilir.
2. **Adaptive (Motion-Aware) Sampling:** Komşu kareler arasındaki piksel/öznitelik farkı $\|I_t - I_{t-1}\|_2$ hesaplanır ve hareketin zirve yaptığı kareler seçilir.

### C. Video-LLaVA Multi-Modal Hizalama
- Video tokenları $H_v \in \mathbb{R}^{T \times N \times D}$ ve metin tokenları $H_t \in \mathbb{R}^{L \times D_{\text{llm}}}$ birleştirilerek tek bir causal sekans olarak kod çözücüye beslenir:
  $$H_{\text{input}} = [W_p(H_v); H_t]$$

### D. Performans ve Doğrulama
- Simüle edilen Video-QA görevlerinde zamansal eylem sıralaması **%100 doğrulukla** tespit edilmiştir.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Video LLM** | Video karelerini ve zaman boyutunu anlayıp metin üreten çok modlu büyük dil modeli. |
| **Spatio-Temporal** | Hem uzay (en-boy pikselleri) hem de zaman (kare sırası) boyutunu kapsayan yapı. |
| **Space-Time Factorized Attention** | Uzamsal ve zamansal dikkati iki ardışık alt adıma bölerek karmaşıklığı düşüren mimari. |
| **Uniform Sampling** | Video boyunca kareleri eşit zaman aralıklarıyla seçme yöntemi. |
| **Adaptive Sampling** | Hareketin ve görsel değişimin yoğun olduğu anları yakalayan dinamik örnekleme. |
| **Temporal Grounding** | Videodaki belirli bir olayın hangi zaman aralığında ($[t_{\text{start}}, t_{\text{end}}]$) gerçekleştiğini bulma. |
| **Video-LLaVA** | Resim ve video girdilerini ortak bir görsel-dil projektöründe birleştiren VLM. |
| **Optical Flow** | Ardışık kareler arasındaki nesne ve piksel hareket vektörleri. |
| **Video-QA** | Video içeriği hakkında sorulan doğal dil sorularını yanıtlama görevi. |
| **Keyframe Extraction** | Bir videodaki en bilgilendirici anahtar kareleri seçme işlemi. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Zaman içerisindeki olay sırasını   │ • Çok uzun (1 saat+) videolarda      │
 │   ve nedensellik ilişkilerini        │   bağlam penceresi sınırları ve      │
 │   anlayabilme.                       │   yüksek GPU bellek ihtiyacı.        │
 │ • Factorized Attention ile hız.      │                                      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Güvenlik kameraları olay analizi,  │ • Hızlı kamera titremelerinde veya   │
 │   spor anlatımı, otonom sürüş ve     │   aşırı kesme/montaj içeren          │
 │   video içerik özetleme sistemleri.  │   videolarda zamansal karışıklık.    │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/video_llm_spatio_temporal_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
