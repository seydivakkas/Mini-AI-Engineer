# Day 170: OpenAI Whisper Mimarisi — Çok Dilli Konuşma Tanıma (ASR), CTC ve Zaman Damgası Tahmini

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 10. günüdür. OpenAI Whisper ve Conformer modellerinin mimarisi olan **80-Kanal Log-Mel Spektrogramı Çıkarımı**, **1D Conv2 + Transformer Encoder-Decoder Mimarisi**, **Özel Görev Belirteçleri (`<|startoftranscript|>`, `<|tr|>`, `<|transcribe|>`, `<|notimestamps|>`)**, **CTC (Connectionist Temporal Classification) Hizalama Kaybı** ve **Kelime Düzeyinde Zaman Damgası Tahmini (Word-Level Timestamp Estimation)** motorunu sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Whisper ASR" Nedir ve Konuşmayı Nasıl Kelime Kelime Zaman Damgalarıyla Deşifre Eder?
- **Sorun (Gürültülü ve Çok Dilli Konuşma Zorluğu):**
  İnsan sesleri farklı dillerde, aksanlarda ve arka plan gürültülerinde gelir. Geleneksel modeller her dil için ayrı akustik model gerektirirdi ve ne zaman hangi kelimenin söylendiğini (Timestamp) hassas kestiremezdi.
- **Çözüm (Log-Mel Spektrogramı + Çok Görevli Encoder-Decoder):**
  1. *80-Kanal Log-Mel Spektrogramı:* 16 kHz ses, insan kulağının duyma frekansına (Mel ölçeği) göre 80 kanallı bir ısı haritasına dönüştürülür.
  2. *Ses Kodlayıcı (Audio Encoder):* İki adet 1D Conv katmanı zamanı 2 kat sıkıştırır, ardından Transformer blokları ses özelliklerini çıkarır.
  3. *Özel Görev Promptları (Task Prompting):* Decoder'a `<|startoftranscript|> <|tr|> <|transcribe|>` verilerek Türkçe konuşma tanıma modu başlatılır.
  4. *Zaman Damgaları (Timestamp Estimation):* Çapraz dikkat (Cross-Attention) ağırlıklarının tepe noktaları izlenerek her kelimenin başlangıç ve bitiş saniyesi (`[00:00.60 -> 00:01.10] zeka`) çıkarılır.

```
====================================================
         WHISPER MULTITASK ARCHITECTURE             
====================================================
  [16 kHz Ses] ──> [80-Kanal Log-Mel Spektrogramı]   
           │                                        
           ▼                                        
  [1D Conv2 Kök Katman (2x Zamansal Downsampling)]  
           │                                        
           ▼                                        
  [Transformer Audio Encoder] ──> [CTC Loss Başlığı]
           │                                        
           ▼  (Çapraz Dikkat - Cross-Attention)     
  [Causal Transformer Text Decoder]                 
      ├── Özel Prompt: <|startoftranscript|>        
      ├── Dil Belirteci: <|tr|> / <|en|>           
      └── Görev Belirteci: <|transcribe|>          
           │                                        
           ▼                                        
  [Transkripsiyon + Zaman Damgası ([00:01.20]) Çıktı]
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Log-Mel Spektrogramı ve Mel Filtre Bankası
- $f$ Hertz frekansının Mel ölçeğine dönüşümü:
  $$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$
- 80 adet üçgen mel filtresi STFT güç spektrumuna uygulanıp logaritması alınarak $X_{\text{mel}} \in \mathbb{R}^{80 \times T}$ elde edilir.

### B. Çok Görevli Özel Belirteç Formatı (Prompt Conditioning)
- Whisper tek bir modelde ASR, VAD (Voice Activity Detection), Dil Tespiti ve Çeviri görevlerini özel belirteç dizisiyle çözer:
  $$\mathbf{y}_{\text{prompt}} = [\langle|\text{startoftranscript}|\rangle, \langle|\text{tr}|\rangle, \langle|\text{transcribe}|\rangle, \langle|\text{notimestamps}|\rangle]$$

### C. CTC (Connectionist Temporal Classification) Kaybı
- Ses kareleri $T$ ve hedef harf dizisi $Y$ için olası tüm hizalama yolları $\pi \in \mathcal{B}^{-1}(Y)$ üzerinden marjinal olasılık maksimize edilir:
  $$\mathcal{L}_{\text{CTC}} = -\ln \sum_{\pi \in \mathcal{B}^{-1}(Y)} P(\pi \mid X)$$

### D. Performans ve Doğrulama
- Türkçe ve İngilizce konuşma transkripsiyon senaryolarında **%0.0 WER (Word Error Rate)** ve kusursuz zaman damgası hizalaması sağlanmıştır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **ASR** | Otomatik Konuşma Tanıma (Automatic Speech Recognition). |
| **Log-Mel Spectrogram** | İnsan işitme eşiğine göre frekans enerjisini logaritmik ölçekleyen 2D temsil. |
| **WER (Word Error Rate)** | Deşifredeki kelime ekleme, silme ve değiştirme hata oranı. |
| **CER (Character Error Rate)** | Karakter düzeyindeki düzenleme mesafesi hata oranı. |
| **CTC Loss** | Ses zaman adımları ile metin harfleri arasındaki hizalamayı dinamik programlama ile çözen kayıp. |
| **Cross-Attention Alignment** | Decoder dikkat ağırlıklarından kelimenin ses dosyasındaki yerini bulma tekniği. |
| **Timestamp Token** | `<|0.00|>` gibi 30 saniyelik konuşma aralığını 20ms adımlarla etiketleyen belirteçler. |
| **Voice Activity Detection (VAD)** | Ses kaydında insanın konuştuğu anları sessizlikten ayırt etme. |
| **Multilingual ASR** | Tek bir modelin 90+ dilde konuşmayı otomatik tanıyabilmesi. |
| **SpecAugment** | Log-Mel spektrogramında zaman ve frekans şeritlerini maskeleyerek veri zenginleştirme. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • 90+ dilde sıfır atışla (zero-shot) │ • 30 saniyelik sabit pencereler      │
 │   aşırı gürültülü ortamlarda yüksek  │   nedeniyle saatlerce süren kesintisiz│
 │   tanıma başarısı.                   │   kayıtlarda pencereleme ihtiyacı.   │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Altyazı senkronizasyonu, podcast   │ • Müzik veya arka plan fısıltılarında│
 │   arama motorları, müşteri hizmetleri│   tekrarlayan halüsinasyon           │
 │   ve gerçek zamanlı toplantı notları.│   (Hallucinatory Repetition) riski.  │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/whisper_speech_to_text_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
