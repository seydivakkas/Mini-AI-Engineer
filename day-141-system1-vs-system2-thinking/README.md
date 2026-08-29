# Day 141: System 1 (Hızlı/Sezgisel) vs System 2 (Yavaş/Akıl Yürüten) LLM Mimarisi ve FAZ 8 Başlangıcı

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; Daniel Kahneman'ın bilişsel teorisini modern derin akıl yürüten LLM mimarilerine (OpenAI o1, DeepSeek-R1) uyarlayan **System 1 (Hızlı / Sezgisel Otoregresif Çıkarım)** ve **System 2 (Yavaş / Düşünme Bütçeli Adım Adım Mantıksal Akıl Yürütme)** motorlarını, **Test-Time Compute Ölçekleme Yasası**nı ve **Bilişsel Yansıma Testi (CRT)** değerlendirme boru hattını içermektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Neden LLM'ler Hızlı Cevap Verirken Hata Yapar, Düşününce Doğruyu Bulur?
- **System 1 (Hızlı / Refleksif Düşünme):**
  Geleneksel LLM'ler her kelimeyi (token) sabit bir matris çarpımıyla üretir ($O(1)$ süre). Beynimizin "2+2=4" derken otomatik cevap vermesi gibidir. Ancak *"Sopa ve top $1.10, sopa toptan $1 pahalıysa top kaç paradır?"* diye sorduğunuzda System 1 ezberle **"10 cent"** der (YANLIŞ!).
- **System 2 (Yavaş / Derin Akıl Yürütme):**
  Model doğrudan cevap vermek yerine içsel düşünme tokenleri üretir ($\langle \text{think} \rangle \dots \langle /\text{think} \rangle$). Soruyu cebirsel denklemlere döker ($S + T = 1.10, S = T + 1.00 \Rightarrow 2T = 0.10 \Rightarrow T = 0.05$), çelişkileri test eder ve doğrulanmış **"5 cent"** cevabını verir (DOĞRU!).
- **Test-Time Compute (Çıkarım Anı Hesaplama Gücü):**
  Modeli daha büyük eğitmek yerine, çıkarım anında modelin daha fazla düşünmesine (Thinking Budget) izin vererek matematik, mantık ve kodlama başarısını logaritmik olarak artırma sanatıdır!

```
               [Kullanıcı Sorusu: x]
                         │
                         ▼
           [Zorluk ve Sezgisellik Ayrımı]
                 ┌───────┴───────┐
                 ▼               ▼
         [System 1 (Hızlı)]   [System 2 (Yavaş)]
          • 0 Düşünme Tokeni   • <think> Adımları
          • Yüzeysel Çıkarım   • Denklem & Sağlama
          • Gecikme: ~12ms     • Düşünme Bütçesi: N
                 │               │
                 ▼               ▼
          [Tuzak Yanıt: $0.10] [Doğrulandı: $0.05]
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Çift Süreçli Bilişsel Teori (Dual-Process Theory in LLMs)
- System 1 tek adımlı koşullu olasılık $P(y_t \mid y_{<t}, x)$ ile doğrudan çıktı üretirken; System 2 ara akıl yürütme durumları $s_1, s_2, \dots, s_k$ üzerinden arama uzayını genişletir.

### B. Test-Time Compute Ölçekleme Yasası (Test-Time Scaling Law)
- Modelin doğruluğu çıkarım anında harcanan düşünme tokenleri $N_{\text{think}}$ ile logaritmik orantılı olarak artar:
  $$\text{Accuracy}(x, N_{\text{think}}) = \alpha \cdot \log(N_{\text{think}}) + \beta$$

### C. Ara Adım Doğrulama ve Çelişki Denetimi (Consistency Check)
- Her akıl yürütme adımı $s_k$, bir önceki durumla ve problem kısıtlarıyla tutarlılık testine tabi tutulur. Çelişki durumunda geriye doğru zincirleme (Backward Chaining) ile öz-düzeltme tetiklenir.

### D. Bilişsel Yansıma Testi (CRT) Kıyaslama Sonuçları
- **System 1:** Doğruluk **%0.0**, Ortalama Gecikme 12.00 ms, Düşünme Tokeni 0.
- **System 2:** Doğruluk **%100.0**, Ortalama Gecikme 40.02 ms, Düşünme Tokeni 200 ($N=4$ adım).

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **System 1 Thinking** | Bilişsel çaba harcamadan hızlı, sezgisel ve otoregresif çıkarım yapan düşünme modu. |
| **System 2 Thinking** | Düşünme bütçesi harcayarak adım adım mantıksal planlama ve doğrulama yapan mod. |
| **Test-Time Compute** | Modelin çıkarım anında daha uzun düşünmesine izin vererek performans artırma yaklaşımı. |
| **Thinking Budget ($N_{\text{think}}$)** | Modele akıl yürütme için tahsis edilen maksimum ara düşünme tokeni veya adım sayısı. |
| **Thinking Trace ($\langle \text{think} \rangle$)** | Modelin nihai cevaptan önce ürettiği şeffaf içsel akıl yürütme metni. |
| **Cognitive Reflection Test (CRT)** | Sezgisel ilk cevabın yanlış olduğu bilişsel tuzak sorularından oluşan test. |
| **Self-Correction** | Modelin kendi ürettiği ara adımlardaki hataları tespit edip düzeltmesi süreci. |
| **Backward Chaining** | Hedef durumdan geriye doğru adımları izleyerek kanıtlama ve doğrulama yapma tekniği. |
| **Verification Gate** | Ara adımların mantıksal geçerliliğini denetleyen eşik filtre mekanizması. |
| **Latency-Accuracy Trade-off** | Düşünme süresinin (gecikmenin) artması karşılığında doğruluk oranının yükselmesi dengesi. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Karmaşık CRT tuzaklarında %100     │ • Artan düşünme tokenleri sebebiyle  │
 │   matematiksel ve mantıksal doğruluk.│   daha yüksek çıkarım gecikmesi (40ms│
 │ • Şeffaf ve denetlenebilir <think>   │ • Basit sorularda aşırı düşünme      │
 │   akıl yürütme adımları.             │   (Overthinking) kaynak israfı riski.│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Matematik olimpiyatları, kodlama,  │ • Hatalı bir ara adımın sonraki tüm  │
 │   hukuki akıl yürütme ve bilimsel ML.│   adımları saptırması zincir riski.  │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/system1_vs_system2_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
