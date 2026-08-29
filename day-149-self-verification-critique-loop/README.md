# Day 149: Kendi Kendine Doğrulama (Self-Verification) ve İkili Eleştiri Döngüsü (Actor-Critic)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; Reasoning LLM'lerin (OpenAI o1, DeepSeek-R1, QwQ) bir çözümü kullanıcıya teslim etmeden önce kendi mantığını ve hesaplamalarını tersine sağlamasını yapan **Kendi Kendine Doğrulama (Self-Verification)**, **Çözümden Girdiye Ters Sağlama (Reverse Verification / Back-Substitution)**, **İkili Eleştirmen Döngüsü (Actor-Critic Refinement)** ve **Kesinlik Kalibrasyonu** mimarisini sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ İnsanlar ve Reasoning Modelleri Neden "Ters Sağlama" Yapar?
- **Standart LLM (Tek Geçişli - Güvensiz):**
  Bir model cebirsel bir denklem çözdüğünde ($3x + 7 \equiv 2 \pmod 5$), ilk adımda bulduğu $x = 2$ sonucunu hiç denetlemeden kullanıcıya sunar. Ancak $3(2) + 7 = 13 \equiv 3 \pmod 5 \ne 2$ olduğu için sonuç tamamen yanlıştır.
- **Reasoning LLM (Actor-Critic & Self-Verification):**
  Çözücü model (Aktör) bir aday cevap ürettiğinde, bağımsız bir Doğrulayıcı model (Eleştirmen) bu cevabı alır ve orijinal denkleme koyarak ters sağlama (**Back-Substitution**) yapar:
  - $x = 2 \implies 13 \pmod 5 = 3 \ne 2$ $\to$ `[REDDEDİLDİ]`
  - Eleştirmen Aktör'e geri bildirim gönderir: `"Ters sağlama başarısız! 3x = 0 mod 5 olmalıdır."`
  - Aktör çözümü rafine eder: $x = 0 \implies 3(0) + 7 = 7 \equiv 2 \pmod 5$ $\to$ `[ONAYLANDI]`!

```
       SELF-VERIFICATION & CRITIQUE ARCHITECTURE
           [Problem: 3x + 7 = 2 (mod 5)]
                         │
                         ▼
     ┌───────────────────────────────────────┐
     │ 1. ACTOR (Generator): İlk Çözüm Üret  │
     │    x = 2 (Aday Çözüm)                 │
     └───────────────────┬───────────────────┘
                         ▼
     ┌───────────────────────────────────────┐
     │ 2. CRITIC (Verifier): Ters Sağlama Yap│
     │    3*(2) + 7 = 13 mod 5 = 3 != 2      │
     │    ✖ REDDEDİLDİ!                      │
     └───────────────────┬───────────────────┘
                         ▼
     ┌───────────────────────────────────────┐
     │ 3. REFINEMENT: Eleştiri ile Düzeltme  │
     │    3x = 0 mod 5 => x = 0 (Yeni Aday)  │
     └───────────────────┬───────────────────┘
                         ▼
     ┌───────────────────────────────────────┐
     │ 4. FINAL VERIFICATION: 3*(0)+7=7=2✔   │
     └───────────────────────────────────────┘
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Ters Sağlama (Back-Substitution)
- Verilen $f(x) = y$ problemi için bulunan aday çözüm $\hat{x}$:
  $$\text{Doğrulama}(\hat{x}) = \begin{cases} 1, & f(\hat{x}) == y \\ 0, & f(\hat{x}) \neq y \end{cases}$$
- Çözüm uzayını taramak zor olsa da, bir adayın doğruluğunu tersine koyup hesaplamak deterministik ve hızlıdır ($O(1)$).

### B. Actor-Critic İkili Rol Ayrımı (Generator vs Verifier Asymmetry)
- Doğrulamayı yapan Eleştirmen (Critic), çözümü üreten Aktör'den farklı bir prompt veya bağımsız bir doğrulayıcı kafa (PRM) kullanarak önyargıları (Confirmation Bias) sıfırlar.

### C. Doğrulama Tabanlı Rafinasyon Döngüsü (Generate-Verify-Refine)
- Eleştirmenin ürettiği hata mesajı doğrudan Aktör'e bir düzeltme talimatı (Negative Feedback) olarak iletilir.

### D. Kesinlik ve Güven Kalibrasyonu (Certainty Calibration)
- Model ancak ters sağlama başarılı olduğunda güven skorunu $1.0$'a çıkararak halüsinasyon riskini ortadan kaldırır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Self-Verification** | Modelin kendi ürettiği çözümü dışarı sunmadan önce bağımsız olarak denetlemesi. |
| **Back-Substitution** | Bulunan matematiksel sonucun orijinal denklemde yerine konularak eşitliğin sağlanması. |
| **Actor-Critic Loop** | Çözümü üreten (Aktör) ve denetleyen (Eleştirmen) iki mekanizmanın döngüsel etkileşimi. |
| **Verification Asymmetry** | Bir problemi çözmenin çok zor, ancak verilen çözümü doğrulamanın çok kolay olması durumu. |
| **Certainty Calibration** | Modelin iddia ettiği güven skoru ile gerçek matematiksel doğruluk arasındaki uyum. |
| **Negative Feedback** | Eleştirmenin hatalı adım hakkında sunduğu spesifik düzeltme gerekçesi. |
| **Double-Check Protocol** | Kritik görevlerde nihai çıktı öncesi zorunlu ters sağlama adımı. |
| **Confirmation Bias** | Modelin kendi yaptığı hatayı tekrar okuduğunda doğru sanma bilişsel yanılsaması. |
| **Iterative Refinement** | Hata raporuna göre çözümü adım adım iyileştirme süreci. |
| **Deterministic Verifier** | Sembolik kurallara veya ters sağlamaya dayalı kesin denetçi. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Halüsinasyon ve işlem hatalarını   │ • Ek doğrulama adımları sebebiyle    │
 │   %98.5 oranında filtreleme gücü.    │   çıkarım süresinde (TTFT) artış.    │
 │ • Ters sağlama ile kesin matematiksel│ • Eleştirmen model için ek hesaplama │
 │   kanıt üretme kabiliyeti.           │   ve bellek gereksinimi.             │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Hukuk, tıp ve finans gibi sıfır    │ • Doğrulanması çözülmesi kadar zor   │
 │   hata toleranslı alanlarda LLM      │   olan açık uçlu felsefi sorularda   │
 │   güvenilirliğini sağlama.           │   uygulama sınırları.                │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/self_verification_critique_loop_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
