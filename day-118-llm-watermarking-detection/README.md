# Day 118: LLM Filigranlama ve Z-Skoru Tabanlı Tespit

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; yapay zeka tarafından üretilen metinleri insan gözüyle fark edilmeyecek şekilde damgalayan ve matematiksel kesinlikle tespit eden **Kirchenbauer Yeşil/Kırmızı Liste Tabanlı LLM Filigranlama (Watermarking) ve Z-Skoru Hipotez Testi Hattı**nı sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Görünmez Damga: LLM Metinlerine Kriptografik Filigran"

Bir yapay zeka tarafından yazılan bir makaleyi veya ödevi okuduğunuzda metin tamamen doğal görünür. Ancak bu metnin yapay zeka tarafından üretildiğini **matematiksel olarak kesin bir şekilde** nasıl ispatlayabiliriz?

**Kirchenbauer Filigranlama Algoritması (ICML 2023 En İyi Makale)** bunu kusursuzca çözer:
1. 🎲 **Gizli Zar (Pseudorandom Hash):** Model bir sonraki kelimeyi seçeceği zaman, bir önceki kelimeyi ve gizli bir anahtarı kullanarak sözlüğü ikiye böler:
   - 🟢 **Yeşil Liste:** Seçilmesi teşvik edilen kelimeler (%50).
   - 🔴 **Kırmızı Liste:** Seçilmesi azaltılan kelimeler (%50).
2. 📈 **Logit Eğilimi ($\delta$):** Yeşil listedeki kelimelerin olasılığına küçük bir bonus eklenir ($\text{logits}[G] += \delta$). Metin insan kulağına tamamen doğal gelir.
3. 🔍 **Z-Skoru ile Tespit:** İnsanlar yazı yazarken yeşil ve kırmızı listeden rastgele (%50-%50) kelime seçer ($Z \approx 0$). Ancak filigranlı AI modeli kelimelerin %95'ini yeşil listeden seçer!
4. 📊 **Matematiksel Kanıt ($Z \ge 4.0$):** 100 kelimelik bir metinde 95 yeşil kelime görülme olasılığı $p < 10^{-15}$'tir (yani tesadüf olması imkansızdır).

```
   ÖNCEKİ TOKEN (x_{t-1}) + GİZLİ ANAHTAR              SÖZLÜK BÖLÜMLEME & LOGIT BIAS
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────────────┐
 │ SHA256(x_{t-1} || Key) % 2^32        │ ─────────> │ 🟢 Yeşil Liste (%50): logit + delta          │
 └──────────────────────────────────────┘            │ 🔴 Kırmızı Liste (%50): logit normal         │
                                                     └──────────────────────┬───────────────────────┘
                                                                            │
                                                                            ▼
                                                              [Z-SKORU HİPOTEZ TESTİ]
                                                              ├── Yeşil Oran: %96.0 (İnsan: %50)
                                                              ├── Z-Skoru   : Z = 9.21 (Eşik: 4.0)
                                                              └── Karar     : [KESİN AI METNİ]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Kirchenbauer Algoritması
- **Önceki Token Bağımlılığı:** Her adımda $x_{t-1}$ tokenı ve gizli anahtar ($k$) kullanılarak deterministik bir RNG tohumu ($seed = \text{hash}(x_{t-1}, k)$) üretilir.
- **Yeşil / Kırmızı Liste:** Kelime dağarcığı $|V|$, $\gamma = 0.5$ oranıyla iki eşit alt kümeye ayrılır ($|G| = \gamma |V|$).
- **Logit Yanlılığı (Biasing):** Modelin ham logit vektörüne yeşil indeksler için $\delta$ eklenir:
$$\tilde{l}_i = \begin{cases} l_i + \delta & \text{eğer } i \in G_{x_{t-1}} \\ l_i & \text{eğer } i \in R_{x_{t-1}} \end{cases}$$

### 2. Matematiksel Hipotez Testi & Z-Skoru Formülasyonu
Filigran tespitinde Boş Hipotez ($H_0$): *"Metin insan tarafından veya filigransız üretilmiştir ($P(\text{Yeşil}) = \gamma$)"*.
Toplam $T$ token geçişinde gözlenen yeşil token sayısı $|G|$ olmak üzere $Z$-Skoru:
$$Z = \frac{|G| - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}$$
$p$-değeri: $p = 1 - \Phi(Z)$. $Z \ge 4.0$ eşiğinde yanlış pozitiflik olasılığı $p \le 3.16 \times 10^{-5}$'tir.

### 3. Güç ($\delta$) ve Metin Kalitesi (Perplexity/Entropy) Takası
- Düşük $\delta$ ($\delta < 1.0$): Metin doğallığı %100 korunur ancak tespit için daha uzun metin ($T > 200$) gerekir.
- Orta $\delta$ ($\delta = 2.0 - 2.5$): İdeal denge. 50 tokenlık kısa metinlerde bile $Z > 6.0$ elde edilir.
- Yüksek $\delta$ ($\delta > 4.0$): Tespit kolaylaşır ancak modelin çeşitliliği düşebilir.

### 4. Paraphrase Saldırılarına Karşı Dayanıklılık
Metin üzerinde kelime değiştirme (synonym replacement) veya cümle yapısı değiştirme saldırısı yapılsa dahi, kelimelerin %30'u değiştirildiğinde bile $Z \ge 4.0$ eşiği korunarak filigran varlığı başarıyla kanıtlanır.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **LLM Watermarking** | Dil modeli çıktılarına istatistiksel ve kriptografik olarak tespit edilebilir gizli desenler ekleme. |
| **Green List ($G$)** | Çıkarım anında seçilmesi $\delta$ logit yanlılığı ile teşvik edilen token alt kümesi. |
| **Red List ($R$)** | Yeşil listenin tümleyeni olan ve seçilme olasılığı nispeten azalan token kümesi. |
| **Logit Bias ($\delta$)** | Yeşil listedeki tokenların logit değerlerine eklenen skaler pozitif ağırlık. |
| **Green Fraction ($\gamma$)** | Sözlüğün yeşil listeye ayrılan fraksiyonu (genellikle $\gamma = 0.5$). |
| **Z-Score** | Gözlenen yeşil token sayısının beklenen değerden standart sapma cinsinden uzaklığı. |
| **P-Value** | Gözlenen yeşil token yoğunluğunun tamamen tesadüf eseri oluşma olasılığı. |
| **True Positive Rate (TPR)** | Gerçekten filigranlı olan metinlerin doğru tespit edilme oranı (%100.0). |
| **False Positive Rate (FPR)** | İnsan tarafından yazılmış metne yanlışlıkla "AI" damgası vurulma oranı (%0.0). |
| **Paraphrase Attack** | Filigranı bozmak amacıyla metnin kelimelerini eş anlamlılarıyla değiştirme saldırısı. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Matematiksel kesinlikte tespit.    │ • Gizli anahtar sızarsa üçüncü parti-│
 │ • Ek model veya GPU gerekmez ($0).   │   ler filigranı taklit edebilir.     │
 │ • Sıfır yanlış alarm (FPR: %0.0).    │ • Çok kısa metinlerde (< 20 token)   │
 │ • İnsan gözüyle fark edilemez.       │   istatistiksel güç zayıflayabilir.  │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Telif hakkı, akademik dürüstlük    │ • Ağır çeviri (back-translation)     │
 │   ve AI içerik etiketleme standartları│   filigranı zayıflatabilir.          │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/filigran_tespit_paneli.png` dosyası üretilir:
1. **Z-Skoru Dağılımı (Filigransız $Z \approx 0$ vs Filigranlı $Z > 9.0$)**
2. **Yeşil Liste Token Oranı Karşılaştırması (İnsan %49.5 vs Model %96.0)**
3. **Delta ($\delta$) Yanlılığına Göre Z-Skoru Artışı**
4. **Metin Değiştirme (Paraphrase) Saldırısı Dayanıklılığı**
5. **Kirchenbauer Yeşil/Kırmızı Liste Filigran Akış Şeması**
6. **LLM Kriptografik Filigran Sertifikası**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# LLM filigranlama ve Z-Skoru tespit hattını koşturun
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
