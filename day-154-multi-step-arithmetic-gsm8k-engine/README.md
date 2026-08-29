# Day 154: GSM8K & MATH Çok Adımlı Matematiksel Akıl Yürütme Motoru (Program-Aided Language Models - PAL)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; ilkokuldan olimpiyat seviyesine kadar çok adımlı sözel matematik problemlerini çözmek için geliştirilen **GSM8K & MATH Benchmark** standartlarını, **Program-Aided Language Models (PAL)**, **Program of Thoughts (PoT)** ve **İzole Python Aritmetik Yürütücüsü** mimarisini sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Neden LLM'ler Zihinsel Aritmetikte Hata Yapar ve PAL Neden %100 Çözümdür?
- **Zihinsel Aritmetik (Raw CoT / Mental Math) Zayıflığı:**
  Büyük Dil Modelleri (LLMs) özünde birer olasılıksal sonraki-token tahmincileridir. Bir problemi çözerken metin içinde $247 \times 38$ veya karmaşık kesirli adımlar ($15 - 6 = 9 \implies 9 / 2$) yaparken token düzeyinde basamak taşımayı karıştırabilir ve halüsinasyon görebilirler.
- **Program-Aided Language Models (PAL & PoT Paradigması):**
  LLM'in en güçlü olduğu alan **mantıksal akıl yürütme ve problem anlama**, bilgisayarın en güçlü olduğu alan ise **aritmetik ve hesaplamadır**.
  - PAL paradigmasında model doğrudan cevabı hesaplamaz; problemi adım adım değişkenlere sahip temiz bir **Python fonksiyonuna (`def solution(): ...`)** dönüştürür.
  - Bu fonksiyon güvenli bir Python yorumlayıcısında çalıştırılarak **%100 matematiksel kesinlikle** sonuç alınır!

```
        PROGRAM-AIDED LANGUAGE MODEL PIPELINE
  [1. GSM8K Sözel Matematik Problemi Girdisi]
                       │
                       ▼
  [2. LLM Code Generator (PoT Parser)]
    def solution():
        toplam = 15 - (3 * 2)
        kalan = toplam / 2
        return kalan
                       │
                       ▼
  [3. Sandboxed Python Interpreter]: Kodu Çalıştırır
                       │
                       ▼
  [4. Kesin ve Hatasız Aritmetik Sonuç: 4.5]
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Program-Aided Language Models (PAL) ve PoT
- Bir matematik problemi $\mathcal{Q}$ için amaç doğrudan nihai cevap $y$'yi üretmek değil, $y = \text{Exec}(\mathcal{C}_{\text{code}})$ olacak şekilde program $\mathcal{C}$ üretmektir:
  $$P(y | \mathcal{Q}) = \sum_{\mathcal{C}} P(\mathcal{C} | \mathcal{Q}) \cdot \mathbb{I}(\text{Exec}(\mathcal{C}) = y)$$

### B. Çok Adımlı Durum Takibi (Step-by-Step State Tracking)
- Her ara değişken (`kalan_1`, `elde_edilen_gelir`, `toplam_mesafe`) adlandırılmış bellek alanı olarak tutulur, böylece LLM'in dikkat (attention) mekanizmasında bilgi kaybı önlenir.

### C. Çoklu Çözüm Yolu ve Çoğunluk Oylaması (Self-Consistency with PAL)
- Farklı değişken adları ve algoritmik yaklaşımlarla $N$ adet farklı Python kodu üretilir ve dönen sayısal sonuçlar çoğunluk oyuyla (Majority Vote) teyit edilir.

### D. Sandboxed İzole Ortamda Yürütme Güvenliği
- Üretilen dinamik kodun yalnızca aritmetik kütüphaneleri kullanması ve sistem kaynaklarına erişememesi için yerel isim alanı (`local_scope`) kısıtlaması uygulanır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **GSM8K** | Grade School Math 8K; 8.500 adet çok adımlı ilkokul düzeyi sözel matematik veri seti. |
| **MATH Benchmark** | Lise ve üniversite matematik olimpiyatı düzeyindeki zorlu matematik problemleri kıyaslaması. |
| **PAL (Program-Aided LM)** | Aritmetik işlemleri bir programlama dili yorumlayıcısına devreden LLM yöntemi. |
| **PoT (Program of Thoughts)** | Düşünce zincirini (CoT) doğal dil yerine çalıştırılabilir kod olarak ifade etme paradigması. |
| **Mental Math (Zihinsel Aritmetik)** | Modelin herhangi bir dış araç kullanmadan doğrudan token olarak sayı hesaplaması (hataya açıktır). |
| **Self-Consistency** | Modelden çoklu çözüm yolları örnekleyip en çok çıkan ortak cevabı seçme yöntemi. |
| **Sandboxed Execution** | Kodun izole, yan etkisiz ve güvenli bir yerel bellek alanında çalıştırılması. |
| **Intermediate State** | Çok adımlı problem çözümlerinde ara adımlarda hesaplanan değişken değerleri. |
| **Fraction / Decimal Precision** | Kesirli ve ondalık sayılarda basamak yuvarlama ve kayan nokta hassasiyeti. |
| **Code-Interpreted Reasoning** | Kod derleyicisinin deterministik gücünü LLM semantik kavrayışıyla birleştiren hibrit mantık. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Aritmetik ve hesaplama hatalarını  │ • Kod üretimi gerektirdiği için      │
 │   %100 sıfırlama garantisi.          │   çıkarım süresinde ufak ek gecikme. │
 │ • Değişken takibi ile karmaşık çok   │ • Yalnızca kodlanabilir matematiksel │
 │   adımlı problemlerde yüksek başarı. │   sorularda uygulanabilirlik.        │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Finansal raporlama, mühendislik    │ • Güvensiz ortamlarda rastgele kod   │
 │   hesaplamaları ve vergi/KDV botları.│   yürütme (Remote Code Execution).   │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/multi_step_arithmetic_gsm8k_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
