# Day 150: Sembolik Akıl Yürütme: LLM ile Z3 SMT Solver & SymPy Entegrasyonu (FAZ 8 YARI-YOL FİNALİ)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.14%2B-brightgreen.svg?style=flat-square)](https://www.sympy.org/)
[![Z3-Solver](https://img.shields.io/badge/Z3--Solver-4.16%2B-orange.svg?style=flat-square)](https://github.com/Z3Prover/z3)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 8: Derin Akıl Yürütme (Reasoning LLMs)** modülünün 10 günlük yarı-yol dönüm noktası (Mid-Way Milestone) olarak, büyük dil modellerinin (LLM) olasılıksal sınırlarını aşmasını sağlayan **Sembolik Akıl Yürütme (Neuro-Symbolic Reasoning)**, **Z3 SMT (Satisfiability Modulo Theories) Çözücü**, **SymPy Cebirsel ve Analitik Motor** ve **Deterministik Teorem İspatı** mimarisini sıfırdan uygulamaktadır.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Neuro-Symbolic AI Nedir ve Neden LLM'leri Z3/SymPy ile Birleştiriyoruz?
- **Saf LLM'lerin Zayıflığı (Olasılıksal Halüsinasyon):**
  Bir LLM ne kadar büyük olursa olsun ($70\text{B}, 405\text{B}$), bir sonraki token'ı olasılık dağılımından ($P(w_t | w_{<t})$) tahmin eder. Karmaşık cebirsel sadeleştirmelerde, modüler kongrüanslarda veya mantık bulmacalarında (SAT) küçük işlem hataları yapar.
- **Neuro-Symbolic AI Gücü (LLM Sezgisi + Deterministik Kesinlik):**
  1. **LLM (Doğal Dil Anlayışı / Semantic Parser):** Kullanıcının serbest metinli problemini anlar ve bunu matematiksel/mantıksal kısıtlara çevirir.
  2. **SymPy / Z3 (Sembolik Motorlar):** Bu kısıtları alır ve arkasındaki deterministik algoritmalarla (Gröbner temelleri, Simplex, DPLL(T)) çözerek **%100 matematiksel kesinlikle** cevap üretir.

```
         NEURO-SYMBOLIC HYBRID ARCHITECTURE
  [Kullanıcı: Doğal Dil Problemi]
          │
          ▼
  [LLM (Semantic Parser)]: Kısıtları Sembolleştirir
          │
          ├──► [SymPy Engine] : Analitik Türev, Cebir, Kök
          │
          └──► [Z3 SMT Solver]: SAT/UNSAT Kısıt Sağlama
          │
          ▼
  [Deterministik Kanıt & %100 Doğru Çözüm]
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Z3 SMT Çözücü ve DPLL(T) Algoritması
- Z3, Birinci Dereceden Mantık (First-Order Logic) üzerindeki kısıtları çözer:
  $$\Phi = (x + y = 15) \land (x \cdot y = 56) \land (x > 0) \land (y > 0)$$
  - Model kontrolü: $\text{Check}(\Phi) \implies \text{SAT} \implies \mathcal{M} = \{x \mapsto 8, y \mapsto 7\}$.

### B. SymPy ile Sembolik Cebir ve Analitik Hesaplama
- Sayısal yaklaşık değerler (Floating-point) yerine sembolik kesin kesirler ve ifadeler kullanılır:
  $$\frac{d}{dx} \left(x^3 \sin(x)\right) = x^3 \cos(x) + 3x^2 \sin(x), \quad \sin^2(x) + \cos^2(x) \equiv 1$$

### C. Neuro-Symbolic İki Yönlü Çeviri Köprüsü
- Doğal dil $\leftrightarrow$ Z3 SMT-LIB / Python AST çift yönlü güvenli çevirisi.

### D. Kesinlik Karşılaştırması ve Sıfır Halüsinasyon Prensibi
- Saf LLM (%45-%72) vs Neuro-Symbolic (%100) kesinlik farkı.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Neuro-Symbolic AI** | Derin öğrenmenin sezgisel örüntü tanıma gücü ile sembolik mantığın kesinliğini birleştiren paradigma. |
| **Z3 SMT Solver** | Microsoft Research tarafından geliştirilen son teknoloji kısıt sağlama (SMT) motoru. |
| **SymPy** | Python için sembolik matematik, cebir, kalkülüs ve denklem çözme kütüphanesi. |
| **SAT / UNSAT** | Bir mantıksal formülün doğru kılınabilir (Satisfiable) veya çelişkili (Unsatisfiable) olma durumu. |
| **First-Order Logic (FOL)** | Değişkenler, fonksiyonlar ve niceleyiciler ($\forall, \exists$) içeren biçimsel mantık sistemi. |
| **Semantic Parsing** | Doğal dildeki kullanıcı metnini çalıştırılabilir sembolik kod veya mantık kuralına dönüştürme. |
| **DPLL(T)** | SMT çözücülerinde teori kısıtlarını doğrulamak için kullanılan çekirdek algoritma. |
| **Formal Proof** | Her adımı matematiksel ve mantıksal kurallarla kanıtlanmış biçimsel ispat. |
| **Symbolic Differentiation** | Sayısal türev yerine tam analitik cebirsel türev fonksiyonunu üretme. |
| **Linear Congruence** | $ax \equiv b \pmod m$ biçimindeki modüler aritmetik denklemleri. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %100 deterministik matematik ve    │ • NP-Zor kısıt problemlerinde        │
 │   mantık doğruluğu (Sıfır Hata).     │   SMT çözücünün zaman aşımı riski.   │
 │ • LLM halüsinasyonlarını kökten yok  │ • Doğal dilden sembolik koda çeviri  │
 │   eden resmi kanıt mekanizması.      │   aşamasında sözdizimi hataları.     │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Havacılık, savunma, kriptografi    │ • İnsan dilinin belirsiz/muğlak      │
 │   ve finansal sözleşmelerin biçimsel │   ifadelerini biçimsel mantığa       │
 │   doğrulanması (Formal Verification).│   dönüştürmenin zorluğu.             │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/symbolic_math_z3_sympy_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
