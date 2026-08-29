# Day 152: Biçimsel Mantık ve Teorem İspatı (Formal Theorem Proving with Lean 4)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Lean4](https://img.shields.io/badge/Lean-4-blueviolet.svg?style=flat-square)](https://leanprover.github.io/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; DeepMind AlphaProof, OpenAI ve modern yapay zekanın matematiksel olimpiyat (IMO) düzeyinde problem çözmesini sağlayan **Biçimsel Teorem İspatı (Formal Theorem Proving)**, **Lean 4 İnteraktif İspat Asistanı (ITP)**, **Curry-Howard Eşbiçimliliği (Propositions-as-Types)**, **Otomatik Biçimsellendirme (Autoformalization)** ve **Taktik Ağacı Arama Motoru (`induction`, `rfl`, `rw`, `simp`)** mimarisini sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Neden Matematikçiler ve AI Mühendisleri Lean 4 Kullanır?
- **Doğal Dil İspatlarının Zayıflığı (İnformal / Boşluklu İspatlar):**
  İnsanlar veya standart LLM'ler bir teorem ispatlarken metin içinde *"Buradan açıkça görülür ki..."* veya *"Benzer şekilde..."* diyerek kritik adımları atlayabilir ve farkında olmadan yanlış varsayımlara dayanabilir.
- **Biçimsel İspat (Formal Verification & Lean 4):**
  Lean 4'te bir teorem bir **Tip (Type)**, onun ispatı ise o tipi inşa eden bir **Program (Term / Program)** olarak kodlanır (**Curry-Howard Eşbiçimliliği**).
  - İspat adımları **Taktikler (Tactics: `induction`, `rfl`, `rw`)** ile yazılır.
  - Lean 4 çekirdeği (Kernel Type Checker) ispatı harfiyen denetler. Eğer tüm açık hedefler (Goals) kapanırsa (`no goals left`), teorem **%100 matematiksel kesinlikle (Q.E.D.)** kanıtlanmış sayılır!

```
      AUTORMALIZATION & LEAN 4 PROOF PIPELINE
  [Doğal Dil: 'Her n için n + 0 = n olduğunu kanıtla']
                       │
                       ▼
  [LLM Autoformalizer]: Lean 4 Teoremine Çevirir
    theorem add_zero (n : Nat) : n + 0 = n := by ...
                       │
                       ▼
  [Lean 4 Tactic Engine]: Taktikleri Yürütür
    1. induction n  -> Alt hedefler açılır
    2. rfl          -> 0 + 0 = 0 kapandı
    3. rw [hd]      -> succ d + 0 = succ d kapandı
                       │
                       ▼
  [Lean 4 Kernel Type-Checker]: %100 Resmi İspat!
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Curry-Howard Eşbiçimliliği (Propositions-as-Types)
- Biçimsel mantıkta her önerme $\mathcal{P}$ bir veri tipidir.
- Bu önermenin ispatı $p$, o tipten bir nesnedir ($p : \mathcal{P}$).
- İspatı doğrulamak, basitçe derleyicinin tip denetimi yapmasıdır ($\text{TypeCheck}(p) == \mathcal{P}$).

### B. Autoformalization (İnformal Metinden Biçimsel Koda Çeviri)
- LLM, doğal dildeki matematik teoremini Lean 4 sözdizimine ve tip imzasına dönüştürür.

### C. Taktik Tabanlı Hedef Durumu Yönetimi (Proof State Transitions)
- Başlangıç durumu: $\Gamma \vdash n + 0 = n$.
- `induction n` taktiği hedefi ikiye böler:
  1. $\vdash 0 + 0 = 0$ (Taban: `rfl` ile kapanır).
  2. $d : \text{Nat}, hd : d + 0 = d \vdash \text{succ}(d) + 0 = \text{succ}(d)$ (Tümevarım adımı: `rw [hd]` ile kapanır).

### D. Sıfır Halüsinasyonlu İspat Doğrulama
- Lean 4 çekirdeği küçük, güvenilir ve deterministiktir. Yanlış bir ispat adımının derleyiciden geçme olasılığı $0$'dır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Lean 4** | Bağımlı tip teorisine (Dependent Type Theory) dayalı modern interaktif teorem ispat dili. |
| **ITP (Interactive Theorem Prover)** | Kullanıcı veya LLM ile adım adım teorem kanıtlayan yazılım asistanı. |
| **Curry-Howard Isomorphism** | Matematiksel önermeler ile bilgisayar programı tipleri arasındaki derin eşdeğerlik. |
| **Autoformalization** | Doğal dilde yazılmış gayriresmi matematik metinlerini biçimsel ITP koduna otomatik çevirme. |
| **Tactic** | İspat hedefini sadeleştiren, alt hedeflere bölen veya kapatan komut (`induction`, `rfl`, `simp`, `rw`). |
| **Proof State / Goal** | İspatın o anında kanıtlanması gereken açık matematiksel hedef ve hipotezler listesi. |
| **Q.E.D. (Quod Erat Demonstrandum)** | "Böylece kanıtlanmış oldu" anlamına gelen ve ispatın bittiğini belirten terim. |
| **Peano Arithmetic** | Doğal sayıları $0$ ve ardıl fonksiyonu ($\text{succ}$) üzerinden tanımlayan aksiyom sistemi. |
| **Reflexivity (`rfl`)** | Özdeşlik kuralı ($a = a$) ile eşitliğin her iki tarafı denk olduğunda hedefi kapatan taktik. |
| **AlphaProof** | Google DeepMind'ın IMO yarışmasında gümüş madalya kazanan Lean tabanlı formal akıl yürütme yapay zekası. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %100 matematiksel kesinlik ve      │ • Lean 4 sözdizimi ve taktik         │
 │   resmi ispat güvencesi.             │   uzayının çok dik öğrenme eğrisi.   │
 │ • LLM halüsinasyonlarını derleyici   │ • Taktik arama uzayının büyük        │
 │   düzeyinde sıfırlama.               │   kombinatorik patlama riski.        │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Çip tasarımı, havacılık yazılımları│ • Dünyadaki informal matematiğin     │
 │   ve akıllı sözleşmelerin biçimsel   │   henüz çok küçük bir kısmının       │
 │   doğrulanması (Formal Verification).│   Lean Mathlib'e aktarılmış olması.  │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/formal_theorem_proving_lean4_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
