# Day 153: Tümdengelimsel Mantık Doğrulayıcı & Safsata Dedektörü (Logical Fallacy & Deductive Engine)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; LLM'lerin ve akıl yürüten ajanların ürettikleri metinlerdeki argümanları öncül ve sonuçlara ayrıştıran, **Tümdengelimsel Geçerlilik (Deductive Validity)** ve **Sağlamlık (Soundness)** denetimi yapan, hem **Biçimsel Mantıksal Safsataları (Affirming the Consequent, Denying the Antecedent)** hem de **Biçimsel Olmayan Bilişsel Safsataları (Ad Hominem, Straw Man, False Dilemma, Circular Reasoning)** tespit eden sıfırdan inşa edilmiş bir mantık motorudur.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Geçerli Argüman (Valid)" ile "Sağlam Argüman (Sound)" Arasındaki Fark Nedir?
- **Mantıksal Geçerlilik (Validity):**
  Bir argümanın biçimsel yapısıyla ilgilidir. *"Eğer öncüller doğru olsaydı, sonucun yanlış olması imkansız olurdu"* kuralını inceler.
  - Örnek: *"Tüm köpekler 6 bacaklıdır. Karabaş bir köpektir. O halde Karabaş 6 bacaklıdır."* $\to$ **GEÇERLİDİR (Valid)** çünkü biçimsel akıl yürütme kusursuzdur; ancak öncül gerçek dünyada yanlış olduğu için **SAĞLAM DEĞİLDİR (Unsound)**.
- **Mantıksal Sağlamlık (Soundness):**
  $$\text{Sağlamlık (Soundness)} = \text{Biçimsel Geçerlilik (Validity)} \land \text{Tüm Öncüllerin Gerçekte Doğru Olması}$$
  - Örnek: *"Tüm insanlar ölümlüdür. Sokrates bir insandır. O halde Sokrates ölümlüdür."* $\to$ **HEM GEÇERLİ HEM SAĞLAMDIR (%100 Güvenilir)**.

```
       DEDUCTIVE & FALLACY ENGINE ARCHITECTURE
  [1. Doğal Dil Argümanı Girdisi]
                 │
                 ▼
  [2. Öncül-Sonuç Ayrıştırıcı (Premise-Conclusion)]
    Öncül 1: P ⟹ Q | Öncül 2: P | Sonuç: Q
                 │
        ┌────────┴────────┐
        ▼                 ▼
  [3. Biçimsel Mantık]  [4. Safsata Dedektörü]
     (Modus Ponens /     (Ad Hominem, Straw Man,
      Modus Tollens)      Affirming Consequent)
        │                 │
        └────────┬────────┘
                 ▼
  [5. Sağlamlık (Soundness) & Güven Skoru Raporu]
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Modus Ponens vs Sonucun Doğrulanması (Affirming Consequent)
- **Modus Ponens (Geçerli Tümdengelim):**
  $$P \implies Q, \quad P \quad \vdash \quad Q$$
- **Sonucun Doğrulanması (Biçimsel Safsata / Invalid):**
  $$P \implies Q, \quad Q \quad \vdash \quad P \quad (\text{Yanlış! Yerler ıslak diye kesin yağmur yağmış olamaz, arazöz yıkamış olabilir})$$

### B. Biçimsel Olmayan Safsataların Anlamsal Ayrışımı
- **Ad Hominem:** Argüman yerine kişiye saldırı ($X \text{ kötüdür} \vdash P \text{ yanlıştır}$).
- **Straw Man (Korkuluk):** Argümanı karikatürize edip çürütme ($P \to P^* \vdash \neg P$).
- **False Dilemma (Yanlış İkilem):** Ara seçenekleri yok sayma ($P \lor Q$).

### C. Öncül-Sonuç Ayrıştırma Grameri
- Bağlaç analizi (`"dolayısıyla"`, `"o halde"`, `"bu yüzden"`, `"çünkü"`) ile öncüller $\mathcal{P} = \{p_1, \dots, p_n\}$ ve varılan sonuç $c$ yapılandırılır.

### D. LLM Halüsinasyon ve Mantık Doğrulama Entegrasyonu
- Üretilen düşünce zincirlerinin (`<think> ... </think>`) safsatadan arındırılması ve sağlam sonuç garantisi.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Deductive Validity** | Öncüller doğru olduğunda sonucun zorunlu olarak doğru olması durumu. |
| **Soundness** | Bir argümanın hem biçimsel olarak geçerli hem de tüm öncüllerinin fiilen doğru olması. |
| **Syllogism (Kıyas)** | İki öncülden zorunlu bir mantıksal sonuç çıkaran klasik Aristo mantık yapısı. |
| **Modus Ponens** | $P \implies Q$ ve $P$ verildiğinde $Q$ sonucuna varan geçerli tümdengelim kuralı. |
| **Affirming Consequent** | Koşullu önermede sonucun doğrulanmasından şartın zorunlu çıkarılamayacağı biçimsel safsata. |
| **Ad Hominem** | Argümanın içeriği yerine iddiayı savunan kişinin şahsına yapılan geçersiz saldırı. |
| **Straw Man** | Karşı tarafın fikrini aşırı basitleştirip veya abartıp kolayca çürütme safsatası. |
| **False Dilemma** | Birden çok seçenek varken yalnızca iki zıt seçenek varmış gibi sunulan safsata. |
| **Circular Reasoning** | Kanıtlanmak istenen sonucun öncüllerin içinde zaten doğru varsayılması (Petitio Principii). |
| **Premise (Öncül)** | Bir argümanda sonuca ulaşmak için ileri sürülen temel iddia veya varsayım. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Argümanları matematiksel ve felsefi│ • Çok karmaşık ve örtük metaforik    │
 │   kesinlikle denetleme yeteneği.     │   safsataların kural dışı kalması.   │
 │ • LLM'lerdeki mantık hatalarını      │ • Gerçek dünya olgusal doğruluk      │
 │   hızlıca filtreleme ve puanlama.    │   veritabanı gereksinimi.            │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Hukuk, finans, akademik hakemlik ve│ • Karmaşık dillerde anlamsal         │
 │   otonom müzakere ajanları.          │   çift anlamlılık (Equivocation)     │
 │                                      │   tuzakları.                         │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/logical_fallacy_deductive_engine_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
