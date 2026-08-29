# Day 151: Test Odaklı Kod Üretimi (TDD Code-Gen Loop & Self-Repair)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![PyTest](https://img.shields.io/badge/PyTest-7.3%2B-brightgreen.svg?style=flat-square)](https://docs.pytest.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; SWE-bench, Devin, Claude Code ve modern kodlama ajanlarının temelini oluşturan **Test Odaklı Geliştirme (TDD - Test-Driven Development)**, **İzole Kod Yürütme (Sandboxed Execution)**, **Hata Yığını Ayrıştırma (Stack Trace / Traceback Parsing)** ve **Kendi Kendine Hata Ayıklama & Yamalama (Iterative Self-Repair Loop)** mimarisini sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Neden LLM'lerin Kod Üretiminde "TDD Döngüsü" Hayati Önem Taşır?
- **Klasik Kod Üretimi (Tek Geçişli - Kırılgan):**
  Bir LLM'den bir fonksiyon yazması istendiğinde, model genellikle ana mantığı doğru kurar ancak **sınır durumlarını (Edge Cases: boş dize `""`, tek eleman, `None`, büyük sayılar)** gözden kaçırır (`IndexError`, `KeyError` veya eksik tampon boşaltma). Kod test edilmeden teslim edilirse sistem çöker.
- **TDD Tabanlı Kod Ajanı (Generate $\to$ Test $\to$ Traceback $\to$ Repair):**
  1. **Kod Üretimi:** Model ilk taslak kodu yazar.
  2. **İzole Test Koşumu:** Yazılan kod otomatik olarak birim testlerle (PyTest) çalıştırılır.
  3. **Hata Yakalama:** Başarısız olan testlerin `Traceback` ve `AssertionError` çıktıları terminalden yakalanır.
  4. **Otomatik Onarım:** Model bu hata çıktısını analiz eder (`<think> Traceback gösteriyor ki boş stringte IndexError alınıyor... </think>`) ve kodu yamalayarak tüm testler yeşil (%100 PASS) olana kadar döngüyü sürdürür.

```
       TEST-DRIVEN CODE GENERATION ARCHITECTURE
  [1. Görev & Test Belirtimi (Problem Specs)]
                 │
                 ▼
  [2. LLM Code Generator]: İlk Taslak Kodu Üretir
                 │
                 ▼
  [3. Sandboxed PyTest Runner]: Testleri Koşar
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
  [FAIL: Traceback]   [PASS: %100 Başarı]
       │                   └──► Teslim Et!
       ▼
  [4. LLM Debugger / Self-Repair]: Kodu Yamalar
       └──► Döngüyü Başa Sar (Iterative Loop)
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: TDD Doğrulama ve İteratif Onarım Döngüsü
- Görev tanımı $\mathcal{P}$ ve test kümesi $\mathcal{T} = \{t_1, t_2, \dots, t_m\}$ için $k$. turdaki kod $C_k$:
  $$\text{Sonuç}_k = \text{Execute}(C_k, \mathcal{T}) \implies (\text{PassRatio}, \text{Traceback}_k)$$
  - Eğer $\text{PassRatio} < 1.0$ ise: $C_{k+1} = \text{Repair}(C_k, \text{Traceback}_k)$.

### B. Hata Yığını (Stack Trace / Traceback) Bilgi Yoğunluğu
- Hata yığını, modelin nereye odaklanması gerektiğini tam satır numarası ve istisna türüyle (`IndexError: string index out of range`) belirterek arama uzayını $O(1)$ seviyesine daraltır.

### C. Sınır Durumu (Edge Case) Dayanıklılığı
- Boş string `""`, tekil eleman `"A"`, tekrarsız dizi `"ABCD"` gibi uç noktalar birim testlerle güvence altına alınır.

### D. Sandboxed İzole Ortam Güvenliği
- Üretilen dinamik kodun sistem dosyalarına zarar vermemesi için güvenli isim alanı (`local_namespace` / AST doğrulaması) ile çalıştırılması.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **TDD (Test-Driven Development)** | Kod yazmadan önce veya kodla eşzamanlı olarak testleri yazıp doğrulamaya dayalı yazılım süreci. |
| **Self-Repair** | Kodlama ajanının terminal hata çıktılarını okuyup kendi yazdığı koddaki hataları düzeltmesi. |
| **Traceback / Stack Trace** | Program çöktüğünde çağrı yığınını, hata satırını ve istisna türünü gösteren terminal raporu. |
| **Edge Case** | Boş girdi, sıfır, negatif değerler veya sınır limitleri gibi uç durumlar. |
| **Sandboxed Execution** | Kodun ana işletim sisteminden izole, güvenli bir bellek alanında çalıştırılması. |
| **SWE-bench** | LLM tabanlı yazılım mühendisliği ajanlarının GitHub sorunlarını çözme başarımını ölçen kıyaslama standardı. |
| **Pass@k** | $k$ denemede en az bir kez tüm testleri geçen kod üretme olasılığı metriği. |
| **AssertionError** | Birim testteki beklenen değer ile gerçekleşen değer uyuşmadığında fırlatılan hata. |
| **AST (Abstract Syntax Tree)** | Kodun sözdizimsel yapısını ağaç biçiminde temsil eden ve güvenlik denetimi sağlayan yapı. |
| **Buffer Flush** | Dize sıkıştırma algoritmalarında döngü bittiğinde tamponda kalan son veri grubunun sonuca eklenmesi. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %100 doğrulanmış, hatasız ve test  │ • Test suite yetersizse gizli        │
 │   onaylı kod üretme garantisi.       │   mantık hatalarının gözden kaçması. │
 │ • Traceback odaklı hızlı onarım ile  │ • Birden fazla test döngüsü sebebiyle│
 │   token ve zaman tasarrufu.          │   artan çıkarım süresi.              │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Otonom yazılım mühendisliği ve     │ • Güvensiz ortamlarda rastgele kod   │
 │   CI/CD pipeline entegrasyonları.    │   çalıştırmanın getirdiği güvenlik   │
 │                                      │   (Sandbox breakout) riskleri.       │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/code_generation_unit_test_loop_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
