# Day 148: Düşünce Yollarında Geri İzleme (Backtracking) ve Çıkmaz Sokakları Fark Etme

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; OpenAI o1, o3, DeepSeek-R1 ve QwQ modellerinin en kritik akıl yürütme yeteneği olan **Düşünce Yollarında Geri İzleme (Backtracking)**, **Çıkmaz Sokak (Dead-End) Tespiti**, **İçsel Monolog ile Hata Kurtarma (`Wait, that's not right...`)**, **LIFO Düşünce Yığını (Thought Stack)** ve **Kontrol Noktalarına Geri Yükleme (State Rollback)** mekanizmalarını sıfırdan uygulamaktadır.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ İnsanlar ve Reasoning Modelleri Neden Geri İzleme (Backtracking) Yapar?
- **Standart LLM (Tek Geçişli CoT - Geri Dönüşsüz):**
  Klasik bir LLM bir düşünce zincirinde 3. adımda mantık veya hesap hatası yaparsa ($1.10 - 1.00 = 0.10 \implies \text{Top} = \$0.10$), bu hataya kilitlenir ve devamındaki 5 adımı bu yanlış temel üzerine kurarak halüsinasyon üretir.
- **Reasoning LLM (Backtracking & Error Recovery):**
  Model çıkarım anında adımlarını denetler. Bir çelişki tespit ettiğinde içsel bir monolog üretir:
  `<think> Bekle, bu çıkarım hatalı! Top $0.10 olursa Sopa $1.10 olur ve toplam $1.20 çıkar. Geri dönüyorum... </think>`
  Son geçerli kontrol noktasına (**Checkpoint**) geri döner ve alternatif bir düşünce dalını keşfeder.

```
        BACKTRACKING & CHECKPOINT ARCHITECTURE
   [Kontrol Noktası #1: Sopa + Top = 1.10] (CHECKPOINT)
       │
       ├─► [Hatalı Dal: Top = $0.10]
       │   └──► ÇELİŞKİ TESPİT EDİLDİ! (Top=0.10 => Toplam 1.20)
       │   └──► ROLLBACK TO CHECKPOINT #1 [BACKTRACK!]
       │
       └─► [Geçerli Dal: Sopa = Top + 1.00]
           └──► [2 * Top = 0.10]
           └──► [Top = $0.05] (KUSURSUZ ÇÖZÜM!)
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: LIFO Düşünce Yığını ve Checkpoint Motoru
- Düşünce durumları bir Call-Stack üzerinde saklanır:
  $$S = [s_0, s_1, \dots, s_t]$$
  - Her onaylanmış dönüm noktası bir kontrol noktası $c_i \in S$ olarak işaretlenir.
  - Hata durumunda yığından hatalı çerçeveler $pop()$ edilerek son $c_i$ durumuna geri yükleme (**Rollback**) yapılır.

### B. Çıkmaz Sokak (Dead-End) ve Çelişki Tespiti
- Bir adımı denetleyen kural motoru:
  $$\text{Kontrol}(s_t) \implies \begin{cases} \text{Geçerli}, & \text{Tutarlı ise} \\ \text{Çelişki / Çıkmaz}, & s_t \models \bot \text{ ise} \end{cases}$$

### C. İçsel Monolog ve İyileşme Belirteçleri (Self-Correction Tokens)
- Modelin kendi kendini düzeltmesini tetikleyen özel düşünce akışı:
  `"Wait...", "Let me reconsider...", "Actually, that contradicts..."`

### D. Hata Yayılımını Engelleme (Error Compounding Prevention)
- Tek geçişli modellerde hata kümülatif olarak büyürken ($E_t \propto \prod (1 - p_i)$), Backtracking ile hata $O(1)$ sürede izole edilir ve budanır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Backtracking** | Bir arama yolunda çıkmaza girildiğinde son geçerli karara geri dönme algoritması. |
| **Dead-End Detection** | Mevcut durumun hiçbir geçerli çözüme ulaşamayacağının erken tespiti. |
| **State Rollback** | Belleğin veya düşünce yığınının önceki geçerli kontrol noktasına geri yüklenmesi. |
| **Thought Stack** | Düşünce adımlarını ve değişken durumlarını LIFO (Son Giren İlk Çıkar) düzeninde tutan yığın. |
| **Checkpoint** | Doğrulanmış ve güvenli kabul edilen durum kontrol noktası. |
| **Internal Monologue** | Modelin kendi mantığını denetlerken ürettiği içsel ses ve akıl yürütme metni. |
| **Self-Correction** | Modelin dış müdahale olmadan kendi hatasını fark edip düzeltmesi. |
| **Branch Pruning** | Hatalı veya çıkmaz olduğu anlaşılan düşünce dallarının ağaçtan silinmesi. |
| **Cognitive Trap** | Bat & Ball paradoksu gibi sezgisel ama yanlış yanıtlara yönlendiren bilişsel tuzaklar. |
| **Verification Token** | Akıl yürütme zincirinde doğrulama adımını başlatan özel belirteç. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Hatalı adımlarda %100 kurtarma     │ • Geri sarma ve yeniden deneme       │
 │   ve doğru sonuca ulaşma başarımı.   │   sebebiyle artan token tüketimi.    │
 │ • Halüsinasyon zincirlerini kökten   │ • Yığın (Call-Stack) yönetimi için   │
 │   kesme ve budama yeteneği.          │   ek bellek gereksinimi.             │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Matematik, mantık ve kodlama       │ • Çıkmaz tespit mekanizmasının       │
 │   görevlerinde insan düzeyinde       │   hatalı pozitif vermesi halinde     │
 │   problem çözme kabiliyeti.          │   doğru yolların budanması riski.    │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/backtracking_and_error_recovery_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
