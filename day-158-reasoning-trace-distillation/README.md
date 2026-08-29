# Day 158: Büyük Akıl Yürüten Modelin (DeepSeek-R1) Düşünce İncilerini Küçük Modele Damıtma (Reasoning Trace Distillation)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; DeepSeek-R1 ve OpenAI o1 gibi büyük akıl yürüten modellerin (671B MoE) saf RL ile geliştirdiği derin düşünce zincirlerini (`<think> ... </think>`), geri izleme ve öz-düzeltme adımlarını filtreleyerek kompakt öğrenci modellere (1.5B / 7B / 14B) SFT ile aktaran **Reasoning Trace Distillation** mimarisini hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Reasoning Trace Distillation (Düşünce İzi Damıtma)" Nedir ve Neden Devrimseldir?
- **Sorun (Devasa Model Maliyeti):**
  DeepSeek-R1 (671 Milyar parametre) matematik ve kodlama olimpiyatlarında harikalar yaratır; ancak onu çalıştırmak için 8x H100 GPU kümesi ve yüz binlerce dolar gerekir. Küçük modeller (1.5B) ise saf RL ile bu düşünce kalitesini kolay kolay öğrenemez.
- **Çözüm (Öğretmenden Öğrenciye Düşünce Damıtma):**
  1. **Aşama 1 (Öğretmen İzi Toplama):** 671B öğretmen modelden 800.000 adet `<think> ... </think>` içeren derin akıl yürütme verisi üretilir.
  2. **Aşama 2 (Kalite Filtresi):** Hatalı, döngülü ve çöp zincirler elenir; sadece doğru cevaba ulaşan ve "Aha moment" içeren izler saklanır.
  3. **Aşama 3 (Öğrenci SFT Eğitimi):** 1.5B kompakt model bu izlerle eğitilir.
  4. **Sonuç:** 1.5B öğrenci model, MATH benchmark'ında $\%28.6$'dan $\%84.2$'ye fırlayarak öğretmenin $\%91.1$ performansını yakalar!

```
      REASONING TRACE DISTILLATION PIPELINE
  [DeepSeek-R1 671B Öğretmen (Saf RL ile Eğitilmiş)]
           │
           ▼  800k+ Düşünce İzi (<think>...</think>)
  [Düşünce İzi Kalite & Doğruluk Süzgeci]
    - Hatalı / döngülü zincirleri temizle
    - Refleksif 'Aha moment' adımlarını sakla
           │
           ▼  Kürate Edilmiş SFT Veri Seti
  [Küçük Öğrenci Model (Qwen-1.5B / 7B / 14B)]
           │
           ▼
  [R1-Distill Küçük Model (%91 Öğretmen Seviyesi)]
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. SFT Düşünce Damıtma Kayıp Fonksiyonu
- Girdi $x$ ve öğretmenden damıtılmış düşünce izi $y = [t_1, \dots, t_k, \text{</think>}, a_1, \dots, a_m]$ için öğrencinin kaybı:
  $$\mathcal{L}_{\text{distill}}(\theta) = -\sum_{i=1}^{|y|} \log P_{\theta}(y_i \mid x, y_{<i})$$
- Öğrenci, öğretmenin hem düşünce akışını ($t_i$) hem de nihai yanıtını ($a_i$) modellemeyi öğrenir.

### B. "Aha Moment" ve Refleksif Düzeltme Transferi
- Saf SFT standart yanıtlarda sadece doğrudan cevabı taklit ederken; düşünce damıtma modelin kendi kendini sınama ("Wait, let me double check", "Actually...") mekanizmalarını içselleştirmesini sağlar.

### C. Düşünce İzi Kürasyon Kriterleri
- $1000+$ token süren kısırdöngüler ve doğrulanmamış (Ground-Truth False) izler filtrelenerek veri kirliliği (Data Poisoning) $\%0$'a indirilir.

### D. Hesaplama ve Çıkarım Maliyeti Verimliliği
- 1.5B model, 671B modele kıyasla $\sim 400\times$ daha düşük bellek ve $20\times$ daha hızlı çıkarım sağlar.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Reasoning Trace Distillation** | Büyük bir modelin düşündüğü ara adımları küçük bir modele SFT ile aktarma süreci. |
| **Teacher Model (Öğretmen)** | Yüksek kapasiteli, saf RL ile derin akıl yürütme öğrenmiş devasa model (örn: DeepSeek-R1 671B). |
| **Student Model (Öğrenci)** | Uç cihazlarda çalışabilen, damıtılmış veriyle eğitilen kompakt model (örn: Qwen-1.5B). |
| **`<think>` Token Formatı** | Modelin içsel düşünme adımlarını kullanıcı yanıtından ayıran yapısal etiketler. |
| **Aha Moment (Öz-Düzeltme)** | Modelin düşünce zinciri ortasında hatasını fark edip strateji değiştirdiği an. |
| **Trace Curation** | Ham düşünce izlerinin doğruluk ve döngü anomalilerine karşı elenmesi. |
| **SFT (Supervised Fine-Tuning)** | Belirlenen girdi-çıktı çiftleri üzerinde standart çapraz entropi ile ince ayar. |
| **Inference Efficiency** | Düşük bellek ve işlemci gücüyle yüksek başarım elde etme oranı. |
| **Data Filtering Pipeline** | Eğitim verisindeki düşük kaliteli veya hatalı adımları temizleyen süzgeç. |
| **Knowledge Transfer** | Büyük bir yapay zekanın bilişsel örüntülerinin küçük ağlara aktarılması. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • 1.5B/7B gibi küçük modellerde      │ • Öğrenci model, öğretmenin hiç      │
 │   olimpik düzeyde matematik yetisi.  │   görmediği tamamen yeni alanlarda   │
 │ • 400x daha ucuz çıkarım maliyeti.   │   genelleme sınırına takılabilir.    │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Mobil cihazlar, robotik sistemler  │ • Öğretmenin yaptığı sistematik ince │
 │   ve yerel sunucularda DeepSeek-R1   │   hataların öğrenciye de aynen       │
 │   kalitesinde otonom ajanlar.        │   kopyalanması (Error Propagation).  │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/reasoning_trace_distillation_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
