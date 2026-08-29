# Day 120: FAZ 6 BÜYÜK FİNALİ — Aligned LLM Benchmark & Chatbot Arena Şampiyonası

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6 BÜYÜK FİNALİ: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO (Gün 102 - Gün 120)**  
> Bu modül; Faz 6 boyunca geliştirilen tüm hizalama yöntemlerini (SFT, DPO, KTO, ORPO, SimPO, GRPO, Merged, Distilled) çok boyutlu değerlendiren, **LLM-as-a-Judge (MT-Bench, AlpacaEval)**, **Bradley-Terry Tabanlı Dinamik Elo Derecelendirme Motoru (Chatbot Arena)** ve **Pozisyon Yanlılığı (Position Bias) Telafisi** sistemini sıfırdan inşa edip Faz 6'yı başarıyla taçlandırır.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Yapay Zekânın Olimpiyatları: LLM Hakemliği ve Chatbot Arena"

Birbirinden farklı 9 yapay zeka modeli eğittiniz (SFT, DPO, GRPO, Distilled vb.). Peki hangisinin gerçekten daha zeki, daha mantıklı ve daha güvenli olduğunu insanlara yüz binlerce anket yaptırmadan nasıl ölçebiliriz?

**LLM-as-a-Judge ve Chatbot Arena** tam olarak bir **Yapay Zekâ Satranç Ligi** kurar:
1. ⚖️ **LLM Hakemliği (LLM-as-a-Judge):** Çok güçlü bir model (örneğin GPT-4 veya Claude-3.5), yarışmacı modellerin yanıtlarını objektif rubriklere göre 1-10 puan arasında değerlendirir (MT-Bench).
2. 🔄 **Pozisyon Yanlılığı ve Swap Testi:** Hakemler genellikle ilk okudukları yanıtı (Model A) daha çok beğenme eğilimindedir (**Position Bias**). Swap Testi ile modellerin yerleri değiştirilir ($[A, B] \leftrightarrow [B, A]$); karar çelişirse beraberlik verilir.
3. 🏆 **Bradley-Terry Elo Motoru:** Tıpkı satrançta olduğu gibi her model 1000 Elo puanıyla başlar. Güçlü bir modeli yenen model çok puan kazanırken, zayıf modele yenilen çok puan kaybeder.
4. 🥇 **Nihai Lider:** Faz 6 boyunca inşa ettiğimiz tüm tekniklerin meyvesi olarak akıl yürüten ve damıtılmış modeller zirveye yerleşir!

```
     HAKEME GELEN ÇİFTLİ İSTEM (Pairwise)                  BRADLEY-TERRY ELO GÜNCELLEMESİ
 ┌──────────────────────────────────────────┐            ┌──────────────────────────────────────────────┐
 │ Model A (GRPO) vs Model B (Base Model)   │ ─────────> │ E_A = 1 / (1 + 10^((R_B - R_A)/400))         │
 └────────────────────┬─────────────────────┘            │ R_A <- R_A + K * (S_A - E_A)                 │
                      │                                  └──────────────────────┬───────────────────────┘
                      ▼                                                         │
          [SWAP TESTİ İLE POZİSYON DENETİMİ]                                    ▼
          ├── 1. Yön: [A, B] -> Model A Kazandı                   [CHATBOT ARENA LİDERLİK TABLOSU]
          ├── 2. Yön: [B, A] -> Model A Kazandı                   🥇 Distilled / GRPO Reasoning (1251 Elo)
          └── Sonuç: %100 Objektif Zafer                          🥈 ORPO / SimPO / DPO SOTA
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & LLM-as-a-Judge Paradigması
- **MT-Bench Çok Turlu Değerlendirme:** 8 farklı bilişsel alanda (Kodlama, Matematik, Muhakeme, Güvenlik, Yaratıcılık, Rol Yapma, Çıkarım, Özetleme) 1-10 puanlama.
- **AlpacaEval & Pairwise Kazanma Oranı:** İki modelin yanıtlarını kör eşleştirmeyle kıyaslayarak mutlak galibiyet oranlarını çıkarma.

### 2. Pozisyon ve Uzunluk Yanlılığı (Position & Verbosity Bias) Telafisi
- **Position Bias:** Hakem modellerin ilk sırada sunulan yanıta %60+ oranında öncelik vermesi problemidir. Çözüm: Her eşleşme için $A \leftrightarrow B$ yer değiştirilip iki yönlü tutarlılık denetlenir.
- **Verbosity Bias:** Uzun ama boş konuşan modellerin gereksiz yüksek puan almasını engellemek için kod blokları, mantık zincirleri ve kesin ispat adımları ödüllendirilir.

### 3. Bradley-Terry Teoremi & Dinamik Elo Derecelendirme Motoru
İki model arasındaki beklenen kazanma olasılığı:
$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$
Maç sonrası dinamik Elo güncellemesi ($K=32$):
$$R_A \leftarrow R_A + K(S_A - E_A), \quad S_A \in \{1.0 \text{ (Galibiyet)}, 0.5 \text{ (Beraberlik)}, 0.0 \text{ (Mağlubiyet)}\}$$

### 4. Faz 6 Hizalama Yöntemleri Büyük Karşılaştırması
- **Base Model (620 Elo):** Ham ön eğitim, düşük talimat takibi.
- **Packed SFT (891 Elo):** Temel talimat ve format uyumu.
- **DPO / ORPO / SimPO (~1250 Elo):** Yüksek kaliteli insan tercihi hizalaması.
- **GRPO & Distilled (~1252 Elo):** Akıl yürütme düşünce zincirleri (`<think>`) ve derin matematiksel ispat ile **Şampiyonluk!**

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **LLM-as-a-Judge** | Güçlü bir dil modelinin diğer modellerin yanıtlarını hakem olarak değerlendirmesi. |
| **MT-Bench** | 8 farklı yetenek alanında çok turlu diyalog kalitesini ölçen standart LLM benchmarkı. |
| **AlpacaEval** | Hızlı ve otomatik çiftli eşleşmelerle model kazanma oranını hesaplayan test kütüphanesi. |
| **Chatbot Arena** | Modelleri kör dövüşlerle (blind battles) Elo puanına göre sıralayan platform (LMSYS). |
| **Position Bias** | Hakem modelin ilk sırada sunulan yanıta istatistiksel olarak iltimas geçmesi yanılgısı. |
| **Swap Test** | Model sıralamasını tersine çevirerek pozisyon yanlılığını sıfırlayan simetri testi. |
| **Verbosity Bias** | Hakem modellerin daha uzun metinleri daha kaliteli zannetme yanılgısı. |
| **Elo Rating** | Satranç ve oyun teorisinde göreli yetenek seviyesini belirleyen dinamik puanlama sistemi. |
| **Bradley-Terry Model** | İkili karşılaştırmalarda kazanma olasılıklarını modelleyen matematiksel olasılık teorisi. |
| **K-Factor** | Her maç sonucunun Elo puanı üzerindeki maksimum değişim etkisini belirleyen katsayı. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • İnsan değerlendirmesinden 100x hızlı│ • Hakem modelin kendi zeka seviyesi  │
 │ • Swap testi ile %0 pozisyon yanlılığı│   değerlendirme tavanını belirler.   │
 │ • Bradley-Terry Elo ile net sıralama. │ • Çok benzer yanıtlarda beraberlik   │
 │ • MT-Bench ile 8 alanda tam teşhis.   │   oranı artabilir.                   │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Sürekli CI/CD pipeline'larında     │ • Model sağlayıcılarının hakem prompt-│
 │   otomatik regresyon testi kurabilme.│   larına göre overfitting yapma riski│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/faz6_capstone_benchmark_paneli.png` dosyası üretilir:
1. **Chatbot Arena Dinamik Elo Liderlik Tablosu**
2. **MT-Bench 8 Kategori Bazlı Başarım (Base vs DPO vs GRPO)**
3. **Turnuva Karşılaşmaları Kazanma Oranı (%)**
4. **LLM Hakemliği Pozisyon Yanlılığı Denetimi (%0.00 Yanlılık)**
5. **Faz 6 Hizalama & Akıl Yürütme Evrim Ağacı**
6. **FAZ 6 BÜYÜK FİNALİ MEZUNİYET VE ŞAMPİYONLUK SERTİFİKASI**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Faz 6 Capstone turnuvasını ve Elo sıralamasını koşturun
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
