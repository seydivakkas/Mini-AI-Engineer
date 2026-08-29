# Day 116: Evol-Instruct & UltraFeedback ile Sentetik Veri Üretim Hattı

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; modern LLM eğitiminin ve hizalamasının yakıtı olan, tohum talimatları otonom evrimleştiren **Evol-Instruct (In-Depth: Kısıt Ekleme, Derinleştirme, Somutlaştırma, Muhakeme / In-Breadth: Mutasyon)** ve çok boyutlu tercih veri seti üreten **UltraFeedback** hattını sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Sentetik Veri Fabrikası: Tohumdan Zekâ Üretimi"

Yapay zeka modelleri eğitilirken en büyük darboğaz "kaliteli veri" bulmaktır. İnsanların yazdığı sorular genellikle basittir: *"Python'da sıralama nasıl yapılır?"*. Ancak bir modelin uzmanlaşması için karmaşık ve zorlu sorulara ihtiyacı vardır.

**Evol-Instruct ve UltraFeedback** tam olarak bir **Sentetik Veri Fabrikası** gibi çalışır:
1. 🌱 **Tohum İstem (Seed Prompt):** Basit bir soruyla başlarız (*"Sıralama fonksiyonu yaz"*).
2. 🧬 **In-Depth Evrim (Derinleştirme):** Soruya kısıtlar ve zorluklar eklenir: *"Ek bellek $O(1)$ olsun, kütüphane kullanmayın, 10 milyon kullanıcılı senaryoda çalışsın"*.
3. 🌳 **In-Breadth Evrim (Mutasyon):** Soru tamamen farklı bir alana çeşitlendirilir (*"Bu sıralama mantığını DNA dizilimi hizalamasına uyarlayın"*).
4. 🧹 **Eleme Filtresi (Elimination):** Eğer evrilen soru çok kolaysa, anlamsızsa veya tohumun aynısıysa çöpe atılır.
5. ⭐ **UltraFeedback Puanlama:** Modelin ürettiği farklı yanıtlar 4 ayrı boyutta puanlanır ve DPO/SimPO için en iyi (Chosen $y_w$) ve en zayıf (Rejected $y_l$) çiftler üretilir!

```
      TOHUM İSTEM                                        EVRİLMİŞ ULTRAFEEDBACK ÇİFTİ
 ┌───────────────────────────┐      Evol-Instruct       ┌──────────────────────────────────────────────┐
 │ "Sıralama fonksiyonu yaz" │ ───────────────────────> │ Prompt: "... O(1) bellek ve 10M senaryosu"   │
 └───────────────────────────┘    (Kısıt + Muhakeme)    │ Chosen (y_w): Detaylı, hatasız algoritma     │
                                                        │ Rejected (y_l): Yüzeysel, kısıtları unutan   │
                                                        └──────────────────────────────────────────────┘
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Evol-Instruct Evrim Operatörleri
- **Kısıt Ekleme (Add Constraints):** Bellek, zaman, format veya kütüphane sınırlamaları getirir.
- **Derinleştirme (Deepening):** Konunun teorik, matematiksel ve asimptotik temellerine iner.
- **Somutlaştırma (Concretizing):** Soyut kavramları gerçek dünya mühendislik vakalarına (HFT, LiDAR, Dağıtık Sistemler) dönüştürür.
- **Muhakeme Adımını Artırma (Increased Reasoning):** Çok adımlı dedüktif çıkarım zincirleri kurdurur.
- **Mutasyon (In-Breadth):** Farklı disiplinlere çapraz uyarlama yapar.

### 2. Otomatik Eleme ve Kalite Filtreleme (Elimination Heuristics)
Evrilen her istem kabul edilmez:
1. **Jaccard Token Benzerliği:** Tohumla benzerlik $> 0.92$ ise (yetersiz mutasyon) elenir.
2. **Karmaşıklık Kazancı ($\Delta C$):** Yeni karmaşıklık skoru tohumdan küçük veya eşitse elenir.
3. **Uzunluk ve Format Denetimi:** Anlamsız veya bozuk istemler temizlenir.

### 3. UltraFeedback Çok Boyutlu Tercih Puanlaması
Aday yanıtlar 4 kritik eksende 1-5 arası puanlanır:
- **Talimat Takibi:** Kısıtlara eksiksiz uyuldu mu?
- **Teknik Doğruluk:** Kod ve mantık hatasız mı?
- **Faydalılık:** Yanıt açık, pratik ve anlaşılır mı?
- **Muhakeme Derinliği:** Çözüm mantıksal adımlarla gerekçelendirildi mi?

### 4. Endüstriyel Ölçek ve Sentetik Veri Paradigması
- **Llama-3, WizardLM ve DeepSeek-V3:** Günümüzün en güçlü modelleri, trilyonlarca tokenlık sentetik veri ve Evol-Instruct türevleriyle eğitilmektedir.
- **İnsan Verisi Sınırı:** İnsan kaynaklı kaliteli verinin tükenmesiyle birlikte, LLM'in kendi kendini denetlediği sentetik fabrikalar standart hale gelmiştir.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Evol-Instruct** | Basit tohum talimatları otonom kural ve şablonlarla karmaşıklaştıran evrimsel istem motoru. |
| **In-Depth Evolution** | Bir istemin kısıtlarını, derinliğini, somutluğunu ve muhakeme adımlarını artıran dikey evrim. |
| **In-Breadth Evolution** | Bir istemi yeni konu başlıklarına ve disiplinlere uyarlayan yatay mutasyon evrimi. |
| **UltraFeedback** | LLM yanıtlarını 4 farklı boyutta puanlayıp yüksek kaliteli çiftli tercih verisi üreten sistem. |
| **Complexity Gain ($\Delta C$)** | Evrilen istemin tohum isteme kıyasla kazandığı teknik ve yapısal karmaşıklık farkı. |
| **Jaccard Similarity** | İki metin arasındaki kelime kümesi örtüşme oranı ($A \cap B / A \cup B$). |
| **Elimination Criterion** | Kalitesiz, kopya veya yapay zekanın yanıtlayamayacağı bozuk istemleri eleyen kural seti. |
| **Chosen / Rejected ($y_w, y_l$)** | Tercih hizalama eğitiminde (DPO/SimPO) kullanılan yüksek ve düşük puanlı yanıt çifti. |
| **Multi-Turn Evolution** | İstemlerin nesiller boyunca (Gen 1 -> Gen 2 -> Gen 3) kümülatif olarak zorlaştırılması. |
| **Synthetic Flywheel** | Güçlü modellerin kaliteli veri üretip bu veriyle daha da güçlü modeller eğitilmesi döngüsü. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Sınırsız ve ucuz veri üretimi.     │ • Filtrelenmezse model halüsinasyon- │
 │ • Karmaşıklıkta %100+ otonom artış.  │   larını kendi kendine çoğaltabilir. │
 │ • İnsan annotator maliyetini sıfırlar│ • Çok yüksek nesillerde aşırı yapay  │
 │ • DPO/SimPO için anında çift üretir. │   ve okunması zor istemler oluşabilir.│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Nadir alanlarda (HFT, Kuantum)     │ • Kalitesiz değerlendirici modeli    │
 │   özelleşmiş veri setleri kurabilme. │   yanlış tercih çiftleri üretebilir. │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/evol_instruct_paneli.png` dosyası üretilir:
1. **Evol-Instruct Nesiller Arası Karmaşıklık Skoru Artışı (Gen 0 -> Gen 3)**
2. **Kullanılan Evrim Operatörleri Dağılımı**
3. **UltraFeedback 4 Boyutlu Puanlama Profili**
4. **Kalite Filtresi Eleme & Kabul Dağılımı (%87.5 Kabul)**
5. **Sentetik Veri Üretim Boru Hattı Akışı**
6. **Sentetik Veri Kalite Sertifikası**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Sentetik veri üretim hattını koşturun
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
