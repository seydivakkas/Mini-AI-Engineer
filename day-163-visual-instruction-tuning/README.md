# Day 163: Görsel Komut İnce Ayarı (Visual Instruction Tuning / Visual SFT) — Kayıp Maskeleme ve Çok Turlu Görsel Sohbet

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 3. günüdür. LLaVA-Instruct-150k formatındaki çok turlu görsel diyalog verilerini işleyen, görsel tokenları ($256$ adet) ve kullanıcı soru promptlarını kayıp fonksiyonunda ekarte eden (**Target-Only Loss Masking with Label -100**) ve yalnızca asistan yanıtı üzerinde gradyan üreterek modeli eğiten uçtan uca **Görsel SFT (Supervised Fine-Tuning) Motoru** mimarisini sıfırdan PyTorch ile inşa etmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Visual Instruction Tuning" Nedir ve Neden "Kayıp Maskeleme (-100)" Kullanılır?
- **Sorun (Piksel Tahmini Tuzağı):**
  Bir VLM'e girdi olarak $[ \text{Görsel (256 token)} + \text{Soru (14 token)} + \text{Cevap (18 token)} ]$ verdiğinizde, standart dil modelleme kaybı (Autoregressive Next-Token Loss) uygularsanız, model resmin piksellerini veya kullanıcının sorusunu da tahmin etmeye çalışır! Bu hem anlamsızdır hem de modeli bozar.
- **Çözüm (Target-Only Loss Masking):**
  PyTorch'un `CrossEntropyLoss(ignore_index=-100)` yeteneği kullanılarak;
  - Resim tokenlarının etiketleri: `[-100, -100, ...]`
  - Kullanıcı sorusunun etiketleri: `[-100, -100, ...]`
  - Asistan cevabının etiketleri: `[Token_1, Token_2, ...]` olarak ayarlanır.
  Böylece model sadece ve sadece doğru asistan yanıtını üretmeye odaklanır!

```
 GİRDİ DİZİSİ (Forward):
  [IMG_1 ... IMG_256] [HUMAN_PROMPT] [ASSISTANT_RESP]
           │                 │               │
           ▼                 ▼               ▼
 HEDEF ETİKETLER (Labels):
  [-100  ...  -100 ] [ -100 ... -100] [ T_1 ... T_K ]
           │                 │               │
           ▼ (Yoksayılır)    ▼ (Yoksayılır)  ▼
 KAYIP FONKSİYONU:
  CrossEntropy(Logits, Labels, ignore_index=-100)
  ==> Sadece Asistan Tokenlarında Gradyan Üretilir!
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Hedefe Odaklı Maskeli Çapraz Entropi Kaybı Formülü
- Girdi dizisi boyutu $L = L_v + L_p + L_r$ olmak üzere kayıp fonksiyonu:
  $$\mathcal{L}_{\text{V-SFT}} = -\frac{1}{L_r} \sum_{i = L_v + L_p + 1}^{L} \log P(y_i \mid \mathbf{X}_v, y_{<i})$$
- Burada $\mathbf{X}_v$ ($256$ görsel token) ve $y_{\le L_v + L_p}$ (kullanıcı komutu) koşul olarak alınır; gradyan yalnızca asistan tokenları $y_i$ üzerinden akar.

### B. LLaVA-Instruct Veri Seti Kategorizasyonu
1. **Kısa VQA (58k Örnek):** Doğrudan ve net nesne/renk soruları ("Masanın üzerinde ne var?").
2. **Detaylı Açıklama (Detailed Description - 45k Örnek):** Sahneyi zengin ve betimleyici bir dille paragraf olarak anlatma.
3. **Karmaşık Muhakeme (Complex Reasoning - 47k Örnek):** Neden-sonuç ilişkileri ("Sürücü neden yavaşlamalı?").

### C. Çok Turlu (Multi-Turn) Görsel Diyalog Şablonu
- Şablon yapısı:
  ```text
  [USER]: <image>\n{Soru 1} [ASSISTANT]: {Cevap 1} [USER]: {Soru 2} [ASSISTANT]: {Cevap 2}
  ```
- Görsel tokenlar sadece konuşmanın ilk turunda dizinin en başında yer alır; sonraki turlarda tekrar eklenmez.

### D. Gradyan Akışı ve Optimizasyon Dinamikleri
- ViT kodlayıcı dondurulurken (frozen), MLP Projektör ve LLM Transformer blokları eğitilir.
- Testlerimizde 5 adımda kayıp $6.9053 \to 0.2117$ seviyesine inerek **%96.93 iyileşme** sağlamıştır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Visual Instruction Tuning (Visual SFT)** | VLM'leri insan komutlarına ve görsel diyaloglara uyarlamak için yapılan denetimli ince ayar. |
| **Loss Masking** | Kayıp fonksiyonunda belirli tokenların (resim ve prompt) gradyan üretimini durdurma tekniği. |
| **ignore_index=-100** | PyTorch CrossEntropy'de gradyanı sıfırlayan standart kayıp maskeleme indisi. |
| **LLaVA-Instruct-150k** | GPT-4 ile üretilen öncü çok modlu komut takip veri seti. |
| **Target-Only Supervision** | Modelin sadece üretmesi beklenen hedef metin üzerinden cezalandırılması/ödüllendirilmesi. |
| **Detailed Description** | Görseldeki tüm nesne, konum ve eylemleri ayrıntılı betimleme görevi. |
| **Complex Visual Reasoning** | Görsel ipuçlarından mantıksal çıkarım ve neden-sonuç analizi yapma. |
| **Multi-Turn Visual Chat** | Aynı görsel üzerinde kullanıcıyla ardışık soru-cevap yürütme yeteneği. |
| **Prompt Template** | Konuşma rollerini (USER, ASSISTANT) belirten yapılandırılmış metin kalıbı. |
| **Visual SFT Trainer** | Görüntü, metin ve maskeli etiketleri GPU'ya besleyen optimize eğitim döngüsü. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Görsel sohbet ve komut takibinde   │ • Yüksek kaliteli çok modlu          │
 │   akıcı ve tutarlı yanıt üretimi.    │   açıklama verisi üretmenin maliyeti │
 │ • Kayıp maskeleme ile %100 hedefe    │   (GPT-4V API bağımlılığı).          │
 │   odaklı gradyan akışı.              │                                      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Çok modlu asistanlar, medikal VQA, │ • Halüsinasyonlu eğitim verisi       │
 │   görme engelliler için akıllı       │   nedeniyle modelin olmayan          │
 │   görsel rehberler inşa etme.        │   nesneleri uydurma (Hallucination)  │
 │                                      │   riski.                             │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/visual_instruction_tuning_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
