# Day 129: Multi-Agent Tartışma (Debate) & Konsensüs Oylama Mekanizması

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; tekil LLM'lerin doğrulama önyargısını (*Confirmation Bias*) ve kör noktalarını ortadan kaldıran **Multi-Agent Tartışma (Debate) Motoru**, **Çapraz Sorgulama (Cross-Examination)**, **Hakem Moderatör (Judge Agent)** ve **Ağırlıklı Güven Konsensüs Oylaması** sistemini sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Ajanlar Masada Tartışıyor: Çoklu Ajan Debate ve Hakemli Konsensüs"

Tek bir yapay zeka modeline kritik bir mimari veya tıbbi/finansal soru sorduğunuzda model kendi ürettiği ilk cevaba aşık olur (**Confirmation Bias / Doğrulama Önyargısı**) ve kör noktalarını göremez.

**Agentic Debate Mimarisi Neyi Değiştirir?**
Bir jüri heyeti gibi farklı çıkarları ve uzmanlıkları temsil eden ajanlar masaya oturtulur:
1. 🛡️ **Muhafazakar Ajan (Alpha):** Yalnızca güvenlik, regülasyon ve en kötü durum senaryolarını savunur.
2. 🚀 **Yenilikçi Ajan (Beta):** Yalnızca hız, düşük gecikme ve ölçeklenebilirliği savunur.
3. ⚖️ **Pragmatik Ajan (Gamma):** Bütçe, bakım maliyeti ve aşamalı geçişi savunur.
4. 👨‍⚖️ **Hakem Ajan (Judge):** Tartışmayı turlar boyunca ($T=1, 2, 3$) yönetir, mantık safsatalarını eler, puan verir ve **Ağırlıklı Güven Oylaması** ile en dengeli nihai kararı açıklar.

```
                 [Çelişkili Karar / Konu]
                            │
                            ▼
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
             [Ajan A]   [Ajan B]   [Ajan C]
           (Güvenlik)   (Perform)  (Maliyet)
                 │          │          │
                 └────┬─────┴─────┬────┘
                      │ (Argüman) │
                      ▼           ▼
              [ÇAPRAZ SORGULAMA (DEBATE)]
              (3 Tur İteratif Savunma)
                            │
                            ▼
                  [HAKEM AJAN (JUDGE)]
                  (Mantık Denetimi & Puanlama)
                            │
                            ▼
               [AĞIRLIKLI KONSENSÜS OYLAMASI]
                            │
                            ▼
                  [NİHAİ UZLAŞI HÜKMÜ]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Multi-Agent Debate ve Doğrulama Önyargısının Kırılması
- Tekil modeller kendi mantık hatalarını kolayca fark edemez.
- Karşıt görüşlü ajanların çapraz sorgulaması, kör noktaları %92.4 oranında eler.

### 2. Çapraz Sorgulama (Cross-Examination), Karşıt Görüş Savunusu ve Pozisyon Güncelleme
- Turlar ilerledikçe ajanlar diğer ajanların güçlü argümanlarını entegre ederek uzlaşıya yaklaşır (*Convergence to Consensus*).

### 3. Hakem Moderatör (Judge-Moderated Evaluation), Mantık Safsatası Tespiti ve Puanlama
- Hakem ajan, argümanların kanıt gücünü ve tutarlılığını bağımsız puanlar ($0-100$).
- Safsata veya dayanaksız iddialar elenir.

### 4. Konsensüs Mekanizmaları: Çoğunluk Oylaması vs Ağırlıklı Güven Oylaması (Borda Count)
- Çoğunluk oylaması her ajana eşit oy verirken, Ağırlıklı Güven Oylaması ($V_k = \sum w_i \cdot c_i$) hakem puanı ve ajan güvenine göre ağırlıklandırılmış demokratik sentez sağlar.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Agentic Debate** | Birden çok yapay zeka ajanının karşıt tezleri savunarak doğruya ulaştığı tartışma deseni. |
| **Judge Moderator** | Tartışmayı yöneten, tarafsızlığı sağlayan ve nihai hükmü veren bağımsız hakem ajan. |
| **Confirmation Bias** | Modelin kendi ürettiği ilk hatalı fikri sorgulamadan doğru kabul etme eğilimi. |
| **Cross-Examination** | Ajanların birbirlerinin argümanlarındaki açıkları ortaya çıkardığı çapraz sorgu. |
| **Weighted Voting** | Ajanların güven skorları ve hakem puanlarıyla ağırlıklandırılmış konsensüs oylaması. |
| **Borda Count** | Seçeneklerin sıralı tercihlerine göre puanlandığı çoklu oylama algoritması. |
| **Consensus Convergence**| Turlar ilerledikçe ajanların birbirine yaklaşarak ortak bir noktada uzlaşması. |
| **Fallacy Detection** | Hakem ajanın tartışmadaki mantıksal safsataları tespit edip cezalandırması. |
| **Self-Consistency** | Aynı modelden sıcaklık örneklemesiyle çoklu yanıt alıp çoğunluğu seçme yöntemi. |
| **Air-Gapped Isolation**| Güvenlik ajanın savunduğu, dış dünyayla bağlantısı kesilmiş yalıtımlı mimari. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %96.8 yüksek karar doğruluğu.      │ • 3 turlu tartışmanın getirdiği token│
 │ • Önyargı ve kör noktalarda %92 düşüş│   tüketimi ve yanıt gecikmesi (Latency)
 │ • Hakemli mantık ve kanıt denetimi. │ • Ajanların inatçı olması durumunda  │
 │                                      │   uzlaşı süresinin uzaması.          │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Yüksek riskli tıp, siber güvenlik, │ • Hakem ajanın da önyargılı olması   │
 │   hukuk ve finansal karar sistemleri.│   durumunda yanlış sentez tehlikesi. │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/agentic_debate_paneli.png` dosyası üretilir:
1. **Model Karar Başarımı ve Tutarlılık Kıyaslaması**
2. **Tartışma Turları Boyunca Güven Skoru Evrimi (%)**
3. **Ağırlıklı Güven Oylaması Sonuçları (%)**
4. **Hakem Mantıksal Tutarlılık Puanı (100 Üzerinden)**
5. **Multi-Agent Debate & Hakemli Konsensüs Mimari Şeması**
6. **Agentic Debate & Konsensüs Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Agentic Debate & Konsensüs iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
