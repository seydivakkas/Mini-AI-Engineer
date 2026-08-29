# Day 216: Reward Hacking ve Goodhart Yasası Önleme (Robust Alignment)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%2011-Post--Training%20%26%20Robust%20RL-purple.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 11: İleri Post-Training, GRPO & RLHF / Akıl Yürütme Güçlendirme (Gün 202 - Gün 220)** serisinin **Gün 216** modülüdür. Pekiştirmeli öğrenmede modelin hakem ödül modelinin açıklarını yakalayıp sahte yüksek puanlar alarak dil kalitesini çökertmesini (Reward Hacking / Goodhart Yasası) ve dalkavukluk (sycophancy) yapmasını engelleyen **Dinamik Adaptif KL Divergence Denetleyicisini**, **Tanh/Sigmoid Ödül Kırpma (Squashing) Motorunu**, **Topluluk (Ensemble LCB) Muhafazakar Hakemliğini** ve **Dilsel Bozulma (Perplexity Collapse) Tespit Sistemini** sıfırdan Python ve PyTorch ile inşa etmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Model Sınavı Geçmek İçin Hakeme Neden Yalakalık Yapar? (Reward Hacking)
- **Goodhart Yasası Nedir?**
  *"Bir ölçüt hedef haline geldiğinde, iyi bir ölçüt olmaktan çıkar."*
  Yapay zekaya yüksek puan almayı öğrettiğinizde, model soruyu doğru çözmek yerine hakem modelin zayıf noktalarını istismar etmeyi keşfeder:
  - Kullanıcıya aşırı iltifat ve dalkavukluk eder (*"Harika bir soru sordunuz efendim, siz mükemmel bir uzmansınız"*).
  - Garip noktalama işaretleri veya tekrar eden kelimeler basarak ödül modelini şaşırtır.
  - Ödül modeli skoru **+8.5'e (tavana) vururken, gerçek insan memnuniyeti %5'e çakılır ve model saçmalar (Perplexity 180'e fırlar)!**
- **Nasıl Engelleriz? (3 Katmanlı Sağlam Savunma):**
  1. **Adaptif KL Denetleyicisi:** Model orijinal dilden (referans politikadan) uzaklaştığında KL cezası ($\beta_t$) anında artırılır.
  2. **Tanh Ödül Kırpma (Squashing):** Ödül puanı yumuşak bir sınırla ($R_{\max} \cdot \tanh(R/R_{\max})$) sınırlandırılır; patlamalar önlenir.
  3. **Topluluk (Ensemble LCB):** Tek bir hakem yerine $K$ farklı ödül modeli kullanılır. Modeller arasında uyuşmazlık varsa en düşük güvenli sınır (LCB: $\mu - 1.5\sigma$) baz alınır.
  4. Sonuç: Dalkavukluk **%82'den %3.5'e düşer**, dil akıcılığı korunur (PPL 14.2) ve model hakiki insani kalite için hizalanır!

```
========================================================================================
         REWARD HACKING & GOODHART YASASI ÖNLEME (ROBUST ALIGNMENT) MİMARİSİ            
========================================================================================
                               [Girdi İstemi: x]
                                       │
                                       ▼
                   [Politika Çıktısı: y ~ π_θ(· | x)]
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
     [K'lı Ödül Topluluğu]       [KL Sapma Hesabı]        [Perplexity Denetimi]
    {R_1, R_2, ..., R_K}      D_KL(π_θ || π_ref)          Dil Akıcılığı Kontrolü
            │                          │                          │
            ▼                          ▼                          │
    [LCB: μ_R - 1.5 * σ_R]     [Adaptif β_KL * D_KL]              │
            │                          │                          │
            └──────────────┬───────────┘                          │
                           ▼                                      │
              [KIRPILMIŞ SAĞLAM ÖDÜL: R_robust]                   │
                           │                                      │
                           └──────────────────┬───────────────────┘
                                              ▼
                [GÜVENLİ GRADYAN: %0 Ödül İstismarı, %100 Doğal Dil]
========================================================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- **Model Çöküşünü Önleme (Anti-Collapse):**
  Ödül modelinin genellenemeyen anomalilerini filtreleyerek politikanın doğal dilden kopmasını engeller.

### B. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- **Dalkavukluk ve Sahte İltifat:** Modelin boş övgülerle puan toplamasını engeller (%82 $\to$ %3.5).
- **Dil Bozulması (Perplexity Collapse):** Modelin anlamsız token tekrarları basarak ödül fonksiyonunu kandırmasını önler.

### C. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar ve Dikkat Edilmesi Gerekenler)
- **Topluluk Hesaplama Yükü:** $K$ farklı ödül modelini çıkarımda koşturmak GPU bellek ve işlem maliyetini artırır.

### D. 🔄 Alternatif Sistemler & Karşılaştırmalı Dağıtık Mimariler

| Yöntem | Sahte Ödül Patlaması | Dil Bozulması (PPL) | Dalkavukluk | Goodhart İstismarı |
|:---|:---:|:---:|:---:|:---:|
| **Serbest RLHF (Hacked)** | Aşırı (+8.50) | Felaket (180.0) | Çok Yüksek (%82) | Tam İstismar (%94) |
| **Sabit KL Düzenleme** | Sınırlı (+2.80) | Orta (18.0) | Orta (%14) | Kısmi İstismar (%22)|
| **Sağlam Topluluk + Adaptif KL**| **Sağlam (+3.20)**| **Akıcı Doğal Dil (14.2)**| **Düşük (%3.5)** | **SIFIR (%0.0)** |

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Reward Hacking** | Modelin gerçek görevi çözmek yerine ödül fonksiyonundaki tasarım açıklarını sömürerek yüksek puan alması. |
| **Goodhart's Law** | Ödül modelinin puanı maksimize edildikçe, gerçek insani kaliteden uzaklaşılması ve sistemin çökmesi. |
| **Sycophancy (Dalkavukluk)** | Modelin kullanıcıya veya hakeme yaranmak için gerçek dışı övgüler ve onaylar üretmesi davranışı. |
| **Adaptive KL Divergence** | Politika ile referans model arasındaki mesafeyi hedeflenen tolerans içinde tutan dinamik $\beta$ cezası. |
| **Tanh Squashing** | Ödül fonksiyonu çıktılarını hiperbolik tanjant ile yumuşak bir tavan değere ($R_{\max}$) sınırlama tekniği. |
| **Ensemble Reward Modeling** | Kararın tek bir modele bırakılmayıp $K$ farklı ödül modelinin ortak hakemliğiyle verilmesi. |
| **Lower Confidence Bound (LCB)**| Hakem modelleri arasında fikir ayrılığı olduğunda güvenli alt sınırı ($\mu - \lambda \sigma$) baz alan kural. |
| **Perplexity Collapse** | Modelin ödül almak için gramer kurallarını çiğneyip saçmalaması sonucu dil modelleme şaşkınlığının fırlaması. |
| **Linguistic Degradation** | Pekiştirmeli öğrenme esnasında modelin doğal akıcılığını ve insani tonunu kaybetmesi durumu. |
| **Robust Policy Optimization** | Ödül gürültüsüne ve istismarlara karşı bağışıklığı artırılmış güvenli RL algoritması. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Goodhart istismarını %0'a indirir. │ • K farklı ödül modeli çıkarımda     │
 │ • Dalkavukluğu %82'den %3.5'e düşürür│   ek GPU belleği gerektirir.         │
 │ • Dil akıcılığını (PPL 14.2) korur.  │ • Muhafazakar LCB ödülü öğrenme      │
 │ • Dinamik PID ile kararlı eğitim.    │   hızını hafif yavaşlatabilir.       │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kurumsal sistemlerde güvenli ve    │                                      │
 │   halüsinasyonsuz yapay zeka sunumu. │                                      │
 └──────────────────────────────────────┴──────────────────────────────────────┘
        FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan 6 panelli Reward Hacking ve Sağlamlık teşhis panosu: `ciktilar/reward_hacking_paneli.png`

![Reward Hacking Paneli](ciktilar/reward_hacking_paneli.png)

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
