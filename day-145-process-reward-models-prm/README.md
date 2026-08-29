# Day 145: Outcome (ORM) vs Process Reward Models (PRM): Adım Adım Mantıksal Doğruluk Puanlama & Best-of-N

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; OpenAI'ın dönüm noktası çalışması **PRM800K (Lightman et al.)** temellerine dayanan **Outcome-supervised Reward Models (ORM)** vs **Process-supervised Reward Models (PRM)**, **Adım Başına Mantıksal Doğruluk Puanlama (Step-level Supervision)** ve **Best-of-N Akıl Yürütme Sıralayıcısı** mimarisini sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Neden Sadece Sonuca Bakmak (ORM) Yetmez? Neden Her Adımı Denetlemeliyiz (PRM)?
- **Outcome Reward Model (ORM) Yanılgısı:**
  ORM sadece bir öğrencinin sınav kağıdındaki sonuca bakar ($100$ üzerinden $100$ veya $0$). Eğer öğrenci ara işlemlerde saçmalamış ama tesadüfen doğru sonucu yazmışsa (**Lucky Guess / False Positive**), ORM buna tam puan verir ($1.0$). Bu durum modelin mantık halüsinasyonlarını ödüllendirir!
- **Process Reward Model (PRM) Çözümü:**
  PRM, titiz bir öğretmen gibi öğrencinin her bir satırını adım adım denetler ($s_1, s_2, s_3, \dots, s_T$). 
  - İlk işlem hatası yapıldığı anda o adıma düşük puan verir ($0.05$).
  - Çözümün toplam PRM skoru tüm adımların çarpımı ($\prod r_t$) olduğu için, ara işlemi hatalı olan şanslı tahminler **derhal elenir**.
  - OpenAI PRM800K araştırmasında Best-of-N matematik başarısı **%72.4'ten (ORM) %92.8'e (PRM)** fırlamıştır!

```
   Soru (x): Beyzbol sopası + top = $1.10 ...
                 │
   [Adım 1: Sopa + Top = 1.10] ──► PRM: 0.98 [OK]
                 │
   [Adım 2: Sopa = Top + 1.00] ──► PRM: 0.98 [OK]
                 │
   [Adım 3: 2*Top = 0.10]      ──► PRM: 0.98 [OK]
                 │
   [Adım 4: Top = $0.05]       ──► PRM: 0.98 [OK]
                 │
   Kümülatif PRM Skoru = 0.98^4 = 0.922 (KUSURSUZ KANIT!)
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: ORM vs PRM Ödül Formülasyonu
- **ORM Ödülü:** $R_{\text{ORM}}(\tau) = P(y = y^* \mid x, \tau)$ (Yalnızca nihai cevap bazlı skaler değer).
- **PRM Ödülü:** Her düşünce adımı $s_t$ için bağımsız $r_t = P(\text{adım } t \text{ doğrudur} \mid x, s_1, \dots, s_t)$.
- **Yol Skorlama:**
  $$\text{Score}_{\text{PRM}}(\tau) = \prod_{t=1}^T r_t \quad \text{veya} \quad \min_{t=1}^T r_t$$

### B. İlk Hata Tespiti (First Error Step Detection)
- PRM, akıl yürütme zincirindeki ilk sapma noktasını $t^* = \arg\min_t \{r_t < \theta\}$ formülüyle tespit eder ve arama motorlarının (ToT, MCTS) o noktada geri izleme (backtracking) yapmasını sağlar.

### C. Best-of-N Re-ranking Dinamiği
- Modelden $N$ adet düşünce yolu üretildiğinde ($\tau_1, \dots, \tau_N$), PRM en yüksek kümülatif güvene sahip zinciri seçer:
  $$\tau^* = \arg\max_{\tau_i} \text{Score}_{\text{PRM}}(\tau_i)$$

### D. Veri Verimliliği ve Aktif Öğrenme (Active Learning in PRM800K)
- PRM modelleri, insan veya sentetik denetçilerin yalnızca modelin belirsiz olduğu adımları etiketlemesi sayesinde ORM'ye göre kat kat daha az veriyle çok daha yüksek genelleme başarımı sunar.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Process Reward Model (PRM)** | Akıl yürütme zincirinin her bir ara adımını bağımsız olarak puanlayan süreç ödül modeli. |
| **Outcome Reward Model (ORM)** | Sadece nihai cevabın doğruluğuna bakarak tüm zincire tek bir ödül veren sonuç modeli. |
| **Step-level Supervision** | Modeli sadece sonuç üzerinden değil, her bir çıkarım adımının doğruluğu üzerinden eğitme yöntemi. |
| **PRM800K** | OpenAI tarafından yayınlanan 800.000 adım etiketli büyük ölçekli matematiksel PRM veri seti. |
| **False Positive / Lucky Guess** | Ara işlem adımları yanlış olduğu halde nihai cevabı tesadüfen doğru çıkan akıl yürütme yolu. |
| **Best-of-N Re-ranking** | Üretilen $N$ farklı çözüm adayının bir ödül modeliyle puanlanıp en iyisinin seçilmesi. |
| **Credit Assignment Problem** | Hatanın zincirin tam olarak hangi adımından kaynaklandığını belirleme zorluğu (PRM çözer). |
| **Cumulative Step Score** | Bir düşünce yolunun tüm ara adım olasılıklarının çarpımıyla elde edilen toplam yol güveni. |
| **Test-Time Verification** | Model çıkarım yaparken üretilen adımların bir doğrulayıcı/ödül modeli ile filtrelenmesi. |
| **MCTS Value Engine** | Monte Carlo Tree Search ağaç aramalarında düğümleri puanlamak için PRM kullanımı. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Şanslı tahminleri derhal eleme ve  │ • Her adım için ayrı çıkarım         │
 │   Best-of-N'de %92.8 doğruluğa ulaşma│   yapıldığından artan test-time GPU  │
 │ • İlk hata noktasını anında tespit   │   maliyeti.                          │
 │   ederek ağaç aramasını yönlendirme. │ • Adım etiketleme verisinin pahalılığı│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • o1 ve DeepSeek-R1 tarzı muhakeme   │ • PRM modelinin kendisinin yanlış    │
 │   modellerinde MCTS/RL yakıtı olma.  │   adım değerlendirmesi yapma riski.  │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/process_reward_models_prm_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
