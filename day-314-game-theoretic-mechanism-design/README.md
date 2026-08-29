# Day 314: Oyun Teorik Mekanizma Tasarımı ve Çoklu Ajan Nash Pazarlığı

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](#lisans)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Scipy](https://img.shields.io/badge/SciPy-Optimization-blue.svg?style=flat-square)](https://scipy.org/)
[![Tests Passing](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)

Otonom süper-zeka ve çoklu ajan ekosistemlerinde (Multi-Agent Clusters), ajanlar bencil (self-interested) çıkarlara sahip olduğunda, ortak GPU/hesaplama kaynaklarını tüketmek veya kolektif kararlar almak için hileli teklifler (misreporting bids) vererek sistemi çöküşe (Tragedy of the Commons) sürükleyebilirler.

**Day 314**, mikroekonomi ve algoritmik oyun teorisinin en güçlü iki sütununu bir araya getirir:
1. **Vickrey-Clarke-Groves (VCG) Mekanizması:** Ayrık kaynak dağıtımında dürüst teklif vermeyi her ajan için kesin matematiksel baskın strateji (Dominant-Strategy Incentive Compatibility - DSIC) haline getiren dışsallık vergilendirmesi.
2. **Genelleştirilmiş Nash Pazarlık Çözümü (Generalized Nash Bargaining):** Sürekli hesaplama kaynağı bölüşümünde tehdit noktalarını ($d_i$) ve pazarlık güçlerini ($\alpha_i$) dikkate alarak **%100 Pareto etkinliği** ve adil rant artışı sağlayan dışbükey optimizasyon motoru.

---

## 🏗️ 1. Mimari Tasarım ve Matematiksel Temeller

```
========================================================================================
             OYUN TEORİK MEKANİZMA VE NASH PAZARLIK MİMARİSİ (DAY 314)
========================================================================================

   Ajan Değerlemeleri {v_1, ..., v_M}
            |
            +-----> [ VCG Mekanizması (Ayrık Karar) ]
            |       |-- Sosyal Refah Maksimizasyonu: k* = argmax sum(v_j)
            |       \-- Clarke Pivot Dışsallık Vergisi: p_i = max sum_{j!=i} v_j(k) - sum_{j!=i} v_j(k*)
            |            ==> Dürüstlük Baskın Stratejidir (DSIC Garantisi)
            |
            \-----> [ Genelleştirilmiş Nash Pazarlığı (Sürekli Kaynak) ]
                    |-- Kısıt: sum(x_i) <= C (Kapasite) & u_i(x_i) >= d_i (Bireysel Rasyonellik)
                    \-- Amaç: max prod_i (u_i(x_i) - d_i)^{alpha_i}
                         ==> %100 Pareto Etkin ve Adil Artık Rant Dağılımı
========================================================================================
```

### Matematiksel Formülasyon

1. **VCG Sosyal Refah Maksimizasyonu ve Dışsallık Vergisi:**
   $$k^* = \arg\max_{k \in \mathcal{K}} \sum_{j=1}^M b_j(k)$$
   $$p_i = \max_{k \in \mathcal{K}} \sum_{j \ne i} b_j(k) - \sum_{j \ne i} b_j(k^*)$$
   $$u_i = v_i(k^*) - p_i$$

2. **Genelleştirilmiş Nash Pazarlık Çözümü (NBS):**
   $$\max_{\mathbf{x}} \sum_{i=1}^M \alpha_i \ln\left(u_i(x_i) - d_i\right) \quad \text{s.t.} \quad \sum_{i=1}^M x_i \le C, \quad u_i(x_i) \ge d_i$$

---

## 🔬 2. Derinlemesine Mimari Analizler

### Analiz 1: Bencil Ajanlar Neden Dağıtımı Manipüle Eder?
Ajanlar yalnızca kendi çıktısını maksimize etmeye programlandığında, standart açık artırmalarda değerlemelerini şişirerek (overbidding) diğer ajanların meşru görevlerini aç bırakırlar (starvation). VCG mekanizması, bir ajanın sisteme girmesiyle diğer ajanların kaybettiği refahı (dışsallık / externality) o ajana vergi olarak ödeterek hile yapma motivasyonunu sıfırlar.

### Analiz 2: Clarke Pivot Kuralı ve Dürüstlüğün Matematiksel İspatı
Ajan $i$'nin net faydası:
$$u_i(b_i, b_{-i}) = v_i(k^*(b_i, b_{-i})) + \sum_{j \ne i} b_j(k^*(b_i, b_{-i})) - \max_k \sum_{j \ne i} b_j(k)$$
Bu denklemde ajan $i$, kendi $b_i$ teklifiyle yalnızca $k^*$ seçimini etkiler; ancak seçilen $k^*$ zaten $v_i(k) + \sum_{j \ne i} b_j(k)$ toplamını maksimize eder. Dolayısıyla $b_i = v_i$ söylemek, ajanın faydasını her koşulda global tepe noktasına taşır ($u_{\text{truthful}} \ge u_{\text{lying}}$).

### Analiz 3: Sürekli Kaynak Paylaşımında Nash Pazarlığı ve Tehdit Noktaları
Ajanların anlaşmaya varamadığı durumda elde edeceği asgari getiri $d_i$ (threat point / disagreement point) olarak tanımlanır. Nash optimizasyonu, hiç kimseyi tehdit noktasının altında bırakmayacak şekilde ($u_i(x_i) > d_i$) toplam refah rantını logaritmik olarak çarparak paylaştırır.

### Analiz 4: Çoklu Ajan Kümelerinde İletişim Maliyeti ve Ölçeklenebilirlik
Yüzlerce ajandan oluşan dağıtık hesaplama kümelerinde merkezi VCG optimizasyonu $O(M \cdot K)$ karmaşıklığıyla mikrosaniyeler içinde çözülürken, sürekli Nash pazarlığı SLSQP kuadratik programlama ile birkaç iterasyonda küresel optimuma yakınsar.

---

## 📊 3. 6-Panelli Teşhis Panosu İncelemesi

Modül çalıştırıldığında `ciktilar/oyun_teorisi_paneli.png` konumunda üretilen 6 teşhis paneli:

1. **VCG Net Ajan Faydaları:** Her ajanın kazandığı net değerleme ($u_i = v_i - p_i$).
2. **VCG Teşvik Uyumlu Dışsallık Ödemeleri:** Ajanların sisteme yüklediği dışsallık maliyeti ($p_i$).
3. **DSIC Baskın Strateji Doğrulaması:** Dürüst teklif ile manipülatif teklif arasındaki kazanç kıyası ($+0.00$ kayıp riski).
4. **Nash Pazarlığı Hesaplama Kaynağı Dağılımı:** 100 TFLOPS kapasitenin ajan güçlerine göre pasta dağılımı.
5. **Tehdit Noktası Üzerindeki Bireysel Artıklar:** Her ajanın tehdit noktası $d_i$ üzerindeki net kazancı ($+8.67$ ile $+14.77$ TFLOPS eşdeğeri).
6. **Oyun Teorisi & Mekanizma Özeti:** Sosyal refah, Pareto etkinliği ve denge sınıfı konsolide raporu.

---

## 📖 4. Terimler Sözlüğü (Glossary)

- **Mechanism Design (Mekanizma Tasarımı):** İstenen bir sosyal/sistemik sonuca ulaşmak için oyunun kurallarını ve teşvik yapılarını tasarlayan tersine oyun teorisi alanı.
- **VCG Mechanism:** Sosyal refahı maksimize eden ve dürüstlüğü baskın strateji kılan Vickrey-Clarke-Groves açık artırma protokolü.
- **Incentive Compatibility (DSIC):** Diğer oyuncuların ne yaptığına bakılmaksızın doğruyu söylemenin en yüksek getiriyi sağladığı durum.
- **Externality (Dışsallık):** Bir ajanın kararının diğer tüm ajanların refahı üzerinde yarattığı pozitif veya negatif etki.
- **Clarke Pivot Rule:** Ajanların ödemesini, onların yokluğunda diğer ajanların kazanacağı maksimum refaha bağlayan vergilendirme kuralı.
- **Nash Bargaining Solution (NBS):** İşbirlikçi oyun teorisinde tehdit noktalarına göre adil ve aksiyomatik kaynak bölüşümü sağlayan çözüm.
- **Threat Point ($d_i$):** Pazarlık çökerse ajanın çekileceği rezervasyon / asgari hayatta kalma faydası.
- **Pareto Efficiency:** Hiçbir ajanın durumunu kötüleştirmeden başka bir ajanın durumunu iyileştirmenin imkansız olduğu optimal durum.
- **Social Welfare:** Sistemdeki tüm ajanların elde ettiği toplam brüt değerlemelerin toplamı ($\sum v_i$).
- **Nash Product:** Tüm ajanların net rant fazlalıklarının ağırlıklı geometrik çarpımı ($\prod (u_i - d_i)^{\alpha_i}$).

---

## ⚖️ 5. SWOT Analizi

```
+----------------------------------------------------+----------------------------------------------------+
| 🟢 GÜÇLÜ YÖNLER (STRENGTHS)                        | 🟡 ZAYIF YÖNLER (WEAKNESSES)                       |
| • Matematiksel olarak kanıtlanmış DSIC dürüstlüğü  | • VCG mekanizmasının gelir garantisi vermemesi     |
| • %100 Pareto etkinliği ve kaynak tam kullanımı    |   (bazen negatif veya düşük vergi toplanabilir)    |
| • Çatışan hedeflere sahip ajanlarda sıfır kilitlenme| • Ajan sayısı $10^5$ olduğunda optimizasyon yükü  |
+----------------------------------------------------+----------------------------------------------------+
| 🔵 FIRSATLAR (OPPORTUNITIES)                       | 🔴 TEHDİTLER (THREATS)                             |
| • Dağıtık GPU/TPU bulutlarında otonom kaynak takası| • Ajanların gizli koalisyonlar kurarak (collusion) |
| • Ajanlar arası API ve veri pazaryeri protokolleri |   fiyat kırma riski                                |
+----------------------------------------------------+----------------------------------------------------+
```

---

## 🚀 6. Hızlı Başlangıç

### Kurulum
```bash
pip install -r gereksinimler.txt
```

### Ana Akışı Çalıştırma
```bash
python ana_akis.py
```

### Testleri Çalıştırma (8/8 Unit Test)
```bash
pytest testler/test_oyun_teorisi.py -v
```

---

## 👨‍🏫 7. Senior AI / SRE Mentor Soru-Cevap

**Soru 1: VCG Mekanizmasında bir ajan bütçe kısıtına sahipse (Budget-Constrained Agents) dürüstlük garantisi bozulur mu?**  
*Cevap:* Evet. Klasik VCG quasilinear (doğrusal para) fayda varsayar. Ajanların ödeme yapacak sınırlı bütçesi varsa, "Clinched Auction" veya bütçe kısıtlı uyarlamalı mekanizmalar kullanılarak dürüstlük ve refah dengesi korunur.

**Soru 2: Nash Pazarlığı ile Shapley Değeri (Shapley Value) arasındaki fark nedir?**  
*Cevap:* Shapley değeri, koalisyonel oyunlarda her ajanın marjinal katkısını ölçerek ödülü paylaştırır. Nash Pazarlığı ise tehdit noktalarını ($d_i$) ve pazarlık güçlerini dikkate alarak iki veya daha fazla ajanın anlaşma alanındaki (bargaining set) ortak artık rantı aksiyomatik olarak bölüşmesini sağlar.

**Soru 3: Dağıtık yapay zeka ajan kümelerinde VCG ve Nash protokolleri nerede kullanılır?**  
*Cevap:* VCG, yüksek öncelikli eğitim görevlerinin hangi GPU düğümlerinde çalıştırılacağını belirlemek için kullanılırken; Nash Pazarlığı, sürekli çıkarım yapan LLM servisleri arasında dinamik bant genişliği ve KV-Cache VRAM paylaşımında kullanılır.

---

## 📜 Lisans

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Bu modül eğitim ve araştırma amaçlıdır. İzinsiz kopyalanamaz ve ticari amaçla kullanılamaz.
