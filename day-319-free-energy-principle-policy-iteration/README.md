# Day 319: Serbest Enerji Prensibi ile Sürekli Politika İterasyonu (Free Energy Principle & Active Inference)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](#lisans)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-blue.svg?style=flat-square)](https://numpy.org/)
[![Tests Passing](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)

Geleneksel Pekiştirmeli Öğrenme (RL - Reinforcement Learning), dışarıdan elle tasarlanmış skaler ödül fonksiyonlarına (reward engineering) ihtiyaç duyar ve seyrek ödül (sparse reward) ortamlarında rastgele keşif (epsilon-greedy) yaparak tıkanır. Biyolojik beyinler ve otonom süper-zeka sistemleri ise ödül maksimizasyonu yerine **Karl Friston'ın Serbest Enerji Prensibi (Free Energy Principle - FEP)** ve **Aktif Çıkarım (Active Inference)** kuramına göre çalışır.

**Day 319**, algıyı (perception) Varyasyonel Serbest Enerji ($F$) minimizasyonu, eylem planlamasını (action planning) ise Beklenen Serbest Enerji ($G(\pi)$) optimizasyonu olarak formüle eden, epistemik merak (information gain) ile pragmatik hedef sömürüsünü tek bir Bayesyen çerçevede birleştiren bir **Aktif Çıkarım Politika İterasyon Motoru** sunar.

---

## 🏗️ 1. Mimari Tasarım ve Matematiksel Temeller

```
========================================================================================
            SERBEST ENERJİ VE AKTİF ÇIKARIM POLİTİKA DÖNGÜSÜ (DAY 319)
========================================================================================

             Gözlem o_t <----------------+ (Çevre Tepkisi)
                 |                       |
                 v                       |
   [ 1. ALGI: VARYASYONEL SERBEST ENERJİ F ] ---> Sürpriz Sınırını Minimize Et
   q(s_t) = softmax( ln A[o_t, :] + ln(B[:, :, a_{t-1}] q_{t-1}) )
   F = D_KL(q(s_t) || P(s_t)) - E_q[ln P(o_t | s_t)]
                 |
                 v
   [ 2. PLANLAMA: BEKLENEN SERBEST ENERJİ G(pi) ]
   G(pi) = Pragmatik(pi) [Hedef Tercihi C] + Epistemik(pi) [Bilgi Kazanımı / Merak]
                 |
                 v
   [ 3. EYLEM SEÇİMİ: POLİTİKA POSTERİORU ] ----> P(pi) = softmax(-gamma * G(pi))
   a_t = argmax P(pi) ---------------------------+
========================================================================================
```

### Matematiksel Formülasyon

1. **Varyasyonel Serbest Enerji (VFE - Algısal Güncelleme):**
   $$F = \underbrace{D_{\text{KL}}(q(s) \parallel P(s))}_{\text{Karmaşıklık (Complexity)}} - \underbrace{\mathbb{E}_{q(s)}[\ln P(o \mid s)]}_{\text{Doğruluk (Accuracy)}} \ge -\ln P(o) \quad (\text{Sürpriz Üst Sınırı})$$

2. **Beklenen Serbest Enerji (EFE - Eylem Planlama):**
   $$G(\pi) = \underbrace{-\mathbb{E}_{q(o, s \mid \pi)}[\ln P(o)]}_{\text{Pragmatik Değer (Pragmatic / Instrumental)}} - \underbrace{\mathbb{E}_{q(s \mid \pi)}[D_{\text{KL}}(P(o \mid s) \parallel q(o \mid \pi))]}_{\text{Epistemik Bilgi Kazanımı (Epistemic / Curiosity)}}$$

3. **Politika Dağılımı ve Eylem:**
   $$P(\pi) = \sigma(-\gamma G(\pi))$$

---

## 🔬 2. Derinlemesine Mimari Analizler

### Analiz 1: Pekiştirmeli Öğrenme Neden Serbest Enerji Karşısında Yetersiz Kalır?
Standart RL'de ajan ortamı yalnızca skaler ödül $r_t$ aldığında öğrenir; ödül yoksa körlemesine gezinir. Aktif Çıkarımda ise ajan, dünyadaki belirsizliği azaltan (epistemic value) eylemleri matematiksel bir zorunluluk olarak seçer (intrinsic motivation / curiosity).

### Analiz 2: Varyasyonel Serbest Enerji ($F$) ve Homeostaz
Canlı sistemler ve AGI çekirdekleri, çevrelerine karşı sürprizlerini ($-\ln P(o)$) ve entropilerini minimumda tutarak varlıklarını sürdürürler (homeostasis). VFE'nin minimize edilmesi, inanç durumu $q(s)$ ile fiziksel gerçeklik arasındaki uyumsuzluğu giderir.

### Analiz 3: Epistemik Bilgi Kazanımı vs Pragmatik Hedef Sömürüsü Dengesi
EFE denklemi ($G(\pi)$), iki zıt gücü tek bir potansiyelde birleştirir: Ajan durumdan emin değilse $D_{\text{KL}}$ terimi baskın çıkarak ajanı ipucu aramaya yönlendirir (Epistemic Foraging). Belirsizlik çözüldüğünde ise öncelikli tercihler vektörü $C(o)$ devreye girerek ödülü sömürür (Pragmatic Exploitation).

### Analiz 4: Hassasiyet ($\gamma$) Parametresinin Rolü
Politika seçimindeki $\gamma$ (precision), ajanın kendi tahminlerine ve ortamın gürültüsüne duyduğu güveni temsil eder. Yüksek $\gamma$, ajanı hedefe deterministik odaklarken, düşük $\gamma$ kaotik ve stokastik ortamlarda aşırı özgüveni engeller.

---

## 📊 3. 6-Panelli Teşhis Panosu İncelemesi

Modül çalıştırıldığında `ciktilar/serbest_enerji_paneli.png` konumunda üretilen 6 teşhis paneli:

1. **Algısal Sürpriz ve Serbest Enerji Azalımı:** Zaman adımları boyunca $F(t)$'nin düşüş grafiği.
2. **Epistemik Keşif vs Pragmatik Sömürü:** Toplam $G(\pi)$, epistemik bilgi kazanımı ve pragmatik değer eğrileri.
3. **Gizli Durum İnanç Belirsizliği:** Shannon entropisi $H(q(s))$'nin sıfıra inerek inancın kesinleşmesi.
4. **Ajan Durum Geçiş Rotası:** Başlangıç durumundan hedefe uzanan durum geçiş adımları.
5. **Üretici Model Olabilirlik Matrisi ($A = P(o \mid s)$):** Olasılık dağılım haritası.
6. **FEP Politika İterasyon Raporu:** Hedef başarısı, epistemik kazanç ve FEP sınıflandırma özeti.

---

## 📖 4. Terimler Sözlüğü (Glossary)

- **Free Energy Principle (FEP):** Kendi kendini organize eden tüm adaptif sistemlerin serbest enerjilerini minimize ettiği evrensel fiziksel prensip.
- **Active Inference (Aktif Çıkarım):** Algı ve eylemin aynı serbest enerji fonksiyonunu minimize etmek için birlikte çalıştığı Bayesyen karar kuramı.
- **Variational Free Energy (VFE):** Algısal çıkarımda sürprizin üst sınırını oluşturan ve inanç güncellemesinde kullanılan metrik ($F$).
- **Expected Free Energy (EFE):** Gelecekteki olası politikaların doğuracağı sürpriz ve belirsizliği puanlayan planlama fonksiyonu ($G(\pi)$).
- **Epistemic Value:** Bilgi kazanımı, merak ve ortamdaki belirsizliği azaltma değeri.
- **Pragmatic Value:** Ajanın önceden tanımlanmış hedeflerine ve hayatta kalma tercihlerine ulaşma derecesi.
- **Generative Model (Üretici Model):** Ajanın çevrenin gizli durumları ($S$) ve gözlemleri ($O$) hakkındaki içsel olasılıksal haritası ($A, B, C, D$).
- **Precision ($\gamma$):** Ajanın inançlarına veya politika dağılımına atadığı güven katsayısı / ters varyans.
- **Homeostasis (Homeostaz):** Bir sistemin iç dengesini ve düşük entropili varlığını koruma süreci.
- **Belief State ($q(s)$):** Ajanın dünyanın gizli durumları hakkındaki güncel olasılık dağılımı.

---

## ⚖️ 5. SWOT Analizi

```
+----------------------------------------------------+----------------------------------------------------+
| 🟢 GÜÇLÜ YÖNLER (STRENGTHS)                        | 🟡 ZAYIF YÖNLER (WEAKNESSES)                       |
| • Elle ödül mühendisliği ihtiyacını ortadan        | • Büyük durum uzaylarında (continuous high-dim)    |
|   kaldıran doğal içsel motivasyon (curiosity)      |   matris çarpımlarının hesaplama yükü              |
| • Epistemik keşif ve pragmatik sömürüyü birleştirme| • B matrisinin doğru modellenmesi gerekliliği      |
| • Matematiksel olarak kanıtlanmış homeostaz        |                                                   |
+----------------------------------------------------+----------------------------------------------------+
| 🔵 FIRSATLAR (OPPORTUNITIES)                       | 🔴 TEHDİTLER (THREATS)                             |
| • Otonom uzay araçları ve bilinmeyen gezegen       | • Kötü yapılandırılmış C tercihlerinde ajanın      |
|   keşiflerinde kendi kendine öğrenme               |   yerel serbest enerji çukurlarına sıkışması       |
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
pytest testler/test_serbest_enerji.py -v
```

---

## 👨‍🏫 7. Senior AI / SRE Mentor Soru-Cevap

**Soru 1: Q-Learning / PPO ile Aktif Çıkarım arasındaki en temel felsefi ve pratik fark nedir?**  
*Cevap:* PPO ve Q-Learning hedef ödülü maksimize etmeye çalışır ($R \to \infty$). Aktif Çıkarım ise sürprizi ve entropiyi minimize etmeye çalışır ($F \to 0$). Aktif çıkarım ajanı, ortamı anlamayı (epistemic value) hayatta kalmanın ve hedefe ulaşmanın ön koşulu olarak görür.

**Soru 2: Epistemik merak (curiosity) ajanın sonsuza kadar gereksiz detayları incelemesine yol açmaz mı ("Dark Room Problem")?**  
*Cevap:* Hayır. Karl Friston'ın yanıtladığı gibi, ajanın üretici modelinde $C$ vektörü (aç kalmama, donmama, hedefe ulaşma tercihleri) bulunur. Ajan karanlık odada kaldığında içsel serbest enerjisi zamanla fırlar; bu yüzden karanlık odayı terk etmek zorunda kalır.

**Soru 3: Derin Sinir Ağları ile Aktif Çıkarım nasıl birleştirilir (Deep Active Inference)?**  
*Cevap:* Ayrık $A$ ve $B$ matrisleri yerine VAE (Variational Autoencoder) ve RNN/Transformer latent modelleri kullanılır; serbest enerji diferansiyellenebilir kayıp olarak geri yayılımla (backprop) optimize edilir.

---

## 📜 Lisans

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Bu modül eğitim ve araştırma amaçlıdır. İzinsiz kopyalanamaz ve ticari amaçla kullanılamaz.
