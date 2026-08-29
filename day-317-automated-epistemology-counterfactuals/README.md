# Day 317: Otonom Epistemoloji — Karşı-Olgusal Hipotez Test Laboratuvarı (Automated Epistemology & Counterfactuals)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](#lisans)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-blue.svg?style=flat-square)](https://numpy.org/)
[![Tests Passing](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)

Standart derin öğrenme modelleri ve LLM'ler, Judea Pearl'ün **Nedensellik Merdiveni'nin (Causal Hierarchy)** yalnızca en alt basamağında (Seviye 1: İstatistiki Korelasyon ve Gözlem) çalışırlar. Bu durum, modellerin karıştırıcı değişkenleri (confounders) gerçek nedensellik sanarak ölümcül yanılsamalara (spurious correlation) kapılmasına ve *"Eğer bu kararı almasaydık ne olurdu?"* gibi karşı-olgusal (counterfactual) sorulara cevap verememesine neden olur.

**Day 317**, Yapısal Nedensel Modeller (Structural Causal Models - SCM), do-calculus müdahale operatörleri ve Pearl'ün 3-adımlı **Kaçırma-Müdahale-Tahmin (Abduction-Action-Prediction)** algoritmasını kullanarak Otonom Süper-Zekaya (ASI) gerçek epistemolojik akıl yürütme yeteneği kazandıran bir karşı-olgusal deney motoru sunar.

---

## 🏗️ 1. Mimari Tasarım ve Matematiksel Temeller

```
========================================================================================
             YAPISAL NEDENSEL MODEL VE KARŞI-OLGUSAL MİMARİ (DAY 317)
========================================================================================

   [ Gözlenen Olgusal Gerçeklik ] (z, x, m, y)
                  |
                  v
   [ 1. ADIM: KAÇIRMA (ABDUCTION) ] ---------> Gözlenemeyen Dışsal Gürültüleri Çıkarsama
   u_Z = z,  u_X = x - 0.8*z,  u_M = m - 1.2*x,  u_Y = y - (0.5*z + 1.5*m + 0.4*x)
                  |
                  v
   [ 2. ADIM: MÜDAHALE (ACTION) ] -----------> do(X = x') Hipotezi (Gelen Okları Kes)
                  |
                  v
   [ 3. ADIM: TAHMİN (PREDICTION) ] ---------> Karşı-Olgusal Evrenin Çıktısını Hesapla
   m_CF = 1.2 * x' + u_M
   y_CF = 0.5 * z + 1.5 * m_CF + 0.4 * x' + u_Y  ==> Y_{X=x'}(u) (Karşı-Olgusal Sonuç)
========================================================================================
```

### Pearl'ün 3-Basamaklı Nedensellik Hiyerarşisi

| Seviye | Adı | Matematiksel Gösterim | Temel Soru | Yapay Zeka Yeteneği |
|:---|:---|:---|:---|:---|
| **L1** | **Gözlem (Association)** | $P(y \mid x)$ | *"X'i görürsem Y ne olur?"* | Klasik LLM / Sınıflandırma |
| **L2** | **Müdahale (Intervention)** | $P(y \mid \text{do}(x))$ | *"X'i zorla değiştirirsem Y ne olur?"* | Aktif Ajan / A/B Testi |
| **L3** | **Karşı-Olgusal (Counterfactuals)** | $P(y_{x'} \mid x, y)$ | *"X'i yapmıştık ve Y oldu; peki X' yerine X=0 yapsaydık ne olurdu?"* | Epistemolojik Akıl Yürütme (ASI) |

### Nedensel Yol Ayrışımı (Mediation Analysis)

1. **Ortalama Tedavi Etkisi (ATE):**
   $$\text{ATE} = \mathbb{E}[Y \mid \text{do}(X=1)] - \mathbb{E}[Y \mid \text{do}(X=0)] = 2.20$$

2. **Doğal Doğrudan Etki (Natural Direct Effect - NDE):**
   $$\text{NDE} = \mathbb{E}[Y_{X=1, M=M(0)} - Y_{X=0, M=M(0)}] = 0.40$$

3. **Doğal Dolaylı Etki (Natural Indirect Effect - NIE):**
   $$\text{NIE} = \mathbb{E}[Y_{X=1, M=M(1)} - Y_{X=1, M=M(0)}] = 1.80 \quad (\text{Aracı: } X \to M \to Y)$$

---

## 🔬 2. Derinlemesine Mimari Analizler

### Analiz 1: Makine Öğrenimi Neden Nedensellik Olmadan Çuvallar? (Confounding Bias)
Gözlemsel veride $Z$ değişkeni (örneğin hava sıcaklığı veya gizli risk faktörü) hem $X$'i hem de $Y$'yi aynı anda etkilediğinde, korelasyon katsayısı $E[Y|X] = 3.8036$ olarak şişer. Model, $X$'in tek başına sonucu uçurduğunu sanır. do-Calculus müdahalesi uygulandığında karıştırıcı $Z$'nin etkisi nötralize edilir ve saf nedensel etki $\text{ATE} = 2.2000$ olarak saptanır ($1.6036$ yanlılık boşluğu elenir).

### Analiz 2: Nedensel Merdivenin 3. Basamağı: Bireysel Karşı-Olgusallık
Bir otonom cerrahi robot veya otonom savunma sistemi, gerçekleşmiş tek bir vakayı ($N=1$) inceleyip *"Bu hastaya A ilacı yerine B ilacını verseydik iyileşir miydi?"* sorusunu yanıtlayabilmelidir. Bu seviye, evrenin o ana özgü gizli arka plan gürültüsünü ($\mathbf{u}$) geriye dönük kaçırma (abduction) ile bulup yeni bir paralel evren simülasyonu çalıştırmayı gerektirir.

### Analiz 3: Doğal Doğrudan ve Dolaylı Etki Ayrışımı (Mediation Analysis)
$X$ kararının sonucu $Y$ üzerindeki etkisinin ne kadarının doğrudan ($X \to Y$), ne kadarının ise $M$ aracısı ($X \to M \to Y$) üzerinden aktığını matematiksel olarak ayrıştırmak, modelin mekanistik nedensellik zincirini şeffaf şekilde kanıtlamasını sağlar ($\text{NDE} = 0.40$, $\text{NIE} = 1.80$).

### Analiz 4: Tutarlılık Aksiyomu (Consistency Axiom: $Y_{X=x}(u) \equiv Y(u)$)
Eğer bir bireye uygulanan hipotetik müdahale zaten o bireyin gerçekte yaşadığı durumla aynıysa ($x' = x$), karşı-olgusal modelin tahmin ettiği sonuç bireyin gerçek sonucuyla ($y$) birebir aynı olmak zorundadır. Motorumuz bu aksiyomu **%100 kesinlikle** doğrular.

---

## 📊 3. 6-Panelli Teşhis Panosu İncelemesi

Modül çalıştırıldığında `ciktilar/epistemoloji_paneli.png` konumunda üretilen 6 teşhis paneli:

1. **Pearl'ün 3-Basamaklı Nedensellik Hiyerarşisi:** Gözlem ($3.804$), Müdahale ATE ($2.200$) ve Karşı-Olgusallık seviyelerinin nicel büyüklükleri.
2. **Nedensel Yol Ayrışımı (Mediation Analysis):** Toplam ATE'nin NDE ($0.40$) ve NIE ($1.80$) bileşenlerine tam ayrışımı.
3. **do-Calculus Müdahale Tepki Eğrisi:** Farklı müdahale değerlerinde $E[Y|\text{do}(X=x)]$ doğrusal tepkisi.
4. **Bireysel Karşı-Olgusal Akıl Yürütme:** 4 gerçek vakanın hipotetik $X=0$ karşı-olgusal sonuçları ve bireysel tedavi etkileri.
5. **Karıştırıcı Değişken Yanlılık Boşluğu:** Gözlem yanlılığı ($1.604$) ile saf nedensellik ($2.200$) kıyası.
6. **Epistemolojik Nedensellik Modeli Özeti:** Tutarlılık skoru (%100), SCM parametreleri ve seviye 3 epistemoloji raporu.

---

## 📖 4. Terimler Sözlüğü (Glossary)

- **Structural Causal Model (SCM):** Değişkenler arasındaki sebep-sonuç ilişkilerini deterministik fonksiyonlar ve dışsal gürültülerle tanımlayan yapısal model.
- **Pearl's Causal Hierarchy:** Nedensel akıl yürütmeyi Gözlem (L1), Müdahale (L2) ve Karşı-Olgusallık (L3) olarak 3 katmana ayıran kuram.
- **do-Calculus:** Nedensel yönlendirilmiş çizgelerde müdahalelerin ($do(X)$) olasılıksal karşılığını türeten cebirsel kurallar bütünü.
- **Abduction (Kaçırma):** Gözlenen olgusal verilerden, sisteme etki eden gözlenemez dışsal arka plan faktörlerini ($U$) geriye dönük hesaplama adımı.
- **Average Treatment Effect (ATE):** Bir müdahalenin tüm popülasyon üzerindeki beklenen ortalama saf nedensel etkisi.
- **Natural Direct Effect (NDE):** Aracı değişkenler sabit tutulduğunda müdahalenin hedef üzerindeki doğrudan nedensel etkisi.
- **Natural Indirect Effect (NIE):** Müdahalenin aracı değişken üzerinden hedefe dolaylı olarak aktardığı nedensel etki.
- **Confounder (Karıştırıcı Değişken):** Hem sebebi hem de sonucu ortak etkileyerek sahte korelasyon üreten ortak ata düğüm.
- **Spurious Correlation:** Aralarında hiçbir nedensel bağ olmayan iki değişkenin karıştırıcılar nedeniyle birlikte hareket etmesi.
- **Consistency Axiom:** Bir bireyin gerçekte maruz kaldığı müdahalenin karşı-olgusal çıktısının gerçek sonuçla özdeş olması aksiyomu.

---

## ⚖️ 5. SWOT Analizi

```
+----------------------------------------------------+----------------------------------------------------+
| 🟢 GÜÇLÜ YÖNLER (STRENGTHS)                        | 🟡 ZAYIF YÖNLER (WEAKNESSES)                       |
| • Seviye 3 karşı-olgusal akıl yürütme yeteneği     | • Nedensel çizgenin (DAG) doğru modellenmesi       |
| • Karıştırıcı değişken yanlılıklarını sıfırlama    |   veya uzman bilgisi gerektirmesi                  |
| • %100 aksiyomatik tutarlılık güvencesi            | • Yüksek boyutlu karmaşık non-lineer SCM'lerde     |
| • Şeffaf doğrudan/dolaylı etki ayrışımı            |   kaçırma (abduction) adımı optimizasyon yükü      |
+----------------------------------------------------+----------------------------------------------------+
| 🔵 FIRSATLAR (OPPORTUNITIES)                       | 🔴 TEHDİTLER (THREATS)                             |
| • Otonom tıp, hukuk ve sigortacılıkta kök-neden    | • Gözlenmeyen karıştırıcıların (unobserved         |
|   analizi ve kusur tespiti                         |   confounders) latent modelleri saptırma riski     |
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
pytest testler/test_epistemoloji_karsiolgusal.py -v
```

---

## 👨‍🏫 7. Senior AI / SRE Mentor Soru-Cevap

**Soru 1: Standart bir LLM'e *"Eğer X olmasaydı ne olurdu?"* diye sorduğumuzda karşı-olgusal akıl yürütme yapmış olmuyor mu?**  
*Cevap:* Hayır. LLM'ler internetteki metinlerde geçen olası hikaye devamlarını (Seviye 1 koşullu olasılık $P(\text{metin} \mid \text{soru})$) otoregresif olarak tahmin eder. Arka plandaki yapısal nedensel denklemleri çözmediği için dışsal gürültüleri kaçıramaz (no abduction) ve karıştırıcı yanlılıklarına teslim olur.

**Soru 2: Do-Calculus ile Karşı-Olgusal (Counterfactual) arasındaki temel fark nedir?**  
*Cevap:* Do-calculus popülasyon düzeyinde çalışır: *"Gelecekte herkese X ilacı verirsek ortalama ölüm oranı ne olur?"* (Seviye 2). Karşı-Olgusal analiz ise birey düzeyinde çalışır: *"Dün ameliyatta vefat eden Ahmet Bey'e B ilacı verilmiş olsaydı bugün hayatta kalır mıydı?"* (Seviye 3).

**Soru 3: Gözlenemeyen karıştırıcı değişkenler (unobserved confounders) olduğunda karşı-olgusal çıkarım nasıl kurtarılır?**  
*Cevap:* Araç değişkenler (Instrumental Variables - IV), Front-Door Kriteri veya Duyarlılık Analizi (Sensitivity Bounds) kullanılarak nedensel etkiler ve karşı-olgusal aralıklar kesin matematiksel sınırlar içine hapsedilir.

---

## 📜 Lisans

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Bu modül eğitim ve araştırma amaçlıdır. İzinsiz kopyalanamaz ve ticari amaçla kullanılamaz.
