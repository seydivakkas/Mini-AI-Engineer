# Day 318: Nöro-Sembolik Sürekli Mantık ve Bulanık Teorem Doğrulama (Neuro-Symbolic Continuous Logic & Differentiable Theorem Proving)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](#lisans)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-blue.svg?style=flat-square)](https://numpy.org/)
[![Tests Passing](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)

Klasik derin öğrenme modelleri (LLM'ler ve sinir ağları) yüksek boyutlu kalıp tanımada (pattern recognition) mükemmel olmalarına rağmen, kesin mantıksal kuralları (First-Order Logic - FOL) aksiyomatik olarak takip edemezler; sık sık kendi kendileriyle çelişir ve mantıksal halüsinasyonlar üretirler. Klasik sembolik yapay zeka (Prolog, SAT çözücüler) ise katı boole mantığı nedeniyle gürültülü gerçek dünya verilerine uyum sağlayamaz ve gradyan tabanlı öğrenilemez.

**Day 318**, mantıksal önermeleri sürekli $[0, 1]$ aralığına eşleyen **Üçgensel Normlar (T-Norms: Product, Łukasiewicz, Gödel)** ve türetilebilir geriye zincirleme (**Differentiable Soft Backward-Chaining**) algoritmalarını birleştirerek sinir ağlarının hem veriyle eğitilmesini hem de katı mantıksal aksiyomları kanıtlamasını sağlayan bir **Nöro-Sembolik Teorem Doğrulama Motoru** sunar.

---

## 🏗️ 1. Mimari Tasarım ve Matematiksel Temeller

```
========================================================================================
             NÖRO-SEMBOLİK SÜREKLİ MANTIK VE TEOREM KANITLAYICI (DAY 318)
========================================================================================

   Varlık Gömümleri (e_i in R^D) & İlişki Matrisleri (W_P, W_A in R^{D x D})
                               |
                               v
   [ 1. SÜREKLİ ÖNERME SKORLAMA ] ----------> Pred(h, t) = sigmoid(e_h^T W e_t / temp)
                               |
                               v
   [ 2. TÜRETİLEBİLİR GERİYE ZİNCİRLEME ] --> Ancestor(X, Z) = Base(P(X, Z)) v
                                             Max_Y [ P(X, Y) ^ Ancestor(Y, Z) ]
                               |
                               v
   [ 3. T-NORM MANTIKSAL BAĞLAÇLAR ] -------> Łukasiewicz T-Norm:
                                             T(a, b) = max(0, a + b - 1)
                                             S(a, b) = min(1, a + b)
                                             I(a, b) = min(1, 1 - a + b)
                               |
                               v
   [ 4. BİRLEŞİK KAYIP OPTİMİZASYONU ] ------> L = L_task + lambda * (1 - tau(Aksiyomlar))
========================================================================================
```

### T-Norm Sürekli Mantık Çerçeveleri

| İşlem | Product T-Norm | Łukasiewicz T-Norm | Gödel T-Norm |
|:---|:---|:---|:---|
| **Ve (Conjunction $T(a,b)$)** | $a \cdot b$ | $\max(0, a + b - 1)$ | $\min(a, b)$ |
| **Veya (Disjunction $S(a,b)$)** | $a + b - a b$ | $\min(1, a + b)$ | $\max(a, b)$ |
| **İse (Implication $I(a,b)$)** | $\min(1, b / a)$ | $\min(1, 1 - a + b)$ | $1 \text{ if } a \le b \text{ else } b$ |
| **Değil (Negation $N(a)$)** | $1 - a$ | $1 - a$ | $1 - a$ |

---

## 🔬 2. Derinlemesine Mimari Analizler

### Analiz 1: Sembolik ile Sinirsel Arasındaki Uçurum Neden Kapanmalıdır?
Salt sinirsel modeller (pure neural), $A \implies B$ ve $B \implies C$ gerçeklerini ezberlese bile $A \implies C$ sonucunu tutarlı şekilde türetemeyebilir. Nöro-sembolik sürekli mantık, bilgi çizgesindeki ilişkileri gradyanla öğrenirken aynı zamanda mantıksal geçişlilik (transitivity) kuralını doğrudan kayıp fonksiyonuna ($\mathcal{L}_{\text{logic}}$) enjekte eder.

### Analiz 2: Neden Łukasiewicz T-Normu Tercih Edilir?
Product T-Norm uzun kanıtlama zincirlerinde ($0.8 \times 0.8 \times 0.8 = 0.512$) hızla sıfıra yaklaşarak kaybolan gradyan (vanishing gradient) üretir. Gödel T-Normu ise türevlenemeyen basamak fonksiyonlarına sahiptir. Łukasiewicz T-Normu, parçalı doğrusal yapısıyla gradyan akışını ($dL/da = \pm 1$) kusursuz ileterek kararlı eğitim sağlar.

### Analiz 3: Türetilebilir Geriye Zincirleme (Soft Backward Chaining)
Bir $Ancestor(X, Z)$ sorgusu geldiğinde motor, latent uzayda olası tüm aracı $Y$ varlıkları üzerinden yumuşak birleşim ($T(P(X, Y), A(Y, Z))$) hesaplar. Böylece sembolik Prolog benzeri bir ispat ağacı gradyan tabanlı tensör matris çarpımlarıyla $O(N)$ karmaşıklığında yürütülür.

### Analiz 4: Aksiyom İhlal Cezası ve Kanıtlama Başarısı
Modelimiz 50 eğitim adımında toplam kaybı $0.1805$'e indirirken aksiyom sağlama oranını **%86.11** seviyesine çıkarmış ve görünmeyen test sorgularını (örneğin Alice $\to$ Dave atalık ilişkisini) **%100 kanıtlama doğruluğu** ile çözmüştür.

---

## 📊 3. 6-Panelli Teşhis Panosu İncelemesi

Modül çalıştırıldığında `ciktilar/noro_sembolik_paneli.png` konumunda üretilen 6 teşhis paneli:

1. **Nöro-Sembolik Gradyan Optimizasyonu:** Birleşik kaybın $0.1805$'e iniş grafiği.
2. **Mantıksal Kurallara Uyum Gelişimi:** Ortalama aksiyom doğrulanma oranının %86.11'e yükselişi.
3. **FOL Aksiyom Skorları:** Taban Kural (%91.6), Geçişlilik (%92.9) ve Asimetri (%73.8) başarıları.
4. **Türetilebilir Teorem Kanıtlama Başarısı:** Test sorgularının ($Ancestor(0, 2)$, $Ancestor(0, 3)$, vb.) $1.000$ doğruluk eşiğini aşması (%100 başarı).
5. **T-Norm Sürekli Mantık Bağlaçları Kıyası:** Product, Łukasiewicz ve Gödel normlarının kesişim dinamikleri.
6. **Nöro-Sembolik Çıkarım Motoru Özeti:** Teorem doğruluğu (%100), T-Norm tipi ve mantıksal seviye konsolide raporu.

---

## 📖 4. Terimler Sözlüğü (Glossary)

- **Neuro-Symbolic AI:** Sinir ağlarının öğrenme gücü ile sembolik mantığın kesinliğini birleştiren yapay zeka paradigması.
- **Continuous Logic (Sürekli Mantık):** Doğruluk değerlerini $\{0, 1\}$ ayrık kümesi yerine $[0, 1]$ sürekli aralığında tanımlayan bulanık mantık kuramı.
- **T-Norm (Triangular Norm):** Sürekli mantıkta mantıksal 'VE' (conjunction) bağlacının matematiksel genellemesi.
- **T-Conorm (S-Norm):** Sürekli mantıkta mantıksal 'VEYA' (disjunction) bağlacının matematiksel genellemesi.
- **Residual Implication:** T-norm ile uyumlu, $a \implies b$ önermesinin $[0, 1]$ arasındaki sürekli doğruluk derecesi.
- **Łukasiewicz Logic:** $T(a, b) = \max(0, a + b - 1)$ ile tanımlanan, gradyan akışına en uygun sürekli mantık sistemi.
- **Soft Backward Chaining:** İspat ağacını latent embedding uzayında yumuşak mantıksal bağlaçlarla geriye doğru tarayan algoritma.
- **First-Order Logic (FOL):** Nesneler, yüklemler ve niceleyiciler ($\forall, \exists$) içeren birinci dereceden mantık dili.
- **Axiom Violation Loss:** Modelin mantıksal çelişki üretmesini cezalandıran diferansiyellenebilir düzenlileştirme terimi.
- **Transitivity (Geçişlilik):** $P(X, Y) \wedge P(Y, Z) \implies P(X, Z)$ aksiyomunun korunması.

---

## ⚖️ 5. SWOT Analizi

```
+----------------------------------------------------+----------------------------------------------------+
| 🟢 GÜÇLÜ YÖNLER (STRENGTHS)                        | 🟡 ZAYIF YÖNLER (WEAKNESSES)                       |
| • %100 teorem kanıtlama ve çıkarım doğruluğu       | • Çok derin ispat ağaçlarında (depth > 5)          |
| • Gradyan temelli uçtan uca eğitilebilirlik        |   hesaplama karmaşıklığı                           |
| • Mantıksal halüsinasyonları aksiyomatik engelleme | • Kural tabanının önceden tanımlanma ihtiyacı      |
+----------------------------------------------------+----------------------------------------------------+
| 🔵 FIRSATLAR (OPPORTUNITIES)                       | 🔴 TEHDİTLER (THREATS)                             |
| • Yasal mevzuat, tıp ve siber güvenlik denetimi    | • Çelişkili aksiyomlar verildiğinde mantıksal      |
| • Otonom AGI sistemlerinde sıfır hata toleransı    |   çöküş (logical inconsistency / deadlock)         |
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
pytest testler/test_noro_sembolik.py -v
```

---

## 👨‍🏫 7. Senior AI / SRE Mentor Soru-Cevap

**Soru 1: Standart bir Transformer'a yüzlerce mantık kuralı prompt edersek nöro-sembolik bir sistem elde etmiş olmaz mıyız?**  
*Cevap:* Hayır. Prompting, modelin dikkat mekanizmasına ipucu verir ancak modelin iç durumunun (latent representations) mantıksal aksiyomları kesin olarak ihlal etmeyeceğini matematiksel olarak garanti edemez. Nöro-sembolik optimizasyon, mantık kurallarını diferansiyellenebilir kısıtlar olarak ağırlıklara kazır.

**Soru 2: Çelişkili veya döngüsel mantık kuralları verilirse sistem ne yapar?**  
*Cevap:* Sürekli mantıkta çelişkiler $I(A, \neg A)$ kaybını yükseltir. Soft relaxation sayesinde ayrık bir Prolog gibi kilitlenmez (deadlock); çelişkiyi yumuşak olasılık uzayında dağıtarak Pareto-optimal bir denge noktası arar.

**Soru 3: Büyük ölçekli bilgi çizgelerinde (milyonlarca varlık) soft unification nasıl ölçeklenir?**  
*Cevap:* Tüm $Y$ varlıklarını taramak yerine, k-NN vektör arama (FAISS / HNSW) indeksleri kullanılarak yalnızca en yakın semantik komşular ispat zincirine dahil edilir; böylece arama karmaşıklığı $O(N)$'den $O(\log N)$'e düşürülür.

---

## 📜 Lisans

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Bu modül eğitim ve araştırma amaçlıdır. İzinsiz kopyalanamaz ve ticari amaçla kullanılamaz.
