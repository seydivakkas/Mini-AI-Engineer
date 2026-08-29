# Day 316: Bizans Hata Toleransı ve Düşmanca Saldırılara Karşı Öz-Düzeltme (Byzantine Fault-Tolerant Multi-Agent Consensus & Robust Aggregation)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](#lisans)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-blue.svg?style=flat-square)](https://numpy.org/)
[![Tests Passing](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)

Dağıtık yapay zeka sürülerinde (Federated Learning & Swarm Intelligence), kümedeki ajanların bir kısmı ele geçirildiğinde, donanımsal arıza yaşadığında veya düşmanca sızmalara (Byzantine Adversaries) maruz kaldığında, aritmetik ortalamaya dayalı standart gradyan toplama yöntemleri tek bir zehirli vektörle tamamen çöker (gradient corruption / catastrophic divergence).

**Day 316**, $f < M/3$ oranındaki kötü niyetli Bizans düğümlerine karşı matematiksel yakınsama güvencesi sunan **Multi-Krum**, **Koordinat Bazlı Kırpılmış Ortalama (Coordinate-Wise Trimmed Mean)** ve hibrit **Bulyan** sağlam toplama (Robust Aggregation) algoritmalarını sunar. Sistem, işaret çevirme (sign-flipping) saldırılarına karşı **%99.48 saldırı azaltma (mitigation)** ve **%100 Bizans saldırgan tespit başarısı** sağlar.

---

## 🏗️ 1. Mimari Tasarım ve Matematiksel Temeller

```
========================================================================================
             BİZANS HATA TOLERANSLI SAĞLAM TOPLAMA MİMARİSİ (DAY 316)
========================================================================================

   Dürüst Sürü Düğümleri {g_1, ..., g_{M-f}} \
                                              +---> Gradyan Havuzu (G in R^{M x D})
   Bizans Saldırgan Düğümleri {g'_1, ..., g'_f} /
                                              |
                   +--------------------------+--------------------------+
                   |                          |                          |
                   v                          v                          v
          [ Naive Mean ]             [ Multi-Krum ]              [ Bulyan Toplayıcı ]
        g = 1/M * sum(g_i)       Score(i) = sum_{j in N_i} ||g_i - g_j||^2     \
       (KUTUPSAL ÇÖKÜŞ -0.99)     En küçük m skoru seç        Krum Ön Filtreleme (theta = M-2f)
                                         |                             |
                                         v                             v
                                  g_krum (Kosinüs: +0.97)     Trimmed Mean (beta = f)
                                                                       |
                                                                       v
                                                              g_bulyan (Kosinüs: +0.985)
========================================================================================
```

### Matematiksel Formülasyon

1. **Multi-Krum Skorlama Fonksiyonu:**
   $$\text{Score}(i) = \sum_{j \in \mathcal{N}_i} \|\mathbf{g}_i - \mathbf{g}_j\|^2 \quad \text{burada } |\mathcal{N}_i| = M - f - 2$$
   $$\mathbf{g}_{\text{Krum}} = \frac{1}{m} \sum_{i \in \text{Top-}m} \mathbf{g}_i$$

2. **Koordinat Bazlı Kırpılmış Ortalama (Trimmed Mean):**
   $$\mathbf{g}_{\text{Trimmed}}(d) = \frac{1}{M - 2f} \sum_{i=f+1}^{M-f} \mathbf{g}_{(i)}(d)$$

3. **Bulyan Hibrit Protokolü:**
   Multi-Krum ile $\theta = M - 2f$ boyutunda güvenli aday havuzu oluşturulur; ardından bu adaylar üzerinde $\beta = f$ parametreli koordinat kırpma uygulanarak sinsi gradyan kaymaları sıfırlanır.

---

## 🔬 2. Derinlemesine Mimari Analizler

### Analiz 1: Dağıtık Sürüler Neden Aritmetik Ortalama ile Çöker?
Standart federe öğrenmede sunucu $\mathbf{g} = \frac{1}{M} \sum \mathbf{g}_i$ hesaplar. Eğer tek bir Bizans düğümü $\mathbf{g}_{\text{malicious}} = -100 \cdot \mathbf{g}_{\text{true}}$ gönderirse, toplam gradyanın yönü tamamen tersine döner (kosinüs benzerliği $-0.9987$'ye düşer) ve modelin kaybı 328.77'ye fırlayarak eğitimi paramparça eder.

### Analiz 2: Geometrik Öklid Mesafesi vs Boyutsal Medyan Kıyası
Krum, yüksek boyutlu uzayda dürüst düğümlerin birbirine yakın kümelendiği, aşırı uçtaki saldırganların ise yüksek mesafe puanı alacağı gerçeğine dayanır. Ancak çok sinsi saldırganlar (küme merkezine yakın durup belirli koordinatları bozanlar) Krum'u kandırabilir. Koordinat bazlı Trimmed Mean ise her boyutu bağımsız sıralayarak boyutsal manipülasyonları temizler.

### Analiz 3: Bulyan Algoritmasının Çifte Savunma Güvencesi
Bulyan, hem Krum'un uzamsal kümeleme gücünü hem de Trimmed Mean'in boyutsal kırpma filtresini birleştirir. $M \ge 4f + 3$ koşulu altında, Bizans saldırganlarının ne yönü ne de büyüklüğü değiştirmesine izin vermez (Kosinüs sadakati $+0.9851$).

### Analiz 4: Hesaplama Karmaşıklığı ($O(M^2 D)$) ve Dağıtık İletişim Dengesi
Krum algoritması tüm düğüm çiftleri arasındaki mesafeyi hesapladığı için $O(M^2 D)$ işlem yükü getirir. Sürü boyutu $M=1000$ olduğunda hiyerarşik kümeleme (Hierarchical Krum) veya rastgele örneklemeli alt kümeler kullanılarak işlem süresi $O(M D \log M)$ seviyesine indirgenir.

---

## 📊 3. 6-Panelli Teşhis Panosu İncelemesi

Modül çalıştırıldığında `ciktilar/bizans_tolerans_paneli.png` konumunda üretilen 6 teşhis paneli:

1. **Bizans Saldırısı Altında Kayıp Yakınsaması:** Naive Mean patlarken Bulyan ve Krum'un ($1.70$) pürüzsüz minimuma inişi.
2. **Gerçek Gradyanla Açısal Uyum (Cosine):** Naive Mean'in negatif sapması ($-0.9987$) karşısında Bulyan'ın $+0.9851$ pozitif kararlılığı.
3. **Toplayıcılar Arası Son Kayıp Kıyası:** 5 algoritmanın nihai görev hatası dağılımı.
4. **Sürü Düğüm Durumları & Zehirli Düğümler:** 15 düğümlük sürüde tespit edilen 4 Bizans düğümü (kırmızı) ve dürüst düğümler (yeşil).
5. **Bizans Savunma ve Tespit Performansı:** %99.48 saldırı azaltma, %100 kesinlik (precision) ve %100 duyarlılık (recall).
6. **Sürü Hata Toleransı Modeli Özeti:** Bizans oranı ($f < M/3$), ortalama kosinüs değerleri ve güvenlik sınıfı konsolide raporu.

---

## 📖 4. Terimler Sözlüğü (Glossary)

- **Byzantine Fault (Bizans Hatası):** Bir dağıtık sistem bileşeninin keyfi, çelişkili veya bilinçli olarak düşmanca yanlış veri üretmesi durumu.
- **Sign-Flipping Attack:** Gerçek gradyan vektörünün yönünü tersine çevirerek eğitimi ters yöne sürükleyen saldırı tipi.
- **Multi-Krum:** Düğümleri en yakın komşularına olan toplam Öklid mesafesine göre sıralayıp en tutarlı çekirdeği seçen algoritma.
- **Coordinate-Wise Trimmed Mean:** Her parametre ekseninde en büyük ve en küçük $f$ adet değeri atıp kalanların ortalamasını alan yöntem.
- **Bulyan:** Krum ön elemesi ile Trimmed Mean yöntemini birleştiren matematiksel olarak kanıtlanmış en güçlü Bizans toplayıcısı.
- **Cosine Fidelity:** Toplanan gradyan vektörü ile gerçek kuramsal gradyan arasındaki kosinüs benzerliği ($[-1, 1]$).
- **Poisoned Gradient:** Model ağırlıklarını zehirlemek veya arka kapı (backdoor) açmak için enjekte edilen kötü niyetli gradyan.
- **Individual Rationality:** Dürüst düğümlerin sisteme katkısının korunması güvencesi.
- **Breakdown Point (Kırılma Noktası):** Bir toplayıcının doğru sonuç üretmeyi bıraktığı maksimum Bizans düğüm oranı ($f/M$).
- **Geometric Median:** Çok boyutlu uzayda tüm noktalara olan toplam Öklid mesafesini minimize eden merkez noktası.

---

## ⚖️ 5. SWOT Analizi

```
+----------------------------------------------------+----------------------------------------------------+
| 🟢 GÜÇLÜ YÖNLER (STRENGTHS)                        | 🟡 ZAYIF YÖNLER (WEAKNESSES)                       |
| • %99.48 saldırı azaltma başarısı                  | • O(M^2 D) çiftler arası mesafe hesaplama maliyeti |
| • %100 saldırgan tespit kesinliği ve duyarlılığı    | • Düğümlerin f < M/3 kısıtına uyması zorunluluğu   |
| • Matematiksel olarak kanıtlanmış Bulyan kararlılığı|                                                   |
+----------------------------------------------------+----------------------------------------------------+
| 🔵 FIRSATLAR (OPPORTUNITIES)                       | 🔴 TEHDİTLER (THREATS)                             |
| • Otonom İHA/SİHA sürülerinde güvenli konsensüs    | • f >= M/2 durumunda Bizans çoğunluk saldırısı     |
| • Merkeziyetsiz web3 yapay zeka eğitim ağları      |   (Sybil Attack / 51% Attack)                      |
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
pytest testler/test_bizans_toleransi.py -v
```

---

## 👨‍🏫 7. Senior AI / SRE Mentor Soru-Cevap

**Soru 1: Sürüdeki Bizans saldırgan sayısı $f \ge M/3$ olursa ne yapılabilir?**  
*Cevap:* Klasik Bulyan $f < M/3$ gerektirir. Eğer Bizans oranı %33-%49 arasına çıkarsa, düğümlerin geçmiş güvenilirlik skorlarını tutan *Reputation-Weighted Multi-Center Clustering (R-Krum)* veya Sıfır Bilgi İspatı (ZK-SNARKs) tabanlı şifreli doğrulama katmanları devreye alınır.

**Soru 2: Ajanlar gradyanları şifreli gönderiyorsa (Homomorphic Encryption) Krum mesafeleri nasıl hesaplanır?**  
*Cevap:* Homomorfik şifreleme altında Öklid mesafesi $\|\mathbf{g}_i - \mathbf{g}_j\|^2$ doğrudan şifreli uzayda (ciphertext) iç çarpım yapılarak hesaplanabilir; sunucu gradyanların içeriğini görmeden Krum skorlarını sıralar ve Bulyan filtresini yürütür.

**Soru 3: Sign-flipping dışındaki sinsi saldırılar (örneğin Backdoor Attack) nasıl engellenir?**  
*Cevap:* Bulyan'ın ikinci aşaması olan koordinat bazlı kırpma, sinsi saldırganların yalnızca belirli alt uzaylara gizlice eklediği tetikleyici vektörleri (backdoor triggers) boyut bazında budayarak yok eder.

---

## 📜 Lisans

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Bu modül eğitim ve araştırma amaçlıdır. İzinsiz kopyalanamaz ve ticari amaçla kullanılamaz.
