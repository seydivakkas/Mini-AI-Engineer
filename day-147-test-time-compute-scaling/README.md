# Day 147: Test-Time Compute Scaling Yasaları: Çıkarım Zamanı Hesaplama Bütçesi & Pareto Sınırları

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; OpenAI o1, o3 ve Snell et al. (UC Berkeley) çalışmalarına dayanan **Test-Time Compute Scaling Yasaları (Çıkarım Zamanı Hesaplama Yasaları)**, **Hesaplama Bütçesi Dağıtımı ($N_{\text{compute}} = N_{\text{tokens}} \times N_{\text{samples}}$)**, **Arama Derinliği vs Genişliği Ticareti (Depth vs Width Trade-off)** ve **Pareto Verimlilik Eğrileri** mimarisini sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Yapay Zekada Yeni Çağ: Neden Modeli Daha Büyük Eğitmek Yerine Çıkarım Sırasında Daha Çok Düşündürüyoruz?
- **Eski Paradigma (Pre-training Scaling):**
  Daha zeki bir yapay zeka için modeli yüz milyarlarca parametreyle eğitmek ($8\text{B} \to 70\text{B} \to 405\text{B}$) gerekiyordu. Ancak model eğitim bütçeleri yüz milyonlarca dolara ulaştı ve veri sınırına (Data Wall) gelindi. Ayrıca model çıkarım anında her soruya tek seferde ve sabit sürede ($O(1)$) cevap vermek zorundaydı.
- **Yeni Paradigma (Test-Time Compute Scaling):**
  Zor bir soru karşısında bir insanın yaptığı gibi modelin çıkarım anında **daha fazla token harcayarak düşünmesine**, farklı stratejileri simüle etmesine ve kendi adımlarını denetlemesine izin verilir.
  - **Kritik Pareto Keşfi:** $8\text{B}$ parametreli hafif bir model $+ 16\times$ Test-Time Compute bütçesi, tek geçişli ($1\times$) $70\text{B}$ devasa bir modeli geride bırakmaktadır!
  - **Güç Yasası (Power-Law):** Çıkarım anındaki token/örnek sayısı $N$ arttıkça hata oranı $\alpha N^{-\beta}$ formülüyle istikrarlı biçimde düşer.

```
       TEST-TIME COMPUTE PARADIGM SHIFT
  ┌──────────────────────────────────────────────┐
  │ [Pre-training Scaling - Eski]                │
  │ • Model Boyutu: 70B veya 405B                │
  │ • Statik Tek Geçiş ($O(1)$)                  │
  │ • Bellek Tüketimi: 140GB+ VRAM               │
  │ • Doğruluk: %76.0                            │
  └──────────────────────┬───────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────┐
  │ [Test-Time Compute Scaling - Yeni (o1 / R1)] │
  │ • Model Boyutu: 8B (Kompakt ve Hızlı)        │
  │ • Dinamik Düşünme Bütçesi: 16x Token         │
  │ • Bellek Tüketimi: 16GB VRAM                 │
  │ • Doğruluk: %81.0 (Dev Modeli Geçti!)       │
  └──────────────────────────────────────────────┘
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Test-Time Power-Law Scaling Yasası
- Çıkarım bütçesi $N$ ile hata oranı arasındaki ilişki:
  $$\text{Hata}(N) = \alpha \cdot N^{-\beta} + \gamma, \quad \text{Doğruluk}(N) = 1.0 - \text{Hata}(N)$$
  - $\alpha$: Başlangıç problem zorluğu / hata katsayısı ($\approx 0.65$).
  - $\beta$: Scaling verimlilik katsayısı ($\approx 0.35 - 0.50$).
  - $\gamma$: Asimptotik indirgenemez taban hata ($\approx 0.05$).

### B. Arama Derinliği ($D$) vs Genişliği ($K$) Ticareti (Depth vs Width Trade-off)
- Sabit bir $N_{\text{token}}$ bütçesinde:
  - **Sadece Genişlik (Paralel Örnekleme / SC):** $K=32, D=1 \implies \text{Doğruluk} \approx \%82.6$.
  - **Sadece Derinlik (Tekil Uzun Zincir):** $K=1, D=32 \implies \text{Doğruluk} \approx \%78.8$.
  - **Dengeli Ağaç (MCTS / ToT):** $K=5, D=5 \implies \text{Doğruluk} \approx \mathbf{\%94.2}$ (Optimal Nokta!).

### C. Pareto Verimlilik Sınırı (Pareto Frontier)
- Donanım maliyeti, bellek ayak izi ve gecikme fonksiyonu optimize edilerek en yüksek doğruluk / en düşük maliyet noktaları seçilir.

### D. Test-Time Compute Allocation (Dinamik Bütçe Tahsisi)
- Kolay sorularda minimum token ($N=1$), zor sorularda maksimum token ($N=64-256$) harcanarak küresel maliyet minimize edilir.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Test-Time Compute** | Modelin çıkarım (inference) anında düşünmek için harcadığı ek hesaplama ve token bütçesi. |
| **Power-Law Scaling** | Hesaplama bütçesi arttıkça hata oranının log-lineer olarak azalmasını açıklayan güç yasası. |
| **Pareto Frontier** | Maliyet ve performans arasında biri iyileştirilirken diğeri kötüleşmeyen en verimli optimal eğri. |
| **Depth vs Width Trade-off** | Hesaplama bütçesini daha çok alternatif yol türetmeye (genişlik) mi yoksa derin adımlara mı ayırma kararı. |
| **FLOPs Budget** | Modelin bir soruyu çözerken gerçekleştirmesine izin verilen toplam kayan nokta işlem sayısı. |
| **Inference-Time Search** | Çıkarım anında MCTS veya ToT gibi arama ağaçlarıyla çözüm uzayını tarama süreci. |
| **Data Wall** | Pre-training aşamasında dünyadaki yüksek kaliteli metin verisinin tükenmesi eşiği. |
| **Adaptive Compute Budget** | Sorunun zorluğuna göre dinamik olarak token sayısını artıran veya azaltan mekanizma. |
| **Parallel Sampling Efficiency** | Birden fazla düşünce yolunu aynı anda paralel GPU çekirdeklerinde koşmanın getirdiği hız. |
| **Latency Penalty** | Test-time compute artırıldığında kullanıcının cevabı bekleme süresindeki uzama. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Küçük modellerle (8B) devasa       │ • Çok adımlı düşünme sebebiyle       │
 │   modelleri (70B) alt etme gücü.     │   uzayan ilk yanıt süresi (TTFT).    │
 │ • Pre-training maliyetini çıkarım    │ • Token başına artan API ve GPU      │
 │   hesaplamasına kaydırma esnekliği.  │   operasyon maliyeti.                │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Donanım kısıtlı kurumsal sistem-   │ • Modelin çıkarım anında sonsuz      │
 │   lerde hafif modellerle yüksek akıl │   döngüye veya gevezeliğe girmesi    │
 │   yürütme sağlama (Edge AI).         │   (Overthinking / Verbosity).        │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/test_time_compute_scaling_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
