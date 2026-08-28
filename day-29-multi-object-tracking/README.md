# Day 29: Çoklu Nesne Takibi & Kalman Filtresi / DeepSORT (Multi-Object Tracking)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; otonom sürüş, akıllı şehir gözetimi ve video analitiğinde nesneleri kareler boyunca tutarlı tekil kimliklerle (**Persistent Track IDs**) izleyen **Çoklu Nesne Takibi (Multi-Object Tracking - MOT)** sistemini sıfırdan ele alır. **8 Boyutlu Kalman Filtresi ile Durum Kestirimi**, **Derin Görsel Re-ID (Yeniden Tanımlama) Gömmeleri**, **Mahalanobis Kapılama (Gating)**, **Macar Algoritması (Hungarian Algorithm)** ve **CLEAR MOT / IDF1** metriklerini kapsayan 6 panelli endüstri standardı bir teşhis panosu (Diagnostic Dashboard) sunar.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Tracking-by-Detection Paradigması
Modern çoklu nesne takibi iki bağımsız bileşenin senkronize çalışmasıyla yürütülür:
1. **Nesne Dedektörü (Object Detector):** Her video karesinde nesnelerin anlık konumlarını bağımsız tespit eder ($t$ anında $[x_1, y_1, x_2, y_2]$).
2. **Takipçi (Tracker):** Zamansal tutarlılığı kurarak $t-1$ anındaki nesne ile $t$ anındaki tespiti eşleştirir ve nesneye ömrü boyunca sabit bir `Track ID` atar.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │               VİDEO KARESİ (t anı)                       │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
       [DEDEKTÖR (YOLO / Faster R-CNN)]                                [KALMAN FİLTRESİ]
 ┌───────────────────────────────────────────┐                  ┌───────────────────────────────┐
 │ 1. Nesne Sınırlayıcı Kutuları             │                  │ 3. Konum & Hız Tahmini        │
 │ 2. Görsel Kırpıntılar (Re-ID 128D Embedding)                 │    - x_{k|k-1} = F * x_{k-1}  │
 └─────────────────────┬─────────────────────┘                  └───────────────┬───────────────┘
                       │                                                        │
                       └─────────────────────────────┬──────────────────────────┘
                                                     │
                                                     ▼
                                       ┌───────────────────────────┐
                                       │ MACAR ALGORİTMASI EŞLEME  │
                                       │ - Mahalanobis Kapılama    │
                                       │ - Re-ID Kosinüs Mesafesi  │
                                       │ - İki Parçalı Eşleme      │
                                       └─────────────┬─────────────┘
                                                     │
                                                     ▼
                                       ┌───────────────────────────┐
                                       │   GÜNCELLENMİŞ TAKİPÇİ    │
                                       │   (Sabit ID & Yörünge)    │
                                       └───────────────────────────┘
```

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Video kareleri boyunca tespit edilen nesneleri Kalman Filtresi ve Re-ID öznitelik benzerliği (DeepSORT/ByteTrack) ile kesintisiz takip etmek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Kareler arası kimlik değişimlerini (ID Switch), kısa süreli oklüzyonları ve hareket belirsizliklerini en aza indirir.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Uzun süreli tam oklüzyonlarda veya kameralar arası geçişte kimlik kaybı yaşanabilir.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- ByteTrack, BoT-SORT, OC-SORT veya Transformer tabanlı Trackers (TrackFormer).

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Çoklu Nesne Takibi (MOT)** | *Multi-Object Tracking* | Video kareleri boyunca birden çok nesnenin konumunu tespit edip her birine benzersiz bir kimlik (ID) atayarak izleme görevi. |
| **Kalman Filtresi** | *Kalman Filter* | Gürültülü ölçümler altında nesnenin bir sonraki karedeki konum ve hızını tahmin eden ve güncelleyen doğrusal durum kestiricisi. |
| **Macar Algoritması** | *Hungarian Algorithm (Munkres)* | Önceki izler (tracks) ile yeni tespitler (detections) arasındaki maliyet matrisini çözen optimum iki parçalı eşleme algoritması. |
| **Kimlik Değişimi (ID Switch)** | *Identity Switch (IDSW)* | Takip edilen bir nesnenin kimlik numarasının başka bir nesneyle karışması veya yanlış atanması hatası. |

---

## 2. Kalman Filtresi ile Durum Kestirimi (Bewley et al., SORT 2016)

Kalman filtresi, dedektörün ölçüm gürültüsünü ve kısa süreli hedef kaçırmalarını (Missing Detections) süzerek nesnenin gelecekteki konumunu doğrusal sabit hızlı (Constant Velocity) hareket modeliyle tahmin eder.

#### A. 8 Boyutlu Durum Vektörü
$$\mathbf{x} = [u, v, \gamma, h, \dot{u}, \dot{v}, \dot{\gamma}, \dot{h}]^T$$
- $u, v$: Bounding box merkez koordinatları
- $\gamma$: En-boy oranı ($w / h$)
- $h$: Kutu yüksekliği
- $\dot{u}, \dot{v}, \dot{\gamma}, \dot{h}$: Karşılık gelen hız türevleri

#### B. Tahmin (Predict) ve Güncelleme (Update) Adımları
1. **Tahmin (A Priori):**
   $$\hat{\mathbf{x}}_{k|k-1} = F \hat{\mathbf{x}}_{k-1|k-1}, \quad P_{k|k-1} = F P_{k-1|k-1} F^T + Q$$
2. **Ölçüm Güncellemesi (A Posteriori):**
   $$\mathbf{y}_k = \mathbf{z}_k - H \hat{\mathbf{x}}_{k|k-1}, \quad S_k = H P_{k|k-1} H^T + R$$
   $$K_k = P_{k|k-1} H^T S_k^{-1}$$
   $$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_k \mathbf{y}_k, \quad P_{k|k} = (I - K_k H) P_{k|k-1}$$

#### C. Mahalanobis Mesafesi & Kapılama (Gating)
Fiziksel olarak imkansız eşleşmeleri engellemek için Kalman kestirim belirsizliği ($S$) üzerinden Mahalanobis mesafesi hesaplanır:
$$d^{(1)}(i, j) = (\mathbf{z}_j - H \hat{\mathbf{x}}_i)^T S_i^{-1} (\mathbf{z}_j - H \hat{\mathbf{x}}_i) \le \tau_M$$
- 4 serbestlik dereceli $\chi^2$ dağılımında $\%95$ güven aralığı için $\tau_M = 9.4877$'dir.

---

### 3. DeepSORT: Görsel Re-ID Gömmeleri & Kapanma (Occlusion) Yönetimi

Klasik SORT yalnızca konum/IoU kullandığı için nesneler birbirinin önünden geçtiğinde (Kapanma / Occlusion) kimlikler karışır (**ID Switch**). **DeepSORT (Wojke et al., 2017)** her nesneden 128 boyutlu L2-normalize bir Re-ID öznitelik vektörü ($r_j, \|r_j\|_2 = 1$) çıkarır.

1. **Re-ID Kosinüs Mesafesi:**
   Her takipçi $i$, son $N$ kareden toplanan bir Re-ID görünüş galerisi $\mathcal{R}_i = \{r_k^{(i)}\}$ tutar:
   $$d^{(2)}(i, j) = \min \{ 1 - r_j^T r_k^{(i)} \mid r_k^{(i)} \in \mathcal{R}_i \}$$
2. **Birleşik Maliyet & Macar Algoritması:**
   Maliyet matrisi $C_{i, j} = \lambda d^{(1)}(i, j) + (1-\lambda) d^{(2)}(i, j)$ oluşturulur ve **Macar Algoritması (Hungarian / Munkres)** ile global optimal eşleme yapılır.
3. **Takipçi Yaşam Döngüsü:**
   - **TENTATIVE (Deneme):** Yeni doğan takipçi; $n_{\text{init}}=3$ ardışık tespit alana kadar kesin kabul edilmez.
   - **CONFIRMED (Onaylı):** Aktif izlenen hedef.
   - **DELETED (Silinmiş):** $max_{\text{age}}=30$ kare boyunca güncellenmeyen kayıp hedefler bellekten silinir.

---

### 4. Çoklu Nesne Takibi Değerlendirme Metrikleri

#### A. MOTA (Multiple Object Tracking Accuracy)
Dedektör hatalarını (FP, FN) ve kimlik değişimlerini (IDSW) ölçer:
$$\text{MOTA} = 1 - \frac{\sum_t (\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}$$

#### B. IDF1 (Identification F1-Score)
Bir nesnenin tüm video boyunca ne kadar süre doğru kimlikle izlendiğini ölçer:
$$\text{IDF1} = \frac{2 \text{IDTP}}{2 \text{IDTP} + \text{IDFP} + \text{IDFN}}$$

#### C. IDSW (Kimlik Değişimi Sayısı)
Gerçek bir nesnenin video akışında farklı bir takip ID'sine geçme sayısıdır.

---

## 📊 Deneysel Sonuçlar ve Metrik Tablosu

40 Karelik Kesişen Yörünge ve Kapanma (Occlusion) Simülasyonunda Elde Edilen Çıktılar:

```
================================================================================
>>> CLEAR MOT & IDF1 Metrik Sonuçları (DeepSORT)
================================================================================
[+] Toplam Video Karesi      : 40 Kare (512x384 Çözünürlük)
[+] Toplam Ground Truth Kutu : 156 Adet
[+] Kimlik Değişimi (IDSW)   : 0 Adet (Kusursuz Kimlik Koruma)
[+] Yanlış Pozitif (FP)      : 2 Adet
[+] Kaçırılan Hedef (FN)     : 4 Adet
```

### Kapsamlı Metrik Çizelgesi

| Metrik Adı | Skor (%) | Açıklama |
|---|---|---|
| **MOTA (Doğruluk)** | **%96.15** | Genel dedeksiyon ve takip başarısı |
| **IDF1 (Kimlik F1)** | **%98.05** | Uzun vadeli kimlik sürekliliği skoru |
| **Hassasiyet (Precision)** | **%98.68** | Takip edilen kutuların doğruluk oranı |
| **Anma (Recall)** | **%97.44** | Sahnedeki nesnelerin yakalanma oranı |
| **MT (Çoğunlukla İzlenen)** | **4 / 4 (%100)** | Hedeflerin ömrünün $\ge \%80$'inde kesintisiz izlenmesi |

---

## 🛠️ Dizin Yapısı

```
day-29-multi-object-tracking/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, torch, opencv, seaborn vb.
├── ana_akis.py                      # Uçtan uca video simülasyonu, DeepSORT ve MOT değerlendirmesi
├── README.md                        # Detaylı teorik ve mentorluk dokümantasyonu (220+ Satır)
├── src/
│   ├── __init__.py
│   ├── kalman_filtresi.py           # 8B Kalman Filtresi & Mahalanobis Kapılama motoru
│   ├── reid_cikarici.py             # 128D L2 Normalize Re-ID Görsel Embedding Çıkarıcı
│   ├── takipci_yoneticisi.py        # Takipçi yaşam döngüsü, galeri havuzu ve Macar Eşleme
│   ├── mot_metrik_motoru.py         # CLEAR MOT (MOTA, IDF1, IDSW, MT, ML) metrik hesaplayıcı
│   ├── video_sahne_simulasyonu.py   # Kesişen yörüngeli ve kapanmalı yapay video simülatörü
│   └── gorsellestirici.py           # 6 panelli MOT teşhis panosu (Dashboard) çizici
├── testler/
│   ├── __init__.py
│   └── test_multi_object_tracking.py # 7 adet kapsamlı birim test
└── ciktilar/
    └── coklu_nesne_takip_paneli.png # 6 panelli teşhis panosu görseli
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışın Çalıştırılması
```bash
python ana_akis.py
```

### 3. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/takipci_yoneticisi.py` içine zaman aşımına göre aşamalı eşleme yapan **Matching Cascade** mantığını ekleyin (Daha yakın zamanda görülen takipçilere eşlemede öncelik tanınması).

**Çözüm:**
```python
def matching_cascade(self, tespitler, embeddings):
    # Yaş farkına göre aşamalı eşleme (time_since_update = 1..max_age)
    eslesmeler = []
    kalan_tespitler = list(range(len(tespitler)))
    
    for age in range(1, self.max_age + 1):
        aday_takipciler = [i for i, t in enumerate(self.takipciler) if t.time_since_update == age]
        if not aday_takipciler or not kalan_tespitler:
            continue
        # Yalnızca bu yaş grubundaki takipçilerle Hungarian eşlemesi yap
        sub_eslesmeler, _, kalan_tespitler = self._alt_esle(aday_takipciler, kalan_tespitler, tespitler, embeddings)
        eslesmeler.extend(sub_eslesmeler)
    return eslesmeler
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** İki yayanın kameraya paralel yürüyüp birbirinin önünden geçtiği (Tam Kapanma / Full Occlusion) bir senaryoda, standart SORT algoritması neden kimlik değişimi (ID Switch) yaşar? DeepSORT'taki Görünüş Galerisi (Appearance Gallery) ve Mahalanobis Kapılaması (Gating) bu sorunu nasıl çözer?

> **Cevap:**
> 1. **SORT Başarısızlığı:** Standart SORT yalnızca sınırlayıcı kutuların uzamsal çakışmasına (IoU) ve Kalman hareket tahminine dayanır. İki yaya üst üste bindiğinde kutuların IoU'su $\%80$'in üzerine çıkar. Kapanma anında dedektör tek bir kutu ürettiğinde veya iki kutunun merkezleri çakıştığında, Macar Algoritması arkadaki yayanın takipçisini öndeki yayanın tespitine atayabilir; böylece yayalar ayrıldığında **kimlikler tamamen yer değiştirir (ID Switch)**.
> 2. **DeepSORT Çözümü:** DeepSORT, her takipçi için geçmişte kaydedilmiş Re-ID gömmelerinden oluşan bir galeri ($\mathcal{R}_i$) tutar. Yayalar ayrıldığında, öndeki yayanın kırmızı ceketi ile arkadaki yayanın mavi ceketi arasındaki Re-ID Kosinüs Mesafesi ($d^{(2)} \approx 0.85$) eşik değerini ($\tau_{\text{reid}} = 0.40$) aşarak yanlış eşleşmeyi **yasaklar (Gate Out)**. Böylece arkadaki yaya $3-5$ kare boyunca görünmese dahi ortaya çıktığında eski kimliğiyle doğru şekilde yeniden eşleşir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
