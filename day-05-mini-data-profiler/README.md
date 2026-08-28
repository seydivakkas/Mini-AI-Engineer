# Day 05: Otomatik Tabüler Veri Profilleyici (Mini Data Profiler)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.2+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.13+-blue.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; ağır ve harici bağımlılıkları olan büyük profilleme kütüphanelerine (ydata-profiling vb.) ihtiyaç duymadan, doğrudan **Pandas, NumPy ve SciPy** kullanarak herhangi bir veri çerçevesinin (DataFrame) eksiklik oranlarını, kardinalitesini, çarpıklık (Skewness), basıklık (Kurtosis) değerlerini ve anlamsal veri tiplerini otomatik çıkaran ve CI/CD hatlarına uygun **Markdown & Konsol raporları** üreten hafif bir veri profilleme motorudur.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
MLOps ve veri mühendisliği boru hatlarında her gün binlerce yeni veri akışı (batch data) sisteme girer. 
- Gelen veride aniden eksik değer oranı fırladı mı?
- Verinin dağılımı (örneğin sensör ölçümü) sağa doğru aşırı çarpıklaştı mı (drift oluştu mu)?
- Bir sütun aslında kategorik mi, benzersiz kimlik (ID) mi, yoksa sürekli sayısal bir değişken mi?

Ağır araçlar gigabaytlarca bellek tüketip CI/CD süreçlerini dakikalarca bekletirken; hafif, saf Python tabanlı bir **Mini Profiler** saniyeler içinde veri sağlığını doğrular.

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Veri Profili Çıkarma** | *Data Profiling* | Veri setinin eksiklik, benzersizlik, tip dağılımı ve istatistiksel momentlerini otomatik olarak özetleyen analiz süreci. |
| **Çarpıklık (Asimetri)** | *Skewness* | Veri dağılımının ortalamaya göre asimetrisini ölçen 3. moment istatistiği ($>0$ sağa çarpık, $<0$ sola çarpık). |
| **Basıklık** | *Kurtosis* | Veri dağılımının kuyruk kalınlığını ve tepe sivriliğini normal dağılıma göre ölçen 4. moment istatistiği. |
| **Kardinalite** | *Categorical Cardinality* | Kategorik bir sütundaki benzersiz (unique) kategori sayısı; yüksek kardinalite one-hot encoding boyut patlamasına yol açar. |
| **Çeyrekler Arası Açıklık (IQR)** | *Interquartile Range* | Verinin 75. persentili ($Q_3$) ile 25. persentili ($Q_1$) arasındaki fark; aykırı değer eşiklerinin temelidir. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. Çarpıklık (Skewness / Üçüncü Moment)
Veri dağılımının simetriden ne kadar saptığını ölçer:

$$\text{Skewness} = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^3}{\left(\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2\right)^{3/2}}$$

- **$\text{Skew} \approx 0$:** Simetrik (Normal) dağılım.
- **$\text{Skew} > 1$:** **Sağa Çarpık (Pozitif Çarpıklık / Uzun Sağ Kuyruk).** Endüstride çoğu hata süresi, maaş veya piksel parlaklığı sağa çarpıktır. Bu değişkenlere model öncesi $log(1 + x)$ dönüşümü uygulanmalıdır.
- **$\text{Skew} < -1$:** Sola Çarpık (Negatif Çarpıklık).

#### B. Basıklık (Kurtosis / Dördüncü Moment)
Dağılımın kuyruk kalınlığını ve sivrilik derecesini ölçer:

$$\text{Kurtosis} = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^4}{\left(\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2\right)^2} - 3$$
*(Balıkçı / Fisher tanımında normal dağılımın basıklığı $0$'dır).*
- **$\text{Kurtosis} > 0$ (Leptokurtik):** Kuyruklar çok kalındır; veride **aşırı aykırı değer (outlier)** riski vardır!

#### C. Kardinalite ve Anlamsal Veri Tipi Çıkarımı
Teknik veri tipi (`dtype`) her zaman verinin anlamsal doğasını yansıtmaz.
- Eğer benzersiz değer oranı $> 0.95$ ve tamsayı/metin ise $\to$ **`benzersiz_kimlik` (ID)**.
- Eğer benzersiz değer sayısı $\le 20$ veya oranı $< 0.05$ ise $\to$ **`kategorik`**.
- Eğer benzersiz değer sayısı tam $2$ ise $\to$ **`ikili` (Boolean)**.
- Sayısal ve sürekli değişkenler $\to$ **`sayisal_surekli`**.

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Ağır Araçların Bellek Şişmesi:** `pandas-profiling` 1 GB'lık bir veride 8 GB RAM tüketerek OOM (Out Of Memory) hatası verebilir. Hafif profilleme tek geçişli döngülerle yapılmalıdır.
2. **Kardinalite Yanılgısı:** Sayısal bir float sütun yüksek benzersizlik oranına sahip diye ID olarak etiketlenmemelidir; sadece tamsayı ve metinler ID olabilir.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
              Ham Veri Tablosu (DataFrame)
                            │
                            ▼
              ┌───────────────────────────┐
              │    MiniVeriProfilleyici   │
              └─────────────┬─────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
[Eksik Değer &     [Kardinalite &      [Çarpıklık &
 Bellek Analizi]    Anlamsal Tip]       Basıklık]
- Eksiklik %'si     - ID, Kategori,     - Skewness
- Toplam RAM          Sürekli Sayısal   - Kurtosis
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
              ┌───────────────────────────┐
              │      RaporOlusturucu      │
              └─────────────┬─────────────┘
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
     [Konsol Tablosu]              [profil_raporu.md]
```

---

## 💻 Konsol Çalıştırma Çıktısı

```text
======================================================================
>>> AŞAMA 1: Mini Veri Profilleyicinin Çalıştırılması
======================================================================
Tablo Satır Sayısı      : 600
Tablo Sütun Sayısı      : 6
Toplam Bellek Tüketimi  : 27.23 KB
Eksik Değer Barındıran  : 3 sütun

======================================================================
>>> AŞAMA 2: Sütun İstatistik Profili Özeti
======================================================================
sutun_adi                 | anlamsal_tip     | eksiklik_% | benzersiz | carpiklik | basiklik  
-----------------------------------------------------------------------------------------------
parca_seri_no             | benzersiz_kimlik | 0.00%      | 600       | -         | -         
uretim_hatti              | kategorik        | 0.00%      | 3         | -         | -         
sensor_sicaklik_c         | sayisal_surekli  | 0.00%      | 600       | 0.01      | -0.06     
hata_bekleme_suresi_sn    | sayisal_surekli  | 4.17%      | 576       | 2.16      | 5.42      
kusur_tespit_edildi_mi    | ikili            | 0.00%      | 2         | -         | -         
kamera_kalibrasyon_skoru  | sayisal_surekli  | 6.67%      | 561       | -0.05     | -0.36     

======================================================================
>>> AŞAMA 3: Markdown Profil Raporunun Diske Yazılması
======================================================================
[V] Markdown raporu başarıyla kaydedildi: profil_raporu.md
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Veri Kayması (Data Drift) Ön Uyarı Modülü**

İki farklı veri seti verildiğinde (örneğin Dün Gelen Veri vs. Bugün Gelen Veri), sayısal sütunların ortalamalarında veya medyanlarında %20'den fazla sapma olup olmadığını tespit eden bir kontrol fonksiyonu geliştir.

### Görev Tanımı:
[`src/veri_profilleyici.py`](./src/veri_profilleyici.py) sınıfına şu metodu ekle:

```python
def veri_kaymasi_kontrolu(
    self,
    referans_df: pd.DataFrame,
    yeni_df: pd.DataFrame,
    sapma_esigi: float = 0.20
) -> List[str]:
```
Ortalaması %20'den fazla değişen sütunların adını bir liste olarak döndür.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> Bir özniteliğin **çarpıklık (skewness) değeri $+2.5$** çıktığında (aşırı sağa çarpık), bu değişkeni doğrudan bir lineer regresyon veya yapay sinir ağına vermek neden gradyanların kararsızlaşmasına yol açar? Neden bu sütuna **Logaritmik Dönüşüm ($log(1 + x)$)** veya **Box-Cox dönüşümü** uygularız?

---

## 📂 Dizin Yapısı

```
day-05-mini-data-profiler/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (pandas, numpy, scipy, pytest)
├── ana_akis.py                 # Konsol laboratuvar çalıştırma betiği
├── profil_raporu.md            # Otomatik üretilen Markdown raporu
├── src/
│   ├── __init__.py
│   ├── veri_profilleyici.py    # MiniVeriProfilleyici çekirdek sınıfı
│   └── rapor_olusturucu.py     # Konsol ve Markdown rapor üreteci
└── testler/
    └── test_profilleyici.py    # 6 adet birim testi (6 passed)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Laboratuvar Akışını Çalıştırma
```bash
python ana_akis.py
```

### 3. Testleri Koşma
```bash
python -m pytest testler/test_profilleyici.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
