# Day 92: Eğitim Öncesi Veri Sözleşmesi Testleri ve Hazır Bulunuşluk (Readiness) Kontrolleri

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Data Quality: Great Expectations / Deepchecks Standard](https://img.shields.io/badge/Data_Contract-Readiness_Gate-darkgreen.svg?style=flat-square)](#1-🎯-günün-konusu--teorikmatematiksel-derinlik)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_training_contract.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin on birinci gününde; derin öğrenme modelleri eğitilmeden önce GPU/TPU kümelerinin hatalı, eksik, sınır dışı (out-of-bounds) veya sızıntılı (data leakage) veriler yüzünden boşa harcanmasını engelleyen katı bir **Eğitim Öncesi Veri Sözleşmesi ve Hazır Bulunuşluk Kapısı (Pre-Training Data Contract & Readiness Gate)** mimarisini sıfırdan kuruyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Modern derin öğrenme modellerinin eğitimi saatler, günler ve yüz binlerce dolar değerinde bulut GPU kaynağı tüketir. Eğitim boru hatlarında karşılaşılan en büyük tuzak, verideki ölümcül kusurların (silent bugs) eğitim başladıktan saatler sonra (örneğin 50. epoch'ta Loss'un `NaN` olmasıyla) fark edilmesidir:

1. **"Çöp Girerse Çöp Çıkar" (Garbage In, Garbage Out) Kuralı:**
   Veri setindeki tek bir `NaN` veya `Inf` değeri, geri yayılım (backpropagation) esnasında tüm model ağırlıklarını zehirleyerek tüm eğitimi sıfırlar.
2. **Yapay Metrik Şişmesi (Data Leakage & Contamination):**
   Eğitim (Train) setindeki görsellerin bir kısmı yanlışlıkla Doğrulama (Val/Test) setine sızmışsa, model genelleme yapmadığı halde test skoru %99 çıkar; canlıya alındığında ise model tamamen çöker.
3. **Katı Giriş Kapısı (Readiness Gate):**
   Veri seti belirlenen şema, tensör çözünürlüğü, piksel sınırları, sınıf dengesi ve 0-sızıntı kriterlerini %100 sağlamadığı sürece GPU eğitim motorunun başlatılmasına izin verilmez.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Boşa Giden GPU Hesaplama Maliyetini Önleme:**
  Bozuk veri yüzünden 3. günde çöken bir eğitimi ilk saniyede engelleyerek donanım ve zaman israfını sıfıra indirir.
- **Sessiz Veri Sızıntısını (Data Leakage) Kriptografik Olarak Yakalama:**
  SHA-256 tensör parmak izleri ile Train ve Val arasındaki kesişimleri anında bularak hatalı benchmark sonuçlarını önler.
- **Dengesiz ve Nadir Sınıf Krizlerini Önceden Haber Verme:**
  Bazı sınıflarda yalnızca 1-2 örnek kaldığında modelin o sınıfları unutmasını (catastrophic class starvation) engelleyip `ClassImbalance` uyarısı üretir.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Milyarlarca Örnekte Hash Hesaplama Maliyeti:**
  Çok büyük veri setlerinde (ör. ImageNet-22k, 14M görsel) tüm veri üzerinde tam SHA-256 hesaplamak disk I/O darboğazı yaratabilir (bu durumda MinHash veya LSH tercih edilmelidir).
- **Semantik Benzerlik vs Birebir Eşleşme:**
  Hash tabanlı sızıntı dedektörü pikselleri birebir aynı olanları anında yakalar; ancak hafifçe kırpılmış veya döndürülmüş (augmented) sızıntılar için embedding tabanlı k-NN taraması gerekir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Veri Doğrulama Çözümü | Şema & Tip Kontrolü | Sızıntı (Leakage) Tespiti | PyTorch Tensör Uyumu | CI/CD Kapı Mantığı |
|---|---|---|---|---|
| **Bizim Veri Sözleşmesi Motorumuz** | **Tam (Shape, Dtype, Range)** | **Kriptografik SHA-256** | **Doğrudan Native Tensör** | **Katı Gate (Pass/Fail/Block)** |
| **Great Expectations** | Tabüler / SQL Odaklı | Zayıf (Tabüler Hash) | Dolaylı (DataFrame) | Evet |
| **Deepchecks** | Kapsamlı ML / Görsel Testleri | Mevcut (Yakın Eşleşme) | Ağır Kütüphane | Evet |
| **Pydantic v2** | Yüksek Hızlı Şema Doğrulama | Yok | Tekil Nesne Odaklı | Kod Seviyesinde |
| **Pandas Schema Validation** | Sadece Tabüler DataFrame | Yok | Yok | Manuel Script |

---

## 📐 Matematiksel Formülasyon

### 1. Veri Sızıntısı ve Kirlenme İndeksi (Contamination Index)
Eğitim tensörleri kümesi $\mathcal{D}_{\text{train}}$ ve Doğrulama kümesi $\mathcal{D}_{\text{val}}$ olmak üzere, hash fonksiyonu $H(x)$ ile sızıntı oranı $L_{\text{val}}$:

$$L_{\text{val}} = \frac{|\{v \in \mathcal{D}_{\text{val}} \mid \exists t \in \mathcal{D}_{\text{train}}, H(t) = H(v)\}|}{|\mathcal{D}_{\text{val}}|}$$

Katı sözleşme kuralı:

$$L_{\text{val}} \le \epsilon_{\text{leakage}} \quad (\epsilon = 0.00)$$

### 2. Sınıf Dengesizlik Oranı (Imbalance Ratio - IR)
Toplam $C$ sınıf için sınıf frekansları $N_1, N_2, \dots, N_C$ olmak üzere:

$$\text{IR} = \frac{\max_{c} N_c}{\min_{c} N_c}$$

Sözleşme kontrolü:

$$\text{IR} \le \tau_{\text{imbalance}} \quad (\text{ör. } \tau = 8.0)$$

### 3. Shannon Entropi ile Dağılım Denge Skoru
Sınıf olasılıkları $p_c = \frac{N_c}{N_{\text{toplam}}}$ için normalize edilmiş entropi $\mathcal{H}_{\text{norm}}$:

$$\mathcal{H}_{\text{norm}} = -\frac{1}{\ln(C)} \sum_{c=1}^C p_c \ln(p_c) \quad \in [0, 1]$$

$\mathcal{H}_{\text{norm}} \to 1$ ise dağılım mükemmel dengelidir; $\mathcal{H}_{\text{norm}} \to 0$ ise aşırı çökmüş (collapsed) bir dağılım vardır.

---

## 📖 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Terim | Tanım | Endüstriyel Önemi |
|---|---|---|
| **Data Contract (Veri Sözleşmesi)** | Veri üreticisi ile model geliştiricisi arasında verinin şemasını, tiplerini, sınırlarını ve kalitesini garanti eden bağlayıcı kurallar bütünü. | Veri hattı bozulmalarının modele ulaşmasını önler. |
| **Readiness Gate (Hazır Bulunuşluk Kapısı)** | Tüm kalite kontrolleri geçilmeden eğitimin başlatılmasını engelleyen otomatik MLOps karar noktası. | GPU kaynaklarının boşa harcanmasını engeller. |
| **Data Leakage (Veri Sızıntısı)** | Test/Doğrulama verisine ait bilgilerin doğrudan veya dolaylı olarak eğitim setine karışması durumu. | Canlıda çöken ancak testte mükemmel gözüken sahte modelleri engeller. |
| **Contamination Rate** | Doğrulama setindeki örneklerin eğitim setinde de bulunan yüzdesi ($|\text{Train} \cap \text{Val}| / |\text{Val}|$). | Benchmark doğruluğunun en temel güvenlik metriğidir. |
| **Cryptographic Tensor Hashing** | Tensör değerlerinin sabit hassasiyette yuvarlanarak SHA-256 özetlerinin çıkarılması. | Milyonlarca veride kopyaları (duplicates) $O(N)$ karmaşıklıkla bulur. |
| **Out-of-Bounds (Sınır Aşımı)** | Piksel veya öznitelik değerlerinin beklenen $[ \min, \max ]$ sınırları dışına taşması. | Normalizasyon hatalarını ve sensör arızalarını anında yakalar. |
| **NaN/Inf Infection** | Tensör içindeki tanımsız (`NaN`) veya sonsuz (`Inf`) sayıların varlığı. | Gradient patlaması ve eğitimin çökmesine yol açan 1 numaralı düşmandır. |
| **Imbalance Ratio (IR)** | En kalabalık sınıf ile en az örneğe sahip sınıf arasındaki frekans oranı. | Modelin azınlık sınıfları tamamen görmezden gelmesini engeller. |
| **Blocking Failure (Bloke Edici İhlal)** | Sistemin çalışmasını imkansız kılan ve derhal durdurma gerektiren kritik hata. | CI/CD boru hattını kırarak hatalı model çıkmasını önler. |
| **Silent Data Corruption** | Hata fırlatmayan ancak görselin içeriğini (ör. siyah kareler, yanlış ölçekleme) bozan veri kusurları. | Otomatik istatistiksel kontroller olmadan fark edilmesi imkansızdır. |

---

## 📊 SWOT Analizi (Karar Matrisi)

```
┌──────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│                   GÜÇLÜ YÖNLER (S)               │                  ZAYIF YÖNLER (W)                │
│ • Eğitimi başlamadan önce denetleyip GPU koruma. │ • Çok büyük verilerde disk I/O ve hash süresi.   │
│ • SHA-256 ile %100 kesin veri sızıntısı tespiti. │ • Augmentation yapılmış sızıntıları kaçırabilir. │
│ • Sıfır dış bağımlılık, hafif PyTorch entegrasyon│ • Sözleşme sınırlarının iyi ayarlanması gerekir. │
├──────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│                  FIRSATLAR (O)                   │                   TEHDİTLER (T)                  │
│ • CI/CD & Airflow/Kubeflow eğitim pipeline kapısı│ • Aşırı katı kuralların geçerli veriyi engellemes│
│ • Otomatik veri temizleme ve filtreleme motoru.  │ • Dağılım değişiminde sözleşmenin güncellenmemesi│
│ • Model kartlarına (Model Card) veri kanıtı eklem│ • Veri tipleri değişiminde pipeline kırılmaları. │
└──────────────────────────────────────────────────┴──────────────────────────────────────────────────┘
```

---

## 🧪 Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

### 🎯 Görev: SimHash / LSH ile Yakın-Kopya (Near-Duplicate) Sızıntı Dedektörü
Pikselleri birebir aynı olmayan ancak üzerinde hafif gürültü veya parlaklık farkı bulunan görsel sızıntılarını yakalayan **Kaba Tensör SimHash (Perceptual Hash)** sınıfını geliştirin.

### 💡 Eksiksiz Çalışan Çözüm Kodu:

```python
import torch
import numpy as np

class PerceptualHashSizintiDedektoru:
    """Görsel tensörlerini düşük çözünürlüklü ortalama eşikleme ile parmak izine dönüştürür."""
    def __init__(self, kucuk_boyut: int = 8):
        self.kucuk_boyut = kucuk_boyut

    def parmak_izi_cikar(self, tensör: torch.Tensor) -> int:
        if tensör.ndim == 3:
            tensör = tensör.unsqueeze(0)
        # 1. 8x8 boyutuna indir ve gri tonlamaya çevir
        kucuk = torch.nn.functional.interpolate(tensör.mean(dim=1, keepdim=True), size=(self.kucuk_boyut, self.kucuk_boyut))
        kucuk_np = kucuk.squeeze().cpu().numpy()
        ortalama = np.mean(kucuk_np)
        bitler = (kucuk_np > ortalama).flatten()
        parmak_izi = 0
        for bit in bitler:
            parmak_izi = (parmak_izi << 1) | int(bit)
        return parmak_izi

    def hamming_mesafesi(self, h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count('1')

# Test ve Doğrulama
dedektor = PerceptualHashSizintiDedektoru()
x1 = torch.randn(1, 3, 32, 32)
x2 = x1 + torch.randn(1, 3, 32, 32) * 0.05  # Hafif gürültülü sızıntı
x3 = torch.randn(1, 3, 32, 32)  # Tamamen bağımsız görsel

h1 = dedektor.parmak_izi_cikar(x1)
h2 = dedektor.parmak_izi_cikar(x2)
h3 = dedektor.parmak_izi_cikar(x3)

mesafe_sizinti = dedektor.hamming_mesafesi(h1, h2)
mesafe_bagimsiz = dedektor.hamming_mesafesi(h1, h3)

print(f"Sızıntı Hamming Mesafesi: {mesafe_sizinti} bit (Çok Yakın)")
print(f"Bağımsız Görsel Mesafesi: {mesafe_bagimsiz} bit (Uzak)")
assert mesafe_sizinti <= 5
print("✓ Perceptual Hash Dedektörü Başarıyla Doğrulandı!")
```

---

## 📜 Lisans & Metaveri

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

### ❓ Derin Teknik Kontrol Sorusu:
> *"Bir Vision modelinin eğitiminde Validation Loss 1. epoch'ta 0.05 çıkıyor ve Test Accuracy %99.2 olarak ölçülüyor. Ancak model canlı web kamerası akışında %60 doğrulukla çalışıyor. Veri boru hattında yapılan incelemede veri artırma (augmentation) adımının train/val ayrımından (split) ÖNCE yapıldığı tespit edildi. Bu durumun matematiksel ve mimari sebebi nedir?"*

### 💡 Mentorluk Açıklaması ve Çözüm:
Bu senaryo, makine öğrenimindeki en sinsi ve ölümcül felaketlerden biri olan **Ön-İşleme Veri Sızıntısıdır (Pre-Split Augmentation Leakage)**:

1. **Kök Neden:**
   Augmentation işlemi split'ten önce yapıldığında, aynı orijinal görselin döndürülmüş hali Train setine, renk tonu değiştirilmiş hali ise Validation setine düşer.
2. **Matematiksel Etki:**
   Model genelleme öğrenmez; doğrudan o spesifik görselin ayırt edici özniteliklerini (örneğin arka plandaki gürültü desenini) ezberler (memorization). Train ve Val setleri istatistiksel olarak bağımsız ($\mathcal{D}_{\text{train}} \perp \mathcal{D}_{\text{val}}$) olma özelliğini kaybeder.
3. **Mühendislik Çözümü:**
   - **Katı Kural:** Train/Val/Test ayrımı (split) **daima ham veride (raw data)** ve hasta/kullanıcı/görsel ID bazlı gruplama (`GroupKFold`) ile yapılmalıdır.
   - **Eğitim Öncesi Veri Sözleşmesi:** Eğitim öncesi SHA-256 veya Perceptual Hash tabanlı `VeriSizintiDedektoru` çalıştırılarak Train ve Val setleri arasındaki benzerlik oranı sıfırlanmalıdır.
