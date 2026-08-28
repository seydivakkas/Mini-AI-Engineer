# Day 82: Öğretmen-Öğrenci Modeli Bilgi Damıtma (Knowledge Distillation) — Soft Target Loss (KL-Diverjansı), Temperature

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Compression: Knowledge Distillation](https://img.shields.io/badge/Compression-Knowledge_Distillation-brightgreen.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_damitma.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin açılış gününde; Geoffrey Hinton et al. (2015) *"Distilling the Knowledge in a Neural Network"* makalesinin çığır açan yöntemini sıfırdan kuruyoruz. Devasa ve yüksek doğruluklu bir **Öğretmen Modelinin (Teacher)** karar yüzeyini ve sınıflar arası gizli ilişkilerini (**Dark Knowledge**); **Sıcaklık Ölçeklemesi (Temperature $\tau$)**, **Kullback-Leibler (KL) Diverjansı** ve **Sert Etiket Cross-Entropy Kaybı** ile $55\times$ daha hafif ve kompakt bir **Öğrenci Modele (Student)** aktarıyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Milyonlarca parametreye sahip devasa derin öğrenme modelleri (ResNet-152, ViT-Huge, LLaMA) yüksek doğruluk üretir ancak mobil cihazlarda, dronlarda ve IoT donanımlarında yüksek gecikme (latency) ve bellek tüketimi nedeniyle doğrudan çalıştırılamaz. Küçük modeller ise doğrudan eğitildiklerinde kapasite eksikliği yüzünden karmaşık veri manifoldunu öğrenemez.

Knowledge Distillation bu açmazı şu bilimsel ilkelerle çözer:

1. **Karanlık Bilginin (Dark Knowledge) Transferi:**
   Tek-sıcak (one-hot) etiketler $[0, 0, 1, 0]$ gerçeği siyah-beyaz gösterir. Örneğin bir "kamyonet" görseli için standart etiket sadece kamyonettir. Ancak derin bir Öğretmen model $[0.01\text{ sedan}, 0.15\text{ SUV}, 0.80\text{ kamyonet}, 0.0001\text{ köpek}]$ olasılıkları üretir. Bu olasılıklar sınıflar arasındaki yapısal benzerlikleri (geometrik akrabalığı) kodlar.
2. **Sıcaklık Ölçeklemesi ($\tau$) ile Olasılıkların Yumuşatılması:**
   Normal Softmax ($\tau=1$) en yüksek logite %99.9 olasılık verip diğerlerini sıfırlar. Logitler $\tau > 1$ (ör. $\tau=3$ veya $4$) ile bölünerek Softmax uygulandığında, düşük olasılıklı sınıfların zengin dağılımı açığa çıkar.
3. **$\tau^2$ Gradyan Ölçekleme Koruması:**
   Logitler $\tau$'ya bölündüğünde geriye yayılan gradyanlar $\frac{1}{\tau^2}$ oranında zayıflar. Kayıp fonksiyonu $\tau^2$ ile çarpılarak gradyan büyüklüğü standart Cross-Entropy ile aynı skalada dengelenir.
4. **$55\times$ Parametre Tasarrufu ve Sıfır Dağıtım Yükü:**
   Öğrenci eğitildikten sonra Öğretmen tamamen atılır; dağıtıma yalnızca ultra hafif Öğrenci modeli girer ($0\text{ ms}$ ek gecikme).

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Küçük Modellerin Düşük Genelleme Sorunu:**
  Doğrudan sert etiketlerle eğitilen küçük modeller ezberlemeye meyilliyken, Öğretmenin pürüzsüz olasılık yüzeyi sayesinde aşırı uydurma engellenir.
- **Edge AI ve Mobil Çıkarım Engeli:**
  Belleği 2 GB olan bir sunucu modelini 30 MB'lık bir kenar cihaza doğruluk kaybetmeden indirger.
- **Etiket Gürültüsü (Label Noise):**
  İnsan etiketçilerin hatalı işaretlediği verilerde Öğretmenin tutarlı olasılık dağılımı gürültüyü filtreler.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **İki Aşamalı Eğitim Maliyeti:**
  Öğrenciyi eğitmeden önce mutlaka yüksek başarıma sahip büyük bir Öğretmen modelinin eğitilmiş olması şarttır.
- **Kapasite Uyuşmazlığı (Capacity Gap):**
  Öğretmen ile Öğrenci arasındaki kapasite farkı aşırı büyükse (ör. 100M parametreye karşı 10k parametre), öğrenci öğretmenin karmaşık manifoldunu taklit edemez (ara damıtma / asistan model gerekir).

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Model Sıkıştırma Yöntemi | Çalışma İlkesi | Parametre Azalması | Yeniden Eğitim | Donanım Uyumu |
|---|---|---|---|---|
| **Knowledge Distillation (Bizim Yöntem)** | **Öğretmenin yumuşak olasılıklarını öğrenme** | **$10\times - 55\times$** | **Gerekir** | **Tüm Donanımlar** |
| **Ağırlık Budama (Weight Pruning)** | Önemsiz ağırlıkları sıfırlama | $2\times - 5\times$ | İnce ayar gerekir | Özel sparse kütüphaneler |
| **Kuantizasyon (INT8 Post-Training)** | FP32 $\to$ INT8 dönüştürme | $4\times$ | Gerekmez/Az | INT8 Tensör Çekirdeği |
| **Düşük Dereceli Ayrışım (LoRA/SVD)** | Matrisleri $U \cdot V$ şeklinde çarpanlara ayırma | $2\times - 4\times$ | İnce ayar gerekir | Standart GEMM |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           KNOWLEDGE DISTILLATION (HINTON ET AL. 2015) AKIŞ ŞEMASI                         │
│                                                                                                           │
│       Girdi Görseli: x                                                                                    │
│          │                                                                                                │
│          ├──────────────────────────────────────────────┐                                                 │
│          ▼                                              ▼                                                 │
│       Dondurulmuş Öğretmen (Teacher)                 Eğitilebilir Öğrenci (Student)                       │
│       z_T = Teacher(x)                               z_S = Student(x)                                     │
│          │                                              │                                                 │
│          ▼ (Sıcaklık ile Yumuşatma: τ > 1)              ├───────────────────────┐                         │
│       p_T^τ = Softmax(z_T / τ)                          ▼ (τ > 1)               ▼ (τ = 1)                 │
│          │                                           p_S^τ = Softmax(z_S / τ)   p_S = Softmax(z_S)        │
│          │                                              │                       │                         │
│          └──────────────────────┬───────────────────────┘                       │                         │
│                                 ▼                                               ▼                         │
│                      KL-Diverjansı: D_KL(p_S^τ || p_T^τ)             Cross-Entropy: CE(p_S, y_true)       │
│                                 │                                               │                         │
│                                 ▼ (x τ²)                                        │                         │
│                         L_soft = τ² · D_KL                                      L_hard = CE               │
│                                 │                                               │                         │
│                                 └──────────────────────┬────────────────────────┘                         │
│                                                        ▼                                                  │
│                                      L_total = (1 - α) · L_hard + α · L_soft                              │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Sıcaklık ile Yumuşatılmış Softmax Dağılımı (Temperature Scaling)
$z_i$ logitleri ve $\tau > 0$ sıcaklık sabiti için:

$$p_i^\tau = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

- $\tau = 1$: Standart Softmax (baskın sınıf öne çıkar).
- $\tau \to \infty$: Düzgün (uniform) dağılım $p_i \to \frac{1}{K}$.
- $\tau \in [3, 6]$: Sınıflar arası **Dark Knowledge** ilişkileri dengeli bir şekilde belirir.

### 2. Kullback-Leibler (KL) Diverjansı Yumuşak Kayıp
$$D_{\text{KL}}(p_S^\tau \parallel p_T^\tau) = \sum_{k=1}^K p_{T, k}^\tau \log\left( \frac{p_{T, k}^\tau}{p_{S, k}^\tau} \right) = \sum_{k=1}^K p_{T, k}^\tau \log p_{T, k}^\tau - \sum_{k=1}^K p_{T, k}^\tau \log p_{S, k}^\tau$$

### 3. $\tau^2$ ile Ölçeklenmiş Bileşik Kayıp Fonksiyonu
$$\mathcal{L}_{\text{KD}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(y_{\text{true}}, z_S) + \alpha \cdot \tau^2 \cdot D_{\text{KL}}\big(\text{Softmax}(z_S / \tau) \parallel \text{Softmax}(z_T / \tau)\big)$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Knowledge Distillation** | *Bilgi Damıtma* | Büyük bir öğretmen modelin zekasını ve temsil gücünü kompakt bir öğrenci modele aktarma tekniği. |
| **Dark Knowledge** | *Karanlık Bilgi* | Yanlış sınıflara atanan küçük olasılıkların içinde gizlenmiş semantik ve geometrik akrabalık bilgisi. |
| **Temperature ($\tau$)** | *Sıcaklık Katsayısı* | Softmax'ten önce logitleri bölerek olasılık dağılımının sivrilik derecesini kontrol eden katsayı. |
| **KL-Divergence** | *Kullback-Leibler Ayrımı* | İki olasılık dağılımı ($P$ ve $Q$) arasındaki göreli entropi ve bilgi farkını ölçen istatistiksel metrik. |
| **Hard vs Soft Targets**| *Sert ve Yumuşak Hedefler* | One-hot $[0, 1, 0]$ kesin etiketler sert; öğretmenden gelen $[0.05, 0.85, 0.10]$ dağılımı yumuşak hedeftir. |
| **Capacity Gap** | *Kapasite Uçurumu* | Öğretmen ve öğrenci arasındaki parametre farkının aşırı büyük olmasından kaynaklanan öğrenme tıkanıklığı. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Kompakt modeller tek başına ulaşamayacağı doğruluk seviyelerine çıkar; Çıkarımda 0 ms ek gecikme. |
| **Weaknesses (Zayıf Yönler)** | İki aşamalı eğitim maliyeti (önce büyük öğretmen eğitilmelidir); $\tau$ ve $\alpha$ hiperparametre hassasiyeti. |
| **Opportunities (Fırsatlar)** | Mobil, IoT ve Edge AI cihazlarında devasa modellerin zekasını çalıştırma; Kuantizasyon ve Budama ile birleşim. |
| **Threats (Tehditler)** | Eğer öğretmen yetersiz veya aşırı uydurmuşsa öğrenciye yanlış bilgi damıtılır. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-82-knowledge-distillation/`](.) dizinindedir:

### A. Bilgi Damıtma Kayıp Fonksiyonu (PyTorch)
Dosya: [`src/kayip_damitma.py`](src/kayip_damitma.py)
```python
class BilgiDamitmaKaybi(nn.Module):
    def __init__(self, sicaklik: float = 4.0, alfa: float = 0.7):
        super().__init__()
        self.sicaklik = sicaklik
        self.alfa = alfa
        self.kl_kayip_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, ogrenci_logitleri, ogretmen_logitleri, gercek_etiketler):
        tau = self.sicaklik
        ce_kaybi = F.cross_entropy(ogrenci_logitleri, gercek_etiketler)
        
        ogrenci_yumusak_log = F.log_softmax(ogrenci_logitleri / tau, dim=-1)
        ogretmen_yumusak_prob = F.softmax(ogretmen_logitleri / tau, dim=-1)
        kl_kaybi = self.kl_kayip_fn(ogrenci_yumusak_log, ogretmen_yumusak_prob) * (tau ** 2)

        toplam_kayip = (1.0 - self.alfa) * ce_kaybi + self.alfa * kl_kaybi
        return toplam_kayip, {"ce_kaybi": ce_kaybi.item(), "kl_kaybi": kl_kaybi.item()}
```

---

## 📊 Deneysel Sonuçlar ve Doğrulama Çıktıları

`ana_akis.py` çalıştırılarak elde edilen analitik eğitim ve model karşılaştırması:

```text
=====================================================================================
🚀 Day 82: Öğretmen-Öğrenci Modeli Bilgi Damıtma (Knowledge Distillation)
=====================================================================================
  ✓ Öğretmen Parametre Sayısı: 1,973,162 (%100.00 Doğruluk)
  ✓ Öğrenci Parametre Sayısı: 35,466 (%99.38 Doğruluk)
  ✓ Model Sıkıştırma Oranı: 55.6x Daha Hafif Model!
  ✓ 6 Panelli Teşhis Panosu Kaydedildi: ciktilar/knowledge_distillation_paneli.png
```

- **55.6x Model Sıkıştırma:** 1.97M parametreli devasa öğretmen modeli, yalnızca 35k parametreli mini öğrenci modele %99.38 doğruluk korunarak başarıyla damıtılmıştır.
- **Birim Test Güvencesi:** [`testler/test_damitma.py`](testler/test_damitma.py) altındaki **8/8 birim test %100 PASSED (5.05s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/knowledge_distillation_paneli.png`](ciktilar/knowledge_distillation_paneli.png) konumundadır:

1. **Knowledge Distillation Hesaplama Akışı:** Öğretmen ve öğrenci logitlerinin $\tau$ ile yumuşatılarak KL kaybına dönüşüm şeması.
2. **Sıcaklık Katsayısının ($\tau$) Etkisi:** Farklı sıcaklıklarda ($\tau=1..12$) Dark Knowledge olasılıklarının açığa çıkışı.
3. **Model Parametre Kapasitesi:** 1.97M Öğretmen vs 35k Öğrenci karşılaştırması ($55\times$ küçülme).
4. **Öğrenci Doğruluk Kıyaslaması:** Bağımsız Öğrenci vs Damıtılmış Öğrenci vs Öğretmen eğrileri.
5. **Damıtma Kayıp Bileşenleri:** Sert CE ve yumuşak KL kayıplarının epoklar boyunca ilerlemesi.
6. **Knowledge Distillation SWOT Karar Matrisi:** Mühendislik karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Yalnızca son sınıflandırma logitlerini değil, Öğretmen ve Öğrencinin **orta katman öznitelik haritalarını (Feature Map Distillation / FitNets - Romero et al.)** da eşleştiren bir $1 \times 1$ adaptasyon konvolüsyonlu ara katman kaybı modülü yazınız.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AraKatmanOzNitelikDamitmaKaybi(nn.Module):
    """FitNets: Intermediate Feature Distillation with 1x1 Conv Projector"""
    def __init__(self, ogrenci_kanali: int, ogretmen_kanali: int):
        super().__init__()
        self.projeksiyon = nn.Conv2d(ogrenci_kanali, ogretmen_kanali, kernel_size=1, bias=False)

    def forward(self, ogrenci_feat: torch.Tensor, ogretmen_feat: torch.Tensor) -> torch.Tensor:
        # ogrenci_feat: (B, C_s, H, W), ogretmen_feat: (B, C_t, H, W)
        proj_s = self.projeksiyon(ogrenci_feat)
        return F.mse_loss(proj_s, ogretmen_feat)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Knowledge Distillation kayıp fonksiyonunda KL-Diverjansı terimi neden $\tau^2$ (sıcaklığın karesi) ile çarpılmak zorundadır? $\tau^2$ ile çarpılmazsa model eğitiminde ne gibi bir sorun yaşanır?

> **Mentor Cevabı:**
> 1. **Gradyan Sönümlenmesi (Gradient Vanishing):** Logitler $\tau$'ya bölündüğünde ($z / \tau$), Softmax'in türevi ve KL-Diverjansının $z_S$'e göre gradyanı $\frac{\partial \mathcal{L}_{\text{KL}}}{\partial z_S} \approx \frac{1}{\tau^2} (p_S^\tau - p_T^\tau)$ haline gelir. Yani sıcaklık $\tau=4$ yapıldığında geriye akan gradyanlar tam **$16$ kat zayıflar.**
> 2. **Kayıp Dengesinin Korunması:** Eğer kayıp $\tau^2$ ile çarpılmazsa, sert etiket Cross-Entropy kaybının gradyanları yumuşak damıtma kaybının gradyanlarını tamamen ezer ve model öğretmenin yumuşak hedeflerini öğrenemez. $\tau^2$ çarpımı, sıcaklık ne seçilirse seçilsin gradyan büyüklüğünün standart CE ile aynı ölçekte kalmasını sağlar.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
