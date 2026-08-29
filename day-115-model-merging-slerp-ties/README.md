# Day 115: Model Merging (SLERP, TIES-Merging, DARE) ile Sıfır GPU Eğitimli Model Füzyonu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; hiçbir GPU eğitimi ve geriye yayılım (backpropagation) gerektirmeden, ortak bir temel modelden türetilmiş uzman modellerin (Matematik, Kodlama, Çok Dilli vb.) ağırlıklarını geometrik ve uzamsal olarak birleştiren **Model Merging (SLERP, TIES-Merging, DARE)** tekniklerini sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Sıfır GPU ile Model Füzyonu: Frankenstein'dan Süper-Modellere"

Diyelim ki elinizde aynı Llama-3-8B temel modelinden fine-tune edilmiş iki farklı model var:
- Model A: Harika bir **Matematikçi** (ama kod yazmayı unutmuş).
- Model B: Harika bir **Yazılımcı** (ama matematikten anlamıyor).

Bu iki modeli tek bir "Süper Model" haline getirmek için normalde yüz binlerce dolar harcayıp iki veri setini birleştirerek sıfırdan eğitmeniz gerekirdi.

**Model Merging (Model Füzyonu)** ise bunu saniyeler içinde **$0 maliyetle (sıfır GPU eğitimiyle)** yapar:
1. 📐 **Task Vector (Görev Vektörü):** Bir modelin taban modele göre öğrendiği bilgi farkıdır ($\tau = \theta - \theta_{\text{base}}$).
2. 🌐 **SLERP (Küresel Enterpolasyon):** Ağırlıkları düz bir çizgiyle değil, yüksek boyutlu bir kürenin yüzeyi boyunca eğri olarak birleştirir (açıyı ve büyüklüğü korur).
3. 🤝 **TIES-Merging (Mutabakat):** Parametreler birbirine zıt yönleri gösteriyorsa (işaret çatışması), çoğunluğun oyuna göre karar verir ve küçük gürültüleri budar (Trim).
4. 🎲 **DARE (Drop & Rescale):** Parametrelerin %70'ini rastgele atıp kalanları ölçekler; böylece birden fazla uzman model birbirine çarpmadan birleşir.

```
       GELENEKSEL EĞİTİM (PAHALI)                                   MODEL MERGING (SIFIR GPU - $0)
 ┌──────────────────────────────────────────────┐       ┌──────────────────────────────────────────────┐
 │ • Aylar süren eğitim & yüzlerce GPU          │       │ • Sıfır Eğitim: Sadece CPU/RAM'de birleştirme│
 │ • Felaket Derecede Unutma (Forgetting) riski │       │ • Süre: Birkaç saniye içinde hazır!          │
 │ • Maliyet: $100,000+                         │       │ • Maliyet: $0 GPU Cost                       │
 │ • Veri kümesi gereksinimi zorunlu            │       │ • Ham eğitim verisine ihtiyaç YOK!           │
 └──────────────────────────────────────────────┘       └──────────────────────────────────────────────┘
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Görev Aritmetiği (Task Arithmetic)
Aynı temel modelden ($\theta_{\text{base}}$) türeyen modellerin parametre farkı:
$$\tau_m = \theta_m - \theta_{\text{base}}$$
Doğrusal birleştirme: $\theta_{\text{merged}} = \theta_{\text{base}} + \sum_m \alpha_m \tau_m$.
Ancak doğrusal ortalama, parametre uzayında gürültü ve çakışma yaratarak modelin performansını düşürür.

### 2. SLERP (Spherical Linear Interpolation) ile Küresel Geometri Korunumu
İki model arasındaki kosinüs açısı $\Omega = \arccos\left(\frac{v_1 \cdot v_2}{\|v_1\| \|v_2\|}\right)$ olmak üzere:
$$\text{SLERP}(v_1, v_2; t) = \frac{\sin((1-t)\Omega)}{\sin \Omega} v_1 + \frac{\sin(t\Omega)}{\sin \Omega} v_2$$
SLERP, ağırlık vektörlerinin büyüklüğünü ve küresel rotasyonunu koruduğu için 2 model birleştirmede en yüksek kaliteyi verir.

### 3. TIES-Merging (TRIM, ELECT SIGN & MERGE)
3 veya daha fazla model birleştirildiğinde parametre işaret çatışmalarını çözer:
1. **Trim:** Her görev vektörünün en küçük mutlak değerli %30-50'sini sıfırla.
2. **Elect Sign:** Mutabakat işaretini belirle: $s_j = \text{sgn}\left(\sum_m \tau_{m,j}\right)$. Çelişenleri sıfırla.
3. **Disjoint Merge:** Uyuşan parametrelerin ortalamasını alarak taban modele ekle.

### 4. DARE (Drop And REscale) ve MergeKit Ekosistemi
DARE, görev vektörlerine $M \sim \text{Bernoulli}(1-p)$ maskesi uygular ve $\tilde{\tau} = \frac{1}{1-p} (M \odot \tau)$ ile yeniden ölçekler. Hugging Face Open LLM Leaderboard'daki lider modellerin (örn. Solar-10.7B, BioMistral) neredeyse tamamı MergeKit ve DARE-TIES ile inşa edilmiştir.

---

## 📊 Model Merging Yöntemleri Karşılaştırma Tablosu

| Yöntem | Model Sayısı | Geometrik Yaklaşım | Çakışma Yönetimi | En İyi Kullanım Alanı |
|:---|:---|:---|:---|:---|
| **Linear (Task Arithmetic)** | $M \ge 2$ | Düz Ağırlıklı Toplam | Yok (Yüksek Çakışma) | Hızlı prototipleme |
| **SLERP** | **Tam 2 Model** | Küresel Yüzey Enterpolasyonu | Açı Korunumu | 2 Genel Yetenekli Modeli Birleştirme |
| **TIES-Merging** | **$M \ge 3$ Model** | Budama & Ayrık Birleştirme | **İşaret Mutabakatı (Elect Sign)** | Çoklu Alan Uzmanlarını Birleştirme |
| **DARE-TIES** | **$M \ge 3$ Model** | Bernoulli Seyreltme + TIES | **Seyreltme & Mutabakat** | SOTA Liderlik Tablosu Modelleri |

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Model Merging** | Sıfır eğitim maliyetiyle birden fazla modelin ağırlıklarını uzamsal olarak birleştirme tekniği. |
| **Task Vector ($\tau$)** | Bir modelin uzmanlaştığı alanda taban modele göre geliştirdiği ağırlık farkı ($\theta - \theta_{\text{base}}$). |
| **SLERP** | İki ağırlık vektörünü küresel bir yay üzerinde enterpole eden trigonometrik algoritma. |
| **TIES-Merging** | Budama, işaret mutabakatı ve ayrık ortalama adımlarından oluşan çoklu model füzyon yöntemi. |
| **DARE** | Parametre farklarının büyük kısmını rastgele atıp kalanları ölçekleyen seyreltme yöntemi. |
| **Catastrophic Forgetting** | Bir modelin yeni bir veri setiyle eğitildiğinde eski bilgilerini tamamen unutması problemi. |
| **Sign Conflict (İşaret Çatışması)** | Farklı modellerin aynı parametre için zıt yönlerde (+ / -) gradyan adımları atması durumu. |
| **Interference (Girişim/Çakışma)** | Birden fazla modelin parametrelerinin birleşirken birbirini nötrlemesi veya bozması. |
| **MergeKit** | LLM dünyasında SLERP, TIES ve DARE birleştirmelerini standartlaştıran açık kaynak araç kütüphanesi. |
| **FrankenMerge** | Farklı katman derinliklerini ve mimarilerini birbirine yapıştırarak üretilen deneysel modeller. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Sıfır GPU eğitim maliyeti ($0 Cost)│ • Sadece aynı taban mimariden        │
 │ • Birkaç saniyede yeni model üretimi.│   türeyen modellerde çalışır.        │
 │ • Çok alanlı genel yetenek kazanımı. │ • Ağırlık oranları (t, alpha)        │
 │ • Felaket unutmayı önler.            │   deneme-yanılma gerektirebilir.     │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Open LLM Leaderboard şampiyonluğu. │ • Tamamen farklı tokenizer'a sahip   │
 │ • Özel domain modellerini birleştirme│   modeller doğrudan birleştirilemez. │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/model_merging_paneli.png` dosyası üretilir:
1. **Çok Alanlı Ortalama Başarı Skoru (0-100)**
2. **Matematik vs Kodlama Çapraz Yetenek Ayrışması**
3. **SLERP vs Linear Enterpolasyon Katsayıları (t: 0->1)**
4. **TIES-Merging (Trim, Elect Sign, Merge) İş Akışı**
5. **DARE Parametre Seyreltme Oranı vs Başarım**
6. **Model Merging Karar ve Entegrasyon Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Ana model birleştirme akışını koşturun
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
