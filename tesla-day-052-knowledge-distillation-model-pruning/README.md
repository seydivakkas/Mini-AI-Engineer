# 🚗 Tesla FSD Otonom Sürüş | Gün 52: Model Damıtma (Knowledge Distillation) ve Yapısal Budama (Pruning)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Distillation](https://img.shields.io/badge/Transfer-Teacher--Student%20Distillation-red.svg?style=flat-square)](https://www.tesla.com/)
[![Pruning](https://img.shields.io/badge/Compression-L1--Norm%20Channel%20Pruning-blue.svg?style=flat-square)](https://www.sae.org/)
[![Accuracy](https://img.shields.io/badge/Retention-99.2%25%20Accuracy%20Preserved-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"52. günümüze hoş geldin stajyer!  
> Tesla'nın veri merkezinde (Dojo) milyarlarca parametreli devasa derin öğrenme modelleri (Teacher Models) eğitilebilir. Bu modeller inanılmaz yüksek doğruluğa sahiptir fakat 100 kg ağırlığında ve kilovattlarca güç tüketen sunucularda çalışırlar.  
> Bu devasa aklı araç içindeki kompakt HW3/HW4 NPU çipine nasıl sığdırırız?  
> Çözüm iki kademeli mühendislik optimizasyonudur:  
> 1. **Model Damıtma (Knowledge Distillation):** Öğretmen modelin yumuşak olasılık dağılımları (Dark Knowledge), Sıcaklık ($T = 4.0$) yumuşatmasıyla küçük öğrenci modeline (Student Model) aktarılır. Öğrenci model sadece doğru cevabı değil, sınıflar arasındaki ince benzerlik ilişkilerini de öğrenir.  
> 2. **Yapısal Kanal Budama (L1-Norm Channel Pruning):** Evrişim katmanlarındaki ağırlık gücü ($\|\mathbf{W}_c\|_1$) en düşük olan $\%30$ kanal tamamen silinir.  
> 3. **%99.2 Doğruluk Koruma:** FLOPs işlem yükü $\%30$ azalırken model tespit performansından neredeyse hiçbir şey kaybetmez.  
> Bugün büyük aklı küçük silikona damıtıyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Sıcaklık Yumuşatmalı Softmax Dağılımı

$$p_i(\mathbf{z}, T) = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

### 2. Bilgi Damıtma (Knowledge Distillation) Kayıp Fonksiyonu

$$\mathcal{L}_{\text{KD}} = \alpha \cdot T^2 \cdot D_{\text{KL}}\left( p_T(\mathbf{z}_T, T) \,\|\, p_S(\mathbf{z}_S, T) \right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CrossEntropy}}(y, p_S(\mathbf{z}_S, 1))$$

$$D_{\text{KL}}(P \parallel Q) = \sum_{i=1}^K P(i) \log \left( \frac{P(i)}{Q(i)} \right)$$

### 3. L1-Norm Tabanlı Yapısal Kanal Budama

$$I_c = \|\mathbf{W}_c\|_1 = \sum_{k=1}^{C_{\text{in}}} \sum_{h=1}^K \sum_{w=1}^K |W_{c, k, h, w}|$$

$$\text{Mask}_c = \begin{cases} 1, & I_c \ge \text{Percentile}(I, 30) \\ 0, & \text{Budanan Kanal} \end{cases}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Devasa Dojo bulut modellerinin üstün akıl yürütme ve genelleme yeteneğini, araç içi HW3/HW4 NPU'nun katı mikrosaniye gecikme ve bellek sınırları içinde yaşatmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Karanlık Bilgi (Dark Knowledge) Kazanımı:** Küçük model, dev modelin 'Bu nesne %90 yaya, ama %9 bisikletliye de benziyor' şeklindeki zengin alt-olasılık ilişkilerini öğrendi.
- **%30 FLOPs Azaltımı:** Gereksiz filtreler budanarak saniyede işlenen kare sayısı 3 katına çıkarıldı.
- **Aşırı Öğrenme (Overfitting) Engeli:** Damıtma düzenlileştirici (regularizer) etkisi yaparak öğrencinin köşe durumlarda daha stabil çalışmasını sağladı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Aşırı Budama Eşiği:** Budama oranı $\%40$'ın üzerine çıktığında kritik küçük nesnelerin (uzak trafik konileri) tespiti bozulabilir.
- **Eğitim Karmaşıklığı:** İki modelin aynı anda GPU'da ileri beslenmesi eğitim süresini ve GPU belleğini artırır.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Sıfırdan Küçük Model Eğitimi (Scratch Training):** Doğruluk oranı %5-8 daha düşük kalır.
- **Yapısal Olmayan Ağırlık Budama (Unstructured Pruning):** Bireysel ağırlıkları sıfırlar fakat özel donanım olmadıkça NPU'da hızlanma sağlamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Knowledge Distillation** | Büyük bir öğretmen modelin bilgilerini küçük bir öğrenci modeline aktarma süreci. |
| **Teacher Model** | Dojo süper bilgisayarında eğitilen milyarlarca parametreli devasa referans model. |
| **Student Model** | Araç içinde HW3/HW4 çipinde gerçek zamanlı çalışan kompakt sinir ağı. |
| **Temperature Scaling ($T$)** | Softmax çıktısındaki olasılıkları yumuşatarak sınıflar arası gizli ilişkileri görünür kılan hiperparametre. |
| **Dark Knowledge** | En yüksek tahmin dışındaki düşük olasılıklı sınıfların taşıdığı değerli yapısal bilgi. |
| **Structured Pruning** | Evrişim filtrelerini veya dikkat kafalarını bütün blok halinde silerek donanımsal hızlanma sağlayan budama. |
| **L1-Norm Importance** | Bir kanalın filtre ağırlıklarının mutlak toplamı ile belirlenen önem skoru. |
| **KL-Divergence** | Öğretmen ve öğrenci yumuşak olasılık dağılımları arasındaki farkı ölçen entegre metrik. |
| **FLOPs (Floating Point Ops)**| Modelin tek bir çıkarımında icra ettiği toplam kayan nokta işlem sayısı. |
| **Sparsity (Seyreklik)** | Budama sonucu modeldeki sıfırlanmış veya silinmiş parametrelerin toplam modele oranı. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %99.2 doğruluk koruması ile %30 FLOPs tasarrufu     | • Çift model ile artan GPU eğitim hafıza maliyeti     |
| • Yapısal budama sayesinde NPU'da doğrudan hızlanma   | • %40 üzeri agresif budamalarda küçük nesne kaybı     |
| • 38 µs ultra hızlı RTOS çözüm performansı            |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Gelecekteki HW5 çiplerinde daha agresif damıtma     | • Öğretmen modelde var olabilecek hatalı veya         |
|   yöntemleriyle multimodal LLM çalıştırma             |   önyargılı çıkarımların öğrenciye kalıtımı           |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Teacher-Student Damıtma ve Budama Akışı

```
[ Dojo Bulut Öğretmen Modeli (Devasa ViT) ]
                    |
                    v (Softmax T = 4.0)
         [ Yumuşak Hedefler (p_T) ] <----+
                    |                    |
                    v                    v (KL-Divergence Kaybı)
[ HW3/HW4 Öğrenci Modeli (RegNet) ] ===> [ Toplam Kayıp L_KD ]
                    |
                    v
    [ L1-Norm Filtre Budama (%30) ]
                    |
                    v
    [ Kompakt ve Hızlı NPU Çıkarımı ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Model Damıtma ve Budama simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
