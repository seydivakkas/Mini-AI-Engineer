# 🚗 Tesla FSD Otonom Sürüş | Gün 45: HydraNet Mimarisi (Paylaşılan Omurga ve Görev Kafaları)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![HydraNet](https://img.shields.io/badge/Architecture-Tesla%20Multi--Task%20HydraNet-red.svg?style=flat-square)](https://www.tesla.com/)
[![Backbone](https://img.shields.io/badge/Backbone-RegNet%20+%20BiFPN%20Pyramid-blue.svg?style=flat-square)](https://www.sae.org/)
[![Loss](https://img.shields.io/badge/Loss-Homoscedastic%20Uncertainty%20Weighing-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"45. günümüze ve 5. BÜYÜK FAZIMIZA hoş geldin stajyer!  
> Tesla FSD aracının aynı anda onlarca farklı görevi yerine getirmesi gerekir:  
> - Çevredeki araçları ve yayaları 3D kutularla tespit etmek,  
> - Şerit çizgilerini 3. derece polinomlarla takip etmek,  
> - Trafik ışıklarını ve yeşil ışık geri sayım sürelerini sınıflandırmak,  
> - Sürülebilir yol zeminini piksel piksel segmente etmek.  
> Eğer her görev için ayrı bir derin sinir ağı (Monolithic Network) çalıştırsaydık, Tesla FSD bilgisayarı (HW3/HW4 NPU) aşırı ısınıp çökerdi.  
> Tesla bunu **HydraNet (Çok Başlı Yılan)** mimarisiyle çözdü:  
> 1. **Paylaşılan Omurga (Shared Backbone):** RegNet ve BiFPN çok ölçekli öznitelik piramidi tüm görevler için görüntüyü **tek bir kez** işler.  
> 2. **Hafif Görev Kafaları (Task Heads):** Ortak öznitelik havuzundan beslenen küçük kafalar kendi özel tahminlerini paralel olarak yapar.  
> 3. **%70+ Hesaplama ve Bellek Tasarrufu:** 8 kamera için 32 ayrı model yerine yalnızca 8 omurga çıkarımı yapılır.  
> 4. **Homoscedastic Belirsizlik Ağırlıklı Kayıp:** Farklı ölçeklerdeki görev kayıpları otomatik dengelenir.  
> Bugün Tesla'nın en ünlü yapay zeka omurgasını inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Paylaşılan Omurga ve Çoklu Görev Çıkarımı

$$\mathbf{F}_{\text{shared}} = \text{BiFPN}\Big(\text{RegNet}(\mathbf{I}_{\text{camera}})\Big)$$

$$\hat{\mathbf{y}}_{\text{objects}} = \text{Head}_{\text{obj}}(\mathbf{F}_{\text{shared}}), \quad \hat{\mathbf{y}}_{\text{lanes}} = \text{Head}_{\text{lane}}(\mathbf{F}_{\text{shared}})$$

### 2. Homoscedastic Belirsizlik Ağırlıklı Çoklu Görev Kaybı

$$\mathcal{L}_{\text{total}}(\mathbf{W}, \sigma_1, \dots, \sigma_K) = \sum_{k=1}^K \left( \frac{1}{2 \sigma_k^2} \mathcal{L}_k(\mathbf{W}) + \log(\sigma_k) \right)$$

### 3. 3. Derece Şerit Polinom Modeli

$$y(x) = c_0 + c_1 \cdot x + c_2 \cdot x^2 + c_3 \cdot x^3$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Aracın üzerindeki 8 kamera için nesne, şerit, trafik ışığı ve zemin modellerini ayrı ayrı çalıştırmanın yaratacağı devasa NPU bellek ve işlem yükünü engellemek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **NPU Aşırı Yüklenmesi:** Tek bir omurga hesaplamasıyla 4 farklı görevin aynı anda çıkarılmasını sağlayarak $\%72$ işlem gücü kazandırdı.
- **Görevler Arası Transfer Öğrenmesi:** Şerit özniteliklerinin yol yüzeyi tespitine, nesne özniteliklerinin ise derinlik tahminine yardımcı olması sağlandı.
- **Dinamik Kafa Güncellemesi:** Yeni bir görev (örneğin hız levhası okuma) eklendiğinde tüm omurgayı baştan eğitmeden sadece ilgili kafayı eğitme esnekliği sağlandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Negative Transfer (Zararlı Etkileşim):** Bir görev kafasındaki hata gradyanı ortak omurga ağırlıklarını bozarak başka bir görevin doğruluğunu düşürebilir (Gradient Surgery / PCGrad gerekir).
- **Kapasite Darboğazı:** Omurga boyutu çok küçük tutulursa tüm görevlerin karmaşıklığını taşıyamaz.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Ayrık Monolitik Ağlar:** Her görev için bağımsız model. Çok yüksek GPU/NPU tüketir.
- **Uçtan Uca Monolitik Transformer:** Çok güçlüdür fakat ara görevlerin (araç, ışık, şerit) ayrı ayrı denetlenmesini zorlaştırır.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **HydraNet** | Tek bir paylaşılan omurgadan çok sayıda uzmanlaşmış görev kafası çıkaran çoklu görev mimarisi. |
| **Shared Backbone** | Görüntüden genel görsel öznitelikleri (kenarlar, dokular, şekiller) çıkaran ortak derin sinir ağı. |
| **Task Head** | Ortak öznitelikleri alıp belirli bir hedefe (şerit, nesne, ışık) dönüştüren hafif sinir ağı katmanı. |
| **BiFPN** | Çift yönlü öznitelik piramidi; farklı çözünürlükteki katmanları ağırlıklı olarak birleştiren yapı. |
| **Negative Transfer** | Bir görevin eğitiminin ortak omurgayı bozarak başka bir görevin başarımını düşürmesi problemi. |
| **Homoscedastic Loss** | Görevlerin kendi içsel belirsizliklerine ($\sigma_k$) göre dinamik ağırlıklandırılan çoklu kayıp fonksiyonu. |
| **Lane Polynomial** | Yol çizgilerinin eğriliğini $x$ boyuna mesafesine göre $y$ yanal konumunda ifade eden polinom. |
| **RegNet** | Tesla'nın NPU donanımında en yüksek bellek bant genişliği verimliliği sağlayan regüle omurga mimarisi. |
| **Gradient Surgery (PCGrad)** | Çoklu görev gradyanlarının birbirini zıt yönde iptal etmesini önleyen izdüşüm tekniği. |
| **Drivable Area Mask** | Aracın tekerlek basabileceği güvenli asfalt alanlarını belirten 2D olasılık matrisi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %72 NPU/GPU hesaplama ve bellek tasarrufu           | • Görevler arası gradyan çatışması (Negative Transfer)|
| • Modüler görev kafası ekleme/çıkarma yeteneği        | • Omurga çökerse tüm kafaların aynı anda körleşmesi   |
| • Ortak öznitelik zenginliği ve transfer öğrenme      |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • 3D Voxel Occupancy ve NeRF başlıkları ile           | • Donanım seviyesinde NPU tensör çekirdeği            |
|   uçtan uca dünya modeli desteği                      |   paralelleştirme optimizasyon zorluğu                |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla FSD HydraNet Mimarisi

```
                           [ 8 Kamera Girdisi ]
                                     |
                                     v
                      [ RegNet + BiFPN Paylaşılan Omurga ]
                                     |
         +-----------------+---------+---------+-----------------+
         |                 |                   |                 |
         v                 v                   v                 v
[ 3D Nesne Kafası ] [ Şerit Kafası ]   [ Trafik Işığı ]  [ Sürülebilir Alan ]
 - 3D BBox           - 3. Derece Poly   - Renk & Ok       - 2D Zemin Maskesi
 - Sınıf Olasılığı   - Sol/Sağ Şerit    - Geri Sayım      - Yol Sınırları
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana HydraNet çoklu görev simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
