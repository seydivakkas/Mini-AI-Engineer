# 🚗 Tesla FSD Otonom Sürüş | Gün 54: Tesla Veri Fabrikası (Otomatik 3D Yörünge ve Sentetik Veri Üretim Hattı)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![AutoLabeling](https://img.shields.io/badge/DataFactory-Tesla%20Auto--Labeling%20Pipeline-red.svg?style=flat-square)](https://www.tesla.com/)
[![Bidirectional](https://img.shields.io/badge/Smoothing-Bidirectional%20Temporal%20Fusion-blue.svg?style=flat-square)](https://www.sae.org/)
[![IoU](https://img.shields.io/badge/Quality-0.965%203D%20IoU%20Ground%20Truth-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"54. günümüze hoş geldin stajyer!  
> Araç yolda giderken 'nedensellik kısıtı' (Causality) vardır; yani araç geleceği ($t+1, t+2$) göremez, sadece geçmişi bilir. Bu yüzden araç içi çevrimiçi algılama gürültülüdür ve titreşim içerir.  
> Ancak video klip Tesla Dojo süper bilgisayarına yüklendiğinde artık nedensellik kısıtı ortadan kalkar: Tüm sürüşün başlangıcı ve sonu bulutun hafızasındadır!  
> Tesla bu süper bilgisayar gücünü **Otomatik Etiketleme Fabrikası (Auto-Labeling Pipeline)** ile zemin gerçeği üretmek için kullanır:  
> 1. **Çift Yönlü Zamansal Düzeltme (Bidirectional Temporal Smoothing):** Hem geçmişten ileriye hem de gelecekten geriye doğru çalışan filtreler nesne yörüngelerini titreşimsiz, milimetre hassasiyetinde pürüzsüzleştirir.  
> 2. **Çoklu Sürüş Hizalama (Multi-Trip Alignment):** Farklı Tesla'ların aynı yoldan gündüz, gece, yağmurda geçişleri üst üste bindirilerek statik zemin haritası ve şerit koordinatları kusursuzca çıkarılır.  
> 3. **Sentetik Sahne Üretimi (Synthetic Augmentation):** Güneşli bir günde kaydedilen video sahnesi NeRF/Diffusion motorunda geceye, yoğun sise veya sağanak yağmura dönüştürülerek modelin köşe durum dayanıklılığı artırılır.  
> 4. **0.965 IoU Kalitesi:** İnsan etiketçilerin aylar süren çalışması sıfır maliyetle saniyeler içinde aşılır.  
> Bugün Tesla'nın veri imparatorluğunun üretim bandını kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Çift Yönlü Zamansal Düzeltme (Bidirectional Smoothing)

$$x_t^* = \sum_{k=-K}^K w_k \cdot x_{t+k}, \quad \sum_{k=-K}^K w_k = 1.0$$

### 2. Çoklu Sürüş Nokta Bulutu Eşleşmesi (Iterative Closest Point - ICP)

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^N \left\| \mathbf{p}_i^{(1)} - \left( \mathbf{R} \mathbf{p}_i^{(2)} + \mathbf{t} \right) \right\|^2$$

### 3. 3D Bounding Box Hacimsel Kesişim Oranı (IoU)

$$\text{IoU}_{3\text{D}} = \frac{\text{Vol}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Vol}(B_{\text{pred}} \cup B_{\text{gt}})} = \frac{\text{Vol}(B_{\text{inter}})}{\text{Vol}(B_{\text{pred}}) + \text{Vol}(B_{\text{gt}}) - \text{Vol}(B_{\text{inter}})}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Milyarlarca saatlik FSD sürüş klibini insan müdahalesine gerek kalmadan süper bilgisayarda otomatik etiketleyip, modelleri besleyen sonsuz bir veri döngüsü kurmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Zamansal Titreme (Temporal Jitter):** Çevrimiçi kamerada anlık kaybolan veya titreşen nesneler ileri-geri enterpolasyonla kalıcı ve sabit kutulara bağlandı.
- **Karanlık ve Sis Sınırı:** Gündüz temiz kaydedilen yol haritası gece çekilen zorlu klibe otomatik zemin gerçeği olarak aktarıldı (Multi-Trip Mapping).
- **Sentetik Veri Çeşitliliği:** Nadir görülen ekstrem hava koşulları yapay olarak üretilerek kaza riski olmadan model eğitildi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **GPS / Poz Kayması:** Çoklu sürüşlerin birleştirilmesinde araç poz kestiriminde sürüklenme (drift) varsa ICP hizalama süresi uzayabilir.
- **Süper Bilgisayar Enerjisi:** Milyonlarca klibin çift yönlü NeRF/ICP süreçleri yüksek elektrik ve veri depolama hacmi tüketir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Hindistan/Meksika Dış Kaynaklı İnsan Etiketleme:** 3D uzayda insan derinliği kestiremez, fahiş maliyetlidir ve son derece yavaştır.
- **Sentetik Oyun Motorları (Unreal/CARLA):** 'Sim-to-Real' uçurumu vardır; gerçek kamera lens bozulmalarını tam taklit edemez.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Auto-Labeling** | Süper bilgisayarda video kliplerinden insan müdahalesiz 3D zemin gerçeği kutuları ve şeritler çıkarma hattı. |
| **Bidirectional Smoothing**| Gelecek ve geçmiş video karelerini aynı anda kullanarak nesne pozisyonlarını pürüzsüzleştiren zamansal filtre. |
| **Multi-Trip Alignment** | Farklı zamanlarda aynı sokaktan geçen araçların 3D algılarını harita koordinatlarında eşleştirme. |
| **3D Box IoU** | İki 3D kutunun hacimsel kesişiminin birleşimine oranı ile ölçülen etiketleme doğruluk metriği. |
| **Synthetic Augmentation**| Gerçek videolara fiziksel tabanlı yağmur, sis, kar veya gece ışık efektleri ekleyen jeneratör. |
| **Iterative Closest Point (ICP)**| İki 3D nokta bulutu arasındaki en iyi rotasyon ve öteleme matrisini bulan optimizasyon algoritması. |
| **Ground Truth (Zemin Gerçeği)**| Derin öğrenme modelinin eğitildiği kesin doğru kabul edilen etiket kümesi. |
| **Sim-to-Real Gap** | Simülasyonda eğitilen modellerin gerçek fiziksel dünyada başarısız olmasına yol açan gerçeklik farkı. |
| **Offline Optimization** | Araç içinde gerçek zamanlı çalışma zorunluluğu olmadan bulutta yüksek hassasiyetle icra edilen hesaplama. |
| **Dojo Data Engine** | Tesla'nın otomatik etiketleme, model damıtma ve yeniden eğitim döngüsünü yöneten yapay zeka fabrikası. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 otomatik, insan maliyeti sıfır ($0.00)         | • Devasa Dojo bulut depolama ve bant genişliği        |
| • 0.965 yüksek 3D IoU zemin gerçeği kalitesi          | • Aşırı GPS sapmalarında ICP hizalama zamanı          |
| • Sentetik hava/ışık varyasyonları ile zenginleştirme |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tüm küresel Tesla filosu verilerini tek bir devasa  | • Dinamik olarak değişen yol yapılarında (yeni asfalt)|
|   dünya modelinde birleştirme                         |   eski sürüş verilerinin güncelliğini yitirmesi       |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Otomatik Etiketleme Hattı Mimarisi

```
[ Filo Video Klipleri (Gölge Modu Tetiklemeleri) ]
                       |
                       v
[ Çift Yönlü Zamansal Düzeltme (Bidirectional Smoothing) ]
                       |
                       v
[ Çoklu Sürüş ICP Hizalaması (Multi-Trip Mapping) ]
                       |
        +--------------+--------------+
        |                             |
        v                             v
[ 3D BBox IoU Doğrulaması ]   [ Sentetik Hava Durumu Çeşitlendirme ]
- 0.965 Kalite Skoru           - Yağmur, Sis, Gece
        \                             /
         \                           /
          v                         v
     [ Milyonlarca Saatlik 3D Zemin Gerçeği Veri Seti ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Otomatik Etiketleme simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
