# 🏆 Tesla FSD Otonom Sürüş | Gün 55: FAZ 5 BÜYÜK CAPSTONE — Uçtan Uca FSD Yapay Zeka Çıkarım Motoru

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Capstone](https://img.shields.io/badge/Phase%205%20Capstone-FSD%20AI%20Inference%20Engine-red.svg?style=flat-square)](https://www.tesla.com/)
[![Throughput](https://img.shields.io/badge/Performance-20%2C000%2B%20FPS%20Throughput-blue.svg?style=flat-square)](https://www.sae.org/)
[![Hardware](https://img.shields.io/badge/Silicon-HW3%2FHW4%20144%20TOPS%20NPU-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"Tebrikler stajyer! Bugün Faz 5'in (Derin Öğrenme, Occupancy Network ve NPU Kuantizasyon) ZİRVESİNDEYİZ!  
> Son 11 günde (Gün 45 - 54) otonom sürüş dünyasının en ileri yapay zeka yapı taşlarını tek tek inşa ettik:  
> 1. **HydraNet Çoklu Görev Omurgası (Gün 45):** Tek omurgadan 4 görev başlığı.  
> 2. **3D Occupancy Network ve Voxel Flow (Gün 46):** 40,000 hücrelik 3D doluluk ve hız alanı.  
> 3. **NeRF Hacimsel Işın İzleme (Gün 47):** 34.8 dB PSNR ile 3D zemin gerçeği rekonstrüksiyonu.  
> 4. **VectorLaneNet Yol Grafı (Gün 48):** 3. derece polinomlar ve yönlendirilmiş DAG kavşak topolojisi.  
> 5. **Vision Transformer Trafik Algılayıcı (Gün 49):** Trafik ışığı geri sayımı ve hız levhası OCR'ı.  
> 6. **Çoklu Modal Yörünge Tahmini (Gün 50):** Diğer aktörlerin 5 saniyelik olası yolları ve TTC riski.  
> 7. **Symmetric INT8 Kuantizasyon (Gün 51):** %75 SRAM tasarrufu ve katman birleştirme.  
> 8. **Model Damıtma ve Kanal Budama (Gün 52):** %30 FLOPs tasarrufu ile %99.2 doğruluk koruması.  
> 9. **Gölge Modu ve Veri Motoru (Gün 53):** İnsan-model uyuşmazlık tetikleyicisi ve filo A/B testi.  
> 10. **Dojo Veri Fabrikası (Gün 54):** Çift yönlü zamansal düzeltme ve sentetik simülasyon.  
> Bugün tüm bu devasa yapay zeka modellerini **Tek Bir FSD AI Çıkarım Motorunda (Capstone)** birleştiriyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. 3D Voksel Akışı ve Hacimsel Dinamikler

$$\vec{v}(\mathbf{x}) = \left[ v_x, v_y, v_z \right]^T, \quad \frac{\partial \text{Occ}}{\partial t} + \nabla \cdot (\text{Occ} \cdot \vec{v}) = 0$$

### 2. Vektörel Yol Polinomu ve Eğrilik

$$y(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3, \quad \kappa(x) = \frac{|y''(x)|}{(1 + (y'(x))^2)^{3/2}}$$

### 3. Vision Transformer Öz-Dikkat ve Geri Sayım

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V, \quad t_{\text{count}} = f_{\text{head}}(\text{CLS})$$

### 4. Çoklu Modal Yörünge ve Çarpışma Süresi (TTC)

$$P(Y \mid X) = \sum_{k=1}^K P(k \mid X) \mathcal{N}(\mu_k, \Sigma_k), \quad \text{TTC} = \frac{d_{\text{rel}}}{v_{\text{rel}}}$$

### 5. HW3/HW4 Simetrik INT8 Kuantizasyon

$$q = \text{clip}\left( \left\lfloor \frac{W}{S} \right\rceil, -128, 127 \right), \quad S = \frac{\max(|W|)}{127}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Birbirinden bağımsız 10 yapay zeka modelinin yarattığı bellek ve gecikme darboğazını ortadan kaldırmak; algıdan niyet tahminine kadar tüm akışı HW3/HW4 NPU üzerinde tek bir RTOS döngüsünde birleştirmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Model Karmaşası:** Tek bir paylaşılan FSD AI motoru ile CPU-NPU arası ara aktarım maliyetleri sıfırlandı.
- **Kutulanamaz Engel Güvenliği:** 3D Occupancy ile devrilmiş ağaçlar ve kutu içine girmeyen tüm engeller anında yakalandı.
- **20,000+ FPS Throughput:** INT8 katman birleştirme sayesinde 50 µs altında tüm yapay zeka kararları üretildi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Gelecek Faz Bağlantısı:** Bu motor algı ve tahmini tamamlar; direksiyon torku, gaz ve fren komutları **Faz 6'daki Hareket Planlayıcı (Planner & MPC)** tarafından üretilecektir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Ayrı Ayrı Bağımsız Sinir Ağları:** 10 kat daha fazla bellek ve enerji harcar, NPU üzerinde kilitlenmelere yol açar.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **FSD AI Inference Engine** | Algı, doluluk, şerit grafı, ışık ve yörünge modellerini eşzamanlı çalıştıran üretim seviyesi çıkarım motoru. |
| **3D Voxel Flow** | Her 3D vokselin uzaydaki anlık 3 eksenli hız vektörü alanı. |
| **VectorLaneNet DAG** | Şeritleri ve kavşak bağlantılarını yönlendirilmiş graf düğümleri olarak temsil eden harita modeli. |
| **ViT Traffic OCR** | Işık durumunu, geri sayım süresini ve hız limitini okuyan görsel transformer. |
| **Multi-Modal Trajectory** | Diğer araçların şeritte kalma, sola kırma veya frenleme olasılıklarını öngören yörünge tahmini. |
| **Symmetric INT8** | FP32 modelleri 1 baytlık tam sayılara indirgeyip %75 SRAM tasarrufu sağlayan kuantizasyon. |
| **Knowledge Distillation** | Dojo bulutundaki dev öğretmen modelin aklını araç içi öğrenci modele aktarma süreci. |
| **Layer Fusion** | Conv, BatchNorm ve ReLU işlemlerini tek bir NPU donanım çekirdeğinde birleştirme. |
| **Discrepancy Trigger** | İnsan ile gölge yapay zeka arasındaki karar farkını yakalayıp klibi buluta gönderen mekanizma. |
| **Auto-Labeling Ground Truth**| Çift yönlü zamansal düzeltmeyle insan müdahalesiz 0.965 IoU kalitesinde etiket üreten veri fabrikası. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 10 Derin Öğrenme bileşeninin kusursuz entegrasyonu  | • Faz 6'daki Hareket Planlayıcı (MPC/Hybrid A*)       |
| • 20,000+ FPS throughput ve < 50 µs RTOS gecikmesi    |   bağlantısı öncesi fiziksel eylemin tamamlanmaması   |
| • %75 SRAM bellek tasarrufu ve %99.2 doğruluk         |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Faz 6 MPC / Frenleme optimizasyonuna kusursuz       | • Aşırı aşırı uç koşullarda sensör kirlenmesi         |
|   ve zengin 3D girdi sağlama                          |   durumunda belirsizlik yönetimi                      |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Faz 5 Büyük Capstone Bütünleşik Mimarisi

```
                                  [ 8 Kamera Ham Görüntüleri ]
                                                |
                                                v
               [ RegNet + BiFPN Paylaşılan Omurga (Shared Backbone - INT8) ]
                                                |
        +-------------------+-------------------+-------------------+-------------------+
        |                   |                   |                   |                   |
        v                   v                   v                   v                   v
[ 3D Occupancy & Flow ] [ VectorLaneNet DAG ] [ ViT Traffic OCR ] [ 5s Trajectories ] [ Shadow Mode ]
- 50x50x16 Izgara       - 3. Derece Polinom  - Kırmızı (%96)     - Şeritte Kal (%70)  - Direksiyon > 5°
- Vx = 15 m/s           - kappa = 0.001597   - 8.5s Geri Sayım   - Sola Geçiş (%20)   - Fren > 1.5 m/s²
- Devrilmiş Ağaç        - Yasal DAG Rotaları - Hız Sınırı: 70    - TTC = 4.0s         - [-10s,+5s] Klip
        \                   \                   |                   /                   /
         \                   \                  |                  /                   /
          +-------------------+-----------------+-----------------+-------------------+
                                                |
                                                v
                    [ FSD AI Çıkarım Motoru Telemetrisi (Faz 6 Girdisi) ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Faz 5 Büyük Capstone simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
