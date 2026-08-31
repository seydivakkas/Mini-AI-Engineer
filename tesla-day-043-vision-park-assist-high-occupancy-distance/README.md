# 🚗 Tesla FSD Otonom Sürüş | Gün 43: Tesla Vision Park Asistanı ve Yüksek Çözünürlüklü Mesafe Kestirimi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![VisionPark](https://img.shields.io/badge/Architecture-No--USS%20Tesla%20Vision%20Park-red.svg?style=flat-square)](https://www.tesla.com/)
[![Occupancy](https://img.shields.io/badge/Grid-3D%20Voxel%20Occupancy%20%285cm%29-blue.svg?style=flat-square)](https://www.sae.org/)
[![BlindSpotMemory](https://img.shields.io/badge/Memory-Bumper%20Blind%20Spot%20Temporal%20Memory-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"43. günümüze hoş geldin stajyer!  
> Tesla 2022 yılı sonu itibarıyla araçlarındaki 12 adet ultrasonik park sensörünü (USS) kaldırdı. Sektör ilk başta 'Kameralar tamponun hemen altındaki alçak kaldırımları nasıl görecek?' diyerek şüpheyle yaklaştı.  
> Tesla bu problemi **High-Occupancy Network** ve **Kör Nokta Zamansal Belleği (Blind Spot Temporal Memory)** ile çözdü:  
> 1. **3D Voxel Doluluk Izgarası (5 cm Çözünürlük):** Kameralardan gelen derinlik ve 3D rekonstrüksiyon verisi aracın etrafında 5 santimetrelik voksel hücrelerine bölünür.  
> 2. **Kör Nokta Belleği:** Araç kaldırıma veya duvara yaklaşırken, nesne kameraların görüş açısından çıkıp ön/arka tamponun kör noktasına girdiğinde sistem bu engeli unutup silmez; aracın odometri hareketiyle ($\Delta x_{\text{ego}}$) geriye doğru kaydırarak hafızada tutmaya devam eder.  
> 3. **360° Işın Atma (Ray-Casting):** Araç gövdesinden dışarıya doğru $1^\circ$ aralıklarla ışınlar gönderilerek en yakın engel mesafesi ($d_{\min}(\theta)$) santimetre düzeyinde ölçülür.  
> 4. **Kademeli Uyarı ve STOP:** Mesafe $30\text{ cm}$ altına indiğinde sesli ve görsel 'STOP' protokolü devreye girer.  
> Bugün Tesla Vision Park Asistanı motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Voxel Doluluk Güncellemesi (Log-Odds Formülasyonu)

$$L(O_{x,y} \mid z_{1:t}) = L(O_{x,y} \mid z_{1:t-1}) + \log\left(\frac{P(O_{x,y} \mid z_t)}{1 - P(O_{x,y} \mid z_t)}\right)$$

### 2. Kör Nokta Zamansal Öteleme (Temporal Memory Warp)

$$\mathbf{P}_{\text{mem}, t} = \mathbf{P}_{\text{mem}, t-1} - \begin{bmatrix} \Delta X_{\text{ego}} \\ \Delta Y_{\text{ego}} \end{bmatrix}$$

### 3. 360 Derece Işın Atma (Ray-Casting) Mesafe Konturu

$$d_{\min}(\theta) = \min_{r \in [r_{\text{veh}}, r_{\max}]} \left\{ r - r_{\text{veh}} \mid \text{Grid}(r \cos\theta, r \sin\theta) > \tau_{\text{occ}} \right\}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Tamponlardaki ultrasonik sensör deliklerini, kablolama ağırlığını ve sensör donanım maliyetlerini ortadan kaldırarak saf görüntü işleme ile tam 360° park desteği sunmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Kaldırım Yüksekliği ve Şekli:** Noktasal USS sensörlerinin aksine, kaldırımın 3D uzantısını ve eğimini tam olarak modelledi.
- **Kör Nokta Görüş Kaybı:** Tampon altı engeller zamansal hafıza ile korunarak çarpma riski sıfırlandı.
- **Kesintisiz 360° Kontur:** Sadece 12 noktadan değil, aracın tüm çevresinden sürekli mesafe çizgisi üretti.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Park Halindeyken Önüne Konulan Nesneler:** Araç park halindeyken (kameralar kapalıyken) tamponun dibine konan bir kutu/çocuk ilk kalkışta görülemeyebilir.
- **Güneş Parlaması ve Çamur:** Arka kamera çamurla kaplandığında geri park hassasiyeti düşebilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Geleneksel Ultrasonik Sensörler (USS):** Donanım maliyeti ve kablolama yükü yüksektir; sadece yakın mesafede noktasal çalışır.
- **Çevre Görüş Kameralı 360 AVM (Around View Monitor):** Sadece 2D görüntü gösterir; metrik santimetre mesafesi hesaplayamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Tesla Vision Park Assist** | Ultrasonik sensörler olmadan sadece kameralar ve yapay zeka ile çalışan park yardım sistemi. |
| **High-Occupancy Network** | 3D uzaydaki boş ve dolu vokselleri santimetre çözünürlükte tahmin eden derin sinir ağı. |
| **Voxel Grid** | 3D uzayı küçük küplere bölen hacimsel ızgara temsil yapısı (Tesla için $5\text{ cm} \times 5\text{ cm}$). |
| **Blind Spot Memory** | Kameraların göremediği tampon altı bölgelerdeki engelleri hafızada tutan zamansal bellek. |
| **Ray-Casting** | Araç merkezinden dışarıya doğru ışınlar göndererek ilk temas edilen engelin mesafesini bulma tekniği. |
| **Distance Contour** | Araç çevresi boyunca $360^\circ$ ölçülen en yakın engellerin çizdiği mesafe poligonu. |
| **STOP Threshold** | Aracın durması gereken asgari güvenlik sınırı ($30\text{ cm}$). |
| **Log-Odds** | Olasılıkların çarpımı yerine toplamıyla güncellenen kararlı Bayesyen ızgara güncelleme yöntemi. |
| **Curb Detection** | Kaldırım kenarlarını ve yüksekliğini tespit ederek jant sürtmelerini önleme fonksiyonu. |
| **Ego-Motion Translation** | Aracın tekerlek odometrisiyle yer değiştirmesine göre hafıza ızgarasının kaydırılması. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Sıfır ek donanım maliyeti (USS'siz saf kamera)      | • Park halinde araç kapalıyken yaklaşan nesneleri     |
| • 5 cm yüksek çözünürlüklü 360° mesafe algısı         |   ilk açılışta görememe riski                         |
| • Kaldırım ve duvar hatlarını tam 3D gösterme         |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Cybertruck ve yeni nesil Robotaxi için tam otonom   | • Kirli veya aşırı su damlacıklı arka kamera          |
|   kendi kendine park etme (AutoPark) altyapısı        |   mercekleri                                          |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Vision Park Asistanı Mimarisi

```
[ 8 Kamera Ham Görüntüleri ] ===> [ Occupancy Network (3D Voxel) ]
                                                |
                                                v
[ Tekerlek Odometrisi (dx, dy) ] ===> [ Kör Nokta Zamansal Belleği ]
                                                |
                                                v
                                  [ 360° Işın Atma (Ray-Casting) ]
                                                |
                                                v
                                  [ Santimetre Konturu & STOP İkazı ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana park asistanı simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
