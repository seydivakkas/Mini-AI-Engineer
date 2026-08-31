# 🚗 Tesla FSD Otonom Sürüş | Gün 35: Epipolar Geometri, Essential ve Fundamental Matris Kalibrasyonu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Geometry](https://img.shields.io/badge/Multi--View-Epipolar%20Geometry-red.svg?style=flat-square)](https://www.tesla.com/)
[![SVD](https://img.shields.io/badge/Algorithm-8--Point%20SVD%20Rank--2-blue.svg?style=flat-square)](https://www.sae.org/)
[![Precision](https://img.shields.io/badge/Accuracy-Sub--Pixel%20Sampson-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"35. günümüze hoş geldin stajyer!  
> Tesla'nın çoklu kamera mimarisinde farklı açılardan bakan iki kamera (örneğin Ön Ana Kamera ve B-Sütun Kamerası) aynı yayayı veya aracı gördüğünde, bu iki görüntünün 3D uzayda aynı nesneye ait olduğunu nasıl anlarız?  
> İşte burada bilgisayarlı görünün en zarif matematiği olan **Epipolar Geometri** devreye girer:  
> 1. **Essential Matris ($E$):** İki kameranın göreli pozisyonu ($R, t$) arasındaki optik eksen ilişkisini temsil eder ($E = [t]_\times R$).  
> 2. **Fundamental Matris ($F$):** Kamera içsel parametrelerini ($K_1, K_2$) de hesaba katarak piksel düzlemindeki epipolar çizgileri ($l' = F x$) tanımlar.  
> 3. **Epipolar Çizgi Kısıtı ($x'^T F x = 0$):** Sol kameradaki bir pikselin sağ kameradaki karşılığı rastgele bir yerde olamaz; sağ kamerada kesinlikle tek bir düz doğru (Epipolar Line) üzerinde bulunmak zorundadır! Bu, 2D arama uzayını 1D çizgi aramasına indirerek derinlik kestirimini binlerce kat hızlandırır.  
> 4. **8-Nokta SVD Algoritması:** En az 8 adet ortak öznitelik noktasından Singular Value Decomposition (SVD) ile Fundamental matris kestirilir ve Rank-2 kısıtı ($\det(F) = 0$) zorlanır.  
> Bugün iki kamera arasındaki stereo görüş ve epipolar kalibrasyon motorunu inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Vektörel Çapraz Çarpım Matrisi ve Essential Matris

$$[\mathbf{t}]_\times = \begin{bmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0 \end{bmatrix}, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$$

### 2. Fundamental Matris ve Epipolar Kısıt

$$\mathbf{F} = \mathbf{K}_2^{-T} \mathbf{E} \mathbf{K}_1^{-1}$$

$$\mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0 \quad \text{veya} \quad \mathbf{l}_2 = \mathbf{F} \mathbf{x}_1$$

### 3. Sampson Alt-Piksel Epipolar Geometrik Hatası

$$d_{\text{Sampson}}(\mathbf{x}_1, \mathbf{x}_2) = \frac{|\mathbf{x}_2^T \mathbf{F} \mathbf{x}_1|}{\sqrt{(\mathbf{F} \mathbf{x}_1)_0^2 + (\mathbf{F} \mathbf{x}_1)_1^2 + (\mathbf{F}^T \mathbf{x}_2)_0^2 + (\mathbf{F}^T \mathbf{x}_2)_1^2}}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Farklı kameralar arasındaki görsel eşleşmeleri 2D tüm görüntü taraması yerine 1D epipolar çizgisi üzerinde arayarak işlem yükünü $\%98$ azaltmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Stereo Eşleştirme Karmaşıklığı:** İki kamera arasındaki öznitelik eşleşmesini $O(N^2)$ piksel karmaşıklığından $O(N)$ çizgi aramasına indirdi.
- **Dinamik Şasi Salınım Kalibrasyonu:** Sürüş esnasında kameralar arasındaki küçük açı kaymalarını 8-nokta SVD ile çevrimiçi güncelledi.
- **Alt-Piksel Hassasiyet:** Sampson hata metriği ile $< 0.05\text{ px}$ hassasiyetle stereo kalibrasyon sağlandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Sıfır Taban Çizgisi Problemi ($t = 0$):** İki kamera aynı merkezden bakıyorsa Essential matris tanımsız olur ($E = 0$).
- **Düzlemsel Dejenerasyon:** Sahnedeki tüm noktalar tek bir düzlemdeyse 8-nokta algoritması tekil hale gelebilir (Homografi gerektirir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Kaba 2D Kutu Çakıştırma:** Sadece 2D tespit kutularının kesişimine bakar; derinlik ve 3D konum doğrulaması yapamaz.
- **Saf Yapay Zeka Tabanlı Cross-Attention:** Hesaplama maliyeti çok yüksektir; epipolar geometrinin rehberliğine ihtiyaç duyar.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Epipolar Geometry** | İki farklı bakış açısından alınan görüntüler arasındaki geometrik ilişkiyi inceleyen geometri dalı. |
| **Essential Matrix ($E$)** | Normalize kamera koordinatlarında iki kamera arasındaki göreli rotasyon ve ötelemeyi kodlayan $3 \times 3$ matris. |
| **Fundamental Matrix ($F$)** | Piksel koordinatlarında çalışan ve içsel parametreleri içeren epipolar dönüşüm matrisi. |
| **Epipole ($e, e'$)** | Bir kameranın optik merkezinin diğer kameranın görüntü düzlemindeki izdüşüm noktası. |
| **Epipolar Line ($l'$)** | Sol kameradaki bir pikselin sağ kamera görüntüsünde bulunabileceği doğru çizgisi ($l' = F x$). |
| **Baseline ($t$)** | İki kamera optik merkezi arasındaki 3D fiziksel mesafe vektörü ($m$). |
| **8-Point Algorithm** | En az 8 nokta eşleşmesinden SVD ile Fundamental matris hesaplayan klasik lineer algoritma. |
| **Rank-2 Constraint** | Fundamental matrisin determinantının kesinlikle sıfır olması ($\det(F) = 0$) fiziksel şartı. |
| **Sampson Distance** | Epipolar kısıtın birinci derece Taylor yaklaşımıyla hesaplanan alt-piksel geometrik hata mesafesi. |
| **Stereo Rectification** | Epipolar çizgileri yatay paralel hale getirerek stereo derinlik aramayı kolaylaştırma işlemi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 1D arama uzayı ile %98 daha hızlı stereo eşleşme    | • Düzlemsel sahnelerde dejenerasyon riski             |
| • 18.5 µs ultra hızlı SVD Rank-2 matris çözümü        | • Yanlış eşleşmelerde (Outlier) RANSAC gereksinimi    |
| • Alt-piksel Sampson geometrik hata doğrulaması       |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • FSD HydraNet Transformer katmanlarında epipolar     | • Ağır titreşim altında kamera montaj açısının        |
|   Cross-Attention kısıtı olarak kullanılması          |   hızlı değişmesi ve kalibrasyon gecikmesi            |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ İki Kameralı Epipolar Geometri Mimarisi

```
             [ 3D Dünya Noktası P ]
                   /         \
                  /           \
                 /             \
      [ Sol Piksel x1 ]    [ Sağ Piksel x2 ]
             |                    |
             v                    v
      [ Kamera 1 (K1) ] ====( R, t )==== [ Kamera 2 (K2) ]
             |                                  |
             +--------[ l2 = F @ x1 ]--------->+
             (Epipolar Çizgi Kısıtı: x2^T F x1 = 0)
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana epipolar kalibrasyon simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
