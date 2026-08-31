# 🚗 Tesla FSD Otonom Sürüş | Gün 68: GPU Hızlandırmalı Donanım Renderleme ve 3D Araç Görselleştirme (OpenGL/Vulkan)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Graphics](https://img.shields.io/badge/Graphics-OpenGL%20%2F%20Vulkan%20MVP-red.svg?style=flat-square)](https://www.tesla.com/)
[![3D Engine](https://img.shields.io/badge/Pipeline-3D%20World%20to%20Screen%20Projection-blue.svg?style=flat-square)](https://www.khronos.org/)
[![Performance](https://img.shields.io/badge/Budget-Sub--100%C2%B5s%20Vertex%20Transform-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"68. günümüze hoş geldin stajyer!  
> Tesla V12 kullanıcı arayüzünün en büyüleyici parçası, ekranın sol tarafında sürekli akan **3D FSD Otonom Sürüş Dünyasıdır**.  
> Aracınızın kendi 3D modeli, etrafındaki yayalar, diğer arabalar, çöp kutuları, yol şeritleri ve turkuaz/cyan renkli hedef sürüş yörüngesi milimetrik bir kesinlikle ekrana çizilir:  
> 1. **Model-View-Projection (MVP) Pipeline:** 3D dünyadaki her bir nesne önce kendi koordinatından dünya uzayına ($\mathbf{M}$), ardından sanal takip kamerası uzayına ($\mathbf{V}$) ve son olarak perspektif kırpma uzayına ($\mathbf{P}$) taşınır.  
> 2. **Kırpma ve NDC Dönüşümü:** Kamera görüş açısının dışında kalan nesneler elenir ve kalanlar $[-1, 1]$ normalize cihaz koordinatlarına (NDC) bölünür.  
> 3. **Ekran Uzayı İzdüşümü:** NDC koordinatları Tesla'nın $1920 \times 1200$ dokunmatik ekran piksellerine haritalanır.  
> 4. **Sabit 60 FPS GPU Render Garantisi:** $16.6\text{ ms}$ kare bütçesi içinde tüm dönüşümler tamamlanır.  
> Bugün Tesla'nın 3D otonom sürüş görselleştirme motorunun grafik çekirdeğini inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Model-View-Projection (MVP) Birleşik Dönüşümü

$$\mathbf{MVP} = \mathbf{P}_{\text{proj}} \cdot \mathbf{V}_{\text{view}} \cdot \mathbf{M}_{\text{model}}$$

$$\mathbf{p}_{\text{clip}} = \mathbf{MVP} \cdot \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = \begin{bmatrix} x_{\text{clip}} \\ y_{\text{clip}} \\ z_{\text{clip}} \\ w_{\text{clip}} \end{bmatrix}$$

### 2. Normalize Cihaz Koordinatları (NDC) ve Ekran Haritalaması

$$\mathbf{p}_{\text{ndc}} = \frac{1}{w_{\text{clip}}} \begin{bmatrix} x_{\text{clip}} \\ y_{\text{clip}} \\ z_{\text{clip}} \end{bmatrix}, \quad -1 \le x_{\text{ndc}}, y_{\text{ndc}}, z_{\text{ndc}} \le 1$$

$$u = \left( \frac{x_{\text{ndc}} + 1}{2} \right) \cdot W_{\text{screen}}, \quad v = \left( \frac{1 - y_{\text{ndc}}}{2} \right) \cdot H_{\text{screen}}$$

### 3. Kamera LookAt Görünüm (View) Matrisi

$$\mathbf{f} = \frac{\mathbf{p}_{\text{target}} - \mathbf{p}_{\text{cam}}}{\|\mathbf{p}_{\text{target}} - \mathbf{p}_{\text{cam}}\|}, \quad \mathbf{s} = \frac{\mathbf{f} \times \mathbf{u}_{\text{up}}}{\|\mathbf{f} \times \mathbf{u}_{\text{up}}\|}, \quad \mathbf{u} = \mathbf{s} \times \mathbf{f}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
FSD sinir ağının algıladığı 3D voksel ve vektör dünyasını sürücüye gerçek zamanlı ve güven verici bir 3D perspektifte aktarmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Sürücü Güven Eksikliği:** Sistemin neyi görüp neyi görmediğini (araçlar, şeritler, yayalar) şeffafça 3D ekrana yansıtarak güven inşa etti.
- **Düşük Gecikmeli Render:** CPU yerine GPU gölgelendiricileri (Shaders) kullanarak binlerce tepe noktasını mikrosaniyelerde dönüştürdü.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Aşırı Poligon Yoğunluğu:** Yoğun kavşaklarda yüzlerce yaya ve aracın aynı anda render edilmesi GPU bant genişliğini zorlayabilir (LOD - Level of Detail seviyelendirmesi gerekir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **2D Kuşbakışı (Top-Down 2D Map):** Klasik navigasyonlar kullanır; ancak sürücüye aracın arkasından akıcı bir 3D perspektif hissi veremez.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **MVP Matrix** | Model, View ve Projection matrislerinin çarpımıyla elde edilen birleşik grafik dönüşüm matrisi. |
| **Model Matrix ($\mathbf{M}$)**| Nesnenin yerel koordinatlarını 3D dünya koordinatlarına taşıyan öteleme ve döndürme matrisi. |
| **View Matrix ($\mathbf{V}$)** | 3D dünyayı sanal takip kamerasının bakış açısına göre konumlandıran LookAt matrisi. |
| **Projection Matrix ($\mathbf{P}$)**| 3D kamera uzayını perspektif derinlik katarak kırpma uzayına dönüştüren matris. |
| **Clip Space** | Kamera görüş piramidi (Frustum) içindeki nesnelerin sınırlandığı 4D homojen uzay. |
| **NDC (Normalized Device Coordinates)**| Kırpma uzayının $w$ bileşenine bölünmesiyle elde edilen $[-1, 1]$ aralığındaki koordinat sistemi. |
| **Screen Space** | Ekrandaki gerçek piksel koordinatları ($1920 \times 1200$). |
| **Vertex Shader** | Her bir 3D tepe noktasını GPU üzerinde paralel dönüştüren grafik programı. |
| **Frustum Culling** | Kameranın görüş açısının dışında kalan 3D nesneleri render edilmeden eleyen optimizasyon. |
| **FSD Visualization** | Tesla ekranındaki 3D otonom sürüş avatarı ve çevre render arayüzü. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 donanım hızlandırmalı GPU matris boru hattı   | • Karmaşık 3D modellerde yüksek GPU bellek kullanımı  |
| • 45 µs ultra hızlı tepe noktası dönüşümü            | • LOD uygulanmazsa kalabalık sahnelerde FPS düşüşü   |
| • Sürücüye tam şeffaflık ve güven sağlayan 3D avatar |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Ray-Tracing ve Unreal Engine seviyesinde fotorealistik| • Ekran GPU çekirdeğinin aşırı ısınması durumunda   |
|   Tesla Robotaksi yolcu eğlence ekranı görselleştirmesi |   kare hızının 30 FPS'e düşmesi                       |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla 3D Grafik Render Pipeline Şeması

```
[ 3D Dünya Nesneleri (Ego Araç, Şeritler, FSD Yolu, Çevre Araçlar) ]
                                |
                                v
             [ 1. Model Matrisi Dönüşümü (M_model) ]
                                |
                                v
             [ 2. Kamera LookAt Dönüşümü (V_view) ]
                                |
                                v
          [ 3. Perspektif Projeksiyon Matrisi (P_proj) ]
                                |
                                v
          [ 4. Kırpma Uzayı & NDC Bölümü (p_clip / w) ]
                                |
                                v
       [ 5. 2D Ekran Piksel İzdüşümü (1920x1200 Framebuffer) ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana 3D GPU Render simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
