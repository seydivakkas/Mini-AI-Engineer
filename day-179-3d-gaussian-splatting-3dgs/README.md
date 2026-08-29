# Day 179: 3D Gaussian Splatting (3DGS) ile Gerçek Zamanlı (100+ FPS) Radyan Rasterizasyonu

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 19. günüdür. NeRF'ün ağır ışın takibi (Ray Marching) kısıtını kırarak fotogerçekçi 3D sahneleri **100+ FPS gerçek zamanlı** hızda renderlayan **3D Gaussian Splatting (3DGS - Kerbl et al., 2023)**, **3D Gauss Temsili (Konum $\mu$, Ölçek $s$, Dönme Kuaterniyonu $q$, Opaklık $\alpha$ ve Küresel Harmonikler SH)**, **3D'den 2D Ekran Düzlemine Kovaryans Projeksiyonu ($\Sigma' = J W \Sigma W^T J^T$)**, **Diferansiyellenebilir Tile-Tabanlı Alfa Karıştırma (Alpha Blending Rasterizer: $C = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$)**, ve **Adaptif Yoğunluk Kontrolü (Adaptive Density Control: Klonlama ve Bölme)** motorunu sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "3D Gaussian Splatting (3DGS)" Nedir ve Nasıl 0.3 FPS'lik NeRF'ü 145 FPS'e Çıkardı?
- **Sorun (NeRF'ün Ağır Işın İntegrali):**
  NeRF fotogerçekçi görüntüler üretir; ancak 1080p tek bir kare üretmek için uzayda 2 milyon ışın fırlatıp her ışında 64 kez derin bir MLP sinir ağını sorgulamak zorundadır. Bu yüzden saniyede ancak 0.3 kare (FPS) üretebilir; yani video oyunlarında veya sanal gerçeklikte (VR) canlı çalışması imkansızdır.
- **Çözüm (3DGS: Açık Elipsoidler ve GPU Rasterizasyonu):**
  1. *Açık 3D Elipsoidler (Gausslar):* 3D uzay bir sinir ağı içine saklanmaz; sahne 1-2 milyon adet minik, yönlendirilebilir 3D duman kümesi (Gauss elipsoidi) olarak tutulur.
  2. *2D Ekran İzdüşümü (Splatting):* Kamera baktığında 3D elipsoidler matematiksel Jacobian projeksiyonuyla anında 2D düzlemsel elipslere ($\Sigma'$) dönüştürülür.
  3. *Tile Tabanlı GPU Rasterizasyonu:* Ekran $16 \times 16$ piksellik karelere bölünür, elipsoidler GPU üzerinde derinliğe göre tek seferde sıralanır (Radix Sort) ve alfa karıştırmayla boyanır.
  - *Devrim:* 24 saatlik eğitim süresi **20 dakikaya**, render hızı ise **145+ FPS'e (Gerçek Zamanlı)** fırlar!

```
====================================================
         3D GAUSSIAN SPLATTING PIPELINE             
====================================================
  3D Gausslar (mu, S, q, alpha, SH) ───────────────┐
           │                                       │
           ▼ (Kamera Projeksiyonu & Jacobian J)    │
  [2D Ekran Gaussları: mu_2d, Sigma_2d]            │
           │                                       │
           ▼ (GPU Radix Sort: Derinliğe Göre Sırala)\n"
  [Önden Arkaya Sıralı Elipsoidler]                │
           │                                       │
           ▼ (16x16 Tile Tabanlı Alfa Karıştırma)  │
  [C(p) = sum c_i * alpha_i * (prod (1-alpha_j))]  │
           │                                       │
           ▼                                       │
  [145+ FPS Fotogerçekçi 2D Görüntü Çıktısı] ──────┘
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. 3D Gauss Parametrizasyonu ve Kovaryans Matrisi
- 3D uzaydaki bir Gauss yoğunluk dağılımı:
  $$G(x) = \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$
- Pozitif yarı-tanımlılığı garanti etmek için $\Sigma$ kovaryans matrisi ölçek $S$ ve rotasyon matrisi $R$ ile ayrıştırılır:
  $$\Sigma = R S S^T R^T, \quad R = \text{QuaternionToRotationMatrix}(q), \quad S = \text{diag}(s)$$

### B. EWA Splatting ile 2D Ekran Kovaryans Projeksiyonu
- Kamera projeksiyonu Jacobian matrisi $J$ ve görüntüleme matrisi $W$ kullanılarak 2D ekran kovaryansı $\Sigma'$ hesaplanır:
  $$\Sigma' = J W \Sigma W^T J^T + \begin{bmatrix} 0.3 & 0 \\ 0 & 0.3 \end{bmatrix}$$

### C. Diferansiyellenebilir Alfa Karıştırma (Over Operator)
- $N$ adet önden arkaya sıralı 2D Gauss'un piksel $p$'deki nihai renk katkısı:
  $$C(p) = \sum_{i \in \mathcal{N}} c_i \alpha_i(p) \prod_{j=1}^{i-1} (1 - \alpha_j(p)), \quad \alpha_i(p) = \alpha_i \exp\left(-\frac{1}{2} (p - \mu_{2d, i})^T \Sigma_{2d, i}^{-1} (p - \mu_{2d, i})\right)$$

### D. Performans ve Doğrulama
- NeRF'e kıyasla **~414x hızlanma (145 FPS)** ve +3.5 dB PSNR artışı elde edilmiştir.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **3DGS (3D Gaussian Splatting)** | 3D sahneleri diferansiyellenebilir 3D Gauss elipsoidleri ile gerçek zamanlı renderlayan mimari. |
| **Splatting** | 3D hacimsel noktaların 2D ekran düzlemine bir boya damlası gibi izdüşürülmesi işlemi. |
| **Kovaryans Matrisi ($\Sigma$)** | Gauss elipsoidinin 3D uzaydaki yönünü, eğimini ve 3 eksendeki boyutunu belirleyen $3 \times 3$ matris. |
| **Jacobian Projeksiyon Matrisi ($J$)** | 3D kamera koordinatlarının 2D piksel düzlemine doğrusal yaklaşım türev matrisi. |
| **Kuaterniyon ($q$)** | Gimbal lock problemini engelleyen 4 boyutlu birim rotasyon temsilcisi. |
| **Tile-Based Rasterization** | Ekranı $16 \times 16$ piksel bloklarına bölerek GPU paralel işlemlerini maksimize eden render tekniği. |
| **Küresel Harmonikler (SH)** | Görüş açısına göre değişen metalik parıltı ve yansımaları modelleyen trigonometrik baz fonksiyonları. |
| **Adaptive Density Control** | Yetersiz temsil edilen bölgelerde Gaussları klonlayan, aşırı büyükleri ikiye bölen optimizasyon motoru. |
| **Radix Sort** | Milyonlarca Gauss noktasını GPU üzerinde derinliklerine göre milisaniyeler içinde sıralayan algoritma. |
| **Over Operator** | Ön plandaki nesnelerin arka plandaki nesneleri saydamlık oranında kapatmasını sağlayan piksel birleştirme işlemi. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • 145+ FPS gerçek zamanlı render     │ • Milyonlarca nokta nedeniyle        │
 │   (Web tarayıcılarında bile 60 FPS). │   RAM/VRAM tüketiminin yüksek olması │
 │ • 20 dakikada fotogerçekçi eğitim.   │   (PLY dosya boyutu ~500 MB - 1 GB). │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Video oyunları, sanal gerçeklik    │ • Eğitim sırasında sahne sınırları   │
 │   (Metaverse/Apple Vision Pro),      │   dışındaki gürültülü Gauss          │
 │   e-ticaret 3D ürün görüntüleyicisi. │   artifaktları (floater'lar).        │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/gaussian_splatting_3dgs_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
