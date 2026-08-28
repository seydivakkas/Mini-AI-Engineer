# 🏷️ Day 95: MiniViT v1 Sürüm Adayı (Release Candidate) ve Uçtan Uca Regresyon Testleri

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Hugging Face: Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg?style=flat-square)](https://huggingface.co/)
[![Quality Gate: PASSED](https://img.shields.io/badge/Quality%20Gate-PASSED-brightgreen.svg?style=flat-square)](src/regresyon_motoru.py)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_regresyon_rc.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin on üçüncü gününde; eğitilmiş ve Hugging Face formatında paketlenmiş Mini Vision Transformer modelimizi, genel canlı dağıtım (Day 96) öncesinde dondurarak **v1.0.0-rc1 (Release Candidate)** sürüm adayı haline getiriyor ve uçtan uca **Kalite Kapısı (Quality Gate)** regresyon testleriyle mühürlüyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Yazılım mühendisliğinde kod değişikliklerinin mevcut işlevleri bozmadığını doğrulamak için uygulanan regresyon testleri, Makine Öğrenimi Mühendisliğinde (MLOps) çok daha kritik ve çok boyutludur. Bir modelin üretime çıkmadan önce **Release Candidate (RC)** aşamasından geçirilmesinin bilimsel ve operasyonel gerekçeleri şunlardır:

1. **Sessiz Sayısal Bozulmaların (Silent Numerical Regressions) Engellenmesi:**
   Kütüphane güncellemeleri, tensör tipi dönüşümleri (float32 $\rightarrow$ bfloat16) veya serileştirme hataları modelin çıktılarında beklenmedik sayısal sapmalara yol açabilir. **Altın Veri Seti (Golden Dataset)** regresyon testi, dondurulmuş referans tensörler ile model logits çıktılarının tolerans sınırları ($< 10^{-4}$) içinde kaldığını kesinleştirir.
2. **SLA ve Gecikme Bütçesi Garantisi (Latency Budget SLA):**
   Üretim mikroservislerinde p95/p99 gecikmelerinin kontrolden çıkması tüm sistemi kilitleyebilir. RC kalite kapısı, çıkarım sürelerinin SLA bütçesini ($P50 \le 5.0$ ms, $P95 \le 10.0$ ms) aşmadığını garantiler.
3. **Bellek Sızıntısı (Memory Leak) Güvencesi:**
   Ardışık çıkarımlar sırasında PyTorch hesaplama graflarının veya tensör tamponlarının temizlenmemesi podların OOM (Out Of Memory) ile çökmesine yol açar. Bellek kararlılık testi 100 ardışık çıkarımda RAM/VRAM tüketimini denetler.
4. **Kriptografik Bütünlük ve İmzalı Manifesto (`RELEASE_MANIFEST.json`):**
   Tüm paket artefaktlarının (`model.safetensors`, `config.json`, `preprocessor_config.json`, `README.md`) SHA-256 karma özetleri hesaplanır ve dondurulmuş sürüm etiketiyle (`v1.0.0-rc1`) mühürlenir.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Üretime Bozuk Model Çıkışını Önleme:** Kalite kapısındaki 5 kriterden herhangi biri başarısız olduğunda sistem dağıtımı otomatik olarak `NO-GO` ile durdurur.
- **Güvensiz ve İmzalanmamış Model Transferlerini Engelleme:** Dağıtım hattındaki (CI/CD) modellerin SHA-256 hash'leri denetlenerek paket üzerinde oynama veya eksik dosya transferi tespit edilir.
- **Canlı Ortam Pod Çökmelerini (OOM) Önleme:** Bellek sızıntısı testi sayesinde üretimde uzun süre çalışan asenkron API servislerinin kararlılığı önceden teyit edilir.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Sentetik vs Gerçek Dünya Veri Temsili:** Altın veri seti belirli senaryoları kapsar; üretim ortamındaki bilinmeyen uç durumlar (edge cases) için canlı gözlemlenebilirlik (Day 91) gereklidir.
- **Donanım Spesifik Gecikme Sapmaları:** RC testinin yapıldığı yerel GPU/CPU ile üretim Kubernetes kümesindeki GPU mimarisi farklıysa gecikme bütçelerinde sapmalar yaşanabilir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Regresyon & Dağıtım Yaklaşımı | Altın Veri Sayısal Testi | SLA Gecikme Bütçesi | Bellek Sızıntısı Analizi | Kriptografik SHA-256 Manifestosu | Otomasyon Seviyesi |
|---|---|---|---|---|---|
| **MiniViT RC1 Kalite Kapısı (Bizim)** | **Evet ($< 10^{-4}$)** | **Evet (P50/P95)** | **Evet (100 Döngü)** | **Evet (`RELEASE_MANIFEST.json`)** | **Tam Otomatik (CI/CD)** |
| **Geleneksel Model Checkpoint** | Hayır | Hayır | Hayır | Hayır | Manuel |
| **Temel Unit Test Yaklaşımı** | Kısmi | Hayır | Hayır | Hayır | Düşük |
| **Sadece Model Registry (MLflow)** | Hayır | Hayır | Hayır | Kısmi (Artefakt Hash) | Orta |

---

## 📐 Matematiksel Formülasyon

### 1. Altın Veri Seti Sayısal Sapma Formülü (Chebyshev / $L_\infty$ Normu)
Modelimizin çıkarım logits tensörü $y_{\text{rc}} \in \mathbb{R}^{B \times C}$ ve referans dondurulmuş logits tensörü $y_{\text{ref}} \in \mathbb{R}^{B \times C}$ arasındaki maksimum mutlak fark hesaplanır:

$$\Delta_{\text{max}} = \| y_{\text{rc}} - y_{\text{ref}} \|_{\infty} = \max_{i, j} |y_{\text{rc}}^{(i, j)} - y_{\text{ref}}^{(i, j)}|$$

Kalite Kapısı Kabul Koşulu:

$$\Delta_{\text{max}} < \epsilon \quad (\text{burada } \epsilon = 10^{-4})$$

### 2. SLA Gecikme Yüzdelikleri (Percentile Latency)
$N$ adet ardışık çıkarım gecikmesi ölçümü $T = \{t_1, t_2, \dots, t_N\}$ küçükten büyüğe sıralandığında ($t_{(1)} \le t_{(2)} \le \dots \le t_{(N)}$):

$$P_{50} = t_{(\lceil 0.50 \cdot N \rceil)}, \quad P_{95} = t_{(\lceil 0.95 \cdot N \rceil)}$$

SLA Kabul Kriteri:

$$P_{50} \le 5.0\,\text{ms} \quad \text{ve} \quad P_{95} \le 10.0\,\text{ms}$$

### 3. Bellek Artış / Sızıntı Oranı (Memory Drift Ratio)
$K$ adımlık ardışık çıkarım sürecinde başlangıç RSS belleği $M_0$ ve bitiş RSS belleği $M_K$ için:

$$\Delta M_{\%} = \max\left(0, \frac{M_K - M_0}{M_0} \times 100\right) \le \tau_{\text{mem}} \quad (\tau_{\text{mem}} = 2.0\%)$$

---

## 📖 Kapsamlı Teknik Terimler Sözlüğü

| Terim | Tanım | Endüstriyel Önemi |
|---|---|---|
| **Release Candidate (RC)** | Üretime çıkmaya hazır, tüm geliştirmeleri tamamlanmış ve dondurulmuş sürüm adayı. | Modelin nihai sürüm öncesi kararlılığını mühürler. |
| **Quality Gate (Kalite Kapısı)** | Modelin bir üst ortama (Staging $\rightarrow$ Production) geçebilmesi için sağlaması gereken katı test kriterleri bütünü. | Hatalı veya standart altı modellerin üretime sızmasını önler. |
| **Golden Dataset (Altın Veri Seti)** | Modelin sayısal doğruluğunu doğrulamak için dondurulmuş, beklenen girdi-çıktı çiftlerinden oluşan referans veri kümesi. | Sayısal ve mimari regresyonların anında yakalanmasını sağlar. |
| **SLA (Service Level Agreement)** | Modelin sunması gereken maksimum gecikme ve minimum doğruluk taahhüdü. | Mikroservis mimarilerinde kullanıcı deneyimi ve throughput güvencesidir. |
| **Memory Leak (Bellek Sızıntısı)** | Çıkarım sırasında temizlenmeyen tensörlerin RAM veya VRAM'i tüketerek sistemi çökertmesi. | Uzun süreli çalışan servislerin stabilitesi için kritik denetimdir. |
| **SHA-256 Manifestosu** | Tüm model dosyalarının kriptografik karma özetlerini ve üstverilerini içeren JSON dosyası. | Tedarik zinciri güvenliği ve bütünlük doğrulaması sağlar. |
| **P95 Latency** | Yapılan isteklerin %95'inin tamamlandığı süre eşiği. | Kuyruk ve spike gecikmelerini ölçen en gerçekçi performans metriğidir. |

---

## 📊 SWOT Analizi (Karar Matrisi)

```
┌──────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│                   GÜÇLÜ YÖNLER (S)               │                  ZAYIF YÖNLER (W)                │
│ • Çok boyutlu (sayısal, metrik, SLA, bellek) test│ • Altın veri setinin periyodik bakımı gerekir.   │
│ • Kriptografik SHA-256 ile tam izlenebilirlik.   │ • Donanım farklılıklarında gecikme sapabilir.    │
│ • Otomatik GO / NO-GO dağıtım kararı.            │ • Bellek testleri iterasyon süresini uzatabilir. │
├──────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│                  FIRSATLAR (O)                   │                   TEHDİTLER (T)                  │
│ • CI/CD pipeline (GitHub Actions) tam entegrasyon│ • Sessiz kütüphane sürüm uyumsuzlukları.         │
│ • Canlı dağıtım (Day 96) için sıfır risk.        │ • Üretim veri kayması (data drift) belirsizliği. │
│ • Kurumsal MLOps standartlarına %100 uyum.       │ • Beklenmedik işletim sistemi I/O gecikmeleri.   │
└──────────────────────────────────────────────────┴──────────────────────────────────────────────────┘
```

---

## 🚀 Çalıştırma ve Doğrulama

### 1. Birim ve Regresyon Testlerini Çalıştırma (PyTest)
```bash
pytest testler/test_regresyon_rc.py -v
```

### 2. Sürüm Adayı Paketleme ve Kalite Kapısını Çalıştırma
```bash
python ana_akis.py
```

### 3. Çıktı Dosyaları
- `surum_adayi_paketi/RELEASE_MANIFEST.json` (Kriptografik SHA-256 manifestosu)
- `surum_adayi_paketi/model.safetensors` (SafeTensors model ağırlıkları)
- `surum_adayi_paketi/config.json` (MiniViT mimari konfigürasyonu)
- `ciktilar/minivit_rc1_regresyon_paneli.png` (6-panelli teşhis panosu)

---

## 📜 Lisans & Metaveri

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
