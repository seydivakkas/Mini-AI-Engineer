# 🌐 Day 96: MiniViT v1.0 Hugging Face Canlı Dağıtımı & Canlı Model Demosu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Hugging Face: Hub & Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces%20%26%20Hub-yellow.svg?style=flat-square)](https://huggingface.co/)
[![Gradio: Live Demo](https://img.shields.io/badge/Gradio-Web%20UI-orange.svg?style=flat-square)](app.py)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_hf_public_release.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin on dördüncü gününde; Mini Vision Transformer (**MiniViT v1.0**) modelimizi genel kullanıma açıyor, Hugging Face Model Hub üzerinde canlı dağıtım paketini oluşturuyor, `transformers.pipeline` uyumlu çıkarım motorunu inşa ediyor ve Hugging Face Spaces üzerinde çalışan etkileşimli bir **Gradio Web Demosu (`app.py`)** ayağa kaldırıyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Eğitilen ve kalite kapılarından (Day 95) geçen bir yapay zeka modelinin endüstriyel değer kazanabilmesi için geliştiricilere ve son kullanıcılara iki temel erişim kanalı sunması gerekir:
1. **Programatik Erişim (API / Pipeline):** Geliştiricilerin tek satırda modeli indirip uygulamalarına entegre edebilmesi (`AutoModelForImageClassification.from_pretrained(...)` veya `pipeline(...)`).
2. **Görsel & Etkileşimli Erişim (Gradio / Spaces Playground):** Ürün yöneticilerinin, etiketleyicilerin ve son kullanıcıların teknik kod yazmadan web tarayıcısı üzerinden kendi fotoğraflarını yükleyerek modelin performansını canlı deneyimleyebilmesi.

Hugging Face Spaces ve Gradio entegrasyonu, modelin yaygınlaştırılmasını (democratization) ve gerçek kullanıcı geri bildirimlerinin toplanmasını sağlar.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Model Dağıtımında Entegrasyon Sürtünmesini Sıfırlama:** Kullanıcıların karmaşık PyTorch kurulumları veya özel ağırlık yükleme kodları yazma zorunluluğunu ortadan kaldırır.
- **Canlı Görsel Hata Ayıklama (Visual Debugging):** Gradio arayüzü sayesinde modelin hangi sınıflarda tereddüt ettiğini (Top-5 olasılık çubukları) ve çıkarım gecikmesini anlık olarak görselleştirir.
- **Bulut Tabanlı Sıfır Kurulumlu Demo (Zero-Setup Cloud Demo):** `app.py` dosyası ile Hugging Face Spaces üzerinde sunucu yönetimi gerektirmeden çalışan bir web uygulaması sağlar.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Gradio Eşzamanlı İstek Sınırları:** Ücretsiz Hugging Face Spaces katmanında CPU kotaları ve eşzamanlı istek (concurrency) sınırları nedeniyle yoğun trafik altında gecikmeler artabilir.
- **İstemci Tarafı Ön-İşleme Bağımlılığı:** Görüntü formatları (PNG, JPEG, WebP, RGBA) ve renk uzaylarının (RGB/BGR) doğru dönüştürülmesi `MiniViTPipeline` seviyesinde garanti edilmelidir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Dağıtım & Demo Yaklaşımı | Canlı Web Arayüzü | AutoClass API Desteği | Sıfır Kurulumlu Bulut Demo | Geliştirme Hızı |
|---|---|---|---|---|
| **Hugging Face Spaces + Gradio (Bizim)** | **Evet (Modern UI)** | **Evet (`pipeline`)** | **Evet (Spaces)** | **Çok Yüksek (< 1 Saat)** |
| **Streamlit Uygulaması** | Evet | Kısmi | Evet (Streamlit Cloud) | Yüksek |
| **Özel React / FastAPI Servisi** | Evet | Evet | Hayır (Kendi Sunucun) | Düşük (1-2 Gün) |
| **Sadece Model Checkpoint Yayını** | Hayır | Hayır | Hayır | Düşük |

---

## 📐 Matematiksel Formülasyon

### 1. Softmax Olasılık Dağılımı ve Top-K Sıralaması
Modelin sınıflandırma başlığından çıkan ham logits vektörü $z \in \mathbb{R}^C$ (burada $C=10$) için her $i$. sınıfın olasılığı Softmax ile normalize edilir:

$$P(y = i \mid x) = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}$$

Gradio arayüzünde gösterilecek en yüksek $K$ ($K=5$) olasılık kümesi:

$$\text{Top-}K(P) = \operatorname{argtopk}_{i \in \{1, \dots, C\}} P(y = i \mid x)$$

### 2. Görüntü Giriş Normalizasyonu
$H \times W \times 3$ boyutundaki girdi görüntüsünün piksel değerleri $x \in [0, 255]$ önce $[0, 1]$ aralığına çekilir, ardından kanal bazında CIFAR-10 ortalama ($\mu$) ve standart sapma ($\sigma$) değerleriyle standartlaştırılır:

$$x_{\text{norm}}^{(c)} = \frac{\frac{x^{(c)}}{255.0} - \mu_c}{\sigma_c}, \quad c \in \{\text{R, G, B}\}$$

$$\mu = [0.4914, 0.4822, 0.4465], \quad \sigma = [0.2470, 0.2435, 0.2616]$$

---

## 📖 Kapsamlı Teknik Terimler Sözlüğü

| Terim | Tanım | Endüstriyel Önemi |
|---|---|---|
| **Hugging Face Spaces** | Makine öğrenimi web uygulamalarını (Gradio/Streamlit) ücretsiz barındıran bulut platformu. | Modellerin anında halka açılmasını ve test edilmesini sağlar. |
| **Gradio** | Python tabanlı, makine öğrenimi modelleri için hızlı web arayüzleri oluşturan UI kütüphanesi. | Görsel interaktif prototipleme standardıdır. |
| **Transformers Pipeline** | Ham veriden (görüntü/metin) sonuca kadar tüm ön/son işleme aşamalarını kapsayan yüksek seviyeli çıkarım API'si. | Geliştiricilere tek satırda model çalıştırma kolaylığı sunar. |
| **Top-K Prediction** | Modelin en yüksek olasılık atadığı ilk $K$ sınıfı ve güven skorlarını listeleme yöntemi. | Sınıflandırma belirsizliğini ve alternatif tahminleri incelemeyi sağlar. |
| **Model Widget** | Hugging Face Model Card sayfasında doğrudan tarayıcı üzerinden çıkarım yapmayı sağlayan etkileşimli bileşen. | Hub ziyaretçilerinin modeli indirmeden denemesini sağlar. |

---

## 📊 SWOT Analizi (Karar Matrisi)

```
┌──────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│                   GÜÇLÜ YÖNLER (S)               │                  ZAYIF YÖNLER (W)                │
│ • Hem Python API hem interaktif Web UI desteği.  │ • Spaces CPU katmanında sınırlı donanım kaynağı. │
│ • SafeTensors ile hızlı ve güvenli yükleme.      │ • Yoğun yük altında kuyruk gecikmeleri.          │
│ • Gradio ile sıfır HTML/CSS gereksinimi.         │ • Model güncellemelerinde cache temizliği gerek. │
├──────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│                  FIRSATLAR (O)                   │                   TEHDİTLER (T)                  │
│ • Topluluk nezdinde geniş görünürlük & yıldızlar │ • Beklenmeyen girdi formatlarında UI hataları.   │
│ • Canlı kullanıcı geri bildirimi toplama.        │ • Üçüncü taraf kütüphane sürüm kırılmaları.      │
│ • Kurumsal portföy ve canlı demo referansı.      │ • Ağ bağlantı kesintileri ve API timeout'ları.   │
└──────────────────────────────────────────────────┴──────────────────────────────────────────────────┘
```

---

## 🚀 Çalıştırma ve Doğrulama

### 1. Birim ve Dağıtım Testlerini Çalıştırma (PyTest)
```bash
pytest testler/test_hf_public_release.py -v
```

### 2. Canlı Dağıtım Akışını Çalıştırma
```bash
python ana_akis.py
```

### 3. Canlı Gradio Web Uygulamasını Başlatma (`app.py`)
```bash
python app.py
```
*Tarayıcınızda `http://127.0.0.1:7860` adresini açarak etkileşimli arayüzü deneyimleyebilirsiniz.*

---

## 📜 Lisans & Metaveri

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
