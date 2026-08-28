# 101 GÜNLÜK BÜYÜK FİNAL: MiniViT-MoE v2 (Sparse Mixture of Experts) Hugging Face Canlı Dağıtımı & Master Mühendislik Manifestosu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Roadmap Status: 101/101 COMPLETED](https://img.shields.io/badge/101--Day%20Roadmap-100%25%20Completed-gold?style=flat-square)](#)

> **GÜN 101 — 101 GÜNLÜK MASTER ROADMAP'İN BÜYÜK FİNALİ (THE GRAND FINALE):**  
> Bu gün, **"101 Günlük Yapay Zeka, Bilgisayarlı Görü, LLM/RAG ve MLOps Mühendisliği Master Programı"**nın nihai zirvesidir. Geliştirilen **MiniViT-MoE v2** mimarisi; **Pre-RMSNorm**, **PyTorch SDPA (FlashAttention-2)**, **Top-K Softmax Router**, **SwiGLU Uzmanları** ve **Auxiliary Load Balancing Loss** mekanizması ile donatılmış, Hugging Face Hub üzerinde canlı `safetensors` ve `Gradio Space` dağıtım paketi olarak yayınlanmıştır.

---

## 1. Dört Zorunlu Analiz ve Mühendislik Derinliği

### 1.1 Hata / İkilem Senaryosu: Yoğun (Dense) Parametre Darboğazı vs Seyrek (Sparse MoE) Ölçeklenme
- **Problem**: Yoğun (Dense) derin öğrenme modellerinde modelin kapasitesini (parametre sayısını) artırmak, orantılı olarak her çıkarım adımındaki FLOPs ve donanım gecikmesini artırır. Model büyüdükçe gerçek zamanlı (real-time) servis edilemez hale gelir.
- **İkilem**: Parametre kapasitesini $3\times - 4\times$ artırırken, çıkarım anındaki FLOPs ve VRAM hesaplama yükünü sabit tutmak nasıl mümkündür?
- **Çözüm**: **Sparse Mixture of Experts (MoE)** mimarisi! Modelde $E=4$ adet uzman inşa edilirken, her bir token için yalnızca en uygun $k=2$ uzman dinamik olarak seçilir (**Top-2 Routing**). Bu sayede toplam parametre kapasitesi 2.37M seviyesine çıkarken, çıkarım esnasında yalnızca 1.32M parametre çalışır (**%44.1 FLOPs tasarrufu**).

### 1.2 Mimari / Tasarım Deseni: Top-K Softmax Router ve SwiGLU Uzmanları
- **Top-K Yönlendirici (Gating Network)**:
  $$H(x) = \text{TopK}(\text{Softmax}(x W_{\text{gate}} + \epsilon), k)$$
  Gelen token vektörü $x \in \mathbb{R}^d$, $W_{\text{gate}} \in \mathbb{R}^{d \times E}$ matrisi ile uzman olasılıklarına izdüşürülür. En yüksek skora sahip $k$ uzman seçilir ve ağırlıkları normalize edilir:
  $$y = \sum_{i \in \text{TopK}} \tilde{w}_i \cdot \text{Expert}_i(x)$$
- **Auxiliary Load Balancing Loss (Yük Dengeleme Kaybı)**:
  Yönlendiricinin sadece 1-2 popüler uzmana kilitlenmesini (Router Collapse) engellemek için Switch Transformer / Mixtral standardında yardımcı kayıp eklenir:
  $$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i, \quad \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{aux}}$$
  Burada $f_i$ uzman $i$'ye yönlendirilen token'ların frekansı, $P_i$ ise ortalama yönlendirme olasılığıdır.

### 1.3 Ölçeklenebilirlik & Dayanıklılık: MLOps ve Hub Yayın Standardı
- **Safetensors Serileştirmesi**: `safetensors.torch.save_file` ile zero-copy bellek eşleme ve Python `pickle` güvenlik açıklarını bertaraf eden güvenli model formatı.
- **Canlı Gradio Spaces Arayüzü**: Model ile birlikte otomatik oluşturulan `app.py` sayesinde Hugging Face Spaces üzerinde tek tıkla canlı interaktif demo başlatılabilir.

### 1.4 Anti-Pattern & Sık Yapılan Hatalar
- **Router Collapse (Yönlendirici Çökmesi)**: Yük dengeleme kaybı ($\mathcal{L}_{\text{aux}}$) kullanılmadığında yönlendirici gradyanları ilk birkaç uzmana yoğunlaştırır ve diğer uzmanlar ölü nöron (dead expert) haline gelir.
- **Top-K Ağırlıklarını Normalize Etmemek**: Seçilen $k$ uzmanın softmax çıktıları doğrudan toplanırsa toplamları 1'den küçük kalır ve tensör genliği düşer; `topk_weights / sum(topk_weights)` normalizasyonu şarttır.

---

## 2. Kapsamlı Teknik Sözlük (10+ Terim)

| Terim | Tanım ve Açıklama |
|---|---|
| **Sparse MoE** | Sparse Mixture of Experts. Her token için yalnızca belirli uzman alt kümelerinin ($k \ll E$) aktif edildiği seyrek mimari. |
| **Top-K Router** | Girdiyi analiz ederek en yüksek olasılığa sahip $k$ adet uzmanı dinamik olarak seçen yönlendirici katman. |
| **Auxiliary Load Balancing Loss** | Uzmanların eşit oranda token işlemesini sağlayan ve yönlendirici çökmesini engelleyen dengeleme kaybı fonksiyonu. |
| **Router Collapse** | Yönlendiricinin tüm token'ları sadece 1 uzmana yönlendirip diğer uzmanları pasif bırakması durumu. |
| **SwiGLU Expert** | SiLU kapılı doğrusal birim mimarisine sahip bireysel uzman ileri besleme (FFN) bloğu. |
| **Active Parameters** | Modelin toplam parametresi yerine, tek bir çıkarım adımında aktif olarak hesaplamaya katılan parametre miktarı. |
| **Sparsity Gain** | Seyreklik kazancı. MoE sayesinde toplam parametre kapasitesine kıyasla sağlanan hesaplama ve FLOPs tasarrufu oranı. |
| **Pre-RMSNorm** | Attention ve MoE katmanlarından önce uygulanan kök ortalama kare normalizasyon standardı. |
| **SDPA** | PyTorch 2.0+ yerleşik Scaled Dot-Product Attention ve FlashAttention-2 bellek optimize çekirdeği. |
| **Model Card** | Modelin kullanım amacını, mimarisini, eğitim metriklerini ve etik sınırlarını belgeleyen standart Hugging Face kartı. |

---

## 3. Mimari SWOT Matrisi

| | Olumlu (Güçlü / Fırsatlar) | Olumsuz (Zayıf / Tehditler) |
|---|---|---|
| **İçsel Faktörler (Internal)** | **Güçlü Yönler (S)**:<br>• Toplam 2.37M kapasiteye rağmen sadece 1.32M aktif FLOPs.<br>• %44.1 FLOPs ve hesaplama tasarrufu.<br>• Mükemmel uzman yük dengesi (E1=%24.8, E2=%25.6, E3=%25.1, E4=%24.5). | **Zayıf Yönler (W)**:<br>• Toplam model ağırlıkları bellekte saklandığı için model dosya boyutu (VRAM) yoğundur. |
| **Dışsal Faktörler (External)** | **Fırsatlar (O)**:<br>• Mixtral 8x7B, DeepSeek-V3 ve Gemini benzeri modern frontier MoE mimarilerinin temelini oluşturur.<br>• Hugging Face Hub üzerinde küresel açık erişim. | **Tehditler (T)**:<br>• Dağıtık çoklu GPU ortamlarında uzmanlar arası iletişim (all-to-all dispatch) darboğazı. |

---

## 4. 101 Günlük Master Yol Haritası — Fazlar ve Başarı Özeti

```
====================================================================================================
           101 GÜNLÜK YAPAY ZEKA VE MLOPS MASTER PROGRAMI — NİHAİ MEZUNİYET TABLOSU
====================================================================================================
• FAZ 1 (Gün 01 - 20) : Python Temelleri, PyTorch Çekirdeği, Tensör Matematiği ve CNN Mimarileri
• FAZ 2 (Gün 21 - 40) : Bilgisayarlı Görü, Nesne Tespiti (YOLO), Segmentasyon ve Veri Boru Hatları
• FAZ 3 (Gün 41 - 65) : Doğal Dil İşleme (NLP), Transformer, LLM Fine-Tuning ve RAG Mimarileri
• FAZ 4 (Gün 66 - 85) : Vision Transformers (ViT), Çok Modlu (Multimodal) Modeller ve CLIP
• FAZ 5 (Gün 86 - 101): Model Sıkıştırma (Pruning/INT8/FP16), Güvenilirlik, Determinizm, FastAPI,
                        Docker Konteynerleştirme, Yük Testleri, SwiGLU/RMSNorm ve BÜYÜK FİNAL: MoE v2!
====================================================================================================
🏆 TOPLAM: 101 GÜN — 800+ PYTEST BİRİM TESTİ — %100 BAŞARI VE MEZUNİYET SERTİFİKASI
====================================================================================================
```

---

## 5. Proje Yapısı ve Kullanım

```bash
day-101-huggingface-minivit-moe-v2/
├── ciktilar/
│   └── minivit_moe_v2_buyuk_final_paneli.png  # 6 Panelli Büyük Final ve Mezuniyet Paneli
├── hf_moe_model_paketi/                       # Standart Hugging Face Hub Yayın Paketi
│   ├── model.safetensors
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── app.py                                # Gradio Canlı Spaces Uygulaması
│   └── README.md                             # Kapsamlı Model Card
├── src/
│   ├── __init__.py
│   ├── konfigurasyon.py                       # MiniViTMoEConfig (MoE parametreleri)
│   ├── moe_katmanlari.py                      # TopKRouter, SwiGLUUzmani, MoEKatmani, Blok
│   ├── model.py                               # MiniViTMoEForImageClassification
│   ├── hub_yoneticisi.py                      # Safetensors ve Hub Dışa Aktarıcısı
│   └── gorsellestirici.py                     # 6 Panelli Teşhis ve Diploma Çizici
├── testler/
│   ├── __init__.py
│   └── test_moe.py                            # 8 Kapsamlı PyTest Birim Testi
├── ana_akis.py                                # Büyük Final Koşum ve Dağıtım Scripti
├── gereksinimler.txt
├── LICENSE
└── README.md
```

### Testleri ve Ana Akışı Çalıştırma

```bash
# 1. Birim Testleri Çalıştır (8/8 PASSED)
pytest day-101-huggingface-minivit-moe-v2/testler -v

# 2. Büyük Final Dağıtımını ve Mezuniyet Panosunu Üret
python day-101-huggingface-minivit-moe-v2/ana_akis.py
```

---

## 6. Lisans

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
