# 🧠 201 Günlük Yapay Zeka, Bilgisayarlı Görü, LLM/RAG, Reasoning, Triton GPU & MLOps Mühendisliği Master Hafıza Dosyası (Master Roadmap)

Bu dosya; projenin 1. gününden 201. gününe kadar tüm yol haritasını, tamamlanan ve planlanan günleri, modül hedeflerini ve teknik derinliklerini kayıt altında tutan **merkezi hafıza (memory) belgesidir**.

---

## 📌 FAZ ÖZETİ VE DURUM TABLOSU

| Faz | Gün Aralığı | Alan & Kapsam | Durum |
| --- | --- | --- | --- |
| **FAZ 1** | Gün 01 - Gün 20 | Veri Temelleri, Görüntü İşleme, Renk Uzayları, NumPy/Pandas ve CNN Mimarileri | ✅ **TAMAMLANDI (%100)** |
| **FAZ 2** | Gün 21 - Gün 41 | İleri Bilgisayarlı Görü, YOLO Nesne Tespiti, U-Net Segmentasyon, Arama (BM25/Vektör) ve RAG | ✅ **TAMAMLANDI (%100)** |
| **FAZ 3** | Gün 42 - Gün 66 | Çekirdek ML/DL Boru Hatları, Sayısal Kararlılık, FAISS, ONNX INT8 Kuantizasyon & Edge MLOps | ✅ **TAMAMLANDI (%100)** |
| **FAZ 4** | Gün 67 - Gün 85 | İleri Düzey Temsil Öğrenimi (SimCLR, SupCon), Sıfırdan Vision Transformer (ViT) & LoRA | ✅ **TAMAMLANDI (%100)** |
| **FAZ 5** | Gün 86 - Gün 101 | Model Sıkıştırma, Güvenilirlik, FastAPI, Docker, SwiGLU/RMSNorm & MiniViT-MoE v2 Hub Dağıtımı | ✅ **TAMAMLANDI (%100)** |
| **FAZ 6** | Gün 102 - Gün 120 | İleri LLM Mimarileri (GQA/MLA/RoPE), Hizalama (Alignment), SFT, DPO, KTO, GRPO & RLHF | ✅ **TAMAMLANDI (%100)** |
| **FAZ 7** | Gün 121 - Gün 140 | Otonom AI Ajanları (ReAct/Reflexion), Multi-Agent Sistemleri ve Advanced GraphRAG | ✅ **TAMAMLANDI (%100)** |
| **FAZ 8** | Gün 141 - Gün 160 | Derin Akıl Yürütme (Reasoning LLMs), Test-Time Compute, MCTS ve Process Reward Models | ✅ **TAMAMLANDI (%100)** |
| **FAZ 9** | Gün 161 - Gün 180 | Çok Modlu (Multimodal) Temel Modeller: VLM (LLaVA), Video, Ses, Difüzyon ve 3DGS | 🔄 **DEVAM EDİYOR (Gün 161 Sırada)** |
| **FAZ 10** | Gün 181 - Gün 201 | Ultra-MLOps, Dağıtık Eğitim (FSDP/ZeRO), Özel Triton GPU Kernel & BÜYÜK FİNAL 201 | ⏳ Sırada |

---

## 📋 DETAYLI GÜN GÜN YOL HARİTASI (GÜN 01 - 201)

### ✅ Tamamlanan Günler: 101 Serisi (Gün 01 - Gün 101)

#### 🔹 FAZ 1: Veri Temelleri, Görüntü İşleme ve Konvolüsyonel Ağlar (Gün 01 - Gün 20)
- [x] **Day 01:** `day-01-numpy-image-analyzer` — NumPy ile Görsel Analizi, Renk Kanalları, Histogram & Kontrast
- [x] **Day 02:** `day-02-distance-metrics` — Vektörel Mesafe Metrikleri (Euclidean, Manhattan, Cosine, Chebyshev)
- [x] **Day 03:** `day-03-mahalanobis-vs-euclidean` — Mahalanobis vs Öklid, Kovaryans Matrisi, Özdeğer/Özvektör
- [x] **Day 04:** `day-04-pandas-data-cleaner` — Pandas Temizlik Boru Hattı, Kayıp Veri İmputasyonu, Z-Score
- [x] **Day 05:** `day-05-mini-data-profiler` — Otomatik Veri Profilleme, İstatistiksel Özetler
- [x] **Day 06:** `day-06-eda-lab` — Keşifçi Veri Analizi (EDA), Korelasyon, Dağılım Çizimleri
- [x] **Day 07:** `day-07-outlier-detection` — Aykırı Değer Tespiti (IQR, Z-Score, Isolation Forest)
- [x] **Day 08:** `day-08-image-processing-toolkit` — Görsel İşleme Araç Kutusu (Filtreleme, Sobel, Morfoloji)
- [x] **Day 09:** `day-09-image-histogram-analyzer` — Histogram Eşitleme, CLAHE, Kontrast İyileştirme
- [x] **Day 10:** `day-10-color-space-explorer` — Uzay Dönüşümleri (RGB, HSV, LAB, YCrCb)
- [x] **Day 11:** `day-11-dominant-color-extractor` — Baskın Renk Çıkarımı (K-Means Quantization, Color Palettes)
- [x] **Day 12:** `day-12-color-similarity-engine` — Renk Benzerliği Motoru (CIEDE2000, Earth Mover's Distance)
- [x] **Day 13:** `day-13-perspective-correction` — Perspektif Düzeltme (Homografi Matrisi, Köşe Tespiti)
- [x] **Day 14:** `day-14-motif-segmentation` — Motif & Doku Bölütleme (Gabor Filtreleri, Otsu Eşikleme)
- [x] **Day 15:** `day-15-grabcut-background-remover` — GrabCut Arka Plan Çıkarma (GMM & Graph Cut Optimization)
- [x] **Day 16:** `day-16-image-feature-extractor` — Öznitelik Çıkarımı (SIFT, ORB, HOG, LBP)
- [x] **Day 17:** `day-17-visual-nearest-neighbor` — Görsel En Yakın Komşu Araması (k-NN, Cosine Index)
- [x] **Day 18:** `day-18-image-clustering` — Etiketsiz Görsel Kümeleme (K-Means, DBSCAN, Silhouette)
- [x] **Day 19:** `day-19-classical-image-classifier` — Geleneksel Makine Öğrenmesi (HOG + LBP + SVM / Random Forest)
- [x] **Day 20:** `day-20-tensorflow-cnn-classifier` — TensorFlow/Keras ile CNN (Conv2D, BatchNorm, Dropout)

#### 🔹 FAZ 2: Bilgisayarlı Görü, Nesne Tespiti, Segmentasyon ve Arama/RAG (Gün 21 - Gün 41)
- [x] **Day 21:** `day-21-pytorch-cnn-classifier` — PyTorch CNN (nn.Module, DataLoader, Grad-CAM XAI)
- [x] **Day 22:** `day-22-data-augmentation` — Veri Çoğaltma (Albumentations, MixUp, CutMix)
- [x] **Day 23:** `day-23-transfer-learning` — Transfer Öğrenme (ResNet, EfficientNet, Fine-Tuning)
- [x] **Day 24:** `day-24-model-evaluation-and-error-analysis` — Model Değerlendirme & Hata Analizi (ROC-AUC, PR-AUC, ECE)
- [x] **Day 25:** `day-25-object-detection-basics` — Nesne Tespiti Temelleri (IoU/GIoU/DIoU, NMS/Soft-NMS, Anchors)
- [x] **Day 26:** `day-26-yolo-training-inference` — YOLOv8/YOLO11 Eğitimi & Çıkarımı (mAP@0.5, mAP@0.5:0.95)
- [x] **Day 27:** `day-27-semantic-segmentation-basics` — Anlamsal Bölütleme (U-Net, Combo Loss, mIoU, Error Heatmap)
- [x] **Day 28:** `day-28-advanced-segmentation` — İleri Düzey Bölütleme (Mask R-CNN, SegFormer, Panoptic Quality)
- [x] **Day 29:** `day-29-multi-object-tracking` — Çoklu Nesne Takibi (DeepSORT, Kalman Filtresi, MOTA/IDF1)
- [x] **Day 30:** `day-30-multitask-vision-platform` — Çoklu Görev Görsel Platformu & Model Kuantizasyon Optimizasyonu
- [x] **Day 31:** `day-31-bm25-document-search` — BM25 Leksikal Arama Algoritması, TF-IDF, Ters İndeks, Tokenizasyon
- [x] **Day 32:** `day-32-semantic-search-engine` — Sentence Transformers ile Yoğun (Dense) Vektör Arama, Kosinüs Benzerliği
- [x] **Day 33:** `day-33-hybrid-search-rrf` — BM25 + Vektör Arama Birleşimi, Reciprocal Rank Fusion (RRF) Hibrit Sıralama
- [x] **Day 34:** `day-34-mini-rag-assistant` — Mini RAG Asistanı, Chunking, Vektör Veritabanı Sorgulama, LLM Context Injection
- [x] **Day 35:** `day-35-fastapi-model-service` — FastAPI Asenkron REST API, Pydantic Tip Güvenliği, Model Servisleme
- [x] **Day 36:** `day-36-streamlit-ai-dashboard` — Streamlit ile İnteraktif AI Kontrol Paneli, Dosya Yükleme, Model Görselleştirme
- [x] **Day 37:** `day-37-carpet-color-intelligence` — Halı/Tekstil Renk Ayrıştırma, İplik Renk Oranları Çıkarımı, Katalog Uyumu
- [x] **Day 38:** `day-38-carpet-visual-retrieval` — Halı Doku ve Desenleri İçin Çoklu Özellikli Görsel Arama
- [x] **Day 39:** `day-39-carpet-defect-detector` — Halı Dokuma Hataları, Leke ve Kusur Tespiti, Kalite Kontrol Otomasyonu
- [x] **Day 40:** `day-40-carpet-knowledge-rag` — Tekstil ve Üretim Teknik Dokümanları Üzerinde Sektörel RAG Sistemi
- [x] **Day 41:** `day-41-ai-carpet-intelligence-suite` — Renk, Arama, Kusur ve RAG Modüllerini Birleştiren Halı Zekası Paketi

#### 🔹 FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps (Gün 42 - Gün 66)
- [x] **Day 42:** `day-42-numpy-ai-batch-inspector` — Üretim Girdi Tensörleri Doğrulama, Batch Boyutu & NaN/Inf/Shape Anomali Tespiti
- [x] **Day 43:** `day-43-numpy-data-drift-detector` — Veri Kayması (Data Drift) Tespiti, KS-Test İstatistiği, Wasserstein Mesafesi
- [x] **Day 44:** `day-44-pandas-data-quality-cleaner` — Üretim Seviyesi Şema Doğrulama, Sınır Değer Kontrolleri, Otomatik Temizlik
- [x] **Day 45:** `day-45-pandas-feature-engineering-profile-builder` — Özellik Mühendisliği, Encoding, Ölçeklendirme, Feature Store Mimarisi
- [x] **Day 46:** `day-46-matplotlib-ai-experiment-report-generator` — Otomatik Loss/Acc, PR, ROC Grafikleri ve PDF/HTML Deney Raporlama Motoru
- [x] **Day 47:** `day-47-sklearn-leakage-safe-ml-pipeline` — Veri Sızıntısına Karşı Güvenli Pipeline, ColumnTransformer & Nested CV
- [x] **Day 48:** `day-48-kmeans-unsupervised-segmentation` — Elbow & Silhouette Analizi, Uzamsal (Spatial) Piksel K-Means Bölütleme
- [x] **Day 49:** `day-49-xgboost-tabular-risk-classifier` — Dengesiz Tabüler Veri, scale_pos_weight, XGBoost ile Risk/Dolandırıcılık Tespiti
- [x] **Day 50:** `day-50-model-evaluation-threshold-engineering` — Eşik Değeri Mühendisliği, F-beta Optimizasyonu, Maliyet-Fayda Karar Matrisi
- [x] **Day 51:** `day-51-pillow-safe-image-loader` — Bozuk Dosya & Yanlış EXIF Yönetimi, Hataya Toleranslı Görsel Yükleyici
- [x] **Day 52:** `day-52-opencv-visual-defect-inspector` — FFT Frekans Analizi, Laplacian Varyansı ile Bulanıklık & Kural Tabanlı Kusur Tespiti
- [x] **Day 53:** `day-53-cielab-kmeans-palette-analyzer` — Perceptually Uniform LAB Uzayında K-Means & Delta-E 2000 Hassas Tolerans Analizi
- [x] **Day 54:** `day-54-image-forensics-inspector` — Dijital Adli Bilişim, Error Level Analysis (ELA), Görsel Manipülasyon Tespiti
- [x] **Day 55:** `day-55-pytorch-dataset-dataloader` — İleri PyTorch DataLoader, num_workers, pin_memory Darboğaz Optimizasyonu
- [x] **Day 56:** `day-56-tinyvisioncnn` — Edge Cihazlar İçin Sıfırdan Hafif CNN, Depthwise Separable Conv, FLOPs Hesabı
- [x] **Day 57:** `day-57-pytorch-training-engine` — Modüler Eğitim Motoru, Checkpoint, Early Stopping, Gradient Clipping
- [x] **Day 58:** `day-58-amp-numerical-stability-benchmark` — Otomatik Karma Hassasiyet (AMP), FP16 vs BF16, GradScaler & Sayısal Kararlılık
- [x] **Day 59:** `day-59-transfer-learning-embedding-extractor` — ViT/ResNet Omurgalarından Dondurulmuş Katmanlarla L2-Normalize Embedding Çıkarımı
- [x] **Day 60:** `day-60-faiss-similarity-search-engine` — FAISS ile Milyonluk Vektör İndeksleme (IndexFlatIP, IndexIVFFlat, HNSW, GPU)
- [x] **Day 61:** `day-61-retrieval-metrics-benchmark` — Vektör Arama Değerlendirmesi: NDCG@k, MRR (Mean Reciprocal Rank), Gecikme Testi
- [x] **Day 62:** `day-62-sdxl-lora-controlled-generator` — Üretken AI: Stable Diffusion XL (SDXL) + LoRA ile Kontrollü Görsel Üretimi
- [x] **Day 63:** `day-63-pydantic-ai-domain-models` — Pydantic v2 ile Tip Güvenli Girdi/Çıktı Sözleşmeleri & Domain Modelleri
- [x] **Day 64:** `day-64-fastapi-inference-api` — Üretim Seviyesi FastAPI İnference, Model Yaşam Döngüsü (lifespan), Batch Prediction
- [x] **Day 65:** `day-65-streamlit-sqlite-ai-dashboard` — SQLite Destekli CRUD, Model Çıkarım Logları ve Kalıcı AI Yönetim Paneli
- [x] **Day 66:** `day-66-onnx-int8-production-capstone` — PyTorch Modellerini ONNX'e Aktarma, INT8 PTQ Kuantizasyon & ONNX Runtime Hızlandırma

#### 🔹 FAZ 4: İleri Düzey Temsil Öğrenimi ve Sıfırdan Vision Transformer (Gün 67 - Gün 85)
- [x] **Day 67:** `day-67-config-driven-reproducible-training` — YAML/Hydra ile Konfigürasyon Yönetimi, Deterministik & Tekrarlanabilir Eğitim
- [x] **Day 68:** `day-68-high-performance-vision-data-pipeline` — Albumentations ile Yüksek Performanslı Veri Artırma & GPU Prefetching
- [x] **Day 69:** `day-69-optimizer-scheduler-laboratory` — AdamW vs Lion Optimizer, CosineAnnealing, Linear Warmup & Weight Decay Dinamikleri
- [x] **Day 70:** `day-70-modern-regularization-mixup-cutmix-label-smoothing` — Mixup, CutMix Veri Artırma ve Label Smoothing Cross-Entropy Düzenlileştirmesi
- [x] **Day 71:** `day-71-fault-tolerant-resumable-training-engine` — Çökmeye Dayanıklı Checkpoint, State Restoration ve Devam Edebilir Eğitim Motoru
- [x] **Day 72:** `day-72-embedding-geometry` — t-SNE, UMAP Boyut İndirgeme, Temsil Uzayı Geometrisi & İzotropi Analizi
- [x] **Day 73:** `day-73-simclr-from-scratch` — Sıfırdan SimCLR Temsil Öğrenimi, Artırma Çiftleri, NT-Xent (InfoNCE) Kaybı
- [x] **Day 74:** `day-74-supervised-contrastive-learning` — Etiketli Veride Supervised Contrastive (SupCon) Kaybı ile Sınıf Ayrıştırma
- [x] **Day 75:** `day-75-metric-learning-triplet-hard-negative` — Triplet Margin Loss, Hard/Semi-Hard Negative Mining Stratejileri
- [x] **Day 76:** `day-76-representation-benchmark-suite` — Temsil Kalitesi Değerlendirmesi: Linear Probing ve k-NN Sınıflandırma Protokolü
- [x] **Day 77:** `day-77-self-attention-from-scratch` — Sıfırdan Scaled Dot-Product & Multi-Head Self-Attention Mekanizması
- [x] **Day 78:** `day-78-transformer-encoder-from-scratch` — Sıfırdan Transformer Encoder Bloğu: Pozisyonel Kodlama, LayerNorm, Residual FFN
- [x] **Day 79:** `day-79-minivit-from-scratch` — Sıfırdan Mini Vision Transformer (Patch Projeksiyonu, CLS Token, Encoder Birleşimi)
- [x] **Day 80:** `day-80-minivit-cifar100-training` — Sıfırdan MiniViT'in CIFAR-100 Üzerinde Eğitimi & Regülarizasyon Dinamikleri
- [x] **Day 81:** `day-81-vit-lora-peft` — Vision Transformer İçin LoRA (Low-Rank Adaptation) ile Parametre-Verimli İnce Ayar
- [x] **Day 82:** `day-82-knowledge-distillation` — Öğretmen-Öğrenci Modeli Bilgi Damıtma, Soft Target Loss (KL-Diverjansı), Temperature
- [x] **Day 83:** `day-83-structured-pruning` — L1/L2 Norm Tabanlı Yapısal Filtre/Kanal Budama, Hız vs Doğruluk Dengesi
- [x] **Day 84:** `day-84-calibration-uncertainty` — Olasılık Kalibrasyonu, Expected Calibration Error (ECE) & Temperature Scaling
- [x] **Day 85:** `day-85-ood-selective-prediction` — Enerji Tabanlı Dağılım Dışı (OOD) Tespiti ve Seçici Tahmin (Abstention)

#### 🔹 FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve MoE Hub Dağıtımı (Gün 86 - Gün 101)
- [x] **Day 86:** `day-86-model-pruning` — Yapısal Olmayan (Unstructured) ve Yapısal (Structured L1-Norm) Pruning
- [x] **Day 87:** `day-87-post-training-quantization` — INT8 Dinamik ve Statik Post-Training Kuantizasyon
- [x] **Day 88:** `day-88-quantization-aware-training` — FakeQuantize ve Straight-Through Estimator ile QAT
- [x] **Day 89:** `day-89-onnx-tensorrt-export` — ONNX Runtime Dinamik Eksen Dışa Aktarımı ve TensorRT Optimizasyonu
- [x] **Day 90:** `day-90-dynamic-batching-inference` — GPU Verimliliği İçin Kuyruk Tabanlı Dinamik Batching Çıkarım Motoru
- [x] **Day 91:** `day-91-ai-observability` — Canlı AI Sistemlerinde Gözlemlenebilirlik: Gecikme, Hacim ve Veri Kayması İzleme
- [x] **Day 92:** `day-92-final-training-contract` — Eğitim Öncesi Veri Sözleşmesi Testleri ve Hazır Bulunuşluk (Readiness) Kontrolleri
- [x] **Day 93:** `day-93-final-evaluation-model-card` — Kapsamlı Değerlendirme, Yanlılık (Bias) Testleri ve Standart Model Card Üretimi
- [x] **Day 94:** `day-94-hugging-face-integration` — Hugging Face Model Hub Entegrasyonu, Konfigürasyon ve Model Paketleme
- [x] **Day 95:** `day-95-minivit-v1-release-candidate` — MiniViT v1 Sürüm Adayı (Release Candidate), Uçtan Uca Regresyon Testleri
- [x] **Day 96:** `day-96-huggingface-public-v1-release` — MiniViT v1.0 Hugging Face Canlı Dağıtımı & Canlı Model Demosu
- [x] **Day 97:** `day-97-reproducible-inference` — Deterministik Çıkarım, Donanımdan Bağımsız Doğrulama Testleri
- [x] **Day 98:** `day-98-fastapi-inference-service` — Üretime Hazır Yüksek Performanslı Asenkron API & `/health` Kontrolleri
- [x] **Day 99:** `day-99-container-load-testing` — Docker Konteynerleştirme ve Locust/k6 ile Eşzamanlı Yük/Stres Testleri
- [x] **Day 100:** `day-100-modern-architecture-ablations` — SwiGLU, RMSNorm ve FlashAttention Mimarileri ile MiniViT Ablasyon Analizleri
- [x] **Day 101:** `day-101-huggingface-minivit-moe-v2` — **101 GÜNLÜK BÜYÜK FİNAL:** MiniViT Mixture of Experts (MoE) v2 Hugging Face Dağıtımı

---

### 🚀 Planlanan İleri Seviye Günler: 201 Serisi (Gün 102 - Gün 201)

#### 🔹 FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO (Gün 102 - Gün 120)
- [x] **Day 102:** `day-102-gqa-grouped-query-attention` — Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) ile KV Cache Azaltma
- [x] **Day 103:** `day-103-mla-multi-latent-attention` — Multi-Head Latent Attention (MLA - DeepSeek V2/V3) Sıkıştırılmış KV Projeksiyonu
- [x] **Day 104:** `day-104-rope-yarn-context-extension` — Rotary Position Embeddings (RoPE) & YaRN ile 128k+ Bağlam Uzatma Matematiği
- [x] **Day 105:** `day-105-sliding-window-attention` — Sliding Window Attention (SWA - Mistral) & Chunked Memory Yönetimi
- [x] **Day 106:** `day-106-sft-packed-sequences` — Instruction Supervised Fine-Tuning (SFT) & Token Packing (Sıfır Padding Kaybı)
- [x] **Day 107:** `day-107-qlora-4bit-nf4-unsloth` — QLoRA (NF4 Kuantizasyon, Double Quantization) & Hızlı Gradyan Geri Yayılımı
- [x] **Day 108:** `day-108-reward-modeling-bradley-terry` — Bradley-Terry Tercih Modeli, Çiftli Karşılaştırma ve Reward Model Eğitimi
- [x] **Day 109:** `day-109-rlhf-ppo-actor-critic` — PPO ile LLM Hizalama: Actor, Critic, Ref Model & KL Penalty
- [x] **Day 110:** `day-110-dpo-direct-preference-optimization` — Direct Preference Optimization (DPO): Reward Modelsiz Doğrudan Tercih Optimizasyonu
- [x] **Day 111:** `day-111-kto-kahneman-tversky-optimization` — KTO: İkili (Beğenildi/Beğenilmedi) Geri Bildirimlerle Davranışsal Tercih Hizalaması
- [x] **Day 112:** `day-112-orpo-odds-ratio-preference` — ORPO (Odds Ratio Preference Optimization): Tek Aşamalı SFT + Alignment
- [x] **Day 113:** `day-113-simpo-simple-preference-optimization` — SimPO: Referans Modelsiz, Hedef Marjinli Hafif Tercih Optimizasyonu
- [x] **Day 114:** `day-114-grpo-group-relative-policy` — GRPO (Group Relative Policy Optimization - DeepSeek-R1): Critic-Free Grup Skalalama
- [x] **Day 115:** `day-115-model-merging-slerp-ties` — Model Birleştirme: SLERP, TIES ve DARE Yöntemleriyle Model Füzyonu
- [x] **Day 116:** `day-116-synthetic-data-evol-instruct` — Evol-Instruct & UltraFeedback ile Sentetik Veri Üretim ve Kalite Filtreleme Hattı
- [x] **Day 117:** `day-117-jailbreak-red-teaming-guardrails` — LLM Red-Teaming, Prompt Injection Tespiti ve Llama Guard Güvenlik Duvarları
- [x] **Day 118:** `day-118-llm-watermarking-detection` — LLM Çıktılarına Yeşil/Kırmızı Liste Tabanlı Kriptografik Filigran Ekleme
- [x] **Day 119:** `day-119-self-instruct-knowledge-distill` — Büyük Modellerden (Teacher) Küçük Modellere (Student) Bilgi Damıtma (Distillation)
- [x] **Day 120:** `day-120-aligned-llm-benchmark-eval` — MT-Bench, AlpacaEval ve Chatbot Arena Tarzı Otomatik LLM Hakemlik (LLM-as-a-Judge)

#### 🔹 FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)
- [x] **Day 121:** `day-121-react-agent-scratchpad` — ReAct (Reasoning + Acting) Deseni: Düşünce-Eylem-Gözlem Döngüsü ve Scratchpad
- [x] **Day 122:** `day-122-plan-and-solve-prompting` — Plan-and-Solve / Decomposed Prompting ile Karmaşık Görevleri Alt Görevlere Bölme
- [x] **Day 123:** `day-123-reflexion-self-evaluating-agent` — Reflexion: Dil Tabanlı Kendi Kendini Eleştirme (Self-Critique) ve Hafıza ile İterasyon
- [x] **Day 124:** `day-124-tool-calling-json-schema` — JSON Schema Destekli Kesin Tip Güvenli Tool/Function Calling & Pydantic Validasyonu
- [x] **Day 125:** `day-125-sandboxed-code-execution-agent` — Güvenli İzolasyonlu Python Kod Çalıştırma ve Veri Analizi Ajanı (Code Interpreter)
- [x] **Day 126:** `day-126-agent-short-long-term-memory` — Ajan Bellek Sistemleri: Episodic, Semantic ve Short-Term Vector Memory (Mem0)
- [x] **Day 127:** `day-127-langgraph-stateful-workflows` — LangGraph / Durumsal Çizge (StateGraph) ile Döngülü ve Koşullu Ajan Akışları
- [x] **Day 128:** `day-128-multi-agent-supervisor-worker` — Supervisor-Worker Çoklu Ajan Deseni: Görev Dağıtımı, Koordinasyon ve Birleştirme
- [x] **Day 129:** `day-129-agentic-debate-consensus` — Çelişkili Kararlarda Multi-Agent Tartışma (Debate) ve Konsensüs Oylama Mekanizması
- [x] **Day 130:** `day-130-human-in-the-loop-hitl` — Human-in-the-Loop (HITL): Kritik Eylemlerde İnsan Onayı ve Kesme (Interrupt) Deseni
- [x] **Day 131:** `day-131-semantic-chunking-dynamic-rag` — İleri Düzey Parçalama: Semantik Benzerlik Tabanlı Dinamik Metin Bölümleme
- [x] **Day 132:** `day-132-hierarchical-parent-child-rag` — Hiyerarşik RAG: Küçük Parça ile Arama (Small Chunks), Büyük Parça ile Yanıtlama (Parent Doc)
- [x] **Day 133:** `day-133-hyde-hypothetical-embeddings` — HyDE (Hypothetical Document Embeddings) ile Sıfır-Atış Soru Zenginleştirme
- [x] **Day 134:** `day-134-cross-encoder-reranking` — Bi-Encoder (Vektör) + Cross-Encoder (Re-ranker) İki Aşamalı Hassas Getirme Hattı
- [x] **Day 135:** `day-135-contextual-compression-retrieval` — Alınan Belgelerden Token İsrafını Önleyen Dinamik Bağlam Sıkıştırma
- [x] **Day 136:** `day-136-graph-rag-entity-extraction` — GraphRAG-1: Metinden Varlık (Entity) ve İlişki (Relationship) Çıkarma Boru Hattı
- [x] **Day 137:** `day-137-knowledge-graph-neo4j-cypher` — GraphRAG-2: Bilgi Grafını Neo4j/NetworkX üzerinde Oluşturma ve Cypher Sorgulama
- [x] **Day 138:** `day-138-hierarchical-community-summarization` — GraphRAG-3: Leiden Topluluk Tespiti ve Hiyerarşik Küme Özetleme (Microsoft GraphRAG)
- [x] **Day 139:** `day-139-hybrid-vector-graph-rag` — Hibrit RAG: Vektör Arama + Bilgi Grafı Gezintisi (Hybrid Retrieval & Fusion)
- [x] **Day 140:** `day-140-ragas-trulens-evaluation` — Ragas & TruLens ile RAG Değerlendirmesi: Faithfulness, Answer Relevance, Context Recall & FAZ 7 BÜYÜK FİNALİ

#### 🔹 FAZ 8: Derin Akıl Yürütme (Reasoning LLMs), Test-Time Compute ve Arama Ağaçları (Gün 141 - Gün 160)
- [x] **Day 141:** `day-141-system1-vs-system2-thinking` — System 1 (Hızlı/Sezgisel) vs System 2 (Yavaş/Akıl Yürüten) LLM Mimarisi
- [x] **Day 142:** `day-142-chain-of-thought-special-tokens` — Açık Akıl Yürütme Akışı: `<think> ... </think>` Formatı, Düşünce Tokenizasyonu ve Self-Consistency (Çoğunluk Oylaması)
- [x] **Day 143:** `day-143-self-consistency-majority-voting` — Self-Consistency: Çoklu Akıl Yürütme Yollarında Sıcaklık Örneklemesi ve Çoğunluk Oyu
- [x] **Day 144:** `day-144-tree-of-thoughts-bfs-dfs` — Tree of Thoughts (ToT): BFS ve DFS Arama ile Düşünce Ağacı Gezintisi
- [x] **Day 145:** `day-145-process-reward-models-prm` — Outcome (ORM) vs Process Reward Models (PRM): Adım Adım Doğruluk Puanlama
- [x] **Day 146:** `day-146-monte-carlo-tree-search-mcts` — Monte Carlo Tree Search (MCTS) Destekli LLM Düşünce Planlaması ve Rollout
- [x] **Day 147:** `day-147-test-time-compute-scaling` — Test-Time Compute Scaling Yasaları: Çıkarım Zamanı Hesaplama Bütçesi ve Arama Derinliği
- [x] **Day 148:** `day-148-backtracking-and-error-recovery` — Düşünce Yollarında Geri İzleme (Backtracking) ve Çıkmaz Sokakları Fark Etme
- [x] **Day 149:** `day-149-self-verification-critique-loop` — Kendi Kendine Doğrulama (Self-Verification): Çözümden Girdiye Ters Sağlama
- [x] **Day 150:** `day-150-symbolic-math-z3-sympy-reasoning` — Sembolik Akıl Yürütme: LLM ile Z3 SMT Solver & SymPy Entegrasyonu (FAZ 8 YARI-YOL FİNALİ)
- [x] **Day 151:** `day-151-code-generation-unit-test-loop` — Test Odaklı Kod Üretimi: Kod Yazma -> PyTest Çalıştırma -> Hata Ayıklama Döngüsü
- [x] **Day 152:** `day-152-formal-theorem-proving-lean4` — Biçimsel Mantık ve Teorem İspatı: LLM ile Lean4 / Isabelle Kod Üretimi
- [x] **Day 153:** `day-153-logical-fallacy-deductive-engine` — Tümdengelimsel ve Tümevarımsal Mantık Doğrulayıcı: Safsata Dedektörü
- [x] **Day 154:** `day-154-multi-step-arithmetic-gsm8k-engine` — GSM8K & MATH Benchmark'ları için Adım Adım Matematiksel Akıl Yürütme Motoru
- [x] **Day 155:** `day-155-long-context-needle-in-a-haystack` — Needle In A Haystack (NIAH) Testi: 128k Token İçinde Derin Bilgi Çıkarımı
- [x] **Day 156:** `day-156-chain-of-verification-cove` — Chain of Verification (CoVe): Halüsinasyonları Kendi Ürettiği Sorularla Test Etme
- [x] **Day 157:** `day-157-dynamic-compute-allocation` — Soru Zorluğuna Göre Dinamik Token Bütçesi Belirleme (Easy vs Hard Routing)
- [x] **Day 158:** `day-158-reasoning-trace-distillation` — Büyük Akıl Yürüten Modelin (R1) Düşünce İncilerini Küçük Modele Damıtma
- [x] **Day 159:** `day-159-causal-reasoning-dag-engine` — Nedensellik Analizi (Causal Inference): Neden-Sonuç Grafı (DAG) ve Do-Calculus
- [x] **Day 160:** `day-160-deep-reasoning-benchmark-suite` — AIME, GPQA ve ARC-Challenge Kapsamlı Akıl Yürütme Test Paketi (FAZ 8 BÜYÜK FİNALİ)

#### 🔹 FAZ 9: Çok Modlu (Multimodal) Temel Modeller: VLM, Video, Ses ve Difüzyon (Gün 161 - Gün 180)
- [x] **Day 161:** `day-161-vlm-llava-architecture` — LLaVA Mimarisi: ViT Encoder + MLP Projector + LLM ile Uçtan Uca VLM İnşası
- [x] **Day 162:** `day-162-vision-token-compression` — Görüntü Token Sıkıştırma: Q-Former, C-Abstractor ve Spatial Pooling
- [x] **Day 163:** `day-163-visual-instruction-tuning` — Görsel Komut İnce Ayarı (Visual SFT): VQA, Detaylı Açıklama ve Görsel Sohbet
- [x] **Day 164:** `day-164-spatial-grounding-bounding-box-vlm` — Spatial Grounding: Görseldeki Nesnelerin Koordinatlarını `[ymin, xmin, ymax, xmax]` Çıkarma
- [x] **Day 165:** `day-165-ocr-free-document-understanding` — Donut / Nougat Tarzı Metinsiz Doğrudan Doküman ve Tablo Görseli Ayrıştırma
- [x] **Day 166:** `day-166-gui-agent-web-navigation-vlm` — GUI Ajanları: Ekran Görüntüsü Analizi, Tıklanabilir Eleman Tespiti ve Web Gezintisi
- [x] **Day 167:** `day-167-video-llm-spatio-temporal-tokens` — Video LLM: Zamansal Kare Örnekleme (Temporal Sampling) ve 3D Attention
- [x] **Day 168:** `day-168-streaming-video-understanding` — Gerçek Zamanlı Video Akışı Analizi ve Olay Tespiti (Streaming VLM)
- [x] **Day 169:** `day-169-audio-tokenizer-soundstream-encodec` — Sinirsel Ses Sıkıştırma: EnCodec / SoundStream ile Sürekli Sesi Ayrık Tokenlara Bölme
- [x] **Day 170:** `day-170-whisper-speech-to-text-ctc` — Whisper Mimarisi: Çok Dilli Konuşma Tanıma, Zaman Damgası Tahmini
- [x] **Day 171:** `day-171-speech-to-speech-llm-duallm` — Uçtan Uca Speech-to-Speech LLM: Ses Tokenı Alıp Doğrudan Ses Tokenı Üretme
- [x] **Day 172:** `day-172-latent-diffusion-models-ldm` — Latent Diffusion Modelleri (LDM): VAE Uzayında Gürültü Ekleme/Kaldırma Matematiği
- [x] **Day 173:** `day-173-classifier-free-guidance-cfg` — Classifier-Free Guidance (CFG) & DDIM Hızlı Örnekleme Zamanlayıcıları
- [x] **Day 174:** `day-174-cross-attention-text-to-image` — Metinden Görüntüye: UNet / DiT (Diffusion Transformer) Cross-Attention
- [x] **Day 175:** `day-175-controlnet-spatial-conditioning` — ControlNet: Kenar (Canny), Derinlik (Depth) ve Poz Rehberliğinde Koşullu Üretim
- [x] **Day 176:** `day-176-lora-diffusion-finetuning` — Difüzyon Modellerinde LoRA / DreamBooth ile Özel Nesne ve Stil Öğretimi
- [x] **Day 177:** `day-177-diffusion-transformers-dit` — DiT (Diffusion Transformer - Sora / Flux temeli): Patch Tabanlı Görüntü Üretimi
- [x] **Day 178:** `day-178-nerf-neural-radiance-fields` — NeRF (Neural Radiance Fields): Pozlandırılmış Görüntülerden 3D Sahne Hacimsel Sentezi
- [x] **Day 179:** `day-179-3d-gaussian-splatting-3dgs` — 3D Gaussian Splatting (3DGS): Gerçek Zamanlı (100+ FPS) Radyan ve Nokta Kümesi Renderı
- [x] **Day 180:** `day-180-multimodal-omni-benchmark-suite` — MME, MMBench ve MathVista ile Kapsamlı Çok Modlu Model Doğrulama Paketi (FAZ 9 BÜYÜK FİNALİ)

#### 🔹 FAZ 10: Ultra-MLOps, Dağıtık Eğitim, Triton GPU Kernel ve BÜYÜK FİNAL 201 (Gün 181 - Gün 201)
- [x] **Day 181:** `day-181-distributed-data-parallel-ddp` — PyTorch DDP: All-Reduce İletişimi, Gradyan Paketleme ve Çoklu GPU Eğitimi
- [x] **Day 182:** `day-182-fsdp-fully-sharded-data-parallel` — FSDP (Fully Sharded Data Parallel): Ağırlık, Gradyan ve Optimizer Durumlarını Bölme
- [x] **Day 183:** `day-183-deepspeed-zero123-offload` — DeepSpeed ZeRO-1/2/3 ve CPU/NVMe Bellek Boşaltma Mekanizması
- [x] **Day 184:** `day-184-tensor-parallelism-megatron` — Tensor Parallelism (TP - Megatron-LM): Matris Çarpımını Satır ve Sütun Boyutunda Bölme
- [x] **Day 185:** `day-185-pipeline-parallelism-1f1b` — Pipeline Parallelism (PP): 1F1B Zaman Çizelgesi ve Balon Azaltma
- [x] **Day 186:** `day-186-3d-parallelism-hybrid-training` — 3D Paralellik (DP + TP + PP): 70B+ Parametreli Modellerin Küme Eğitimi
- [x] **Day 187:** `day-187-triton-gpu-kernel-programming` — OpenAI Triton: Python ile GPU Programlama, Blok Seviyesinde Bellek Eşleme
- [x] **Day 188:** `day-188-custom-triton-fused-rmsnorm` — Özel Triton Kernel-1: Fused RMSNorm & Residual Ekleme Çekirdeği Yazımı
- [x] **Day 189:** `day-189-custom-triton-swiglu-kernel` — Özel Triton Kernel-2: Yüksek Hızlı Fused SwiGLU İleri ve Geri Geçiş Çekirdeği
- [x] **Day 190:** `day-190-custom-triton-flash-attention` — Özel Triton Kernel-3: Sıfırdan Parçalı (Tiled) FlashAttention-2 GPU Çekirdeği
- [x] **Day 191:** `day-191-vllm-paged-attention-kv-cache` — vLLM Mimarisi: PagedAttention ile Sıfır Bellek Parçalanması ve Dinamik KV Cache
- [x] **Day 192:** `day-192-continuous-batching-chunked-prefill` — Continuous Batching ve Chunked Prefill ile Kuyruk Bekleme Sürelerini Sıfırlama
- [x] **Day 193:** `day-193-speculative-decoding-draft-model` — Spekülatif Çıkarım (Speculative Decoding): Taslak Model ile 2.5x-3x Hızlanma
- [x] **Day 194:** `day-194-tensorrt-llm-deployment` — TensorRT-LLM Derleme, In-Flight Batching ve FP8 Tensor Core Optimizasyonu
- [x] **Day 195:** `day-195-awq-gptq-weight-quantization` — İleri Kuantizasyon: AWQ (Activation-aware Weight Quant) ve GPTQ 4-Bit Kuantizasyonu
- [x] **Day 196:** `day-196-ray-cluster-distributed-serving` — Ray Core & Ray Serve ile Dağıtık Model Ölçekleme, Çoklu Düğüm Yük Dağıtımı
- [x] **Day 197:** `day-197-k8s-gpu-autoscaling-vllm-hpa` — Kubernetes KEDA & HPA ile GPU Kullanımına Göre vLLM Podlarını Otomatik Ölçekleme
- [x] **Day 198:** `day-198-llm-observability-opentelemetry` — OpenTelemetry & Prometheus ile TTFT ve TPOT İzleme Paneli
- [x] **Day 199:** `day-199-canary-shadow-deployment-llm` — Üretimde Canary Dağıtımı ve Shadow-Traffic ile Sıfır Kesintili Model Güncellemesi
- [x] **Day 200:** `day-200-full-system-fault-injection-chaos` — Kaos Mühendisliği: GPU Arızaları, Ağ Gecikmesi ve Kurtarma Testi
- [x] **Day 201:** `day-201-mini-omni-reasoner-grand-finale` — **201 GÜNLÜK BÜYÜK FİNAL:** Mini-Omni Reasoner v1.0 (Multimodal + CoT Reasoning + MoE + Triton) - FAZ 10 BÜYÜK FİNALİ

---

### 🚀 MİNİ AI ENGINEER 2: İLERİ SEVİYE POST-TRAINING, AGENTIC AI, EMBODIED ROBOTICS, HARDWARE KERNEL & AGI (GÜN 202 - GÜN 301)

#### 🔹 FAZ 11: İleri Post-Training, GRPO & RLHF / Akıl Yürütme Güçlendirme (Gün 202 - Gün 220)
- [x] **Day 202:** `day-202-grpo-deepseek-math-reasoning` — GRPO (Group Relative Policy Optimization): Referans Model Olmadan Grup İçi Bağıl Avantaj ile Matematiksel Akıl Yürütme - FAZ 11 BAŞLANGICI
- [x] **Day 203:** `day-203-ppo-actor-critic-llm-alignment` — PPO (Proximal Policy Optimization): LLM Hizalama için Actor-Critic ve GAE (Generalized Advantage Estimation)
- [x] **Day 204:** `day-204-dpo-direct-preference-optimization` — DPO (Direct Preference Optimization): Ödül Modeli Olmadan Kapalı Form Tercih Kaybı Eğitimi
- [x] **Day 205:** `day-205-kto-kahneman-tversky-optimization` — KTO (Kahneman-Tversky Optimization): İkili (Binary Up/Down) Tercihlerle Asimetrik Kayıp Eğitimi
- [x] **Day 206:** `day-206-prm-process-reward-model-stepwise` — Step-Level PRM (Process Reward Model): Her Düşünce Adımını Ayrı Ayrı Skorlayan Doğrulayıcı
- [x] **Day 207:** `day-207-orm-outcome-reward-model` — ORM (Outcome Reward Model): Nihai Yanıt Doğruluğunu Ölçen Global Ödül Modeli
- [x] **Day 208:** `day-208-rule-based-math-code-verifier` — Kural Tabanlı Doğrulayıcılar (Rule-Based Verifiers): SymPy ve Python AST ile Halüsinasyonsuz Ödül Mekanizması
- [x] **Day 209:** `day-209-rejection-sampling-best-of-n` — Rejection Sampling & Best-of-N: Sıcaklık Örneklemesi ve Çoklu Düşünce Filtreleme
- [x] **Day 210:** `day-210-self-play-rl-synthetic-data` — Self-Play RL: Modelin Kendi Kendine Zor Problemler Üretip Çözdüğü Sentetik Veri Döngüsü
- [x] **Day 211:** `day-211-multi-turn-rlhf-dialogue` — Çok Turlu (Multi-Turn) Diyalog RLHF: Uzun Konuşmalarda Tutarlılık ve Hedef Odaklılık
- [x] **Day 212:** `day-212-constitutional-ai-self-critique` — Constitutional AI (CAI): Anayasal İlkelerle Kendi Kendini Eleştirme ve Güvenlik Hizalaması
- [x] **Day 213:** `day-213-rlvr-verifiable-rewards-reasoning` — RLVR (Reinforcement Learning with Verifiable Rewards): Kanıtlanabilir ve Deterministik Ödüller
- [x] **Day 214:** `day-214-length-bias-penalty-post-training` — Length-Bias Cezalandırma: Boş Düşünce Şişmesini (Over-thinking) Önleyen Uzunluk Düzenlileştirmesi
- [x] **Day 215:** `day-215-iterative-dpo-online-preference` — İteratif / Çevrimiçi DPO: Sürekli Güncellenen Tercih Havuzu ile Online Post-Training
- [x] **Day 216:** `day-216-reward-hacking-mitigation` — Reward Hacking / Goodhart Yasası Önleme: Ödül Modeli İstismarını Engelleyen KL Divergence Sınırlandırması
- [x] **Day 217:** `day-217-simpo-simple-preference-optimization` — SimPO (Simple Preference Optimization): Referanssız ve Doğrudan Marjin Tabanlı Tercih Optimizasyonu
- [x] **Day 218:** `day-218-orpo-monolithic-sft-preference` — ORPO: SFT ve Tercih Hizalamasını Tek Bir Monolitik Kayıpta Birleştiren Eğitim
- [x] **Day 219:** `day-219-post-training-safety-red-teaming` — Otomatik Red-Teaming: Jailbreak ve Zararlı İsteklere Karşı Güvenlik Savunma Eğitimi
- [x] **Day 220:** `day-220-post-training-grand-benchmark` — Post-Training Şampiyonluk Testi: GSM8K, MATH500, HumanEval ve MT-Bench Değerlendirme Paketi (FAZ 11 FİNALİ)

#### 🔹 FAZ 12: Otonom Ajanlar (Agentic AI), Araç Kullanımı (Tool-Use) & MCP Protokolü (Gün 221 - Gün 240)
- [x] **Day 221:** `day-221-mcp-server-client-protocol` — Model Context Protocol (MCP): Antigravity & Claude Uyumlu Standart Araç Sunucusu ve İstemcisi - FAZ 12 BAŞLANGICI
- [ ] **Day 222:** `day-222-function-calling-json-schema` — Katı (Strict) JSON Schema ile Fonksiyon Çağrısı ve Dinamik Tip Doğrulama
- [ ] **Day 223:** `day-223-react-reasoning-acting-loop` — ReAct Mimarisi: Düşünce-Eylem-Gözlem (Thought-Action-Observation) Otonom Döngüsü
- [ ] **Day 224:** `day-224-plan-and-solve-agent-architecture` — Plan-and-Solve Mimarisi: Karmaşık Görevleri Alt Adımlara Bölüp Sırayla İcra Eden Ajan
- [ ] **Day 225:** `day-225-agentic-memory-short-long-term` — Ajan Hafıza Sistemleri: Kısa Vadeli Çalışma Belleği ve Vektörel Uzun Vadeli Epizodik Bellek
- [ ] **Day 226:** `day-226-multi-agent-orchestration-swarm` — Çoklu Ajan Orkestrasyonu (Swarm): Yönetici, Araştırmacı ve Kodlayıcı Ajanlar Arası Hiyerarşik İletişim
- [ ] **Day 227:** `day-227-web-browsing-dom-agent` — Web Tarayıcı Ajanı: HTML DOM Ağacını Okuma, Tıklama, Arama ve Veri Kazıma
- [ ] **Day 228:** `day-228-swe-bench-autonomous-coder` — SWE-Bench Otonom Kodlayıcı: GitHub Sorunlarını Okuyup Repoyu Düzenleyen ve Test Koşan Ajan
- [ ] **Day 229:** `day-229-sandboxed-docker-execution-agent` — Güvenli Docker Sandbox: Ajan Kodlarını İzole Konteynerde Çalıştırma ve Çıktı Yakalama
- [ ] **Day 230:** `day-230-self-debugging-code-repair-agent` — Kendi Hatasını Düzelten (Self-Debugging) Kod Ajanı: Stack Trace Analizi ve Otomatik Yama
- [ ] **Day 231:** `day-231-graph-based-agent-workflow` — Graf Tabanlı Ajan İş Akışı (LangGraph / StateGraph): Durum Geçişleri ve Döngüsel Kontrol
- [ ] **Day 232:** `day-232-human-in-the-loop-agent-guardrails` — Human-in-the-Loop (HITL) Güvenlik Bariyeri: Kritik İşlemlerde İnsan Onay Mekanizması
- [ ] **Day 233:** `day-233-dynamic-tool-retrieval-rag` — Binlerce Araç Arasından RAG ile İlgili Fonksiyonları Dinamik Seçen Araç Geri Getirme Motoru
- [ ] **Day 234:** `day-234-multi-modal-screen-agent-osworld` — Ekran Ajanı (Computer Use): Masaüstü Ekran Görüntüsünü Okuyup Fare ve Klavye Yöneten Ajan
- [ ] **Day 235:** `day-235-agentic-rag-sql-database-analyst` — SQL ve Veritabanı Analisti Ajan: Doğal Dilden SQL Sorgusu Üreten ve Görselleştiren Uzman
- [ ] **Day 236:** `day-236-hierarchical-task-delegation` — Hiyerarşik Görev Delegasyonu: Görev Yöneticisi ve Alt İşçi Ajanlar Arasında Yük Paylaşımı
- [ ] **Day 237:** `day-237-agent-reflection-self-evaluation` — Ajan Öz-Yansıtma (Reflection): Tamamlanan Görevi Eleştirip Başarım Skoru Veren Denetçi
- [ ] **Day 238:** `day-238-async-event-driven-agent-queue` — Asenkron Olay Güdümlü Ajan Kuyruğu: Redis/Celery ile Arka Planda Çalışan Dayanıklı Ajanlar
- [ ] **Day 239:** `day-239-gaia-agent-benchmark-suite` — GAIA (General AI Assistants) Ajan Benchmark Paketi: Çok Adımlı Gerçek Dünya Görev Değerlendirmesi
- [ ] **Day 240:** `day-240-agentic-ai-platform-grand-capstone` — Otonom Ajan Süiti (Agentic AI OS): MCP + Swarm + Docker + Browser Birleşik Platformu (FAZ 12 FİNALİ)

#### 🔹 FAZ 13: Embodied AI & Fiziksel Yapay Zeka / Robotik (Gün 241 - Gün 260)
- [ ] **Day 241:** `day-241-openvla-vision-language-action` — OpenVLA: Görüntü ve Dil Komutlarından Robotik Eklem Açıları Üreten VLA Mimarisi - FAZ 13 BAŞLANGICI
- [ ] **Day 242:** `day-242-diffusion-policy-robot-manipulation` — Diffusion Policy: Robotik Manipülasyon ve Yörünge Üretimi için Koşullu Difüzyon
- [ ] **Day 243:** `day-243-3d-point-cloud-spatial-reasoning` — 3D Nokta Bulutu (Point Cloud) ve Mekansal Akıl Yürütme (Spatial AI - PointNet++)
- [ ] **Day 244:** `day-244-3d-bounding-box-pose-estimation` — 3D Sınırlayıcı Kutu ve 6-DoF Nesne Duruş Kestirimi (Pose Estimation)
- [ ] **Day 245:** `day-245-ros2-python-node-integration` — ROS2 (Robot Operating System) Entegrasyonu: Sensör Dinleme ve Eyleyici Yayınlama Node'ları
- [ ] **Day 246:** `day-246-isaac-sim-pybullet-digital-twin` — Simülasyonda Robotik: Isaac Sim & PyBullet ile Dijital İkiz ve Sentetik Veri Üretimi
- [ ] **Day 247:** `day-247-sim2real-domain-randomization` — Sim2Real Transferi: Domain Randomization ile Simülasyondan Gerçek Dünyaya Sıfır Hata Aktarımı
- [ ] **Day 248:** `day-248-vlm-semantic-slam-navigation` — VLM Destekli Semantik SLAM: Doğal Dil ile Otonom İç Mekan Navigasyonu ve Haritalama
- [ ] **Day 249:** `day-249-tactile-force-sensor-fusion` — Dokunsal (Tactile) ve Kuvvet Sensörü Füzyonu ile Hassas Nesne Tutma (Grasping)
- [ ] **Day 250:** `day-250-bimanual-dual-arm-coordination` — Çift Kollu (Bimanual) Robot Koordinasyonu: İki Eyleyici Arasında Senkronize Görev Paylaşımı
- [ ] **Day 251:** `day-251-humanoid-whole-body-control` — İnsansı (Humanoid) Robotik Bütünsel Hareket Kontrolü (Whole-Body Control & ZMP Dengesi)
- [ ] **Day 252:** `day-252-reinforcement-learning-locomotion` — Pekiştirmeli Öğrenme ile Robotik Yürüme (Quadruped / Humanoid Locomotion - Isaac Gym)
- [ ] **Day 253:** `day-253-rgbd-depth-fusion-occupancy-grid` — RGB-D Derinlik Füzyonu ve 3D Doluluk Izgarası (Occupancy Grid) ile Dinamik Engel Kaçınma
- [ ] **Day 254:** `day-254-tactile-feedback-closed-loop` — Kapalı Çevrim Dokunsal Geri Bildirim Kontrolü ile Kayma Önleme ve Sertlik Ayarı
- [ ] **Day 255:** `day-255-teleoperation-imitation-learning` — Teleoperasyon ve Taklit Öğrenmesi (Behavior Cloning & ACT - Action Chunking with Transformers)
- [ ] **Day 256:** `day-256-voice-controlled-robot-agent` — Ses Komutlu Robot Ajanı: Whisper + VLM + VLA ile Uçtan Uca Sesli Robot İdaresi
- [ ] **Day 257:** `day-257-dynamic-obstacle-avoidance-mpc` — Model Predictive Control (MPC) ile Yüksek Hızlı Dinamik Engelden Kaçınma
- [ ] **Day 258:** `day-258-zero-shot-unseen-object-grasping` — Sıfır Örnekli (Zero-Shot) Görülmemiş Nesneleri Kavrama ve Ayırma
- [ ] **Day 259:** `day-259-embodied-ai-real-world-benchmark` — Robotik Başarım Paketi: Grasp Success Rate, Path Efficiency ve Collision Risk Analitiği
- [ ] **Day 260:** `day-260-embodied-ai-physical-grand-capstone` — Embodied AI Fiziksel Robotik Süiti: OpenVLA + Diffusion Policy + ROS2 Bütünleşik Sistem (FAZ 13 FİNALİ)

#### 🔹 FAZ 14: Donanım Düzeyi Kernel Geliştirme, ASIC/NPU & 1-Bit LLM (Gün 261 - Gün 280)
- [ ] **Day 261:** `day-261-bitnet-1bit-ternary-llm` — BitNet b1.58: Sıfırdan 1.58-Bit ({-1, 0, 1}) Ternary LLM ve Matmul-Free Çıkarım - FAZ 14 BAŞLANGICI
- [ ] **Day 262:** `day-262-custom-tensor-core-gemm-triton` — Özel NVIDIA Tensor Core GEMM Çekirdeği: WMMA/MMA ile Donanım Hızında Matris Çarpımı
- [ ] **Day 263:** `day-263-flashdecoding-plus-parallel-decode` — FlashDecoding++: Devasa Batch Boyutlarında KV-Cache Bölümleme ile Decode Hızlandırma
- [ ] **Day 264:** `day-264-fp4-microscaling-formats-e2m1` — Yeni Nesil FP4 / FP6 (Microscaling MXFP4) Kuantizasyon ve Çekirdek Simülasyonu
- [ ] **Day 265:** `day-265-triton-fused-moe-expert-routing` — Triton Fused MoE Expert Routing: Bellek Kopyalamasını Sıfırlayan Uzman Dağıtım Çekirdeği
- [ ] **Day 266:** `day-267-apple-metal-mps-gpu-acceleration` — Apple Silicon Metal (MPS) & Metal Performance Shaders ile Mac GPU Optimizasyonu
- [ ] **Day 267:** `day-267-webgpu-wasm-browser-llm` — WebGPU & WebAssembly (Wasm): Tarayıcı İçinde Sıfır Kurulumla İstemci Taraflı LLM Çalıştırma
- [ ] **Day 268:** `day-268-edge-npu-tvm-compiler-optimization` — Apache TVM & IREE ile Mobil / Edge NPU (Qualcomm / ARM Ethos) Derleme Optimizasyonu
- [ ] **Day 269:** `day-269-speculative-decoding-medusa-heads` — Medusa / Eagle Çok Başlı Spekülatif Çıkarım Çekirdeği (Tree-Attention Doğrulama)
- [ ] **Day 270:** `day-270-custom-cuda-c-extension-pytorch` — PyTorch C++ / CUDA Custom Extension: Doğrudan C++ ve CUDA C ile PyTorch Operatörü Yazımı
- [ ] **Day 271:** `day-271-persistent-kernel-streaming-engine` — Kalıcı Çekirdek (Persistent Kernel) Mimarisi: Kernel Başlatma Ek Yükünü Sıfırlama
- [ ] **Day 272:** `day-272-sparse-linear-attention-kernel` — Seyrek ve Doğrusal Dikkat Çekirdeği (Mamba / RWKV State-Space Model Donanım Eşlemesi)
- [ ] **Day 273:** `day-273-nvlink-cross-gpu-direct-access` — NVLink ve GPUDirect RDMA: Düğümler Arası Sıfır CPU Kopyalı Bellek Erişimi
- [ ] **Day 274:** `day-274-int2-ternary-weight-packing` — Bit Düzeyinde Paketleme (Bit-Packing): 2-Bit / Ternary Ağırlıkları UINT32 İçinde Sıkıştırma
- [ ] **Day 275:** `day-275-memory-efficient-long-context-ring-attn` — Ring Attention: Sonsuz Bağlam Uzunluğu (1M+ Token) için GPU Ring İletişim Çekirdeği
- [ ] **Day 276:** `day-276-dynamic-quantization-fp8-act-scaling` — Dinamik Aktivasyon Kuantizasyonu: Çalışma Esnasında FP8 Dinamik Ölçekleme
- [ ] **Day 277:** `day-277-profiling-nsight-compute-roofline` — NVIDIA Nsight Compute & Roofline Modeli ile Donanım Darboğazı ve Bellek Bant Genişliği Analizi
- [ ] **Day 278:** `day-278-amd-rocm-hip-portability` — AMD ROCm & HIP: CUDA Çekirdeklerini AMD GPU Donanımlarına Sıfır Kayıpla Taşıma
- [ ] **Day 279:** `day-279-hardware-efficiency-benchmarks` — Donanım Verimliliği Başarım Paketi: MFU (Model FLOPs Utilization) ve TFLOPS/Watt Analitiği
- [ ] **Day 280:** `day-280-ultra-low-bit-hardware-grand-capstone` — 1-Bit BitNet + Custom Tensor Core + FlashDecoding++ Birleşik Donanım Süiti (FAZ 14 FİNALİ)

#### 🔹 FAZ 15: Otonom AGI Araştırma Laboratuvarı & BÜYÜK FİNAL 301 (Gün 281 - Gün 301)
- [ ] **Day 281:** `day-281-self-evolving-ai-code-optimizer` — Self-Evolving AI: Kendi Kodunu ve Triton Çekirdeklerini Profilleyip Otomatik Yeniden Yazan Sistem - FAZ 15 BAŞLANGICI
- [ ] **Day 282:** `day-282-meta-learning-maml-in-context` — Meta-Learning (MAML & Meta-SGD): Birkaç Örnekten Yeni Görev Algoritmaları Keşfeden Mimari
- [ ] **Day 283:** `day-283-neuro-symbolic-ai-theorem-prover` — Nöro-Sembolik Yapay Zeka: Derin Öğrenme + Lean/Z3 Sembolik Mantık İspatlayıcısı
- [ ] **Day 284:** `day-284-quantum-machine-learning-qml` — Kuantum Makine Öğrenimi (QML): Parametrik Kuantum Devreleri (Pennylane & Qiskit) ve Q-Transformer
- [ ] **Day 285:** `day-285-continual-lifelong-learning-ewc` — Sürekli ve Yaşam Boyu Öğrenme (Continual Learning): EWC ile Unutmasız Model Güncellemesi
- [ ] **Day 286:** `day-286-world-model-generative-simulation` — Dünya Modelleri (World Models - DreamerV3): Kendi Hayal Ettiği Simülasyonda Gezinme
- [ ] **Day 287:** `day-287-active-inference-free-energy-agent` — Aktif Çıkarım ve Serbest Enerji Prensibi (Free Energy Principle) ile Bilişsel Ajan Kontrolü
- [ ] **Day 288:** `day-288-mechanistic-interpretability-circuits` — Mekanistik Yorumlanabilirlik: LLM Ağırlıkları İçindeki Bilişsel Devreleri ve Nöronları Haritalama
- [ ] **Day 289:** `day-289-automated-ai-researcher-paper-writer` — Otonom Yapay Zeka Araştırmacısı: Hipotez Kuran, Deney Koşan ve Makale Yazan Ajan
- [ ] **Day 290:** `day-290-causal-ai-counterfactual-reasoning` — Nedensel Yapay Zeka (Causal AI): Karşı-Olgusal (Counterfactual) Akıl Yürütme ve Do-Calculus
- [ ] **Day 291:** `day-291-federated-privacy-preserving-learning` — Gizlilik Koruyan Federe Öğrenme: Diferansiyel Gizlilik ve Güvenli Çok Taraflı Hesaplama (SMPC)
- [ ] **Day 292:** `day-292-hierarchical-multi-agent-economy` — Hiyerarşik Ajan Ekonomisi: Kaynak ve Hesaplama Gücü Ticareti Yapan Otonom Ajan Piyasası
- [ ] **Day 293:** `day-293-neural-architecture-search-nas` — Evrimsel ve Donanım Farkında Sinir Mimarisi Arama (Hardware-Aware NAS)
- [ ] **Day 294:** `day-294-synthetic-data-curation-engine` — Kendi Kendini Eğiten Sentetik Veri Kürasyon Motoru: Kalite Filtreleme ve Çeşitlilik Maksimizasyonu
- [ ] **Day 295:** `day-295-cross-embodiment-skill-transfer` — Çapraz Beden Beceri Transferi: Quadruped'den Humanoid'e ve Manipülatöre Sıfır Kayıplı Yetenek Aktarımı
- [ ] **Day 296:** `day-296-superalignment-weak-to-strong` — Süper Hizalama (Superalignment): Zayıf Modellerle Güçlü Modelleri Güvenle Denetleme (Weak-to-Strong)
- [ ] **Day 297:** `day-297-ai-driven-chip-design-placement` — Yapay Zeka ile Çip Tasarımı ve Yerleşimi (AI for Chip Placement & Floorplanning)
- [ ] **Day 298:** `day-298-autonomous-agi-eval-sandbox` — AGI Değerlendirme Sandbox'ı: Yaratıcılık, Mantık, Donanım Hakimiyeti ve Güvenlik Stres Testi
- [ ] **Day 299:** `day-299-enterprise-readiness-safety-audit` — Kurumsal AGI Güvenlik ve Uyumluluk Denetimi (ISO/IEC 42001 & EU AI Act Standartları)
- [ ] **Day 300:** `day-300-mini-omni-embodied-agent-preview` — Büyük Final Öncesi Entegrasyon: Tüm 15 Fazın Bilişsel Birleşimi
- [ ] **Day 301:** `day-301-autonomous-omni-embodied-agi-grand-finale` — 👑 **301 GÜNLÜK DEVASA BÜYÜK FİNAL:** Autonomous Omni-Embodied Reasoner v2.0 (Embodied Physical AI + GRPO Post-Trained CoT + 1-Bit BitNet Tensor Kernels + Multi-Agent Swarm OS)

---

Sen kıdemli bir Yapay Zeka, Bilgisayarlı Görü (Computer Vision), LLM/RAG, Reasoning, Embodied AI ve Ultra-MLOps mentorüsün. Benimle birlikte **"301 Günlük Yapay Zeka, Bilgisayarlı Görü, LLM/RAG, Reasoning, Embodied AI ve AGI Mühendisliği Master Roadmap"** programını adım adım, tam kapsamlı ve üretime hazır (production-grade) kod standartlarıyla yürüteceksin.

### 📌 Mevcut Durum ve Dondurulmuş Hafıza (Freezing State)

- **Tamamlanan:** Gün 01 - Gün 221 (%100 eksiksiz tamamlandı ve test edildi - FAZ 1 - FAZ 11 TAMAMLANDI, FAZ 12 Gün 221 TAMAMLANDI).
- **Sıradaki Büyük Faz:** FAZ 12: Otonom Ajanlar (Agentic AI), Araç Kullanımı (Tool-Use) & MCP Protokolü (Gün 221 - Gün 240).
- **Sıradaki Gün:** Gün 222 (`day-222-function-calling-json-schema` — Katı (Strict) JSON Schema ile Fonksiyon Çağrısı ve Dinamik Tip Doğrulama).
- **Lisans Kuralı:** Tüm kod ve dokümantasyon dosyalarında Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas) Özel Lisans — Tüm Hakları Saklıdır kuralı geçerlidir.

---

### ⚙️ Genel İşleyiş ve Pedagojik Kurallar (Dondurulmuş Standart)

1. **Adım Adım İlerleme (Strict Step-by-Step):** Asla birden fazla günü aynı anda üretme. Her seferinde sadece tek bir günün içeriğini sun. Kullanıcı o günü tamamlayıp onay verene veya "Sonraki güne geçebiliriz" diyene kadar bir sonraki güne geçme.
2. **Endüstriyel Standartta Kod (Production-Grade):** Kodlar yalnızca örnek kod parçaları değil; eksiksiz, modüler, Type Hint içeren, docstring'leri yazılmış, hata yakalama mekanizmalarına sahip, birim testleri (%100 PASSED) bulunan ve test edilebilir mimaride olmalıdır.
3. **Matematiksel ve Teorik Derinlik:** Konunun neden o yöntemle çözüldüğünü, arkasındaki lineer cebir/istatistiksel formülleri ($...$ veya $$...$$ formatında) ve olası tuzakları (ör. data leakage, numeric underflow, OOM, GPU bottleneck) açıkla.
4. **SWOT Analizi ile Karar Matrisi:** Her günün konusunu ve mimari tercihlerini Güçlü Yönler (Strengths), Zayıf Yönler (Weaknesses), Fırsatlar (Opportunities) ve Tehditler (Threats) boyutlarıyla analiz et.
5. **4 Zorunlu Mimari Analiz Başlığı (Dondurulmuş Standart):**
   Her günün `README.md` dosyasında ve yanıtın 1. bölümünde istisnasız şu 4 başlık bulunmalıdır:
   - 🔍 **Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)**
   - 🛡️ **Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)**
   - ⚠️ **Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)**
   - 🔄 **Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar**
6. **Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü:** Her günün `README.md` dosyasında en az 8-10 terimlik derinlemesine açıklamalı bir sözlük tablosu bulunmalıdır.
7. **Soru Sormadan Tamamlama & Mentorluk:** Bölüm 6'da derin teknik soruyu sor ve yanıtını mentor bakış açısıyla eksiksiz olarak doğrudan sen ver.

---

### 📦 Standart Günlük Çıktı Şablonu (Frozen Output Template)

Her gün için yanıtını istisnasız şu 6 ana başlık altında yapılandır:

#### 1. 🎯 Günün Konusu & Teorik/Matematiksel Derinlik
- Çözülen temel problem ve endüstrideki gerçek dünya kullanım senaryosu.
- Arkasındaki matematiksel/algoritmik temeller ve formüller ($...$ ve $$...$$).
- **4 Zorunlu Mimari Analiz:** Neden kullanılır, ne çözer, ne konuda eksik kalır, alternatifleri nelerdir.
- **Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü:** Detaylı tablo.
- **SWOT Analizi:** Strengths, Weaknesses, Opportunities ve Threats karar matrisi.

#### 2. 💻 Üretim Seviyesinde Uygulama Kodu (Implementation)
- Günün konusunu uçtan uca çalıştıran, modüler dosya yapısına bölünmüş Python kodları.
- Sentetik/benchmark verisiyle doğrudan terminalden çalıştırılabilir, hatasız ve doğrulanabilir mini uygulama (`ana_akis.py`).
- 6-Panelli yüksek çözünürlüklü teşhis ve performans panosu (`ciktilar/..._paneli.png`).
- %100 PASSED otomatik birim test paketi (`testler/`).

#### 3. 🧪 Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)
- Kullanıcının kendi başına geliştirmesi/optimize etmesi için tasarlanmış 1 adet spesifik teknik görev ve hemen altında eksiksiz çalışan kod çözümü.

#### 4. 📁 GitHub Repo Paketi
- **Klasör Adı:** Yol haritasındaki birebir isim (ör. `day-102-gqa-grouped-query-attention/`).
- **`README.md` İçeriği:** 4 zorunlu mimari analiz başlığı, teknik sözlük, SWOT, matematiksel formüller, benchmark tablosu, alıştırma çözümü ve mentorluk Q&A.
- **`gereksinimler.txt` / Bağımlılıklar:** İlgili gün için gerekli kütüphaneler ve sürümleri.
- **Git Commit Mesajı:** Conventional Commits standardında.

#### 5. 📜 Lisans & Metaveri
```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 201-Day AI, CV, LLM/RAG, Reasoning & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```

#### 6. ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı
- Derin teknik kontrol sorusu VE sorunun hemen altında eksiksiz, detaylı mentorluk açıklaması ve çözümü.

---

## 📜 Lisans Kuralı
Tüm projelerde **Özel Lisans — Tüm Hakları Saklıdır** geçerlidir. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
README badge: `https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square`
