# 🧠 101 Günlük Yapay Zeka, Bilgisayarlı Görü, LLM/RAG ve MLOps Mühendisliği Master Hafıza Dosyası (Master Roadmap)

Bu dosya; projenin 1. gününden 101. gününe kadar tüm yol haritasını, tamamlanan ve planlanan günleri, modül hedeflerini ve teknik derinliklerini kayıt altında tutan **merkezi hafıza (memory) belgesidir**.

---

## 📌 FAZ ÖZETİ VE DURUM TABLOSU

| Faz | Gün Aralığı | Alan & Kapsam | Durum |
|---|---|---|---|
| **FAZ 1** | Gün 01 - Gün 15 | Veri Temelleri, Görüntü İşleme, Renk Uzayları ve Segmentasyon | ✅ **TAMAMLANDI (%100)** |
| **FAZ 2A** | Gün 16 - Gün 30 | İleri Bilgisayarla Görme, YOLO Nesne Tespiti, U-Net, Mask R-CNN, DeepSORT, Çoklu Görev | ✅ **TAMAMLANDI (%100)** |
| **FAZ 2B** | Gün 31 - Gün 41 | Leksikal/Semantik Arama (BM25, Dense, RRF), RAG, API Servisleri ve Sektörel Halı Zekası | 🔄 **BAŞLIYOR (Gün 31'den Devam)** |
| **FAZ 3** | Gün 42 - Gün 66 | Çekirdek ML/DL Boru Hatları, FAISS, Sayısal Kararlılık, ONNX INT8 Kuantizasyon & Edge MLOps | ⏳ Sırada |
| **FAZ 4** | Gün 67 - Gün 81 | İleri Düzey Temsil Öğrenimi (SimCLR, SupCon), Sıfırdan Vision Transformer (ViT) & LoRA | ⏳ Sırada |
| **FAZ 5** | Gün 82 - Gün 101 | Model Sıkıştırma (Distillation, Pruning), Kalibrasyon, MLOps, Hugging Face Dağıtımı & MoE | ⏳ Sırada |

---

## 📋 DETAYLI GÜN GÜN YOL HARİTASI

### ✅ Tamamlanan Günler (Gün 01 - Gün 30)
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

---

### 🚀 Planlanan ve Sırayla İşlenecek Günler (Gün 31 - Gün 101)

#### 🔹 FAZ 2B: Arama Motorları, RAG ve Sektörel Halı/Tekstil Zekası (Gün 31 - Gün 41)
- [x] **Day 31:** `day-31-bm25-document-search` — BM25 Leksikal Arama Algoritması, TF-IDF, Ters İndeks (Inverted Index), Tokenizasyon
- [x] **Day 32:** `day-32-semantic-search-engine` — Sentence Transformers ile Yoğun (Dense) Vektör Arama, Kosinüs Benzerliği
- [x] **Day 33:** `day-33-hybrid-search-rrf` — BM25 + Vektör Arama Birleşimi, Reciprocal Rank Fusion (RRF) Hibrit Sıralama
- [x] **Day 34:** `day-34-mini-rag-assistant` — Mini RAG Asistanı, Chunking, Vektör Veritabanı Sorgulama, LLM Context Injection
- [x] **Day 35:** `day-35-fastapi-model-service` — FastAPI Asenkron REST API, Pydantic Tip Güvenliği, Model Servisleme
- [x] **Day 36:** `day-36-streamlit-ai-dashboard` — Streamlit ile İnteraktif AI Kontrol Paneli, Dosya Yükleme, Model Görselleştirme
- [x] **Day 37:** `day-37-carpet-color-intelligence` — Halı/Tekstil Renk Ayrıştırma, İplik Renk Oranları Çıkarımı, Katalog Uyumu
- [x] **Day 38:** `day-38-carpet-visual-retrieval` — Halı Doku ve Desenleri İçin Çoklu Özellikli (Renk+Doku) Görsel Arama
- [x] **Day 39:** `day-39-carpet-defect-detector` — Halı Dokuma Hataları, Leke ve Kusur Tespiti, Kalite Kontrol Otomasyonu
- [ ] **Day 40:** `day-40-carpet-knowledge-rag` — Tekstil ve Üretim Teknik Dokümanları Üzerinde Sektörel RAG Sistemi
- [ ] **Day 41:** `day-41-ai-carpet-intelligence-suite` — Renk, Arama, Kusur ve RAG Modüllerini Birleştiren Uçtan Uca Halı Zekası Paketi

#### 🔹 FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps (Gün 42 - Gün 66)
- [ ] **Day 42:** `day-42-numpy-ai-batch-inspector` — Üretim Girdi Tensörleri Doğrulama, Batch Boyutu & NaN/Inf/Shape Anomali Tespiti
- [ ] **Day 43:** `day-43-numpy-data-drift-detector` — Veri Kayması (Data Drift) Tespiti, KS-Test İstatistiği, Wasserstein Mesafesi
- [ ] **Day 44:** `day-44-pandas-data-quality-cleaner` — Üretim Seviyesi Şema Doğrulama, Sınır Değer Kontrolleri, Otomatik Temizlik
- [ ] **Day 45:** `day-45-pandas-feature-engineering-profile-builder` — Özellik Mühendisliği, Encoding, Ölçeklendirme, Feature Store Mimarisi
- [ ] **Day 46:** `day-46-matplotlib-ai-experiment-report-generator` — Otomatik Loss/Acc, PR, ROC Grafikleri ve PDF/HTML Deney Raporlama Motoru
- [ ] **Day 47:** `day-47-sklearn-leakage-safe-ml-pipeline` — Veri Sızıntısına Karşı Güvenli Pipeline, ColumnTransformer & Nested CV
- [ ] **Day 48:** `day-48-kmeans-unsupervised-segmentation` — Elbow & Silhouette Analizi, Uzamsal (Spatial) Piksel K-Means Bölütleme
- [ ] **Day 49:** `day-49-xgboost-tabular-risk-classifier` — Dengesiz Tabüler Veri, scale_pos_weight, XGBoost ile Risk/Dolandırıcılık Tespiti
- [ ] **Day 50:** `day-50-model-evaluation-threshold-engineering` — Eşik Değeri Mühendisliği, F-beta Optimizasyonu, Maliyet-Fayda Karar Matrisi
- [ ] **Day 51:** `day-51-pillow-safe-image-loader` — Bozuk Dosya & Yanlış EXIF Yönetimi, Hataya Toleranslı Görsel Yükleyici
- [ ] **Day 52:** `day-52-opencv-visual-defect-inspector` — FFT Frekans Analizi, Laplacian Varyansı ile Bulanıklık & Kural Tabanlı Kusur Tespiti
- [ ] **Day 53:** `day-53-cielab-kmeans-palette-analyzer` — Perceptually Uniform LAB Uzayında K-Means & Delta-E 2000 Hassas Tolerans Analizi
- [ ] **Day 54:** `day-54-image-forensics-inspector` — Dijital Adli Bilişim, Error Level Analysis (ELA), Görsel Manipülasyon Tespiti
- [ ] **Day 55:** `day-55-pytorch-dataset-dataloader` — İleri PyTorch DataLoader, num_workers, pin_memory Darboğaz Optimizasyonu
- [ ] **Day 56:** `day-56-tinyvisioncnn` — Edge Cihazlar İçin Sıfırdan Hafif CNN, Depthwise Separable Conv, FLOPs Hesabı
- [ ] **Day 57:** `day-57-pytorch-training-engine` — Modüler Eğitim Motoru, Checkpoint, Early Stopping, Gradient Clipping
- [ ] **Day 58:** `day-58-amp-numerical-stability-benchmark` — Otomatik Karma Hassasiyet (AMP), FP16 vs BF16, GradScaler & Sayısal Kararlılık
- [ ] **Day 59:** `day-59-transfer-learning-embedding-extractor` — ViT/ResNet Omurgalarından Dondurulmuş Katmanlarla L2-Normalize Embedding Çıkarımı
- [ ] **Day 60:** `day-60-faiss-similarity-search-engine` — FAISS ile Milyonluk Vektör İndeksleme (IndexFlatIP, IndexIVFFlat, HNSW, GPU)
- [ ] **Day 61:** `day-61-retrieval-metrics-benchmark` — Vektör Arama Değerlendirmesi: NDCG@k, MRR (Mean Reciprocal Rank), Gecikme Testi
- [ ] **Day 62:** `day-62-sdxl-lora-controlled-generator` — Üretken AI: Stable Diffusion XL (SDXL) + LoRA ile Kontrollü Görsel Üretimi
- [ ] **Day 63:** `day-63-pydantic-ai-domain-models` — Pydantic v2 ile Tip Güvenli Girdi/Çıktı Sözleşmeleri & Domain Modelleri
- [ ] **Day 64:** `day-64-fastapi-inference-api` — Üretim Seviyesi FastAPI İnference, Model Yaşam Döngüsü (lifespan), Batch Prediction
- [ ] **Day 65:** `day-65-streamlit-sqlite-ai-dashboard` — SQLite Destekli CRUD, Model Çıkarım Logları ve Kalıcı AI Yönetim Paneli
- [ ] **Day 66:** `day-66-onnx-int8-production-capstone` — PyTorch Modellerini ONNX'e Aktarma, INT8 PTQ Kuantizasyon & ONNX Runtime Hızlandırma

#### 🔹 FAZ 4: İleri Düzey Eğitim, Temsil Öğrenimi ve Sıfırdan Vision Transformer (Gün 67 - Gün 81)
- [ ] **Day 67:** `day-67-config-driven-reproducible-training` — YAML/Hydra ile Konfigürasyon Yönetimi, Deterministik & Tekrarlanabilir Eğitim
- [ ] **Day 68:** `day-68-high-performance-vision-data-pipeline` — Albumentations ile Yüksek Performanslı Veri Artırma & GPU Prefetching
- [ ] **Day 69:** `day-69-optimizer-scheduler-laboratory` — AdamW vs Lion Optimizer, CosineAnnealing, Linear Warmup & Weight Decay Dinamikleri
- [ ] **Day 70:** `day-70-modern-regularization-mixup-cutmix-label-smoothing` — Mixup, CutMix Veri Artırma ve Label Smoothing Cross-Entropy Düzenlileştirmesi
- [ ] **Day 71:** `day-71-fault-tolerant-resumable-training-engine` — Çökmeye Dayanıklı Checkpoint, State Restoration ve Devam Edebilir Eğitim Motoru
- [ ] **Day 72:** `day-72-embedding-geometry` — t-SNE, UMAP Boyut İndirgeme, Temsil Uzayı Geometrisi & İzotropi Analizi
- [ ] **Day 73:** `day-73-simclr-from-scratch` — Sıfırdan SimCLR Temsil Öğrenimi, Artırma Çiftleri, NT-Xent (InfoNCE) Kaybı
- [ ] **Day 74:** `day-74-supervised-contrastive-learning` — Etiketli Veride Supervised Contrastive (SupCon) Kaybı ile Sınıf Ayrıştırma
- [ ] **Day 75:** `day-75-metric-learning-triplet-hard-negative` — Triplet Margin Loss, Hard/Semi-Hard Negative Mining Stratejileri
- [ ] **Day 76:** `day-76-representation-benchmark-suite` — Temsil Kalitesi Değerlendirmesi: Linear Probing ve k-NN Sınıflandırma Protokolü
- [ ] **Day 77:** `day-77-self-attention-from-scratch` — Sıfırdan Scaled Dot-Product & Multi-Head Self-Attention Mekanizması
- [ ] **Day 78:** `day-78-transformer-encoder-from-scratch` — Sıfırdan Transformer Encoder Bloğu: Pozisyonel Kodlama, LayerNorm, Residual FFN
- [ ] **Day 79:** `day-79-minivit-from-scratch` — Sıfırdan Mini Vision Transformer (Patch Projeksiyonu, CLS Token, Encoder Birleşimi)
- [ ] **Day 80:** `day-80-minivit-cifar10-training` — Sıfırdan MiniViT'in CIFAR-10 Üzerinde Eğitimi & Regülarizasyon Dinamikleri
- [ ] **Day 81:** `day-81-vit-lora-peft` — Vision Transformer İçin LoRA (Low-Rank Adaptation) ile Parametre-Verimli İnce Ayar

#### 🔹 FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı (Gün 82 - Gün 101)
- [ ] **Day 82:** `day-82-knowledge-distillation` — Öğretmen-Öğrenci Modeli Bilgi Damıtma, Soft Target Loss (KL-Diverjansı), Temperature
- [ ] **Day 83:** `day-83-structured-pruning` — L1/L2 Norm Tabanlı Yapısal Filtre/Kanal Budama, Hız vs Doğruluk Dengesi
- [ ] **Day 84:** `day-84-calibration-uncertainty` — Olasılık Kalibrasyonu, Expected Calibration Error (ECE) & Temperature Scaling
- [ ] **Day 85:** `day-85-ood-selective-prediction` — Enerji Tabanlı Dağılım Dışı (OOD) Tespiti ve Seçici Tahmin (Abstention)
- [ ] **Day 86:** `day-86-robustness-domain-shift` — Görsel Bozulmalar (Bulanıklık/Gürültü) Altında Model Dayanıklılığı & Domain Shift
- [ ] **Day 87:** `day-87-experiment-registry` — MLflow / Weights & Biases ile Merkezi Deney Takibi ve Artefakt Kayıt Sistemi
- [ ] **Day 88:** `day-88-optuna-hpo` — Optuna ile Otomatik Hiperparametre Optimizasyonu (TPE Algoritması, Pruning)
- [ ] **Day 89:** `day-89-model-registry` — Model Kayıt Sistemi, Model Sürümleme, Staging/Production Yaşam Döngüsü
- [ ] **Day 90:** `day-90-dynamic-batching-inference` — GPU Verimliliği İçin Kuyruk Tabanlı Dinamik Batching Çıkarım Motoru
- [ ] **Day 91:** `day-91-ai-observability` — Canlı AI Sistemlerinde Gözlemlenebilirlik: Gecikme, Hacim ve Veri Kayması İzleme
- [ ] **Day 92:** `day-92-final-training-contract` — Eğitim Öncesi Veri Sözleşmesi Testleri ve Hazır Bulunuşluk (Readiness) Kontrolleri
- [ ] **Day 93:** `day-93-final-evaluation-model-card` — Kapsamlı Değerlendirme, Yanlılık (Bias) Testleri ve Standart Model Card Üretimi
- [ ] **Day 94:** `day-94-hugging-face-integration` — Hugging Face Model Hub Entegrasyonu, Konfigürasyon ve Model Paketleme
- [ ] **Day 95:** `day-95-minivit-v1-release-candidate` — MiniViT v1 Sürüm Adayı (Release Candidate), Uçtan Uca Regresyon Testleri
- [ ] **Day 96:** `day-96-huggingface-public-v1-release` — MiniViT v1.0 Hugging Face Canlı Dağıtımı & Canlı Model Demosu
- [ ] **Day 97:** `day-97-reproducible-inference` — Deterministik Çıkarım, Donanımdan Bağımsız Doğrulama Testleri
- [ ] **Day 98:** `day-98-fastapi-inference-service` — Üretime Hazır Yüksek Performanslı Asenkron API & `/health` Kontrolleri
- [ ] **Day 99:** `day-99-container-load-testing` — Docker Konteynerleştirme ve Locust/k6 ile Eşzamanlı Yük/Stres Testleri
- [ ] **Day 100:** `day-100-modern-architecture-ablations` — SwiGLU, RMSNorm ve FlashAttention Mimarileri ile MiniViT Ablasyon Analizleri
- [ ] **Day 101:** `day-101-huggingface-minivit-moe-v2` — **101 GÜNLÜK BÜYÜK FİNAL:** MiniViT Mixture of Experts (MoE) v2 Hugging Face Dağıtımı

---

## 📜 Lisans Kuralı
Tüm projelerde **Özel Lisans — Tüm Hakları Saklıdır** geçerlidir. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas).
