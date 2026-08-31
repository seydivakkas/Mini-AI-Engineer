# 📱 TURKCELL YAPAY ZEKA, VERİ BİLİMİ VE BÜYÜK VERİ MÜHENDİSLİĞİ 200 GÜNLÜK STAJ & PORTFÖY HAFIZA PLANI

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Kurum: Turkcell](https://img.shields.io/badge/Kurumsal-Turkcell%20%7C%20Paycell%20%7C%20fizy%20%7C%20TV%2B%20%7C%20BiP-blue.svg?style=flat-square)](https://www.turkcell.com.tr/)
[![Veri Kümeleri: Açık Kaynak](https://img.shields.io/badge/Veri%20Kaynaklar%C4%B1-Kaggle%20%7C%20HuggingFace%20%7C%20Roboflow%20%7C%20UCI-green.svg?style=flat-square)](https://kaggle.com)
[![Format: Python + Jupyter .ipynb](https://img.shields.io/badge/Format-Python%203.11%20%2B%20Jupyter%20.ipynb-orange.svg?style=flat-square)](https://jupyter.org)

---

## 🎯 Programın Amacı & Kapsamı

Bu hafıza dokümanı, **Turkcell'in 6 ana iş kolunda** (Telekomünikasyon Şebekesi, Fintek/Paycell, Dijital Medya fizy/TV+, NLP/Müşteri Deneyimi, Bilgisayarlı Görü ve MLOps/Büyük Veri) staj ve junior/mid veri bilimci pozisyonlarında doğrudan fark yaratacak, internetteki açık kaynaklı gerçek veri setleriyle birebir doğrulanmış **200 adet uygulamalı mini projenin** mimarisini, veri kaynaklarını, Türkçe değişken adlarını ve `.ipynb` şablonlarını içerir.

Her proje şu standart bileşenlerle inşa edilir:
1. **Veri Kümesi (Kaggle / Hugging Face / Roboflow / UCI)**
2. **Algoritmik / İstatistiksel Model**
3. **%100 Anlaşılır Türkçe Değişken ve Fonksiyon Adları**
4. **Jupyter Notebook (`.ipynb`) Yapısı ve Çalışma Akışı**
5. **Turkcell Staj & Mülakat Odaklı Değerlendirme Sorusu**

> [!TIP]
> **💻 %100 YEREL (LOCAL), ÜCRETSİZ VE AÇIK KAYNAK GÜVENCESİ:**
> Bu müfredattaki 200 projenin tamamı, öğrencinin/mühendisin kendi yerel bilgisayarında (CPU veya standart GPU) ve ücretsiz Google Colab / Kaggle ortamlarında **sıfır maliyetle ($0)** çalışacak şekilde tasarlanmıştır. Hiçbir projede ücretli API anahtarı (OpenAI, Anthropic, Google Cloud Paid API vb.) veya harici ücretli donanım/bulut kaynağı gerekmez. Tüm LLM ve NLP görevleri açık kaynak yerel modeller (Ollama, HuggingFace Transformers, Qwen2.5, LLaMA-3.2, BERTurk) ile yürütülür.

---


# 📚 200 GÜNLÜK MÜFREDAT MODÜL DAĞILIMI

```
[BÖLÜM 1: GÜN 1 - 100]
├── Modül 01: Müşteri Analitiği, Churn, CRM & Gelir Optimizasyonu (Gün 001 - 015)
├── Modül 02: Şebeke, Ağ Trafiği & Zaman Serileri (Gün 016 - 030)
├── Modül 03: Doğal Dil İşleme (NLP), Müşteri Hizmetleri & LLM (Gün 031 - 045)
├── Modül 04: Bilgisayarlı Görü (Computer Vision) & Saha Denetimi (Gün 046 - 060)
├── Modül 05: Fintek, Paycell & Fraud / Dolandırıcılık Tespiti (Gün 061 - 075)
├── Modül 06: Ses İşleme & Çağrı Analitiği (Audio AI) (Gün 076 - 085)
├── Modül 07: Öneri Sistemleri, TV+, fizy & Dijital Servisler (Gün 086 - 095)
└── Modül 08: IoT, Akıllı Şehir & Edge AI (Gün 096 - 100)

[BÖLÜM 2: GÜN 101 - 200]
├── Modül 09: Telekom Şebeke Optimizasyonu, Radyo & 5G Altyapısı (Gün 101 - 115)
├── Modül 10: Fintek / Paycell, Dijital Cüzdan & Alternatif Risk (Gün 116 - 130)
├── Modül 11: Dijital Servisler (TV+, fizy, lifebox, BiP, Dergilik) (Gün 131 - 145)
├── Modül 12: İleri Seviye NLP, LLM, Müşteri Deneyimi & Agentic AI (Gün 146 - 160)
├── Modül 13: Bilgisayarlı Görü, Saha Operasyonları & Güvenlik (Gün 161 - 175)
├── Modül 14: Siber Güvenlik, Ağ Savunması & Tehdit İstihbaratı (Gün 176 - 185)
├── Modül 15: MLOps, Veri Mühendisliği & Dağıtık Akış (Gün 186 - 195)
└── Modül 16: Sürdürülebilirlik, Yeşil Telekom & Enerji Verimliliği (Gün 196 - 200)
```

---

# 🚀 BÖLÜM 1: GÜN 001 – 100

## 📊 Modül 01: Müşteri Analitiği, Churn & CRM (Gün 001 – 015)

### Gün 001: Telco Müşteri Kayıp (Churn) Tahmini
- **İş Alanı:** Turkcell Bireysel Müşteri Analitiği
- **Veri Kaynağı:** [Kaggle - Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Model:** CatBoostClassifier / XGBoost + SHAP Açıklanabilirlik
- **Türkçe Değişkenler:** `musteri_id`, `sozlesme_turu`, `aylik_odeme_tutari`, `toplam_harcama`, `ayrilma_riski_orani`, `model_tahmini`
- **Jupyter Notebook (`gun_001_telco_churn_tahmini.ipynb`):**
  1. Veri yükleme ve eksik değerlerin imputasyonu
  2. Kategorik değişkenlerin One-Hot & Target Encoding dönüşümleri
  3. Sınıf dengesizliği yönetimi (SMOTE / Scale_pos_weight)
  4. CatBoost eğitimi ve ROC-AUC optimizasyonu
  5. Müşteri bazlı churn risk skorlama çıktısının CSV olarak kaydedilmesi
- **Mülakat Sorusu:** Dengesiz churn veri setinde neden Accuracy yerine PR-AUC veya F1-Score kullanılır?

### Gün 002: Müşteri Yaşam Boyu Değeri (CLTV) Modellemesi
- **İş Alanı:** Pazarlama & Gelir Planlama
- **Veri Kaynağı:** [Kaggle - Online Retail & Customer Analytics](https://www.kaggle.com/datasets/vijayuv/onlineretail)
- **Model:** BG/NBD (Beta-Geometric / Negative Binomial) + Gamma-Gamma Monetization Modeli
- **Türkçe Değişkenler:** `abone_id`, `islem_sikligi`, `musteri_yasi_hafta`, `ortalama_fatura_tutari`, `beklenen_gelecek_gelir`
- **Jupyter Notebook (`gun_002_musteri_yasam_boyu_degeri.ipynb`):**
  1. RFM (Recency, Frequency, Monetary) öznitelik türetimi
  2. BG/NBD ile 3 ve 6 aylık beklenen işlem frekansı tahmini
  3. Gamma-Gamma ile ortalama marjinal kâr hesabı
  4. Abone değer segmentasyonu (VIP, Sadık, Riskli)

### Gün 003: RFM Tabanlı Abone Segmentasyonu
- **İş Alanı:** Turkcell CRM & Kampanya Yönetimi
- **Veri Kaynağı:** [Kaggle - Credit Card Customer Segmentation](https://www.kaggle.com/datasets/arjunbhasin2005/ccdata)
- **Model:** K-Means Kümeler + UMAP / PCA Boyut İndirgeme
- **Türkçe Değişkenler:** `son_islem_gunu`, `islem_adedi`, `toplam_odeme_tutari`, `kume_etiketi`, `segment_adi`
- **Jupyter Notebook (`gun_003_rfm_abone_segmentasyonu.ipynb`):**
  1. Logaritmik ölçekleme ve StandardScaler normalizasyonu
  2. Elbow yöntemi ve Silhouette skoru ile optimal K seçimi
  3. Segment profilleme (Şampiyonlar, Uyuyanlar, Kaybedilmemesi Gerekenler)

### Gün 004: Faturasızdan Faturalıya Tarife Terfi (Upselling) Modeli
- **İş Alanı:** Satış & Kanal Yönetimi
- **Veri Kaynağı:** [Kaggle - Bank Marketing / Product Upsell](https://www.kaggle.com/datasets/henriqueyama/bank-marketing)
- **Model:** LightGBM + Optuna Hiperparametre Optimizasyonu
- **Türkçe Değişkenler:** `faturasiz_kullanim_suresi_ay`, `ortalama_tl_yukleme`, `kota_asimi_sikligi`, `faturali_gecis_egilimi`
- **Jupyter Notebook (`gun_004_tarife_terfi_upsell.ipynb`):**
  1. Paket doluluk oranı ve veri tüketim trend analizi
  2. Optuna ile LightGBM hiperparametre araması
  3. Kampanya hedef kitlesi için olasılık eşik optimizasyonu

### Gün 005: Net Promoter Score (NPS) / Memnuniyet Tahmini
- **İş Alanı:** Müşteri Deneyimi Yönetimi (CEM)
- **Veri Kaynağı:** [Kaggle - Customer Satisfaction Dataset](https://www.kaggle.com/datasets/santander-customer-satisfaction)
- **Model:** Random Forest Regressor & Ordinal Regression
- **Türkçe Değişkenler:** `cagri_merkezi_arama_sayisi`, `baglanti_kopma_adedi`, `fatura_itiraz_durumu`, `tahmini_nps_puani`
- **Jupyter Notebook (`gun_005_nps_memnuniyet_tahmini.ipynb`):**
  1. Çok değişkenli korelasyon ve VIF (Multicollinearity) analizi
  2. Öznitelik önem derecelerinin belirlenmesi
  3. Memnuniyetsiz aboneler için erken uyarı raporu

### Gün 006: Faturasız Hat TL/Paket Yükleme Zamanı Tahmini
- **İş Alanı:** Paycell & Dijital Operatör
- **Veri Kaynağı:** [Kaggle - Mobile Money Transaction](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Model:** Survival Analysis (Cox Proportional Hazards) / XGBoost Regressor
- **Türkçe Değişkenler:** `kalan_tl_bakiyesi`, `son_yukleme_uzerinden_gecen_gun`, `tahmini_gelecek_yukleme_gunu`
- **Jupyter Notebook (`gun_006_tl_yukleme_zamani_tahmini.ipynb`)**

### Gün 007: Fatura Ödeme Gecikmesi Tahminleyicisi
- **İş Alanı:** Finans & Alacak Yönetimi
- **Veri Kaynağı:** [Kaggle - Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Model:** XGBoost + Cost-Sensitive Learning
- **Türkçe Değişkenler:** `gecikmis_fatura_adedi`, `son_3_ay_ortalama_fatura`, `gecikme_olasiligi_skoru`
- **Jupyter Notebook (`gun_007_fatura_odeme_gecikmesi.ipynb`)**

### Gün 008: Müşteri İtiraz (Dispute) Olasılığı Modeli
- **İş Alanı:** Fatura İtiraz & Şikayet Yönetimi
- **Veri Kaynağı:** [Kaggle - Consumer Complaint Database](https://www.kaggle.com/datasets/selener/consumer-complaint-database)
- **Model:** Logistic Regression & Gradient Boosting
- **Türkçe Değişkenler:** `aylik_fatura_artis_orani`, `yurt_disi_roaming_harcamasi`, `itiraz_riski_puani`
- **Jupyter Notebook (`gun_008_musteri_itiraz_modeli.ipynb`)**

### Gün 009: Cihaz Yenileme (Handset Upgrade) Eğilimi
- **İş Alanı:** Pasaj (Turkcell E-Ticaret)
- **Veri Kaynağı:** [Kaggle - Mobile Phone Usage Dataset](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset)
- **Model:** Random Forest Classifier
- **Türkçe Değişkenler:** `mevcut_cihaz_yasi_ay`, `batarya_saglik_skoru`, `veri_kullanim_artisi`, `yeni_cihaz_alacagi_tarih`
- **Jupyter Notebook (`gun_009_cihaz_yenileme_egilimi.ipynb`)**

### Gün 010: Abonelik İptal Nedenlerini Sınıflandırma
- **İş Alanı:** Müşteri Kazanım ve İkna Masası
- **Veri Kaynağı:** [Kaggle - Subscription Churn Telecom](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Model:** Multi-Class LightGBM
- **Türkçe Değişkenler:** `iptal_gerekce_kodu`, `rakip_operatore_gecis`, `fiyat_kaynakli_iptal`, `cekim_gucu_sorunu`
- **Jupyter Notebook (`gun_010_iptal_nedenleri_siniflandirma.ipynb`)**

### Gün 011: Çapraz Satış (Cross-Selling) Modeli (TV+, Superonline, Paycell)
- **İş Alanı:** Çoklu Ürün Stratejisi
- **Veri Kaynağı:** [Kaggle - Multi-Product Financial/Telecom Data](https://www.kaggle.com/datasets)
- **Model:** Multi-Output Classifier / Stacking Ensemble
- **Türkçe Değişkenler:** `ev_interneti_aktif`, `fizy_kullanimi_saat`, `paycell_islem_hacmi`, `tvplus_satin_alma_ihtimali`
- **Jupyter Notebook (`gun_011_capraz_satis_cross_sell.ipynb`)**

### Gün 012: Müşteri Kayıp Riski Erken Uyarı Motoru
- **İş Alanı:** Gerçek Zamanlı CRM
- **Veri Kaynağı:** [Kaggle - Telecom Churn BigML](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)
- **Model:** Z-Score Anomalisi + Karar Ağacı
- **Türkçe Değişkenler:** `son_30_gun_veri_degisimi`, `arama_suresi_dusus_orani`, `erken_uyari_tetiklendi`
- **Jupyter Notebook (`gun_012_churn_erken_uyari_motoru.ipynb`)**

### Gün 013: Fiyat Esnekliği (Price Elasticity of Demand) Analizi
- **İş Alanı:** Gelir Yönetimi & Fiyatlandırma
- **Veri Kaynağı:** [Kaggle - Telecom Pricing & Demand](https://www.kaggle.com/datasets)
- **Model:** Log-Log OLS Regresyonu
- **Türkçe Değişkenler:** `paket_fiyat_artisi_yuzde`, `talep_degisimi_yuzde`, `fiyat_esneklik_katsayisi`
- **Jupyter Notebook (`gun_013_fiyat_esnekligi_analizi.ipynb`)**

### Gün 014: Pasifleşen (Dormant) Hatları Geri Kazanım Modeli
- **İş Alanı:** Yeniden Etkinleştirme Kampanyaları
- **Veri Kaynağı:** [Kaggle - Subscription Reactivation](https://www.kaggle.com/datasets)
- **Model:** Uplift Modeling (CausalML / Two-Model Approach)
- **Türkçe Değişkenler:** `inaktif_gun_sayisi`, `kampanya_teklifi`, `uplift_skoru`, `kazanim_ihtimali`
- **Jupyter Notebook (`gun_014_inaktif_hat_kazanim.ipynb`)**

### Gün 015: Dijital Kanallara Geçiş Eğilimi Modeli
- **İş Alanı:** Turkcell Dijital Operatör Dönüşümü
- **Veri Kaynağı:** [Kaggle - Digital Channel Adoption](https://www.kaggle.com/datasets)
- **Model:** XGBoost Classifier
- **Türkçe Değişkenler:** `fiziksel_magaza_ziyaret_sayisi`, `web_giris_sikligi`, `dijitale_gecis_skoru`
- **Jupyter Notebook (`gun_015_dijital_kanal_donusum.ipynb`)**

---

## 📡 Modül 02: Şebeke, Ağ Trafiği & Zaman Serileri (Gün 016 – 030)

### Gün 016: Baz İstasyonu İnternet Trafik Tahmini
- **İş Alanı:** Şebeke Planlama & Kapasite Yönetimi
- **Veri Kaynağı:** [Kaggle - Telecom Italia SMS/Call/Internet Grid Data](https://www.kaggle.com/datasets/marcodena/mobile-phone-activity)
- **Model:** LSTM / Facebook Prophet / Temporal Fusion Transformer
- **Türkçe Değişkenler:** `hucre_id`, `zaman_damgasi`, `saatlik_indirilen_veri_gb`, `yuklenen_veri_gb`, `tahmin_edilen_trafik_gb`
- **Jupyter Notebook (`gun_016_baz_istasyonu_trafik_tahmini.ipynb`):**
  1. Spatio-temporal ızgara verisini hücre bazında ayrıştırma
  2. Saatlik ve haftalık mevsimsellik (Seasonality) çıkarımı
  3. LSTM ile 24 saatlik ileri yönlü trafik tahmini
  4. Aşırı yüklenme (Congestion) eşik kontrolleri

### Gün 017: Ağ İhlal ve Dağıtık Hizmet Engelleme (DDoS) Tespiti
- **İş Alanı:** Turkcell Siber Güvenlik Operasyon Merkezi (SOC)
- **Veri Kaynağı:** [Kaggle - CICIDS2017 / NSL-KDD](https://www.kaggle.com/datasets/cicdataset/cicids2017)
- **Model:** Random Forest & Autoencoder Anomali Tespiti
- **Türkçe Değişkenler:** `kaynak_ip`, `hedef_port`, `paket_uzunluk_ortalamasi`, `saniyedeki_istek_adedi`, `saldiri_etiketi`
- **Jupyter Notebook (`gun_017_ddos_saldiri_tespiti.ipynb`)**

### Gün 018: Şebeke Gecikme (Latency) Anomali Dedektörü
- **İş Alanı:** 4.5G/5G Hizmet Kalitesi (QoS)
- **Veri Kaynağı:** [Kaggle - Numenta Anomaly Benchmark (NAB)](https://www.kaggle.com/datasets/boltzmannbrain/nab)
- **Model:** Isolation Forest & DBSCAN
- **Türkçe Değişkenler:** `ping_gecikme_ms`, `jitter_sapmasi_ms`, `paket_kayip_orani`, `anomali_durumu`
- **Jupyter Notebook (`gun_018_gecikme_anomali_dedektoru.ipynb`)**

### Gün 019: Baz İstasyonu Enerji Tüketimi Optimizasyonu
- **İş Alanı:** Yeşil Şebeke & Enerji Yönetimi
- **Veri Kaynağı:** [Kaggle - Smart Grid Energy Consumption](https://www.kaggle.com/datasets)
- **Model:** Ridge Regression / LightGBM Regressor
- **Türkçe Değişkenler:** `baz_istasyonu_turu`, `sicaklik_derecesi`, `gece_trafik_yuku`, `harcanan_guc_kwh`
- **Jupyter Notebook (`gun_019_baz_istasyonu_enerji_optimizasyonu.ipynb`)**

### Gün 020: 4G/5G Hücre Tıkanıklığı (Cell Congestion) Tahmini
- **İş Alanı:** Radyo Erişim Şebekesi (RAN)
- **Veri Kaynağı:** [Kaggle - Cellular Network QoS Data](https://www.kaggle.com/datasets)
- **Model:** CatBoost Multi-Classification
- **Türkçe Değişkenler:** `rrc_baglanti_sayisi`, `prb_kullanim_orani`, `tikaniklik_seviyesi`
- **Jupyter Notebook (`gun_020_hucre_tikaniklik_tahmini.ipynb`)**

### Gün 021: Mobil Ağ Hız Testi (QoE) Analizi
- **İş Alanı:** Mobil Şebeke Performans Değerlendirmesi & Müşteri Deneyimi (QoE)
- **Veri Kaynağı:** [Kaggle - Ookla Open Network Speedtest Data](https://www.kaggle.com/datasets/kylehatch/ookla-open-network-speedtest-data)
- **Model:** LightGBM Regressor + Coğrafi Hiyerarşik Kümeleme (Spatial K-Fold)
- **Türkçe Değişkenler:** `enlem_boylam_koordinati`, `sinyal_gucu_rsrp_dbm`, `baglanti_tipi_4g_5g`, `tahmini_indirme_hizi_mbps`, `gecikme_suresi_ms`
- **Jupyter Notebook (`gun_021_mobil_ag_hiz_testi_qoe.ipynb`):**
  1. Coğrafi H3/Hexagon ızgara koordinatlarının dönüştürülmesi
  2. Sinyal kalitesi (RSRP/RSRQ) ve baz istasyonu mesafesi öznitelikleri
  3. Spatial K-Fold ile veri sızıntısını (Data Leakage) önleyerek model eğitimi
  4. Hız düşüklüğü yaşanan kör bölgelerin harita üzerinde ısı haritası olarak görselleştirilmesi
- **Mülakat Sorusu:** Coğrafi telekom verilerini eğitirken rastgele K-Fold yerine neden Spatial / Group K-Fold kullanılır?

### Gün 022: Veri Merkezi Sunucu CPU/RAM Aşırı Yükleme Öngörüsü
- **İş Alanı:** Turkcell Bulut & Veri Merkezi Altyapı Yönetimi
- **Veri Kaynağı:** [Kaggle - Google Cloud Cluster Workload Traces](https://www.kaggle.com/datasets)
- **Model:** Gated Recurrent Unit (GRU) / WaveNet + Anomali Eşik Dedektörü
- **Türkçe Değişkenler:** `sunucu_id`, `anlik_cpu_kullanimi_yuzde`, `bellek_kullanimi_mb`, `disk_okuma_yazma_iops`, `asiri_yuklenme_riski_15dk`
- **Jupyter Notebook (`gun_022_sunucu_kaynak_asiri_yuklenme.ipynb`):**
  1. Çoklu sunucu telemetri zaman serisi pencerelenmesi (Rolling Window)
  2. GRU modeli ile 15 dakika sonrasının CPU/RAM kullanım tahmini
  3. Kaynak tükenmesi (Resource Exhaustion) öncesi otomatik pod/konteyner ölçekleme tetikleyicisi
- **Mülakat Sorusu:** Aşırı yükleme tahmininde false negative (yükü kaçırma) riskini minimize etmek için loss fonksiyonu nasıl modifiye edilir?

### Gün 023: Fiber Optik Sinyal Bozulması Tespiti
- **İş Alanı:** Superonline Fiber Omurga & İletim Şebekesi
- **Veri Kaynağı:** [Kaggle - Optical Network Performance](https://www.kaggle.com/datasets)
- **Model:** 1D-CNN (Evrişimli Sinir Ağı) / Support Vector Classifier (SVC)
- **Türkçe Değişkenler:** `fiber_hat_id`, `optik_guc_seviyesi_dbm`, `polarizasyon_mod_dagilimi_pmd`, `kromatik_dagilim_cd`, `kablo_hasar_durumu`
- **Jupyter Notebook (`gun_023_fiber_sinyal_bozulmasi.ipynb`):**
  1. Optik spektral telemetri sinyallerinin Fourier dönüşümü (FFT) ile frekans analizi
  2. 1D-CNN ile fiziksel bükülme, kırılma ve zayıflama sınıflandırması
  3. Erken arıza tespitinde Precision-Recall eğrisi analizi
- **Mülakat Sorusu:** Fiber optik ağlarda PMD ve CD parametreleri sinyal zayıflamasını nasıl etkiler ve ML ile nasıl modellenir?

### Gün 024: DNS Tünelleme ve Zararlı İstek Tespiti
- **İş Alanı:** Şebeke Güvenliği & Tehdit Avcılığı (Threat Hunting)
- **Veri Kaynağı:** [Kaggle - DNS Exfiltration & Tunneling Dataset](https://www.kaggle.com/datasets)
- **Model:** Random Forest + Karakter Düzeyi N-Gram & Shannon Entropi Hesaplayıcı
- **Türkçe Değişkenler:** `sorgulanan_alan_adi`, `alan_adi_uzunlugu`, `shannon_entropi_degeri`, `alt_alan_adi_sayisi`, `dns_tunelleme_riski`
- **Jupyter Notebook (`gun_024_dns_tunelleme_tespiti.ipynb`):**
  1. DNS sorgu metinlerinden entropi, sesli/sessiz harf oranı ve N-gram çıkarma
  2. Zararlı veri sızdırma (Data Exfiltration) tünellerinin tespit edilmesi
  3. Düşük yanlış pozitif (False Positive) oranıyla gerçek zamanlı engelleme kuralları
- **Mülakat Sorusu:** DNS tünelleme saldırısında sorgulanan alan adının Shannon Entropisi neden normal alan adlarından belirgin şekilde yüksektir?

### Gün 025: Radyo Sinyali Yayılım Kaybı (Path Loss) Tahmini
- **İş Alanı:** Radyo Frekans (RF) Planlama & Kule Konumlandırma
- **Veri Kaynağı:** [Kaggle - Radio Propagation Dataset](https://www.kaggle.com/datasets)
- **Model:** XGBoost Regressor / Çok Katmanlı Algılayıcı (MLP)
- **Türkçe Değişkenler:** `anten_yuksekligi_m`, `kullanici_mesafesi_km`, `bina_yogunlugu_morfoloji`, `tasiyici_frekans_mhz`, `tahmini_yol_kaybi_db`
- **Jupyter Notebook (`gun_025_radyo_sinyali_yayilim_kaybi.ipynb`):**
  1. Standart Okumura-Hata ve Cost-231 ampirik modelleriyle karşılaştırma
  2. Coğrafi ve bina morfolojisi özniteliklerinin modele beslenmesi
  3. RMSE ve MAE metrikleri ile klasik formüllere kıyasla doğruluk kazancı analizi
- **Mülakat Sorusu:** Klasik ampirik RF yayılım formülleri yerine Makine Öğrenmesi kullanmanın kentsel alanlardaki en büyük avantajı nedir?

### Gün 026: Hücresel Geçiş (Handover) Başarısızlık Modeli
- **İş Alanı:** Mobilite Yönetimi & Otoyol Kapsama Analitiği
- **Veri Kaynağı:** [Kaggle / UCI - Wireless Handover Analytics](https://archive.ics.uci.edu/ml/datasets.php)
- **Model:** CatBoost Classifier + Zaman Pencereli Öznitelikler
- **Türkçe Değişkenler:** `arac_hizi_kmh`, `kaynak_hucre_sinyali_rsrp`, `hedef_hucre_sinyali_rsrp`, `zaman_histerezis_farki`, `gecis_basarisiz_mi`
- **Jupyter Notebook (`gun_026_handover_gecis_basarisizligi.ipynb`):**
  1. Hızlı araç hareketlerinde sinyal düşüm eğrilerinin analizi
  2. Ping-pong handover (sürekli istasyon değiştirme) tespiti
  3. Başarısız geçişleri önleyici dinamik eşik optimizasyonu
- **Mülakat Sorusu:** Hızlı tren veya otoyollarda gerçekleşen "Too-Late Handover" arızası ML ile nasıl tahmin edilir?

### Gün 027: Ağ Trafiği Protokol ve Uygulama Sınıflandırması
- **İş Alanı:** Derin Paket İnceleme (DPI) & Bant Genişliği Yönetimi
- **Veri Kaynağı:** [Kaggle - ISCX VPN-nonVPN Network Traffic](https://www.kaggle.com/datasets)
- **Model:** 1D-CNN + Random Forest Hibrit Sınıflandırıcı
- **Türkçe Değişkenler:** `paket_akis_suresi_ms`, `ileri_yonlu_paket_boyutu`, `paketler_arasi_sure_iat`, `uygulama_tipi_video_oyun_ses`
- **Jupyter Notebook (`gun_027_ag_trafigi_protokol_siniflandirma.ipynb`):**
  1. PCAP / NetFlow akış özelliklerinin (Flow Statistics) çıkarımı
  2. Şifreli (HTTPS/VPN) trafikte paket boyutu ve zamanlama paternleriyle sınıflandırma
  3. QoS önceliklendirmesi için akış etiketleme pipeline'ı
- **Mülakat Sorusu:** Şifreli ağ trafiğinde (HTTPS/TLS) paket içeriğine bakmadan video veya oyun trafiği nasıl ayırt edilir?

### Gün 028: BGP Yönlendirme Anomalileri ve Rota Sızıntısı Tespiti
- **İş Alanı:** Uluslararası İnternet Omurgası & Rota Güvenliği
- **Veri Kaynağı:** [Kaggle - BGP Routing Anomaly Dataset](https://www.kaggle.com/datasets)
- **Model:** Isolation Forest & One-Class SVM
- **Türkçe Değişkenler:** `as_yol_uzunlugu`, `duyuru_guncelleme_sayisi`, `geri_cekme_mesaji_adedi`, `bgp_anomali_skoru`
- **Jupyter Notebook (`gun_028_bgp_yonlendirme_anomalileri.ipynb`):**
  1. BGP güncelleme mesajlarının (Announce/Withdrawal) zaman serisi analizi
  2. Rota ele geçirme (BGP Hijacking) ve rota sızıntısı (Route Leak) anomali tespiti
  3. Otomatik alarm ve prefix filtreleme öneri motoru
- **Mülakat Sorusu:** BGP Prefix Hijacking saldırısının telekom operatörü üzerindeki etkisi nedir ve anomali tespitiyle nasıl yakalanır?

### Gün 029: Dağıtık Mikroservis Yanıt Süresi (p99) Sapma Analizi
- **İş Alanı:** Turkcell Dijital Servisler Altyapısı & SRE (Site Reliability)
- **Veri Kaynağı:** [Kaggle - Microservices Telemetry Trace](https://www.kaggle.com/datasets)
- **Model:** Quantile Regression Gradient Boosting (p50, p95, p99)
- **Türkçe Değişkenler:** `servis_adi`, `gelen_istek_sayisi_rps`, `veritabani_sorgu_suresi_ms`, `kuyruk_bekleme_suresi`, `tahmini_p99_yanit_ms`
- **Jupyter Notebook (`gun_029_mikroservis_p99_sapma_analizi.ipynb`):**
  1. Dağıtık OpenTelemetry izleme (trace) kayıtlarının analizi
  2. Kuyruk gecikmesi ve veritabanı kilitlerinin p99 kuyruk sapmalarına etkisinin modellenmesi
  3. SLA ihlali oluşmadan önce erken ikaz üretimi
- **Mülakat Sorusu:** Neden ortalama yanıt süresi yerine p99/p99.9 gecikme süreleri optimize edilir?

### Gün 030: Şebeke Alarm Kök Neden Analizi (RCA - Root Cause Analysis)
- **İş Alanı:** Şebeke Yönetim Merkezi (NOC)
- **Veri Kaynağı:** [Kaggle - Telco Telemetry Alert Correlation](https://www.kaggle.com/datasets)
- **Model:** Birliktelik Kuralı Madenciliği (FP-Growth / Apriori) + Graf Tabanlı Nedensellik (Causal Discovery)
- **Türkçe Değişkenler:** `alarm_kodu`, `etkilenen_cihaz_id`, `alarm_zaman_damgasi`, `kok_neden_alarm_mi`, `tetiklenen_alt_alarm_sayisi`
- **Jupyter Notebook (`gun_030_sebeke_alarm_kok_neden_analizi.ipynb`):**
  1. Birbirini tetikleyen yüzlerce alt alarmın (Alarm Storm) filtrelenmesi
  2. Zamansal birliktelik analizi ile ana arıza kaynağının izolasyonu
  3. Saha ekiplerine doğrudan kök arıza noktasını bildiren akıllı bilet (ticket) sistemi
- **Mülakat Sorusu:** Bir fiber kopmasında oluşan yüzlerce ikincil alarm arasından kök nedeni saniyeler içinde izole etmek için hangi algoritmalar kullanılır?

---

## 💬 Modül 03: Doğal Dil İşleme (NLP), Müşteri Hizmetleri & LLM (Gün 031 – 045)

### Gün 031: Telekom Müşteri Şikayetleri Duygu Analizi
- **İş Alanı:** Müşteri Deneyimi & Sosyal Medya Dinleme
- **Veri Kaynağı:** [Kaggle - Turkish Sentiment Analysis / Şikayetvar Dataset](https://www.kaggle.com/datasets)
- **Model:** BERTurk (`dbmdz/bert-base-turkish-cased`) / RoBERTa
- **Türkçe Değişkenler:** `sikayet_metni`, `duygu_sinifi_pozitif_notr_negatif`, `guven_skoru`
- **Jupyter Notebook (`gun_031_sikayet_duygu_analizi.ipynb`):**
  1. Türkçe metin ön işleme (Zemberek/NLTK kök bulma, stop-words temizleme)
  2. HuggingFace Transformers ile BERTurk ince ayarı (Fine-tuning)
  3. Confusion Matrix ve F1 değerlendirmesi

### Gün 032: Müşteri Talebi Intent (Niyet) Sınıflandırma
- **İş Alanı:** Turkcell Dijital Asistan (Chatbot)
- **Veri Kaynağı:** [HuggingFace - Banking77 / Turkish Intent](https://huggingface.co/datasets/banking77)
- **Model:** SetFit (Few-Shot Text Classification) / DistilBERTurk
- **Türkçe Değişkenler:** `kullanici_cumlesi`, `tespit_edilen_niyet`, `niyet_olasiligi`
- **Jupyter Notebook (`gun_032_chatbot_niyet_siniflandirma.ipynb`)**

### Gün 033: Twitter Destek Taleplerini Otomatik Departmana Yönlendirme
- **İş Alanı:** @TurkcellHizmet Sosyal Medya Masası
- **Veri Kaynağı:** [Kaggle - Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **Model:** TF-IDF + LinearSVC / FastText
- **Türkçe Değişkenler:** `tweet_icerigi`, `ilgili_departman_fatura_sebeke_cihaz`, `atanan_oncelik`
- **Jupyter Notebook (`gun_033_sosyal_medya_yonlendirme.ipynb`)**

### Gün 034: Çağrı Metinlerinden Varlık İsmi Çıkarımı (NER - Named Entity Recognition)
- **İş Alanı:** Müşteri Deneyimi & KVKK / PII Maskeleme Masası
- **Veri Kaynağı:** [HuggingFace - wikiann / tr (Turkish NER)](https://huggingface.co/datasets/wikiann)
- **Model:** BERTurk-NER (`dbmdz/bert-base-turkish-cased-ner`) / TokenClassification
- **Türkçe Değişkenler:** `cagri_transkripti`, `bulunan_varlik_etiketi`, `kisi_kurum_lokasyon_turu`, `maskelenmis_metin`
- **Jupyter Notebook (`gun_034_cagri_metinleri_ner.ipynb`):**
  1. Ses transkripti metinlerinin BIO (Begin-Inside-Outside) tokenizasyonu
  2. BERTurk ile ad, soyad, şehir, telefon ve TC kimlik no gibi varlıkların (Entity) tespiti
  3. KVKK uyumu için hassas kişisel verilerin (PII) otomatik maskelenmesi ve anonimizasyon pipeline'ı
- **Mülakat Sorusu:** Türkçe gibi eklemeli (agglutinative) dillerde Token-Level NER yaparken subword tokenizasyonunda BIO etiketleri nasıl hizalanır?

### Gün 035: Şirket İçi Dokümanlar için RAG (Retrieval-Augmented Generation) Asistanı
- **İş Alanı:** Turkcell Akademi & Şirket İçi Bilgi Yönetimi
- **Veri Kaynağı:** [HuggingFace - BilgiQA / Turkish Telecom FAQs](https://huggingface.co/datasets)
- **Model:** LangChain + ChromaDB + `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` + Llama-3-8B-Instruct
- **Türkçe Değişkenler:** `kullanici_sorusu`, `getirilen_dokuman_parcalari`, `vektor_benzerlik_skoru`, `uretilen_yanit`
- **Jupyter Notebook (`gun_035_dokuman_rag_asistani.ipynb`):**
  1. PDF/Markdown telekom politika dokümanlarının recursive character splitter ile parçalanması (Chunking)
  2. ChromaDB vektör veritabanında embedding indeksleme
  3. Semantik arama (Cosine Similarity) ve LLM bağlam enjeksiyonu ile halüsinasyonsuz yanıt üretimi
- **Mülakat Sorusu:** RAG mimarisinde "Lost in the Middle" problemi nedir ve context reranking (Cohere Rerank / Cross-Encoder) ile nasıl çözülür?

### Gün 036: Kullanıcı Yorumları Konu Modellemesi (Topic Modeling)
- **İş Alanı:** Ürün Yönetimi (fizy, TV+, BiP, Paycell App Store Yorumları)
- **Veri Kaynağı:** [Kaggle - Google Play Store Turkcell Apps Reviews](https://www.kaggle.com/datasets)
- **Model:** BERTopic + UMAP Boyut İndirgeme + HDBSCAN Kümeleme
- **Türkçe Değişkenler:** `magaza_yorumu`, `atanan_konu_id`, `konu_anahtar_kelimeleri`, `kullanici_yildiz_puani`
- **Jupyter Notebook (`gun_036_uygulama_yorumlari_bertopic.ipynb`):**
  1. App Store ve Google Play yorumlarının c-TF-IDF ile ağırlıklandırılması
  2. BERTopic ile dinamik konu kümelerinin (örn: "Fatura İtirazı", "Giriş Hatası", "Yavaşlama") çıkarımı
  3. Zaman içindeki konu popülarite trendlerinin görselleştirilmesi
- **Mülakat Sorusu:** Klasik LDA (Latent Dirichlet Allocation) yerine transformer tabanlı BERTopic tercih edilmesinin temel avantajları nelerdir?

### Gün 037: SMS Oltalama (Phishing) ve Sahte Kampanya Filtresi
- **İş Alanı:** Bilgi Güvenliği & Siber Savunma Masası
- **Veri Kaynağı:** [Kaggle - SMS Spam Collection / Turkish Smishing Dataset](https://www.kaggle.com/datasets)
- **Model:** TF-IDF + Multinomial Naive Bayes / RoBERTa Sequence Classification
- **Türkçe Değişkenler:** `sms_govde_metni`, `icerdigi_url_sayisi`, `tehdit_turu_oltalama_normal`, `guvenlik_skoru`
- **Jupyter Notebook (`gun_037_sms_oltalama_filtresi.ipynb`):**
  1. SMS metinlerindeki aciliyet tetikleyicileri ("Hemen tıkla", "Hattınız kapanacak") ve sahte link çıkarımı
  2. Karakter seviyesi N-Gram ve kelime vektörleriyle spam/phishing sınıflandırma
  3. Şebeke SMS Gateway üzerinde gecikmesiz (<5ms) kural ve model inference entegrasyonu
- **Mülakat Sorusu:** Oltalama SMS'i tespitinde yüksek Recall mı yoksa yüksek Precision mı hedeflenir? Müşterinin normal SMS'inin engellenmesi işi nasıl etkiler?

### Gün 038: Müşteri Temsilcisi Çağrı Özeti Çıkarıcı (Abstractive Summarization)
- **İş Alanı:** 532 Çağrı Merkezi Operasyonel Verimlilik
- **Veri Kaynağı:** [HuggingFace - Turkish Text Summarization / TR-News / DialogSum](https://huggingface.co/datasets)
- **Model:** mT5 (`google/mt5-base`) / Turkish-BART İnce Ayar (Fine-Tuning)
- **Türkçe Değişkenler:** `uzun_cagri_diyalogu`, `temsilci_aksiyonu`, `uretilen_kisa_ozet`, `rouge_skoru`
- **Jupyter Notebook (`gun_038_cagri_ozeti_mt5.ipynb`):**
  1. Müşteri ve temsilci arasındaki çok turlu konuşma diyaloğu formatlaması
  2. Seq2Seq mT5 modelinin ROUGE-1, ROUGE-2 ve ROUGE-L metrikleriyle eğitimi
  3. Çağrı sonrasında CRM sistemine otomatik 2 cümlelik özet ve aksiyon kartı basılması
- **Mülakat Sorusu:** Metin özetlemede Extractive (Çıkarımsal) ve Abstractive (Soyutlamalı) yaklaşımlar arasındaki fark nedir? Çağrı merkezi için hangisi uygundur?

### Gün 039: Sosyal Medya Marka Kriz Dedektörü (Sentiment Volatility & Anomaly)
- **İş Alanı:** Kurumsal İletişim & Sosyal Medya Kriz Yönetimi
- **Veri Kaynağı:** [Kaggle - Twitter Brand Sentiment Stream](https://www.kaggle.com/datasets)
- **Model:** Exponential Moving Average (EMA) + Z-Score Volatilite Anomali Dedektörü
- **Türkçe Değişkenler:** `saatlik_olumsuz_tweet_orani`, `hareketli_ortalama_z_skoru`, `kriz_alarmi_seviyesi`, `trend_hashtagler`
- **Jupyter Notebook (`gun_039_sosyal_medya_kriz_dedektoru.ipynb`):**
  1. Canlı Twitter/X akışından saatlik duygu skorlarının (Sentiment Score) hesaplanması
  2. Standart sapma dışına çıkan ani negatif duygu patlamalarının tespiti
  3. Kriz anında öne çıkan anahtar kelimelerin anlık kelime bulutu (WordCloud) analizi
- **Mülakat Sorusu:** Zaman serisinde mevsimsel duygu dalgalanmalarını (gece/gündüz farkı) gerçek bir kriz patlamasından ayırt etmek için hangi istatistiksel filtreler kullanılır?

### Gün 040: SSS (FAQ) Semantik Soru Eşleştirme Motoru
- **İş Alanı:** Turkcell Web & Dijital Operatör Arama Motoru
- **Veri Kaynağı:** [HuggingFace - Turkish Semantic Similarity / STS-tr](https://huggingface.co/datasets)
- **Model:** Sentence-Transformers (`sentence-transformers/all-MiniLM-L6-v2`) + Cosine Similarity
- **Türkçe Değişkenler:** `kullanici_arama_ifadesi`, `veritabani_sss_sorusu`, `semantik_benzerlik_orani`, `onerilen_cevap_id`
- **Jupyter Notebook (`gun_040_sss_semantik_eslestirme.ipynb`):**
  1. Farklı yazılmış ancak aynı anlama gelen soruların (örn: "Faturamı nasıl öderim?" vs "Borç yatırma kanalları") vektörleştirilmesi
  2. Siameze Sinir Ağları ve Cosine Similarity ile en yakın SSS eşleşmesinin bulunması
  3. Arama kutusunda anlık otomatik tamamlama ve cevap kartı getirme pipeline'ı
- **Mülakat Sorusu:** Semantik aramada Cross-Encoder ile Bi-Encoder arasındaki performans ve gecikme (latency) ödünleşimi (trade-off) nedir?

### Gün 041: Çok Dilli Destek Talebi Ayrıştırma ve Tercüme
- **İş Alanı:** Turkcell Global Bilgi & Turist/Yabancı Müşteri Destek Masası
- **Veri Kaynağı:** [HuggingFace - Opus-100 Multilingual Parallel Dataset](https://huggingface.co/datasets/opus100)
- **Model:** FastText Dil Tanıma (`lid.176.bin`) + MarianMT (`Helsinki-NLP/opus-mt-en-tr`, `ar-tr`)
- **Türkçe Değişkenler:** `gelen_mesaj_metni`, `tespit_edilen_dil_kodu`, `turkce_tercume_metni`, `guvenilirlik_skoru`
- **Jupyter Notebook (`gun_041_cok_dilli_destek_tercume.ipynb`):**
  1. Gelen mesajın dilinin (İngilizce, Arapça, Rusça, Almanca vb.) milisaniyeler içinde tespiti
  2. Nöral Makine Çevirisi (NMT) ile müşteri mesajının temsilci ekranına Türkçe çevrilmesi
  3. Temsilcinin Türkçe yanıtının anında hedef dile geri çevrilmesi (Bidirectional Pipeline)
- **Mülakat Sorusu:** Dil tespiti (Language Identification) modelleri kısa metinlerde neden zorlanır ve hibrit kurallarla doğruluk nasıl artırılır?

### Gün 042: Toksik ve Hakaret İçeren Yorum Moderasyonu
- **İş Alanı:** BiP Kanalları & Topluluk İletişim Moderasyonu
- **Veri Kaynağı:** [Kaggle - Turkish Toxic / Offensive Language Dataset](https://www.kaggle.com/datasets)
- **Model:** BERTurk Text Classification (`dbmdz/bert-base-turkish-cased`) + Multi-Label BCEWithLogitsLoss
- **Türkçe Değişkenler:** `mesaj_icerigi`, `hakaret_olasiligi`, `tehdit_olasiligi`, `otomatik_engellendi_mi`
- **Jupyter Notebook (`gun_042_toksik_yorum_moderasyonu.ipynb`):**
  1. Argo, hakaret, nefret söylemi ve tehdit içeren çok etiketli veri temizliği
  2. Dengesiz veri dağılımında Focal Loss / Class Weights ile model optimizasyonu
  3. Gerçek zamanlı sohbet akışlarında küfür ve toksisite filtreleme API'si
- **Mülakat Sorusu:** Multi-label metin sınıflandırmada Binary Cross Entropy ile Categorical Cross Entropy arasındaki fark nedir?

### Gün 043: PDF Abonelik Sözleşmesi Madde ve Taahhüt Çıkarımı
- **İş Alanı:** Hukuk & Kurumsal Satış Sözleşme Otomasyonu
- **Veri Kaynağı:** [HuggingFace - Contract Understanding / CUAD Dataset Adapted to TR](https://huggingface.co/datasets)
- **Model:** LayoutLMv3 / PDFplumber + Regex + Turkish Question Answering BERT
- **Türkçe Değişkenler:** `sozlesme_pdf_yolu`, `taahhut_suresi_ay`, `cayma_bedeli_tutari`, `tespit_edilen_madde_metni`
- **Jupyter Notebook (`gun_043_sozlesme_madde_cikarimi.ipynb`):**
  1. OCR ve PDF parser ile kurumsal abonelik sözleşmelerinin dijitalleştirilmesi
  2. LayoutLM ve QA modeliyle "Taahhüt Süresi", "Ceza Bedeli", "Yetkili İmza" alanlarının tespiti
3. Yapısal JSON çıktısı üreterek ERP/CRM sistemine otomatik kontrat veri aktarımı
- **Mülakat Sorusu:** Doküman AI modellerinde (LayoutLM) sadece metin yerine görsel yerleşim (bounding box) koordinatlarının kullanılmasının önemi nedir?

### Gün 044: Müşteri Temsilcisi Yanıt Kalitesi Skorlama (LLM-as-a-Judge)
- **İş Alanı:** Kalite Güvence (QA) & Müşteri Deneyimi Denetimi
- **Veri Kaynağı:** [HuggingFace - Turkish Customer Service Multi-Turn Conversations](https://huggingface.co/datasets)
- **Model:** LLM-as-a-Judge (Yerel Qwen2.5-7B-Instruct / LLaMA-3.2-3B (Ollama / HuggingFace Transformers) + G-Eval Prompting)
- **Türkçe Değişkenler:** `temsilci_cevabi`, `nezaket_puani_1_5`, `dogruluk_puani_1_5`, `cozum_odaklilik_puani`, `denetim_gerekcesi`
- **Jupyter Notebook (`gun_044_temsilci_kalite_skorlama_llm.ipynb`):**
  1. Temsilci yanıtlarının "Nezaket", "Kurumsal Bilgi Doğruluğu", "Çözüm Hızı" rubriklerine göre puanlanması
  2. Chain-of-Thought (CoT) prompting ile LLM hakemlik değerlendirmesi
  3. İnsan denetçi puanları ile LLM puanları arasındaki Pearson/Spearman korelasyon analizi
- **Mülakat Sorusu:** "LLM-as-a-Judge" yaklaşımında karşılaşılan pozisyonel önyargı (Position Bias) ve uzunluk önyargısı (Verbosity Bias) nasıl engellenir?

### Gün 045: IVR (Sesli Yanıt) Menü Yönlendirme Niyet Modeli
- **İş Alanı:** 532 Sesli Yanıt Sistemi (IVR) Otomasyonu
- **Veri Kaynağı:** [Kaggle - Conversational Intent / Spoken Dialog Dataset](https://www.kaggle.com/datasets)
- **Model:** ConvBERT / Bi-LSTM + Attention Mekanizması
- **Türkçe Değişkenler:** `sesli_komut_metni`, `ana_menu_hedefi_fatura_tarife_puk`, `alt_aksiyon_kodu`, `yonlendirme_guveni`
- **Jupyter Notebook (`gun_045_ivr_sesli_yanit_niyet.ipynb`):**
  1. ASR (Otomatik Konuşma Tanıma) çıktısı olan gürültülü metinlerin normalizasyonu
  2. Hiyerarşik sınıflandırma ile önce Ana Menü, ardından Alt Menü tespiti
  3. Güven skoru %80'in altında kaldığında teyit sorusu soran karar mekanizması
- **Mülakat Sorusu:** Hiyerarşik Niyet Sınıflandırmasında (Hierarchical Intent Classification) Flat Multiclass modele göre ne gibi mimari avantajlar elde edilir?

## 👁️ Modül 04: Bilgisayarlı Görü (Computer Vision) & Saha Denetimi (Gün 046 – 060)

### Gün 046: Baz İstasyonu Kule & Anten Nesne Tespiti
- **İş Alanı:** Saha Operasyonları & Altyapı Denetimi
- **Veri Kaynağı:** [Roboflow Universe - Telecom Tower Antenna Detection](https://universe.roboflow.com/)
- **Model:** YOLOv8n / YOLOv11 Object Detection
- **Türkçe Değişkenler:** `kule_goruntusu`, `tespit_edilen_anten_sayisi`, `sinirlayici_kutu_koordinatlari`, `guven_orani`
- **Jupyter Notebook (`gun_046_kule_anten_tespiti.ipynb`):**
  1. Roboflow üzerinden veri seti indirme ve YAML konfigürasyonu
  2. YOLOv8 ile kule, sektör anteni ve mikrodalga çanak nesneleri üzerinde eğitim
  3. mAP@0.5 ve mAP@0.5:0.95 metrik değerlendirmesi ve test görüntülerinde inference
- **Mülakat Sorusu:** Farklı hava koşullarında (sis, yoğun güneş, kar) drone fotoğraflarından küçük antenleri tespit ederken mAP düşüşünü önlemek için hangi veri artırma (Augmentation) yöntemleri kullanılır?

### Gün 047: Kimlik Kartı & Pasaport Köşe Tespiti ve Segmentasyonu
- **İş Alanı:** Dijital Hat Açılış Süreci (e-KYC)
- **Veri Kaynağı:** [Roboflow - ID Card & Passport Segmentation](https://universe.roboflow.com/)
- **Model:** YOLOv8-Seg / OpenCV Perspective Warp (Dört Köşe Homografi)
- **Türkçe Değişkenler:** `ham_kimlik_fotografi`, `kose_noktalari`, `perspektif_duzeltilmis_kimlik`, `parlama_orani`
- **Jupyter Notebook (`gun_047_kimlik_segmentasyon_kyc.ipynb`):**
  1. Kullanıcının cep telefonuyla açılı çektiği kimlik fotoğrafının köşe segmentasyonu
  2. 4 köşe koordinatı üzerinden OpenCV `getPerspectiveTransform` ve `warpPerspective` ile kuşbakışı düzeltme
  3. Parlama ve yansımaları filtreleyerek OCR öncesi netleştirme pipeline'ı
- **Mülakat Sorusu:** Kimlik kartının perspektif dönüşümünde homografi matrisi $H$ kaç serbestlik derecesine sahiptir ve en az kaç köşe noktası gereklidir?

### Gün 048: Fatura / Fiş Bounding Box OCR Tespiti
- **İş Alanı:** Paycell Fatura Ödeme & Masraf Yönetimi
- **Veri Kaynağı:** [Roboflow - Invoice & Receipt OCR Key Information Extraction](https://universe.roboflow.com/)
- **Model:** PaddleOCR / CRAFT Text Detector + LayoutLM
- **Türkçe Değişkenler:** `fatura_gorseli`, `kurum_adi_kutusu`, `odenecek_tutar_kutusu`, `son_odeme_tarihi_kutusu`
- **Jupyter Notebook (`gun_048_fatura_ocr_bilgi_cikarimi.ipynb`):**
  1. Karmaşık fatura ve makbuz görsellerinde metin bloklarının kutulanması (Bounding Box)
  2. Türkçe karakter destekli PaddleOCR ile fatura tutarı, abone no ve son ödeme tarihinin ayrıştırılması
  3. Yapısal JSON çıktısı üreterek Paycell tek tıkla fatura ödeme API'sine aktarım
- **Mülakat Sorusu:** OCR metin tespitinde CRAFT (Character Region Awareness for Text Detection) algoritmasının klasik kenar buluculara göre üstünlüğü nedir?

### Gün 049: Saha Ekibi İş Güvenliği (Baret / Yelek) Denetimi
- **İş Alanı:** Saha İSG (İş Sağlığı ve Güvenliği) Otomasyonu
- **Veri Kaynağı:** [Roboflow - Construction & Worker PPE Safety Dataset](https://universe.roboflow.com/)
- **Model:** YOLOv8x + Custom Safety Compliance Rules
- **Türkçe Değişkenler:** `saha_kamera_karesi`, `baret_takili_mi`, `reflektorlu_yelek_var_mi`, `isg_ihlal_alarmi`
- **Jupyter Notebook (`gun_049_is_guvenligi_baret_yelek.ipynb`):**
  1. Baz istasyonu montaj ve kule tırmanışlarında işçi, baret ve reflektörlü yelek tespiti
  2. Bounding box kesişimi (IoU) ile bareti takan kişinin eşleştirilmesi
  3. İhlal durumunda anlık alarm ve kule tırmanış durdurma bildirim mekanizması
- **Mülakat Sorusu:** Baret ile işçi gövdesini doğru eşleştirmek için iki bounding box arasındaki spatial overlap (IoU) ilişkisi nasıl kurgulanır?

### Gün 050: Veri Merkezi Sunucu Kablo Hasar & Karmaşa Tespiti
- **İş Alanı:** Turkcell Veri Merkezi Kablolama & Rack Denetimi
- **Veri Kaynağı:** [Roboflow - Server Rack Cable Management & Defect Dataset](https://universe.roboflow.com/)
- **Model:** Mask R-CNN / YOLOv8-Seg Instance Segmentation
- **Türkçe Değişkenler:** `rack_sunucu_gorseli`, `ezilmis_kablo_segmenti`, `kablo_duzen_puani_1_100`, `hava_akisi_engeli_var_mi`
- **Jupyter Notebook (`gun_050_sunucu_kablo_hasar_tespiti.ipynb`):**
  1. Sunucu kabinlerindeki fiber ve ethernet kablolarının piksel düzeyinde segmentasyonu
  2. Bükülmüş, ezilmiş veya hava akışını engelleyen karmaşık kablo demetlerinin sınıflandırılması
  3. Kabin düzen skoru ve revizyon gereken portların işaretlenmesi
- **Mülakat Sorusu:** Semantik Segmentasyon (FCN/U-Net) ile Instance Segmentasyon (Mask R-CNN) arasındaki temel fark kablo ayrıştırmada neden kritiktir?

### Gün 051: Turkcell Bayi İçi Müşteri Sayma & Yoğunluk Isı Haritası
- **İş Alanı:** Perakende Mağazacılık & Bayi Kanalı Analitiği
- **Veri Kaynağı:** [Roboflow - Retail Store Customer Tracking & Density](https://universe.roboflow.com/)
- **Model:** YOLOv8-Pose / ByteTRACK + Kernel Density Estimation (KDE) Isı Haritası
- **Türkçe Değişkenler:** `magaza_kamera_akisi`, `anlik_musteri_sayisi`, `reyon_kalma_suresi_sn`, `yogunluk_isi_haritasi`
- **Jupyter Notebook (`gun_051_bayi_musteri_sayma_isi_haritasi.ipynb`):**
  1. Giriş/çıkış sanal çizgileri üzerinden geçen müşterilerin yönlü sayımı (In/Out Counter)
  2. ByteTRACK ile müşteri izleme (Tracking) ve telefon/aksesuar stantlarında geçirilen sürenin ölçülmesi
  3. Mağaza yerleşim planı üzerine yoğunluk ısı haritası (Heatmap) bindirme
- **Mülakat Sorusu:** Çoklu kamera veya kalabalık sahnelerde müşteri takibinde ID Switch (kimlik karışması) problemi ByteTRACK ile nasıl önlenir?

### Gün 052: Baz İstasyonu Çevresi Yangın & Duman Erken Uyarısı
- **İş Alanı:** Kırsal Altyapı Güvenliği & Afet Yönetimi
- **Veri Kaynağı:** [Roboflow - Wildfire Smoke & Flame Detection](https://universe.roboflow.com/)
- **Model:** YOLOv8 Small + Temporal Smoothing Filter (Yanlış Alarm Engelleyici)
- **Türkçe Değişkenler:** `kamera_goruntusu`, `duman_olasiligi`, `alev_olasiligi`, `yangin_alarmi_tetiklendi_mi`
- **Jupyter Notebook (`gun_052_baz_istasyonu_yangin_duman.ipynb`):**
  1. Ormanlık alanlardaki kule kameralarından duman bulutu ve alev tespiti
  2. Bulut, toz veya sis kaynaklı sahte pozitifleri elemek için ardışık 5 karelik zaman filtresi
  3. İtfaiye ve kriz masasına GPS koordinatlı acil MMS/BiP bildirimi
- **Mülakat Sorusu:** Duman gibi sınırları belirsiz ve amorf nesnelerin tespitinde bounding box regülarizasyonu nasıl optimize edilir?

### Gün 053: SIM Kart Barkod & ICCID Seri Numarası Konumlandırma
- **İş Alanı:** Lojistik & SIM Kart Paketleme Kalite Kontrolü
- **Veri Kaynağı:** [Roboflow - Barcode & QR Code Localization](https://universe.roboflow.com/)
- **Model:** YOLOv8-Nano + PyZbar / OpenCV QR Detector
- **Türkçe Değişkenler:** `sim_kart_kutusu`, `barkod_alani_koordinatlari`, `okunan_iccid_seri_no`, `kod_okunabilir_mi`
- **Jupyter Notebook (`gun_053_sim_kart_barkod_iccid.ipynb`):**
  1. Hareketli konveyör banttaki SIM kartlar üzerinde barkod ve QR kodların bulunması
  2. Bounding box kırpılarak görüntü netleştirme ve ICCID numarasının okunması
  3. Bozuk, çizik veya eksik basılmış SIM kartların otomatik reddedilmesi
- **Mülakat Sorusu:** Düşük çözünürlüklü veya hareket bulanıklığı (Motion Blur) olan görüntülerde barkod okuma başarısı nasıl artırılır?

### Gün 054: Güneş Enerjili İstasyonlarda Panel Kirlilik/Kırık Tespiti
- **İş Alanı:** Yeşil Şebeke & Saha Yenilenebilir Enerji Bakımı
- **Veri Kaynağı:** [Roboflow - Solar Panel Defect & Dust Detection](https://universe.roboflow.com/)
- **Model:** YOLOv8 Classification & Segmentation / EfficientNet-B4
- **Türkçe Değişkenler:** `panel_termal_goruntusu`, `kirlilik_orani_yuzde`, `sicak_nokta_hotspot_var_mi`, `temizlik_bakim_onerisi`
- **Jupyter Notebook (`gun_054_gunes_paneli_kirlilik_tespiti.ipynb`):**
  1. Drone ile çekilen termal ve RGB panel fotoğraflarının analizi
  2. Kuş pisliği, tozlanma veya hücre kırığı (Hotspot) kaynaklı verim kayıplarının tespiti
  3. Saha ekiplerine önleyici temizlik ve panel değişim görev emri oluşturulması
- **Mülakat Sorusu:** Güneş panellerindeki mikro çatlakları tespit etmede RGB kamera ile Termal (FLIR) kamera füzyonunun katkısı nedir?

### Gün 055: Saha Araçları Otomatik Plaka Tanıma (ANPR)
- **İş Alanı:** Turkcell Plaza & Saha Filo Giriş-Çıkış Yönetimi
- **Veri Kaynağı:** [Roboflow - Turkish License Plate Detection & Character OCR](https://universe.roboflow.com/)
- **Model:** YOLOv8 (Plaka Tespiti) + CRNN / Tesseract (Karakter Tanıma)
- **Türkçe Değişkenler:** `arac_on_goruntusu`, `plaka_alani`, `okunan_plaka_metni`, `filo_yetkili_arac_mi`
- **Jupyter Notebook (`gun_055_otomatik_plaka_tanima_anpr.ipynb`):**
  1. Hareket halindeki araçlardan Türk formatına uygun (34 ABC 123) plaka tespiti
  2. Plaka bölgesinin kırpılması, gri tonlama ve adaptif eşikleme (Otsu Thresholding)
  3. Karakter dizisi tanıma ve bariyer otomatik açılış lojiği
- **Mülakat Sorusu:** ANPR sistemlerinde iki aşamalı (Two-Stage: Detection + Recognition) mimari neden End-to-End OCR modellerine göre pratikte daha stabildir?

### Gün 056: Taranan Sözleşmelerde Islak İmza Eksikliği Denetimi
- **İş Alanı:** Müşteri Kabul & Sözleşme Arşiv Denetimi
- **Veri Kaynağı:** [Roboflow - Signature Area & Stamp Detection on Documents](https://universe.roboflow.com/)
- **Model:** Faster R-CNN / YOLOv8 Object Detection
- **Türkçe Değişkenler:** `taranmis_sozlesme_sayfasi`, `imza_kutusu_koordinati`, `imza_mevcut_mu`, `sahte_fotokopi_suphesi`
- **Jupyter Notebook (`gun_056_sozlesme_imza_denetimi.ipynb`):**
  1. PDF sözleşme sayfalarının yüksek çözünürlüklü imajlara dönüştürülmesi
  2. "Müşteri İmzası" ve "Bayi Kaşesi" alanlarının tespiti
  3. Boş bırakılan veya fotokopiyle çoğaltılmış imzasız evrakların otomatik bayiye iade edilmesi
- **Mülakat Sorusu:** Dokümanlarda ıslak mürekkepli imza ile dijital yapıştırılmış imza arasındaki doku (texture) farkı ML ile nasıl ayrıştırılır?

### Gün 057: Antenlerde Kuş Yuvası ve Engel Tespiti
- **İş Alanı:** Radyo Şebeke Kule Bakımı & Sinyal Engeli Önleme
- **Veri Kaynağı:** [Roboflow - Bird Nest & Transmission Line Hazard Detection](https://universe.roboflow.com/)
- **Model:** YOLOv8 Object Detection + Sinyal Kaybı Korelasyonu
- **Türkçe Değişkenler:** `anten_yakin_cekimi`, `kus_yuvasi_tespit_edildi_mi`, `engel_kapatma_yuzdesi`, `tahmini_db_zayiflama`
- **Jupyter Notebook (`gun_057_anten_kus_yuvasi_engeli.ipynb`):**
  1. Periyodik drone uçuş fotoğraflarında mikrodalga çanak ve panel antenlerin incelenmesi
  2. Kuş yuvası, yabani sarmaşık veya metalik korozyon engellerinin tespiti
  3. Sinyal yayılımını bozan fiziksel engeller için saha ekibi yönlendirmesi
- **Mülakat Sorusu:** Yabancı cisim tespitinde dengesiz ve az sayıda pozitif örnek içeren sınıflar için Synthetic Data Generation (Diffusion/GAN) nasıl kullanılır?

### Gün 058: Mobil Uygulama Arayüz Hata (UI Glitch / Buton Kayması) Tespiti
- **İş Alanı:** Mobil QA (Quality Assurance) & TV+, fizy, Paycell Test Masası
- **Veri Kaynağı:** [Roboflow - Mobile UI Elements & Layout Glitch Dataset](https://universe.roboflow.com/)
- **Model:** YOLOv8 Object Detection + Layout Bounding Overlap Checker
- **Türkçe Değişkenler:** `uygulama_ekran_goruntusu`, `ust_uste_binen_butonlar`, `metin_tasma_durumu`, `ui_hata_skoru`
- **Jupyter Notebook (`gun_058_mobil_ui_glitch_tespiti.ipynb`):**
  1. Farklı ekran çözünürlüklerindeki (iOS/Android tablet, telefon) ekran görüntülerinin taranması
  2. Buton, metin alanı ve görsellerin koordinat tespiti
  3. Üst üste binen (Overlap) veya ekrandan taşan UI hatalarının otomatik CI/CD pipeline'ında yakalanması
- **Mülakat Sorusu:** UI otomasyon testlerinde piksel piksel görsel karşılaştırma (Pixel Diff) yerine neden Nesne Tespiti (Object Detection) tercih edilir?

### Gün 059: Yüz Canlılık (Liveness / Anti-Spoofing) Tespiti
- **İş Alanı:** Paycell & Dijital Kimlik Biyometrik Doğrulama
- **Veri Kaynağı:** [Kaggle - CelebA-Spoof / Face Anti-Spoofing Dataset](https://www.kaggle.com/datasets)
- **Model:** MiniFASNet / FeatherNets + 2D-Fourier Spektrum Analizi
- **Türkçe Değişkenler:** `selfie_videosu_karesi`, `canlilik_skoru_0_1`, `saldiri_turu_ekran_maske_kagit`, `islem_onaylandi_mi`
- **Jupyter Notebook (`gun_059_yuz_canlilik_anti_spoofing.ipynb`):**
  1. Kamera önündeki kişinin gerçek canlı mı yoksa ekrandan gösterilen fotoğraf/video mu olduğunun tespiti
  2. Ekran pikselleri moiré paterni ve derinlik analizi
  3. Dijital onaylarda sahteciliği engelleyen milisaniyelik liveness kontrolü
- **Mülakat Sorusu:** Yüz tanıma sistemlerine yapılan Presentation Attack (baskılı kağıt, tablet ekranı, 3D maske) türleri yazılımsal olarak nasıl engellenir?

### Gün 060: Açık Hava Billboard & Reklam Panosu Doğrulama
- **İş Alanı:** Turkcell Pazarlama & Açık Hava Reklam Denetimi
- **Veri Kaynağı:** [Roboflow - Billboard & Outdoor Advertising Dataset](https://universe.roboflow.com/)
- **Model:** YOLOv8 + SIFT / ORB Feature Matching / CLIP Zero-Shot Classification
- **Türkçe Değişkenler:** `saha_sokak_fotografi`, `billboard_alani`, `reklam_kampanya_eslesme_orani`, `kampanya_dogrulandi_mi`
- **Jupyter Notebook (`gun_060_billboard_reklam_dogrulama.ipynb`):**
  1. Şehir içi araç kameralarından billboard ve otobüs durak reklamlarının tespiti
  2. Tespit edilen panodaki görselin aktif Turkcell reklam afişiyle CLIP / SIFT ile eşleştirilmesi
  3. Reklam ajanslarının afiş asma taahhütlerinin otomatik fatura doğrulaması
- **Mülakat Sorusu:** Değişen açı, ışık ve kısmi gölgelenme altında kurumsal reklam afişini doğrulamak için Feature Matching ile Zero-Shot CLIP nasıl birleştirilir?

---

## 💳 Modül 05: Fintek, Paycell & Fraud / Dolandırıcılık Tespiti (Gün 061 – 075)

### Gün 061: Kredi Kartı Dolandırıcılık Tespiti (Aşırı Dengesiz Veri)
- **İş Alanı:** Paycell Risk İzleme Masası & Sahtekarlık Önleme
- **Veri Kaynağı:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Model:** XGBoost + Focal Loss / Autoencoder Reconstruction Error
- **Türkçe Değişkenler:** `islem_tutari`, `pca_bilesenleri`, `sahte_islem_etiketi`, `rekonstruksiyon_hatasi`
- **Jupyter Notebook (`gun_061_kredi_karti_fraud_tespiti.ipynb`):**
  1. %0.17 sınıf oranına sahip dengesiz verinin incelenmesi ve SMOTE/ADASYN sentetik örnekleme
  2. Autoencoder ile normal işlemlerin öğrenilmesi ve hata eşiği belirlenmesi
  3. Precision-Recall Curve (PR-AUC) optimizasyonu ve maliyet matrisi değerlendirmesi
- **Mülakat Sorusu:** Aşırı dengesiz (%0.1) fraud veri setlerinde ROC-AUC metriği neden yanıltıcıdır ve neden PR-AUC (Average Precision) tercih edilmelidir?

### Gün 062: Mobil Para Transferi Sahtekarlık Modeli
- **İş Alanı:** Paycell P2P (Kişiden Kişiye) Transfer Güvenliği
- **Veri Kaynağı:** [Kaggle - PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Model:** CatBoost Classifier + İzolasyon Ormanı (Isolation Forest) Anomali Skoru
- **Türkçe Değişkenler:** `gonderen_bakiye_oncesi`, `gonderen_bakiye_sonrasi`, `transfer_tutari`, `supheli_islem_etiketi`, `islem_turu_p2p_nakit_odeme`
- **Jupyter Notebook (`gun_062_mobil_transfer_sahtekarlik.ipynb`):**
  1. Bakiyeyi tamamen sıfırlayan ani para boşaltma transferlerinin öznitelik mühendisliği
  2. Alıcı ve gönderici hesap hareketleri hız (velocity) metrikleri
  3. Anlık para transfer onay mekanizmasında milisaniyelik risk skorlama modeli
- **Mülakat Sorusu:** Hesap ele geçirme (Account Takeover - ATO) sonrası yapılan "hesabı boşaltma" işlemlerini tespit etmede Feature Store üzerinde hesaplanan hangi zaman pencereli (Rolling Window) değişkenler en etkilidir?

### Gün 063: Paycell Hazır Limit Kredi Risk Skoru
- **İş Alanı:** Paycell Tüketici Finansmanı & Mikro Kredi Skorlama
- **Veri Kaynağı:** [Kaggle - Home Credit Default Risk](https://www.kaggle.com/datasets/c/home-credit-default-risk)
- **Model:** LightGBM Classifier + Optuna Hiperparametre Optimizasyonu + WoE (Weight of Evidence)
- **Türkçe Değişkenler:** `talep_edilen_limit_tutari`, `gelir_duzeyi`, `gecmis_gecikme_gunu`, `temerrut_riski_skoru`, `kredi_notu_puani`
- **Jupyter Notebook (`gun_063_paycell_hazir_limit_kredi_skoru.ipynb`):**
  1. Başvuru, kredi bürosu geçmişi ve taksit ödeme tablolarının birleştirilmesi
  2. WoE ve Information Value (IV) ile değişken eleme ve filtreleme
  3. Gini katsayısı ve Kolmogorov-Smirnov (KS) istatistiği ile kredi risk modeli validasyonu
- **Mülakat Sorusu:** Kredi risk modellerinde modelin ayrım gücünü ölçmek için kullanılan Kolmogorov-Smirnov (KS) istatistiği nedir ve bankacılıkta ideal KS değeri kaçtır?

### Gün 064: Kara Para Aklama (AML) Şüpheli İşlem Ağ Analizi
- **İş Alanı:** Mevzuat Uyumu (Compliance), MASAK Raporlaması & AML Masası
- **Veri Kaynağı:** [Kaggle / IBM - Synthetic AML Transactions](https://www.kaggle.com/datasets)
- **Model:** Graf Sinir Ağları (GNN - Graph Convolutional Network / Node2Vec) + NetworkX
- **Türkçe Değişkenler:** `gonderen_hesap_id`, `alici_hesap_id`, `graf_derece_merkeziyeti`, `dairesel_transfer_halkasi_var_mi`, `aml_risk_skoru`
- **Jupyter Notebook (`gun_064_kara_para_aklama_aml_graf.ipynb`):**
  1. Finansal transferlerin yönlü ve ağırlıklı graf olarak modellenmesi (Nodes: Hesaplar, Edges: Transferler)
  2. Yapılandırma (Smurfing/Structuring) ve dairesel para dolaştırma (Layering) halkalarının tespiti
  3. Node2Vec düğüm gömmeleri ile şüpheli hesap kümelemesi ve MASAK şüpheli işlem bildirimi
- **Mülakat Sorusu:** Kara para aklamada "Smurfing" (parçalayarak yatırma) paternini klasik tabular ML yerine Graf Sinir Ağları (GNN) ile yakalamanın avantajı nedir?

### Gün 065: Fiziksel Kiosk / Paycell Noktası Nakit Talebi Tahmini
- **İş Alanı:** Fiziksel Ödeme Noktaları & Kiosk Lojistiği
- **Veri Kaynağı:** [Kaggle - ATM Cash Demand Forecasting](https://www.kaggle.com/datasets)
- **Model:** Prophet + SARIMAX + Takvim/Maaş Günü Dışsal Değişkenleri
- **Türkçe Değişkenler:** `kiosk_id`, `tarih_damgasi`, `gunluk_cekilen_nakit_tl`, `maas_gunu_mu`, `tahmini_gerekli_nakit`
- **Jupyter Notebook (`gun_065_kiosk_nakit_talebi_tahmini.ipynb`):**
  1. Kiosk bazlı günlük nakit çekim zaman serisinin mevsimsellik ve trend ayrışımı
  2. Maaş günleri, bayramlar ve resmi tatillerin dışsal regresör (Exogenous Variables) olarak eklenmesi
  3. Kiosklarda nakit bitmesini (Cash-out) önleyen optimum lojistik ikmal planlaması
- **Mülakat Sorusu:** Nakit optimizasyonunda maliyet fonksiyonu asimetriktir (nakit bitmesi cezası > fazla nakit bulundurma faiz kaybı). Bu asimetri modelde nasıl cezalandırılır?

### Gün 066: Mobil Ödeme Hata ve Reddetme Tahminleyici
- **İş Alanı:** Paycell Ödeme Ağ Geçidi (Payment Gateway) & Banka Entegrasyonları
- **Veri Kaynağı:** [Kaggle - Online Payment Failure & Gateway Logs](https://www.kaggle.com/datasets)
- **Model:** Random Forest Classifier + Banka Yanıt Kodu Sınıflandırıcı
- **Türkçe Değişkenler:** `kart_bin_kodu`, `banka_kodu`, `pos_ag_gecidi`, `islem_red_kodu`, `basarisizlik_olasiligi`
- **Jupyter Notebook (`gun_066_odeme_reddetme_tahmini.ipynb`):**
  1. Banka provizyon gecikmeleri ve kart limit yetersizliği loglarının incelenmesi
  2. İşlem anında banka pos arızasını öngörüp alternatif banka sanal POS'una dinamik yönlendirme (Smart Routing)
  3. Ödeme başarı oranının (Authorization Rate) %3-5 artırılması simülasyonu
- **Mülakat Sorusu:** Ödeme orkestrasyonunda (Smart Payment Routing) başarısızlık tahmin modeli ile komisyon maliyet optimizasyonu birlikte nasıl çözülür?

### Gün 067: Üye İşyeri (Merchant) Chargeback Risk Puanlaması
- **İş Alanı:** Paycell Sanal POS Üye İşyeri Risk Yönetimi
- **Veri Kaynağı:** [Kaggle - Merchant Risk & Fraud Chargeback Dataset](https://www.kaggle.com/datasets)
- **Model:** CatBoost Classifier + Bayesian Target Encoding
- **Türkçe Değişkenler:** `uye_isyeri_id`, `sektor_mcc_kodu`, `aylik_ciro`, `ters_ibraz_chargeback_orani`, `isyeri_risk_kategorisi`
- **Jupyter Notebook (`gun_067_uye_isyeri_chargeback_riski.ipynb`):**
  1. Yeni ve mevcut üye işyerlerinin işlem hacmi, ortalama sepet büyüklüğü ve iade oranlarının analizi
  2. Visa/Mastercard kurallarına göre chargeback oranı kritik eşiği (%1) aşabilecek üye işyerlerinin tahmini
  3. Riskli işyerlerine bloke gün sayısı ve teminat tutarı artırma aksiyonlarının belirlenmesi
- **Mülakat Sorusu:** Yüksek riskli MCC kodlarına sahip işyerlerinde "Cold Start" (yeni açılan işyeri) durumunda risk skoru nasıl hesaplanır?

### Gün 068: Alternatif Telekom Verileriyle Kredi Notu Üretme
- **İş Alanı:** Finansal Kapsayıcılık & Bankacılık Geçmişi Olmayan (Unbanked) Kullanıcılar
- **Veri Kaynağı:** [Kaggle - Telco-based Credit Scoring / Financial Inclusion](https://www.kaggle.com/datasets)
- **Model:** Explainable Boosting Machine (EBM) / XGBoost + SHAP Değerleri
- **Türkçe Değişkenler:** `faturali_hat_yasi_ay`, `duzenli_fatura_odeme_skoru`, `aylik_ortalama_paket_tutari`, `alternatif_kredi_skoru_300_900`
- **Jupyter Notebook (`gun_068_telekom_alternatif_kredi_notu.ipynb`):**
  1. Telekom kullanım alışkanlıkları (düzenli fatura ödeme, hat yaşı, mobil ödeme sıklığı) öznitelik çıkarımı
  2. Açıklanabilir Yapay Zeka (XAI) ile BDDK ve KKB standartlarına uygun şeffaf kredi skor kartı üretimi
  3. Kredi kartı olmayan genç ve unbanked kitleye mikro kredi limiti açma simülasyonu
- **Mülakat Sorusu:** Finansal kredi skorlamada "Adversarial Disparate Impact" (etik önyargı ve adalet) analizi neden zorunludur ve nasıl test edilir?

### Gün 069: Çoklu / Sahte Hesap (Sybil Attack) ve Bonus Avcılığı Tespiti
- **İş Alanı:** Paycell Kampanya & Kazan Güvenliği
- **Veri Kaynağı:** [Kaggle - Fraudulent Account Registration & Identity Clustering](https://www.kaggle.com/datasets)
- **Model:** DBSCAN / HDBSCAN Yoğunluk Tabanlı Kümeleme + Device Fingerprinting
- **Türkçe Değişkenler:** `cihaz_parmak_izi_hash`, `ip_alt_agi`, `kayit_zaman_araligi_sn`, `ayni_cihazdaki_hesap_sayisi`, `sahte_kullanici_mi`
- **Jupyter Notebook (`gun_069_sahte_hesap_kampanya_istismari.ipynb`):**
  1. Kampanya bonuslarını (ilk kayda 50 TL vb.) suistimal etmek için aynı cihazdan açılan çoklu hesapların tespiti
  2. IP, MAC, IMEI ve kullanım paterni benzerliği üzerinden yoğunluk kümelemesi
  3. Sahte hesap çiftliklerinin anlık kampanya bloke listesine alınması
- **Mülakat Sorusu:** Cihaz parmak izi (Device Fingerprint) sürekli değişen veya emülatör kullanan gelişmiş bot hesaplar davranışsal biyometri ile nasıl yakalanır?

### Gün 070: Dijital Varlık / Kripto Volatilite ve Likidite Tahmini
- **İş Alanı:** Paycell Kripto & Yatırım Servisleri Masası
- **Veri Kaynağı:** [Kaggle - G-Research Crypto Forecasting Dataset](https://www.kaggle.com/datasets/c/g-research-crypto-forecasting)
- **Model:** Temporal Fusion Transformer (TFT) / LightGBM Regressor
- **Türkçe Değişkenler:** `varlik_kodu_btc_eth`, `emir_defteri_derinligi`, `gerceklesen_volatilite_15dk`, `tahmini_fiyat_getirisi`
- **Jupyter Notebook (`gun_070_kripto_volatilite_tahmini.ipynb`):**
  1. Yüksek frekanslı (High-Frequency) işlem ve emir defteri (Order Book) verisi öznitelik mühendisliği
  2. Alış-satış makası (Bid-Ask Spread) ve volatilite tahmini
  3. Kullanıcı alım-satım emirlerinde kayma (Slippage) maliyetini minimize eden likidite tahmini
- **Mülakat Sorusu:** Finansal zaman serilerinde "GARCH" modelleri ile Derin Öğrenme (TFT/LSTM) modelleri volatilite tahmininde nasıl hibritlenir?

### Gün 071: B2B Kurumsal Bayi Tahsilat Gecikmesi Tahmini
- **İş Alanı:** Turkcell Finans & Kurumsal Alacak Yönetimi
- **Veri Kaynağı:** [Kaggle - B2B Invoice Payment Delay & Default Dataset](https://www.kaggle.com/datasets)
- **Model:** Survival Analysis (Cox Proportional Hazards / Random Survival Forests)
- **Türkçe Değişkenler:** `kurumsal_musteri_id`, `fatura_tutari`, `vade_gun_sayisi`, `gecikme_olasiligi`, `tahmini_tahsilat_gunu`
- **Jupyter Notebook (`gun_071_b2b_tahsilat_gecikmesi.ipynb`):**
  1. B2B kurumsal faturaların vadesinde ödenmeme riskinin Yaşam Analizi (Survival Analysis) ile modellenmesi
  2. Erken nakit iskontosu veya yasal takip öncesi hatırlatma aksiyonlarının tetiklenmesi
  3. Şirket nakit akış tahminine (Cash Flow Forecast) dinamik girdi sağlanması
- **Mülakat Sorusu:** Fatura tahsilat tahmininde klasik regresyon yerine Yaşam Analizi (Survival Analysis) kullanmanın "sağdan sansürlü veri" (Censored Data) açısından avantajı nedir?

### Gün 072: POS Harcama Coğrafi Anomali Dedektörü (Spatial Outliers)
- **İş Alanı:** Paycell Kart Güvenliği & Çalıntı Kart Kullanım Tespiti
- **Veri Kaynağı:** [Kaggle - Spatial Transaction & Geolocation Fraud](https://www.kaggle.com/datasets)
- **Model:** Haversine Hız Hesaplayıcı + Isolation Forest
- **Türkçe Değişkenler:** `kart_id`, `onceki_islem_sehri`, `su_anki_islem_sehri`, `gecen_sure_dakika`, `imkansiz_hiz_kmh`, `anomali_skoru`
- **Jupyter Notebook (`gun_072_pos_cografi_anomali_tespiti.ipynb`):**
  1. İki ardışık kart harcaması arasındaki mesafe (Haversine Distance) ve geçen sürenin oranlanması
  2. "İmkansız Seyahat Hızı" (>900 km/s - örn: 10 dk arayla İstanbul ve Berlin harcaması) tespiti
  3. Şüpheli coğrafi atlamalarda karta anında otomatik SMS onay teyidi düşürülmesi
- **Mülakat Sorusu:** Coğrafi mesafe hesaplarken düzlemsel Euclidean mesafe yerine neden Haversine / Vincenty formülü kullanılmalıdır?

### Gün 073: Otomatik Banka/Paycell Slip Harcama Kategorizasyonu
- **İş Alanı:** Paycell Bütçe Yönetimi & Harcama Analitiği ("Nereye Harcadım?")
- **Veri Kaynağı:** [Kaggle - Bank Transaction Classification & Merchant Tagging](https://www.kaggle.com/datasets)
- **Model:** TF-IDF + RoBERTa / FastText Metin Sınıflandırma
- **Türkçe Değişkenler:** `slip_aciklama_metni`, `harcama_kategorisi_market_benzin_eglence`, `kategori_guven_skoru`
- **Jupyter Notebook (`gun_073_harcama_slip_kategorizasyonu.ipynb`):**
  1. POS slip açıklamalarındaki anlamsız kısaltmaların (örn: "BIM MAG 1234 IST" -> Market) temizlenmesi
  2. Metin sınıflandırma ile işlemlerin 15 ana harcama kategorisine atanması
  3. Kullanıcıya aylık grafiksel harcama özeti ve kişisel bütçe önerileri sunulması
- **Mülakat Sorusu:** Harcama açıklama metinlerindeki gürültülü kısaltmaları çözmek için Regex kuralları ile NLP modelleri nasıl kademeli (Cascade) pipeline oluşturur?

### Gün 074: Sadakat Puanı / Cashback İstismarı Tespiti
- **İş Alanı:** Paycell Sadakat Programı & Hediye Dünyası
- **Veri Kaynağı:** [Kaggle - Loyalty Program Abuse & Synthetic Fraud](https://www.kaggle.com/datasets)
- **Model:** K-Means Kümeleme + Mahalanobis Mesafe Anomali Skoru
- **Türkçe Değişkenler:** `kullanici_id`, `kazanilan_puan_adedi`, `puan_harcama_orani`, `iptal_iade_orani`, `istismar_riski_etiketi`
- **Jupyter Notebook (`gun_074_sadakat_puani_istismari.ipynb`):**
  1. Cashback kazanıp ardından siparişi iade eden veya sahte işlemlerle puan biriktiren kullanıcıların tespiti
  2. Normal kullanıcı harcama dağılımı ile fırsatçı istismarcıların çok boyutlu ayrıştırılması
  3. Haksız kazanılan puanların dondurulması ve promosyon kural motorunun güncellenmesi
- **Mülakat Sorusu:** Çok değişkenli anomali tespitinde Mahalanobis mesafesi, değişkenler arasındaki korelasyonu nasıl hesaba katar?

### Gün 075: SIM Swap Sonrası Finansal İşlem Riski Modeli
- **İş Alanı:** Telekom & Bankacılık Ortak Güvenlik Masası (SIM Swap Fraud)
- **Veri Kaynağı:** [Kaggle - Telecom SIM Swap & Banking Fraud Correlation](https://www.kaggle.com/datasets)
- **Model:** XGBoost Classifier + Zaman Kısıtlı Risk Matrisi
- **Türkçe Değişkenler:** `sim_kart_degisim_saati`, `ilk_finansal_islem_saati`, `gecen_sure_saat`, `cihaz_degisti_mi`, `sim_swap_dolandiricilik_riski`
- **Jupyter Notebook (`gun_075_sim_swap_finansal_risk.ipynb`):**
  1. SIM kartın yedek SIM ile yenilenmesi sonrası ilk 48 saatteki yüksek tutarlı transferlerin incelenmesi
  2. Cihaz IMEI değişimi, şifre sıfırlama talepleri ve havale işlemlerinin ortak skorlanması
  3. Bankalara ve Paycell'e anlık "SIM Swap Alarmı" API entegrasyonu simülasyonu
- **Mülakat Sorusu:** SIM Swap dolandırıcılığında telekom operatörü ile finans kuruluşları arasındaki gerçek zamanlı sinyal paylaşımı (Open Gateway API / CAMARA standardı) nasıl çalışır?

---

## 🎧 Modül 06: Ses İşleme & Çağrı Analitiği (Audio AI) (Gün 076 – 085)

### Gün 076: Türkçe Konuşma Tanıma (ASR - Automatic Speech Recognition)
- **İş Alanı:** 532 Çağrı Merkezi Ses Kayıt Transkripsiyonu
- **Veri Kaynağı:** [HuggingFace - Mozilla Common Voice Turkish](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
- **Model:** Açık Kaynak Yerel Whisper-Small (HuggingFace / CPU-GPU) / Wav2Vec2-XLSR-Turkish + CTC Loss
- **Türkçe Değişkenler:** `ses_dosyasi_yolu`, `ornekleme_frekansi_hz`, `metne_dokulen_transkript`, `kelime_hata_orani_wer`, `karakter_hata_orani_cer`
- **Jupyter Notebook (`gun_076_turkce_asr_whisper.ipynb`):**
  1. MP3/WAV ses dosyalarının 16 kHz mono formatına dönüştürülmesi ve Log-Mel spektrogram çıkarımı
  2. Whisper-Small modelinin Türkçe telekom çağrı verisiyle LoRA / Fine-Tuning eğitimi
  3. Word Error Rate (WER) ve Character Error Rate (CER) metrikleri ile model başarımı
- **Mülakat Sorusu:** Gürültülü telekom çağrılarında (8 kHz bant genişliği) ASR modellerinde WER değerini düşürmek için hangi ses önişleme ve Spectral Augmentation (SpecAugment) yöntemleri uygulanır?

### Gün 077: Çağrıda Müşteri Sesinden Duygu & Stres Tespiti (Speech Emotion Recognition)
- **İş Alanı:** Müşteri Deneyimi & Öfkeli Müşteri Erken Uyarı Masası
- **Veri Kaynağı:** [Kaggle - RAVDESS & CREMA-D Audio Dataset](https://www.kaggle.com/datasets)
- **Model:** 1D-CNN + Bi-LSTM / HuBERT + Mel-Frequency Cepstral Coefficients (MFCC)
- **Türkçe Değişkenler:** `ses_dalga_boyu`, `mfcc_katsayilari`, `ses_tonu_perdesi_pitch`, `ofke_stres_seviyesi_0_100`
- **Jupyter Notebook (`gun_077_ses_duygu_stres_analizi.ipynb`):**
  1. Ses sinyalinden MFCC, Chroma, Spectral Contrast ve Pitch (F0) özniteliklerinin çıkarılması
  2. Öfkeli, stresli, sakin ve mutlu duygu durumlarının sınıflandırılması
  3. Çağrı esnasında müşteri öfkesi %80'i aştığında amir (supervisor) ekranına canlı uyarı düşürülmesi
- **Mülakat Sorusu:** Metin tabanlı duygu analizi ile ses dalgası tabanlı duygu analizi (Multimodal SER) arasındaki fark nedir ve ses tonu neden daha dürüst bir duygu sinyalidir?

### Gün 078: Ses Biyometrisi ile Müşteri Kimlik Doğrulama (Speaker Verification)
- **İş Alanı:** 532 Sesli İmza & Şifresiz Kimlik Doğrulama
- **Veri Kaynağı:** [HuggingFace - VoxCeleb1 / VoxCeleb2 Speaker Recognition](https://huggingface.co/datasets)
- **Model:** ECAPA-TDNN / ResNetSE34V2 + Angular Additive Margin Softmax (ArcFace Loss)
- **Türkçe Değişkenler:** `kayitli_ses_vektoru_embedding`, `gelen_cagri_sesi`, `kosinus_benzerlik_skoru`, `kimlik_dogrulandi_mi`
- **Jupyter Notebook (`gun_078_ses_biyometrisi_dogrulama.ipynb`):**
  1. Müşterinin "Turkcell beni sesimden tanır" ses örneğinden 192 boyutlu d-vector / x-vector çıkarımı
  2. Gelen çağrıdaki ses gömmesi ile kayıtlı gömme arasındaki Cosine Similarity karşılaştırması
  3. EER (Equal Error Rate) eğrisi analizi ile güvenlik eşiği optimizasyonu
- **Mülakat Sorusu:** Ses biyometrisinde Text-Dependent (sabit cümle) ile Text-Independent (serbest konuşma) sistemler arasındaki fark ve güvenlik ödünleşimi nedir?

### Gün 079: Ağ Arka Plan Gürültüsü Sınıflandırma ve Gürültü Engelleme
- **İş Alanı:** Çağrı Kalitesi İyileştirme & Ortam Gürültüsü Filtreleme
- **Veri Kaynağı:** [Kaggle - UrbanSound8K Dataset](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
- **Model:** U-Net tabanlı Spektrogram Maskeleme / Deep Complex UNet (DCUNet)
- **Türkçe Değişkenler:** `gurultulu_ses_girisi`, `ortam_turu_trafik_kafe_siren_ruzgar`, `temizlenmis_ses_ciktisi`, `snr_iyilesme_orani_db`
- **Jupyter Notebook (`gun_079_arka_plan_gurultu_filtreleme.ipynb`):**
  1. Kısa Zamanlı Fourier Dönüşümü (STFT) ile sesin frekans-zaman düzlemine aktarılması
  2. Arka plandaki trafik, korna, bebek ağlaması, rüzgar gibi gürültülerin sınıflandırılması
  3. U-Net ile gürültü maskesi tahmin edilerek müşteri sesinin netleştirilmesi (Signal-to-Noise Ratio artırımı)
- **Mülakat Sorusu:** Spektral maskelemede Ideal Ratio Mask (IRM) ile Complex Ratio Mask (cRM) arasındaki faz (phase) koruma farkı nedir?

### Gün 080: VoLTE/VoIP Hatlarında Ses Kalitesi MOS Puanı Tahmini
- **İş Alanı:** Şebeke Ses İletimi Kalitesi (QoS / QoE) Masası
- **Veri Kaynağı:** [HuggingFace - NISQA Non-Intrusive Speech Quality Assessment](https://huggingface.co/datasets)
- **Model:** NISQA (CNN-Self-Attention) / PESQ Referanssız MOS Regresyonu
- **Türkçe Değişkenler:** `paket_kayip_orani`, `ses_gecikmesi_jitter_ms`, `kodek_turu_amr_wb_evs`, `tahmini_mos_puani_1_5`
- **Jupyter Notebook (`gun_080_ses_kalitesi_mos_tahmini.ipynb`):**
  1. VoLTE/VoIP aramalarındaki paket kaybı, jitter ve kodek sıkıştırma bozulmalarının modellenmesi
  2. Referans ses olmadan (Non-Intrusive) yapay zeka ile 1.0 - 5.0 arası Mean Opinion Score (MOS) tahmini
  3. Ses kalitesi 3.5'in altına düşen baz istasyonlarının otomatik şebeke optimizasyonuna bildirilmesi
- **Mülakat Sorusu:** Geleneksel ITU-T P.862 PESQ (Intrusive) ölçümü ile Derin Öğrenme tabanlı NISQA (Non-Intrusive) MOS tahmini arasındaki operasyonel fark nedir?

### Gün 081: IVR Tek Kelimelik Sesli Komut Algılama (Keyword Spotting - KWS)
- **İş Alanı:** 532 Sesli Yanıt Menüsü (IVR) & Edge Cihaz Komut Algılama
- **Veri Kaynağı:** [HuggingFace - Google Speech Commands Dataset](https://huggingface.co/datasets/speech_commands)
- **Model:** Temporal Convolutional Network (TCN) / Squeeze-and-Excitation 1D-CNN (<50 KB Model)
- **Türkçe Değişkenler:** `kisa_ses_kesiti_1sn`, `algilanan_komut_evet_hayir_fatura_iptal`, `komut_olasiligi`
- **Jupyter Notebook (`gun_081_sesli_komut_algilama_kws.ipynb`):**
  1. 1 saniyelik ses pencerelerinden MFCC öznitelik haritası çıkarılması
  2. "Evet", "Hayır", "Fatura", "İptal", "Temsilci" gibi anahtar kelimelerin milisaniyeler içinde tespiti
  3. Edge cihazlarda ve düşük güçlü sunucularda ultra düşük CPU/RAM tüketimiyle inference
- **Mülakat Sorusu:** Keyword Spotting sistemlerinde "Streaming Inference" yaparken kayan pencere (Sliding Window) ve False Trigger önleme mantığı nasıl kurulur?

### Gün 082: Müşteri Temsilcisi Botu için Türkçe TTS (Metinden Sese) Sentezleme
- **İş Alanı:** Turkcell Sesli Dijital Asistan & Çağrı Merkezi Botu
- **Veri Kaynağı:** [HuggingFace - Turkish Single-Speaker Speech Dataset / Common Voice](https://huggingface.co/datasets)
- **Model:** VITS (Variational Inference with adversarial learning for Text-to-Speech) / FastSpeech2
- **Türkçe Değişkenler:** `bot_yanit_metni`, `uretilen_spektrogram`, `sentetik_ses_dalgasi_wav`, `dogallik_mos_skoru`
- **Jupyter Notebook (`gun_082_turkce_tts_ses_sentezleme.ipynb`):**
  1. Türkçe metinlerin fonem (Phoneme) dizilimine dönüştürülmesi
  2. VITS mimarisi ile uçtan uca spektrogram ve dalga formu (Vocoder) üretimi
  3. Doğal tonlamalı, akıcı ve kurumsal Turkcell marka ses karakterinde gerçek zamanlı ses sentezleme
- **Mülakat Sorusu:** TTS modellerinde autoregressive mimariler (Tacotron) yerine non-autoregressive (FastSpeech2/VITS) modellerin tercih edilmesinin gerçek zamanlı çağrı merkezi botlarındaki önemi nedir?

### Gün 083: Çağrıda Konuşmacı Ayrıştırma (Speaker Diarization - Kim Ne Zaman Konuştu?)
- **İş Alanı:** 532 Çağrı Analitiği Masası & Temsilci Müşteri Konuşma Oranı
- **Veri Kaynağı:** [HuggingFace - AMI Meeting Corpus / CallHome Adapted](https://huggingface.co/datasets)
- **Model:** PyAnnote.Audio / Spectral Clustering + Voice Activity Detection (VAD)
- **Türkçe Değişkenler:** `cagri_ses_kaydi`, `konusmaci_etiketi_temsilci_musteri`, `konusma_baslangic_sn`, `konusma_bitis_sn`, `ust_uste_konusma_orani`
- **Jupyter Notebook (`gun_083_konusmaci_ayristirma_diarization.ipynb`):**
  1. Ses sinyalinde konuşma bölgelerinin VAD ile tespiti ve sessizliklerin kırpılması
  2. Kayan pencerelerden ses embedding'leri çıkarılarak Spektral Kümeleme ile ayrıştırma
  3. Temsilcinin müşterinin sözünü kesme (Overlapping Speech) ve müşteri sessizlik sürelerinin analizi
- **Mülakat Sorusu:** Diarization Error Rate (DER) metriği hangi üç alt bileşenden (Speaker Confusion, False Alarm, Missed Detection) oluşur?

### Gün 084: Çağrı Merkezine Gelen Sentetik / Klon Ses (Deepfake Voice) Tespiti
- **İş Alanı:** Paycell & 532 Sesli İşlem Dolandırıcılık Önleme
- **Veri Kaynağı:** [Kaggle - ASVspoof 2021 / Synthetic Speech & Voice Clone Dataset](https://www.kaggle.com/datasets)
- **Model:** RawNet2 / LFCC (Linear Frequency Cepstral Coefficients) + ResNet-18
- **Türkçe Değişkenler:** `cagri_sesi_akisi`, `lfcc_spektrogrami`, `derin_sahtecilik_deepfake_olasiligi`, `islem_guvenlik_blokesi`
- **Jupyter Notebook (`gun_084_deepfake_ses_tespiti.ipynb`):**
  1. TTS ve Voice Conversion algoritmalarının ürettiği faz uyuşmazlıkları ve yüksek frekans kalıntılarının LFCC ile çıkarımı
  2. Canlı insan sesi ile yapay zeka tarafından üretilen klon seslerin ayrıştırılması
  3. Sesli bankacılık/Paycell işlemlerinde deepfake ses saldırısını milisaniyeler içinde bloke etme
- **Mülakat Sorusu:** Deepfake ses tespitinde MFCC yerine neden yüksek frekans detaylarını koruyan LFCC (Linear Frequency Cepstral Coefficients) tercih edilir?

### Gün 085: Çağrı Bekleme Müziği ve Sessizlik Süresi Ölçer (Music vs Speech Segmentation)
- **İş Alanı:** 532 Çağrı Merkezi Operasyonel Verimlilik (AHT - Average Handling Time Analizi)
- **Veri Kaynağı:** [Kaggle - GTZAN Music Speech Classification](https://www.kaggle.com/datasets)
- **Model:** CRNN (Convolutional Recurrent Neural Network) + Zero-Crossing Rate & Spectral Flatness
- **Türkçe Değişkenler:** `cagri_sesi`, `bekleme_muzigi_suresi_sn`, `net_konusma_suresi_sn`, `temsilci_bekletme_orani`
- **Jupyter Notebook (`gun_085_muzik_sessizlik_ayristirma.ipynb`):**
  1. Ses akışının 0.5 saniyelik pencerelerde "Konuşma", "Müzik", "Sessizlik/Gürültü" olarak etiketlenmesi
  2. Müşterinin beklemede (Hold) kaldığı sürelerin otomatik ölçümü
  3. Çağrı sürelerini şişiren gereksiz bekleme müziklerinin ve sessiz anların tespiti
- **Mülakat Sorusu:** Müzik ve konuşma sinyallerini ayırt etmede Spectral Centroid, Spectral Rolloff ve Zero Crossing Rate özniteliklerinin fiziksel anlamı nedir?

---

## 🎬 Modül 07: Öneri Sistemleri, TV+, fizy & Dijital Servisler (Gün 086 – 095)

### Gün 086: fizy Kişiselleştirilmiş Çalma Listesi Öneri Motoru
- **İş Alanı:** fizy Müzik Servisi & Kullanıcı Tutundurma
- **Veri Kaynağı:** [Kaggle - Spotify Million Playlist Dataset](https://www.kaggle.com/datasets)
- **Model:** Implicit Collaborative Filtering (Alternating Least Squares - ALS) + LightFM
- **Türkçe Değişkenler:** `kullanici_id`, `sarki_id`, `dinleme_sayisi`, `oneri_listesi`, `benzerlik_skoru`
- **Jupyter Notebook (`gun_086_fizy_muzik_oneri_motoru.ipynb`):**
  1. Kullanıcı-şarkı örtük geri bildirim (Implicit Feedback - dinleme sayısı, tamamlama oranı) matrisi inşası
  2. ALS matris çarpanlarına ayırma ile gizli özellik (Latent Factors) vektörlerinin çıkarımı
  3. Precision@K, Recall@K ve MAP@K metrikleri ile öneri kalitesinin ölçümü
- **Mülakat Sorusu:** Örtük geri bildirimde (Implicit Feedback) kullanıcıların "dinlememiş olması" negatif geri bildirim midir? ALS formülündeki güven katsayısı ($c_{ui} = 1 + lpha r_{ui}$) bunu nasıl çözer?

### Gün 087: TV+ İçerik Tabanlı Film ve Dizi Öneri Sistemi
- **İş Alanı:** TV+ Dijital Televizyon Servisi
- **Veri Kaynağı:** [Kaggle - The Movies Dataset / TMDB 5000](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- **Model:** TF-IDF + BERTurak İçerik Embedding'leri + Cosine Similarity
- **Türkçe Değişkenler:** `film_id`, `film_ozeti_metni`, `tur_yonetmen_oyuncular`, `icerik_benzerlik_matrisi`, `onerilen_filmler`
- **Jupyter Notebook (`gun_087_tv_plus_icerik_tabanli_oneri.ipynb`):**
  1. Film özetleri, türleri, yönetmen ve oyuncu kadrolarının birleşik metin haline getirilmesi
  2. Vektörleştirme ile içerik benzerlik matrisinin hesaplanması
  3. İzlenen bir filmden yola çıkarak benzer temalı 10 içeriğin milisaniyeler içinde listelenmesi
- **Mülakat Sorusu:** İçerik tabanlı öneri sistemlerinde "Filter Bubble" (Kullanıcıyı sürekli benzer içeriğe hapsetme) problemi nedir ve Serendipity / Diversity metrikleriyle nasıl aşılır?

### Gün 088: Müzik Atlama (Skip) Davranışı Tahminleyici
- **İş Alanı:** fizy Kullanıcı Deneyimi & Çalma Listesi Sıralama Optimizasyonu
- **Veri Kaynağı:** [Kaggle - Spotify Sequential Skip Prediction](https://www.kaggle.com/datasets)
- **Model:** CatBoost Classifier + Sıralı Oturum Öznitelikleri
- **Türkçe Değişkenler:** `oturum_id`, `sarki_sira_no`, `kullanici_onceki_sarkiyi_atladi_mi`, `sarki_enerji_seviyesi`, `atlama_olasiligi`
- **Jupyter Notebook (`gun_088_muzik_atlama_skip_tahmini.ipynb`):**
  1. Dinleme oturumundaki şarkı geçiş dinamiklerinin analizi
  2. Şarkının ilk 30 saniyesinde atlanma riskini artıran akustik ve bağlamsal faktörlerin tespiti
  3. Atlama riski yüksek şarkıları çalma listesinde alt sıralara kaydıran dinamik sıralama motoru
- **Mülakat Sorusu:** Müzik atlama davranışında "Context Awareness" (kullanıcının o an arabada, sporda veya gece dinlemesi) model başarımını nasıl etkiler?

### Gün 089: BiP Çıkartma (Sticker) & Popülerlik Modeli
- **İş Alanı:** BiP Anlık Mesajlaşma & Sosyal Etkileşim
- **Veri Kaynağı:** [Kaggle - Social Media Virality & Sticker Usage](https://www.kaggle.com/datasets)
- **Model:** GBDT / Poisson Regresyonu + Zaman Serisi Eğilim Analizi
- **Türkçe Değişkenler:** `cikartma_paketi_id`, `gunluk_gonderim_sayisi`, `viral_yayilim_katsayisi`, `trend_cikartma_mi`
- **Jupyter Notebook (`gun_089_bip_cikartma_trend_modeli.ipynb`):**
  1. BiP kullanıcılarının gönderdiği çıkartma (sticker) ve emoji kullanım frekanslarının analizi
  2. Trend olan ve viralleşen yeni çıkartma paketlerinin erken tespiti
  3. BiP mağazasında popüler çıkartmaların öne çıkarılması ve öneri şeridine eklenmesi
- **Mülakat Sorusu:** Sayım verisi (Count Data - çıkartma kullanım adedi) modellerken neden Linear Regression yerine Poisson / Negative Binomial Regresyon tercih edilir?

### Gün 090: Video Akışında Uyarlanabilir Bant Genişliği ve QoS Optimizasyonu
- **İş Alanı:** TV+ Video Streaming & DASH / HLS Uyarlanabilir Akış
- **Veri Kaynağı:** [Kaggle - Video Streaming QoS & Buffer Telemetry](https://www.kaggle.com/datasets)
- **Model:** Derin Pekiştirmeli Öğrenme (Deep Q-Network - DQN / A3C)
- **Türkçe Değişkenler:** `anlik_ag_hizi_mbps`, `video_arabellek_dolulugu_sn`, `secilen_cozunurluk_bitrate_kbps`, `donma_orani_rebuffer`
- **Jupyter Notebook (`gun_090_video_akis_qos_dqn.ipynb`):**
  1. 4G/5G dalgalı mobil ağlarda tampon bellek (Buffer) doluluk telemetrisinin simülasyonu
  2. DQN ajanı ile video donmasını (Rebuffering) sıfırlarken maksimum görüntü kalitesi (4K/1080p/720p) seçimi
  3. Kullanıcı QoE skorunu maksimize eden ödül fonksiyonu tasarımı
- **Mülakat Sorusu:** Video streaming ABR (Adaptive Bitrate) kontrolünde kural tabanlı BOLA/Pensieve algoritmalarına karşı Reinforcement Learning'in avantajı nedir?

### Gün 091: fizy Otomatik Spektrogram Tabanlı Müzik Türü & Mod Çıkarımı
- **İş Alanı:** fizy Müzik Kataloğu Otomatik Etiketleme (Auto-Tagging)
- **Veri Kaynağı:** [Kaggle - GTZAN Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Model:** 2D-CNN (ResNet-34) / Mel-Spektrogram Görsel Sınıflandırma
- **Türkçe Değişkenler:** `sarki_ses_kesiti`, `mel_spektrogram_gorseli`, `tahmini_muzik_turu_rock_pop_caz`, `muzik_modu_enerjik_huzunlu`
- **Jupyter Notebook (`gun_091_muzik_turu_mod_cnn.ipynb`):**
  1. Ses dosyalarının 3 saniyelik Mel-Spektrogram görüntülerine dönüştürülmesi
  2. Bilgisayarlı Görü CNN mimarisiyle spektrogram doku ve ritim paternlerinin öğrenilmesi
  3. fizy kataloğuna yeni eklenen bağımsız şarkıların otomatik tür ve mod etiketlemesi
- **Mülakat Sorusu:** Ses sinyallerini 2D Spektrogram haline getirip Görü (Vision) modelleriyle eğitmenin 1D Waveform modellerine göre avantajları ve hesaplama maliyeti nedir?

### Gün 092: Oturum Tabanlı (Session-based) TV Programı Önerisi
- **İş Alanı:** TV+ Canlı Yayın & Giriş Yapmamış Ziyaretçi Önerileri
- **Veri Kaynağı:** [Kaggle - RecSys Challenge Session Dataset](https://www.kaggle.com/datasets)
- **Model:** GRU4Rec / Graph Neural Network (SR-GNN - Session-based RecSys)
- **Türkçe Değişkenler:** `anonim_oturum_id`, `izlenen_kanallar_dizisi`, `kanal_kalma_suresi_sn`, `siradaki_kanal_onerisi`
- **Jupyter Notebook (`gun_092_oturum_tabanli_tv_onerisi.ipynb`):**
  1. Geçmiş kullanıcı profili olmadan sadece o anki ardışık kanal geçişleri üzerinden oturum modelleme
  2. GRU4Rec ile kullanıcının anlık moduna uygun canlı TV kanalı tahmini
  3. TV kumandası kanal değiştirme anında sıradaki en uygun 3 kanalın ekranda önerilmesi
- **Mülakat Sorusu:** Kullanıcı kimliği bilinmeyen anonim oturumlarda (Cold-Start Sessions) klasik Matris Faktörizasyonu neden çalışmaz ve GRU4Rec bunu nasıl çözer?

### Gün 093: Game+ Bulut Oyun Ping ve En Yakın Sunucu Eşleme Modeli
- **İş Alanı:** Turkcell Game+ Bulut Oyun Platformu (GeForce NOW)
- **Veri Kaynağı:** [Kaggle - Cloud Gaming Network Latency & Server Telemetry](https://www.kaggle.com/datasets)
- **Model:** K-Nearest Neighbors (KNN) Regressor / XGBoost Ping Tahmini
- **Türkçe Değişkenler:** `oyuncu_ip_bloğu`, `kullanici_isp_operatoru`, `sunucu_lokasyonu_istanbul_ankara_izmir`, `tahmini_gecikme_ping_ms`
- **Jupyter Notebook (`gun_093_game_plus_sunucu_esleme.ipynb`):**
  1. Oyuncunun IP adresi, ISP routing rotası ve şebeke tipi (Fiber/DSL/5G) öznitelikleri
  2. Her veri merkezi sunucusu için oyun başlamadan önce milisaniye cinsinden ping süresi tahmini
  3. Oyuncuyu en düşük gecikmeye (<15ms) ve en stabil FPS değerine sahip Game+ sunucusuna otomatik atama
- **Mülakat Sorusu:** Bulut oyunda "Jitter" (gecikme dalgalanması) ve paket kaybının FPS stabilitesine etkisi ortalama ping değerinden neden daha kritiktir?

### Gün 094: Dinleyici Yaş ve İlgi Alanı Demografik Tahmini
- **İş Alanı:** fizy & TV+ Hedefli Reklamcılık Masası
- **Veri Kaynağı:** [Kaggle - Last.fm User Demographic & Listening Habits](https://www.kaggle.com/datasets)
- **Model:** Multi-Layer Perceptron (MLP) / CatBoost Classifier
- **Türkçe Değişkenler:** `sanatci_cesitlilik_orani`, `gece_dinleme_yuzdesi`, `dinlenen_turler_vektoru`, `tahmini_yas_grubu_18_25_35`
- **Jupyter Notebook (`gun_094_dinleyici_demografi_tahmini.ipynb`):**
  1. Dinlenen müzik türleri, sanatçılar ve gün içi dinleme saatlerinden demografik öznitelik çıkarımı
  2. Yaş grubu, müzik zevki ve ilgi alanı segmentasyonu (Persona Çıkarımı)
  3. Müşteriye özel reklam ve kişiselleştirilmiş kampanya eşleştirmesi
- **Mülakat Sorusu:** Kullanıcıların açıkça yaşını beyan etmediği durumlarda davranışsal verilerden demografik tahmin yaparken KVKK ve gizlilik (Privacy-Preserving AI) sınırları nasıl korunur?

### Gün 095: Podcast Bölümünden İlgi Çekici Anları Özetleme ve Klipleme
- **İş Alanı:** fizy Podcast Servisi & Sosyal Medya Paylaşım Otomasyonu
- **Veri Kaynağı:** [HuggingFace - Spotify Podcast Dataset](https://huggingface.co/datasets)
- **Model:** Whisper ASR + BERTurk TextRank / Sentence-Transformers Embeddings
- **Türkçe Değişkenler:** `podcast_sesi_yolu`, `bolum_transkripti`, `en_onemli_cumleler_skoru`, `klip_baslangic_bitis_sn`
- **Jupyter Notebook (`gun_095_podcast_one_cikan_anlar.ipynb`):**
  1. 1 saatlik podcast kaydının transkripte dönüştürülmesi ve anlamsal paragraflara bölünmesi
  2. TextRank algoritması ve semantik yoğunluk skorlarıyla en vurucu 60 saniyelik kısmın tespiti
  3. fizy uygulamasında ve sosyal medyada paylaşılmak üzere otomatik 60 saniyelik sesli klip (Highlight Reel) üretimi
- **Mülakat Sorusu:** Uzun ses kayıtlarında öne çıkan anları (Highlights) belirlemede metinsel TextRank ile akustik enerji/kahkaha/ton yükselmesi füzyonu nasıl yapılır?

---

## 🌐 Modül 08: IoT, Akıllı Şehir & Edge AI (Gün 096 – 100)

### Gün 096: Akıllı Şehir Trafik Akışı & Araç Yoğunluğu Haritalama
- **İş Alanı:** Turkcell Akıllı Şehir Çözümleri & IoT Trafik Yönetimi
- **Veri Kaynağı:** [Roboflow Universe - Traffic Density & Vehicle Detection](https://universe.roboflow.com/)
- **Model:** YOLOv8-Nano (Edge TPU / INT8 Quantized) + ByteTRACK
- **Türkçe Değişkenler:** `kamera_kare_goruntusu`, `arac_sinifi_otobus_otomobil_motor`, `dakikalik_gecen_arac_sayisi`, `yogunluk_indeksi_0_1`
- **Jupyter Notebook (`gun_096_akilli_sehir_trafik_yogunluk.ipynb`):**
  1. Şehir kameralarından alınan RTSP video akışından araç tespiti ve çoklu nesne takibi (MOT)
  2. Şerit bazlı araç sayımı ve kavşak yoğunluk ısı haritası (Heatmap) üretimi
  3. Trafik ışık optimizasyon sistemine MQTT protokolüyle anlık yoğunluk telemetrisi aktarımı
- **Mülakat Sorusu:** Edge AI cihazlarında (Raspberry Pi / NVIDIA Jetson) 30 FPS gerçek zamanlı çıkarım sağlamak için INT8 Post-Training Quantization (PTQ) nasıl uygulanır?

### Gün 097: Akıllı Sayaç (Elektrik/Su) Kaçak & Anomali Tespiti
- **İş Alanı:** NB-IoT Akıllı Şebeke (Smart Grid) & Altyapı Sayaç Yönetimi
- **Veri Kaynağı:** [Kaggle - Smart Meter Energy Consumption Data in London](https://www.kaggle.com/datasets/jeanmadev/smart-meters-in-london)
- **Model:** Autoencoder + Isolation Forest + Dynamic Time Warping (DTW)
- **Türkçe Değişkenler:** `sayac_id`, `yarim_saatlik_kwh_tuketim`, `tipik_gunluk_profil_farki`, `kacak_kullanim_anomali_skoru`
- **Jupyter Notebook (`gun_097_akilli_sayac_kacak_tespiti.ipynb`):**
  1. NB-IoT üzerinden gelen 30 dakikalık enerji/su tüketim telemetrilerinin toplanması
  2. Normal tüketim profili ile gerçek kullanım arasındaki DTW mesafe sapmalarının incelenmesi
  3. Sayaç manipülasyonu ve hat kaçağı şüphesi olan abonelerin otomatik tespit edilip saha ekiplerine atanması
- **Mülakat Sorusu:** Zaman serilerinde ani tüketim sıfırlanması ile periyodik bayram/tatil düşüşlerini ayırt etmek için hangi takvimsel ayrıştırma (STL Decomposition) uygulanır?

### Gün 098: IoT İstasyonları Hava Kirliliği (PM2.5) Zaman Serisi Tahmini
- **İş Alanı:** Çevre İzleme & Akıllı Şehir Hava Kalitesi İndeksi (AQI)
- **Veri Kaynağı:** [Kaggle - Air Quality Time Series Data (PM2.5 / NO2 / SO2)](https://www.kaggle.com/datasets)
- **Model:** Bi-LSTM + Attention Mekanizması / Temporal Convolutional Network (TCN)
- **Türkçe Değişkenler:** `hava_istasyon_id`, `pm25_degeri`, `ruzgar_hizi_yonu`, `sicaklik_nem`, `gelecek_24saat_tahmini_aqi`
- **Jupyter Notebook (`gun_098_iot_hava_kirliligi_tahmini.ipynb`):**
  1. Çok değişkenli (Multivariate) çevre sensör telemetrilerinin kayan pencere (Sliding Window) formatına dönüştürülmesi
  2. LSTM-Attention modeli ile önümüzdeki 24 saatin saatlik PM2.5 ve AQI seviyelerinin tahmini
  3. Kritik eşik aşıldığında belediye ve vatandaşlara Turkcell SMS/BiP üzerinden otomatik uyarı gönderilmesi
- **Mülakat Sorusu:** Çok değişkenli zaman serisi tahmininde hava durumu gibi dışsal değişkenlerin (Exogenous Variables) gecikmeli etkileri (Lagged Features) nasıl modellenir?

### Gün 099: Akıllı Otopark Doluluk Tespiti ve Yönlendirme
- **İş Alanı:** Turkcell Akıllı Otopark & Saha IoT Görü
- **Veri Kaynağı:** [Kaggle - PKLot Parking Lot Dataset](https://www.kaggle.com/datasets/allanbr/pklot)
- **Model:** ResNet-18 / MobileNetV3 (Park Yeri Sınıflandırma)
- **Türkçe Değişkenler:** `park_yeri_koordinati_roi`, `doluluk_durumu_dolu_bos`, `bos_kalan_sure_dakika`, `en_yakin_bos_alan_id`
- **Jupyter Notebook (`gun_099_akilli_otopark_doluluk_tespiti.ipynb`):**
  1. Otopark kamerası üzerinden belirlenen her bir park alanının (ROI - Region of Interest) kırpılması
  2. Işık, gölge, yağmur ve gece şartlarında hafif CNN modeli ile dolu/boş tespiti
  3. Mobil uygulamaya ve sokak yönlendirme panolarına anlık boş yer sayısının iletilmesi
- **Mülakat Sorusu:** Otopark kameralarında aşırı güneş yansıması veya araç gölgelerinin boş park yerini "dolu" göstermesini engellemek için hangi veri artırma (Data Augmentation) teknikleri uygulanır?

### Gün 100: Akıllı Tarım Toprak Nemi ve Sulama Karar Motoru
- **İş Alanı:** Turkcell Dijital Tarım & IoT Toprak Sensör Ağı
- **Veri Kaynağı:** [Kaggle - Smart Agriculture & Soil Moisture Sensor Data](https://www.kaggle.com/datasets)
- **Model:** Random Forest Regressor + Kural Tabanlı Uzman Karar Motoru (Fuzzy Logic)
- **Türkçe Değişkenler:** `tarla_bolge_id`, `toprak_nemi_yuzdesi`, `toprak_sicakligi_c`, `gunes_radyasyonu`, `sulama_vanasi_acik_kalma_dakika`
- **Jupyter Notebook (`gun_100_akilli_tarim_sulama_motoru.ipynb`):**
  1. Toprak nem sensörleri, buharlaşma-terleme (Evapotranspiration) ve hava durumu tahminlerinin entegrasyonu
  2. Bitki su ihtiyacı ve solma noktası (Wilting Point) analizi
  3. IoT solenoid vanaların gereksiz su tüketimini önleyecek şekilde otomatik açılıp kapanma karar algoritması
- **Mülakat Sorusu:** IoT tabanlı tarım sistemlerinde pil ömrünü uzatmak için sensörlerin ne sıklıkla veri göndereceğini dinamik ayarlayan Adaptive Sampling mantığı nasıl tasarlanır?

---

# 🚀 BÖLÜM 2: GÜN 101 – 200

## 📡 Modül 09: Telekom Şebeke Optimizasyonu, Radyo & 5G Altyapısı (Gün 101 – 115)

### Gün 101: 5G Massive MIMO Kanal Durum Bilgisi (CSI) Geri Bildirim Sıkıştırma
- **İş Alanı:** 5G Radyo Şebekesi & Anten Verimliliği
- **Veri Kaynağı:** [Kaggle / IEEE DataPort - Massive MIMO CSI Dataset](https://www.kaggle.com/datasets)
- **Model:** Complex-Valued Autoencoder (CsiNet)
- **Türkçe Değişkenler:** `anten_sayisi`, `alt_tasiyici_frekansi`, `ham_csi_matrisi`, `sikistirilmis_vektor`, `rekonstruksiyon_nmse`
- **Jupyter Notebook (`gun_101_5g_mimo_csi_sikistirma.ipynb`):**
  1. Kullanıcı cihazı (UE) ile baz istasyonu (gNodeB) arasındaki karmaşık kanal matrisinin elde edilmesi
  2. Uplink geri bildirim yükünü 1/16 oranına düşüren Derin Öğrenme Autoencoder mimarisi
  3. Normalized Mean Squared Error (NMSE) metriği ile sinyal geri çatım kalitesinin ölçümü
- **Mülakat Sorusu:** FDD 5G sistemlerinde CSI geri bildiriminin kanal gecikmesinden (Channel Aging) önce baz istasyonuna iletilmesinde Autoencoder sıkıştırmasının önemi nedir?

### Gün 102: Mobil Ağ Hücresel Yük Devri (Handover) Başarı Tahmini
- **İş Alanı:** Hareket Halinde Kesintisiz İletişim (Mobility Management)
- **Veri Kaynağı:** [UCI Machine Learning Repository - Wireless Handover Traces](https://archive.ics.uci.edu/)
- **Model:** XGBoost Classifier + Histeresis / Time-to-Trigger Optimizasyonu
- **Türkçe Değişkenler:** `kaynak_hucre_rsrp`, `hedef_hucre_rsrp`, `kullanici_hizi_kmh`, `gecis_basarili_mi`, `ping_pong_gecis_sayisi`
- **Jupyter Notebook (`gun_102_hucre_gecisi_handover_tahmini.ipynb`):**
  1. Hızlı tren ve otoyolda seyahat eden kullanıcıların hücre sinyal gücü (RSRP/RSRQ) değişimlerinin analizi
  2. Başarısız geçişleri ve iki hücre arasında gereksiz git-gel (Ping-Pong Handover) hareketlerini tahmin etme
  3. Dinamik Time-to-Trigger eşiği ayarlayarak çağrı düşme oranını (Call Drop Rate) minimize etme
- **Mülakat Sorusu:** Handover optimizasyonunda "Too Early Handover" ile "Too Late Handover" arasındaki farklar ve şebekeye maliyetleri nelerdir?

### Gün 103: Radyo Erişim Şebekesi (RAN) Güç Tüketimi Tahminleyici
- **İş Alanı:** Şebeke Enerji Verimliliği & Yeşil Telekom Operasyonları
- **Veri Kaynağı:** [Kaggle - Telecom Base Station Power Consumption](https://www.kaggle.com/datasets)
- **Model:** LightGBM Regressor + SHAP Değişken Önem Analizi
- **Türkçe Değişkenler:** `baz_istasyon_id`, `aktif_tasiyici_frekans_sayisi`, `anlik_veri_trafigi_gb`, `enerji_tuketimi_kwh`
- **Jupyter Notebook (`gun_103_ran_guc_tuketim_tahmini.ipynb`):**
  1. Baz istasyonundaki RF yükselticiler, dijital işlem üniteleri ve klima güç tüketimlerinin ayrıştırılması
  2. Gece trafiği düşen saatlerde kapatılabilecek yedek taşıyıcıların enerji tasarruf potansiyeli
  3. İstasyon bazında karbon emisyon ve elektrik faturası maliyet simülasyonu
- **Mülakat Sorusu:** Baz istasyonu güç modellemesinde yükten bağımsız sabit taban güç (Static Power) ile veri yüküne bağlı dinamik güç (Dynamic Power) nasıl ayrıştırılır?

### Gün 104: Baz İstasyonu Anten Açısı (Tilt/Azimuth) Sapma Tespiti
- **İş Alanı:** Saha Kalite Denetimi & Anten Mekanik Yön Sapmaları
- **Veri Kaynağı:** [Telecom Antenna Radiation Pattern Telemetry](https://www.kaggle.com/datasets)
- **Model:** 1D-CNN + Random Forest + Işınım Deseni Sınıflandırma
- **Türkçe Değişkenler:** `anten_id`, `nominal_tilt_acisi`, `olculen_hucre_kapsama_profili`, `mekanik_sapma_derecesi`
- **Jupyter Notebook (`gun_104_anten_tilt_sapma_tespiti.ipynb`):**
  1. Fırtına veya rüzgar nedeniyle yönü kayan antenlerin kapsama alanı sinyal izlerinin incelenmesi
  2. Hücre içi kullanıcı dağılımından fiziksel anten açısı (Electrical/Mechanical Tilt) sapmasının tahmini
  3. Saha ekiplerine otomatik düzeltme iş emri oluşturulması
- **Mülakat Sorusu:** Anten Over-shooting (aşırı uzak menzile yayın yapma) durumunun komşu baz istasyonlarında enterferans (SINR düşüşü) yaratması nasıl modellenir?

### Gün 105: SIM Kart Bağlantı Kopma Sıklığı Sınıflandırma
- **İş Alanı:** Müşteri Deneyimi & Hatalı Donanım / SIM Teşhisi
- **Veri Kaynağı:** [Telecom Network Disconnection & Detach Logs](https://www.kaggle.com/datasets)
- **Model:** CatBoost Classifier + İkili Sınıflandırma
- **Türkçe Değişkenler:** `sim_kart_uretim_serisi`, `gunluk_baglanti_kopma_sayisi`, `cihaz_model_kodu`, `arizali_sim_etiketi`
- **Jupyter Notebook (`gun_105_sim_kart_kopma_teshisi.ipynb`):**
  1. Hücresel şebekeden ani Detach olan ve PIN sorma moduna düşen SIM hareketlerinin filtrelenmesi
  2. Cihaz uyumsuzluğu ile fiziksel SIM kart çip aşınmasının ayrıştırılması
  3. Kronik arızalı SIM kartlı müşterilere otomatik ücretsiz yedek SIM değişim SMS'i tetikleme
- **Mülakat Sorusu:** Ağdan kopma sebeplerinde (Cause Codes) Radio Link Failure (RLF) ile Core Network kaynaklı Detach ayrımı modelde nasıl kullanılır?

### Gün 106: WiFi - LTE/5G Otomatik Ağ Geçiş (Offloading) Karar Motoru
- **İş Alanı:** Turkcell WiFi & Mobil Şebeke Yük Dengeleme
- **Veri Kaynağı:** [Kaggle - Wireless QoS & Network Offloading Traces](https://www.kaggle.com/datasets)
- **Model:** Derin Pekiştirmeli Öğrenme (Deep Q-Learning - DQN)
- **Türkçe Değişkenler:** `wifi_sinyal_gucu_rssi`, `mobil_hiz_mbps`, `kullanici_uygulama_turu_video_oyun_web`, `secilen_ag_wifi_mobil`
- **Jupyter Notebook (`gun_106_wifi_mobil_offloading_dqn.ipynb`):**
  1. Zayıf ve kararsız WiFi ağlarına bağlanıp internet deneyimi bozulan kullanıcı durumlarının simülasyonu
  2. Kullanıcının o anki internet tüketimine (video izleme, online oyun oynama) göre optimum ağ seçimi
  3. Operatör hücresel yükünü azaltırken kullanıcı QoE skorunu maksimize etme
- **Mülakat Sorusu:** WiFi Offloading kararlarında "Ping-Pong Efekti" (iki ağ arasında sürekli geçiş yaparak paket kaybı yaşama) ceza fonksiyonuyla nasıl engellenir?

### Gün 107: 5G Ultra Düşük Gecikme (URLLC) Paket Kuyruk Analizi
- **İş Alanı:** 5G Endüstriyel IoT & Otonom Sürüş Şebeke Altyapısı
- **Veri Kaynağı:** [5G URLLC Packet Latency Telemetry](https://www.kaggle.com/datasets)
- **Model:** Temporal Fusion Transformer (TFT) / Quantile Regresyon (p99.9 Gecikme)
- **Türkçe Değişkenler:** `kuyruk_uzunlugu_paket`, `oncelik_sinifi_qos_flow_id`, `anlik_kanal_kapasitesi`, `p99_gecikme_ms`
- **Jupyter Notebook (`gun_107_5g_urllc_gecikme_analizi.ipynb`):**
  1. Kritik görev (Mission-Critical) iletişim paketlerinin kuyrukta bekleme sürelerinin modellenmesi
  2. p50 yerine p99.9 (Kuyruk Gecikmesi - Tail Latency) öngörüsü
  3. Gecikme 1 ms'nin üzerine çıkma riski taşıdığında şebeke dilimleme (Network Slicing) dinamik kaynak rezervasyonu
- **Mülakat Sorusu:** URLLC SLA garantilerinde ortalama gecikme (Mean Latency) yerine neden Quantile Regresyon ile p99.9 gecikme optimize edilir?

### Gün 108: Baz İstasyonları Arası Enterferans (SINR) Tahmini
- **İş Alanı:** Şebeke Frekans Planlama & Kapsama Kalitesi
- **Veri Kaynağı:** [Kaggle - Cellular Network SINR & Interference Traces](https://www.kaggle.com/datasets)
- **Model:** Graf Dikkat Ağları (Graph Attention Network - GAT)
- **Türkçe Değişkenler:** `baz_istasyon_kordinatlari`, `komsuluk_mesafesi_metre`, `iletim_gucu_dbm`, `sinr_degeri_db`
- **Jupyter Notebook (`gun_108_baz_istasyon_enterferans_gat.ipynb`):**
  1. Komşu hücrelerin baz istasyonu topolojisini uzamsal graf (Spatial Graph) olarak kurma
  2. Komşu hücrelerin aynı frekansta yayın yapmasından doğan enterferansı GAT ile öğrenme
  3. Hücre çıkış güçlerini (Transmission Power) enterferansı minimize edecek şekilde dinamik ayarlama
- **Mülakat Sorusu:** Graf Sinir Ağlarında komşu hücre enterferansını modellerken kenar ağırlıklarının (Edge Weights) mesafeye bağlı sönümlenmesi (Path Loss) nasıl dahil edilir?

### Gün 109: eSIM Profil İndirme Hata Oranı Kümeleme
- **İş Alanı:** Dijital eSIM Servisleri & Uzaktan SIM Yönetimi (RSP - Remote SIM Provisioning)
- **Veri Kaynağı:** [eSIM Download & Activation Logs](https://www.kaggle.com/datasets)
- **Model:** HDBSCAN + UMAP Boyut İndirgeme
- **Türkçe Değişkenler:** `cihaz_eum_kodu`, `profil_indirme_adimi`, `hata_yanit_kodu_sm_dp`, `kume_etiketi`
- **Jupyter Notebook (`gun_109_esim_profil_hata_kumeleme.ipynb`):**
  1. eSIM aktivasyonu sırasında SM-DP+ sunucusu ile cihaz arasındaki TLS/HTTP telemetri loglarının incelenmesi
  2. Belirli cihaz üreticileri ve işletim sistemi sürümlerinde tekrarlayan sessiz aktivasyon hatalarının kümelenmesi
  3. Profil yükleme başarısızlıklarının kök neden analizi
- **Mülakat Sorusu:** Yüksek boyutlu kategorik hata loglarında t-SNE yerine neden UMAP + HDBSCAN kombinasyonu tercih edilir?

### Gün 110: Şehir Izgara Haritasında İnsan Yoğunluğu Çıkarımı
- **İş Alanı:** Turkcell Büyük Veri & Şehir Planlama / Nüfus Hareketliliği
- **Veri Kaynağı:** [Telecom Spatio-Temporal Grid Traffic Dataset](https://www.kaggle.com/datasets)
- **Model:** ConvLSTM / Spatio-Temporal Graph Convolutional Network (ST-GCN)
- **Türkçe Değişkenler:** `h3_altigen_izgara_kodu`, `zaman_dilimi_saat`, `aktif_bagli_telefon_adedi`, `nufus_yogunluk_skoru`
- **Jupyter Notebook (`gun_110_izgara_harita_insan_yogunlugu.ipynb`):**
  1. Şehrin Uber H3 altıgen ızgaralarına (H3 Hexagons - Çözünürlük 8) bölünmesi
  2. Baz istasyonu sinyallerinden saatlik dinamik nüfus yoğunluğu tahmini
  3. Acil afet durumlarında ve büyük etkinliklerde toplanma alanlarının anlık kapasite takibi
- **Mülakat Sorusu:** Zamansal-Uzamsal (Spatio-Temporal) verilerde bağımsız 2D-CNN ve LSTM yerine ConvLSTM kullanmanın bellek ve parametre verimliliği nedir?

### Gün 111: Optik Sinyal-Gürültü Oranı (OSNR) Bozulma Öngörüsü
- **İş Alanı:** Turkcell Superonline Omurga Fiber Altyapısı & DWDM İletim
- **Veri Kaynağı:** [Optical DWDM Telemetry & OSNR Degradation Data](https://www.kaggle.com/datasets)
- **Model:** Random Survival Forest / Gradient Boosting Regressor
- **Türkçe Değişkenler:** `fiber_hat_id`, `dalga_boyu_kanali_nm`, `optik_guc_dbm`, `osnr_degeri_db`, `tahmini_bozulma_suresi_saat`
- **Jupyter Notebook (`gun_111_optik_osnr_bozulma_ongorusu.ipynb`):**
  1. DWDM yükselticilerinden toplanan optik telemetri zaman serilerinin işlenmesi
  2. Lazer yaşlanması ve fiber bükülmelerine bağlı sinyal gürültü oranı (OSNR) düşüşünün erken tespiti
  3. Trafik kesintisi yaşanmadan önce trafiğin alternatif koruma rotasına (Protection Path) aktarılması
- **Mülakat Sorusu:** DWDM optik ağlarda Polarizasyon Modu Dispersiyonu (PMD) ile OSNR arasındaki korelasyon modelde nasıl öznitelikleştirilir?

### Gün 112: LTE Hız Kısıtlama (Throttling) Tespiti ve Adil Kullanım Analizi
- **İş Alanı:** Şebeke Trafik Yönetimi & Adil Kullanım Politikası (FUP)
- **Veri Kaynağı:** [Kaggle - Mobile Network ISP Throttling Traces](https://www.kaggle.com/datasets)
- **Model:** Karar Ağacı / CUSUM (Cumulative Sum Control Chart) Değişim Noktası Tespiti
- **Türkçe Değişkenler:** `abone_id`, `paket_kullanim_orani`, `akistaki_anlik_hiz_kbps`, `hiz_kisitlama_uygulandi_mi`
- **Jupyter Notebook (`gun_112_lte_hiz_kisitlama_tespiti.ipynb`):**
  1. Paket kotasını aşan veya aşırı torrent kullanımı yapan hatlarda uygulanan hız düşüşlerinin izlenmesi
  2. CUSUM algoritması ile veri akış hızındaki ani yapay taban kısıtlamalarının (Bandwidth Shaping) tespiti
  3. Müşteri şikayetlerini önlemek adına kota dolum bildirimlerinin senkronizasyon kontrolü
- **Mülakat Sorusu:** Zaman serisinde ani ortalama değişimlerini (Step Change) yakalamada CUSUM algoritmasının istatistiksel hipotez testi mantığı nedir?

### Gün 113: Araç İçi İletişim (V2X) İletim Gecikmesi Modellemesi
- **İş Alanı:** Otonom Araçlar, C-V2X (Cellular V2X) & Akıllı Ulaşım
- **Veri Kaynağı:** [Connected Vehicles V2X Telemetry Dataset](https://www.kaggle.com/datasets)
- **Model:** Multi-Layer Perceptron (MLP) + Gaussian Process Regression (Belirsizlik Tahmini)
- **Türkçe Değişkenler:** `arac_mesafesi_metre`, `bagli_baz_istasyonu_yuk_orani`, `v2x_mesaj_tipi_cam_denm`, `tahmini_gecikme_ms`, `guven_araligi`
- **Jupyter Notebook (`gun_113_v2x_iletisim_gecikme_modeli.ipynb`):**
  1. Araçtan Araca (V2V) ve Araçtan Altyapıya (V2I) acil fren ve kaza uyarı mesajlarının simülasyonu
  2. Şebeke yükü ve araç hızına bağlı iletim gecikmesi ve belirsizlik (Uncertainty) modellemesi
  3. Kritik güvenlik mesajları için doğrudan sidelink (PC5) iletişimine geçiş kararı
- **Mülakat Sorusu:** Gaussian Process regresyonunda model tahmininin yanında varyans (güven aralığı) üretmenin otonom sürüş güvenliğindeki hayati rolü nedir?

### Gün 114: Telekom Veri Merkezi Sıcaklık Sensör Anomalileri Takibi
- **İş Alanı:** Turkcell Veri Merkezleri & Sunucu Odası İklimlendirme
- **Veri Kaynağı:** [Data Center Temperature & Humidity Sensor Telemetry](https://www.kaggle.com/datasets)
- **Model:** LSTM Autoencoder / DBSCAN Çok Sensörlü Anomali Tespiti
- **Türkçe Değişkenler:** `kabinet_sensor_id`, `giris_havasi_sicakligi_c`, `cikis_havasi_sicakligi_c`, `klima_fani_devir_rpm`, `sicak_nokta_hotspot_riski`
- **Jupyter Notebook (`gun_114_veri_merkezi_sicaklik_anomali.ipynb`):**
  1. Binlerce kabinet içi sıcaklık ve nem sensör verisinin gerçek zamanlı akış analizi
  2. Sıcak hava döngüsü kaçağı ve bölgesel aşırı ısınma (Hot-Spot) noktalarının tahmini
  3. Yangın ve sunucu kapanma riskine karşı dinamik soğutma (CRAC) yönlendirmesi
- **Mülakat Sorusu:** Çok sensörlü IoT ağlarında tek bir sensörün bozulması (Sensor Failure) ile gerçek fiziksel yangın/ısınma durumu nasıl ayırt edilir?

### Gün 115: Kapsama Alanı Olmayan Kör Nokta (Dead Zone) Haritalama
- **İş Alanı:** Şebeke Planlama & Yeni Baz İstasyonu Lokasyon Seçimi
- **Veri Kaynağı:** [OpenCelliD / User Crowdsourced Signal Coverage](https://www.opencellid.org/)
- **Model:** Kriging / Spatial Interpolation + Random Forest Regressor
- **Türkçe Değişkenler:** `enlem_boylam`, `sinyal_seviyesi_dbm`, `arazi_yukseklik_bilgisi`, `bina_yogunlugu`, `kor_nokta_mi`
- **Jupyter Notebook (`gun_115_kor_nokta_kapsama_haritalama.ipynb`):**
  1. Kullanıcıların mobil uygulamalarından toplanan kitle kaynaklı (Crowdsourced) sinyal gücü ölçümleri
  2. Ölçüm yapılmayan kör coğrafi alanlarda uzamsal Kriging enterpolasyonu ile sinyal tahmini
  3. Kapsama boşluklarını kapatmak için yeni mikrosite ve small-cell kurulum önceliklendirmesi
- **Mülakat Sorusu:** Uzamsal enterpolasyonda (Spatial Interpolation) Inverse Distance Weighting (IDW) ile Kriging (Gauss Süreci Tabanlı) arasındaki temel fark nedir?

---

## 💳 Modül 10: Fintek / Paycell, Dijital Cüzdan & Alternatif Risk (Gün 116 – 130)

### Gün 116: Dijital Cüzdan Bakiye Yetersizlik Tahmini
- **İş Alanı:** Paycell Dijital Cüzdan & Otomatik Bakiye Yükleme Önerisi
- **Veri Kaynağı:** [Mobile Wallet Insolvent Users Dataset](https://www.kaggle.com/datasets)
- **Model:** LightGBM Classifier + Zaman Serisi Harcama Paternleri
- **Türkçe Değişkenler:** `kullanici_id`, `mevcut_cuzdan_bakiyesi`, `haftalik_ortalama_harcama`, `bakiye_yetersiz_kalma_olasiligi`
- **Jupyter Notebook (`gun_116_cuzdan_bakiye_yetersizlik.ipynb`):**
  1. Kullanıcının yaklaşan fatura, abonelik ve rutin harcama geçmişinin taranması
  2. Önümüzdeki 3 gün içinde bakiyenin sıfırlanıp işlemin reddedilme riskinin hesaplanması
  3. Kullanıcıya işlem anında zor durumda kalmaması için "Otomatik Yükleme Tanımla" bildirimi gönderilmesi
- **Mülakat Sorusu:** Finansal bakiye yetersizliği tahmininde False Positive (Gereksiz yükleme hatırlatması) maliyeti ile False Negative (Müşterinin kasada kalması) maliyeti nasıl dengelenir?

### Gün 117: QR Kod ile Ödeme Dolandırıcılığı Dedektörü
- **İş Alanı:** Paycell QR ile Ödeme Güvenliği & Sahte Karekod Engelleme
- **Veri Kaynağı:** [QR Transaction Fraud & Merchant Telemetry](https://www.kaggle.com/datasets)
- **Model:** XGBoost + Coğrafi Mesafe ve İşlem Hızı (Velocity) Kuralları
- **Türkçe Değişkenler:** `kullanici_gps_konumu`, `uye_isyeri_gps_konumu`, `qr_okutma_araligi_sn`, `supheli_qr_islem_skoru`
- **Jupyter Notebook (`gun_117_qr_odeme_dolandiricilik.ipynb`):**
  1. QR kodun okutulduğu telefon lokasyonu ile işyerinin kayıtlı adresi arasındaki GPS mesafesi
  2. Başka bir şehirdeki sahte QR kodun uzaktan manipüle edilmesi vakalarının tespiti
  3. Şüpheli QR ödemelerine SMS OTP veya biyometrik yüz onayı zorunluluğu getirilmesi
- **Mülakat Sorusu:** QR dolandırıcılığında "Quishing" (Phishing QR) ile "Merchant Impersonation" arasındaki farklar makine öğrenmesinde nasıl etiketlenir?

### Gün 118: Fatura Taksitlendirme Geri Ödeme Skoru
- **İş Alanı:** Turkcell Finansman & Faturaya Ek Taksitli Cihaz/Hizmet Satışı
- **Veri Kaynağı:** [Kaggle - Loan Default & Installment Repayment Dataset](https://www.kaggle.com/datasets)
- **Model:** CatBoost + Monotonic Constraints (Monoton Artan/Azalan Kurallar)
- **Türkçe Değişkenler:** `fatura_tutari`, `talep_edilen_taksit_sayisi`, `gecmis_gecikme_adedi`, `taksit_odeme_basari_olasiligi`
- **Jupyter Notebook (`gun_118_fatura_taksit_odeme_skoru.ipynb`):**
  1. Abonenin geçmiş 24 aylık fatura ödeme disiplini ve GSM kullanım trendleri
  2. Gelir arttıkça veya gecikme azaldıkça riskin artmasını engelleyen Monoton Kısıtlı GBDT eğitimi
  3. Cihaz taksitlendirme onay limitinin dinamik belirlenmesi
- **Mülakat Sorusu:** Kredi modellerinde regülasyon uyumu için GBDT modellerine "Monotonic Constraints" eklemenin önemi nedir?

### Gün 119: Ön Ödemeli Kart (Prepaid) İnaktif Kullanıcı Tahmini
- **İş Alanı:** Paycell Kart Yaşam Döngüsü Yönetimi & Kart Terk (Dormancy) Önleme
- **Veri Kaynağı:** [Prepaid Card Inactivity & Churn Dataset](https://www.kaggle.com/datasets)
- **Model:** Random Forest + RFM Segmentasyonu
- **Türkçe Değişkenler:** `kart_id`, `son_harcamadan_gecen_gun`, `aylik_yukleme_sikligi`, `inaktif_olma_riski_skoru`
- **Jupyter Notebook (`gun_119_prepaid_kart_inaktiflik_tahmini.ipynb`):**
  1. Paycell fiziksel ve sanal kartlarda harcama sıklığı (Frequency) azalan kullanıcıların tespiti
  2. 60 gün boyunca işlem yapmama (Dormant) riski taşıyan müşterilerin segmentasyonu
  3. Kartı yeniden canlandıracak kişiselleştirilmiş cashback kampanyası önerilmesi
- **Mülakat Sorusu:** Fintekte "Churn" tanımı (hesap kapatma) ile "Dormancy" tanımı (kartı çekmecede unutma) arasındaki fark model hedef değişkenine nasıl yansıtılır?

### Gün 120: Sanal POS Başarılı Geçiş (Authorization Rate) Akıllı Yönlendirici
- **İş Alanı:** Paycell Ödeme Geçidi (Payment Gateway) & Akıllı POS Yönlendirme
- **Veri Kaynağı:** [Payment Gateway Routing & Auth Rates](https://www.kaggle.com/datasets)
- **Model:** Çok Kollu Haydut (Multi-Armed Bandit - Contextual Bandit / Thompson Sampling)
- **Türkçe Değişkenler:** `kart_ailesi_bonus_world_maximum`, `islem_tutari`, `hedef_banka_posu`, `tahmini_onay_orani`, `komisyon_orani`
- **Jupyter Notebook (`gun_120_sanal_pos_yonlendirici_bandit.ipynb`):**
  1. Banka sanal POS'larının anlık kesinti ve başarı oranlarının izlenmesi
  2. Contextual Thompson Sampling ile en yüksek onay oranına ve en düşük komisyona sahip banka POS'unun seçimi
  3. Başarısız işlemlerde müşteriye hissettirmeden milisaniyeler içinde alternatif POS'tan yeniden deneme (Retry Routing)
- **Mülakat Sorusu:** Statik kural tabanlı POS yönlendirme yerine Thompson Sampling kullanmanın keşif-sömürü (Exploration-Exploitation) dengesindeki kazancı nedir?

### Gün 121: Sentetik Kimlik (Synthetic Identity) Dolandırıcılığı Tespiti
- **İş Alanı:** Paycell Kimlik Doğrulama Güvenliği & Sahte Hesap Şebekeleri
- **Veri Kaynağı:** [Synthetic Identity Theft & Fraud Dataset](https://www.kaggle.com/datasets)
- **Model:** İzolasyon Ormanı (Isolation Forest) + Denetimsiz Graf Kümeleme
- **Türkçe Değişkenler:** `tc_kimlik_dogrulama_skoru`, `ayni_adresi_kullanan_farkli_kullanici_sayisi`, `cihaz_parmak_izi_cesitliligi`, `sentetik_kimlik_olasiligi`
- **Jupyter Notebook (`gun_121_sentetik_kimlik_dolandiricilik.ipynb`):**
  1. Gerçek ve sahte kimlik bilgilerinin harmanlanmasıyla oluşturulan "Frankenstein" hesapların incelenmesi
  2. Ortak telefon, e-posta, IP ve cihaz kullanan sahte hesap kümelerinin tespiti
  3. MASAK ve BDDK regülasyonlarına uygun riskli hesap blokajı
- **Mülakat Sorusu:** Gerçek çalınmış kimlik (Stolen Identity) ile parça parça üretilmiş Sentetik Kimlik (Synthetic Identity) arasındaki davranışsal farklar nelerdir?

### Gün 122: Paycell P2P Transfer Anomali Tespiti
- **İş Alanı:** Cüzdandan Cüzdana Transfer Masası & Şüpheli Akış İzleme
- **Veri Kaynağı:** [Kaggle - PaySim Financial Transactions](https://www.kaggle.com/datasets)
- **Model:** Autoencoder + Mahalanobis Distance
- **Türkçe Değişkenler:** `gonderici_hesap_yasi_gun`, `alici_hesap_yasi_gun`, `transfer_saati`, `transfer_tutari_tl`, `anomali_skoru`
- **Jupyter Notebook (`gun_122_p2p_transfer_anomali_tespiti.ipynb`):**
  1. Kullanıcıların alışılmış transfer saatleri, kişi listesi ve tutar dağılımlarının modellenmesi
  2. Gece yarısı yeni açılmış bir hesaba yapılan sıra dışı yüksek tutarlı transferlerin anomali tespiti
  3. Transfer tutarını geçici askıya alıp müşteriden SMS/BiP ile çift onay isteme
- **Mülakat Sorusu:** Tek değişkenli Z-Score anomali testi yerine çok değişkenli Mahalanobis Distance veya Autoencoder kullanmanın avantajı nedir?

### Gün 123: Çalıntı Kart Harcama Hızı (Velocity Fraud) Modeli
- **İş Alanı:** Paycell Kart Güvenliği & Seri Küçük Harcama Dedektörü
- **Veri Kaynağı:** [Credit Card Velocity Attack Telemetry](https://www.kaggle.com/datasets)
- **Model:** Kayan Pencere (Sliding Window) + LightGBM Classifier
- **Türkçe Değişkenler:** `kart_id`, `son_5dk_islem_sayisi`, `son_1saat_toplam_harcama`, `farkli_uye_isyeri_sayisi`, `kart_bloke_edilsin_mi`
- **Jupyter Notebook (`gun_123_harcama_hizi_velocity_fraud.ipynb`):**
  1. Kredi kartı çalındığında dolandırıcıların kartın limitini test etmek için yaptığı seri küçük harcamaların (Card Testing) analizi
  2. 1, 5 ve 15 dakikalık pencerelerde hesaplanan işlem frekans metrikleri
  3. Hızlı harcama saldırısı (Velocity Attack) anında kartı otomatik geçici kullanıma kapatma
- **Mülakat Sorusu:** Gerçek zamanlı akış işleme motorlarında (Apache Flink / Spark Streaming) kayan pencere (Sliding Window) özniteliklerini düşük gecikmeyle nasıl hesaplarız?

### Gün 124: Üye İşyeri (Merchant) Günlük Ciro Tahminleme Modeli
- **İş Alanı:** Paycell Üye İşyeri Hizmetleri & Erken Finansman / POS Kredisi
- **Veri Kaynağı:** [Merchant Daily Revenue Time Series Dataset](https://www.kaggle.com/datasets)
- **Model:** Prophet + LightGBM Hibrit Zaman Serisi Modeli
- **Türkçe Değişkenler:** `uye_isyeri_id`, `gunluk_islem_adedi`, `gunluk_ciro_tl`, `sektor_trend_katsayisi`, `gelecek_7gun_tahmini_ciro`
- **Jupyter Notebook (`gun_124_uye_isyeri_ciro_tahmini.ipynb`):**
  1. Restoran, market ve e-ticaret üye işyerlerinin mevsimsel ve haftalık ciro dinamiklerinin modellenmesi
  2. Gelecek hafta ve ayın günlük cirosunun tahmini
  3. Ciroya dayalı Paycell POS Kredisi tekliflerinin otomatik oluşturulması
- **Mülakat Sorusu:** Zaman serisinde tatil günleri ve özel gün (Black Friday / Anneler Günü) etkilerini Prophet tatil parametreleriyle nasıl yönetirsiniz?

### Gün 125: Otomatik Fatura Talimatı İptal Riski Puanlama
- **İş Alanı:** Fatura Tahsilat Masası & Doğrudan Borçlandırma Sistemi (DBS)
- **Veri Kaynağı:** [Auto-Debit & Recurring Payment Cancellation Data](https://www.kaggle.com/datasets)
- **Model:** XGBoost Classifier + SHAP Analizi
- **Türkçe Değişkenler:** `musteri_id`, `fatura_turu_su_elektrik_turkcell`, `gecmis_talimat_yasi_ay`, `son_3ayda_bakiye_yetersiz_sayisi`, `iptal_etme_riski`
- **Jupyter Notebook (`gun_125_otomatik_fatura_iptal_riski.ipynb`):**
  1. Otomatik ödeme talimatının peş peşe provizyon alamama (bakiye yetersiz) loglarının incelenmesi
  2. Müşterinin talimatı kaldırma ve manuel ödemeye dönme riskinin modellenmesi
  3. Talimat iptali gerçekleşmeden önce müşteriye Paycell Hazır Limit veya yedek kart tanımlama fırsatı sunulması
- **Mülakat Sorusu:** Tekrarlayan abonelik ve talimat verilerinde "Sağdan Sansürleme" (Right-Censored Data) problemine karşı model validasyonu nasıl kurulmalıdır?

### Gün 126: Kurumsal Şirket Hatları Harcama Limiti Dinamik Hesaplayıcı
- **İş Alanı:** Turkcell Kurumsal Müşteri Yönetimi & B2B Risk Masası
- **Veri Kaynağı:** [B2B Telecom Expense & Usage Dataset](https://www.kaggle.com/datasets)
- **Model:** Ridge Regresyonu + K-Means Kurumsal Şirket Kümeleme
- **Türkçe Değişkenler:** `sirket_sektoru`, `sirket_calisan_hat_sayisi`, `gecmis_aylik_toplam_fatura`, `onerilen_dinamik_harcama_limiti`
- **Jupyter Notebook (`gun_126_kurumsal_harcama_limiti.ipynb`):**
  1. Kurumsal firmaların çalışanlarına tahsis ettiği hatların yurt dışı arama ve veri dolaşım harcamalarının modellenmesi
  2. Şirket cirosu ve hat adedine göre aşırı fatura riskini önleyen dinamik harcama limitlerinin hesaplanması
  3. Limit aşımında kurumsal filo yöneticisine anlık onay ekranı iletilmesi
- **Mülakat Sorusu:** Çok değişkenli doğrusal regresyonda aşırı çoklu doğrusal bağlantı (Multicollinearity) durumunda neden Ridge veya Lasso regülarizasyonu kullanılır?

### Gün 127: Dijital Altın / Döviz Alım-Satım Eğilimi Tahmini
- **İş Alanı:** Paycell Yatırım & Emtia / Döviz Hizmetleri
- **Veri Kaynağı:** [Retail FX & Precious Metal Trading Behavior](https://www.kaggle.com/datasets)
- **Model:** Random Forest Classifier + Makroekonomik Göstergeler
- **Türkçe Değişkenler:** `kullanici_cuzdan_bakiyesi`, `altin_fiyat_volatilitesi_7g`, `kullanici_onceki_alim_sayisi`, `altin_doviz_alma_olasiligi`
- **Jupyter Notebook (`gun_127_dijital_altin_alim_egilimi.ipynb`):**
  1. Piyasa altın/dolar kuru hareketleri ile Paycell cüzdan kullanıcılarının alım-satım eğilimlerinin analizi
  2. Maaş dönemlerinde yatırım yapmaya meyilli kullanıcı kitlelerinin tespiti
  3. Uygun kur anında kişiselleştirilmiş "Komisyonsuz Altın Al" bildirimlerinin gönderilmesi
- **Mülakat Sorusu:** Kullanıcı yatırım davranışını modellerken piyasa fiyatlarını modele doğrudan mutlak değer olarak vermek yerine neden logaritmik getiri (Log Return) verilir?

### Gün 128: Fiziksel POS Terminal Donanım Arıza Öngörüsü
- **İş Alanı:** Paycell POS Donanım Operasyonları & Saha Değişim Lojistiği
- **Veri Kaynağı:** [POS Hardware Telemetry & Failure Logs](https://www.kaggle.com/datasets)
- **Model:** Weibull Yaşam Dağılımı / Survival Analysis + CatBoost
- **Türkçe Değişkenler:** `pos_cihaz_modeli`, `gunluk_islem_adedi`, `kart_okuma_hata_orani`, `pil_voltaji`, `arizalanma_gunu_tahmini`
- **Jupyter Notebook (`gun_128_pos_donanim_ariza_ongorusu.ipynb`):**
  1. POS cihazlarından toplanan batarya voltajı, manyetik/çip okuma hata logları ve tuş takımı telemetrilerinin analizi
  2. Cihazın tamamen bozulmadan önceki arıza sinyallerinin tespiti
  3. Üye işyeri mağdur olmadan önce yeni POS cihazının kargo ile önceden sevk edilmesi
- **Mülakat Sorusu:** Donanım arıza tahmininde Yaşam Analizi (Survival Analysis) ve Weibull Dağılımının MTBF (Mean Time Between Failures) hesaplamasındaki rolü nedir?

### Gün 129: Harcama Lokasyonu ile Hücresel Konum Uyuşmazlığı Modeli
- **İş Alanı:** Paycell & GSM Şebeke Ortak Güvenlik Masası (Geo-Fraud)
- **Veri Kaynağı:** [Telecom Geolocation & POS Transaction Mismatch Data](https://www.kaggle.com/datasets)
- **Model:** Haversine Coğrafi Mesafe Formülü + Gradient Boosting
- **Türkçe Değişkenler:** `musteri_baz_istasyon_konumu`, `pos_cihazi_sehir_kordinati`, `aradaki_kusucusu_mesafe_km`, `klon_kart_riski`
- **Jupyter Notebook (`gun_129_harcama_hucresel_konum_uyusmazligi.ipynb`):**
  1. Fiziksel kartla harcama yapılan POS terminalinin koordinatları ile abonenin o an bağlı olduğu baz istasyonu koordinatlarının karşılaştırılması
  2. Müşteri İstanbul'dayken kartın İzmir'de fiziksel olarak çekilmesi durumunun milisaniyeler içinde tespiti
  3. Klon kart kopyalama saldırılarına karşı işlemi anında reddetme ve SMS ile uyarma
- **Mülakat Sorusu:** Coğrafi dolandırıcılık tespitinde GPS kapalıyken operatör baz istasyonu üçgenleme (Cell-Tower Triangulation) konum doğruluğu ve tolerans marjı nasıl hesaplanır?

### Gün 130: Sadakat Puanı / Hediye Bakiye Suiistimal Dedektörü
- **İş Alanı:** Paycell & Turkcell Bizbize Sadakat Programı Güvenliği
- **Veri Kaynağı:** [Loyalty Points Abuse & Cashback Fraud](https://www.kaggle.com/datasets)
- **Model:** K-Means + Mahalanobis Distance Anomali Skoru
- **Türkçe Değişkenler:** `kullanici_id`, `kazanilan_puan_adedi`, `puan_harcama_hizi`, `iptal_iade_orani`, `suiistimal_etiketi`
- **Jupyter Notebook (`gun_130_sadakat_puani_suiistimal_tespiti.ipynb`):**
  1. Alışveriş yapıp hediye puan kazandıktan sonra siparişi iptal eden organize kullanıcıların tespiti
  2. Çoklu hesaplar üzerinden puan birleştirme ve nakde çevirme paternlerinin modellenmesi
  3. Haksız kazanç sağlayan hesapların puan transferlerinin dondurulması
- **Mülakat Sorusu:** Sadakat programlarında "Puan Arbitrajı" (Point Arbitrage) yapan bot ağlarını tespit etmek için hangi Graf ve Kümeleme algoritmaları birlikte kullanılır?

---

## 📱 Modül 11: Dijital Servisler (TV+, fizy, lifebox, BiP, Dergilik) (Gün 131 – 145)

### Gün 131: fizy Şarkı Benzerliği Vektör Arama Motoru
- **İş Alanı:** fizy Müzik Servisi & Benzer Şarkıyı Çal (Song Radio) Özelliği
- **Veri Kaynağı:** [Kaggle - Spotify 1.2M+ Songs Audio Features & Embeddings](https://www.kaggle.com/datasets)
- **Model:** CLMR (Contrastive Learning of Musical Representations) + FAISS (Facebook AI Similarity Search)
- **Türkçe Değişkenler:** `sarki_vektoru_128d`, `akustik_dans_edilebilirlik_valans`, `sorgu_sarki_id`, `en_yakin_10_sarki_id`
- **Jupyter Notebook (`gun_131_fizy_vektor_arama_faiss.ipynb`):**
  1. Şarkıların ses dalgalarından ve müzikal özniteliklerinden 128 boyutlu yoğun vektörler (Embeddings) çıkarma
  2. Milyonlarca şarkı vektörünün FAISS HNSW (Hierarchical Navigable Small World) indeksine yüklenmesi
  3. Çalan bir şarkıya akustik ve janr olarak en yakın parçaların <5 milisaniyede bulunup sonsuz radyo akışı oluşturulması
- **Mülakat Sorusu:** Milyonlarca vektör arasında arama yaparken k-NN brute-force arama yerine FAISS IVF-PQ (Inverted File Product Quantization) kullanmanın bellek ve hız farkı nedir?

### Gün 132: TV+ Video Başlangıç Gecikmesi (Startup Latency) Regresyonu
- **İş Alanı:** TV+ Video Oynatıcı Mühendisliği & QoE Optimizasyonu
- **Veri Kaynağı:** [Video Player Startup Latency & CDN Telemetry](https://www.kaggle.com/datasets)
- **Model:** LightGBM Regressor + Optuna
- **Türkçe Değişkenler:** `cdn_sunucu_id`, `istemci_cihaz_turu_smart_tv_mobil`, `ag_turu_fiber_wifi_4g`, `video_baslama_suresi_ms`
- **Jupyter Notebook (`gun_132_tv_plus_video_baslama_gecikmesi.ipynb`):**
  1. Kullanıcı "Oynat" butonuna bastıktan sonra ilk video karesinin ekrana gelme süresinin (Time-to-First-Frame) analizi
  2. CDN sunucu gecikmesi, TLS el sıkışma süresi ve ilk segment boyutunun gecikmeye etkisinin modellenmesi
  3. Gecikmeyi düşürmek için en uygun CDN rotasının dinamik seçilmesi
- **Mülakat Sorusu:** Video oynatıcılarda "Initial Chunk Pre-fetching" ve HLS LL-HLS (Low Latency HLS) protokollerinin başlangıç süresine etkisi nedir?

### Gün 133: lifebox Fotoğraf Otomatik Albümleme & Sahne Sınıflandırma
- **İş Alanı:** lifebox Bulut Depolama & Akıllı Fotoğraf Gruplama
- **Veri Kaynağı:** [Places365 / ImageNet Scene Recognition Dataset](http://places2.csail.mit.edu/)
- **Model:** ConvNeXt-Tiny / ResNet-50 Fine-Tuning
- **Türkçe Değişkenler:** `kullanici_fotografi`, `sahne_etiketi_plaj_dag_dogum_gunu_dugun`, `guven_orani`
- **Jupyter Notebook (`gun_133_lifebox_sahne_siniflandirma.ipynb`):**
  1. Kullanıcının lifebox'a yüklediği fotoğrafların sahne ve mekan içeriklerinin tespiti
  2. "Yaz Tatili", "Doğum Günü", "Konserler", "Doğa & Manzara" otomatik albümlerinin oluşturulması
  3. Akıllı arama çubuğunda "Plaj fotoğraflarımı göster" sorgularına anında yanıt üretme
- **Mülakat Sorusu:** Kullanıcı fotoğraflarının gizliliği için bulutta ağır modeller çalıştırmak yerine mobil cihaz üzerinde (On-Device AI) CoreML / TensorFlow Lite ile çıkarım yapmanın avantajları nelerdir?

### Gün 134: TV+ İzleyici Diziyi Bırakma (Drop-off) Dakikası Tahmini
- **İş Alanı:** TV+ İçerik Satın Alma & Senaryo / Kurgu İzleyici Tutundurma
- **Veri Kaynağı:** [OTT Video Viewing Session & Drop-off Data](https://www.kaggle.com/datasets)
- **Model:** Yaşam Analizi (Cox Proportional Hazards) + Survival Tree
- **Türkçe Değişkenler:** `icerik_id`, `izleyici_toplam_izleme_dakikasi`, `dizi_bolum_uzunlugu_dk`, `birakma_olasiligi_egrisi`
- **Jupyter Notebook (`gun_134_tv_plus_dizi_birakma_dakikasi.ipynb`):**
  1. Dizinin hangi dakikalarında izleyicilerin sıkılıp videoyu kapattığının (Drop-off Curve) analizi
  2. Dizinin ilk 15 dakikasında bırakılma riskini artıran tempo, sahne türü ve ses dinamiklerinin incelenmesi
  3. İçerik öneri motoruna sadece tamamlanma oranı yüksek dizilerin öncelikli beslenmesi
- **Mülakat Sorusu:** Video izleme verilerinde videoyu yarıda kapatanlar ile sonuna kadar izleyenleri "Right-Censored" olarak Cox PH modelinde nasıl formüle ederiz?

### Gün 135: fizy Çalma Listesi Devam Ettirme Modeli
- **İş Alanı:** fizy Otomatik Çalma Listesi Tamamlama (Playlist Continuation)
- **Veri Kaynağı:** [Kaggle - Spotify Million Playlist Continuation Challenge](https://www.kaggle.com/datasets)
- **Model:** Word2Vec (Item2Vec / Song2Vec) + İki Kuleli Derin Öğrenme (Two-Tower DSSM)
- **Türkçe Değişkenler:** `calma_listesi_id`, `mevcut_sarki_listesi`, `onerilen_siradaki_sarkilar`, `sarki_vektor_benzerligi`
- **Jupyter Notebook (`gun_135_fizy_playlist_devam_ettirme.ipynb`):**
  1. Kullanıcının oluşturduğu 5-10 şarkılık çalma listesinin müzikal temasının öğrenilmesi
  2. Item2Vec ile aynı çalma listesinde sıkça birlikte yer alan şarkıların vektör temsillerinin çıkarılması
  3. Çalma listesinin sonuna otomatik olarak akışı bozmayacak 10 yeni şarkı önerilmesi
- **Mülakat Sorusu:** NLP'deki Word2Vec (Skip-gram) algoritması müzik çalma listelerine nasıl uyarlanır ve pencere boyutu (Context Window) neyi ifade eder?

### Gün 136: lifebox Tekrarlanan / Kopya Fotoğrafları Temizleme
- **İş Alanı:** lifebox Depolama Alanı Tasarrufu & Galeri Temizleme Asistanı
- **Veri Kaynağı:** [Kaggle - Duplicate Image Detection & Near-Duplicates](https://www.kaggle.com/datasets)
- **Model:** Algısal Karma (Perceptual Hashing - pHash / dHash) + CNN Embedding Mesafesi
- **Türkçe Değişkenler:** `fotograf_1_hash`, `fotograf_2_hash`, `hamming_mesafesi`, `benzerlik_orani`, `silme_onerisi`
- **Jupyter Notebook (`gun_136_lifebox_kopya_fotograf_temizleme.ipynb`):**
  1. Seri çekim (Burst Mode) ile peş peşe çekilmiş neredeyse aynı fotoğrafların pHash ile tespiti
  2. Hamming mesafesi < 5 olan fotoğrafların kümelenmesi
  3. En net, gözleri açık ve odaklanmış en iyi karenin seçilip diğerlerinin silinmesi için kullanıcıya önerilmesi
- **Mülakat Sorusu:** Kriptografik özetleme (MD5/SHA256) ile Algısal Özetleme (pHash) arasındaki temel fark nedir ve fotoğraf boyutlandırıldığında pHash neden değişmez?

### Gün 137: Dergilik / Dijital Yayın İlgi Alanı Kişiselleştirme
- **İş Alanı:** Turkcell Dergilik & Dijital Gazete / Makale Öneri Akışı
- **Veri Kaynağı:** [Microsoft News Dataset (MIND) / News Recommendation](https://msnews.github.io/)
- **Model:** NRMS (Neural News Recommendation with Multi-Head Self-Attention) + BERTurk
- **Türkçe Değişkenler:** `kullanici_okuma_gecmisi`, `haber_basligi_metni`, `kategori_ekonomi_spor_teknoloji`, `tiklama_tahmin_skoru_ctr`
- **Jupyter Notebook (`gun_137_dergilik_haber_makale_onerisi.ipynb`):**
  1. Haber başlıkları ve özetlerinin Multi-Head Self-Attention ile haber vektörlerine dönüştürülmesi
  2. Kullanıcının geçmişte okuduğu haberlerden anlık ilgi alanı profilinin çıkarılması
  3. Dergilik ana sayfasında kişiselleştirilmiş "Sizin İçin Seçtiklerimiz" haber akışı sıralaması
- **Mülakat Sorusu:** Haber öneri sistemlerinde haberlerin çok hızlı eskimesi (Cold-Start & Freshness) sorununu NRMS modeli nasıl çözer?

### Gün 138: TV+ Altyazı ve Ses Senkronizasyon Kayması Dedektörü
- **İş Alanı:** TV+ İçerik Kalite Kontrol (QC) & Altyazı Uyumu
- **Veri Kaynağı:** [Video Audio Subtitle Synchronization Dataset](https://www.kaggle.com/datasets)
- **Model:** Whisper ASR (Ses Transkripsiyonu) + Levenshtein / DTW Hizalama
- **Türkçe Değişkenler:** `video_ses_izlek`, `altyazi_srt_dosyasi`, `ses_zamani_sn`, `altyazi_zamani_sn`, `senkron_kayma_ms`
- **Jupyter Notebook (`gun_138_tv_plus_altyazi_senkron_kaymasi.ipynb`):**
  1. Videonun Türkçe ses kanalından Whisper ile otomatik zaman damgalı konuşma metni çıkarma
  2. SRT altyazı dosyası ile çıkarılan metnin Dynamic Time Warping (DTW) ile hizalanması
  3. 500 ms'den fazla kayma olan bozuk altyazıların yayına girmeden önce otomatik düzeltilmesi
- **Mülakat Sorusu:** Altyazı metni ile ASR metni arasındaki çeviri/ifade farklılıklarına rağmen zaman hizalamasını doğru yapmak için hangi fonetik hizalama algoritmaları kullanılır?

### Gün 139: BiP Sesli Mesaj Arka Plan Gürültüsü Temizleme
- **İş Alanı:** BiP Anlık Mesajlaşma & Sesli Mesaj Netleştirme
- **Veri Kaynağı:** [VoiceBank-DEMAND / Speech Enhancement Dataset](https://datasync.ed.ac.uk/)
- **Model:** Demucs / Wave-U-Net 1D Ses Dalgası Ayrıştırma
- **Türkçe Değişkenler:** `ham_sesli_mesaj_wav`, `tahmini_gurultu_sinyali`, `netlestirilmis_ses_wav`, `pesq_skoru`
- **Jupyter Notebook (`gun_139_bip_sesli_mesaj_netlestirme.ipynb`):**
  1. Rüzgarlı havada, sokakta veya metroda kaydedilmiş gürültülü sesli mesajların toplanması
  2. Wave-U-Net ile doğrudan zaman düzleminde insan sesi ile ortam gürültüsünün ayrıştırılması
  3. BiP oynatıcısına "Gürültüyü Azalt" butonu entegrasyonu ile kristal netliğinde ses iletimi
- **Mülakat Sorusu:** Zaman-Frekans düzleminde (STFT Spektrogram) gürültü temizleme ile doğrudan Dalga Formu (Time-Domain Waveform) temizleme arasındaki faz hatası farkı nedir?

### Gün 140: TV+ Canlı Yayın Eşzamanlı İzleyici (CCU) Yük Tahmini
- **İş Alanı:** TV+ Canlı Maç Yayınları & CDN Sunucu Kapasite Planlama
- **Veri Kaynağı:** [Live Streaming CCU & Viewership Time Series Data](https://www.kaggle.com/datasets)
- **Model:** SARIMAX + Derbi/Maç Başlama Saati Dışsal Regresörleri
- **Türkçe Değişkenler:** `kanal_id`, `canli_yayin_turu_derbi_haber_dizi`, `tarih_saat`, `anlik_es_zamanli_izleyici_ccu`, `tahmini_gerekli_cdn_gbps`
- **Jupyter Notebook (`gun_140_tv_plus_canli_yayin_ccu_tahmini.ipynb`):**
  1. Süper Lig derbi maçları öncesinde izleyici sayısındaki ani patlamanın (Traffic Spike) modellenmesi
  2. Maçın 15 dakika öncesi ve devre arasında oluşacak tepe izleyici (Peak CCU) tahmini
  3. Sunucu çökmesini ve donmaları önlemek için CDN bant genişliğinin önceden otomatik ölçeklenmesi (Auto-Scaling)
- **Mülakat Sorusu:** Canlı yayın izleyici tahmininde geçmiş trendlerin ötesinde "Maçtaki gol anı" veya "Kırmızı kart" gibi anlık olayların yarattığı trafik sıçramaları nasıl yönetilir?

### Gün 141: fizy Podcast Açıklamalarından Otomatik Kategori Çıkarma
- **İş Alanı:** fizy Podcast Kataloğu Düzenleme & Arama Keşfi
- **Veri Kaynağı:** [Spotify Podcast Descriptions & Transcripts](https://huggingface.co/datasets)
- **Model:** BERTurk Text Classification / Zero-Shot NLI Classifier
- **Türkçe Değişkenler:** `podcast_basligi`, `bolum_aciklama_metni`, `tahmini_ana_kategori_teknoloji_tarih_spor_psikoloji`, `etiket_olasiligi`
- **Jupyter Notebook (`gun_141_fizy_podcast_kategori_siniflandirma.ipynb`):**
  1. Podcast yayıncılarının girdiği serbest metin açıklamaların temizlenmesi
  2. BERTurk ile 20 farklı ana podcast kategorisine çok sınıflı (Multi-Class) atama
  3. fizy Keşfet sekmesinde podcastlerin doğru konu başlıkları altında listelenmesi
- **Mülakat Sorusu:** Yeni bir podcast kategorisi eklendiğinde modeli yeniden eğitmeden sınıflandırma yapmak için Zero-Shot Classification (NLI Entailment) nasıl kullanılır?

### Gün 142: lifebox Belge/Fiş Tarama Otomatik Perspektif Düzeltme
- **İş Alanı:** lifebox Belge Tarayıcı & Fatura / Sözleşme Arşivleme
- **Veri Kaynağı:** [Roboflow - Document Edge Detection & Dewarping](https://universe.roboflow.com/)
- **Model:** YOLOv8-Pose (4 Köşe Tespiti) + OpenCV Perspektif Dönüşümü (WarpPerspective)
- **Türkçe Değişkenler:** `egik_cekilmis_belge_fotografi`, `tespit_edilen_4_kose_noktasi`, `duzlestirilmis_a4_ciktisi`
- **Jupyter Notebook (`gun_142_lifebox_belge_perspektif_duzeltme.ipynb`):**
  1. Masanın üzerinde açılı çekilmiş fiş veya sözleşmenin 4 köşe noktasının YOLOv8-Pose ile tespiti
  2. OpenCV `getPerspectiveTransform` ile görüntünün taranmış gibi A4 formatında düzeltilmesi
  3. Kontrast artırma ve gölge temizleme filtreleri uygulayarak okunabilir PDF üretme
- **Mülakat Sorusu:** Köşe tespiti yaparken klasik Hough Transform köşe bulma yerine Derin Öğrenme Pose Estimation kullanmanın buruşuk ve gölgeli kağıtlardaki üstünlüğü nedir?

### Gün 143: Game+ Bulut Oyun Paket Kaybı Tolerans ve Bitrate Ayarlayıcı
- **İş Alanı:** Turkcell Game+ (GeForce NOW) Oyun Deneyimi Mühendisliği
- **Veri Kaynağı:** [Cloud Gaming WebRTC Network Telemetry](https://www.kaggle.com/datasets)
- **Model:** Bulanık Mantık (Fuzzy Logic Controller) / Karar Ağacı
- **Türkçe Değişkenler:** `anlik_paket_kaybi_yuzdesi`, `rtt_gecikme_ms`, `ekran_cozunurlugu_fps`, `dinamik_bitrate_kbps`
- **Jupyter Notebook (`gun_143_game_plus_bitrate_kontrol.ipynb`):**
  1. Mobil veya ev internetinde anlık paket kaybı (%2 - %5) yaşandığı anların tespiti
  2. Oyunda takılma (Stuttering) olmaması için video bitrate'ini ve FEC (Forward Error Correction) oranını dinamik ayarlama
  3. Ağ toparlandığında gecikmesiz olarak 1080p 60 FPS kalitesine geri dönme
- **Mülakat Sorusu:** Bulut oyunda WebRTC GCC (Google Congestion Control) algoritmasının Delay-Based ve Loss-Based kontrol mekanizmaları nasıl çalışır?

### Gün 144: TV+ Dizi/Film Fragman Heyecan Puanlaması
- **İş Alanı:** TV+ Pazarlama & En Çarpıcı Fragman Karesini Seçme
- **Veri Kaynağı:** [Video Trailer Action & Emotion Arousal Dataset](https://www.kaggle.com/datasets)
- **Model:** 3D-CNN (Video ResNet - R(2+1)D) + Akustik Enerji Birleşimi
- **Türkçe Değişkenler:** `video_kesiti_10sn`, `gorsel_hareket_skoru`, `ses_patlama_enerjisi`, `heyecan_arousal_puani_0_100`
- **Jupyter Notebook (`gun_144_tv_plus_fragman_heyecan_puani.ipynb`):**
  1. Fragmandaki aksiyon, kamera hareket hızı ve müzik yükselişlerinin multimodal analizi
  2. Fragmanın en yüksek heyecan (Arousal) ve dikkat çeken 5 saniyelik kısmının tespiti
  3. TV+ ana sayfa afişinde otomatik oynatılacak hareketli önizleme (Preview GIF/Video) üretimi
- **Mülakat Sorusu:** Video işlemede 2D-CNN + LSTM yerine 3D-CNN (Spatiotemporal Convolutions) kullanmanın hareket dinamiklerini yakalamadaki farkı nedir?

### Gün 145: fizy Türkçe Şarkı Sözlerinden Ruh Hali (Mood) Çıkarımı
- **İş Alanı:** fizy Mod Listeleri (Mutlu, Hüzünlü, Enerjik, Sakin, Odaklanma)
- **Veri Kaynağı:** [Turkish Song Lyrics & Mood Dataset](https://huggingface.co/datasets)
- **Model:** BERTurk + Focal Loss (Dengesiz Duygu Sınıfları)
- **Türkçe Değişkenler:** `sarki_adi`, `turkce_sarki_sozleri`, `tahmini_ruh_hali_huzun_neseli_romantik_ofke`, `valans_arousal_skoru`
- **Jupyter Notebook (`gun_145_fizy_sarki_sozu_ruh_hali.ipynb`):**
  1. Türkçe şarkı sözlerindeki edebi anlatım, metafor ve duygusal tonların BERTurk ile modellenmesi
  2. Şarkıların "Hüzünlü Akşamlar", "Motivasyon & Spor", "Aşk Şarkıları" tematik listelerine atanması
  3. Kullanıcının o anki ruh haline uygun şarkı arama motorunun desteklenmesi
- **Mülakat Sorusu:** Metin tabanlı müzik duygu analizinde (Valence-Arousal) akustik müzik tonu ile şarkı sözlerinin zıt olduğu (örneğin hüzünlü sözlerin hareketli ritimle söylenmesi) durumlar nasıl çözülür?

---

## 🤖 Modül 12: İleri Seviye NLP, LLM, Müşteri Deneyimi & Agentic AI (Gün 146 – 160)

### Gün 146: Müşteri Temsilcisi Yanıt Kalitesi Denetleyicisi (LLM-as-a-Judge)
- **İş Alanı:** Turkcell Global Bilgi Kalite Güvence & Temsilci Değerlendirme
- **Veri Kaynağı:** [Customer Support Conversation Quality Dataset](https://huggingface.co/datasets)
- **Model:** Yerel Qwen2.5-7B-Instruct / LLaMA-3.2-3B (Ollama / HuggingFace Yerel) + G-Eval Kalite Rubriği
- **Türkçe Değişkenler:** `cagri_transkripti`, `temsilci_cevabi`, `degerlendirme_kriteri_empati_cozum_nezaket`, `kalite_puani_1_5`, `gerekce_metni`
- **Jupyter Notebook (`gun_146_llm_as_a_judge_kalite_denetimi.ipynb`):**
  1. Çağrı merkezi görüşme metinlerinin LLM'e değerlendirme rubriği (Empati, Doğru Bilgi, Çözüm Hızı) ile beslenmesi
  2. G-Eval metodu ile zincirleme düşünme (Chain-of-Thought) kullanılarak 1-5 arası objektif puanlama
  3. Temsilcilere gelişim alanlarına dair otomatik koçluk geri bildirimi üretimi
- **Mülakat Sorusu:** LLM-as-a-Judge sistemlerinde "Position Bias" (ilk verilen cevabı kayırma) ve "Verbosity Bias" (uzun cevaba yüksek puan verme) sorunları nasıl hafifletilir?

### Gün 147: Telekom Terimleri için Alana Özel (Domain-Specific) Word2Vec / FastText
- **İş Alanı:** Turkcell Ar-Ge & Alana Özel NLP Altyapısı
- **Veri Kaynağı:** [Turkcell Help Center, FAQ & Forum Texts](https://www.turkcell.com.tr/)
- **Model:** FastText (Subword Embeddings) / Word2Vec CBOW
- **Türkçe Değişkenler:** `telekom_terimi_voenabler_apn_taahhut`, `en_yakin_kelimeler`, `vektor_uzayi_gorsellestirme`
- **Jupyter Notebook (`gun_147_telekom_fasttext_kelime_vektorleri.ipynb`):**
  1. Milyonlarca telekom yardım sayfası, tarife detayı ve kullanıcı forum yazılarının toplanması
  2. Alt-kelime (Subword) bilgisi kullanan FastText modeli ile "faturamdaki", "taahhütsüz", "APN" gibi sektörel kelimelerin eğitilmesi
  3. Yazım hatalı yazılan telekom terimlerinin en yakın doğru anlamsal karşılığının bulunması
- **Mülakat Sorusu:** Türkçe gibi sondan eklemeli dillerde kelime bazlı Word2Vec yerine karakter n-gram tabanlı FastText kullanmanın OOV (Out-of-Vocabulary) kelimelerdeki avantajı nedir?

### Gün 148: Fatura PDF'lerinden Yapılandırılmış JSON Çıkaran LLM Ajanı
- **İş Alanı:** Kurumsal Müşteri Masası & Otomatik Fatura İçe Aktarma
- **Veri Kaynağı:** [Kaggle - Telecom Invoice PDF/Text Dataset](https://www.kaggle.com/datasets)
- **Model:** LangChain / LlamaIndex + Pydantic Structured Output Parser
- **Türkçe Değişkenler:** `fatura_metni`, `abone_no`, `fatura_kesim_tarihi`, `odenecek_tutar_tl`, `kdv_oiv_vergileri_json`
- **Jupyter Notebook (`gun_148_fatura_pdf_json_llm_ajani.ipynb`):**
  1. Karmaşık telekom fatura PDF'lerinden metin ve tablo bloklarının çıkarılması
  2. Pydantic şeması ile zorunlu alanların (Fatura No, Vergi Kalemleri, Paket Aşım Ücreti) tanımlanması
  3. LLM Function Calling ile sıfır şema hatasıyla doğrulanmış JSON çıktısı üretimi
- **Mülakat Sorusu:** LLM çıktılarının JSON formatında kesin ve hatasız gelmesini garanti altına almak için JSON Mode ve Grammar-based Sampling nasıl çalışır?

### Gün 149: Canlı Sohbet Müşteri Sinir Seviyesi (Frustration) İzleyici
- **İş Alanı:** BiP & Turkcell Web Canlı Destek Masası
- **Veri Kaynağı:** [Customer Support Live Chat Frustration Traces](https://huggingface.co/datasets)
- **Model:** BERTurk + Temporal Attention (Mesaj Sırası Ağırlandırma)
- **Türkçe Değişkenler:** `oturum_mesaj_listesi`, `buyuk_harf_unlem_orani`, `cevap_bekleme_suresi_sn`, `sinirlilik_skoru_0_100`
- **Jupyter Notebook (`gun_149_canli_sohbet_sinir_seviyesi.ipynb`):**
  1. Sohbet esnasında müşterinin yazdığı ardışık mesajlardaki öfke artışının takibi
  2. BÜYÜK HARF kullanımı, tekrarlayan soru sorma ve botun anlayamaması kaynaklı sinir seviyesi tahmini
  3. Sinir skoru %75'i geçtiğinde botun devreden çıkıp sohbeti anında kıdemli müşteri temsilcisine aktarması (Human-in-the-Loop)
- **Mülakat Sorusu:** Canlı sohbette tek bir cümlenin bağımsız duygu analizi ile tüm konuşma akışının kümülatif sinir seviyesi (Contextual Frustration) arasındaki fark nasıl modellenir?

### Gün 150: Şikayet Metinlerinden Kök Neden Hiyerarşisi Çıkarma
- **İş Alanı:** Şikayet Yönetimi & Operasyonel Hata Analizi (Sikayetvar / 532)
- **Veri Kaynağı:** [Kaggle - Turkish Telecom Complaints Dataset](https://www.kaggle.com/datasets)
- **Model:** Hiyerarşik Metin Sınıflandırma (Hierarchical BERTurk / SetFit)
- **Türkçe Değişkenler:** `sikayet_metni`, `ana_kategori_sebeke_fatura_kampanya`, `alt_kategori_cekmiyor_fatura_asimi`, `kok_neden_kodu`
- **Jupyter Notebook (`gun_150_sikayet_kok_neden_hiyerarsisi.ipynb`):**
  1. Şikayet metinlerinin 3 seviyeli hiyerarşik taksonomiye (Ana Alan -> Alt Problem -> Kök Neden) göre etiketlenmesi
  2. Seviyeli sınıflandırıcı (Local Classifier per Parent Node) mimarisi ile uçtan uca tahmin
  3. Şebeke arızası kaynaklı şikayetlerin doğrudan ilgili bölge saha operasyon birimine yönlendirilmesi
- **Mülakat Sorusu:** Düz çok sınıflı (Flat Multi-Class) model yerine Hiyerarşik Sınıflandırma kullanmanın sınıflar arası mantıksal tutarlılığa katkısı nedir?

### Gün 151: RAG için Hibrit Vektör + BM25 Arama Motoru (Hybrid Search)
- **İş Alanı:** Turkcell İntranet Bilgi Bankası & Müşteri Asistanı RAG
- **Veri Kaynağı:** [Turkcell Knowledge Base & Help Articles](https://www.turkcell.com.tr/)
- **Model:** BM25 (Sparse) + BGE-M3 / BERTurk Embeddings (Dense) + Reciprocal Rank Fusion (RRF)
- **Türkçe Değişkenler:** `kullanici_sorusu`, `bm25_anahtar_kelime_skorlari`, `vektor_anlamsal_benzerlik_skorlari`, `rrf_birlestirilmis_siralama`
- **Jupyter Notebook (`gun_151_hibrit_arama_rag_bm25_vektor.ipynb`):**
  1. Telekom teknik terimleri (Örn: "VoLTE ayarı", "APN internet") içeren sorgularda BM25 ve Vektör aramasının ayrı ayrı çalıştırılması
  2. Reciprocal Rank Fusion (RRF) formülü ile iki farklı arama sonucunun en iyi şekilde harmanlanması
  3. Cross-Encoder Reranker ile en alakalı 3 dokümanın seçilip LLM bağlamına iletilmesi
- **Mülakat Sorusu:** RAG mimarilerinde sadece Vektör araması (Dense Search) kullanıldığında nadir teknik ürün kodlarında ve hata numaralarında neden başarısız olunur ve BM25 bunu nasıl çözer?

### Gün 152: Çağrı Metninden Kampanya Kabul İhtimali Puanlama
- **İş Alanı:** Dış Arama (Outbound Telemarketing) & Paket Satış Masası
- **Veri Kaynağı:** [Telemarketing Call Transcripts & Campaign Success](https://www.kaggle.com/datasets)
- **Model:** TabNet / CatBoost (Metin Embedding + Müşteri CRM Verisi Füzyonu)
- **Türkçe Değişkenler:** `gorusme_transkripti_embedding`, `musteri_mevcut_tarife_tutari`, `onerilen_kampanya_fiyati`, `teklifi_kabul_etme_olasiligi`
- **Jupyter Notebook (`gun_152_kampanya_kabul_ihtimali_nlp.ipynb`):**
  1. Görüşmenin ilk 30 saniyesinde müşterinin konuşma tarzı ve itiraz cümlelerinin analizi
  2. Müşteri demografisi ile konuşma metni gömmelerinin birleştirilerek teklif başarı ihtimali tahmini
  3. Kabul ihtimali düşük olduğunda temsilcinin ekranına alternatif indirimli paket önerisi düşürülmesi
- **Mülakat Sorusu:** Tabular veriler ile serbest metin verilerini (Multimodal Tabular-NLP) aynı modelde birleştirmede Late Fusion ve Early Fusion mimarileri nasıl tasarlanır?

### Gün 153: Otomatik Tarife Detay Özeti Üretici
- **İş Alanı:** Dijital Kanallar & Web/Mobil Tarife Karşılaştırma Sayfası
- **Veri Kaynağı:** [Telecom Tariff Terms & Conditions Contracts](https://www.kaggle.com/datasets)
- **Model:** mT5-Base / Turkish BART (Soyutlayıcı Özetleme - Abstractive Summarization)
- **Türkçe Değişkenler:** `uzun_tarife_yasal_metni`, `uretilen_3_maddelik_ozet`, `rouge_1_rouge_2_rouge_l_skorlari`
- **Jupyter Notebook (`gun_153_tarife_ozetleme_mt5.ipynb`):**
  1. 10 sayfalık hukuki ve teknik tarife taahhütnamelerinin özetleme modeline beslenmesi
  2. Müşterinin bilmesi gereken kritik maddeleri (Cayma bedeli, İnternet kotası, Aşım ücreti) madde madde çıkaran özetleme
  3. ROUGE ve BLEU metrikleri ile özet kalitesinin ve olgusal doğruluğunun (Factuality) ölçümü
- **Mülakat Sorusu:** Hukuki ve finansal metin özetlemede LLM halüsinasyonlarını (Hallucination) önlemek ve olgusal doğruluğu denetlemek için hangi Guardrails sistemleri kullanılır?

### Gün 154: Sosyal Medya Rakip Operatör Kampanya Karşılaştırma Analizörü
- **İş Alanı:** Pazarlama Stratejisi & Rekabet İstihbaratı Masası
- **Veri Kaynağı:** [Twitter / X Telecom Mentions & Campaign Feedback](https://www.kaggle.com/datasets)
- **Model:** RoBERTa-Turkish Aspect-Based Sentiment + Named Entity Recognition (NER)
- **Türkçe Değişkenler:** `tweet_metni`, `bahsedilen_operator_turkcell_vodafone_telekom`, `karsilastirma_kriteri_fiyat_kapsama_hiz`, `duygu_kutbu`
- **Jupyter Notebook (`gun_154_rakip_kampanya_analizoru.ipynb`):**
  1. Sosyal medyada rakip operatörler hakkında atılan tweetlerin gerçek zamanlı taranması
  2. Fiyat, internet hızı ve müşteri hizmetleri başlıklarında rakip memnuniyet/şikayet oranlarının kıyaslanması
  3. Rakibin zayıf kaldığı bölgelere özel karşı Turkcell kampanyası önerme motoru
- **Mülakat Sorusu:** Aspect-Based Sentiment Analysis (ABSA) ile genel cümle düzeyinde sentiment analizi arasındaki fark nedir?

### Gün 155: Sesli Yanıt (IVR) Fonetik Benzerlik Eşleştirici
- **İş Alanı:** 532 Sesli Yanıt Menüsü & Şive/Aksan/Yazım Hatası Toleransı
- **Veri Kaynağı:** [Turkish Phonetic Pronunciation & Voice Queries](https://huggingface.co/datasets)
- **Model:** Double Metaphone / Soundex Turkish Adaptation + Levenshtein Distance
- **Türkçe Değişkenler:** `kullanici_sesli_ifadesi_metni`, `fonetik_kod_uretilen`, `eslesen_menu_komutu`, `fonetik_benzerlik_orani`
- **Jupyter Notebook (`gun_155_ivr_fonetik_eslestirici.ipynb`):**
  1. Kullanıcının şiveli veya yanlış telaffuz ettiği komutların (Örn: "patura", "fatıra", "kontur") fonetik kodlarının çıkarılması
  2. Doğru IVR menü komutlarıyla ("Fatura Ödeme", "TL Yükleme") fonetik kod benzerliği eşleştirmesi
  3. Yanlış anlama oranını %40 azaltarak müşteriyi doğru menüye aktarma
- **Mülakat Sorusu:** Fonetik algoritmaların (Double Metaphone) ASR sonrası niyet eşleştirmede saf metinsel Levenshtein mesafesine göre avantajı nedir?

### Gün 156: Abonelik Sözleşmesi Cayma Bedeli ve Taahhüt Maddesi Bulucu
- **İş Alanı:** Hukuk Masası & Dijital Sözleşme Analizi
- **Veri Kaynağı:** [Telecom Legal Contracts & Addendums](https://www.kaggle.com/datasets)
- **Model:** LayoutLMv3 / DeBERTa-v3 Question Answering (Extractive QA)
- **Türkçe Değişkenler:** `sozlesme_pdf_goruntusu`, `soru_cayma_bedeli_nasil_hesaplanir`, `cevap_metin_kesiti`, `guven_skoru`
- **Jupyter Notebook (`gun_156_sozlesme_cayma_bedeli_qa.ipynb`):**
  1. Taranmış sözleşme sayfalarındaki görsel yerleşim ve metinlerin LayoutLMv3 ile işlenmesi
  2. "Taahhüt süresi ne kadar?", "Erken iptal halinde hangi indirimler geri alınır?" sorularına sözleşmeden doğrudan cevap çıkarma
  3. Müşteriye ve temsilciye kanuni hakları anında gösteren sözleşme yardımcısı
- **Mülakat Sorusu:** Extractive QA ile Generative QA arasındaki fark nedir ve yasal sözleşmelerde neden Extractive QA tercih edilir?

### Gün 157: Chatbot için Few-Shot Niyet Genişletici Sentetik Veri Pipeline'ı
- **İş Alanı:** BiP Dijital Asistan Eğitimi & Veri Artırma (Data Augmentation)
- **Veri Kaynağı:** [Turkish Chatbot Intent Dataset](https://huggingface.co/datasets)
- **Model:** Yerel Qwen2.5-3B-Instruct / LLaMA-3.2-1B (Ollama / HuggingFace Yerel) + Cosine Similarity Filtresi
- **Türkçe Değişkenler:** `ornek_niyet_cumlesi`, `uretilen_sentetik_varyasyonlar`, `anlamsal_benzerlik_esigi`, `filtrelenmis_egitim_verisi`
- **Jupyter Notebook (`gun_157_few_shot_sentetik_niyet_verisi.ipynb`):**
  1. 5 adet örnek kullanıcı cümlesinden yola çıkarak LLM ile 50 farklı alternatif soru varyasyonu üretme
  2. Üretilen cümlelerin özgünlük ve semantik kayma kontrolünden (Sentence Embeddings ile) geçirilmesi
  3. NLU niyet sınıflandırıcısının eğitim veri setinin 10 katına çıkarılması
- **Mülakat Sorusu:** Sentetik veri üretiminde "Model Collapse" ve semantik kayma (Semantic Drift) riskleri nasıl engellenir?

### Gün 158: Boyut Tabanlı Müşteri Memnuniyetsizliği (Aspect-Based Sentiment)
- **İş Alanı:** Müşteri Deneyimi Ölçümleme (NPS / CSAT Masası)
- **Veri Kaynağı:** [Customer Feedback & Review Survey Dataset](https://www.kaggle.com/datasets)
- **Model:** DeBERTa-v3 Multi-Task Learning (Aspect + Sentiment Classification)
- **Türkçe Değişkenler:** `anket_yorumu`, `tespit_edilen_boyutlar_fiyat_hiz_musteri_hizmetleri`, `boyut_duygu_skorlari`
- **Jupyter Notebook (`gun_158_aspect_based_sentiment_deberta.ipynb`):**
  1. "İnternetiniz çok hızlı ama faturalar çok pahalı" gibi çoklu duygu içeren cümlelerin ayrıştırılması
  2. Hız boyutuna Pozitif, Fiyat boyutuna Negatif etiket atanması
  3. Ürün ekiplerine departman bazlı net memnuniyet puanı (Aspect NPS) raporlanması
- **Mülakat Sorusu:** Multi-Task Learning mimarilerinde Aspect Extraction ve Sentiment Classification görevlerinin ortak temsil katmanından öğrenilmesinin yararı nedir?

### Gün 159: E-posta Destek Talebi Otomatik Cevap Taslağı Üretici
- **İş Alanı:** Turkcell Global Bilgi E-posta Destek Masası
- **Veri Kaynağı:** [Customer Support Email Dataset](https://huggingface.co/datasets)
- **Model:** RAG + Yerel Qwen2.5-7B-Instruct / LLaMA-3.2-3B (Ollama / HuggingFace Yerel) + Türkçe Kurumsal Şablon Motoru
- **Türkçe Değişkenler:** `gelen_musteri_epostasi`, `ilgili_cozum_dokumani`, `uretilen_cevap_taslagi`, `temsilci_onayladi_mi`
- **Jupyter Notebook (`gun_159_eposta_otomatik_cevap_taslagi.ipynb`):**
  1. Müşteriden gelen e-postanın konusunun, abone bilgilerinin ve talebinin tespiti
  2. Şirket içi çözüm rehberinden ilgili politikanın çekilerek kurumsal dilde kişiselleştirilmiş yanıt taslağı hazırlanması
  3. Temsilcinin tek tıkla inceleyip onaylayabileceği e-posta arayüzü entegrasyonu
- **Mülakat Sorusu:** Müşteri destek botlarında "Human-in-the-Loop" (insan onaylı yapay zeka) yaklaşımının marka güvenilirliğindeki rolü nedir?

### Gün 160: Müşteri İletişim Dili (Resmi vs Samimi) Belirleme ve Ton Eşleme
- **İş Alanı:** Dijital İletişim & Kişiselleştirilmiş İletişim Tonu (Tone of Voice)
- **Veri Kaynağı:** [Turkish Formality & Register Dataset](https://huggingface.co/datasets)
- **Model:** BERTurk Formality Classifier + Style Transfer Prompting
- **Türkçe Değişkenler:** `musteri_mesaji`, `resmiyet_puani_0_100`, `uygun_cevap_stili_resmi_kurumsal_genc_samimi`, `uyarlanan_cevap`
- **Jupyter Notebook (`gun_160_iletisim_dili_ton_esleme.ipynb`):**
  1. Kullanıcının hitap şeklinden ("Merhabalar efendim" vs "selam naber") resmiyet derecesinin tespiti
  2. Genç kullanıcıya enerjik/samimi, kurumsal kullanıcıya resmi/saygılı dilde yanıt üreten stil transferi
  3. Müşteri memnuniyetini ve iletişim bağını güçlendiren dinamik ton uyarlaması
- **Mülakat Sorusu:** NLP'de Formality Classification ve Text Style Transfer için kullanılan metrikler (BLEU, Formality Accuracy, PPL) nelerdir?

---

## 🛡️ Modül 13: Bilgisayarlı Görü, Saha Operasyonları & Güvenlik (Gün 161 – 175)

### Gün 161: Saha Teknisyenleri Düşme / Hareketsizlik Algılama
- **İş Alanı:** İş Sağlığı ve Güvenliği (İSG) & Kule Tırmanış Güvenliği
- **Veri Kaynağı:** [Kaggle - Human Fall Detection Video Dataset](https://www.kaggle.com/datasets)
- **Model:** YOLOv8-Pose + ST-GCN (Spatio-Temporal Graph Convolutional Network)
- **Türkçe Değişkenler:** `kamera_goruntusu`, `insan_iskelet_eklem_koordinatlari`, `dikey_hiz_ivmesi`, `dusme_alarmi_tetiklendi_mi`
- **Jupyter Notebook (`gun_161_isg_dusme_hareketsizlik_algilama.ipynb`):**
  1. Baz istasyonu kulelerine tırmanan teknisyenlerin 17 eklem noktasının (Pose Estimation) tespiti
  2. İskelet koordinatlarının dikey hızındaki ani düşüş ve ardından gelen hareketsizliğin modellenmesi
  3. Düşme algılandığı anda acil durum merkezine ve saha şefine anlık GPS koordinatlı SMS/Çağrı alarmı gönderme
- **Mülakat Sorusu:** Video tabanlı düşme tespitinde klasik optik akış (Optical Flow) yerine Pose-based GCN kullanmanın ışık değişimlerine karşı dayanıklılığı nedir?

### Gün 162: Kule Tırmanış Emniyet Kemeri (Harness) Takma Denetimi
- **İş Alanı:** Saha Operasyonları İSG Denetimi & Drone ile Teftiş
- **Veri Kaynağı:** [Roboflow - Safety Harness & PPE Detection](https://universe.roboflow.com/)
- **Model:** YOLOv8x Object Detection
- **Türkçe Değişkenler:** `kule_dron_goruntusu`, `emniyet_kemeri_var_mi`, `kanca_kuleye_takili_mi_iou`, `guvenli_tirmanis_onayi`
- **Jupyter Notebook (`gun_162_emniyet_kemeri_denetimi.ipynb`):**
  1. Kuleye tırmanan teknisyenin vücut kemeri (Harness), karabina kancası ve yaşam hattı bağlantısının tespiti
  2. Karabina kancasının kule demirine bağlı olup olmadığının Bounding Box kesişimi (IoU) ile analizi
  3. Kancayı takmadan tırmanan personelin tespit edilerek tırmanış izninin iptal edilmesi
- **Mülakat Sorusu:** Küçük nesnelerin (karabina kancası, bağlantı ipi) yüksekten çekilen drone görüntülerinde doğru tespiti için SAHI (Slicing Aided Hyper Inference) nasıl uygulanır?

### Gün 163: Optik Fiber Ekleme Noktası (Splice) Mikroskobik Kusur Tespiti
- **İş Alanı:** Superonline Fiber Optik Altyapı Kalite Denetimi
- **Veri Kaynağı:** [Fiber Optic Splice Microscopy Defect Dataset](https://www.kaggle.com/datasets)
- **Model:** Mask R-CNN / YOLOv8-Seg (Segmentasyon & Kusur Alanı Ölçümü)
- **Türkçe Değişkenler:** `mikroskop_goruntusu`, `ek_yeri_cam_cekirdek_hizasi`, `hava_kabarcigi_catlak_alani_piksel`, `ek_kalite_onayi`
- **Jupyter Notebook (`gun_163_fiber_ek_kusur_segmentasyonu.ipynb`):**
  1. Füzyon ek cihazından alınan optik fiber mikroskop görüntülerinin işlenmesi
  2. Fiber çekirdeklerindeki eksen kayması, çatlak, hava kabarcığı ve yanık kusurlarının piksel bazlı segmentasyonu
  3. Sinyal kaybı (dB) standardı aşan kalitesiz eklerin teknisyene anında bildirilip yeniletilmesi
- **Mülakat Sorusu:** Endüstriyel mikroskobik kalite kontrolünde piksel bazlı kusur alanının hesaplanması ve kabul/ret eşiğinin belirlenmesinde Dice Loss nasıl kullanılır?

### Gün 164: Sokak Kameralarından Açık/Kırık Menhol Kapağı Tespiti
- **İş Alanı:** Altyapı Güvenliği, Fiber Kablo Hırsızlığı Önleme & Belediye Güvenliği
- **Veri Kaynağı:** [Roboflow - Manhole Cover Defect & Open Lid Dataset](https://universe.roboflow.com/)
- **Model:** YOLOv8-Nano (Edge AI / Araç Üstü Kamera)
- **Türkçe Değişkenler:** `yol_goruntusu`, `kapak_durumu_kapali_acik_kirik`, `gps_koordinati`, `tehlike_seviyesi`
- **Jupyter Notebook (`gun_164_acik_menhol_kapagi_tespiti.ipynb`):**
  1. Turkcell saha devriye araçlarının kameralarından menhol kapaklarının tespiti
  2. Açık bırakılmış veya kırılmış menhollerin tespit edilerek kablo hırsızlığı ve can güvenliği tehlikesinin haritalanması
  3. Saha nöbetçi ekiplerine acil müdahale bildiriminin iletilmesi
- **Mülakat Sorusu:** Hareket halindeki araç kameralarında (Motion Blur) nesne tespit performansını artırmak için hangi deblurring ve shutter hızı filtreleri kullanılır?

### Gün 165: Kırsal İstasyonlarda Tel Örgü İhlali & İzinsiz Giriş Algılama
- **İş Alanı:** Kırsal Baz İstasyonu Fiziksel Güvenliği & Akıllı Kamera
- **Veri Kaynağı:** [Perimeter Intrusion & Fence Crossing Video Dataset](https://www.kaggle.com/datasets)
- **Model:** YOLOv8 + Virtual Fence Line Crossing (Sanal Çit İhlali Algoritması)
- **Türkçe Değişkenler:** `guvenlik_kamera_goruntusu`, `sanal_cit_cizgisi_koordinatlari`, `insan_arac_hareket_vektoru`, `izinsiz_giris_alarmi`
- **Jupyter Notebook (`gun_165_tel_orgu_izinsiz_giris_tespiti.ipynb`):**
  1. Kamera açısında tel örgü sınırının sanal koordinat çizgisi (Virtual Tripwire) olarak tanımlanması
  2. İnsan veya aracın tel örgüyü aşma yönünün vektörel takibi
  3. Kedi, köpek veya rüzgarda sallanan ağaç kaynaklı False Alarm (Yanlış Alarm) filtreleme
- **Mülakat Sorusu:** Güvenlik kameralarında hayvan ve rüzgar kaynaklı yanlış alarmları (False Positive) önlemek için optik akış ve insan sınıflandırma eşiği nasıl birleştirilir?

### Gün 166: Sunucu Odası Yangın Tüpü ve Acil Çıkış Engel Denetimi
- **İş Alanı:** Veri Merkezi İSG & Yangın Güvenliği Denetimi
- **Veri Kaynağı:** [Roboflow - Fire Extinguisher & Blocked Exit Detection](https://universe.roboflow.com/)
- **Model:** YOLOv8 Object Detection
- **Türkçe Değişkenler:** `guvenlik_kamerasi_goruntusu`, `yangin_tupu_yeri_bos_mu`, `kapi_onu_engel_var_mi`, `isg_ceza_puani`
- **Jupyter Notebook (`gun_166_yangin_tupu_cikis_engeli_denetimi.ipynb`):**
  1. Veri merkezi koridorlarındaki yangın tüplerinin yerinde olup olmadığının kontrolü
  2. Acil çıkış kapılarının önüne bırakılmış koli, sunucu kasası gibi engellerin tespiti
  3. İSG kurallarına aykırı durumların otomatik fotoğraflanarak bina yönetimine raporlanması
- **Mülakat Sorusu:** Sabit güvenlik kameralarında arka plan çıkarma (Background Subtraction) ile derin öğrenme nesne tespitini birleştirmenin hesaplama maliyetine etkisi nedir?

### Gün 167: Dijital Hat Başvurusunda Canlı Selfie ve Kimlik Fotoğrafı Doğrulama
- **İş Alanı:** Dijital Hat Açılışı (e-KYC) & Yüz Biyometrisi Eşleme
- **Veri Kaynağı:** [LFW / CelebA Face Verification Dataset](https://vis-www.cs.umass.edu/lfw/)
- **Model:** FaceNet / InsightFace (ArcFace) + MTCNN Yüz Hizalama
- **Türkçe Değişkenler:** `kimlik_foto_yuzu`, `canli_selfie_yuzu`, `yuz_vektorleri_512d`, `kosinus_benzerligi`, `ayni_kisi_mi_onayi`
- **Jupyter Notebook (`gun_167_canli_selfie_kimlik_dogrulama.ipynb`):**
  1. Kimlik kartı üzerindeki vesikalık fotoğraftan ve canlı selfie'den yüz tespiti ve 512 boyutlu gömme çıkarımı
  2. ArcFace modeli ile iki yüz arasındaki Cosine Similarity mesafesinin hesaplanması
  3. Benzerlik eşiği %85'in üzerinde olduğunda dijital hat başvurusuna otomatik onay verilmesi
- **Mülakat Sorusu:** Farklı ışık, yaş farkı ve çözünürlük altındaki yüz doğrulama modellerinde False Acceptance Rate (FAR) ile False Rejection Rate (FRR) dengesi nasıl kurulur?

### Gün 168: Bayi Raf Standı Planogram Uyumluluk Kontrolü
- **İş Alanı:** Turkcell Mağazacılık & Perakende Raf Tanzim Teşhir Denetimi
- **Veri Kaynağı:** [Roboflow - Retail Shelf Product Detection & Planogram](https://universe.roboflow.com/)
- **Model:** YOLOv8 + Planogram Kural Matrisi Eşleme
- **Türkçe Değişkenler:** `magaza_raf_fotografi`, `tespit_edilen_telefon_aksesuar_urunleri`, `raf_sira_duzeni`, `planogram_uyum_yuzdesi`
- **Jupyter Notebook (`gun_168_magaza_raf_planogram_denetimi.ipynb`):**
  1. Mağaza müdürünün çektiği telefon ve aksesuar standı fotoğraflarının incelenmesi
  2. Ürünlerin merkez tarafından belirlenen planogram sırasına (Örn: En pahalı cihazın en üst rafta olması) uygunluğunun tespiti
  3. Hatalı dizilen veya eksik ürünlerin mağaza personeline anında bildirilmesi
- **Mülakat Sorusu:** Yoğun nesneli perakende raf görüntülerinde (Dense Object Detection) ürünlerin birbirini örtmesi (Occlusion) durumunda Non-Maximum Suppression (NMS) eşikleri nasıl ayarlanır?

### Gün 169: SIM Kart Çip Çizik ve Kusur Tespiti
- **İş Alanı:** SIM Kart Fabrika Üretim Hattı Kalite Kontrolü
- **Veri Kaynağı:** [Kaggle - PCB Defect Detection Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- **Model:** Autoencoder Reconstruction Anomaly Detection / YOLOv8-Seg
- **Türkçe Değişkenler:** `sim_cip_mikroskop_goruntusu`, `altin_kaplama_cizik_alani`, `kopuk_devre_izi`, `kusurlu_sim_reddi`
- **Jupyter Notebook (`gun_169_sim_kart_cip_kusur_tespiti.ipynb`):**
  1. Üretim bandından saniyede 10 adet geçen SIM kart altın çip yüzeylerinin taranması
  2. Normal çip görüntüsüyle eğitilen Autoencoder ile çizik, leke ve devre kopukluklarının tespiti
  3. Kusurlu SIM kartların hava üfleyici mekanizmayla ıskarta kutusuna ayrılması
- **Mülakat Sorusu:** Endüstriyel görsel kontrolde sadece hatasız ürünlerle eğitilen Denetimsiz Anomali Tespiti (Unsupervised Anomaly Detection - PatchCore / Padim) yöntemlerinin avantajı nedir?

### Gün 170: Baz İstasyonu Jeneratör Yağ/Yakıt Sızıntısı Tespiti
- **İş Alanı:** Saha Altyapı Bakımı & Çevre Güvenliği
- **Veri Kaynağı:** [Industrial Liquid Leakage & Puddle Segmentation Dataset](https://www.kaggle.com/datasets)
- **Model:** SegNet / U-Net (Sıvı Birikintisi Segmentasyonu)
- **Türkçe Değişkenler:** `jenerator_odasi_goruntusu`, `sizinti_alani_piksel`, `sivi_turu_motor_yagi_dizel`, `yangin_cevre_kirliligi_riski`
- **Jupyter Notebook (`gun_170_jenerator_yakit_sizintisi_segmentasyonu.ipynb`):**
  1. Jeneratör tabanında oluşan yağ ve mazot sızıntılarının güvenlik kamerasıyla segmentasyonu
  2. Zemin yansıması ve su birikintisinden ayırt etmek için renk ve doku analizi
  3. Jeneratörün susuz/yağsız kalarak yanmasını önleyecek acil servis çağrısının açılması
- **Mülakat Sorusu:** Sıvı birikintileri gibi belirgin geometrik şekli olmayan (Amorphous Objects) nesnelerin segmentasyonunda doku ve gradyan kayıp fonksiyonlarının önemi nedir?

### Gün 171: Fırtınada Kule Rüzgar Salınımı ve Yapısal Eğrilik Ölçümü
- **İş Alanı:** Telekom Kule Yapısal Sağlık İzleme (Structural Health Monitoring)
- **Veri Kaynağı:** [Tower Sway & Vibration Video Telemetry](https://www.kaggle.com/datasets)
- **Model:** Faz Tabanlı Optik Akış (Phase-Based Optical Flow) + Sub-pixel Edge Tracking
- **Türkçe Değişkenler:** `yuksek_cozunurluklu_kule_videosu`, `tepe_noktasi_yer_degistirme_piksel`, `salinim_genligi_cm`, `rezonans_tehlikesi`
- **Jupyter Notebook (`gun_171_kule_ruzgar_salinimi_olculmesi.ipynb`):**
  1. Fırtınalı havalarda telekom kulelerinin tepe noktasının milimetrik yer değiştirmesinin video ile takibi
  2. Sub-pixel hassasiyetle rüzgar salınım genliği ve doğal titreşim frekansının hesaplanması
  3. Metal yorulması ve kule devrilme riskine karşı erken yapısal alarm üretimi
- **Mülakat Sorusu:** Video üzerinden mikroskobik titreşim ve salınımları büyütüp ölçmek için kullanılan "Eulerian Video Magnification" tekniği nasıl çalışır?

### Gün 172: Aşınmış ve Hasarlı Karekod Düzeltme & Okuma
- **İş Alanı:** Saha Envanter Takibi & Yıpranmış Etiket Okuma
- **Veri Kaynağı:** [Damaged QR Code & Barcode Restoration Dataset](https://www.kaggle.com/datasets)
- **Model:** Generative Adversarial Network (SRGAN / Pix2Pix) + Pyzbar / OpenCV QR Decoder
- **Türkçe Değişkenler:** `hasarli_karekod_goruntusu`, `onaricidan_gecen_karekod`, `okunan_cihaz_seri_no`, `kod_cozme_basarili_mi`
- **Jupyter Notebook (`gun_172_hasarli_karekod_onarimi_gan.ipynb`):**
  1. Sahadaki cihazların üzerinde yırtılmış, çizilmiş veya solmuş karekodların fotoğraflanması
  2. GAN tabanlı görüntü iyileştirme ile eksik karekod piksellerinin onarılması ve kontrast artırımı
  3. Standart QR okuyucuların çözemediği kodların %90 oranında başarıyla okunması
- **Mülakat Sorusu:** QR kodların kendi içindeki Reed-Solomon hata düzeltme kapasitesi (Error Correction Level L/M/Q/H) yetersiz kaldığında Derin Öğrenme süper-çözünürlük nasıl destek olur?

### Gün 173: Veri Merkezi Kabinet Kapak Açık Unutulma Dedektörü
- **İş Alanı:** Veri Merkezi Güvenliği & Soğutma Verimliliği
- **Veri Kaynağı:** [Server Rack Door Open/Close State Dataset](https://www.kaggle.com/datasets)
- **Model:** MobileNetV3 Sınıflandırıcı / YOLOv8
- **Türkçe Değişkenler:** `koridor_kamera_goruntusu`, `kabinet_id`, `kapi_durumu_acik_kapali`, `acik_kalma_suresi_dakika`
- **Jupyter Notebook (`gun_173_kabinet_kapak_acik_unutulma.ipynb`):**
  1. Sunucu kabinetlerinin kapaklarının açık/kapalı durumunun güvenlik kamerasıyla izlenmesi
  2. 15 dakikadan uzun süre açık kalan kapakların soğutma hava akışını (Hot Aisle / Cold Aisle) bozmasını engelleme
  3. Görevli personele "Kabinet 42B kapağı açık unutuldu" uyarısı gönderme
- **Mülakat Sorusu:** Veri merkezi koridorlarındaki perspektif bozulmaları ve dar açı kameralarda sınıflandırma doğruluğunu artırmak için Spatial Transformer Networks (STN) nasıl uygulanır?

### Gün 174: Saha Projektör ve Gece Aydınlatma Arıza Tespiti
- **İş Alanı:** Saha Fiziksel Güvenliği & Gece Görüş Şartları
- **Veri Kaynağı:** [Night CCTV & Lighting Failure Dataset](https://www.kaggle.com/datasets)
- **Model:** Parlaklık Histogramı Analizi + Hafif CNN Sınıflandırıcı
- **Türkçe Değişkenler:** `gece_kamera_kare_goruntusu`, `ortalama_parlaklik_lux`, `aydinlatma_arizali_mi`, `guvenlik_riski_puani`
- **Jupyter Notebook (`gun_174_saha_aydinlatma_ariza_tespiti.ipynb`):**
  1. Gece saatlerinde istasyon aydınlatma projektörlerinin çalışıp çalışmadığının kamera görüntüsünden tespiti
  2. Ampul patlaması veya elektrik kesintisi kaynaklı karanlıkta kalan kör bölgelerin analizi
  3. Güvenlik zafiyeti oluşmadan önce aydınlatma onarım iş emrinin açılması
- **Mülakat Sorusu:** Gece görüşlü (Infrared - IR) kameralarda IR LED aydınlatması ile harici görünür ışık projektör arızası görüntü işlemede nasıl ayrıştırılır?

### Gün 175: Drone Görüntüsünden Kule Paslanma ve Korozyon Analizi
- **İş Alanı:** Kule Bakım Onarım & Yapısal Ömür Uzatma
- **Veri Kaynağı:** [Roboflow - Metal Rust & Corrosion Segmentation Dataset](https://universe.roboflow.com/)
- **Model:** YOLOv8-Seg / DeepLabV3+ (Pas Alanı Yüzdesi Ölçümü)
- **Türkçe Değişkenler:** `dron_fotografi`, `tespit_edilen_pas_maskesi`, `toplam_metal_yuzey_alani`, `paslanma_orani_yuzde`, `boya_bakim_aciliyeti`
- **Jupyter Notebook (`gun_175_kule_paslanma_korozyon_analizi.ipynb`):**
  1. Otonom drone teftişinde çekilen kule metal ayak fotoğraflarının piksellerine ayrıştırılması
  2. Yüzeydeki pas ve korozyon alanlarının DeepLabV3+ ile segmentasyonu
  3. Pas oranı %15'i geçen kulelerin pas sökücü ve galvaniz boya programına dahil edilmesi
- **Mülakat Sorusu:** Dış ortam çekimlerinde metal üzerindeki çamur, sarı yaprak lekeleri ile gerçek korozyon pasını ayırt etmek için renk uzayları (HSV / LAB) nasıl kullanılır?

---

## 🔒 Modül 14: Siber Güvenlik, Ağ Savunması & Tehdit İstihbaratı (Gün 176 – 185)

### Gün 176: Botnet Komuta Kontrol (C2) Periyodik Ağ Sinyali Tespiti
- **İş Alanı:** Turkcell Siber Savunma Merkezi (SOC) & Botnet Avcılığı
- **Veri Kaynağı:** [Stratosphere IPS - CTU-13 Malware & Botnet Dataset](https://www.stratosphereips.org/datasets-ctu13)
- **Model:** Otoregresif Fourier Dönüşümü (FFT) + Random Forest Classifier
- **Türkçe Değişkenler:** `kaynak_ip`, `hedef_ip`, `paket_zaman_araligi_ms`, `periyodiklik_gucu_fft`, `botnet_c2_olasiligi`
- **Jupyter Notebook (`gun_176_botnet_c2_periyodik_sinyal_tespiti.ipynb`):**
  1. Zararlı yazılım bulaşmış cihazların Komuta Kontrol (C2) sunucusuna gönderdiği "Beaconing" (kalp atışı) sinyallerinin analizi
  2. Ağ paket zaman aralıklarına FFT uygulayarak periyodik düzenli sinyallerin tespiti
  3. İlgili zararlı IP adresinin Turkcell omurga güvenlik duvarında otomatik karantinaya alınması
- **Mülakat Sorusu:** C2 trafiğinde tespit edilmemek için rastgele bekleme süresi (Jitter) ekleyen gelişmiş botnetler FFT ve otokorelasyon analizini nasıl atlatmaya çalışır?

### Gün 177: Web Uygulaması API İstismar (Exploit) Tespiti
- **İş Alanı:** Web Uygulama Güvenlik Duvarı (WAF) & API Güvenliği
- **Veri Kaynağı:** [CSIC 2010 HTTP Dataset / OWASP API Top 10](https://www.kaggle.com/datasets)
- **Model:** RoBERTa / Character-Level CNN (SQLi, XSS, Path Traversal Sınıflandırma)
- **Türkçe Değişkenler:** `http_istek_url`, `istek_govdesi_body`, `istek_basliklari_headers`, `saldiri_turu_sqli_xss_rce_normal`, `saldiri_skoru`
- **Jupyter Notebook (`gun_177_waf_api_istismar_tespiti.ipynb`):**
  1. Turkcell ve Paycell API uç noktalarına gelen HTTP istek parametrelerinin karakter dizisi analizi
  2. SQL Enjeksiyonu, XSS, Remote Code Execution (RCE) saldırı kalıplarının tespiti
  3. Kural tabanlı imzaların yakalayamadığı sıfır-gün (Zero-Day) varyasyonlarının derin öğrenmeyle bloklanması
- **Mülakat Sorusu:** Character-Level CNN modellerinin kelime bazlı NLP modellerine göre URL ve Payload analizindeki üstünlüğü (özel karakterleri ve kaçış dizilerini yakalama) nedir?

### Gün 178: VPN ve Tor Anonim Ağ Trafiği Sınıflandırma
- **İş Alanı:** Ağ Güvenliği & Şifrelenmiş Trafik Karakterizasyonu
- **Veri Kaynağı:** [UNB ISCX - Tor-NonTor & VPN-nonVPN Dataset](https://www.unb.ca/cic/datasets/tor.html)
- **Model:** XGBoost + Paket Boyut Dağılımı ve Entropi Analizi
- **Türkçe Değişkenler:** `akis_suresi_sn`, `gelen_giden_paket_boyut_orani`, `akistaki_bayt_entropisi`, `trafik_turu_vpn_tor_normal`
- **Jupyter Notebook (`gun_178_vpn_tor_ag_trafigi_siniflandirma.ipynb`):**
  1. Paket içerikleri şifreli (TLS/SSL) olsa bile paket boyutları, akış süreleri ve zamanlama izlerinden öznitelik çıkarma
  2. Tor köprüleri ve VPN tünelleri üzerinden akan şüpheli trafiğin sınıflandırılması
  3. Güvenlik politikalarına aykırı veri sızdırma (Data Exfiltration) kanallarının tespiti
- **Mülakat Sorusu:** Şifreli trafikte Deep Packet Inspection (DPI) yapılamadığında akış seviyesi (Flow-Level) istatistiksel değişkenler sınıflandırma için nasıl yeterli olur?

### Gün 179: Güvenlik Duvarı Loglarından Port Tarama (Port Scan) Tespiti
- **İş Alanı:** Şebeke Güvenliği & Erken Tehdit İstihbaratı
- **Veri Kaynağı:** [Kaggle - Network Firewall Log Data (Port Scan & DoS)](https://www.kaggle.com/datasets)
- **Model:** Kayan Pencere (Sliding Window) + LightGBM / DBSCAN
- **Türkçe Değişkenler:** `kaynak_ip`, `hedef_port_sayisi_10sn`, `basarisiz_syn_paket_orani`, `tarama_turu_syn_scan_fin_scan`, `ip_bloke_edilsin_mi`
- **Jupyter Notebook (`gun_179_port_tarama_tespiti.ipynb`):**
  1. Güvenlik duvarı loglarında kısa sürede yüzlerce farklı porta bağlantı açmaya çalışan IP'lerin izlenmesi
  2. Nmap SYN Scan, FIN Scan ve XMAS Scan saldırı paternlerinin tespiti
  3. Saldırgan IP'nin saniyeler içinde otomatik dinamik erişim engelleme listesine (Blacklist) eklenmesi
- **Mülakat Sorusu:** Yavaş ve sinsi port taraması (Slow Scan - günlere yayılan tarama) yapan gelişmiş saldırganlar kayan pencere dedektörlerini nasıl aşar ve bu durum Graf analiziyle nasıl yakalanır?

### Gün 180: Sahte Baz İstasyonu (IMSI Catcher / Stingray) Sinyal Avcısı
- **İş Alanı:** Hücresel Ağ Güvenliği & Abone Gizliliği Savunması
- **Veri Kaynağı:** [Cellular Rogue Base Station Telemetry & Signal Anomaly Dataset](https://www.kaggle.com/datasets)
- **Model:** İzolasyon Ormanı (Isolation Forest) + Kural Tabanlı Şebeke Parametre Denetimi
- **Türkçe Değişkenler:** `hucre_kimligi_ci`, `sinyal_gucu_rsrp`, `sifreleme_istegi_a5_0_a5_1`, `sessiz_sms_adedi`, `sahte_baz_istasyonu_olasiligi`
- **Jupyter Notebook (`gun_180_sahte_baz_istasyonu_tespiti.ipynb`):**
  1. Çevrede aniden beliren, çok yüksek sinyal gücüyle telefonları 2G'ye düşürmeye (Downgrade Attack) çalışan sahte hücrelerin tespiti
  2. A5/0 (Şifresiz) iletişim dayatması yapan ve kullanıcı IMSI numaralarını toplayan cihazların analizi
  3. Kullanıcı cihazlarına ve şebeke kontrol merkezine sahte istasyon ihbarı düşürülmesi
- **Mülakat Sorusu:** 2G şebekelerinin tek yönlü kimlik doğrulaması (Mutual Authentication eksikliği) zafiyetinden yararlanan IMSI Catcher saldırıları 5G Standalone (SA) mimarisinde SUCI protokolüyle nasıl engellenir?

### Gün 181: Bellek Dökümünden (Memory Dump) Zararlı Yazılım Tespiti
- **İş Alanı:** Uç Nokta Güvenliği (EDR) & Adli Bilişim (Digital Forensics)
- **Veri Kaynağı:** [UNB CIC-MalMem-2022 / Obfuscated Memory Malware Dataset](https://www.unb.ca/cic/datasets/malmem-2022.html)
- **Model:** Random Forest + CatBoost Classifier
- **Türkçe Değişkenler:** `aktif_proses_sayisi`, `enjekte_edilen_dll_sayisi`, `gizli_thread_adedi`, `zararli_yazilim_sinifi_ransomware_spyware_trojan`
- **Jupyter Notebook (`gun_181_bellek_dokumu_zararli_yazilim.ipynb`):**
  1. Sunucu RAM bellek dökümünden (Volatility Framework çıktısı) çıkarılan sistem çağrıları ve proses öznitelikleri
  2. Diske dosya yazmadan doğrudan bellekte çalışan (Fileless Malware) ve gizlenen tehditlerin tespiti
  3. Fidye yazılımı (Ransomware) bulaşmış sunucunun ağdan otomatik izole edilmesi
- **Mülakat Sorusu:** Dosyasız zararlı yazılımların (Fileless Malware) klasik imza tabanlı antivirüsleri atlatma stratejileri nelerdir ve bellek analizi bunu nasıl çözer?

### Gün 182: Şüpheli İç Tehdit (Insider Threat) Davranış Analizi
- **İş Alanı:** Kurumsal Bilgi Güvenliği & Veri Sızıntısı Önleme (DLP)
- **Veri Kaynağı:** [Carnegie Mellon University (CERT) - Insider Threat Test Dataset](https://kilthub.cmu.edu/)
- **Model:** LSTM Autoencoder / Isolation Forest (Kullanıcı Davranış Biyometrisi)
- **Türkçe Değişkenler:** `kullanici_id`, `mesai_disi_giris_sayisi`, `indirilen_dosya_hacmi_mb`, `usb_bellek_takildi_mi`, `ic_tehdit_skoru`
- **Jupyter Notebook (`gun_182_ic_tehdit_davranis_analizi.ipynb`):**
  1. Şirket çalışanlarının e-posta, dosya kopyalama, USB kullanımı ve mesai saatleri aktivitelerinin profillenmesi
  2. İstifa öncesi kritik müşteri veritabanlarını indiren şüpheli çalışan hareketlerinin anomali tespiti
  3. Bilgi Güvenliği ekibine gerçek zamanlı risk uyarı alarmı düşürülmesi
- **Mülakat Sorusu:** Kullanıcı ve Varlık Davranış Analitiğinde (UEBA) "Role-Based Baseline" ile "Individual Baseline" oluşturmanın farkı nedir?

### Gün 183: SSH / RDP Kaba Kuvvet (Brute Force) Saldırı Dedektörü
- **İş Alanı:** Veri Merkezi Sunucu Savunması & SSH Koruması
- **Veri Kaynağı:** [Kaggle - SSH/RDP Brute Force Authentication Logs](https://www.kaggle.com/datasets)
- **Model:** Markov Zinciri (Markov Chain) / Logistic Regression
- **Türkçe Değişkenler:** `kaynak_ip`, `dakikadaki_basarisiz_giris_sayisi`, `denenen_kullanici_adlari_entropisi`, `brute_force_olasiligi`
- **Jupyter Notebook (`gun_183_ssh_brute_force_tespiti.ipynb`):**
  1. Linux ve Windows sunuculardaki `auth.log` ve `Security Event 4625` kayıtlarının taranması
  2. Sözlük saldırısı (Dictionary Attack) ile rastgele kullanıcı adı deneyen botların tespiti
  3. Fail2ban benzeri yapay zeka ajanının IP'yi firewall üzerinden anında engellemesi
- **Mülakat Sorusu:** Dağıtık kaba kuvvet saldırılarında (Password Spraying - binlerce farklı IP'den tek bir şifre deneme) geleneksel IP limitleri neden yetersiz kalır?

### Gün 184: Açık Kaynak Git Depolarında Sızdırılmış API Key & Token Avcısı
- **İş Alanı:** Turkcell DevSecOps & Kaynak Kod Güvenliği
- **Veri Kaynağı:** [SecretBench / Leaked API Keys & Secrets Dataset](https://github.com/)
- **Model:** Regex + Shannon Entropisi + Fine-Tuned CodeBERT
- **Türkçe Değişkenler:** `kod_satiri_metni`, `shannon_entropi_degeri`, `anahtar_turu_aws_turkcell_jwt_openai`, `gizli_anahtar_mi`
- **Jupyter Notebook (`gun_184_sizdirilmis_api_key_avcisi.ipynb`):**
  1. GitHub ve GitLab depolarına atılan kod commit'lerinin taranması
  2. Yüksek entropili rastgele karakter dizilerinin (API Token, Private Key, DB Şifresi) CodeBERT ile filtrelenmesi
  3. Test ve sahte anahtarları eleyip gerçek sızan şirket anahtarlarını anında iptal ettirme (Revoke) pipeline'ı
- **Mülakat Sorusu:** Statik Regex kuralları ile yüksek entropili string tarama yaklaşımının yüksek False Positive üretmesi sorunu CodeBERT ile nasıl çözülür?

### Gün 185: DGA (Domain Generation Algorithm) ile Üretilen Sahte Alan Adı Tespiti
- **İş Alanı:** Turkcell Güvenli DNS Servisi & Zararlı Yazılım Engelleme
- **Veri Kaynağı:** [Kaggle - DGA Domain Detection Dataset](https://www.kaggle.com/datasets)
- **Model:** 1D-CNN + Bi-LSTM (Karakter Düzeyinde Alan Adı Sınıflandırma)
- **Türkçe Değişkenler:** `sorgulanan_alan_adi_domain`, `sesli_sessiz_harf_orani`, `n_gram_entropisi`, `dga_alan_adi_olasiligi`
- **Jupyter Notebook (`gun_185_dga_sahte_alan_adi_tespiti.ipynb`):**
  1. DNS sunucularına gelen anlamsız ve rastgele harf dizisi alan adlarının (Örn: `xkz83jqlm.com`) tespiti
  2. Karakter seviyesinde eğitilen Bi-LSTM modeli ile meşru alan adları ile zararlı yazılımların türettiği DGA alan adlarının ayrıştırılması
  3. DNS seviyesinde engelleme (DNS Sinkholing) uygulayarak botnet enfeksiyonunu durdurma
- **Mülakat Sorusu:** DGA alan adı tespitinde sözlük tabanlı DGA (Dictionary-based DGA - Örn: `sunflowermonkeytable.com`) türlerinin yakalanmasında Word-Level embedding modellerinin önemi nedir?

---

## ⚙️ Modül 15: MLOps, Veri Mühendisliği & Dağıtık Akış (Gün 186 – 195)

### Gün 186: Feature Store (Feast) ile Canlı Müşteri Öznitelik Deposu
- **İş Alanı:** Turkcell Büyük Veri & Gerçek Zamanlı Yapay Zeka Platformu
- **Veri Kaynağı:** [Kaggle - Telco Customer 360 Feature Store Traces](https://www.kaggle.com/datasets)
- **Model:** Feast Feature Store + Redis (Online Store) + Parquet / S3 (Offline Store)
- **Türkçe Değişkenler:** `abone_id`, `son_24s_kullanilan_data_mb`, `anlik_kalan_bakiye`, `feature_view_tanimi`
- **Jupyter Notebook (`gun_186_feature_store_feast.ipynb`):**
  1. Müşteri özniteliklerinin (Offline eğitim ve Online gerçek zamanlı çıkarım için) tek bir kaynakta tanımlanması
  2. Redis üzerinde milisaniyelik gecikmeyle (Low Latency Feature Retrieval) canlı değişken sunumu
  3. Model eğitimi sırasında zaman yolculuğu (Point-in-Time Correctness / Time Travel) ile veri sızıntısının önlenmesi
- **Mülakat Sorusu:** MLOps mimarilerinde Feature Store kullanmanın "Training-Serving Skew" (Eğitim ve Canlı Dağıtım Arasındaki Tutarsızlık) sorununu çözmedeki rolü nedir?

### Gün 187: Streaming K-Means ile Bellek Üzerinde Ağ Paket Kümeleme
- **İş Alanı:** Şebeke Trafik İzleme & Anlık Akış Kümeleme (Stream Mining)
- **Veri Kaynağı:** [Real-Time Network Stream Packet Traces](https://www.kaggle.com/datasets)
- **Model:** River / Scikit-Multiflow Streaming Mini-Batch K-Means
- **Türkçe Değişkenler:** `akis_bayt_hizi`, `paket_araligi_ms`, `anlik_kume_merkezleri`, `yeni_kume_anomali_mi`
- **Jupyter Notebook (`gun_187_streaming_kmeans_ag_trafigi.ipynb`):**
  1. Disk üzerine kaydetmeden doğrudan RAM bellekten akan ağ paketlerinin tek geçişte (Single-Pass) kümelenmesi
  2. Kayan veri akışına uyum sağlayan dinamik küme merkezi güncellemesi
  3. Alışılmadık yeni trafik kümesi oluştuğunda anında uyarı tetikleme
- **Mülakat Sorusu:** Streaming ML algoritmalarında "Concept Drift" (Kavram Kayması) ve bellek kısıtı (Memory Bounded Processing) nasıl yönetilir?

### Gün 188: Delta Lake ile PostgreSQL Veritabanı CDC Pipeline Simülasyonu
- **İş Alanı:** Turkcell Veri Ambarı & Canlı Veri Akış Hattı (Data Lakehouse)
- **Veri Kaynağı:** [PostgreSQL Debezium CDC Simulated Stream](https://www.kaggle.com/datasets)
- **Model:** Delta Lake PySpark / DuckDB Delta Entegrasyonu (ACID Transactions)
- **Türkçe Değişkenler:** `islem_turu_insert_update_delete`, `abone_id`, `degisen_alanlar_json`, `delta_tablo_versiyonu`
- **Jupyter Notebook (`gun_188_delta_lake_cdc_pipeline.ipynb`):**
  1. Debezium formatında gelen Change Data Capture (CDC) veri akışının simülasyonu
  2. Delta Lake tablosuna `MERGE INTO` (Upsert) komutuyla verilerin tutarlı yazılması
  3. Delta Time Travel ile geçmiş bir andaki veri tablosuna geri dönme (Snapshot Isolation)
- **Mülakat Sorusu:** Klasik Veri Ambarları (Data Warehouse) yerine Data Lakehouse (Delta Lake / Apache Iceberg) mimarisinin büyük veri ML projelerindeki maliyet ve hız avantajları nelerdir?

### Gün 189: MLflow ile Model Sürümleme ve Otomatik A/B Testi
- **İş Alanı:** MLOps Model Yönetimi & Canlıya Alma Pipeline'ı
- **Veri Kaynağı:** [Telco Churn Model Experiment Registry Data](https://www.kaggle.com/datasets)
- **Model:** MLflow Tracking + Model Registry + Canary Deployment Simülasyonu
- **Türkçe Değişkenler:** `deney_id`, `model_versiyonu_v1_v2`, `roc_auc_skoru`, `canli_trafik_bolusumu_yuzde`
- **Jupyter Notebook (`gun_189_mlflow_model_surumleme_ab_test.ipynb`):**
  1. Farklı hiperparametrelerle eğitilen CatBoost ve LightGBM modellerinin metriklerinin MLflow'a kaydedilmesi
  2. En iyi modelin otomatik olarak "Production" aşamasına terfi ettirilmesi
  3. Canlı trafiğin %90'ını eski modele, %10'unu yeni modele (Canary Release) yönlendiren A/B test pipeline'ı
- **Mülakat Sorusu:** MLOps'ta Model Drift tespit edildiğinde CI/CD boru hattı üzerinden otomatik yeniden eğitimi (Continuous Training - CT) tetikleyen mimari nasıl kurulur?

### Gün 190: Graf Tabanlı Dolandırıcılık Şebekesi Analizi
- **İş Alanı:** Turkcell & Paycell Ortak Sahtekarlık Şebekesi Çökertme
- **Veri Kaynağı:** [Kaggle - Financial Fraud Graph Network Dataset](https://www.kaggle.com/datasets)
- **Model:** NetworkX + Louvain Topluluk Tespiti (Community Detection) + PageRank
- **Türkçe Değişkenler:** `dugum_id_kullanici_cihaz_kart`, `kenar_turu_para_transferi_ortak_wifi`, `topluluk_kume_no`, `merkeziyet_puani`
- **Jupyter Notebook (`gun_190_graf_dolandiricilik_sebekesi_analizi.ipynb`):**
  1. Kullanıcılar, cihazlar, kredi kartları ve ortak IP'lerin devasa bir graf olarak modellenmesi
  2. Louvain algoritması ile birbirine sıkı sıkıya bağlı organize dolandırıcılık çetelerinin kümelenmesi
  3. Çetenin elebaşı olan merkezi düğümün (High Degree & Betweenness Centrality) tespiti ve emniyet birimlerine bildirilmesi
- **Mülakat Sorusu:** Graf analizinde PageRank ve Betweenness Centrality metriklerinin dolandırıcılık şebekelerindeki kök hesapları bulmadaki matematiksel anlamı nedir?

### Gün 191: PyTorch DDP / Ray ile Dağıtık Tabular Model Eğitimi Simülasyonu
- **İş Alanı:** Büyük Veri Eğitimi & Dağıtık Yapay Zeka Altyapısı
- **Veri Kaynağı:** [10M+ Rows Synthetic Telco Big Dataset](https://www.kaggle.com/datasets)
- **Model:** PyTorch DistributedDataParallel (DDP) / Ray Train
- **Türkçe Değişkenler:** `gpu_rank_id`, `veri_parcasi_batch`, `gradyan_senkronizasyonu_all_reduce`, `egitim_hizlanma_orani`
- **Jupyter Notebook (`gun_191_dagitik_pytorch_ddp_egitimi.ipynb`):**
  1. 10 milyon satırlık büyük veri setinin parçalanarak (Data Parallelism) çoklu GPU simülasyonuna dağıtılması
  2. PyTorch DDP ile gradyanların All-Reduce operasyonuyla senkronize edilmesi
  3. Tek GPU'ya kıyasla 4x eğitim hızı kazanımının ve bellek kullanımının ölçülmesi
- **Mülakat Sorusu:** PyTorch'ta DataParallel (DP) ile DistributedDataParallel (DDP) arasındaki fark nedir ve Python GIL (Global Interpreter Lock) engeli DDP ile nasıl aşılır?

### Gün 192: Veri Kalitesi ve Şema Kayması (Data Drift / KS-Test) Takipçisi
- **İş Alanı:** MLOps İzleme & Veri Kalite Güvencesi (Evidently AI / Great Expectations)
- **Veri Kaynağı:** [Production Telemetry with Induced Data Drift](https://www.kaggle.com/datasets)
- **Model:** Kolmogorov-Smirnov (KS) Testi + Population Stability Index (PSI)
- **Türkçe Değişkenler:** `degisken_adi`, `referans_dagilim`, `canli_uretim_dagilimi`, `psi_degeri`, `veri_kaymasi_var_mi`
- **Jupyter Notebook (`gun_192_data_drift_psi_ks_testi.ipynb`):**
  1. Canlıya alınan modele gelen müşteri verilerinin dağılımının her hafta referans eğitim verisiyle karşılaştırılması
  2. Sayısal değişkenlerde KS-Test p-değeri ve PSI (Population Stability Index) hesaplanması
  3. PSI > 0.25 olduğunda modelin performansının düşeceği uyarısını verip yeniden eğitim alarmı açılması
- **Mülakat Sorusu:** Population Stability Index (PSI) formülü nedir ve PSI < 0.1, 0.1-0.25, >0.25 aralıklarının operasyonel anlamı nedir?

### Gün 193: Coğrafi Hücre Verilerini Uber H3 Hexagon ile Hiyerarşik İndeksleme
- **İş Alanı:** Turkcell Coğrafi Bilgi Sistemleri (GIS) & Uzamsal Büyük Veri
- **Veri Kaynağı:** [Cellular Tower Coordinates & Signal Heatmap](https://www.opencellid.org/)
- **Model:** Uber H3 Kütüphanesi (Spatial Indexing) + Geopandas + Kepler.gl
- **Türkçe Değişkenler:** `enlem_boylam`, `h3_index_res_7`, `h3_index_res_9`, `altigen_toplam_trafik_gb`
- **Jupyter Notebook (`gun_193_uber_h3_cografi_indeksleme.ipynb`):**
  1. Milyonlarca GPS noktasının Uber H3 altıgen indeks kodlarına dönüştürülmesi
  2. Farklı çözünürlüklerde (Resolution 7 ilçe bazlı, Resolution 9 sokak bazlı) hiyerarşik trafik toplulaştırması (Aggregation)
  3. SQL sorgularında ağır coğrafi kesişim (Polygon Intersection) işlemlerini O(1) integer eşleşmesine indirgeme
- **Mülakat Sorusu:** Coğrafi analizlerde kare ızgaralar yerine H3 altıgen ızgaralarının (Hexagonal Grid) tercih edilmesinin komşuluk mesafesi (Equidistant Neighbors) açısından geometrik nedeni nedir?

### Gün 194: ONNX Runtime ile Düşük Gecikmeli (<2ms) Model Servisleme
- **İş Alanı:** Canlı Çağrı Merkezi & Milisaniyelik Çıkarım (Low Latency Inference)
- **Veri Kaynağı:** [Pre-trained Churn & Fraud PyTorch/XGBoost Models](https://www.kaggle.com/datasets)
- **Model:** ONNX Runtime (C++ / Python Engine) + FP16 Quantization
- **Türkçe Değişkenler:** `model_onnx_dosyasi`, `giris_tensör_boyutu`, `saf_python_cikarim_suresi_ms`, `onnx_cikarim_suresi_ms`
- **Jupyter Notebook (`gun_194_onnx_runtime_hizli_servisleme.ipynb`):**
  1. PyTorch ve CatBoost modellerinin standart ONNX (Open Neural Network Exchange) formatına dönüştürülmesi
  2. ONNX Runtime optimize grafik motoru ile CPU üzerinde çıkarım süresinin 25 ms'den 1.8 ms'ye düşürülmesi
  3. FastAPI microservice ile yük testi (Load Testing) yapılarak saniyede 5000 istek karşılama testi
- **Mülakat Sorusu:** Derin öğrenme ve ağaç modellerinde ONNX dönüşümü sırasında yapılan Operator Fusion ve Constant Folding optimizasyonları nasıl çalışır?

### Gün 195: Event-Driven Fatura Kesim ve SMS Bildirim Akış Hattı (Kafka Simülasyonu)
- **İş Alanı:** Faturalama & Anlık Olay Güdümlü (Event-Driven) Mimariler
- **Veri Kaynağı:** [Simulated Telecom CDR & Billing Event Stream](https://www.kaggle.com/datasets)
- **Model:** Kafka Python (Producer/Consumer) + Faust Streaming Engine
- **Türkçe Değişkenler:** `olay_turu_arama_bitti_sms_atildi_kota_bitti`, `abone_id`, `kafka_topic_adi`, `anlik_fatura_tutari_tl`
- **Jupyter Notebook (`gun_195_kafka_event_driven_faturalama.ipynb`):**
  1. Abonenin yaptığı her arama ve internet kullanımının Kafka `telecom-cdr-events` konusuna akıtılması
  2. Faust akış işleyicisi ile abonenin kotasının bittiği milisaniyede anlık SMS bildirim olayının üretilmesi
  3. Ay sonu toplu fatura kesme yükünü ortadan kaldıran gerçek zamanlı faturalama mimarisi
- **Mülakat Sorusu:** Dağıtık mesajlaşma kuyruklarında (Kafka) "At-least-once" ile "Exactly-once" işleme garantisi (Idempotency) faturalama sistemlerinde nasıl sağlanır?

---

## 🌱 Modül 16: Sürdürülebilirlik, Yeşil Telekom & Enerji Verimliliği (Gün 196 – 200)

### Gün 196: Baz İstasyonu Güneş Paneli Üretimi ve Karbon Ayak İzi Tahmini
- **İş Alanı:** Turkcell Sürdürülebilirlik & Yenilenebilir Enerji Yönetimi
- **Veri Kaynağı:** [Solar Power Generation & Weather Telemetry Dataset](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- **Model:** CatBoost Regressor + Güneş Işınım Fiziği Modeli
- **Türkçe Değişkenler:** `gunes_paneli_id`, `gunes_isinim_degeri_w_m2`, `panel_sicakligi_c`, `gunluk_uretilen_kwh`, `onlenen_karbon_salinimi_kg`
- **Jupyter Notebook (`gun_196_gunes_paneli_karbon_ayak_izi.ipynb`):**
  1. Kırsal baz istasyonlarındaki güneş paneli üretim verilerinin hava durumu tahminleriyle birleştirilmesi
  2. İstasyonun ertesi gün şebekeden çekeceği elektrik ve güneşten karşılayacağı enerjinin tahmini
  3. Kapsama alanı başına engellenen karbon salınımının (Scope 1 & Scope 2) raporlanması
- **Mülakat Sorusu:** Güneş paneli verimlilik modellemesinde hava sıcaklığının artmasının panel voltajını düşürmesi (Negatif Sıcaklık Katsayısı) makine öğrenmesinde nasıl temsil edilir?

### Gün 197: Veri Merkezi PUE (Güç Kullanım Verimliliği) Optimizasyonu
- **İş Alanı:** Veri Merkezi Enerji Yönetimi & Yeşil Bilişim
- **Veri Kaynağı:** [Data Center PUE & Cooling Power Telemetry](https://www.kaggle.com/datasets)
- **Model:** Derin Pekiştirmeli Öğrenme (PPO - Proximal Policy Optimization) / Gradient Boosting
- **Türkçe Değişkenler:** `it_yuk_tuketimi_kw`, `sogutma_klima_gucu_kw`, `dis_ortam_hava_sicakligi_c`, `anlik_pue_degeri`, `onerilen_chiller_set_noktasi`
- **Jupyter Notebook (`gun_197_veri_merkezi_pue_optimizasyonu.ipynb`):**
  1. Veri merkezinin PUE (Power Usage Effectiveness = Toplam Güç / IT Gücü) metriğinin gerçek zamanlı modellenmesi
  2. Dış hava sıcaklığı düştüğünde klimaları kapatıp dış havayla soğutma (Free Cooling) kararı
  3. PUE değerini 1.45'ten 1.18'e düşürerek yıllık milyonlarca kWh elektrik tasarrufu sağlama
- **Mülakat Sorusu:** Veri merkezi soğutma optimizasyonunda PUE'yi düşürürken sunucuların güvenli çalışma sıcaklığı sınırını (ASHRAE standartları) korumak için kısıtlı optimizasyon (Constrained RL) nasıl kurulur?

### Gün 198: Akıllı Uyku Modu (Sleep Mode) ile Gece RAN Enerji Tasarrufu
- **İş Alanı:** 5G/4G Radyo Şebekesi Otomasyonu & Enerji Tasarrufu
- **Veri Kaynağı:** [Cellular Traffic Volume & Base Station Power Data](https://www.kaggle.com/datasets)
- **Model:** Q-Learning / Karar Ağacı (Kural Tabanlı Uyku Karar Motoru)
- **Türkçe Değişkenler:** `hucre_id`, `gece_saati_02_06`, `anlik_kullanici_sayisi`, `hucre_uyku_moduna_gecsin_mi`, `tasarruf_edilen_watt`
- **Jupyter Notebook (`gun_198_akilli_uyku_modu_ran_tasarrufu.ipynb`):**
  1. Gece 02:00 - 06:00 arasında trafiğin sıfıra yaklaştığı baz istasyonu hücrelerinin tespiti
  2. Kapsamayı tek bir ana hücreye devredip kapasite taşıyıcılarını (MIMO katmanlarını) uyku moduna (Micro Sleep) alma
  3. Ani bir acil çağrı veya trafik artışı olduğunda hücreyi milisaniyeler içinde uyandırma mekanizması
- **Mülakat Sorusu:** Baz istasyonlarında "Deep Sleep" modu ile "Micro Sleep" modu arasındaki uyanma gecikmesi ve enerji tasarrufu ödünleşimi nedir?

### Gün 199: Elektronik Atık (E-Waste) Eski Modem/Kart Parça Sınıflandırıcı
- **İş Alanı:** Döngüsel Ekonomi & İade Edilen Cihaz Yenileme (Refurbishment)
- **Veri Kaynağı:** [Roboflow - Electronic Waste & PCB Component Detection](https://universe.roboflow.com/)
- **Model:** YOLOv8 Object Detection & Classification
- **Türkçe Değişkenler:** `iade_modem_kart_fotografi`, `tespit_edilen_parcalar_kondansator_cip_port`, `yenilenebilir_mi_geri_donusum_mu`
- **Jupyter Notebook (`gun_199_elektronik_atik_modem_siniflandirma.ipynb`):**
  1. Abonelerin iade ettiği eski modem, TV+ kutusu ve ağ kartlarının kamera altında taranması
  2. Yanmış/şişmiş kondansatörler ile sağlam parçaların tespiti
  3. Sağlam cihazların temizlenip yeniden ekonomiye kazandırılması, bozukların değerli maden geri dönüşümüne ayrılması
- **Mülakat Sorusu:** E-atık ayrıştırmada optik konveyör bant üzerinde hareket eden nesnelerin gerçek zamanlı sınıflandırılmasında kameranın FPS ve tetikleme (Trigger) senkronizasyonu nasıl tasarlanır?

### Gün 200: Aşırı Hava Koşullarının (Fırtına/Kar) Şebeke Arıza Riskine Etkisi
- **İş Alanı:** Afet Yönetimi, Kriz Masası & Dayanıklı Şebeke Mühendisliği
- **Veri Kaynağı:** [Extreme Weather & Telecom Outage Correlation Dataset](https://www.kaggle.com/datasets)
- **Model:** XGBoost Classifier + SHAP Etkileşim Grafikleri + Meteoroloji API Entegrasyonu
- **Türkçe Değişkenler:** `bolge_id`, `beklenen_ruzgar_hizi_kmh`, `kar_kalinligi_cm`, `sicaklik_eksi_derece`, `sebeke_kesinti_ariza_riski_0_100`
- **Jupyter Notebook (`gun_200_asiri_hava_sebeke_ariza_riski.ipynb`):**
  1. MGM (Meteoroloji Genel Müdürlüğü) fırtına ve yoğun kar uyarıları ile geçmiş şebeke elektrik kesintilerinin birleştirilmesi
  2. Hangi ilçelerdeki baz istasyonlarının jeneratör yakıtına veya aküye düşme riski taşıdığının tahmini
  3. Fırtına öncesinde mobil jeneratör ve acil müdahale araçlarının riskli bölgelere önceden konuşlandırılması
- **Mülakat Sorusu:** Afet ve aşırı hava koşullarında telekom dayanıklılığını (Network Resilience) sağlamak için Makine Öğrenmesi destekli "Proaktif Afet Yönetimi" nasıl kurgulanır?

---

## 📜 Özel Lisans & Telif Hakkı

```
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
amaçlı olarak paylaşılmıştır.

YASAKLAR:
  1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
  2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
  3. Alt lisanslanamaz, satılamaz veya devredilemez.
  4. Tersine mühendislik yapılamaz.

İZİN VERİLEN KULLANIM:
  - GitHub üzerinde görüntüleme ve okuma.
  - Kişisel öğrenim amacıyla kodu inceleme (kopyalamadan).

YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
İzin talepleri için: GitHub @seydivakkas
```
