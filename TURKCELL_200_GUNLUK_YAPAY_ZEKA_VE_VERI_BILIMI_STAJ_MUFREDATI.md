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
- **Model:** LLM-as-a-Judge (Llama-3-70B / GPT-4o-mini Evaluation Prompting)
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

---

## 👁️ Modül 04: Bilgisayarlı Görü (Computer Vision) & Saha Denetimi (Gün 046 – 060)

### Gün 046: Baz İstasyonu Kule & Anten Nesne Tespiti
- **İş Alanı:** Saha Operasyonları & Altyapı Denetimi
- **Veri Kaynağı:** [Roboflow Universe - Telecom Tower Antenna Detection](https://universe.roboflow.com/)
- **Model:** YOLOv8n / YOLOv11 Object Detection
- **Türkçe Değişkenler:** `kule_goruntusu`, `tespit_edilen_anten_sayisi`, `sinirlayici_kutu_koordinatlari`, `guven_orani`
- **Jupyter Notebook (`gun_046_kule_anten_tespiti.ipynb`):**
  1. Roboflow üzerinden veri seti çekme ve YAML konfigürasyonu
  2. YOLOv8 ile kule ve anten nesneleri üzerinde model eğitimi
  3. mAP@0.5 ve mAP@0.5:0.95 metrik değerlendirmesi

### Gün 047: Kimlik Kartı & Pasaport Köşe Tespiti ve Segmentasyonu
- **İş Alanı:** Dijital Hat Açılış Süreci (e-KYC)
- **Veri Kaynağı:** [Roboflow - ID Card & Passport Segmentation](https://universe.roboflow.com/)
- **Model:** YOLOv8-Seg / OpenCV Perspective Warp
- **Türkçe Değişkenler:** `ham_kimlik_fotografi`, `kose_noktalari`, `perspektif_duzeltilmis_kimlik`
- **Jupyter Notebook (`gun_047_kimlik_segmentasyon_kyc.ipynb`)**

### Gün 048 – Gün 060 Hızlı Liste:
- **Gün 048:** Fatura / Fiş Bounding Box OCR Tespiti (Roboflow Invoice Extraction)
- **Gün 049:** Saha Ekibi İş Güvenliği (Baret / Yelek) Denetimi (Worker PPE Safety)
- **Gün 050:** Veri Merkezi Sunucu Kablo Hasar & Karmaşa Tespiti (Server Cable Defects)
- **Gün 051:** Turkcell Bayi İçi Müşteri Sayma & Yoğunluk Isı Haritası (People Counter)
- **Gün 052:** Baz İstasyonu Çevresi Yangın & Duman Erken Uyarısı (Wildfire & Smoke YOLO)
- **Gün 053:** SIM Kart Barkod & ICCID Seri Numarası Konumlandırma (Barcode/QR Detection)
- **Gün 054:** Güneş Enerjili İstasyonlarda Panel Kirlilik/Kırık Tespiti (Solar Panel Defects)
- **Gün 055:** Saha Araçları Otomatik Plaka Tanıma (ANPR YOLO + Tesseract OCR)
- **Gün 056:** Taranan Sözleşmelerde Islak İmza Eksikliği Denetimi (Signature Area Detection)
- **Gün 057:** Antenlerde Kuş Yuvası ve Engel Tespiti (Bird Nest & Hazard Detection)
- **Gün 058:** Mobil Uygulama Arayüz Hata (UI Glitch / Buton Kayması) Tespiti (UI Elements)
- **Gün 059:** Yüz Canlılık (Liveness / Anti-Spoofing) Tespiti (CelebA-Spoof / MiniFASNet)
- **Gün 060:** Açık Hava Billboard & Reklam Panosu Doğrulama (Billboard Verification)

---

## 💳 Modül 05: Fintek, Paycell & Fraud / Dolandırıcılık Tespiti (Gün 061 – 075)

### Gün 061: Kredi Kartı Dolandırıcılık Tespiti (Aşırı Dengesiz Veri)
- **İş Alanı:** Paycell Risk İzleme Masası
- **Veri Kaynağı:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Model:** XGBoost + Focal Loss / Autoencoder Reconstruction Error
- **Türkçe Değişkenler:** `islem_tutari`, `pca_bilesenleri`, `sahte_islem_etiketi`, `rekonstruksiyon_hatasi`
- **Jupyter Notebook (`gun_061_kredi_karti_fraud_tespiti.ipynb`):**
  1. %0.17 sınıf oranına sahip dengesiz verinin incelenmesi
  2. Autoencoder ile normal işlemlerin öğrenilmesi ve hata eşiği belirlenmesi
  3. Precision-Recall Curve (PR-AUC) optimizasyonu

### Gün 062: Mobil Para Transferi Sahtekarlık Modeli
- **İş Alanı:** Paycell P2P Transfer Güvenliği
- **Veri Kaynağı:** [Kaggle - PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Model:** CatBoost Classifier + Anomali Skorlama
- **Türkçe Değişkenler:** `gonderen_bakiye_oncesi`, `gonderen_bakiye_sonrasi`, `transfer_tutari`, `supheli_islem`
- **Jupyter Notebook (`gun_062_mobil_transfer_sahtekarlik.ipynb`)**

### Gün 063 – Gün 075 Hızlı Liste:
- **Gün 063:** Paycell Hazır Limit Kredi Risk Skoru (Home Credit Default Risk)
- **Gün 064:** Kara Para Aklama (AML) Şüpheli İşlem Ağ Analizi (IBM AML Data)
- **Gün 065:** Fiziksel Kiosk / ATM Nakit Talebi Tahmini (ATM Cash Forecasting)
- **Gün 066:** Mobil Ödeme Hata ve Reddetme Tahminleyici (Online Payment Failure)
- **Gün 067:** Üye İşyeri (Merchant) Chargeback Risk Puanlaması (Merchant Risk Data)
- **Gün 068:** Alternatif Telekom Verileriyle Kredi Notu Üretme (Credit Score Classification)
- **Gün 069:** Çoklu / Sahte Hesap (Sybil Attack) Tespiti (Fraudulent Account Clustering)
- **Gün 070:** Dijital Varlık / Kripto Volatilite Tahmini (G-Research Crypto Data)
- **Gün 071:** B2B Kurumsal Bayi Tahsilat Gecikmesi Tahmini (B2B Invoice Payment Delay)
- **Gün 072:** POS Harcama Coğrafi Anomali Dedektörü (Spatial Transaction Outliers)
- **Gün 073:** Otomatik Banka/Paycell Slip Harcama Kategorizasyonu (Transaction Tagging)
- **Gün 074:** Sadakat Puanı / Cashback İstismarı Tespiti (Loyalty Program Abuse)
- **Gün 075:** SIM Swap Sonrası Finansal İşlem Riski Modeli (Telecom-Banking Fraud)

---

## 🎧 Modül 06: Ses İşleme & Çağrı Analitiği (Audio AI) (Gün 076 – 085)

### Gün 076: Türkçe Konuşma Tanıma (ASR)
- **İş Alanı:** 532 Çağrı Merkezi Ses Kayıt Transkripsiyonu
- **Veri Kaynağı:** [HuggingFace - Mozilla Common Voice Turkish](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
- **Model:** OpenAI Whisper-Small / Wav2Vec2-XLSR-Turkish
- **Türkçe Değişkenler:** `ses_dosyasi_yolu`, `ornekleme_frekansi_hz`, `metne_dokulen_transkript`, `kelime_hata_orani_wer`
- **Jupyter Notebook (`gun_076_turkce_asr_whisper.ipynb`)**

### Gün 077 – Gün 085 Hızlı Liste:
- **Gün 077:** Çağrıda Müşteri Sesinden Duygu & Stres Tespiti (RAVDESS / CREMA-D)
- **Gün 078:** Ses Biyometrisi ile Müşteri Kimlik Doğrulama (VoxCeleb Speaker Verification)
- **Gün 079:** Ağ Arka Plan Gürültüsü Sınıflandırma (UrbanSound8K)
- **Gün 080:** VoLTE/VoIP Hatlarında Ses Kalitesi MOS Puanı Tahmini (NISQA Speech Quality)
- **Gün 081:** IVR Tek Kelimelik Sesli Komut Algılama (Speech Commands Dataset)
- **Gün 082:** Müşteri Temsilcisi Botu için Türkçe TTS (Metinden Sese) Sentezleme
- **Gün 083:** Çağrıda Konuşmacı Ayrıştırma (Speaker Diarization PyAnnote)
- **Gün 084:** Çağrı Merkezine Gelen Sentetik / Klon Ses (Deepfake Voice) Tespiti (ASVspoof)
- **Gün 085:** Çağrı Bekleme Müziği ve Sessizlik Süresi Ölçer (Music vs Speech)

---

## 🎬 Modül 07: Öneri Sistemleri, TV+, fizy & Dijital Servisler (Gün 086 – 095)

### Gün 086: fizy Kişiselleştirilmiş Çalma Listesi Öneri Motoru
- **İş Alanı:** fizy Müzik Servisi
- **Veri Kaynağı:** [Kaggle - Spotify Million Playlist Dataset](https://www.kaggle.com/datasets)
- **Model:** Implicit Collaborative Filtering (ALS / LightFM) + Matrix Factorization
- **Türkçe Değişkenler:** `kullanici_id`, `sarki_id`, `dinleme_sayisi`, `oneri_listesi`, `benzerlik_skoru`
- **Jupyter Notebook (`gun_086_fizy_muzik_oneri_motoru.ipynb`)**

### Gün 087 – Gün 095 Hızlı Liste:
- **Gün 087:** TV+ İçerik Tabanlı Film ve Dizi Öneri Sistemi (MovieLens / TMDB)
- **Gün 088:** Müzik Atlama (Skip) Davranışı Tahminleyici (Spotify Sequential Skip)
- **Gün 089:** BiP Çıkartma (Sticker) & Popülerlik Modeli (Social Media Virality)
- **Gün 090:** Video Akışında Uyarlanabilir Bant Genişliği ve QoS Optimizasyonu
- **Gün 091:** fizy Otomatik Spektrogram Tabanlı Müzik Türü & Mod Çıkarımı (GTZAN CNN)
- **Gün 092:** Oturum Tabanlı (Session-based) TV Programı Önerisi (GRU4Rec / RecSys)
- **Gün 093:** Game+ Bulut Oyun Ping ve En Yakın Sunucu Eşleme Modeli
- **Gün 094:** Dinleyici Yaş ve İlgi Alanı Demografik Tahmini (Last.fm Listening Habits)
- **Gün 095:** Podcast Bölümünden İlgi Çekici Anları Özetleme (Spotify Podcast Transcripts)

---

## 🌐 Modül 08: IoT, Akıllı Şehir & Edge AI (Gün 096 – 100)

- **Gün 096:** Akıllı Şehir Trafik Akışı & Araç Yoğunluğu Haritalama (Roboflow Traffic)
- **Gün 097:** Akıllı Sayaç (Elektrik/Su) Kaçak & Anomali Tespiti (London Smart Meter)
- **Gün 098:** IoT İstasyonları Hava Kirliliği (PM2.5) Zaman Serisi Tahmini
- **Gün 099:** Akıllı Otopark Doluluk Tespiti (PKLot Parking Space YOLO)
- **Gün 100:** Akıllı Tarım Toprak Nemi ve Sulama Karar Motoru (Soil Moisture IoT)

---

# 🚀 BÖLÜM 2: GÜN 101 – 200

## 📡 Modül 09: Telekom Şebeke Optimizasyonu, Radyo & 5G Altyapısı (Gün 101 – 115)

### Gün 101: 5G Massive MIMO Kanal Durum Bilgisi (CSI) Geri Bildirim Sıkıştırma
- **İş Alanı:** 5G Radyo Şebekesi & Anten Verimliliği
- **Veri Kaynağı:** [Kaggle / IEEE DataPort - Massive MIMO CSI Dataset](https://www.kaggle.com/datasets)
- **Model:** Complex-Valued Autoencoder (CsiNet)
- **Türkçe Değişkenler:** `anten_sayisi`, `alt_tasiyici_frekansi`, `ham_csi_matrisi`, `sikistirilmis_vektor`, `rekonstruksiyon_nmse`
- **Jupyter Notebook (`gun_101_5g_mimo_csi_sikistirma.ipynb`)**

### Gün 102 – Gün 115 Hızlı Liste:
- **Gün 102:** Mobil Ağ Hücresel Yük Devri (Handover) Başarı Tahmini (UCI Handover Traces)
- **Gün 103:** Radyo Erişim Şebekesi (RAN) Güç Tüketimi Tahminleyici (Tower Fuel Data)
- **Gün 104:** Baz İstasyonu Anten Açısı (Tilt/Azimuth) Sapma Tespiti (Radiation Pattern Data)
- **Gün 105:** SIM Kart Bağlantı Kopma Sıklığı Sınıflandırma (Disconnection Logs)
- **Gün 106:** WiFi - LTE/5G Otomatik Ağ Geçiş (Offloading) Karar Motoru (QoS Traces)
- **Gün 107:** 5G Ultra Düşük Gecikme (URLLC) Paket Kuyruk Analizi (Low Latency Data)
- **Gün 108:** Baz İstasyonları Arası Enterferans (SINR) Tahmini (SINR Interference Data)
- **Gün 109:** eSIM Profil İndirme Hata Oranı Kümeleme (Remote SIM Provisioning)
- **Gün 110:** Şehir Izgara Haritasında İnsan Yoğunluğu Çıkarımı (Spatio-Temporal Grid)
- **Gün 111:** Optik Sinyal-Gürültü Oranı (OSNR) Bozulma Öngörüsü (Fiber Telemetry)
- **Gün 112:** LTE Hız Kısıtlama (Throttling) Tespiti (ISP Throttling Traces)
- **Gün 113:** Araç İçi İletişim (V2X) İletim Gecikmesi Modellemesi (Connected Vehicles V2X)
- **Gün 114:** Telekom Veri Merkezi Sıcaklık Sensör Anomalileri Takibi
- **Gün 115:** Kapsama Alanı Olmayan Kör Nokta (Dead Zone) Haritalama (Signal Coverage)

---

## 💳 Modül 10: Fintek / Paycell, Dijital Cüzdan & Alternatif Risk (Gün 116 – 130)

- **Gün 116:** Dijital Cüzdan Bakiye Yetersizlik Tahmini (Mobile Wallet Insolvent Users)
- **Gün 117:** QR Kod ile Ödeme Dolandırıcılığı Dedektörü (QR Transaction Fraud)
- **Gün 118:** Fatura Taksitlendirme Geri Ödeme Skoru (Installment Repayment Risk)
- **Gün 119:** Ön Ödemeli Kart (Prepaid) İnaktif Kullanıcı Tahmini (Prepaid Churn)
- **Gün 120:** Sanal POS Başarılı Geçiş (Authorization Rate) Akıllı Yönlendirici (Payment Gateway)
- **Gün 121:** Sentetik Kimlik (Synthetic Identity) Dolandırıcılığı Tespiti (Identity Records)
- **Gün 122:** Paycell P2P Transfer Anomali Tespiti (Isolation Forest P2P)
- **Gün 123:** Çalıntı Kart Harcama Hızı (Velocity Fraud) Modeli (Velocity Checks)
- **Gün 124:** Üye İşyeri (Merchant) Günlük Ciro Tahminleme Modeli (Revenue Forecast)
- **Gün 125:** Otomatik Fatura Talimatı İptal Riski Puanlama (Auto-Debit Churn)
- **Gün 126:** Kurumsal Şirket Hatları Harcama Limiti Dinamik Hesaplayıcı
- **Gün 127:** Dijital Altın / Döviz Alım-Satım Eğilimi Tahmini (Retail FX Behavior)
- **Gün 128:** Fiziksel POS Terminal Donanım Arıza Öngörüsü (POS Telemetry)
- **Gün 129:** Harcama Lokasyonu ile Hücresel Konum Uyuşmazlığı Modeli (Geo-Fraud)
- **Gün 130:** Sadakat Puanı / Hediye Bakiye Suiistimal Dedektörü (Cashback Abuse)

---

## 📱 Modül 11: Dijital Servisler (TV+, fizy, lifebox, BiP, Dergilik) (Gün 131 – 145)

- **Gün 131:** fizy Şarkı Benzerliği Vektör Arama Motoru (FAISS / Qdrant Audio Embeddings)
- **Gün 132:** TV+ Video Başlangıç Gecikmesi (Startup Latency) Regresyonu
- **Gün 133:** lifebox Fotoğraf Otomatik Albümleme & Sahne Sınıflandırma (Places365 CNN)
- **Gün 134:** TV+ İzleyici Diziyi Bırakma (Drop-off) Dakikası Tahmini
- **Gün 135:** fizy Çalma Listesi Devam Ettirme Modeli (Sequential Playlist Continuation)
- **Gün 136:** lifebox Tekrarlanan / Kopya Fotoğrafları Temizleme (Perceptual Hashing)
- **Gün 137:** Dergilik / Dijital Yayın İlgi Alanı Kişiselleştirme (MIND News Recommendation)
- **Gün 138:** TV+ Altyazı ve Ses Senkronizasyon Kayması Dedektörü
- **Gün 139:** BiP Sesli Mesaj Arka Plan Gürültüsü Temizleme (U-Net Denoising)
- **Gün 140:** TV+ Canlı Yayın Eşzamanlı İzleyici (CCU) Yük Tahmini (Live TV Viewership)
- **Gün 141:** fizy Podcast Açıklamalarından Otomatik Kategori Çıkarma
- **Gün 142:** lifebox Belge/Fiş Tarama Otomatik Perspektif Düzeltme (Dewarping YOLO)
- **Gün 143:** Game+ Bulut Oyun Paket Kaybı Tolerans ve Bitrate Ayarlayıcı
- **Gün 144:** TV+ Dizi/Film Fragman Heyecan Puanlaması (Trailer Emotion)
- **Gün 145:** fizy Türkçe Şarkı Sözlerinden Ruh Hali (Mood) Çıkarımı

---

## 🤖 Modül 12: İleri Seviye NLP, LLM, Müşteri Deneyimi & Agentic AI (Gün 146 – 160)

- **Gün 146:** Müşteri Temsilcisi Yanıt Kalitesi Denetleyicisi (LLM-as-a-Judge)
- **Gün 147:** Telekom Terimleri için Alana Özel (Domain-Specific) Word2Vec / FastText
- **Gün 148:** Fatura PDF'lerinden Yapılandırılmış JSON Çıkaran LLM Ajanı (LangChain)
- **Gün 149:** Canlı Sohbet Müşteri Sinir Seviyesi (Frustration) İzleyici
- **Gün 150:** Şikayet Metinlerinden Kök Neden Hiyerarşisi Çıkarma (Complaint Hierarchy)
- **Gün 151:** RAG için Hibrit Vektör + BM25 Arama Motoru (Hybrid Search)
- **Gün 152:** Çağrı Metninden Kampanya Kabul İhtimali Puanlama (Telemarketing Text)
- **Gün 153:** Otomatik Tarife Detay Özeti Üretici (Text Summarization)
- **Gün 154:** Sosyal Medya Rakip Operatör Kampanya Karşılaştırma Analizörü
- **Gün 155:** Sesli Yanıt (IVR) Fonetik Benzerlik Eşleştirici (Double Metaphone)
- **Gün 156:** Abonelik Sözleşmesi Cayma Bedeli ve Taahhüt Maddesi Bulucu
- **Gün 157:** Chatbot için Few-Shot Niyet Genişletici Sentetik Veri Pipeline'ı
- **Gün 158:** Boyut Tabanlı Müşteri Memnuniyetsizliği (Aspect-Based Sentiment: Hız, Fiyat, Destek)
- **Gün 159:** E-posta Destek Talebi Otomatik Cevap Taslağı Üretici
- **Gün 160:** Müşteri İletişim Dili (Resmi vs Samimi) Belirleme ve Ton Eşleme

---

## 🛡️ Modül 13: Bilgisayarlı Görü, Saha Operasyonları & Güvenlik (Gün 161 – 175)

- **Gün 161:** Saha Teknisyenleri Düşme / Hareketsizlik Algılama (Worker Fall Detection)
- **Gün 162:** Kule Tırmanış Emniyet Kemeri (Harness) Takma Denetimi (YOLO Safety)
- **Gün 163:** Optik Fiber Ekleme Noktası (Splice) Mikroskobik Kusur Tespiti
- **Gün 164:** Sokak Kameralarından Açık/Kırık Menhol Kapağı Tespiti
- **Gün 165:** Kırsal İstasyonlarda Tel Örgü İhlali & İzinsiz Giriş Algılama
- **Gün 166:** Sunucu Odası Yangın Tüpü ve Acil Çıkış Engel Denetimi
- **Gün 167:** Dijital Hat Başvurusunda Canlı Selfie ve Kimlik Fotoğrafı Doğrulama (Facenet)
- **Gün 168:** Bayi Raf Standı Planogram Uyumluluk Kontrolü (Retail Shelf Compliance)
- **Gün 169:** SIM Kart Çip Çizik ve Kusur Tespiti (PCB Defect Detection)
- **Gün 170:** Baz İstasyonu Jeneratör Yağ/Yakıt Sızıntısı Tespiti (Industrial Leak Detection)
- **Gün 171:** Fırtınada Kule Rüzgar Salınımı ve Yapısal Eğrilik Ölçümü
- **Gün 172:** Aşınmış ve Hasarlı Karekod Düzeltme & Okuma
- **Gün 173:** Veri Merkezi Kabinet Kapak Açık Unutulma Dedektörü
- **Gün 174:** Saha Projektör ve Gece Aydınlatma Arıza Tespiti
- **Gün 175:** Drone Görüntüsünden Kule Paslanma ve Korozyon Analizi (Rust Segmentation)

---

## 🔒 Modül 14: Siber Güvenlik, Ağ Savunması & Tehdit İstihbaratı (Gün 176 – 185)

- **Gün 176:** Botnet Komuta Kontrol (C2) Periyodik Ağ Sinyali Tespiti (CTU-13 Dataset)
- **Gün 177:** Web Uygulaması API İstismar (Exploit) Tespiti (CSIC HTTP Dataset)
- **Gün 178:** VPN ve Tor Anonim Ağ Trafiği Sınıflandırma (ISCX Tor Dataset)
- **Gün 179:** Güvenlik Duvarı Loglarından Port Tarama (Port Scan) Tespiti
- **Gün 180:** Sahte Baz İstasyonu (IMSI Catcher / Stingray) Sinyal Avcısı
- **Gün 181:** Bellek Dökümünden (Memory Dump) Zararlı Yazılım Tespiti (CIC-MalMem-2022)
- **Gün 182:** Şüpheli İç Tehdit (Insider Threat) Davranış Analizi (CERT Dataset)
- **Gün 183:** SSH / RDP Kaba Kuvvet (Brute Force) Saldırı Dedektörü
- **Gün 184:** Açık Kaynak Git Depolarında Sızdırılmış API Key & Token Avcısı
- **Gün 185:** DGA (Domain Generation Algorithm) ile Üretilen Sahte Alan Adı Tespiti

---

## ⚙️ Modül 15: MLOps, Veri Mühendisliği & Dağıtık Akış (Gün 186 – 195)

- **Gün 186:** Feature Store (Feast) ile Canlı Müşteri Öznitelik Deposu
- **Gün 187:** Streaming K-Means ile Bellek Üzerinde Ağ Paket Kümeleme
- **Gün 188:** Delta Lake ile PostgreSQL Veritabanı CDC Pipeline Simülasyonu
- **Gün 189:** MLflow ile Model Sürümleme ve Otomatik A/B Testi
- **Gün 190:** Graf Tabanlı Dolandırıcılık Şebekesi Analizi (NetworkX / Neo4j)
- **Gün 191:** PyTorch DDP / Ray ile Dağıtık Tabular Model Eğitimi Simülasyonu
- **Gün 192:** Veri Kalitesi ve Şema Kayması (Data Drift / KS-Test) Takipçisi
- **Gün 193:** Coğrafi Hücre Verilerini Uber H3 Hexagon ile Hiyerarşik İndeksleme
- **Gün 194:** ONNX Runtime ile Düşük Gecikmeli (<2ms) Model Servisleme
- **Gün 195:** Event-Driven Fatura Kesim ve SMS Bildirim Akış Hattı (Kafka Simülasyonu)

---

## 🌱 Modül 16: Sürdürülebilirlik, Yeşil Telekom & Enerji Verimliliği (Gün 196 – 200)

- **Gün 196:** Baz İstasyonu Güneş Paneli Üretimi ve Karbon Ayak İzi Tahmini
- **Gün 197:** Veri Merkezi PUE (Güç Kullanım Verimliliği) Optimizasyonu
- **Gün 198:** Akıllı Uyku Modu (Sleep Mode) ile Gece RAN Enerji Tasarrufu
- **Gün 199:** Elektronik Atık (E-Waste) Eski Modem/Kart Parça Sınıflandırıcı (YOLO)
- **Gün 200:** Aşırı Hava Koşullarının (Fırtına/Kar) Şebeke Arıza Riskine Etkisi

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
