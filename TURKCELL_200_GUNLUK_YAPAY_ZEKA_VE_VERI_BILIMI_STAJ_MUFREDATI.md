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
