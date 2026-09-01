# 📱 TURKCELL YAPAY ZEKA, VERİ BİLİMİ VE BÜYÜK VERİ MÜHENDİSLİĞİ 300 GÜNLÜK STAJ & PORTFÖY HAFIZA PLANI

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Kurum: Turkcell](https://img.shields.io/badge/Kurumsal-Turkcell%20%7C%20Paycell%20%7C%20fizy%20%7C%20TV%2B%20%7C%20BiP-blue.svg?style=flat-square)](https://www.turkcell.com.tr/)
[![Veri Kümeleri: Açık Kaynak](https://img.shields.io/badge/Veri%20Kaynaklar%C4%B1-Kaggle%20%7C%20HuggingFace%20%7C%20Roboflow%20%7C%20UCI-green.svg?style=flat-square)](https://kaggle.com)
[![Format: Python + Jupyter .ipynb](https://img.shields.io/badge/Format-Python%203.11%20%2B%20Jupyter%20.ipynb-orange.svg?style=flat-square)](https://jupyter.org)

---

## 🎯 Programın Amacı & Kapsamı

Bu hafıza dokümanı, **Turkcell'in 6 ana iş kolunda** (Telekomünikasyon Şebekesi, Fintek/Paycell, Dijital Medya fizy/TV+, NLP/Müşteri Deneyimi, Bilgisayarlı Görü ve MLOps/Büyük Veri) staj ve junior/mid veri bilimci pozisyonlarında doğrudan fark yaratacak, internetteki açık kaynaklı gerçek veri setleriyle birebir doğrulanmış **300 adet uygulamalı mini projenin** mimarisini, veri kaynaklarını, Türkçe değişken adlarını ve `.ipynb` şablonlarını içerir.

Her proje şu standart bileşenlerle inşa edilir:
1. **Veri Kümesi (Kaggle / Hugging Face / Roboflow / UCI)**
2. **Algoritmik / İstatistiksel Model**
3. **%100 Anlaşılır Türkçe Değişken ve Fonksiyon Adları**
4. **Jupyter Notebook (`.ipynb`) Yapısı ve Çalışma Akışı**
5. **Turkcell Staj & Mülakat Odaklı Değerlendirme Sorusu**

> [!TIP]
> **💻 %100 YEREL (LOCAL), ÜCRETSİZ VE AÇIK KAYNAK GÜVENCESİ:**
> Bu müfredattaki 300 projenin tamamı, öğrencinin/mühendisin kendi yerel bilgisayarında (CPU veya standart GPU) ve ücretsiz Google Colab / Kaggle ortamlarında **sıfır maliyetle ($0)** çalışacak şekilde tasarlanmıştır. Hiçbir projede ücretli API anahtarı (OpenAI, Anthropic, Google Cloud Paid API vb.) veya ağır üretken LLM donanım/çalışma gereksinimi yoktur. Tüm NLP ve veri bilimi görevleri açık kaynak, deterministik, hafif modeller (BERTurk, DeBERTa, Sentence-Transformers, FastText, Scikit-learn, Regex) ile doğrudan CPU/GPU üzerinde milisaniyeler içinde yürütülür.

---


# 📚 200 GÜNLÜK MÜFREDAT MODÜL DAĞILIMI

```
[BÖLÜM 1: GÜN 1 - 100]
├── Modül 01: Müşteri Analitiği, Churn, CRM & Gelir Optimizasyonu (Gün 001 - 015)
├── Modül 02: Şebeke, Ağ Trafiği & Zaman Serileri (Gün 016 - 030)
├── Modül 03: Doğal Dil İşleme (NLP), Müşteri Hizmetleri & Semantik Arama (Gün 031 - 045)
├── Modül 04: Bilgisayarlı Görü (Computer Vision) & Saha Denetimi (Gün 046 - 060)
├── Modül 05: Fintek, Paycell & Fraud / Dolandırıcılık Tespiti (Gün 061 - 075)
├── Modül 06: Ses İşleme & Çağrı Analitiği (Audio AI) (Gün 076 - 085)
├── Modül 07: Öneri Sistemleri, TV+, fizy & Dijital Servisler (Gün 086 - 095)
└── Modül 08: IoT, Akıllı Şehir & Edge AI (Gün 096 - 100)

[BÖLÜM 2: GÜN 101 - 200]
├── Modül 09: Telekom Şebeke Optimizasyonu, Radyo & 5G Altyapısı (Gün 101 - 115)
├── Modül 10: Fintek / Paycell, Dijital Cüzdan & Alternatif Risk (Gün 116 - 130)
├── Modül 11: Dijital Servisler (TV+, fizy, lifebox, BiP, Dergilik) (Gün 131 - 145)
├── Modül 12: İleri Seviye NLP, Semantik Arama, Metin Madenciliği & Müşteri Deneyimi (Gün 146 - 160)
├── Modül 13: Bilgisayarlı Görü, Saha Operasyonları & Güvenlik (Gün 161 - 175)
├── Modül 14: Siber Güvenlik, Ağ Savunması & Tehdit İstihbaratı (Gün 176 - 185)
├── Modül 15: MLOps, Veri Mühendisliği & Dağıtık Akış (Gün 186 - 195)
└── Modül 16: Sürdürülebilirlik, Yeşil Telekom & Enerji Verimliliği (Gün 196 - 200)

[BÖLÜM 3: GÜN 201 - 300]
├── Modül 17: Deterministik & Yerel Akıllı Ajanlar (Local Agentic AI, NOC & SOC Otomasyonu) (Gün 201 - 220)
├── Modül 18: 5G Advanced, O-RAN & Akıllı Şebeke Optimizasyonu (Simülasyon & Hafif RL) (Gün 221 - 235)
├── Modül 19: Hafif Çok Modlu (Multimodal) Çıkarım & Yerel Görü-Metin Füzyonu (Gün 236 - 250)
├── Modül 20: Edge AI, TinyML & Modem/CPE Üzeri Gömülü Yapay Zeka (Gün 251 - 265)
├── Modül 21: Büyük Ölçekli Grafik Sinir Ağları (GNN) & NetworkX Topoloji Analizi (Gün 266 - 280)
└── Modül 22: Güvenilir AI, XAI, Kuantum Hazırlığı & 5 Büyük Mezuniyet Capstone (Gün 281 - 300)
```

---

# 🚀 BÖLÜM 1: GÜN 001 – 100

## 📊 Modül 01: Müşteri Analitiği, Churn & CRM (Gün 001 – 015)

### Gün 001: Telco Müşteri Kayıp (Churn) Tahmini
- **İş Alanı:** Turkcell Bireysel Müşteri Analitiği & Terk Önleme Masası
- **Veri Kaynağı:** [Kaggle - Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Model:** CatBoostClassifier / XGBoost + SHAP Açıklanabilirlik
- **Türkçe Değişkenler:** `musteri_id`, `sozlesme_turu`, `aylik_odeme_tutari`, `toplam_harcama`, `ayrilma_riski_orani`, `model_tahmini`
- **Jupyter Notebook (`gun_001_telco_churn_tahmini.ipynb`):**
  1. Telekom abonelik, tarife ve fatura verilerinin yüklenip eksik değerlerin imputasyonu
  2. Kategorik değişkenlerin One-Hot & Target Encoding dönüşümleri ve SMOTE sınıf dengelemesi
  3. CatBoost eğitimi, ROC-AUC / PR-AUC optimizasyonu ve SHAP değişken önem sıralaması
- **Mülakat Sorusu:** Aşırı dengesiz churn veri setlerinde neden doğruluk (Accuracy) metriği yerine PR-AUC ve F1-Score tercih edilmelidir?

### Gün 002: Müşteri Yaşam Boyu Değeri (CLTV) Modellemesi
- **İş Alanı:** Pazarlama & Abone Değeri Gelir Planlama
- **Veri Kaynağı:** [Kaggle - Online Retail & Customer Analytics](https://www.kaggle.com/datasets/vijayuv/onlineretail)
- **Model:** BG/NBD (Beta-Geometric / Negative Binomial) + Gamma-Gamma Monetization Modeli
- **Türkçe Değişkenler:** `abone_id`, `islem_sikligi`, `musteri_yasi_hafta`, `ortalama_fatura_tutari`, `beklenen_gelecek_gelir`
- **Jupyter Notebook (`gun_002_musteri_yasam_boyu_degeri.ipynb`):**
  1. RFM (Recency, Frequency, Monetary) telekom özniteliklerinin çıkarılması
  2. BG/NBD ile 3 ve 6 aylık beklenen işlem frekansı tahmini
  3. Gamma-Gamma modeli ile müşteri başına net marjinal kâr hesabı ve değer segmentasyonu
- **Mülakat Sorusu:** BG/NBD modelinde "Müşterinin Canlı Olma Olasılığı" (Probability of Being Alive - P_Alive) formülü nasıl çalışır?

### Gün 003: RFM Tabanlı Abone Segmentasyonu
- **İş Alanı:** Turkcell CRM & Hedefli Kampanya Yönetimi
- **Veri Kaynağı:** [Kaggle - Credit Card Customer Segmentation](https://www.kaggle.com/datasets/arjunbhasin2005/ccdata)
- **Model:** K-Means Kümeler + UMAP / PCA Boyut İndirgeme
- **Türkçe Değişkenler:** `son_islem_gunu`, `islem_adedi`, `toplam_odeme_tutari`, `kume_etiketi`, `segment_adi`
- **Jupyter Notebook (`gun_003_rfm_abone_segmentasyonu.ipynb`):**
  1. Logaritmik dönüşüm ve StandardScaler normalizasyonu ile veri önişleme
  2. Elbow yöntemi ve Silhouette skoru ile optimal K küme sayısının seçimi
  3. Segment profilleme (Şampiyonlar, Sadık Aboneler, Uyuyanlar, Riskli Grup) ve görselleştirme
- **Mülakat Sorusu:** K-Means kümelemesinde aykırı değerlerin (Outliers) küme merkezlerini saptırmasını engellemek için K-Medoids veya DBSCAN nasıl tercih edilir?

### Gün 004: Faturasızdan Faturalıya Tarife Terfi (Upselling) Modeli
- **İş Alanı:** Satış & Dijital Kanal Gelir Artırma Masası
- **Veri Kaynağı:** [Kaggle - Bank Marketing / Product Upsell](https://www.kaggle.com/datasets/henriqueyama/bank-marketing)
- **Model:** LightGBM + Optuna Hiperparametre Optimizasyonu
- **Türkçe Değişkenler:** `faturasiz_kullanim_suresi_ay`, `ortalama_tl_yukleme`, `kota_asimi_sikligi`, `faturali_gecis_egilimi`
- **Jupyter Notebook (`gun_004_tarife_terfi_upsell.ipynb`):**
  1. Faturasız hatların paket doluluk oranı ve veri tüketim trend analizi
  2. Optuna ile LightGBM hiperparametre araması ve Cross-Validation
  3. Faturalıya geçiş ihtimali en yüksek %10'luk kitle için hedefli kampanya eşik optimizasyonu
- **Mülakat Sorusu:** Pazarlama kampanyalarında "Cumulative Gains Chart" ve "Decile Table" analizi hedef kitle seçiminde nasıl kullanılır?

### Gün 005: Net Promoter Score (NPS) / Memnuniyet Tahmini
- **İş Alanı:** Müşteri Deneyimi Yönetimi (CEM) & Memnuniyetsizlik Erken Teşhisi
- **Veri Kaynağı:** [Kaggle - Customer Satisfaction Dataset](https://www.kaggle.com/datasets/santander-customer-satisfaction)
- **Model:** Random Forest Regressor & Ordinal Regression
- **Türkçe Değişkenler:** `cagri_merkezi_arama_sayisi`, `baglanti_kopma_adedi`, `fatura_itiraz_durumu`, `tahmini_nps_puani`
- **Jupyter Notebook (`gun_005_nps_memnuniyet_tahmini.ipynb`):**
  1. Çağrı merkezi kayıtları, şebeke kopma logları ve fatura itirazlarının birleştirilmesi
  2. Random Forest Regressor ile 0-10 arası tahmini NPS puanı modellemesi
  3. Memnuniyetsiz (Detractor: 0-6 puan) aboneler için müşteri hizmetlerine otomatik alarm düşürme
- **Mülakat Sorusu:** NPS gibi sıralı kategorik (Ordinal) hedef değişkenlerde standart Regresyon yerine Ordinal Logistic Regression kullanmanın avantajı nedir?

### Gün 006: Faturasız Hat TL/Paket Yükleme Zamanı Tahmini
- **İş Alanı:** Paycell & Dijital Operatör TL Yükleme Masası
- **Veri Kaynağı:** [Kaggle - Mobile Money Transaction](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Model:** Yaşam Analizi (Survival Analysis - Cox Proportional Hazards) / XGBoost Regressor
- **Türkçe Değişkenler:** `kalan_tl_bakiyesi`, `son_yukleme_uzerinden_gecen_gun`, `ortalama_harcama_hizi`, `tahmini_gelecek_yukleme_gunu`
- **Jupyter Notebook (`gun_006_tl_yukleme_zamani_tahmini.ipynb`):**
  1. Abonenin günlük bakiye erime hızı ve geçmiş yükleme aralıklarının hesaplanması
  2. Cox PH yaşam analizi ile bakiyenin sıfırlanacağı günün tahmini
  3. Hat kapanmadan 24 saat önce aboneye kişiselleştirilmiş indirimli paket teklif SMS'i tetikleme
- **Mülakat Sorusu:** Faturasız abonelerin paket yenileme döngüsünde "Sağdan Sansürlü Veri" (Censored Data) kavramı nasıl ele alınır?

### Gün 007: Fatura Ödeme Gecikmesi Tahminleyicisi
- **İş Alanı:** Finans & Alacak Risk Yönetimi
- **Veri Kaynağı:** [Kaggle - Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Model:** XGBoost + Cost-Sensitive Learning (Maliyet Duyarlı Öğrenme)
- **Türkçe Değişkenler:** `gecikmis_fatura_adedi`, `son_3_ay_ortalama_fatura`, `fatura_tutari`, `gecikme_olasiligi_skoru`
- **Jupyter Notebook (`gun_007_fatura_odeme_gecikmesi.ipynb`):**
  1. Geçmiş fatura ödeme disiplini, gecikme gün sayıları ve son fatura tutarının modellenmesi
  2. Borcunu geciktirecek abonelerin sınıflandırılmasında maliyet matrisi optimizasyonu
  3. Riskli abonelere vade gününden 2 gün önce otomatik hatırlatma bildirimi gönderilmesi
- **Mülakat Sorusu:** Cost-Sensitive Learning yaklaşımında yanlış negatif (gecikecek faturayı tahmin edememe) maliyeti modele nasıl ağırlık olarak verilir?

### Gün 008: Müşteri İtiraz (Dispute) Olasılığı Modeli
- **İş Alanı:** Fatura İtiraz & Şikayet Önleme Masası
- **Veri Kaynağı:** [Kaggle - Consumer Complaint Database](https://www.kaggle.com/datasets/selener/consumer-complaint-database)
- **Model:** Logistic Regression & Gradient Boosting (GBDT)
- **Türkçe Değişkenler:** `aylik_fatura_artis_orani`, `yurt_disi_roaming_harcamasi`, `paket_asimi_tutari`, `itiraz_riski_puani`
- **Jupyter Notebook (`gun_008_musteri_itiraz_modeli.ipynb`):**
  1. Beklenmedik yüksek fatura (Bill Shock - Roaming / Kota Aşımı) yaşayan abonelerin tespiti
  2. Fatura kesim anında itiraz etme olasılığının hesaplanması
  3. İtiraz riski yüksek müşterilere fatura açıklandığında otomatik açıklayıcı harcama dökümü iletilmesi
- **Mülakat Sorusu:** "Bill Shock" (Fatura Şoku) vakalarında açıklanabilirlik için SHAP Force Plot grafikleri temsilci ekranında nasıl kullanılır?

### Gün 009: Cihaz Yenileme (Handset Upgrade) Eğilimi
- **İş Alanı:** Pasaj (Turkcell E-Ticaret) & Cihaz Kampanyaları
- **Veri Kaynağı:** [Kaggle - Mobile Phone Usage Dataset](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset)
- **Model:** Random Forest Classifier + Feature Importance
- **Türkçe Değişkenler:** `mevcut_cihaz_yasi_ay`, `batarya_saglik_skoru`, `veri_kullanim_artisi`, `cihaz_yenileme_olasiligi`
- **Jupyter Notebook (`gun_009_cihaz_yenileme_egilimi.ipynb`):**
  1. Abonenin şebekeye bağlandığı telefonun model yılı, donanım yetersizlikleri ve kullanım süresinin analizi
  2. 24 aydan eski telefon kullanan abonelerin yeni model alma eğilimlerinin sınıflandırılması
  3. Pasaj üzerinde bütçesine uygun taksitli cihaz teklifinin kişiselleştirilmesi
- **Mülakat Sorusu:** Cihaz yenileme tahmininde "Look-alike Modeling" (yeni telefon alan müşterilere benzer profilleri bulma) tekniği nasıl uygulanır?

### Gün 010: Abonelik İptal Nedenlerini Sınıflandırma
- **İş Alanı:** Müşteri Kazanım & İkna / Retention Masası
- **Veri Kaynağı:** [Kaggle - Subscription Churn Telecom](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Model:** Multi-Class LightGBM (Çok Sınıflı Ayrıştırma)
- **Türkçe Değişkenler:** `iptal_gerekce_kodu`, `rakip_operatore_gecis`, `fiyat_kaynakli_iptal`, `cekim_gucu_sorunu`, `hizmet_memnuniyetsizligi`
- **Jupyter Notebook (`gun_010_iptal_nedenleri_siniflandirma.ipynb`):**
  1. Ayrılmak isteyen müşterilerin iptal gerekçelerinin (Fiyat, Şebeke, Rakip Kampanya) çok sınıflı etiketlenmesi
  2. LightGBM Multi-Class modeli ile ayrılma nedeninin önceden tahmini
  3. İkna temsilcisine müşterinin asıl rahatsızlığına özel (Örn: Şebeke sorunuysa ek anten/indirim, fiyatsa alt tarife) teklif sunulması
- **Mülakat Sorusu:** Multi-class sınıflandırmada Log-Loss (Cross-Entropy) kaybı ile One-vs-Rest (OvR) yaklaşımı arasındaki hesaplama ve ayrım gücü farkı nedir?

### Gün 011: Çapraz Satış (Cross-Selling) Modeli (TV+, Superonline, Paycell)
- **İş Alanı:** Çoklu Ürün Stratejisi & Müşteri Başına Ortalama Gelir (ARPU)
- **Veri Kaynağı:** [Kaggle - Multi-Product Financial/Telecom Data](https://www.kaggle.com/datasets)
- **Model:** Multi-Output Classifier / Stacking Ensemble
- **Türkçe Değişkenler:** `ev_interneti_aktif`, `fizy_kullanimi_saat`, `paycell_islem_hacmi`, `tvplus_satin_alma_ihtimali`
- **Jupyter Notebook (`gun_011_capraz_satis_cross_sell.ipynb`):**
  1. Sadece GSM kullanan abonelerin dijital servis (TV+, fizy, Superonline) kullanım potansiyelinin analizi
  2. Multi-Output Classifier ile her bir yan ürün için ayrı ayrı satın alma olasılığı üretimi
  3. Sepet analizi (Market Basket Analysis) ile birlikte satılma olasılığı yüksek ikili ürün paketleri tasarımı
- **Mülakat Sorusu:** Çapraz satışta "Next Best Action" (NBA) karar motoru oluştururken ürün kâr marjı ile satın alma olasılığı nasıl ağırlıklandırılır?

### Gün 012: Müşteri Kayıp Riski Erken Uyarı Motoru
- **İş Alanı:** Gerçek Zamanlı CRM & Davranışsal Değişim Dedektörü
- **Veri Kaynağı:** [Kaggle - Telecom Churn BigML](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)
- **Model:** Z-Score Anomalisi + Karar Ağacı
- **Türkçe Değişkenler:** `son_30_gun_veri_degisimi`, `arama_suresi_dusus_orani`, `sms_sayisi_azalma`, `erken_uyari_tetiklendi`
- **Jupyter Notebook (`gun_012_churn_erken_uyari_motoru.ipynb`):**
  1. Abonenin geçmiş 6 aylık ortalama tüketimi ile son 2 haftalık tüketimi arasındaki Z-Score sapmalarının hesaplanması
  2. İkinci SIM kart takarak Turkcell hattını pasifleştiren kullanıcıların anında tespiti
  3. Hat tamamen kapatılmadan 15 gün önce proaktif müdahale mekanizması
- **Mülakat Sorusu:** Zaman serisinde hareketli ortalama (Rolling Average) sapmalarını takip ederek trend kırılmalarını tespit etmenin avantajı nedir?

### Gün 013: Fiyat Esnekliği (Price Elasticity of Demand) Analizi
- **İş Alanı:** Gelir Yönetimi & Dinamik Tarife Fiyatlandırma
- **Veri Kaynağı:** [Kaggle - Telecom Pricing & Demand](https://www.kaggle.com/datasets)
- **Model:** Log-Log OLS Regresyonu (Ekonometrik Talep Eğrisi)
- **Türkçe Değişkenler:** `paket_fiyat_artisi_yuzde`, `talep_degisimi_yuzde`, `fiyat_esneklik_katsayisi`, `gelir_maksimizasyon_fiyati`
- **Jupyter Notebook (`gun_013_fiyat_esnekligi_analizi.ipynb`):**
  1. Geçmiş tarife zamları ve paket satış adetlerinin logaritmik dönüşümü
  2. Log-Log regresyon eğimi ile esneklik katsayısının ($E_d = \% \Delta Q / \% \Delta P$) hesaplanması
  3. Toplam geliri maksimize eden optimum paket fiyat noktasının simülasyonu
- **Mülakat Sorusu:** Fiyat esnekliği $E_d < -1$ (Esnek Talep) olan bir pakette fiyat artışı yapıldığında şirketin toplam gelirine ne olur?

### Gün 014: Pasifleşen (Dormant) Hatları Geri Kazanım Modeli
- **İş Alanı:** Yeniden Etkinleştirme & Causal Inference
- **Veri Kaynağı:** [Kaggle - Subscription Reactivation](https://www.kaggle.com/datasets)
- **Model:** Uplift Modeling (CausalML / Two-Model Approach)
- **Türkçe Değişkenler:** `inaktif_gun_sayisi`, `kampanya_teklifi`, `uplift_skoru`, `ikna_edilebilir_musteri_mi`
- **Jupyter Notebook (`gun_014_inaktif_hat_kazanim.ipynb`):**
  1. A/B test verisinde kampanya alan ve almayan pasif kullanıcıların modellenmesi
  2. Two-Model ve X-Learner yaklaşımlarıyla salt kampanyanın yarattığı net etki (Uplift) skorlaması
  3. Sadece kampanya verildiğinde geri dönecek "İkna Edilebilir" (Persuadables) kitlenin seçilip bütçe israfının önlenmesi
- **Mülakat Sorusu:** Uplift modellemede "Uyuyan Köpekler" (Sleeping Dogs - kampanya verildiğinde rahatsız olup hattını tamamen kapatanlar) nasıl elenir?

### Gün 015: Dijital Kanallara Geçiş Eğilimi Modeli
- **İş Alanı:** Turkcell Dijital Operatör Dönüşümü & Maliyet Azaltma
- **Veri Kaynağı:** [Kaggle - Digital Channel Adoption](https://www.kaggle.com/datasets)
- **Model:** XGBoost Classifier + ROC-AUC
- **Türkçe Değişkenler:** `fiziksel_magaza_ziyaret_sayisi`, `web_giris_sikligi`, `dijital_uygulama_kullanim_puani`, `dijitale_gecis_skoru`
- **Jupyter Notebook (`gun_015_dijital_kanal_donusum.ipynb`):**
  1. Fiziksel mağazadan ve çağrı merkezinden işlem yapan abonelerin demografik ve işlem alışkanlıklarının incelenmesi
  2. Dijital kanallara (Turkcell Uygulaması) geçiş potansiyeli yüksek kitlelerin sınıflandırılması
  3. Uygulamayı ilk kez indirenlere özel hediye internet kampanyası yönlendirmesi
- **Mülakat Sorusu:** Dijital kanal benimseme modellerinde dijitalleşen müşterinin şirkete sağladığı işlem başı maliyet (Cost-per-Contact) tasarrufu nasıl hesaplanır?

### Gün 016: Baz İstasyonu İnternet Trafik Tahmini
- **İş Alanı:** Şebeke Planlama & Kapasite Yönetimi
- **Veri Kaynağı:** [Kaggle - Telecom Italia SMS/Call/Internet Grid Data](https://www.kaggle.com/datasets/marcodena/mobile-phone-activity)
- **Model:** LSTM / Facebook Prophet / Temporal Fusion Transformer
- **Türkçe Değişkenler:** `hucre_id`, `zaman_damgasi`, `saatlik_indirilen_veri_gb`, `yuklenen_veri_gb`, `tahmin_edilen_trafik_gb`
- **Jupyter Notebook (`gun_016_baz_istasyonu_trafik_tahmini.ipynb`):**
  1. Spatio-temporal ızgara telemetri verisini hücre bazında ayrıştırma
  2. Saatlik ve haftalık mevsimsellik (Seasonality) ve trend çıkarımı
  3. LSTM ile 24 saatlik ileri yönlü trafik tahmini ve aşırı yüklenme (Congestion) eşik kontrolleri
- **Mülakat Sorusu:** Telekom zaman serilerinde tatil, maç veya afet anlarındaki ani trafik sıçramalarını (Traffic Spikes) LSTM modellerine dışsal değişken (Exogenous Feature) olarak nasıl beslersiniz?

### Gün 017: Ağ İhlal ve Dağıtık Hizmet Engelleme (DDoS) Tespiti
- **İş Alanı:** Turkcell Siber Güvenlik Operasyon Merkezi (SOC)
- **Veri Kaynağı:** [Kaggle - CICIDS2017 / NSL-KDD](https://www.kaggle.com/datasets/cicdataset/cicids2017)
- **Model:** Random Forest & Autoencoder Anomali Tespiti
- **Türkçe Değişkenler:** `kaynak_ip`, `hedef_port`, `paket_uzunluk_ortalamasi`, `saniyedeki_istek_adedi`, `saldiri_etiketi`
- **Jupyter Notebook (`gun_017_ddos_saldiri_tespiti.ipynb`):**
  1. Ağ akış (Flow) verilerinden paket frekansı, bayt oranı ve TCP bayraklarının (SYN, ACK) çıkarılması
  2. Random Forest ve Autoencoder ile DDoS, Port Tarama ve Brute Force saldırılarının tespiti
  3. Siber güvenlik duvarında saldırgan IP'lerin saniyeler içinde otomatik karantinaya alınması
- **Mülakat Sorusu:** DDoS saldırılarında hacimsel (Volumetric - SYN Flood) ile uygulama katmanı (Layer 7 - HTTP Flood) saldırılarını ayırt etmede hangi akış öznitelikleri kritiktir?

### Gün 018: Şebeke Gecikme (Latency) Anomali Dedektörü
- **İş Alanı:** 4.5G/5G Hizmet Kalitesi (QoS) & Kesintisiz İletişim Masası
- **Veri Kaynağı:** [Kaggle - Numenta Anomaly Benchmark (NAB)](https://www.kaggle.com/datasets/boltzmannbrain/nab)
- **Model:** İzolasyon Ormanı (Isolation Forest) & DBSCAN
- **Türkçe Değişkenler:** `ping_gecikme_ms`, `jitter_sapmasi_ms`, `paket_kayip_orani`, `anomali_durumu`
- **Jupyter Notebook (`gun_018_gecikme_anomali_dedektoru.ipynb`):**
  1. Baz istasyonları ve omurga yönlendiricilerden toplanan milisaniyelik ping ve jitter telemetrilerinin işlenmesi
  2. İzolasyon Ormanı ile ağdaki ani gecikme ve paket kaybı anomalilerinin tespiti
  3. Şebeke operasyon merkezine (NOC) otomatik kök neden inceleme alarmı tetikleme
- **Mülakat Sorusu:** Gerçek zamanlı akış verisinde İzolasyon Ormanı ile Z-Score anomali tespitinin hesaplama karmaşıklığı ve çok değişkenli duyarlılık farkı nedir?

### Gün 019: Baz İstasyonu Enerji Tüketimi Optimizasyonu
- **İş Alanı:** Yeşil Şebeke & Enerji Yönetimi
- **Veri Kaynağı:** [Kaggle - Smart Grid Energy Consumption](https://www.kaggle.com/datasets)
- **Model:** Ridge Regression / LightGBM Regressor
- **Türkçe Değişkenler:** `baz_istasyonu_turu`, `sicaklik_derecesi`, `gece_trafik_yuku`, `harcanan_guc_kwh`
- **Jupyter Notebook (`gun_019_baz_istasyonu_enerji_optimizasyonu.ipynb`):**
  1. Baz istasyonunun soğutma, radyo frekans yükseltici ve dijital işlem ünitelerinin güç tüketimlerinin ayrıştırılması
  2. Dış hava sıcaklığı ve anlık veri trafiğine bağlı güç regresyon modelinin eğitilmesi
  3. Gece trafiği düşen saatlerde kapatılabilecek yedek taşıyıcıların enerji tasarruf potansiyeli simülasyonu
- **Mülakat Sorusu:** Baz istasyonu enerji optimizasyonunda şebeke kalitesinden (QoS / çağrı düşme oranı) ödün vermeden güç tasarrufu sağlamak için kısıtlı optimizasyon (Constrained Optimization) nasıl kurulur?

### Gün 020: 4G/5G Hücre Tıkanıklığı (Cell Congestion) Tahmini
- **İş Alanı:** Radyo Erişim Şebekesi (RAN) & Trafik Yönetimi
- **Veri Kaynağı:** [Kaggle - Cellular Network QoS Data](https://www.kaggle.com/datasets)
- **Model:** CatBoost Multi-Classification
- **Türkçe Değişkenler:** `rrc_baglanti_sayisi`, `prb_kullanim_orani`, `aktif_kullanici_adedi`, `tikaniklik_seviyesi`
- **Jupyter Notebook (`gun_020_hucre_tikaniklik_tahmini.ipynb`):**
  1. Fiziksel Kaynak Blokları (PRB - Physical Resource Block) doluluk oranları ve RRC bağlantı telemetrilerinin işlenmesi
  2. CatBoost ile hücrenin 1 saat sonra "Normal", "Yoğun", "Kritik Tıkalı" durumuna geçme olasılığının tahmini
  3. Tıkanıklık öncesinde komşu hücrelere otomatik yük aktarma (Load Balancing) kararı
- **Mülakat Sorusu:** 4G/5G şebekelerinde PRB (Physical Resource Block) kullanım oranı ile kullanıcı indirme hızı (Throughput) arasındaki ters orantılı ilişki modelde nasıl kullanılır?


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

## 💬 Modül 03: Doğal Dil İşleme (NLP), Müşteri Hizmetleri & Semantik Arama (Gün 031 – 045)

### Gün 031: Telekom Müşteri Şikayetleri Duygu Analizi
- **İş Alanı:** Müşteri Deneyimi & Sosyal Medya Dinleme Masası
- **Veri Kaynağı:** [Kaggle - Turkish Sentiment Analysis / Şikayetvar Dataset](https://www.kaggle.com/datasets)
- **Model:** BERTurk (`dbmdz/bert-base-turkish-cased`) / RoBERTa + Focal Loss
- **Türkçe Değişkenler:** `sikayet_metni`, `duygu_sinifi_pozitif_notr_negatif`, `guven_skoru`, `tahmin_edilen_etiket`
- **Jupyter Notebook (`gun_031_sikayet_duygu_analizi.ipynb`):**
  1. Türkçe metin ön işleme (Noktalama, stop-words temizleme, küçük harfe çevirme)
  2. HuggingFace Transformers ile BERTurk modelinin 3 sınıflı duygu sınıflandırması için ince ayarı (Fine-Tuning)
  3. Dengesiz veri dağılımında Confusion Matrix, PR-AUC ve F1-Score değerlendirmesi
- **Mülakat Sorusu:** Türkçe gibi morfolojik olarak zengin ve sondan eklemeli dillerde WordPiece/BPE tokenizasyonu kök-ek ayrımını nasıl ele alır?

### Gün 032: Müşteri Talebi Intent (Niyet) Sınıflandırma
- **İş Alanı:** Turkcell Dijital Asistan & Çağrı Botu (NLU Masası)
- **Veri Kaynağı:** [HuggingFace - Banking77 / Turkish Intent](https://huggingface.co/datasets/banking77)
- **Model:** SetFit (Few-Shot Sentence-Transformers) / DistilBERTurk
- **Türkçe Değişkenler:** `kullanici_cumlesi`, `tespit_edilen_niyet`, `niyet_olasiligi`, `yonlendirilen_islem_kodu`
- **Jupyter Notebook (`gun_032_chatbot_niyet_siniflandirma.ipynb`):**
  1. Kullanıcının yazdığı serbest metin mesajlarının ("faturamı ödemek istiyorum", "kalan internetim kaç GB") tokenizasyonu
  2. SetFit mimarisi ile az sayıda örnekle (Few-Shot) 77 farklı niyet sınıfının eğitilmesi
  3. Chatbot akışında eşik değerin (%85) üzerindeki niyetlerin ilgili servis API'sine yönlendirilmesi
- **Mülakat Sorusu:** Chatbot niyet sınıflandırmasında "Out-of-Scope / Fallback" (kapsam dışı veya anlaşılamayan niyet) tespiti için Softmax olasılığı dışında hangi güvenilirlik metrikleri kullanılır?

### Gün 033: Twitter Destek Taleplerini Otomatik Departmana Yönlendirme
- **İş Alanı:** @TurkcellHizmet Sosyal Medya Masası & Akıllı Bilet Yönlendirme (Ticket Routing)
- **Veri Kaynağı:** [Kaggle - Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **Model:** TF-IDF (N-Gram) + LinearSVC / FastText Çok Sınıflı Yönlendirici
- **Türkçe Değişkenler:** `tweet_icerigi`, `ilgili_departman_fatura_sebeke_cihaz_paket`, `atanan_oncelik_seviyesi`, `tahmini_cozum_ekibi`
- **Jupyter Notebook (`gun_033_sosyal_medya_yonlendirme.ipynb`):**
  1. @TurkcellHizmet etiketli tweetlerin temizlenmesi, kullanıcı adlarının ve emojilerin işlenmesi
  2. TF-IDF unigram/bigram matrisi ile LinearSVC ve FastText modellerinin eğitilmesi
  3. Gelen tweetin saniyeler içinde doğru operasyon kuyruğuna (Fatura, Şebeke, Pasaj vb.) yönlendirilmesi
- **Mülakat Sorusu:** Yüksek hacimli çağrı/destek metinlerini departmanlara yönlendirirken LinearSVC/FastText gibi hafif modellerin derin transformatörlere göre gecikme (Latency) ve CPU maliyeti avantajları nelerdir?


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

### Gün 035: Şirket İçi Dokümanlar için Semantik Vektör Arama Motoru
- **İş Alanı:** Turkcell Akademi & Şirket İçi Bilgi Yönetimi
- **Veri Kaynağı:** [HuggingFace - BilgiQA / Turkish Telecom FAQs](https://huggingface.co/datasets)
- **Model:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` + ChromaDB / FAISS Vektör İndeksi + Cosine Similarity
- **Türkçe Değişkenler:** `kullanici_sorusu`, `getirilen_dokuman_parcalari`, `vektor_benzerlik_skoru`, `en_uygun_pasaj`
- **Jupyter Notebook (`gun_035_dokuman_semantik_arama.ipynb`):**
  1. PDF/Markdown telekom politika dokümanlarının recursive character splitter ile parçalanması (Chunking)
  2. ChromaDB vektör veritabanında embedding indeksleme
  3. Kullanıcı sorusuna en uygun ilk 3 teknik kılavuz pasajının milisaniyeler içinde Cosine Similarity ile sıralanıp getirilmesi
- **Mülakat Sorusu:** Vektör aramasında Dense Retrieval ile Sparse Retrieval (TF-IDF/BM25) arasındaki fark nedir ve hangi senaryolarda hangisi üstündür?

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

### Gün 044: Müşteri Temsilcisi Yanıt Kalitesi ve Nezaket Skorlama Modeli
- **İş Alanı:** Kalite Güvence (QA) & Müşteri Deneyimi Denetimi (Global Bilgi Masası)
- **Veri Kaynağı:** [HuggingFace - Turkish Customer Service Multi-Turn Conversations](https://huggingface.co/datasets)
- **Model:** BERTurk (`dbmdz/bert-base-turkish-cased`) + Nezaket/Empati Kural Sözlüğü (Politeness Lexicon) + Cosine Semantik Relevans
- **Türkçe Değişkenler:** `temsilci_cevabi`, `nezaket_sozluk_skoru`, `cozum_anlamsal_uyum_skoru`, `nihai_kalite_puani_1_100`
- **Jupyter Notebook (`gun_044_temsilci_kalite_skorlama.ipynb`):**
  1. Çağrı transkriptlerinden müşteri sorusu ile temsilci cevabının ayrıştırılması
  2. Kural ve sözlük tabanlı nezaket/profesyonellik belirteçlerinin taranması
  3. BERTurk ile temsilci yanıtının standart kurumsal çözüm rehberiyle semantik benzerliğinin hesaplanıp 1-100 arası puanlanması
- **Mülakat Sorusu:** Doğal Dil İşlemede kural tabanlı sözlük (Lexicon-based) yaklaşımları ile derin öğrenme embedding skorlarını birleştirmenin açıklanabilirlik ve hesaplama hızı avantajı nedir?

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

## 🤖 Modül 12: İleri Seviye NLP, Semantik Arama, Metin Madenciliği & Müşteri Deneyimi (Gün 146 – 160)

### Gün 146: Müşteri Temsilcisi Yanıt Kalitesi ve Çözüm Uyumu Denetleyicisi
- **İş Alanı:** Turkcell Global Bilgi Kalite Güvence & Otomatik Değerlendirme
- **Veri Kaynağı:** [HuggingFace - Customer Support Conversation Quality Dataset](https://huggingface.co/datasets)
- **Model:** BERTurk Cross-Encoder + Doğal Dil Çıkarımı (NLI - Entailment/Contradiction Sınıflandırma)
- **Türkçe Değişkenler:** `musteri_sorusu`, `temsilci_cevabi`, `mantiksal_uyum_entailment_skoru`, `nezaket_puani`, `kalite_notu_1_100`
- **Jupyter Notebook (`gun_146_temsilci_kalite_denetimi_nli.ipynb`):**
  1. Çağrı transkriptlerinden soru ve yanıt çiftlerinin ayrıştırılması
  2. NLI modeli ile cevabın soruyu mantıksal olarak karşılayıp karşılamadığının (Doğruluk/Çelişki) tespiti
  3. LLM gerektirmeyen deterministik ve milisaniyelik kalite denetim raporlaması
- **Mülakat Sorusu:** Metin uyumunu değerlendirmede Bi-Encoder (Cosine Similarity) yerine Cross-Encoder kullanmanın dikkat mekanizması (Cross-Attention) açısından sağladığı doğruluk artışı nedir?

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

### Gün 148: Fatura PDF'lerinden Yapılandırılmış JSON Veri Çıkarıcı
- **İş Alanı:** Kurumsal Müşteri Masası & Otomatik Fatura İçe Aktarma
- **Veri Kaynağı:** [Kaggle - Telecom Invoice PDF/Text Dataset](https://www.kaggle.com/datasets)
- **Model:** PDFPlumber + Regex Kural Motoru + SpaCy / BERTurk NER (Slot Filling)
- **Türkçe Değişkenler:** `fatura_metni`, `abone_no`, `fatura_kesim_tarihi`, `odenecek_tutar_tl`, `kdv_oiv_vergileri_json`
- **Jupyter Notebook (`gun_148_fatura_pdf_json_ayristirici.ipynb`):**
  1. Fatura PDF'lerinden metin blokları ve tabloların koordinat bazlı ayrıştırılması
  2. Regex ve Varlık İsmi Tanıma (NER) ile Fatura No, Vergi Kalemleri ve Tutar alanlarının çıkarılması
  3. Deterministik ve sıfır hata payıyla doğrulanmış JSON çıktısı üretimi
- **Mülakat Sorusu:** Doküman işlemede (Information Extraction) kural tabanlı Regex/NER yaklaşımlarının üretim ortamında LLM'lere göre hız ve güvenilirlik avantajları nelerdir?

### Gün 149: Canlı Sohbet Müşteri Sinir Seviyesi (Frustration) İzleyici
- **İş Alanı:** BiP & Turkcell Web Canlı Destek Masası
- **Veri Kaynağı:** [HuggingFace - Customer Support Live Chat Frustration Traces](https://huggingface.co/datasets)
- **Model:** BERTurk + Temporal Attention (Mesaj Sırası Ağırlıklandırma)
- **Türkçe Değişkenler:** `oturum_mesaj_listesi`, `buyuk_harf_unlem_orani`, `cevap_bekleme_suresi_sn`, `sinirlilik_skoru_0_100`
- **Jupyter Notebook (`gun_149_canli_sohbet_sinir_seviyesi.ipynb`):**
  1. Sohbet esnasında müşterinin yazdığı ardışık mesajlardaki öfke artışının takibi
  2. BÜYÜK HARF kullanımı, tekrarlayan soru sorma ve botun anlayamaması kaynaklı sinir seviyesi tahmini
  3. Sinir skoru %75'i geçtiğinde botun devreden çıkıp sohbeti anında kıdemli müşteri temsilcisine aktarması (Human-in-the-Loop)
- **Mülakat Sorusu:** Canlı sohbette tek bir cümlenin bağımsız duygu analizi ile tüm konuşma akışının kümülatif sinir seviyesi (Contextual Frustration) arasındaki fark nasıl modellenir?

### Gün 150: Şikayet Metinlerinden Kök Neden Hiyerarşisi Çıkarma
- **İş Alanı:** Şikayet Yönetimi & Operasyonel Hata Analizi (Şikayetvar / 532)
- **Veri Kaynağı:** [Kaggle - Turkish Telecom Complaints Dataset](https://www.kaggle.com/datasets)
- **Model:** Hiyerarşik Metin Sınıflandırma (Hierarchical BERTurk / SetFit)
- **Türkçe Değişkenler:** `sikayet_metni`, `ana_kategori_sebeke_fatura_kampanya`, `alt_kategori_cekmiyor_fatura_asimi`, `kok_neden_kodu`
- **Jupyter Notebook (`gun_150_sikayet_kok_neden_hiyerarsisi.ipynb`):**
  1. Şikayet metinlerinin 3 seviyeli hiyerarşik taksonomiye (Ana Alan -> Alt Problem -> Kök Neden) göre etiketlenmesi
  2. Seviyeli sınıflandırıcı (Local Classifier per Parent Node) mimarisi ile uçtan uca tahmin
  3. Şebeke arızası kaynaklı şikayetlerin doğrudan ilgili bölge saha operasyon birimine yönlendirilmesi
- **Mülakat Sorusu:** Düz çok sınıflı (Flat Multi-Class) model yerine Hiyerarşik Sınıflandırma kullanmanın sınıflar arası mantıksal tutarlılığa katkısı nedir?

### Gün 151: RAG için Hibrit Vektör + BM25 Telekom Arama Motoru
- **İş Alanı:** Turkcell İntranet Bilgi Bankası & Teknik Doküman Arama
- **Veri Kaynağı:** [HuggingFace - Turkish Tech Documentation & FAQ](https://huggingface.co/datasets)
- **Model:** BM25 (Sparse) + BGE-M3 / Sentence-Transformers (Dense) + Reciprocal Rank Fusion (RRF) + Cross-Encoder Reranker
- **Türkçe Değişkenler:** `kullanici_sorusu`, `bm25_anahtar_kelime_skorlari`, `vektor_anlamsal_benzerlik_skorlari`, `rrf_birlestirilmis_siralama`, `en_uygun_pasajlar`
- **Jupyter Notebook (`gun_151_hibrit_arama_bm25_vektor.ipynb`):**
  1. Teknik dökümanlarda hem birebir anahtar kelime (BM25) hem de anlamsal kavram (Dense Embeddings) araması çalıştırma
  2. Reciprocal Rank Fusion (RRF) formülü ile sparse ve dense arama skorlarının ağırlıklı birleştirilmesi
  3. Cross-Encoder Reranker ile en alakalı ilk 3 teknik dokümanın milisaniyeler içinde listelenmesi
- **Mülakat Sorusu:** Teknik telekom kodları ve hata terimleri aramasında neden tek başına vektör araması yetersiz kalır ve BM25 ile hibritleme şarttır?

### Gün 152: Çağrı Metninden Kampanya Kabul İhtimali Puanlama
- **İş Alanı:** Dış Arama (Outbound Telemarketing) & Paket Satış Masası
- **Veri Kaynağı:** [Kaggle - Telemarketing Call Transcripts & Campaign Success](https://www.kaggle.com/datasets)
- **Model:** TabNet / CatBoost (Metin TF-IDF/Embedding + Müşteri CRM Verisi Füzyonu)
- **Türkçe Değişkenler:** `gorusme_transkripti_embedding`, `musteri_mevcut_tarife_tutari`, `onerilen_kampanya_fiyati`, `teklifi_kabul_etme_olasiligi`
- **Jupyter Notebook (`gun_152_kampanya_kabul_ihtimali_nlp.ipynb`):**
  1. Görüşmenin ilk 30 saniyesinde müşterinin konuşma tarzı ve itiraz kelimelerinin analizi
  2. Müşteri demografisi ile konuşma metni gömmelerinin birleştirilerek teklif başarı ihtimali tahmini
  3. Kabul ihtimali düşük olduğunda temsilcinin ekranına alternatif indirimli paket önerisi düşürülmesi
- **Mülakat Sorusu:** Tabular veriler ile serbest metin verilerini (Multimodal Tabular-NLP) aynı modelde birleştirmede Late Fusion ve Early Fusion mimarileri nasıl tasarlanır?

### Gün 153: Abonelik Taahhütnameleri için Ekstraktif (Çıkarıcı) Metin Özetleme
- **İş Alanı:** Dijital Kanallar & Web/Mobil Tarife Karşılaştırma Sayfası
- **Veri Kaynağı:** [Kaggle - Telecom Tariff Terms & Conditions Contracts](https://www.kaggle.com/datasets)
- **Model:** Extractive TextRank / LexRank + BERTurk Cümle Gömmeleri (Halüsinasyonsuz Özetleyici)
- **Türkçe Değişkenler:** `uzun_tarife_yasal_metni`, `cumle_onem_skorlari`, `cikarilan_3_maddelik_ozet`, `rouge_1_rouge_2_rouge_l_skorlari`
- **Jupyter Notebook (`gun_153_tarife_ekstraktif_ozetleme.ipynb`):**
  1. 10 sayfalık hukuki ve teknik tarife taahhütnamelerinin cümlelere ayrıştırılması
  2. TextRank graf algoritması ile metnin en bilgilendirici ve kritik 3 cümlesinin doğrudan metinden çekilmesi
  3. Üretken model kullanmadan sıfır halüsinasyon riskiyle anında özet çıkarma
- **Mülakat Sorusu:** Hukuki ve finansal sözleşmelerde Abstractive (Üretici) özetleme yerine Extractive (Çıkarıcı) özetleme kullanmanın yasal uyumluluk avantajı nedir?

### Gün 154: Sosyal Medya Rakip Operatör Kampanya Karşılaştırma Analizörü
- **İş Alanı:** Pazarlama Stratejisi & Rekabet İstihbaratı Masası
- **Veri Kaynağı:** [Kaggle - Twitter / X Telecom Mentions & Campaign Feedback](https://www.kaggle.com/datasets)
- **Model:** RoBERTa-Turkish Aspect-Based Sentiment + Named Entity Recognition (NER)
- **Türkçe Değişkenler:** `tweet_metni`, `bahsedilen_operator_turkcell_vodafone_telekom`, `karsilastirma_kriteri_fiyat_kapsama_hiz`, `duygu_kutbu`
- **Jupyter Notebook (`gun_154_rakip_kampanya_analizoru.ipynb`):**
  1. Sosyal medyada rakip operatörler hakkında atılan tweetlerin gerçek zamanlı taranması
  2. Fiyat, internet hızı ve müşteri hizmetleri başlıklarında rakip memnuniyet/şikayet oranlarının kıyaslanması
  3. Rakibin zayıf kaldığı bölgelere özel karşı Turkcell kampanyası önerme motoru
- **Mülakat Sorusu:** Aspect-Based Sentiment Analysis (ABSA) ile genel cümle düzeyinde sentiment analizi arasındaki fark nedir?

### Gün 155: Sesli Yanıt (IVR) Fonetik Benzerlik Eşleştirici
- **İş Alanı:** 532 Sesli Yanıt Menüsü & Şive/Aksan/Yazım Hatası Toleransı
- **Veri Kaynağı:** [HuggingFace - Turkish ASR Transcriptions](https://huggingface.co/datasets)
- **Model:** Double Metaphone / Soundex Turkish Adaptation + Levenshtein / Jaro-Winkler Distance
- **Türkçe Değişkenler:** `kullanici_sesli_ifadesi_metni`, `fonetik_kod_uretilen`, `eslesen_menu_komutu`, `fonetik_benzerlik_orani`
- **Jupyter Notebook (`gun_155_ivr_fonetik_eslestirici.ipynb`):**
  1. Kullanıcının şiveli veya yanlış telaffuz ettiği komutların (Örn: "patura", "fatıra", "kontur") fonetik kodlarının çıkarılması
  2. Doğru IVR menü komutlarıyla ("Fatura Ödeme", "TL Yükleme") fonetik kod benzerliği eşleştirmesi
  3. Yanlış anlama oranını %40 azaltarak müşteriyi doğru menüye aktarma
- **Mülakat Sorusu:** Fonetik algoritmaların (Double Metaphone) ASR sonrası niyet eşleştirmede saf metinsel Levenshtein mesafesine göre avantajı nedir?

### Gün 156: Abonelik Sözleşmesi Cayma Bedeli ve Taahhüt Maddesi Bulucu
- **İş Alanı:** Hukuk Masası & Dijital Sözleşme Analizi
- **Veri Kaynağı:** [Kaggle - Telecom Legal Contracts & Addendums](https://www.kaggle.com/datasets)
- **Model:** LayoutLMv3 / DeBERTa-v3 Question Answering (Extractive QA - Metin İçi Konum Bulucu)
- **Türkçe Değişkenler:** `sozlesme_pdf_goruntusu`, `soru_cayma_bedeli_nasil_hesaplanir`, `cevap_metin_kesiti`, `guven_skoru`
- **Jupyter Notebook (`gun_156_sozlesme_cayma_bedeli_qa.ipynb`):**
  1. Taranmış sözleşme sayfalarındaki görsel yerleşim ve metinlerin LayoutLMv3 ile işlenmesi
  2. "Taahhüt süresi ne kadar?", "Erken iptal halinde hangi indirimler geri alınır?" sorularına sözleşmeden doğrudan cevap çıkarma
  3. Müşteriye ve temsilciye kanuni hakları anında gösteren sözleşme yardımcısı
- **Mülakat Sorusu:** Extractive QA ile Generative QA arasındaki fark nedir ve yasal sözleşmelerde neden Extractive QA tercih edilir?

### Gün 157: Chatbot için Few-Shot Niyet Genişletici Sentetik Veri Pipeline'ı
- **İş Alanı:** BiP Dijital Asistan Eğitimi & Veri Artırma (Data Augmentation)
- **Veri Kaynağı:** [HuggingFace - Turkish Intent Expansion](https://huggingface.co/datasets)
- **Model:** EDA (Easy Data Augmentation - Synonym Replacement via WordNet/FastText) + Back-Translation (Helsinki-NLP Opus-MT tr-en-tr)
- **Türkçe Değişkenler:** `ornek_3_niyet_cumlesi`, `uretilen_50_varyasyon`, `anlamsal_benzerlik_esigi`, `zenginlestirilmis_egitim_seti`
- **Jupyter Notebook (`gun_157_few_shot_niyet_genisletici_eda.ipynb`):**
  1. 3 adet örnek kullanıcı cümlesinden yola çıkarak Eş Anlamlı Değiştirme (Synonym Replacement) ve Rastgele Ekleme/Silme uygulama
  2. Geriye Çeviri (Back-Translation: Türkçe -> İngilizce -> Türkçe) ile doğal ve çeşitli alternatif cümleler üretme
  3. Üretilen 50 cümlenin Sentence-Transformers ile semantik kayma filtresinden geçirilip eğitim setine eklenmesi
- **Mülakat Sorusu:** NLP'de Kural Tabanlı Veri Artırma (EDA) ve Back-Translation tekniklerinin model genelleme kabiliyetine (Generalization) etkisi nedir?

### Gün 158: Boyut Tabanlı Müşteri Memnuniyetsizliği (Aspect-Based Sentiment)
- **İş Alanı:** Müşteri Deneyimi Ölçümleme (NPS / CSAT Masası)
- **Veri Kaynağı:** [Kaggle - Aspect Based Sentiment Telecom](https://www.kaggle.com/datasets)
- **Model:** DeBERTa-v3 / BERTurk Multi-Task Learning (Aspect Extraction + Sentiment Polarity)
- **Türkçe Değişkenler:** `anket_yorumu`, `tespit_edilen_boyutlar_fiyat_hiz_musteri_hizmetleri`, `boyut_duygu_skorlari`
- **Jupyter Notebook (`gun_158_aspect_based_sentiment_deberta.ipynb`):**
  1. "İnternetiniz çok hızlı ama faturalar çok pahalı" gibi çoklu duygu içeren cümlelerin ayrıştırılması
  2. Hız boyutuna Pozitif, Fiyat boyutuna Negatif etiket atanması
  3. Ürün ekiplerine departman bazlı net memnuniyet puanı (Aspect NPS) raporlanması
- **Mülakat Sorusu:** Multi-Task Learning mimarilerinde Aspect Extraction ve Sentiment Classification görevlerinin ortak temsil katmanından öğrenilmesinin yararı nedir?

### Gün 159: E-posta Destek Talebi Otomatik Cevap Taslağı Üretici
- **İş Alanı:** Turkcell Global Bilgi E-posta Destek Masası
- **Veri Kaynağı:** [HuggingFace - Customer Email Response Generation](https://huggingface.co/datasets)
- **Model:** BERTurk Niyet & Varlık Çıkarıcı (Slot Filling) + Jinja2 Kurumsal Şablon Motoru
- **Türkçe Değişkenler:** `gelen_musteri_epostasi`, `tespit_edilen_talep_turu`, `cikarilan_parametreler_fatura_tarih_tutar`, `uretilen_cevap_taslagi`
- **Jupyter Notebook (`gun_159_eposta_sablon_cevap_motoru.ipynb`):**
  1. Gelen e-postanın konusunun, abone bilgilerinin ve talebinin BERTurk ile sınıflandırılması
  2. Müşteri parametrelerinin (İsim, Fatura Tutarı, Paket Adı) şablondaki yerlerine (Slots) otomatik yerleştirilmesi
  3. Temsilcinin tek tıkla inceleyip onaylayabileceği kurumsal dilde hatasız e-posta taslağı üretimi
- **Mülakat Sorusu:** Müşteri destek sistemlerinde serbest metin üreten kontrolsüz modeller yerine Slot-Filling ve Şablon Motoru (Template Engine) kullanmanın marka güvenliği avantajı nedir?

### Gün 160: Müşteri İletişim Dili (Resmi vs Samimi) Belirleme ve Ton Eşleme
- **İş Alanı:** Dijital İletişim & Kişiselleştirilmiş İletişim Tonu (Tone of Voice)
- **Veri Kaynağı:** [HuggingFace - Formality Classification](https://huggingface.co/datasets)
- **Model:** BERTurk Formality Classifier + Kural Tabanlı Şablon Eşleme (Formal / Informal Response Selector)
- **Türkçe Değişkenler:** `musteri_mesaji`, `resmiyet_puani_0_100`, `uygun_cevap_stili_resmi_kurumsal_genc_samimi`, `secilen_cevap_sablonu`
- **Jupyter Notebook (`gun_160_iletisim_dili_ton_esleme.ipynb`):**
  1. Kullanıcının hitap şeklinden ("Merhabalar efendim" vs "selam naber") resmiyet derecesinin tespiti
  2. Genç kullanıcıya enerjik/samimi, kurumsal kullanıcıya resmi/saygılı dilde yanıt şablonu seçimi
  3. Müşteri memnuniyetini ve iletişim bağını güçlendiren dinamik ton uyarlaması
- **Mülakat Sorusu:** NLP'de Formality Classification için kullanılan öznitelikler (kibar ekler, emoji kullanımı, argo/jargon sıklığı) modelde nasıl ağırlıklandırılır?

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


---

---
# BÖLÜM 3: GÜN 201 - 300 (İLERİ SEVİYE UZMANLIK, EDGE AI, 5G & O-RAN, MULTIMODAL VE GNN)
> **💻 %100 YEREL (LOCAL), ÜCRETSİZ VE AÇIK KAYNAK GÜVENCESİ:**
> Bu bölümdeki projelerin tamamı, öğrencinin/mühendisin kendi yerel bilgisayarında (CPU veya standart GPU) ve ücretsiz Google Colab / Kaggle ortamlarında **sıfır maliyetle ($0)** çalışacak şekilde tasarlanmıştır. Hiçbir projede ücretli API anahtarı (OpenAI, Anthropic, Google Cloud Paid API vb.) veya ağır üretken LLM donanım/çalışma gereksinimi yoktur. Tüm NLP, ses, görüntü, grafik ve telekom optimizasyon görevleri açık kaynak, deterministik, hafif kütüphaneler (Scikit-learn, LightGBM, XGBoost, NetworkX, OpenCV, MobileNetV3, Librosa, SimPy, SciPy, River, FastText, m2cgen) ile doğrudan CPU/GPU üzerinde milisaniyeler içinde yürütülür.


## Modül 17: Deterministik & Yerel Akıllı Ajanlar (Local Agentic AI)

---

### Gün 201: NOC Otonom Alarm Triyaj ve Kök Neden Analiz Ajanı
- **İş Alanı:** Şebeke Operasyon Merkezi (NOC) & Otonom Arıza Yönetimi
- **Veri Kaynağı:** Turkcell NOC Alarm Günlükleri & Netcool Sentetik Olay Kayıtları
- **Model:** Sonlu Durum Makinesi (FSM) + Regex Log Ayrıştırıcı + Kural Tabanlı Triyaj Motoru
- **Türkçe Değişkenler:** `alarm_id, istasyon_kodu, alarm_seviyesi, kok_neden_kategorisi, otomatik_bilet_acilsin_mi, triyaj_aksiyonu`
- **Kapsam:** Baz istasyonlarından gelen binlerce eş zamanlı alarmı korelasyon kurallarıyla analiz eder; elektrik kesintisi kaynaklı ikincil transmisyon alarmlarını eleyerek tek bir kök neden arıza kaydı üretir.
- **Jupyter Notebook (`gun_201_noc_otonom_alarm_triyaj_ajani.ipynb`):**
  1. Ham SNMP/Syslog alarm akışının ayrıştırılması ve zaman penceresi (time-window) kümelemesi
  2. Güç kesintisi ile transmisyon kopması arasındaki nedensellik ağacının kurulması
  3. Sahaya çıkacak nöbetçi ekibe otomatik yönlendirilen arıza bileti ve öncelik skorunun üretilmesi
- **Mülakat Sorusu:** *"Birbirini tetikleyen alarm fırtınalarında (Alarm Flapping) sistemin kaynak tüketimini önlemek için yerel durum makinesinde debouncing ve sliding-window nasıl kurgulanır?"*

### Gün 202: Paycell Şüpheli Transfer ve Dolandırıcılık Araştırma Ajanı
- **İş Alanı:** Fintek / Paycell Güvenliği & Dolandırıcılık İnceleme Masası
- **Veri Kaynağı:** Paycell Sentetik İşlem Kayıtları & Kullanıcı Cüzdan Hareketleri
- **Model:** NetworkX Graf Dolaşımı + Kural Tabanlı Şüphe Skoru Ajanı
- **Türkçe Değişkenler:** `gonderici_id, alici_id, transfer_tutari_tl, islem_sikligi_dakika, supheli_halka_puani, bloke_edilsin_mi`
- **Kapsam:** Yeni açılan cüzdanlar üzerinden kısa sürede peş peşe yapılan parçalı para transferlerini ve aracı hesapları tespit ederek otonom bloke ve inceleme kararı alır.
- **Jupyter Notebook (`gun_202_paycell_supheli_transfer_ajan.ipynb`):**
  1. Son 15 dakikalık para transferi işlemlerinden yönlü işlem grafının (Directed Graph) çıkarılması
  2. Giden ve gelen para arasındaki hız (velocity) ve işlem bölünme (smurfing) kurallarının denetimi
  3. Kara liste eşleşmesi durumunda otomatik cüzdan dondurma ve şüpheli işlem bildirim raporu
- **Mülakat Sorusu:** *"Finansal dolandırıcılıkta kural tabanlı ajanlar ile makine öğrenimi modellerinin bir arada çalıştığı hibrit karar mimarisi nasıl kurgulanır?"*

### Gün 203: Otomatik Müşteri SLA Kesinti İtiraz ve İade Hesaplama Ajanı
- **İş Alanı:** Kurumsal Müşteri Hizmetleri & Sözleşme SLA Yönetimi
- **Veri Kaynağı:** Kurumsal Metro-Ethernet Kesinti Kayıtları & Abone SLA Taahhütleri
- **Model:** Zaman Serisi Aralık Kesişim Motoru (Interval Tree) + Mantıksal Karar Ajanı
- **Türkçe Değişkenler:** `kurumsal_abone_id, taahhut_edilen_uptime, gerceklesen_kesinti_dakika, ceza_katsayisi, iade_tutari_tl`
- **Kapsam:** Kurumsal fiber müşterilerinin internet kesintisi şikayetlerinde şebeke telemetrisini otomatik sorgular; sözleşmedeki %99.9 erişilebilirlik taahhüdünün aşılıp aşılmadığını hesaplayarak faturaya otomatik iade yansıtır.
- **Jupyter Notebook (`gun_203_kurumsal_sla_kesinti_iade_ajani.ipynb`):**
  1. Abone bazlı SLA taahhüt metrikleri ve geçmiş kesinti loglarının zaman kesişimlerinin hesaplanması
  2. Mücbir sebep (planlı bakım, fiber kablo kopması) ayrıştırması ile net kesinti süresinin bulunması
  3. Fatura dönemi için cezai şart iade matrisinin hesaplanması ve onay dekontu çıktısı
- **Mülakat Sorusu:** *"SLA hesaplamasında planlı bakım pencereleri ile beklenmedik kesintileri log zaman damgalarından ayrıştırmanın deterministik yöntemi nedir?"*

### Gün 204: Çağrı Merkezi Müşteri Temsilcisi Akıllı Copilot Ajanı
- **İş Alanı:** 532 Müşteri Hizmetleri & Temsilci Destek Sistemleri
- **Veri Kaynağı:** Turkcell Müşteri Çağrı Transkriptleri & Çözüm Kataloğu
- **Model:** TF-IDF + Cosine Benzerliği + Kuralcı Slot-Filling (Yerel Deterministik)
- **Türkçe Değişkenler:** `musteri_ifadesi, tespit_edilen_niyet, onerilen_cozum_adimi, temsilci_onay_butonu, cagri_suresi_tasarrufu_sn`
- **Kapsam:** Müşteri konuşurken metne dönüşen diyalogdan abonenin ana problemini anlık çıkarır ve temsilcinin ekranına en uygun 3 çözüm butonunu milisaniyeler içinde getirir.
- **Jupyter Notebook (`gun_204_cagri_merkezi_copilot_ajani.ipynb`):**
  1. Müşteri ifadesindeki anahtar telekom terimlerinin (PUK kodu, fatura aşımı, eSIM) yerel niyetle eşlenmesi
  2. Müşterinin tarife ve fatura durumuyla eşleşen dinamik çözüm kartının ekrana getirilmesi
  3. Temsilcinin tek tıkla aksiyon almasını sağlayan simüle API çağrısı ve çağrı süresi optimizasyonu
- **Mülakat Sorusu:** *"Çağrı merkezinde gerçek zamanlı copilot çalışırken gecikmenin (latency) < 100ms tutulması için TF-IDF matris önbelleklemesi (caching) nasıl yapılır?"*

### Gün 205: Self-Healing RAN: Otomatik Baz İstasyonu İyileştirme Ajanı
- **İş Alanı:** Radyo Erişim Şebekesi (RAN) & Otonom Hücre İyileştirme
- **Veri Kaynağı:** Baz İstasyonu KPI Telemetrisi (E-RAB Drop Rate, Sinyal Gücü, CPU)
- **Model:** Karar Ağacı Mantığı + Simüle Netconf/YANG Ajan Döngüsü
- **Türkçe Değişkenler:** `hucre_id, dusme_orani_yuzde, kilitlenme_durumu, onerilen_aksiyon_soft_reset, basari_durumu`
- **Kapsam:** Hücre kilitlenmesi veya aşırı çağrı düşme oranı yaşayan baz istasyonlarında insan müdahalesine gerek kalmadan adım adım Soft Reset, Anten Tilt Düzeltmesi ve Güç Dengeleme aksiyonlarını uygular.
- **Jupyter Notebook (`gun_205_self_healing_ran_otonom_ajan.ipynb`):**
  1. Hücresel KPI'ların eşik değer aşım (Threshold Cross Alert) kontrolü
  2. Sırasıyla en az kesintili otonom aksiyon planının (Kanal reset -> Güç artırma -> Sektör yönlendirme) seçimi
  3. Müdahale sonrası 5 dakikalık test çağrısı simülasyonu ile hücrenin düzeldiğinin teyit edilmesi
- **Mülakat Sorusu:** *"Telekom şebekelerinde 'Self-Healing' mekanizması kurgulanırken sonsuz döngüye (reboot loop) girmeyi engelleyen devre kesici (Circuit Breaker) paterni nasıl kodlanır?"*

### Gün 206: Kurumsal Müşteri Şartname ve İhale Kural Denetim Ajanı
- **İş Alanı:** Turkcell Kurumsal Satış & İhale Operasyonları Masası
- **Veri Kaynağı:** Kamu ve Özel Sektör İhale Şartname Metinleri (Sentetik PDF/Metin)
- **Model:** Düzenli İfadeler (Regex) Şablon Motoru + JSON Schema Validator
- **Türkçe Değişkenler:** `sartname_metni, aranan_kriter_hiz_mbps, istenen_ip_blok_sayisi, turkcell_karsilama_orani, uygunluk_raporu`
- **Kapsam:** Yüzlerce sayfalık kurumsal ihale şartnamelerini saniyeler içinde tarayarak hız, IP blok adedi, gecikme ve taahhüt şartlarının Turkcell altyapısı ile karşılanabilirliğini denetler.
- **Jupyter Notebook (`gun_206_ihale_sartname_kural_denetim_ajani.ipynb`):**
  1. Şartname metnindeki sayısal taahhütlerin ve cezai şart maddelerinin şablonlarla taranması
  2. Turkcell kurumsal ürün kataloğu sınırları (Maksimum hız, Metro-Ethernet kapsama) ile kuralcı eşleşme
  3. Teklif mektubu için yeşil/kırmızı uygunluk matrisi ve riskli maddeler dökümünün hazırlanması
- **Mülakat Sorusu:** *"Hukuki ve teknik sözleşmelerde yapılandırılmamış metinlerden regex ile bilgi çıkarırken yanlış eşleşmeleri (False Positive) önleyen bağlam sınırlayıcılar nelerdir?"*

### Gün 207: Akıllı Roaming (Yurt Dışı Dolaşım) Ülke ve Paket Eşleme Ajanı
- **İş Alanı:** Uluslararası Dolaşım (Roaming) & Dinamik Müşteri Deneyimi
- **Veri Kaynağı:** Uluslararası Şebeke Kodları (MCC-MNC) & Roaming Tarife Veri Tabanı
- **Model:** Ağaç Arama (Prefix Tree / Trie) + Deterministik Fiyat Kuralı
- **Türkçe Değişkenler:** `abone_id, baglanilan_mcc_mnc, tespit_edilen_ulke, mevcut_paket_kapsiyor_mu, onerilen_roaming_paketi`
- **Kapsam:** Yurt dışına çıkan abonenin telefonunun bağlandığı yabancı baz istasyonunun MCC-MNC kodunu anında ülkeye ve operatöre çevirerek en tasarruflu Turkcell dolaşım paketini önerir.
- **Jupyter Notebook (`gun_207_akilli_roaming_esleme_ajani.ipynb`):**
  1. Signal 7 / Diameter sinyalleşmesinden gelen MCC-MNC kodunun Trie yapısıyla ayrıştırılması
  2. Abonenin mevcut tarifesinin o ülkede 'Tarifen Yurt Dışında Geçerli' kapsamında olup olmadığının kontrolü
  3. Fatura şokunu önlemek amacıyla aboneye gidecek kişiselleştirilmiş bilgilendirme mesajının üretilmesi
- **Mülakat Sorusu:** *"Global operatör kodlarının (MCC-MNC) hızlı aranmasında Hash Table yerine Trie (Önek Ağacı) veri yapısı kullanmanın bellek ve arama karmaşıklığı avantajı nedir?"*

### Gün 208: Sosyal Medya Kriz ve Eskalasyon Karar Ajanı
- **İş Alanı:** Kurumsal İletişim, PR & Müşteri Deneyimi Kriz Masası
- **Veri Kaynağı:** Twitter/X ve Şikayetvar Turkcell Etiketli Sentetik Gönderiler
- **Model:** Sözlük Tabanlı Duygu Şiddeti Puanlayıcı + Eşik Tetikleme Mantığı
- **Türkçe Değişkenler:** `tweet_id, takipci_sayisi, icerik_metni, ofke_katsayisi_0_1, viral_kriz_riski, nobetci_mudure_sms_gitsin_mi`
- **Kapsam:** Sosyal medyada Turkcell hakkında yazılan şikayetleri takipçisi yüksek hesaplar, hukuki tehdit içeren kelimeler ve öfke şiddetine göre tarayarak viral krizleri dakikalar öncesinden yakalar.
- **Jupyter Notebook (`gun_208_sosyal_medya_kriz_eskalasyon_ajani.ipynb`):**
  1. Kriz sözlüğü (BTK, mahkeme, boykot, toplu kesinti) ağırlıklandırması ile gönderi taraması
  2. Takipçi sayısı ve etkileşim hızının logaritmik kriz yayılma katsayısına dönüştürülmesi
  3. Kriz skoru %80'i aşan durumlarda nöbetçi kriz masasına SMS/E-posta eskalasyon bildirimi
- **Mülakat Sorusu:** *"Sosyal medya kriz tespitinde sahte bot hesapların yarattığı yapay öfke dalgalarını gerçek müşteri infialinden ayıran filtreleme kuralları nelerdir?"*

### Gün 209: SIM-Kart & eSIM Aktivasyon ve ICCID Denetim Ajanı
- **İş Alanı:** Bayi Operasyonları & Dijital Kanal Hat Aktivasyon Masası
- **Veri Kaynağı:** SIM Kart Barkod Serileri & ICCID / IMSI / EID Veri Şablonları
- **Model:** Luhn Modülo 10 Algoritması + IIN (Issuer Identification) Doğrulama
- **Türkçe Değişkenler:** `girilen_iccid_no, luhn_gecerli_mi, operator_kodu_8990, esim_eid_uzunlugu, aktivasyon_onaylandi_mi`
- **Kapsam:** Mağazalardan veya web'den girilen 19-20 haneli SIM kart ICCID numaralarının Luhn checksum algoritmasıyla matematiksel doğruluğunu ve Turkcell seri aralığını anında teyit eder.
- **Jupyter Notebook (`gun_209_sim_iccid_luhn_denetim_ajani.ipynb`):**
  1. ICCID numarasının Türkiye ve Turkcell ön ekleri (899001...) açısından denetimi
  2. Luhn modülo 10 algoritması ile hatalı veya yanlış basılmış son basamak kontrolü
  3. eSIM için 32 haneli EID kimlik formatının doğrulanması ve otomatik profil atama
- **Mülakat Sorusu:** *"Kredi kartlarında ve telekom SIM kartlarında kullanılan Luhn algoritmasının tek basamaklı yazım hatalarını yakalamadaki matematiksel garantisi nedir?"*

### Gün 210: Bayi Cihaz Stok ve Yeniden Sipariş Ajanı
- **İş Alanı:** Tedarik Zinciri & Mağaza Lojistik Planlama
- **Veri Kaynağı:** Turkcell Mağazaları Akıllı Telefon & Aksesuar Satış Geçmişi
- **Model:** EOQ (Ekonomik Sipariş Miktarı) + Poisson Talep Dağılımı + Dinamik Emniyet Stoğu
- **Türkçe Değişkenler:** `magaza_kodu, urun_kodu, gunluk_ortalama_satis, teslim_suresi_gun, kritik_stok_seviyesi, siparis_adedi`
- **Kapsam:** Bayilerde telefon ve modem stoklarının tükenmesini önlemek için geçmiş satış hızına göre emniyet stoğu hesaplar ve tedarik merkezine otomatik sipariş emri oluşturur.
- **Jupyter Notebook (`gun_210_bayi_cihaz_stok_otonom_siparis_ajani.ipynb`):**
  1. Mağaza satış hızının ortalama ve varyansının Poisson dağılımı ile modellenmesi
  2. Tedarikçi teslim süresi (Lead Time) gecikmelerine karşı dinamik güvenlik stoğu hesabı
  3. Depo sipariş tetikleyicisinin otomatik üretilmesi ve fazla stok maliyeti optimizasyonu
- **Mülakat Sorusu:** *"Tedarik zincirinde 'Kamçı Etkisi'ni (Bullwhip Effect) azaltmak için bayi sipariş ajanında kullanılan Poisson talep modelinin üstünlüğü nedir?"*

### Gün 211: BiP Grup Mesajı Spam ve Oltalama Filtreleme Ajanı
- **İş Alanı:** BiP Mesajlaşma Platformu Güvenliği & İçerik Moderasyonu
- **Veri Kaynağı:** BiP Genel Kanalları Sentetik Mesaj Akışı & Spam Kelime Veri Seti
- **Model:** TF-IDF + Naive Bayes + Shannon URL Entropi Analizi (Yerel CPU)
- **Türkçe Değişkenler:** `mesaj_id, gonderici_id, icerik_metni, url_entropi_skoru, spam_olasiligi, mesaj_gizlensin_mi`
- **Kapsam:** BiP açık kanallarında paylaşılan bedava kontör, hediye çarkı ve kripto vaatli dolandırıcılık linklerini URL yapısı ve metin özellikleri üzerinden milisaniyeler içinde sessize alır.
- **Jupyter Notebook (`gun_211_bip_spam_filtreleme_ajani.ipynb`):**
  1. Mesaj içerisindeki linklerin çıkarılması ve karmaşık/rastgele domain entropisinin ölçülmesi
  2. Naive Bayes ile oltalama metin örüntülerinin (şifre gir, ödül kazan) sınıflandırılması
  3. Spam tespit edilen kullanıcının kanala mesaj atma yetkisinin geçici olarak kısıtlanması
- **Mülakat Sorusu:** *"Kötü niyetli bağlantıların tespitinde Shannon Bilgi Entropisi (Entropy) alan adı karmaşıklığını yakalamada neden etkilidir?"*

### Gün 212: Açık Sokak Haritası (OSM) ile Baz İstasyonları ve Yol Kapsama Analiz Ajanı
- **İş Alanı:** CBS (GIS) Kapsama Mühendisliği & Yol Güzergahı Şebeke Planlama
- **Veri Kaynağı:** OpenStreetMap (OSM) Istanbul Roads & Telecom Towers POI Açık Veri Kümesi (Kaggle/OSM)
- **Model:** GeoPandas / Shapely Mekânsal Tampon (Buffer) + Mesafe Kesişim Motoru
- **Türkçe Değişkenler:** `yol_segment_id, yol_turu_otoban_cadde, yakin_baz_istasyonu_mesafe_m, kapsama_altinda_mi, zayif_kapsama_uyarisi`
- **Kapsam:** OpenStreetMap'ten indirilen gerçek şehir yol ağında baz istasyonlarına 300 metreden uzak kalan kör sokakları tespit edip yeni anten öneri listesi çıkarır.
- **Jupyter Notebook (`gun_212_osm_yol_sebeke_kapsama_ajani.ipynb`):**
  1. OSM yol ağının GeoPandas ile yüklenip baz istasyonu noktaları etrafında 300m tampon (Buffer) oluşturulması
  2. Mekânsal kesişim (Spatial Join) ile kapsama dışı kalan kör sokak segmentlerinin ayrıştırılması
  3. Planlama birimine acil mikro-hücre (Small Cell) kurulması gereken sokakların listesinin raporlanması
- **Mülakat Sorusu:** *"OSM vektör verisinde çizgi (LineString) geometrileri ile baz istasyonu nokta (Point) geometrilerini R-Tree mekânsal indeksiyle saniyeler içinde kesiştirmenin mantığı nedir?"*

### Gün 213: TV+ Canlı Yayın Donma ve Buffer Teşhis Ajanı
- **İş Alanı:** Dijital TV+, CDN Mühendisliği & Akış Servisleri Kalitesi
- **Veri Kaynağı:** TV+ OTT Oynatıcı Telemetrisi (Buffer Süresi, Bitrate Düşüşü, FPS)
- **Model:** İstatistiksel Eşik Analizörü + Kayan Pencere (Moving Average) Teşhis Motoru
- **Türkçe Değişkenler:** `kullanici_id, kanal_adi, anlik_bitrate_kbps, tampon_bellek_saniye, darbogaz_yeri_cdn_mi_ev_wi_fi_mi`
- **Kapsam:** Canlı maç yayınında kullanıcının yayını donduğunda telemetriyi analiz eder; sorunun Turkcell CDN sunucusundan mı yoksa abonenin evindeki Wi-Fi zayıflığından mı kaynaklandığını teşhis eder.
- **Jupyter Notebook (`gun_213_tvplus_yayin_donma_teshis_ajani.ipynb`):**
  1. İstemci oynatıcıdan gelen saniyelik buffer sağlık metriklerinin normalize edilmesi
  2. Aynı CDN sunucusuna bağlı diğer 1000 kullanıcının genel sağlığı ile bireysel sağlık kıyası
  3. Kullanıcı ekranına otomatik 'Wi-Fi sinyaliniz zayıf, 5GHz bandına geçin' yönlendirmesi üretilmesi
- **Mülakat Sorusu:** *"Video akışında (HLS/DASH) QoE (Quality of Experience) ölçümünde 'Rebuffering Ratio' ile 'Bitrate Switching Frequency' dengesi nasıl kurulur?"*

### Gün 214: Bulut Sunucu Kaynak İsrafı ve FinOps Tasarruf Ajanı
- **İş Alanı:** Turkcell Bulut Teknolojileri & IT Altyapı FinOps Masası
- **Veri Kaynağı:** OpenStack / VMware Sunucu CPU, RAM ve Disk Kullanım Günlükleri
- **Model:** Eşik Tabanlı Kural Motoru + Lineer Regresyon Trend Tahmini
- **Türkçe Değişkenler:** `sunucu_id, son_7gun_maks_cpu, ortalama_ram_gb, atil_durumda_mi, onerilen_kucultme_orani, aylik_tasarruf_tl`
- **Kapsam:** Turkcell veri merkezlerindeki on binlerce sanal sunucuyu tarayarak 7 gün boyunca %5'in altında CPU kullanan atıl makineleri bulur; küçültme (downsize) önerileriyle bütçe tasarrufu sağlar.
- **Jupyter Notebook (`gun_214_bulut_finops_tasarruf_ajani.ipynb`):**
  1. Sunucu telemetrisinin quantile analizi (95th percentile CPU kullanımı hesabı)
  2. Zombi sunucuların (aktif ağ trafiği olmayan ama açık kalan makineler) tespiti
  3. Otomatik küçültme tavsiye tablosu ve sağlayacağı elektrik/lisans tasarrufu hesabı
- **Mülakat Sorusu:** *"Bulut kaynak optimizasyonunda ortalama CPU yerine 95. veya 99. yüzdelik dilim (percentile) kullanımına bakmanın sebebi nedir?"*

### Gün 215: SQL Sorgu Güvenliği ve İndeks Öneri Ajanı
- **İş Alanı:** Büyük Veri Mimarisi & Veritabanı Yönetim Sistemleri (DBA)
- **Veri Kaynağı:** PostgreSQL / Oracle Yavaş Sorgu (Slow Query) Log Dosyaları
- **Model:** SQLGlot AST (Soyut Sözdizim Ağacı) Ayrıştırıcı + Heuristik İndeksleyici
- **Türkçe Değişkenler:** `sorgu_metni, calisma_suresi_ms, taranan_tablo, where_kolonlari, onerilen_index_sql, tehlikeli_full_scan_mi`
- **Kapsam:** Veritabanına gönderilen sorguların AST ağacını yerel olarak çıkarır; WHERE ve JOIN koşullarında indeks olmayan kolonları bularak DBA ekibine hazır `CREATE INDEX` komutları üretir.
- **Jupyter Notebook (`gun_215_sql_optimizasyon_ve_indeks_ajani.ipynb`):**
  1. Yavaş çalışan SQL sorgularının SQLGlot ile ayrıştırılarak WHERE ve JOIN filtrelerinin çıkarılması
  2. Tablo üzerinde Sequential Scan (Tam Tablo Taraması) yapan sorguların tespit edilmesi
  3. Bileşik indeks (Composite Index) önerisinin üretilmesi ve beklenen sorgu hızlanma tahmini
- **Mülakat Sorusu:** *"İlişkisel veritabanlarında B-Tree indeksleme yapılırken sütun sıralamasının (Cardinality / Seçicilik) önemi nedir?"*

### Gün 216: Siber Güvenlik Loglarında CVE Tehdit Eşleme Ajanı
- **İş Alanı:** Siber Tehdit İstihbaratı (CTI) & SOC Analist Destek
- **Veri Kaynağı:** Web Sunucu Access Logları & NVD (National Vulnerability Database) CVE Listesi
- **Model:** Aho-Corasick Çoklu Dize Arama Algoritması + Regex Eşleme
- **Türkçe Değişkenler:** `istek_url, gelen_ip, tespit_edilen_cve_id, saldiri_turu_rce_sqli, engelleme_kurali_iptables`
- **Kapsam:** WAF ve sunucu erişim loglarında bilinen güvenlik açıklarını (Log4j, Spring4Shell vb.) Aho-Corasick algoritması ile saniyede yüz binlerce satır hızla tarayıp CVE koduyla eşler.
- **Jupyter Notebook (`gun_216_cve_tehdit_esleme_ajani.ipynb`):**
  1. Son 1 saatlik HTTP istek url ve gövdelerinin yerel olarak okunması
  2. Aho-Corasick algoritması ile 2000'den fazla bilinen saldırı deseninin tek geçişte aranması
  3. Saldırgan IP adreslerinin otomatik Linux iptables kuralı formatına dökülmesi
- **Mülakat Sorusu:** *"N adet metin desenini M uzunluğundaki log akışında ararken N kez Regex çalıştırmak yerine Aho-Corasick algoritması kullanmanın zaman karmaşıklığı farkı nedir?"*

### Gün 217: Şeffaf Fatura Açıklama ve İtiraz Çözüm Robotu
- **İş Alanı:** Faturalama Sistemleri (Billing) & Müşteri Şikayet Yönetimi
- **Veri Kaynağı:** Turkcell Bireysel Müşteri Son 2 Dönem Detaylı Fatura Kalemleri
- **Model:** Deterministik Ayrıştırıcı Motor + Kuralcı Metin Üretim Şablonu
- **Türkçe Değişkenler:** `abone_id, onceki_fatura_tl, guncel_fatura_tl, fark_tutari_tl, temel_sebep_asım_vergi_servis, aciklama_metni`
- **Kapsam:** Abonenin faturasının neden yükseldiğini merak ettiği durumlarda iki fatura arasındaki farkı kalem kalem (ek paket, cayma bedeli, telsiz kullanım vergisi, yurt dışı arama) inceler ve net Türkçe açıklama yazar.
- **Jupyter Notebook (`gun_217_seffaf_fatura_aciklama_ajani.ipynb`):**
  1. Geçen ay ile bu ayın detaylı fatura kalemlerinin (Roaming, SMS, Veri Aşımı, Katma Değerli Servisler) fark tablosu
  2. En büyük artışın yaşandığı ana kalemlerin oran bazlı sıralanması
  3. Müşterinin anlayacağı sadelikte 'Faturanız geçen aya göre 45 TL fazla, çünkü 1GB ek internet paketi aldınız' mesajı çıktısı
- **Mülakat Sorusu:** *"Finansal sistemlerde müşteri açıklama motoru kurgularken deterministik kural motoru kullanmanın üretken LLM modellerine göre yasal ve denetimsel güvencesi nedir?"*

### Gün 218: Bayi Standart Denetim ve Gizli Müşteri Puanlayıcı Ajan
- **İş Alanı:** Satış Kanalları Denetimi & Kalite Güvence (QA)
- **Veri Kaynağı:** Bayi Müşteri Temsilcisi Satış Ses Transkriptleri (Sentetik Metin)
- **Model:** Kelimeler Arası N-Gram Kapsama Skoru + Rubrik Puanlama Algoritması
- **Türkçe Değişkenler:** `bayi_id, gorusme_metni, selamlama_yapildi_mi, kampanya_anlatildi_mi, vedalasma_tam_mi, kalite_puani_100`
- **Kapsam:** Mağaza satış görüşmelerinin dökümlerini inceleyerek Turkcell standart selamlama, cihaz sigortası teklifi, numara taşıma bilgilendirmesi ve vedalaşma aşamalarını kontrol edip puanlar.
- **Jupyter Notebook (`gun_218_bayi_denetim_rubrik_ajani.ipynb`):**
  1. Turkcell Kurumsal Satış Standartları kontrol listesinin (Checklist) hazırlanması
  2. Transkriptte zorunlu cümle ve anahtar kelime öbeklerinin esnek n-gram eşlemesi
  3. Bayi bazında aylık kalite karnesi ve eğitim alması gereken personelin raporlanması
- **Mülakat Sorusu:** *"Çağrı metinlerinde tam eşleşmeyen ancak eş anlamlı selamlama ifadelerini (Örn: 'Merhaba', 'İyi günler') yerel olarak puanlamada Word2Vec Cosine eşiği nasıl kullanılır?"*

### Gün 219: KVKK / PII Hassas Veri Sızıntı Önleme Ajanı
- **İş Alanı:** Bilgi Güvenliği, Uyum & Hukuk Teknolojileri (RegTech)
- **Veri Kaynağı:** Turkcell Çağrı Notları, Müşteri Talepleri & Veritabanı Metin Alanları
- **Model:** TCKN Modülo 10/11 Doğrulayıcı + Kredi Kartı Luhn + GSM Regex Filtresi
- **Türkçe Değişkenler:** `ham_metin, tespit_edilen_tckn_sayisi, maskelenmis_metin, kvkk_ihlal_seviyesi, guvenli_metin_ciktisi`
- **Kapsam:** Veritabanına ve analiz sistemlerine giden serbest metin alanlarında unutulan T.C. Kimlik Numarası, kredi kartı ve cep telefonu bilgilerini algoritmik olarak doğrulayıp `***` ile maskeler.
- **Jupyter Notebook (`gun_219_kvkk_pii_veri_maskeleme_ajani.ipynb`):**
  1. 11 haneli sayıların TCKN matematiksel kuralı (1. 3. 5. 7. 9. basamaklar toplamı formülü) ile doğrulanması
  2. Kredi kartı numaralarının ve IBAN kalıplarının regex ile taranıp maskelenmesi
  3. Maskelenmiş metnin analitik modellere güvenle beslenmesini sağlayan temizleme hattı
- **Mülakat Sorusu:** *"Basit bir 11 haneli sayı dizisinin rastgele bir sayı mı yoksa gerçek bir TCKN mi olduğunu doğrulayan algoritmik modülo formülü nedir?"*

### Gün 220: Multi-Agent Şebeke Kurtarma Simülasyonu (Modül Capstone)
- **İş Alanı:** Uçtan Uca Otonom Ağ Yönetimi & Orkestrasyon
- **Veri Kaynağı:** Büyük Şebeke Çökmesi Sentetik Olay Günlüğü & Alarm Verisi
- **Model:** 3 Senkronize Durum Makinesi Ajanı (Gözlemci + Karar Verici + Uygulayıcı)
- **Türkçe Değişkenler:** `olay_adimi, gozlemci_durumu, teshis_edilen_hata, uygulanan_cozum_komutu, sebeke_kurtarildi_mi`
- **Kapsam:** Büyük bir fırtına sonrası çöken transmisyon hattında 3 farklı yerel ajan (Log Gözlemcisi, Rota Planlayıcısı ve Komut Yürütücüsü) sırayla haberleşerek şebekeyi ayağa kaldırır.
- **Jupyter Notebook (`gun_220_multi_agent_sebeke_kurtarma_simulasyonu.ipynb`):**
  1. Gözlemci Ajanın (Monitor) alarm günlüğünü okuyup kopan hattı tespit etmesi
  2. Karar Verici Ajanın (Planner) yedek mikrodalga link üzerinden alternatif rota çizmesi
  3. Uygulayıcı Ajanın (Executor) yönlendirme tablolarını güncelleyerek trafiği kurtarması
- **Mülakat Sorusu:** *"Çok ajanlı (Multi-Agent) deterministik sistemlerde ajanlar arası mesajlaşmada 'Yarış Durumu'nu (Race Condition) önlemek için Actor Modeli nasıl uygulanır?"*


## Modül 18: 5G Advanced, O-RAN & Akıllı Şebeke Optimizasyonu

---

### Gün 221: CRAWDAD Açık Verisiyle Çok Kullanıcılı Wi-Fi/Hücresel Frekans Tahsis Motoru
- **İş Alanı:** Kablosuz Şebeke Optimizasyonu & Dinamik Frekans Tahsisi
- **Veri Kaynağı:** CRAWDAD Wireless Network Trace Dataset (IEEE / Dartmouth Açık Verisi)
- **Model:** Scikit-learn Random Forest + Açgözlü (Greedy) Frekans Atayıcı
- **Türkçe Değişkenler:** `kullanici_mac_adresi, anlik_rssi_dbm, talep_edilen_bant_mbps, atanan_kanal_frekansi, cakisma_orani`
- **Kapsam:** CRAWDAD açık veri setindeki binlerce gerçek kablosuz bağlantı ölçümünü kullanarak en az çakışma yaratan dinamik kanal ve frekans atamasını yapar.
- **Jupyter Notebook (`gun_221_crawdad_kablosuz_frekans_tahsisi.ipynb`):**
  1. CRAWDAD gerçek kablosuz telemetri akışından kullanıcı bazlı sinyal ve trafik taleplerinin çıkarılması
  2. Kanal kapasitesi ve parazit matrisini optimize eden hafif Random Forest çıkarımı
  3. Aynı anda bağlanan yüzlerce cihaz için çakışmasız frekans tablosunun üretilmesi
- **Mülakat Sorusu:** *"Açık spektrumda dinamik kanal ataması yapılırken gizli düğüm (Hidden Node Problem) çakışmaları makine öğrenimi öznitelikleriyle nasıl tespit edilir?"*

### Gün 222: Veri Merkezi Sunucu Sıcaklık ve PUE Enerji Tasarrufu Optimizasyonu
- **İş Alanı:** Bulut Altyapısı, Veri Merkezi Mühendisliği & Yeşil Enerji
- **Veri Kaynağı:** Kaggle - ASHRAE Great Energy Predictor & Data Center Thermal Sensor Open Dataset
- **Model:** LightGBM Regresyon + Eşik Tabanlı Akıllı Soğutma Karar Motoru
- **Türkçe Değişkenler:** `sunucu_oda_sicakligi_c, cpu_yuk_yuzdesi, dis_hava_sicakligi, sogutma_enerjisi_kwh, pue_verimlilik_orani`
- **Kapsam:** Kaggle ASHRAE veri setini kullanarak sunucu yükü ve dış hava sıcaklığına göre soğutma sistemlerini dinamik ayarlayarak PUE enerji tüketimini minimize eder.
- **Jupyter Notebook (`gun_222_veri_merkezi_pue_enerji_optimizasyonu.ipynb`):**
  1. Kaggle veri setindeki sunucu elektrik ve ortam sıcaklığı ölçümlerinin zaman serisi analizi
  2. LightGBM ile dış hava koşullarına göre gelecek saatlik soğutma ihtiyacının tahmini
  3. PUE (Power Usage Effectiveness) oranını 1.2 altına çeken soğutma ayar kurallarının üretilmesi
- **Mülakat Sorusu:** *"Veri merkezlerinde PUE (Power Usage Effectiveness) metriği nasıl hesaplanır ve serbest soğutma (Free Cooling) devreye girdiğinde ML modeli bunu nasıl dengeler?"*

### Gün 223: Network Slicing Dinamik Kaynak Tahsis Motoru
- **İş Alanı:** 5G Core & Ağ Dilimleme (Slicing) Kaynak Yönetimi
- **Veri Kaynağı:** 3 Farklı Dilim (eMBB Video, URLLC Otonom Araç, mMTC Sayaç) Trafik Verisi
- **Model:** Tabular Q-Learning (Bellek < 1MB, Süre < 5ms)
- **Türkçe Değişkenler:** `dilim_turu, anlik_istek_sayisi, garanti_edilen_bant_genisligi, atanan_prb_kaynak_blok, sla_ihlal_orani`
- **Kapsam:** 5G şebekesinde ambulans veya otonom araç gibi gecikmeye tahammülü olmayan kritik dilimlere acil durumlarda diğer dilimlerden kaynak aktaran dinamik pekiştirmeli öğrenme motorudur.
- **Jupyter Notebook (`gun_223_5g_network_slicing_q_learning.ipynb`):**
  1. Üç farklı 5G diliminin SLA gereksinimlerinin (Gecikme < 1ms vs Bant Genişliği > 100Mbps) tanımlanması
  2. Q-Learning ödül fonksiyonunda SLA ihlaline yüksek ceza puanı verilmesi
  3. Şebeke yükü %100'e ulaştığında kritik dilimin kesintiye uğramadığının doğrulanması
- **Mülakat Sorusu:** *"Network Slicing ortamında eMBB dilimi trafiği patladığında URLLC diliminin PRB (Physical Resource Block) izolasyonu donanım seviyesinde nasıl garanti edilir?"*

### Gün 224: Gerçek Ağ Trafiği Akış Sınıflandırması (QoS Paket Analitiği)
- **İş Alanı:** Çekirdek Şebeke Trafik Mühendisliği & Servis Kalitesi (QoS)
- **Veri Kaynağı:** Kaggle - Network Packet Flow (ISCX VPN-nonVPN / CICFlowMeter Open Dataset)
- **Model:** Random Forest Sınıflandırıcı + SHAP Öznitelik Katkısı
- **Türkçe Değişkenler:** `akis_suresi_ms, paket_boyutu_ortalama, tcp_pencere_boyutu, protokol_turu, tahmin_edilen_servis_video_voip_web`
- **Kapsam:** ISCX gerçek ağ trafiği veri kümesindeki 80'den fazla akış özelliğini kullanarak şifreli paketlerin türünü derin paket inceleme (DPI) yapmadan sınıflandırır.
- **Jupyter Notebook (`gun_224_gercek_ag_akisi_qos_siniflandirma.ipynb`):**
  1. CICFlowMeter özniteliklerinin (Paket boyutu varyansı, akışlar arası varış süresi) yüklenmesi
  2. Random Forest ile şifreli paket akışının Video, Sesli Arama veya Web olarak sınıflandırılması
  3. Kritik VoIP ve acil durum akışlarına QoS öncelik kuyruğu (DiffServ EF) atanması
- **Mülakat Sorusu:** *"Şifreli ağ trafiğinde (HTTPS/QUIC) paket içeriğini görmeden akış istatistikleri (Inter-Arrival Time, Packet Length) üzerinden uygulama sınıflandırması nasıl yapılır?"*

### Gün 225: Yüksek Hızlı Tren (YHT) Seyir Halinde Hücresel Sinyal Düşüş Tahmini
- **İş Alanı:** Hareketli Kapsama Mühendisliği & Ulaşım Telekomünikasyonu
- **Veri Kaynağı:** Kaggle - High-Speed Rail Wireless Channel Open Dataset / Train Movement Telemetry
- **Model:** Genişletilmiş Kalman Filtresi (EKF) + Ridge Regresyonu
- **Türkçe Değişkenler:** `tren_hizi_kmh, baz_istasyonuna_mesafe_m, olculan_rsrp_dbm, tahmin_edilen_sinyal_kaybi, erken_handover_tetikle`
- **Kapsam:** Saatte 250 km hızla giden trenden toplanmış gerçek sinyal telemetrisini Kalman filtresiyle düzelterek tünel veya baz istasyonu geçişlerinde çağrı kopmasını önceden tahmin eder.
- **Jupyter Notebook (`gun_225_yuksek_hizli_tren_sinyal_tahmini.ipynb`):**
  1. Yüksek hızlı tren güzergahındaki gerçek RSRP sinyal dalgalanmalarının okunması
  2. Genişletilmiş Kalman Filtresi ile gürültülü ve Doppler etkili sinyal serisinin düzeltilmesi
  3. Sinyalin -110 dBm altına ineceği anlarda erken el değiştirme (Fast Handover) sinyali üretimi
- **Mülakat Sorusu:** *"Hızlı tren seyahatlerinde Doppler kayması ve hızlı sönümleme (Fast Fading) baz istasyonu el değiştirmelerinde (Handover Failure) nasıl bir ping-pong etkisine yol açar?"*

### Gün 226: Gerçek Çok Katlı Bina İç Mekan Sinyal Gücü (RSRP) Haritalama
- **İş Alanı:** İç Mekan Kapsama Mühendisliği & Akıllı Bina Altyapısı
- **Veri Kaynağı:** UCI Machine Learning Repository - UJIIndoorLoc WiFi/Cellular Fingerprint Dataset
- **Model:** K-En Yakın Komşu (KNN Regresyonu) + Çok Çıktılı Random Forest
- **Türkçe Değişkenler:** `kat_numarasi, bina_kodu, sinyal_gucu_vektoru_520, tahmin_edilen_x_y_metre, konumlandirma_hatasi_m`
- **Kapsam:** UCI UJIIndoorLoc açık veri setindeki 21.000'den fazla gerçek sinyal parmak izini kullanarak çok katlı binalarda kör noktaları ve sinyal haritasını santimetre hassasiyetinde çıkarır.
- **Jupyter Notebook (`gun_226_ic_mekan_sinyal_haritalama_ujiindoorloc.ipynb`):**
  1. UJIIndoorLoc veri setindeki 520 erişim noktası sinyal gücü vektörlerinin normalize edilmesi
  2. KNN ve Random Forest ile kullanıcının bulunduğu katın ve (X, Y) koordinatlarının tahmini
  3. Binadaki sinyalin zayıf olduğu kör noktaların 3 boyutlu ısı haritasının çıkarılması
- **Mülakat Sorusu:** *"İç mekan konumlandırmada radyo parmak izi (Fingerprinting) yönteminde çevresel değişikliklerin (insan yoğunluğu, mobilya) yarattığı sinyal sapmaları nasıl kalibre edilir?"*

### Gün 227: RadioML Açık Verisiyle Radyo Frekans Modülasyonu ve Parazit Sınıflandırma
- **İş Alanı:** Bilişsel Radyo (Cognitive Radio) & RF Spektrum İzleme Masası
- **Veri Kaynağı:** Kaggle - RadioML 2016.10A / GNU Radio Açık Modülasyon Veri Kümesi
- **Model:** 1D-CNN (PyTorch Hafif CPU) + FFT Spektral Enerji Çıkarımı
- **Türkçe Değişkenler:** `iq_ornek_dizisi_128, snr_degeri_db, tahmin_edilen_modulasyon_qam16_qpsk_bpsk, gurultu_orani, sinyal_tanindi_mi`
- **Kapsam:** Dünyanın en ünlü açık RF veri seti olan RadioML'deki gerçek I/Q ham sinyallerini inceleyerek parazitli ortamda modülasyon türünü (BPSK, QPSK, 16QAM, WBFM) milisaniyede tanır.
- **Jupyter Notebook (`gun_227_radioml_rf_modulasyon_siniflandirma.ipynb`):**
  1. 220.000 örnekli RadioML veri setinden 128 boyutlu I ve Q kanallarının çıkarılması
  2. Hafif 1D-CNN mimarisiyle farklı SNR seviyelerinde (-20 dB ile +18 dB) model eğitimi
  3. Bilinmeyen veya parazitli sinyallerin anında tespit edilip spektrum raporu üretilmesi
- **Mülakat Sorusu:** *"I/Q (In-phase / Quadrature) sinyal temsilinin genlik ve faz bilgisini korumadaki matematiksel avantajı nedir ve derin öğrenmeye 2 kanallı dizi olarak nasıl beslenir?"*

### Gün 228: Baz İstasyonu Trafik Dijital İkizi (Network Digital Twin)
- **İş Alanı:** Şebeke Planlama & Simülasyon Tabanlı Test Mühendisliği
- **Veri Kaynağı:** Gerçek Şebeke Hücre Konfigürasyonları ve Abone Hareket Profilleri
- **Model:** Ayrık Olay Simülasyonu (Discrete-Event Simulation - SimPy)
- **Türkçe Değişkenler:** `simulasyon_zamani_dakika, baglanan_abone_sayisi, veri_indirme_hizi_mbps, paket_kaybi_yuzde, yeni_anten_eklensin_mi`
- **Kapsam:** Canlı şebekeye dokunmadan önce, bir stadyum veya meydandaki baz istasyonunun sanal ikizini kurarak 50.000 kişilik kalabalıkta şebekenin nasıl davranacağını test eder.
- **Jupyter Notebook (`gun_228_baz_istasyonu_dijital_ikizi_simpy.ipynb`):**
  1. SimPy ortamında baz istasyonu kaynaklarının ve kullanıcı bağlantı süreçlerinin kodlanması
  2. Konser başlangıcı gibi ani yük artışlarının simüle edilmesi
  3. Yeni bir frekans bandı açıldığında paket kaybının nasıl düştüğünün sanal ortamda kanıtlanması
- **Mülakat Sorusu:** *"Telekomünikasyonda Dijital İkiz (Digital Twin) oluştururken zaman serisi tahmin modelleri yerine Ayrık Olay Simülasyonu (DES) kullanmanın avantajı nedir?"*

### Gün 229: 5G Kapalı Alan Konumlandırma (Sub-Meter Positioning)
- **İş Alanı:** Akıllı Depolar, Endüstri 4.0 & İç Mekan Konumlandırma
- **Veri Kaynağı:** İç Mekan Mikro Baz İstasyonu RSSI ve Uçuş Süresi (ToF) Verileri
- **Model:** Trilaterasyon Geometrisi + K-En Yakın Komşu (KNN) Düzeltmesi
- **Türkçe Değişkenler:** `anten_mesafe_metre_1_2_3, tahmin_edilen_x_y, gercek_x_y, konum_hatasi_metre, forklift_guvenli_bolgede_mi`
- **Kapsam:** GPS sinyalinin girmediği fabrikalarda ve AVM'lerde, cihazın 3 farklı 5G mikro hücresine olan mesafesini ölçerek 1 metrenin altında hassasiyetle konumunu belirler.
- **Jupyter Notebook (`gun_229_5g_kapali_alan_konumlandirma.ipynb`):**
  1. Sinyal zayıflama modelinden (Log-Distance Path Loss) mesafelerin kestirimi
  2. Üç çemberin kesişim noktasını bulan en küçük kareler (Least Squares) trilaterasyonu
  3. Çok yollu yansıma (Multipath) kaynaklı hataların KNN ile kalibre edilmesi
- **Mülakat Sorusu:** *"İç mekan konumlandırmada ToA (Time of Arrival), TDoA (Time Difference of Arrival) ve RSSI yöntemlerinin doğruluk ve donanım karmaşıklığı kıyası nasıldır?"*

### Gün 230: V2X Araçlar Arası Acil Mesajlaşma Gecikme Analizi
- **İş Alanı:** Bağlantılı Araçlar, Otonom Sürüş & C-V2X İletişimi
- **Veri Kaynağı:** C-V2X Sidelink (PC5) Doğrudan Araç Haberleşmesi Paket Kayıtları
- **Model:** Weibull Dağılımı + Uç Değer Analizi (Extreme Value Theory)
- **Türkçe Değişkenler:** `arac_id, mesafe_metre, iletim_gecikmesi_ms, paket_ulasma_orani_pdr, carpisma_riski_var_mi`
- **Kapsam:** Otonom araçların baz istasyonuna gitmeden doğrudan birbirine attığı 'Acil Fren' uyarılarının 5 milisaniyenin altında kalıp kalmadığını aşırı gecikme olasılık modelleriyle test eder.
- **Jupyter Notebook (`gun_230_v2x_gecikme_ve_guvenilirlik_analizi.ipynb`):**
  1. Araç yoğunluğuna göre paket çarpışmalarının ve kuyruk gecikmelerinin modellenmesi
  2. Weibull dağılımı uydurularak gecikmenin 10ms'yi aşma ihtimalinin (99.999% güvenilirlik) hesabı
  3. Fren mesafesini kurtaran kritik güvenlik menzilinin doğrulanması
- **Mülakat Sorusu:** *"C-V2X Sidelink iletişiminde baz istasyonu kontrolü olmadan araçların radyo frekans kaynaklarını kendi seçtiği Mode 4 mekanizması nasıl çalışır?"*

### Gün 231: Milano Telekom CDR Açık Verisiyle Şehir Hücresel Trafik Sıkışıklığı Tahmini
- **İş Alanı:** Şehir Şebeke Trafik Planlama & Kapasite Yönetimi
- **Veri Kaynağı:** Harvard Dataverse / Kaggle - Telecom Italia Big Data Challenge Open CDR Dataset
- **Model:** Facebook Prophet / SARIMAX ile Saatlik Hücresel Veri Trafiği Tahmini
- **Türkçe Değişkenler:** `grid_alan_id, zaman_saat, internet_hacmi_mb, sms_sayisi, cagri_hacmi, kapasite_asimi_riski`
- **Kapsam:** Telecom Italia tarafından kamuya açılan gerçek 10.000 ızgaralık hücresel veri trafiğini inceleyerek akşam saatlerindeki internet tıkanmalarını 24 saat önceden tahmin eder.
- **Jupyter Notebook (`gun_231_milano_telekom_cdr_trafik_tahmini.ipynb`):**
  1. Milano şehir ızgara haritasındaki saatlik internet tüketim zaman serilerinin yüklenmesi
  2. Prophet modeliyle haftalık döngüsellik (Seasonality) ve tatil günü etkilerinin modellenmesi
  3. Gelecek 24 saatte baz istasyonu kapasitesini %85 üzerinde zorlayacak yoğun bölgelerin haritalanması
- **Mülakat Sorusu:** *"Büyük veri ölçeğinde mekânsal-zamansal (Spatio-Temporal) telekom CDR trafiğini modellerken mevsimsellik (Seasonality) ve tatil günleri etkisi nasıl ayrıştırılır?"*

### Gün 232: Endüstriyel Özel 5G (Private 5G) Jitter İstatistiksel Analitiği
- **İş Alanı:** Turkcell Kurumsal Özel Şebekeler & Akıllı Fabrika Çözümleri
- **Veri Kaynağı:** Fabrika PLC ve Sensör Ağ Paket Gecikme Değişimi (Jitter) Kayıtları
- **Model:** İstatistiksel Süreç Kontrolü (SPC) + Z-Skor Anomali Dedektörü
- **Türkçe Değişkenler:** `paket_no, gecikme_degisimi_mikrosaniye, ust_kontrol_limiti, jitter_anomalisi_var_mi, montaj_hatti_uyarisi`
- **Kapsam:** Otomotiv fabrikasındaki robot kollarının senkron çalışabilmesi için gereken mikrosaniye seviyesindeki jitter sapmalarını denetleyerek montaj hattında hata oluşmasını önler.
- **Jupyter Notebook (`gun_232_private_5g_jitter_spc_analizi.ipynb`):**
  1. Paketler arası varış zamanı farkının (Inter-Arrival Jitter) mikrosaniye hassasiyetinde hesabı
  2. Shewhart Kontrol Grafikleri ile istatistiksel sınırların (UCL / LCL) belirlenmesi
  3. Robotların senkronizasyonunun bozulma riskine karşı alarm üretilmesi
- **Mülakat Sorusu:** *"TSN (Time-Sensitive Networking) ve 5G URLLC entegrasyonunda 'Jitter' değerinin sıfıra yakın tutulabilmesi için radyo katmanında hangi zamanlama protokolleri kullanılır?"*

### Gün 233: Mobil Kenar Bilişim (MEC) Görev Boşaltma Karar Motoru
- **İş Alanı:** MEC (Multi-Access Edge Computing) & Dağıtık Bulut Mimarisi
- **Veri Kaynağı:** Kullanıcı Cihaz Pil Seviyesi, CPU Yükü ve Ağ Gecikme Profilleri
- **Model:** 0-1 Knapsack (Sırt Çantası) Dinamik Programlama Algoritması
- **Türkçe Değişkenler:** `gorev_id, hesaplama_yuku_mips, aktarim_boyutu_mb, yerel_enerji_mwh, kenar_enerji_mwh, gorev_kenara_gitsin_mi`
- **Kapsam:** Telefonun pili azaldığında veya işlemcisi yetmediğinde, ağır bir yapay zeka görevini baz istasyonundaki yerel kenar sunucuya mı göndermeli yoksa telefonda mı çözmeli kararını optimize eder.
- **Jupyter Notebook (`gun_233_mec_gorev_bosaltma_knapsack.ipynb`):**
  1. Yerel çalışma enerjisi ile ağdan transfer enerjisinin analitik modellemesi
  2. Gecikme bütçesi kısıtı altında toplam enerji tüketimini minimize eden dinamik programlama
  3. Farklı sinyal kalitesi koşullarında en doğru hesaplama boşaltma kararlarının doğrulanması
- **Mülakat Sorusu:** *"Mobil Kenar Bilişimde (MEC) 'Task Offloading' yaparken kablosuz kanal kalitesi aniden düşerse görevin kenara aktarılması neden ters tepebilir?"*

### Gün 234: Gerçek Hava Durumu ve Yağışın Mikrodalga Radyo Link Sinyal Zayıflamasına Etkisi
- **İş Alanı:** Mikrodalga Transmisyon Mühendisliği & Meteorolojik Sinyal Analizi
- **Veri Kaynağı:** Open-Meteo Historical Weather API & ITU-R P.838 Açık Yağış Zayıflama Verisi
- **Model:** SciPy Eğri Uydurma + ITU-R P.838 Yağış Sönümleme Formülasyonu
- **Türkçe Değişkenler:** `yagis_miktari_mm_saat, link_mesafesi_km, tasiyici_frekans_ghz, polarizasyon_yatay_dikey, sinyal_zayiflama_db`
- **Kapsam:** Open-Meteo gerçek geçmiş yağış verisi ile mikrodalga transmisyon linklerinin dB kayıplarını modelleyerek şiddetli sağanaklarda linkin kopmasını önceden uyarır.
- **Jupyter Notebook (`gun_234_yagis_mikrodalga_link_zayiflama_modeli.ipynb`):**
  1. Open-Meteo API'sinden saatlik yağış şiddeti (mm/h) verilerinin çekilip eşlenmesi
  2. ITU-R P.838 standardı (gamma = k * R^alpha) katsayılarının SciPy ile optimize edilmesi
  3. Şiddetli fırtınada mikrodalga linkte beklenen dB kaybının hesaplanarak otomatik uyarı verilmesi
- **Mülakat Sorusu:** *"Mikrodalga frekanslarında (13-38 GHz) yağmur damlalarının çapı dalga boyuna yaklaştığında Rayleigh saçılması yerine Mie saçılmasının devreye girmesi sinyali nasıl etkiler?"*

### Gün 235: Otomatik Frekans ve Fiziksel Hücre Kimliği (PCI) Planlama (Modül Capstone)
- **İş Alanı:** RAN Planlama Mühendisliği & Şebeke Kurulum Otomasyonu
- **Veri Kaynağı:** Türkiye Geneli Baz İstasyonu Coğrafi Konumları ve Komşuluk Listeleri
- **Model:** Graf Renklendirme (Graph Coloring) Açgözlü (Greedy) Algoritması
- **Türkçe Değişkenler:** `istasyon_sayisi_500, komsu_hucre_kenarlari, atanabilen_pci_havuzu_504, pci_cakisma_sayisi, minimum_renk_sayisi`
- **Kapsam:** Yeni kurulan 500 baz istasyonuna, birbirini gören komşu kuleler aynı kodu almayacak şekilde 504 adet PCI kimliğini sıfır çakışma ile otomatik dağıtan optimizasyon motorudur.
- **Jupyter Notebook (`gun_235_otomatik_pci_frekans_planlama.ipynb`):**
  1. Baz istasyonlarının coğrafi komşuluk grafının (Delaunay Nirengisi) kurulması
  2. Welsh-Powell ve DSATUR graf renklendirme algoritmaları ile PCI kodlarının atanması
  3. PCI Collision (aynı hücre kodu) ve PCI Confusion (aynı komşuya sahip olma) denetimi
- **Mülakat Sorusu:** *"LTE ve 5G şebekelerinde 'PCI Collision' ile 'PCI Confusion' kavramları arasındaki fark nedir ve eldeki 504 kod sınırlı olduğunda ne yapılır?"*


## Modül 19: Hafif Çok Modlu (Multimodal) Çıkarım & Yerel Görü-Metin Füzyonu

---

### Gün 236: Saha Ekipmanı Görsel Arıza Danışmanı
- **İş Alanı:** Saha Operasyonları & Teknisyen Mobil Yardım Terminali
- **Veri Kaynağı:** Baz İstasyonu Güç Panosu ve Akü Fotoğrafları Veri Seti
- **Model:** PyTorch MobileNetV3 (Hafif Görsel Çıkarım) + Kural Tabanlı Çözüm Adımları
- **Türkçe Değişkenler:** `pano_fotografi, tespit_edilen_parca, yanik_sigorta_var_mi, guvenlik_riski_seviyesi, teknisyene_talimat_metni`
- **Kapsam:** Saha teknisyeninin cep telefonuyla çektiği elektrik panosu fotoğrafındaki yanık sigortayı veya gevşek kabloyu yerel MobileNet modeliyle milisaniyede tanıyıp Türkçe onarım adımlarını ekrana döker.
- **Jupyter Notebook (`gun_236_saha_pano_ariza_gorsel_danisman.ipynb`):**
  1. Saha panosu görselinin 224x224 boyutuna getirilip MobileNetV3 ile sınıflandırılması
  2. Tespit edilen arıza sınıfı (Örn: Aşırı Isınmış Şalter) ile ilişkili prosedürün kütüphaneden çekilmesi
  3. Önce elektriği kes uyarısı içeren güvenli iş adımı yönergesinin oluşturulması
- **Mülakat Sorusu:** *"Mobil cihazlarda çalışan hafif evrişimli ağlarda (MobileNet) standart evrişim yerine 'Depthwise Separable Convolution' kullanmanın işlem yükünü azaltma mantığı nedir?"*

### Gün 237: Kimlik Kartı & Belge Düzeni Doğrulama Motoru
- **İş Alanı:** Müşteri Kabul Masası (KYC) & Dijital Başvuru Denetimi
- **Veri Kaynağı:** Sentetik T.C. Kimlik Kartı & Sürücü Belgesi Şablon Görüntüleri
- **Model:** OpenCV Kenar / Perspektif Düzeltme + Yerel Karakter Şablon Eşleme
- **Türkçe Değişkenler:** `belge_fotografi, dort_kose_koordinatlari, duzeltilmis_belge, seri_no_yeri_dogru_mu, onay_puani`
- **Kapsam:** Cep telefonuyla yamuk veya açılı çekilen kimlik kartı fotoğraflarının kenarlarını bularak görüntüyü düzeltir (Perspective Warp) ve çip, fotoğraf, TCKN alanlarının doğru yerde olup olmadığını denetler.
- **Jupyter Notebook (`gun_237_kimlik_belge_perspektif_ve_duzen_dogrulama.ipynb`):**
  1. Canny kenar bulma ve en büyük 4 köşeli konturun (Kare/Dikdörtgen) tespiti
  2. Perspektif dönüşümü (Four-Point Perspective Transform) ile belgenin taranmış gibi hizalanması
  3. Fotoğraf, çip ve unvan bölgelerinin piksel koordinat kontrolleriyle sahtecilik denetimi
- **Mülakat Sorusu:** *"Görüntü işlemede homografi matrisi (Homography Matrix) ile perspektif düzeltme yaparken en az kaç adet eşleşen referans noktasına ihtiyaç vardır?"*

### Gün 238: fizy Şarkı Kapağı Renk Paleti ve Mod Eşleme
- **İş Alanı:** Dijital Müzik Platformu fizy & Akıllı UI Temalandırma
- **Veri Kaynağı:** Albüm Kapak Görselleri & Şarkı Sözleri Metin Verisi
- **Model:** K-Means Renk Kümeleme (Görsel) + VADER / Sözlük Mod Analizi (Metin)
- **Türkçe Değişkenler:** `album_kapagi_resmi, hakim_3_renk_rgb, sarki_sozu_duygu_skoru, sarki_modu_enerjik_huzunlu, fizy_ui_arka_plan_rengi`
- **Kapsam:** Şarkı çalarken albüm kapağındaki hâkim renkleri çıkarır ve şarkı sözünün neşeli/hüzünlü moduna göre fizy uygulamasının arayüz arka plan rengini dinamik olarak değiştirir.
- **Jupyter Notebook (`gun_238_fizy_album_kapak_mod_esleme.ipynb`):**
  1. Albüm kapağı piksellerinin K-Means (k=3) ile kümelenerek dominant renk paletinin çıkarılması
  2. Şarkı sözlerindeki kelimelerden pozitif/negatif enerji skorunun hesaplanması
  3. Müzik dinleme ekranı için kontrastı yüksek erişilebilir dinamik CSS renk kodu üretimi
- **Mülakat Sorusu:** *"Görüntü piksellerinden renk paleti çıkarırken RGB uzayı yerine Lab veya HSV renk uzayında kümeleme yapmanın algısal (Perceptual) avantajı nedir?"*

### Gün 239: TV+ Sahne Arama: Altyazı ve Kare Zamanı Eşleme
- **İş Alanı:** TV+ Akıllı Video İndeksleme & Kullanıcı Arama Deneyimi
- **Veri Kaynağı:** Dizi Video Parçaları ve Senkronize SRT Altyazı Dosyaları
- **Model:** TF-IDF Altyazı Arama + Sahne Değişim Dedektörü (Kare Histogramı)
- **Türkçe Değişkenler:** `arama_metni, altyazi_satirlari, sahne_baslangic_saniyesi, sahne_bitis_saniyesi, eslesen_video_kare_zamani`
- **Kapsam:** Kullanıcı TV+'ta 'İstanbul Boğazı'nda ayrılık sahnesi' veya belirli bir repliği aradığında, altyazı ve sahne geçişlerini birleştirerek doğrudan ilgili sahnenin başladığı saniyeyi ekrana getirir.
- **Jupyter Notebook (`gun_239_tvplus_sahne_ve_altyazi_arama.ipynb`):**
  1. SRT altyazı dosyasındaki zaman damgalarının saniye bazında indekslenmesi
  2. Metin benzerliği ile aranan repliğin geçtiği anın yakalanması
  3. Sahne değişim noktalarıyla (Shot Boundary) senkronize edilerek tam sahne başlangıcına atlanması
- **Mülakat Sorusu:** *"Uzun videolarda sahne geçişi (Shot Transition) tespiti yaparken ardışık kareler arası piksel farkı yerine renk histogramı farkı kullanmak neden kameranın küçük hareketlerine karşı daha dayanıklıdır?"*

### Gün 240: Mağaza Vitrini Müşteri İlgi ve Yoğunluk Takibi
- **İş Alanı:** Turkcell Mağazacılık & Akıllı Perakende Analitiği
- **Veri Kaynağı:** Mağaza Vitrini Güvenlik Kamerası Örnek Video Kayıtları
- **Model:** HOG (Histogram of Oriented Gradients) + Lineer SVM İnsan Algılayıcı
- **Türkçe Değişkenler:** `kamera_karesi, tespit_edilen_insan_sayisi, vitrin_onunde_kalma_saniyesi, ilgi_seviyesi_yuksek_mi, gunluk_ziyaretci`
- **Kapsam:** Mağaza vitrinindeki yeni telefon veya kampanya afişinin önünden geçen insanların vitrin önünde kaç saniye durakladığını yerel kamerayla anonim olarak ölçer.
- **Jupyter Notebook (`gun_240_magaza_vitrini_musteri_ilgi_takibi.ipynb`):**
  1. HOG öznitelik çıkarıcı ve SVM ile vitrin önünden geçen yayaların tespiti
  2. Basit centroid tracking (ağırlık merkezi takibi) ile her kişinin vitrin önündeki kalış süresinin ölçümü
  3. Duraklama süresi 5 saniyeyi aşan müşterilerin 'İlgili Müşteri' olarak etiketlenmesi ve ısı haritası
- **Mülakat Sorusu:** *"Görüntü işlemede HOG (Histogram of Oriented Gradients) özniteliklerinin nesnenin renginden bağımsız olarak kenar ve şekil geometrisini yakalamadaki gücü nereden gelir?"*

### Gün 241: Fatura Tablosu Satır ve Sütun Çıkarıcı
- **İş Alanı:** Kurumsal Dijitalleşme & Otomatik Muhasebe Entegrasyonu
- **Veri Kaynağı:** Taranmış Kurumsal Telekom Fatura Görselleri (PNG/JPG)
- **Model:** OpenCV Morfolojik Filtreler (Yatay/Dikey Çizgi Ayrıştırma) + OCR
- **Türkçe Değişkenler:** `fatura_gorseli, dikey_cizgiler, yatay_cizgiler, tespit_edilen_hucre_sayisi, cikarilan_fatura_tablosu_df`
- **Kapsam:** Fotoğrafı çekilmiş faturalardaki tablo ızgarasını morfolojik filtrelerle bularak hücreleri tek tek böler; internet, SMS ve konuşma tutarlarını doğrudan Pandas DataFrame'ine dönüştürür.
- **Jupyter Notebook (`gun_241_fatura_tablo_morfolojik_ayristirma.ipynb`):**
  1. Görüntüye yatay ve dikey morfolojik çekirdekler (Kernels) uygulanarak tablo çizgilerinin çıkarılması
  2. Çizgilerin kesişim noktalarından tablo hücrelerinin (Cells) koordinatlarının bulunması
  3. Hücre içindeki metinlerin sırayla okunarak Excel/CSV uyumlu tablo formatına getirilmesi
- **Mülakat Sorusu:** *"Morfolojik görüntü işlemede 'Erosion' (Aşındırma) ve 'Dilation' (Genişletme) operasyonlarının ardışık uygulanmasıyla (Opening/Closing) tablodaki gürültü çizgileri nasıl elenir?"*

### Gün 242: Kule Pas ve Cıvata Deformasyon Otomatik Not Üreticisi
- **İş Alanı:** Kule Muayene Mühendisliği & Altyapı Bakım Raporlama
- **Veri Kaynağı:** Baz İstasyonu Demir Konstrüksiyon ve Cıvata Yakın Çekim Fotoğrafları
- **Model:** Renk Eşikleme (HSV Pas Maskesi) + Kontur Alanı Oranı + Kuralcı Raporlama
- **Türkçe Değişkenler:** `civata_fotografi, pas_alani_piksel, toplam_civata_alani, pas_yuzdesi, uretilen_teknik_rapor_paragrafi`
- **Kapsam:** Fotoğraftaki kule bağlantı cıvatalarında pas rengini (turuncu/kahverengi tonları) HSV uzayında izole eder; pas oranı %30'u aştığında teknisyen adına otomatik teknik rapor paragrafı yazar.
- **Jupyter Notebook (`gun_242_kule_pas_civata_raporlayici.ipynb`):**
  1. Görüntünün HSV renk uzayına çevrilerek pas renk aralığının maskelenmesi
  2. Kontur analiziyle cıvata üzerindeki korozyon alanının yüzdeye vurulması
  3. '3 numaralı sektör flanşında %38 paslanma tespit edildi, acil galvaniz yenileme gerekir' metni üretimi
- **Mülakat Sorusu:** *"Metal yüzeylerdeki pası tespit ederken gölge ve ışık parlamalarını elemek için HSV uzayında V (Value) kanalı yerine H (Hue) ve S (Saturation) kanallarına odaklanmanın faydası nedir?"*

### Gün 243: Çift Kanallı Çağrı Analitiği (Ses Enerjisi + Metin Füzyonu)
- **İş Alanı:** 532 Kalite Güvence & Müşteri Temsilcisi Değerlendirme
- **Veri Kaynağı:** Çağrı Merkezi 2 Kanallı WAV Ses Kayıtları & Metin Dökümleri
- **Model:** Librosa RMS Ses Enerjisi + TF-IDF Negatif Kelime Skoru Füzyonu
- **Türkçe Değişkenler:** `ses_dosyasi_wav, transkript_metni, ses_enerjisi_z_skor, metin_ofke_puani, birlestirilmis_gerginlik_skoru`
- **Kapsam:** Müşterinin ses kaydındaki bağırma/ses yükselme anları (RMS enerjisi) ile söylediği olumsuz kelimeleri zaman ekseninde senkronize ederek gerçek öfke anlarını hatasız yakalar.
- **Jupyter Notebook (`gun_243_ses_ve_metin_cift_kanalli_cagri_analizi.ipynb`):**
  1. Librosa ile ses dalgasının RMS (Kök Ortalama Kare) enerji profilinin saniye saniye çıkarılması
  2. Metin transkriptindeki şikayet kelimelerinin zaman damgalarıyla eşleştirilmesi
  3. Hem sesin yükseldiği hem olumsuz kelimenin söylendiği anların çağrının en kritik anı olarak işaretlenmesi
- **Mülakat Sorusu:** *"Çok modlu (Multimodal) veri füzyonunda Erken Füzyon (Early Fusion) ile Geç Füzyon (Late Fusion) mimarileri arasındaki fark nedir?"*

### Gün 244: TV+ Video Özet Klip Üreticisi
- **İş Alanı:** TV+ Sosyal Medya İçerik Yönetimi & Tanıtım Videoları
- **Veri Kaynağı:** Dizi ve Film Bölümlerinden Alınmış Örnek Kısa Video Dosyaları
- **Model:** Kareler Arası Renk Histogramı Farkı + Hareket Vektörü Büyüklüğü
- **Türkçe Değişkenler:** `video_dosyasi, kare_gecis_farklari, en_dinamik_10_sahne, birlestirilmis_fragman_video, islem_suresi_sn`
- **Kapsam:** 2 saatlik bir filmin karelerini tarayarak hareketin en yoğun ve sahnelerin en hızlı değiştiği kısımları seçer; sosyal medyada paylaşılacak 30 saniyelik dinamik tanıtım klibi üretir.
- **Jupyter Notebook (`gun_244_tvplus_otomatik_fragman_ozetleme.ipynb`):**
  1. Video akışından saniyede 2 kare örneklenerek (Frame Sampling) ardışık histogram farklarının hesabı
  2. Hareket skoru en yüksek olan tepe noktalarının (Peaks) bulunması
  3. Seçilen en iyi 5 sahne kesitinin OpenCV ile arka arkaya birleştirilerek kaydedilmesi
- **Mülakat Sorusu:** *"Video özetleme algoritmalarında sadece durağan kareleri (Keyframe) seçmek ile dinamik video segmentleri seçmek arasındaki hesaplama karmaşıklığı farkı nedir?"*

### Gün 245: BiP İçin Dinamik Çıkartma (Sticker) Üreteci
- **İş Alanı:** BiP Kullanıcı Etkileşimi & Kişiselleştirilmiş İletişim
- **Veri Kaynağı:** Turkcell Maskot Şablonları & Tebrik/Kutlama Metin Verisi
- **Model:** PIL (Pillow) Vektör Çizim Motoru + Dinamik Tipografi & Kenarlık
- **Türkçe Değişkenler:** `kullanici_yazisi, secilen_maskot_png, yazi_font_boyutu, otomatik_kenar_rengi, uretilen_sticker_webp`
- **Kapsam:** Kullanıcının yazdığı 'Hayırlı Cumalar', 'Doğum Günün Kutlu Olsun' gibi metinleri Turkcell maskot görseli üzerine otomatik metin sarma (Word Wrap) ve çıkartma kenarlığı ile basıp BiP formatında WebP üretir.
- **Jupyter Notebook (`gun_245_bip_otomatik_sticker_tipografi.ipynb`):**
  1. Metin uzunluğuna göre font boyutunun dinamik ölçeklenmesi ve satırlara bölünmesi
  2. Yazının okunabilirliğini artıran dış çizgi (Stroke / Outline) ve gölge efekti eklenmesi
  3. Şeffaf arka planlı WebP formatında çıkartmanın oluşturulup belleğe yazılması
- **Mülakat Sorusu:** *"Dinamik görsel üretiminde piksel tabanlı raster grafikler (PNG) yerine vektörel yaklaşımların ve font render motorlarının bellek performansı nasıl optimize edilir?"*

### Gün 246: Şebeke Topoloji Şeması Görsel Bağlantı Çıkarıcı
- **İş Alanı:** Ağ Dokümantasyonu & Saha Şeması Dijitalleştirme
- **Veri Kaynağı:** Kağıda Çizilmiş veya Visio ile Yapılmış Ağ Şeması Görselleri
- **Model:** Hough Çizgi Dönüşümü + Kontur Sınırlayıcı Kutu (Bounding Box) Eşleme
- **Türkçe Değişkenler:** `sema_gorseli, bulunan_cihaz_kutulari, bulunan_baglanti_cizgileri, baglanti_matrisi_router_switch`
- **Kapsam:** Fotoğrafı çekilmiş bir ağ bağlantı şemasındaki sunucu/router kutularını ve aralarındaki bağlantı çizgilerini bularak şemayı JSON formatında topoloji grafına çevirir.
- **Jupyter Notebook (`gun_246_sebeke_semasi_gorsel_okuyucu.ipynb`):**
  1. HoughLinesP ile şemadaki yatay ve dikey bağlantı kablolarının tespit edilmesi
  2. Dikdörtgen cihaz simgelerinin kontur alanlarıyla koordinatlarının bulunması
  3. Hangi çizginin hangi iki kutuya temas ettiğinin tespitiyle bağlantı listesinin çıkarılması
- **Mülakat Sorusu:** *"Hough Transform algoritmasında piksel uzayından parametre uzayına (Rho-Theta uzayı) geçişin kesintili çizgileri birleştirmedeki rolü nedir?"*

### Gün 247: Sesli ve Metin Tabanlı Çift Modlu Müşteri Arama Motoru
- **İş Alanı:** Arama Sistemleri & Müşteri Bilgi Bankası Erişimi
- **Veri Kaynağı:** Turkcell SSS Metinleri & Sesli Sorulmuş Kısa WAV Kayıtları
- **Model:** FastText Kelime Vektörleri + Ses MFCC Spektral Benzerliği
- **Türkçe Değişkenler:** `arama_sorgusu_ses, arama_sorgusu_metin, metin_vektor_skoru, ses_profil_skoru, ortak_siralanmis_sss_sonuclari`
- **Kapsam:** Abonenin hem sesli olarak sorduğu hem metin olarak yazdığı soruları ortak bir benzerlik uzayında puanlayarak en alakalı yardım makalesini milisaniyede döndürür.
- **Jupyter Notebook (`gun_247_cift_modlu_sss_arama_motoru.ipynb`):**
  1. Yazılı sorgunun FastText önceden eğitilmiş Türkçe vektörleriyle temsil edilmesi
  2. Sesli sorgudan çıkarılan MFCC katsayılarının ses benzerliği kütüphanesiyle eşleşmesi
  3. İki skorun ağırlıklı toplamıyla (Late Fusion) en doğru SSS maddesinin sıralanması
- **Mülakat Sorusu:** *"Ses öznitelikleri (MFCC) ile metin özniteliklerini (Kelime Embedding) aynı arama skorunda birleştirirken normalizasyon (Z-score / MinMax) neden zorunludur?"*

### Gün 248: lifebox Görsel Kategori ve Nesne Etiketleyici
- **İş Alanı:** lifebox Bulut Depolama & Akıllı Fotoğraf Yönetimi
- **Veri Kaynağı:** Kullanıcı Fotoğraf Koleksiyonları (Manzara, Yemek, Belge, Hayvan)
- **Model:** PyTorch MobileNetV3-Small (Yerel CPU, Model Boyutu 10MB)
- **Türkçe Değişkenler:** `kullanici_fotografi, tahmin_edilen_etiketler, guven_skorlari, otomatik_album_adi, islem_suresi_kare_ms`
- **Kapsam:** lifebox kullanıcılarının fotoğraflarını buluta göndermeden kendi telefonlarında/bilgisayarlarında 10 milisaniyede sınıflandırarak 'Yemekler', 'Plaj', 'Kediler' albümlerine otomatik dağıtır.
- **Jupyter Notebook (`gun_248_lifebox_hafif_gorsel_etiketleyici.ipynb`):**
  1. Fotoğrafın küçültülüp normalize edilerek MobileNetV3-Small modeline beslenmesi
  2. İlk 3 tahmin sınıfının Türkçe etiket karşılıklarıyla (Örn: Deniz Kıyısı) eşlenmesi
  3. Fotoğraf EXIF tarih bilgisiyle birleştirilerek akıllı albüm dizininin oluşturulması
- **Mülakat Sorusu:** *"Kullanıcı gizliliği gereği fotoğrafları sunucuya göndermeden cihaz üzerinde (On-Device) etiketlemenin bant genişliği ve KVKK açısından kritik avantajı nedir?"*

### Gün 249: Güvenlik Kamerası Hareket ve Bölge İhlali Günlükleyicisi
- **İş Alanı:** Veri Merkezi Güvenliği & Kritik Tesis İzleme
- **Veri Kaynağı:** Veri Merkezi Koridor Güvenlik Kamerası Video Kayıtları
- **Model:** Arka Plan Çıkarma (MOG2) + Poligon Alan İhlali + Zaman Günlüğü
- **Türkçe Değişkenler:** `video_karesi, guvenli_olmayan_poligon_alani, hareketli_nesne_konturu, ihlal_baslangic_saati, log_kaydi`
- **Kapsam:** Kamera görüntüsünde çizilen kırmızı güvenlik poligonuna (Örn: Sunucu kabinlerinin arkası) bir hareket girdiğinde olayın başlangıç ve bitiş saniyesini güvenlik raporu olarak diske yazar.
- **Jupyter Notebook (`gun_249_guvenlik_alani_ihlal_gunlukleyici.ipynb`):**
  1. OpenCV BackgroundSubtractorMOG2 ile hareketli piksellerin maskelenmesi
  2. Hareket merkezinin tanımlanan poligonun (PointPolygonTest) içinde olup olmadığının kontrolü
  3. İhlal anının ekran görüntüsünün kaydedilip metin günlüğüne (Audit Log) yazılması
- **Mülakat Sorusu:** *"Arka plan çıkarma algoritmalarında (MOG2) aydınlatma değişimlerine (ışık açılması/gölge) karşı öğrenme katsayısı (Learning Rate) nasıl kalibre edilir?"*

### Gün 250: Multimodal Saha Destek Terminali (Modül Capstone)
- **İş Alanı:** Entegre Saha Otomasyonu & Mobil Teknisyen Terminali
- **Veri Kaynağı:** Saha Fotoğrafları ve Teknisyen Sesli Soruları Sentetik Verisi
- **Model:** Görsel Kontur Renk Analizi + Ses Anahtar Kelime + Kural Füzyonu
- **Türkçe Değişkenler:** `terminal_giris_resim, terminal_giris_ses, tespit_edilen_kablo_rengi, sesli_komut_onay, nihai_onay_durumu`
- **Kapsam:** Teknisyenin çektiği fiber kablo demeti fotoğrafındaki doğru rengi (Mavi/Turuncu) doğrulayan ve teknisyenin sesli 'Ek yapıldı' teyidini dinleyerek iş emrini kapatan entegre terminaldir.
- **Jupyter Notebook (`gun_250_multimodal_saha_destek_terminali.ipynb`):**
  1. Fotoğraftaki kablo uçlarının renk histogramı ile doğru porta takıldığının denetimi
  2. Mikrofon sesinden teknisyenin onay kelimesinin eşik tabanlı algılanması
  3. Her iki modülden gelen doğrulamanın birleşmesiyle merkezi sisteme başarı raporu iletimi
- **Mülakat Sorusu:** *"Saha şartlarında gürültülü ses ve kötü aydınlatmalı görseller birleştirilirken modüllerden birinin başarısız olması durumunda Fallback (Geri çekilme) stratejisi nasıl kurulur?"*


## Modül 20: Edge AI, TinyML & Modem/CPE Üzeri Gömülü Yapay Zeka

---

### Gün 251: NSL-KDD Açık Verisiyle Ağ Saldırı Tespiti İçin ANSI C Koduna Derlenen Karar Ağacı
- **İş Alanı:** Modem/Router Gömülü Siber Güvenlik & Ağ Geçidi Tehdit Engelleme
- **Veri Kaynağı:** Kaggle / UNB - NSL-KDD Network Intrusion Detection Open Dataset
- **Model:** Scikit-Learn Karar Ağacı -> m2cgen ile Saf ANSI C Kodu Derleme
- **Türkçe Değişkenler:** `protokol_turu, servis_adi_http_ftp, paket_hata_orani, uretilen_c_fonksiyonu, saldiri_turu_dos_probe`
- **Kapsam:** Dünyanın en bilinen ağ saldırı veri seti NSL-KDD üzerinde eğitilen modeli hiçbir harici bağımlılığı olmayan saf ANSI C koduna derler; modem donanımında mikrosaniyede saldırı yakalar.
- **Jupyter Notebook (`gun_251_nsl_kdd_saldiri_tespiti_c_kodu.ipynb`):**
  1. NSL-KDD veri setindeki 41 ağ akış özniteliğiyle Scikit-Learn üzerinde Karar Ağacı eğitimi
  2. Modelin m2cgen aracıyla saf ANSI C fonksiyonuna (`predict()`) dönüştürülmesi
  3. Üretilen C kodunun GCC ile derlenerek modem işlemcisinde sıfır RAM sızıntısıyla test edilmesi
- **Mülakat Sorusu:** *"Makine öğrenimi modelleri m2cgen gibi araçlarla saf C fonksiyonlarına dönüştürüldüğünde derleyici optimizasyonları (GCC -O3) çalışma zamanını nasıl hızlandırır?"*

### Gün 252: Aşırı Düşük Bellekli INT8 Sayısal Kuantalama (Quantization)
- **İş Alanı:** Edge AI & Uç Cihazlarda Model Küçültme
- **Veri Kaynağı:** Modem Trafik Tipi Sınıflandırma Sentetik Verisi (VoIP, Web, Oyun, Video)
- **Model:** 3 Katmanlı PyTorch MLP -> Post-Training Quantization (FP32 -> INT8)
- **Türkçe Değişkenler:** `fp32_model_boyutu_kb, int8_model_boyutu_kb, dogruluk_kaybi_yuzde, bellek_tasarrufu_kat, cikarim_hizi_x`
- **Kapsam:** 32-bit kayan noktalı (FP32) yapay sinir ağı ağırlıklarını 8-bit tam sayıya (INT8) dönüştürerek model boyutunu %75 küçültür ve bellek kısıtlı IoT çiplerine sığdırır.
- **Jupyter Notebook (`gun_252_int8_sayisal_kuantalama_edge.ipynb`):**
  1. PyTorch üzerinde FP32 temel sınıflandırma modelinin eğitilmesi
  2. Ölçekleme faktörü (Scale) ve sıfır noktası (Zero-point) hesaplanarak INT8 kuantalama yapılması
  3. FP32 ve INT8 modellerinin doğruluk ve bellek ayak izlerinin kıyaslanması
- **Mülakat Sorusu:** *"Post-Training Quantization (PTQ) ile Quantization-Aware Training (QAT) arasındaki doğruluk ve eğitim süresi farkı nedir?"*

### Gün 253: IoT Akıllı Su/Elektrik Sayacı Anomali Tespiti
- **İş Alanı:** NB-IoT (Narrowband IoT) & Akıllı Şebeke Sensörleri
- **Veri Kaynağı:** Pille Çalışan Akıllı Sayaç Saatlik Tüketim Akışı
- **Model:** Half-Space Trees (Akış Tabanlı Çevrim İçi Anomali Tespiti - River)
- **Türkçe Değişkenler:** `saatlik_tuketim_m3, pil_voltaji, anomali_skoru, kacak_su_uyarisi, pil_omru_yil`
- **Kapsam:** 10 yıl pil ömrü hedeflenen NB-IoT sayaçlarında, geçmiş veriyi saklamadan her yeni tüketim verisiyle kendini güncelleyen ultra hafif akış tabanlı su kaçağı dedektörü kodlar.
- **Jupyter Notebook (`gun_253_nb_iot_sayac_akilli_anomali.ipynb`):**
  1. Belleksiz akış (Streaming) veri yapısının ve saatlik tüketim simülasyonunun kurulması
  2. River kütüphanesi Half-Space Trees modeliyle anlık anomali skorunun çıkarılması
  3. Ani gece tüketimi patlamalarında boru patlağı uyarısının üretilmesi
- **Mülakat Sorusu:** *"Kaynak kısıtlı IoT çiplerinde toplu (Batch) öğrenme yerine tek geçişli akış (One-pass Online Learning) algoritmaları neden zorunludur?"*

### Gün 254: Tarım İHA'ları İçin Ağaç Sayımı ve Sağlık Analitiği
- **İş Alanı:** Turkcell Dijital Tarım & İHA Kenar Bilişimi
- **Veri Kaynağı:** Zeytin ve Fındık Bahçesi Sentetik Hava Fotoğrafları
- **Model:** Basit Blob Algılama (Laplacian of Gaussian - LoG) + ExG Bitki İndeksi
- **Türkçe Değişkenler:** `iha_fotografi, bulunan_agac_sayisi, exg_yesillik_indeksi, sararan_agac_sayisi, tarla_verim_tahmini`
- **Kapsam:** Drone uçarken Raspberry Pi / Coral Edge TPU üzerinde doğrudan ağaç tepelerini sayar ve yapraklardaki sararmaları Aşırı Yeşil İndeksi (ExG) ile tespit edip çiftçiye harita çıkarır.
- **Jupyter Notebook (`gun_254_dijital_tarim_iha_agac_sayimi.ipynb`):**
  1. Bitki örtüsünü vurgulayan ExG (Excess Green = 2G - R - B) dönüşümünün uygulanması
  2. Laplacian of Gaussian (LoG) dairesel filtrelerle ağaç taçlarının sayılması
  3. Klorofil kaybı yaşayan kurak ağaçların koordinatlarının tespiti ve raporlanması
- **Mülakat Sorusu:** *"Hassas tarımda multispektral kameralar olmadan standart RGB görüntülerden bitki sağlığı indeksleri (ExG, VARI) nasıl türetilir?"*

### Gün 255: FCC Açık Genişbant Ölçüm Verisiyle Ev İnterneti Hız Testi ve Kalite (QoE) Tahmini
- **İş Alanı:** Sabit İnternet Servis Kalitesi (QoE) & Müşteri Modem Hız Denetimi
- **Veri Kaynağı:** FCC.gov / Kaggle - Measuring Broadband America Open Dataset
- **Model:** Random Forest Regresyon + İstatistiksel Hız Sınıflandırması
- **Türkçe Değişkenler:** `abone_hiz_profili_mbps, gerceklesen_indirme_hizi, yukleme_hizi, paket_kaybi_orani, qoe_memnuniyet_puani_1_5`
- **Kapsam:** ABD Federal İletişim Komisyonu'nun (FCC) yayınladığı milyonlarca gerçek modem ölçüm verisinden yararlanarak taahhüt edilen hız ile gerçekleşen hız arasındaki sapmayı tahmin eder.
- **Jupyter Notebook (`gun_255_fcc_genisbant_hiz_ve_qoe_analizi.ipynb`):**
  1. FCC açık veri tabanından modem bazlı indirme hızı, gecikme ve paket kaybı satırlarının filtrelenmesi
  2. Taahhüt edilen tarife hızına ulaşamayan hatları tahmin eden Random Forest modelinin kurulması
  3. Kullanıcı modem ekranına anlık hat kalitesi ve Wi-Fi iyileştirme tavsiyesinin basılması
- **Mülakat Sorusu:** *"Hız testlerinde TCP indirme hızının ilk 5 saniyesindeki yavaş başlangıç (Slow Start) etkisini arındırarak kararlı durum (Steady-State) bant genişliği nasıl ölçülür?"*

### Gün 256: Giyilebilir Cihazlar İçin Düşme Algılama (TinyML)
- **İş Alanı:** Turkcell Yaşam / Akıllı Saat & Yaşlı Güvenliği Teknolojileri
- **Veri Kaynağı:** 3 Eksenli İvmeölçer (IMU) Sentetik Düşme ve Yürüme Verisi
- **Model:** SVM RBF Çekirdeği -> 8 Sabit Katsayılı Doğrusal Sınıflandırıcı
- **Türkçe Değişkenler:** `ivme_x_y_z, toplam_ivme_vektoru_g, serbest_dusse_ani_var_mi, carpma_siddeti, acil_yardim_sms_at`
- **Kapsam:** Akıllı saat sensöründen gelen ivme değerlerinde önce serbest düşme (0G) ardından ani darbe tepe noktasını yakalayarak yaşlıların düştüğünü anında algılar ve yakınlarına SMS atar.
- **Jupyter Notebook (`gun_256_akilli_saat_dusme_algilama_tinyml.ipynb`):**
  1. 3 eksenli ivmenin büyüklük vektörünün (Signal Magnitude Area) hesaplanması
  2. Serbest düşme vadisi ve çarpma tepesi özniteliklerinin çıkarılması
  3. Basit eşik kuralları ve SVM ile oturma/yatma hareketlerinden düşmenin ayrıştırılması
- **Mülakat Sorusu:** *"Mikrodenetleyicilerde (Cortex-M0/M4) kayan noktalı sayı birimi (FPU) olmadığında sensör verisi 'Fixed-Point' aritmetiğiyle nasıl işlenir?"*

### Gün 257: Araç İçi Takip Cihazında Agresif Sürüş Tespiti
- **İş Alanı:** Kopilot Filo Yönetimi & Araç Telemetrisi Çözümleri
- **Veri Kaynağı:** OBD-II ve Dahili Jiroskop/İvmeölçer Saniyede 10Hz Veri Akışı
- **Model:** Kayan Pencere Standart Sapması + Dinamik Eşikleme
- **Türkçe Değişkenler:** `hiz_km_h, boyuna_ivme_g, yanal_ivme_g, ani_fren_sayisi, sert_viraj_sayisi, surucu_guvenlik_puani_100`
- **Kapsam:** Turkcell Kopilot OBD cihazı üzerinde hücresel veriyi harcamadan, şoförün sert fren, ani gazlama ve makas hareketlerini yerel sensörde puanlayıp filo yöneticisine özet gönderir.
- **Jupyter Notebook (`gun_257_kopilot_agresif_surus_tespiti.ipynb`):**
  1. Aracın boyuna (Longitudinal) ve yanal (Lateral) ivmelerinin filtrelenmesi
  2. 0.3G üzeri ani yavaşlamaların ve sert şerit değiştirmelerin yakalanması
  3. 100 üzerinden filo güvenli sürüş karnesinin hesaplanması ve yakıt israfı analitiği
- **Mülakat Sorusu:** *"Araç hareket halindeyken OBD cihazının montaj açısından kaynaklanan eğim açısı yerçekimi vektörü yardımıyla yazılımsal olarak nasıl sıfırlanır (Calibration)?"*

### Gün 258: Sensör Verisi İçin Hafif Sıkıştırma (Gorilla Algoritması)
- **İş Alanı:** IoT Büyük Veri İletimi & Pil ve Bant Genişliği Tasarrufu
- **Veri Kaynağı:** Baz İstasyonu Sıcaklık ve Nem Sensörü Yüksek Frekanslı Zaman Serisi
- **Model:** Facebook Gorilla Kayan Noktalı XOR Sıkıştırma Algoritması
- **Türkçe Değişkenler:** `ham_sicaklik_degeri, onceki_deger, xor_farki, bas_ve_kuyruk_sifirlari, sikistirilmis_bit_dizisi, sikistirma_orani`
- **Kapsam:** IoT sensörlerinin ürettiği 64-bit kayan noktalı ölçümleri ardışık XOR farkları üzerinden kayıpsız sıkıştırarak NB-IoT üzerinden gönderilen veri boyutunu %80 azaltır.
- **Jupyter Notebook (`gun_258_gorilla_iot_zaman_serisi_sikistirma.ipynb`):**
  1. Kayan noktalı IEEE-754 sayı formatının ikili (Binary) bit düzeyinde temsil edilmesi
  2. Ardışık iki ölçüm arasındaki XOR farkının ve ortak sıfırların kodlanması
  3. 1000 adet sensör ölçümünün orijinal ve sıkıştırılmış bayt boyutlarının kıyaslanması
- **Mülakat Sorusu:** *"Zaman serisi veritabanlarında (InfluxDB, Prometheus) kullanılan Gorilla sıkıştırma algoritmasının Delta-of-Delta zamanlama ve XOR kayan nokta prensipleri nelerdir?"*

### Gün 259: Akıllı Ev Gateway Üzerinde DNS Tünelleme Dedektörü
- **İş Alanı:** Ev İnterneti Siber Güvenliği & Siber Tehdit Engelleme
- **Veri Kaynağı:** Modem Üzerinden Geçen Anlık DNS İstekleri Günlüğü
- **Model:** Shannon Entropisi + Alan Adı Uzunluğu + Heuristik Karar Motoru
- **Türkçe Değişkenler:** `sorgulanan_alan_adi, alt_alan_adi_uzunlugu, karakter_entropi_skoru, txt_kayit_sikligi, veri_kacirma_suphesi`
- **Kapsam:** Evdeki virüslü bir cihazın gizlice veri sızdırmak için kullandığı karmaşık ve anlamsız uzunluktaki DNS sorgularını modem CPU'sunda milisaniyede yakalar ve engeller.
- **Jupyter Notebook (`gun_259_modem_dns_tunneling_dedektoru.ipynb`):**
  1. DNS sorgularındaki alt alan adlarının (Subdomain) uzunluk ve karakter dağılımının incelenmesi
  2. Normal alan adları (google.com) ile tünelleme (x89f2a.evil.com) entropisinin kıyası
  3. Şüpheli cihaz IP'sinin modem güvenlik duvarında (iptables) anında engellenmesi
- **Mülakat Sorusu:** *"Siber saldırganların güvenlik duvarlarını aşmak için standart DNS 53 numaralı portunu veri kaçırma (Exfiltration) kanalı olarak kullanma yöntemi nasıl çalışır?"*

### Gün 260: Kamera Sensöründe Yerel Optik Akış ile Araç Sayımı
- **İş Alanı:** Akıllı Şehirler, Trafik Yönetimi & Saha Uç Analitiği
- **Veri Kaynağı:** Kavşak Trafik Kamerası Örnek Düşük Çözünürlüklü Video Kareleri
- **Model:** Lucas-Kanade Seyrek Optik Akış (Sparse Optical Flow) + Sanal Çizgi Kesişimi
- **Türkçe Değişkenler:** `video_karesi, kose_noktalari, optik_akis_vektorleri_dx_dy, cizgiyi_gecen_arac_sayisi, akici_trafik_mi`
- **Kapsam:** Trafik ışığı direğindeki mini kamerada ağır derin öğrenme modellerine ihtiyaç duymadan köşe noktalarının hareket yönünü takip ederek şeritten geçen araçları sayar.
- **Jupyter Notebook (`gun_260_optik_akis_arac_sayimi.ipynb`):**
  1. Shi-Tomasi köşe bulucu ile yoldaki belirgin araç köşelerinin tespiti
  2. Lucas-Kanade optik akış algoritması ile noktaların bir sonraki karedeki konum takibi
  3. Tanımlanan sanal sayım çizgisini dikey yönde kesen araçların sayısının kaydedilmesi
- **Mülakat Sorusu:** *"Optik akış yönteminde 'Parlaklık Sabitliği' (Brightness Constancy) varsayımı nedir ve ani ışık değişimlerinde neden zorlanır?"*

### Gün 261: Endüstriyel Motor Titreşim Analizi (Edge FFT)
- **İş Alanı:** Kestirimci Bakım & Fabrika IoT Titreşim Sensörleri
- **Veri Kaynağı:** Rulman ve Motor Mili Yüksek Hızlı İvmeölçer Verisi (10 kHz)
- **Model:** Hızlı Fourier Dönüşümü (FFT) + Spektral Tepe Frekans Eşleme
- **Türkçe Değişkenler:** `zaman_sinyali_ivme, frekans_spektrumu_fft, rulman_hata_frekansi_bpfi, tepe_genlik_degeri, bakim_zamani_geldi_mi`
- **Kapsam:** Fabrikadaki soğutma pompalarının motor miline takılı sensörde titreşimi frekans spektrumuna çevirir; rulman bilyasındaki çatlağın karakteristik frekansını yakalayarak motor yanmadan uyarır.
- **Jupyter Notebook (`gun_261_motor_titresim_fft_kestirimci_bakim.ipynb`):**
  1. Zaman domeni ivme verisine Hanning pencere fonksiyonu uygulanması
  2. Numpy FFT ile sinyalin frekans spektrumuna (0 - 5000 Hz) dönüştürülmesi
  3. Motor dönüş hızının harmoniklerindeki beklenmedik genlik artışlarının alarm üretmesi
- **Mülakat Sorusu:** *"Titreşim analizinde Hızlı Fourier Dönüşümü (FFT) uygulanmadan önce spektral sızıntıyı (Spectral Leakage) önlemek için neden pencereleme (Windowing) yapılır?"*

### Gün 262: Gerçek Fotovoltaik Santral Verisiyle Güneş Enerjisi Üretim Tahmini
- **İş Alanı:** Yeşil Şebeke, Kırsal Enerji & Solar Hibrit Güç Sistemleri
- **Veri Kaynağı:** Kaggle - Solar Power Generation Data (2 Gerçek Santral Verisi)
- **Model:** XGBoost Regressor + Saatlik Trend Ayrıştırması
- **Türkçe Değişkenler:** `ortam_sicakligi_c, modul_sicakligi_c, gunes_isinimi_w_m2, anlik_ac_guc_kw, gunluk_toplam_verim_kwh`
- **Kapsam:** Kaggle'daki gerçek güneş enerjisi santrali telemetrisini kullanarak ortam sıcaklığı ve ışınımdan üretilecek DC/AC gücü saat saat tahmin eder; batarya şarj planı oluşturur.
- **Jupyter Notebook (`gun_262_gunes_enerjisi_uretim_tahmini_xgboost.ipynb`):**
  1. 68.000 satırlık gerçek santral sensör verilerindeki ışınım ve panel sıcaklığı korelasyonu
  2. XGBoost ile 15 dakikalık aralıklarla üretilecek elektrik enerjisinin (kW) tahmini
  3. Kırsal baz istasyonu bataryasının geceye yetip yetmeyeceğinin enerji bilançosu analitiği
- **Mülakat Sorusu:** *"Güneş panellerinde ortam sıcaklığı arttıkça panel veriminin düşmesi (Sıcaklık Katsayısı etkisi) makine öğrenimi özniteliklerinde lineer olmayan bir ilişki olarak nasıl modellenir?"*

### Gün 263: Modem Üzeri Otomatik Kanal Seçimi (MAB Epsilon-Greedy)
- **İş Alanı:** Ev İçi Wi-Fi Deneyimi & Girişim Yönetimi
- **Veri Kaynağı:** Apartmandaki Komşu Wi-Fi Sinyalleri ve Kanal Çakışma Metrikleri
- **Model:** Çok Kollu Haydut (Multi-Armed Bandit) Epsilon-Greedy Algoritması
- **Türkçe Değişkenler:** `kanal_havuzu_1_6_11, secilen_kanal, elde_edilen_hiz_mbps, ortalama_kanal_odulleri, en_iyi_kanal_secimi`
- **Kapsam:** Apartmanda herkes aynı Wi-Fi kanalını kullandığında, modemin belirli aralıklarla sessizce diğer kanalları deneyip en az gürültülü ve en hızlı kanala kalıcı olarak geçmesini sağlar.
- **Jupyter Notebook (`gun_263_wifi_kanal_secimi_mab_epsilon_greedy.ipynb`):**
  1. 13 adet 2.4 GHz kanalının potansiyel parazit ortamının simülasyonu
  2. %90 ihtimalle bilinen en iyi kanalı kullanırken %10 ihtimalle diğer kanalları keşfetme (Exploration vs Exploitation)
  3. Modemin 1 hafta içinde en temiz kanala kilitlenerek hız testini %40 artırdığının gösterilmesi
- **Mülakat Sorusu:** *"Çok Kollu Haydut (Multi-Armed Bandit) algoritmalarında Epsilon-Greedy ile UCB (Upper Confidence Bound) arasındaki keşif stratejisi farkı nedir?"*

### Gün 264: Yapay Zeka Modeli Karar Ağacından Gömülü C Koduna Dönüştürücü
- **İş Alanı:** Model Derleme Teknolojileri & Gömülü Sistem Mühendisliği
- **Veri Kaynağı:** Turkcell Müşteri Ayrılma (Churn) Eğitim Veri Kümesi
- **Model:** Özel Python AST Üretici (Scikit-Learn Tree -> Saf `if-else` C Fonksiyonu)
- **Türkçe Değişkenler:** `agac_yapisi, sol_cocuk, sag_cocuk, esik_degerleri, uretilen_c_kodu_metni, gcc_derleme_durumu`
- **Kapsam:** Scikit-learn ile eğitilen bir karar ağacını ayrıştırarak hiçbir harici kütüphane gerektirmeyen, doğrudan donanım mikroçiplerine gömülebilen saf iç içe `if-else` C kodunu sıfırdan üretir.
- **Jupyter Notebook (`gun_264_karar_agaci_c_koduna_derleyici.ipynb`):**
  1. Scikit-learn Karar Ağacının `tree_` iç yapısındaki düğüm, özellik ve eşik değerlerinin okunması
  2. Özyinelemeli (Recursive) fonksiyon ile C sözdiziminde `float predict(float* features)` fonksiyonu yazımı
  3. Üretilen C kodunun Python çıktısıyla birebir aynı tahminleri ürettiğinin birim testi
- **Mülakat Sorusu:** *"Makine öğrenimi modellerini C koduna çevirirken (Transpilation) derin ağaçların yol açtığı 'Instruction Cache Miss' problemi nasıl optimize edilir?"*

### Gün 265: Akıllı Uç Cihaz (Edge AI) Ağ Geçidi Mimarisi (Modül Capstone)
- **İş Alanı:** Bütünleşik Edge Donanım Mimarisi & Uçtan Uca IoT Otomasyonu
- **Veri Kaynağı:** Sensör Verisi + Wi-Fi Telemetrisi + Ağ Trafiği Ortak Akışı
- **Model:** Boru Hattı Mimarisi (Gorilla Sıkıştırma + C-Kodu Tahmini + INT8 Filtreleme)
- **Türkçe Değişkenler:** `ham_veri_paketi, sikistirilmis_boyut, yerel_c_tahmin_sonucu, buluta_aktarilsin_mi, toplam_gecikme_mikrosaniye`
- **Kapsam:** Modem veya IoT Gateway üzerinde çalışan, veriyi yerel C koduyla sınıflandıran, sadece anomali olduğunda Gorilla ile sıkıştırıp Turkcell bulutuna yollayan tam teşekküllü uç yazılımdır.
- **Jupyter Notebook (`gun_265_edge_ai_ag_gecidi_mimarisi.ipynb`):**
  1. Sensör verisinin bellek üzerinde anlık okunması ve yerel C tahmin fonksiyonundan geçirilmesi
  2. Normal durumlarda verinin yerelde tutulması, kritik durumlarda sıkıştırılarak kuyruğa alınması
  3. Uç cihazın bant genişliği tüketimini %92 oranında azalttığının simülasyon testi
- **Mülakat Sorusu:** *"Uç bilişimde (Edge Computing) 'Cloud-First' mimariden 'Local-First with Cloud Backup' mimarisine geçişin güvenlik, bant genişliği ve operasyonel maliyet kazanımları nelerdir?"*


## Modül 21: Büyük Ölçekli Grafik Sinir Ağları & NetworkX Topoloji Analizi

---

### Gün 266: TeleGeography Açık Verisiyle Küresel Denizaltı Fiber Kablo Ağında Kritik Düğüm Analizi
- **İş Alanı:** Uluslararası Transmisyon, Denizaltı Kablo Hatları & İnternet Dayanıklılığı
- **Veri Kaynağı:** TeleGeography Submarine Cable Map Public Open Dataset (GitHub / Kaggle)
- **Model:** NetworkX Arasındalık Merkeziliği (Betweenness Centrality) + Köprü (Bridge) Analizi
- **Türkçe Değişkenler:** `kablo_adi, inis_istasyonu_sehri, bagli_ulke_sayisi, arasindalik_skoru, kablo_koparsa_izole_olacak_ulkeler`
- **Kapsam:** TeleGeography'nin açık GitHub deposundaki gerçek denizaltı fiber kablo koordinatlarını NetworkX grafına dönüştürerek Süveyş Kanalı veya Cebelitarık gibi tek arıza noktalarını bulur.
- **Jupyter Notebook (`gun_266_telegeography_denizalti_kablo_kritik_dugum.ipynb`):**
  1. TeleGeography GeoJSON açık verisinden dünya denizaltı fiber kablo hatlarının ve karaya çıkış noktalarının graf yapısına aktarılması
  2. Brandes algoritması ile küresel veri akışını taşıyan en kritik transit iniş istasyonlarının bulunması
  3. Kızıldeniz veya Akdeniz'deki ana kablolar koptuğunda Türkiye'nin internet trafiğinin yedek rotalarının tespiti
- **Mülakat Sorusu:** *"Küresel internet omurgasında denizaltı kablo kopmalarında (Örn: Kızıldeniz çapa kazaları) trafiğin alternatif okyanus rotalarına sapması graf teorisinde dinamik kenar silme ile nasıl test edilir?"*

### Gün 267: IP / MPLS Ağlarında K-En Kısa Yedekli Yol Algoritması
- **İş Alanı:** Çekirdek Şebeke Yönlendirme (Routing) & Hızlı Yeniden Rotalama (FRR)
- **Veri Kaynağı:** MPLS Router Gecikme ve Kapasite Ağırlıklı Topoloji Verisi
- **Model:** Yen's K-Shortest Path Algoritması + Kesişmeyen Hat Filtresi
- **Türkçe Değişkenler:** `kaynak_router, hedef_router, en_kisa_1_yol, yedek_2_yol, alternatif_3_yol, toplam_gecikme_ms`
- **Kapsam:** İstanbul-Ankara arası veri akışında ana fiber hattı koptuğunda sıfır kesintiyle devreye girecek, ana hatla hiçbir ortak santral veya kablo kanalı paylaşmayan 3 bağımsız yedek rota çıkarır.
- **Jupyter Notebook (`gun_267_mpls_yen_k_en_kisa_rota.ipynb`):**
  1. Ağ gecikmeleri ağırlık kabul edilerek Dijkstra ile ilk en kısa yolun bulunması
  2. Yen algoritması ile kenarları sırayla devre dışı bırakarak en iyi alternatif 3 yolun türetilmesi
  3. Ana yol ve yedek yolların düğüm kesişimsizliğinin (Node-Disjoint Path) doğrulanması
- **Mülakat Sorusu:** *"Telekomünikasyonda Düğüm Bağımsız (Node-Disjoint) yollar ile Kenar Bağımsız (Edge-Disjoint) yollar arasındaki fark güvenilirlik açısından neden hayatidir?"*

### Gün 268: Çağrı Trafik Ağından Çıkarımla Topluluk Tespiti (Louvain)
- **İş Alanı:** Büyük Veri Analitiği & Sosyal Ağ Tabanlı Müşteri Kümeleri
- **Veri Kaynağı:** Anonimleştirilmiş CDR Arama ve Mesajlaşma Yönlü Grafı (10.000 Düğüm)
- **Model:** Louvain Modülerlik Maksimizasyonu Algoritması (NetworkX)
- **Türkçe Değişkenler:** `abone_a, abone_b, gorusme_sayisi_agirlik, atanan_topluluk_id, ag_modulerlik_skoru, aile_arkadas_kumesi`
- **Kapsam:** Abonelerin birbirleriyle yaptıkları arama sıklıklarını analiz ederek doğal aile, arkadaş veya iş çevrelerini (Toplulukları) bularak grup içi indirimli konuşma paketleri önerir.
- **Jupyter Notebook (`gun_268_cagri_grafindan_louvain_topluluk_tespiti.ipynb`):**
  1. Telefon aramalarından ağırlıklı yönlü grafın ve komşuluk matrisinin oluşturulması
  2. Louvain algoritmasıyla ağın modülerliğini maksimize eden alt toplulukların bulunması
  3. Topluluk içi iletişim yoğunluğu ile topluluklar arası köprü kişilerin tespiti
- **Mülakat Sorusu:** *"Graf analizinde 'Modülerlik' (Modularity) metriği nedir ve Louvain algoritmasının hiyerarşik kümeleme mantığı nasıl çalışır?"*

### Gün 269: Baz İstasyonları Arası Handover Trafiği Spektral Kümeleme
- **İş Alanı:** RAN Mühendisliği & Konum Güncelleme Alanı (LAC/TAC) Optimizasyonu
- **Veri Kaynağı:** Baz İstasyonları Arasındaki Başarılı/Başarısız Geçiş (Handover) Sayıları
- **Model:** Graf Laplasyen Matrisi + Spektral Kümeleme (Spectral Clustering)
- **Türkçe Değişkenler:** `kaynak_hucre, hedef_hucre, handover_sayisi, laplasyen_ozvektorleri, atanan_tac_bolgesi`
- **Kapsam:** Arabayla giderken telefonun baz istasyonları arasında geçiş yapma sıklıklarını inceleyerek, gereksiz sinyalleşmeyi azaltacak en ideal Takip Alanı (Tracking Area Code - TAC) sınırlarını çizer.
- **Jupyter Notebook (`gun_269_handover_grafindan_spektral_kumeleme.ipynb`):**
  1. Hücreler arası handover akışlarından simetrik benzerlik matrisinin kurulması
  2. Normalize Edilmiş Graf Laplasyen (Normalized Laplacian) matrisinin özdeğer ayrışımı
  3. En küçük özvektörler üzerinde K-Means ile baz istasyonlarının coğrafi kümelerinin bulunması
- **Mülakat Sorusu:** *"Graf Laplasyen matrisinin ikinci en küçük özdeğerine (Fiedler Değeri) neden 'Cebirsel Bağlantılılık' denir ve ağ bölümlendirmede ne ifade eder?"*

### Gün 270: Graf Sinir Ağı (GCN) ile Baz İstasyonu Trafik Tahmini
- **İş Alanı:** Gelecek Nesil Yapay Zeka & Uzay-Zaman Şebeke Tahmini
- **Veri Kaynağı:** 100 Hücrenin Coğrafi Mesafe Grafı ve Saatlik GB Veri Trafiği
- **Model:** Graf Evrişimli Ağ (Graph Convolutional Network - GCN) Matematiksel Çarpımı
- **Türkçe Değişkenler:** `komsuluk_matrisi_a, ozellik_matrisi_x, agirlik_matrisi_w, duzeltilmis_laplasyen, gelecek_saat_tahmini`
- **Kapsam:** Bir baz istasyonundaki trafik artışının komşu istasyonlara nasıl yayıldığını komşuluk matrisi ve öznitelik çarpımıyla (A_hat * X * W) modelleyerek gelecek saat yükünü tahmin eder.
- **Jupyter Notebook (`gun_270_gcn_baz_istasyon_trafik_tahmini.ipynb`):**
  1. Baz istasyonları arası mesafelerden eşiklenmiş komşuluk matrisinin oluşturulması
  2. Kipf & Welling GCN formülünün (D^-0.5 * A_tilde * D^-0.5 * X * W) saf Numpy ile kodlanması
  3. Sadece tek bir hücrenin geçmişine bakan modellere göre mekânsal komşu bilgisinin sağladığı %25 hata iyileşmesi
- **Mülakat Sorusu:** *"GCN (Graph Convolutional Networks) katmanlarında kendi kendine döngü ekleme (Self-loop) ve derece matrisi normalizasyonu yapılmazsa sayısal patlama (Exploding Gradient) neden oluşur?"*

### Gün 271: Dolandırıcı SIM Halka Tespiti (Çevrim ve Klike Analizi)
- **İş Alanı:** Gelir Güvencesi (Revenue Assurance) & Dolandırıcılık İstihbaratı
- **Veri Kaynağı:** Aynı Gün Açılan SIM Kartların Arama ve SMS İletişim Ağı
- **Model:** Yönlü Çevrim Bulucu (Tarjan SCC) + Bron-Kerbosch Maksimal Klike
- **Türkçe Değişkenler:** `hat_numarasi, aranan_hat, tespit_edilen_kapali_halka, klike_eleman_sayisi, organize_dolandiricilik_mi`
- **Kapsam:** Korsan çağrı merkezlerinin veya dolandırıcıların birbirini arayarak şebekeyi meşgul ettiği veya puan topladığı kapalı iletişim halkalarını (Cycle) ve tam bağlı çeteleri (Clique) yakalar.
- **Jupyter Notebook (`gun_271_sim_dolandirici_halka_klike_analizi.ipynb`):**
  1. İletişim akışında yönlü grafların oluşturulması ve Tarjan algoritmasıyla güçlü bağlı bileşenlerin tespiti
  2. Bron-Kerbosch algoritması ile 5 veya daha fazla kişinin birbirini karşılıklı aradığı tam klikelerin bulunması
  3. Organize dolandırıcılık şüphesiyle SIM kartların tek tıkla toplu incelemeye sevk edilmesi
- **Mülakat Sorusu:** *"Graf teorisinde Maksimal Klike (Maximal Clique) bulma probleminin NP-Complete olmasının sebebi nedir ve telekom gibi büyük graflarda nasıl sınırlandırılır?"*

### Gün 272: Dinamik Ağlarda Bilgi Yayılımı (Epidemik Model Simülasyonu)
- **İş Alanı:** Pazarlama Analitiği, Viral Yayılım & Ağ Virüs Bulaşması
- **Veri Kaynağı:** Müşterilerin 'Arkadaşını Getir' Kampanyası Davet İletişim Grafı
- **Model:** SIR (Susceptible-Infectious-Recovered) Graf Bulaşma Simülasyonu
- **Türkçe Değişkenler:** `dugum_sayisi, bulasma_olasiligi_beta, iyilesme_orani_gamma, zaman_adimi_gun, toplam_kampanyaya_katilan`
- **Kapsam:** Yeni bir Turkcell dijital servisinin veya kampanyasının abonelerin arkadaş çevreleri üzerinden kulaktan kulağa nasıl bir hızla yayılacağını fiziksel salgın modeliyle simüle eder.
- **Jupyter Notebook (`gun_272_sir_epidemik_kampanya_yayilimi.ipynb`):**
  1. NetworkX üzerinde Barabási-Albert ölveksiz (Scale-Free) ağ yapısının kurulması
  2. SIR diferansiyel denklemlerinin graf kenarları üzerinde ayrık zamanlı simülasyonu
  3. En yüksek dereceye (Degree) sahip 'Fenomen / Lider' aboneler hedeflendiğinde kampanyanın yayılma hızı
- **Mülakat Sorusu:** *"Ağlarda bilgi veya hastalık yayılımında temel üreme katsayısı (R0) ile grafın spektral yarıçapı (Spectral Radius) arasındaki matematiksel ilişki nedir?"*

### Gün 273: Kişiselleştirilmiş PageRank ile Müşteri Nüfuz ve Etki Skoru
- **İş Alanı:** Pazarlama Analitiği & Etkileyici (Influencer) Müşteri Tespiti
- **Veri Kaynağı:** Aboneler Arası Aylık Arama Hacimleri ve Süreleri Matrisi
- **Model:** Kişiselleştirilmiş PageRank (Personalized PageRank - PPR) Güç İterasyonu
- **Türkçe Değişkenler:** `abone_id, gorusme_suresi_kenar_agirligi, damping_faktoru_085, pagerank_etki_skoru, vip_musteri_mi`
- **Kapsam:** Abonelerin sadece kaç kişiyi aradığına değil, aradığı kişilerin de ne kadar önemli/nüfuzlu olduğuna bakarak çevrelerine yön veren 'Kanaat Önderi' aboneleri PageRank ile tespit eder.
- **Jupyter Notebook (`gun_273_musteri_etki_skoru_pagerank.ipynb`):**
  1. Geçiş olasılıkları matrisinin (Stochastic Matrix) arama süreleriyle ağırlıklandırılması
  2. Güç iterasyonu (Power Iteration) ile PageRank vektörünün yakınsayana kadar hesaplanması
  3. En etkili %1'lik abone grubuna özel VIP müşteri sadakat tekliflerinin tanımlanması
- **Mülakat Sorusu:** *"PageRank algoritmasında 'Damping Factor' (Genellikle 0.85) kullanılmasının matematiksel ve graf yapısındaki (Örn: Örümcek Tuzakları / Spider Traps) sebebi nedir?"*

### Gün 274: Graph Attention Network (GAT) ile Hücre Anomali Tespiti
- **İş Alanı:** Gelişmiş Şebeke Teşhisi & Dikkat Mekanizmalı Ağ Analitiği
- **Veri Kaynağı:** Baz İstasyonu KPI Özellikleri ve Komşu Hücre Bağlantıları
- **Model:** Görselleştirilebilir Dikkat Katsayıları (Attention Weights) ile GAT Katmanı
- **Türkçe Değişkenler:** `hucre_ozellik_vektoru, komsu_dikkat_katsayisi_alpha, agirlikli_komsuluk_toplami, anomali_olasiligi`
- **Kapsam:** Her komşunun aynı öneme sahip olmadığını kabul eden Dikkat (Attention) mekanizmasıyla, baz istasyonunun sinyal probleminde hangi komşunun daha çok suçlu olduğunu tespit eder.
- **Jupyter Notebook (`gun_274_gat_hucre_anomali_dikkat_mekanizmasi.ipynb`):**
  1. Hücre çiftleri arasındaki özellik benzerliğini ölçen Softmax dikkat mekanizmasının kodlanması
  2. Her düğümün yeni temsilinin komşularının dikkat ağırlıklı ortalamasıyla güncellenmesi
  3. Komşularından anormal derecede farklı davranan izole veya hatalı hücrelerin yakalanması
- **Mülakat Sorusu:** *"Graph Convolutional Network (GCN) ile Graph Attention Network (GAT) arasındaki temel mimari fark nedir ve GAT dinamik komşuluk ağırlıklarını nasıl öğrenir?"*

### Gün 275: Data Center Sunucu Rafı Kablolama ve Yerleşim Minimizasyonu
- **İş Alanı:** Veri Merkezi Altyapısı & Enerji/Kablo Maliyeti Optimizasyonu
- **Veri Kaynağı:** Sunucular Arası Saniyelik Veri Trafiği Matrisi ve Boş Kabin Konumları
- **Model:** Karesel Atama Problemi (QAP) + Benzetimli Tavlama (Simulated Annealing)
- **Türkçe Değişkenler:** `sunucu_sayisi_100, kabin_sayisi_20, sunucular_arasi_trafik_gbps, kabin_mesafeleri_metre, toplam_kablo_metresi`
- **Kapsam:** Birbiriyle çok yoğun veri alışverişi yapan sunucuları aynı kabine veya en yakın raflara yerleştirerek veri merkezindeki fiber kablo uzunluğunu ve gecikmeyi minimize eder.
- **Jupyter Notebook (`gun_275_veri_merkezi_raf_yerlesimi_qap.ipynb`):**
  1. Sunucular arası trafik hacmi matrisi ile raflar arası fiziksel mesafe matrisinin çarpımı
  2. Benzetimli Tavlama (Simulated Annealing) ile rastgele yerleşimlerin optimize edilmesi
  3. Kablo maliyetinde ve ana omurga anahtar yükünde sağlanan %35 tasarrufun gösterilmesi
- **Mülakat Sorusu:** *"Karesel Atama Problemi (Quadratic Assignment Problem - QAP) neden NP-Hard sınıfındadır ve telekom tesis yerleşimlerinde sezgisel (Heuristic) yöntemler nasıl kullanılır?"*

### Gün 276: Node2Vec ile Şebeke Düğümleri Temsili (Graph Embedding)
- **İş Alanı:** Şebeke Makine Öğrenimi & Grafik Temsil Öğrenimi
- **Veri Kaynağı:** Router ve Switch Cihazları Topoloji Grafı
- **Model:** İki Parametreli Rastgele Yürüyüş (Random Walk p, q) + Word2Vec (Skip-Gram)
- **Türkçe Değişkenler:** `dugum_id, donus_parametresi_p, disari_cikma_parametresi_q, cikarilan_64_boyutlu_vektor, kosinus_benzerligi`
- **Kapsam:** Bir ağdaki yönlendiricilerin bağlantı yapılarını 64 boyutlu vektörlere dönüştürür; aynı roldeki (Örn: Çekirdek Router, Kenar Switch) cihazların haritada yan yana düşmesini sağlar.
- **Jupyter Notebook (`gun_276_node2vec_sebeke_dugum_vektorleri.ipynb`):**
  1. Genişlemesine (BFS) veya Derinlemesine (DFS) keşfi kontrol eden p ve q parametreli rastgele yürüyüşler
  2. Yürüyüş dizilerinin cümle, düğümlerin kelime gibi kabul edilerek Word2Vec ile eğitilmesi
  3. Vektör uzayında t-SNE görselleştirmesi ile şebeke katmanlarının (Core/Access) ayrışımının teyidi
- **Mülakat Sorusu:** *"Node2Vec algoritmasında p (Return) ve q (In-out) hiperparametreleri graf üzerinde Homofili (Homophily) ve Yapısal Rol (Structural Equivalence) kavramlarını nasıl dengeler?"*

### Gün 277: Büyük Ölçekli IP Trafiğinde Graf Çekirdeği (K-Core) Ayrıştırması
- **İş Alanı:** İnternet Değişim Noktası (IXP) & DDoS Dayanıklılık Analizi
- **Veri Kaynağı:** Otonom Sistemler (AS) BGP Yönlendirme Grafı (Border Gateway Protocol)
- **Model:** K-Cores ve K-Trusses Budama Algoritması (NetworkX)
- **Türkçe Değişkenler:** `as_numarasi, otonom_sistem_derecesi, k_degeri, ana_cekirdek_dugumleri, cevre_uc_noktalar`
- **Kapsam:** İnternetin küresel BGP grafındaki zayıf uç düğümleri katman katman soyarak en içteki kırılmaz ana çekirdeği (Core Tier-1 Operatörler) bularak küresel siber saldırı direncini ölçer.
- **Jupyter Notebook (`gun_277_bgp_grafinda_k_core_ayrisimi.ipynb`):**
  1. Derecesi k'dan az olan düğümlerin sırayla budanarak (Pruning) alt çekirdeklere inilmesi
  2. Her düğümün ait olduğu maksimum çekirdek katmanının (Coreness) hesaplanması
  3. Çekirdekte yer alan omurga operatörlerin çökmesi senaryosunda internetin parçalanma simülasyonu
- **Mülakat Sorusu:** *"Büyük ölçekli karmaşık ağlarda (Complex Networks) K-Core ayrıştırmasının derece dağılımına (Degree Distribution) göre grafın yoğun merkezini yakalamadaki farkı nedir?"*

### Gün 278: Stanford SNAP İnternet Otonom Sistemleri (AS) Arasında Maksimum Akış ve Min-Cut Analizi
- **İş Alanı:** BGP Yönlendirme, İnternet Değişim Noktaları (IXP) & Trafik Darboğaz Analizi
- **Veri Kaynağı:** Stanford SNAP - Autonomous Systems (AS) Internet Topology Graph Open Dataset
- **Model:** Edmonds-Karp Maksimum Akış (Max-Flow) & Minimum Kesme (Min-Cut) Teoremi
- **Türkçe Değişkenler:** `kaynak_as_numarasi, hedef_as_numarasi, baglanti_kapasitesi, tasinabilen_maksimum_trafik, kritik_darbogaz_as_kenarlari`
- **Kapsam:** Stanford Üniversitesi SNAP açık veri tabanındaki gerçek BGP otonom sistemler grafı üzerinde iki büyük operatör arasındaki maksimum veri akışını ve tıkanıklığa yol açan kritik darboğaz hatlarını belirler.
- **Jupyter Notebook (`gun_278_stanford_snap_as_max_flow_min_cut.ipynb`):**
  1. Stanford SNAP BGP otonom sistemler topoloji grafının ve kenar kapasitelerinin yüklenmesi
  2. Edmonds-Karp BFS artırma yolu algoritmasıyla iki otonom sistem arasındaki maksimum veri akışının bulunması
  3. Ağda tıkanmaya yol açan 'Minimum Kesme' (Min-Cut) kritik darboğaz kenarlarının tespiti
- **Mülakat Sorusu:** *"BGP yönlendirme grafında iki AS arasındaki Minimum Kesme (Min-Cut) kenarları tespit edildiğinde, trafiğin otonom sistemleri tıkamadan dengelenmesi (Anycast / Peering) nasıl kurgulanır?"*

### Gün 279: Bağlantı Tahmini (Link Prediction) ile Abone Aile Paketi Önerisi
- **İş Alanı:** Çapraz Satış (Cross-Sell) & CRM Sosyal Ağ Önericisi
- **Veri Kaynağı:** Mevcut Müşteri Arama İletişim Grafı ve Ortak Arkadaş Sayıları
- **Model:** Adamic-Adar İndeksi + Jaccard Benzerliği + Resource Allocation
- **Türkçe Değişkenler:** `abone_1, abone_2, ortak_aradiklari_kisi_sayisi, adamic_adar_skoru, potansiyel_aile_bireyi_mi`
- **Kapsam:** Birbirini doğrudan aramayan ama ortak aradıkları kişiler aynı olan aboneleri (Örn: Anne ve Çocuk) Adamic-Adar graf metriğiyle bularak Turkcell 'Bizce Aile' paketine davet eder.
- **Jupyter Notebook (`gun_279_baglanti_tahmini_adamic_adar.ipynb`):**
  1. İki abone arasındaki ortak komşuların derecelerinin ters logaritmasıyla (Adamic-Adar) skorlanması
  2. Herkesin aradığı popüler numaraların (Örn: Çağrı merkezi) ortak arkadaş gürültüsünden elenmesi
  3. En yüksek bağ olasılığına sahip abone çiftlerine otomatik aile tarifesi teklif üretimi
- **Mülakat Sorusu:** *"Sosyal ağlarda bağlantı tahmininde Adamic-Adar metriğinin basit Ortak Komşu (Common Neighbors) sayısına göre popüler/merkezi düğümleri cezalandırma mantığı nasıldır?"*

### Gün 280: Kendi Kendini İyileştiren Dinamik Ağ Topolojisi Simülatörü (Modül Capstone)
- **İş Alanı:** Akıllı Otonom Ağlar & Graf Tabanlı Arıza Kurtarma
- **Veri Kaynağı:** 500 Düğümlü Dinamik Ağ Topolojisi ve Anlık Düğüm Çökme Senaryoları
- **Model:** Dinamik Graf Güncelleme + Yerel Yeniden Bağlanma Heuristiği
- **Türkçe Değişkenler:** `toplam_dugum_500, coken_dugumler, otomatik_eklenen_telsiz_baglanti, ag_baglantili_kaldi_mi, onarim_suresi_ms`
- **Kapsam:** Ağdaki rastgele 20 düğüm aynı anda çöktüğünde grafın parçalanıp izole adacıklar oluşturmasını önlemek için insansız hava araçları veya mikrodalga linklerle grafı saniyede birleştiren tam otonom simülasyondur.
- **Jupyter Notebook (`gun_280_kendi_kendini_iyilestiren_graf_simulasyonu.ipynb`):**
  1. Graf bağlantılılığının (Connected Components) anlık olarak izlenmesi
  2. Ayrılan bileşenlerin ağırlık merkezleri arasına minimum maliyetli köprü kenarların açılması
  3. Ağ parçalanmasının 100 milisaniye içinde tamamen ortadan kaldırıldığının görselleştirilmesi
- **Mülakat Sorusu:** *"Graf teorisinde 'Perkolasyon Eşiği' (Percolation Threshold) nedir ve telekom şebekelerinde rastgele arızalar ile hedefli saldırıların ağ bütünlüğü üzerindeki etkisi nasıl farklılaşır?"*


## Modül 22: Güvenilir AI, XAI, Kuantum Hazırlığı & 5 Büyük Mezuniyet Capstone

---

### Gün 281: SHAP ile Müşteri Kredi / Churn Karar Açıklayıcısı
- **İş Alanı:** Model Açıklanabilirliği (XAI) & Finansal/Hukuki Uyum
- **Veri Kaynağı:** Paycell Kredi Skorlama ve Müşteri Ayrılma Veri Seti
- **Model:** LightGBM + TreeSHAP (Ağaç Açıklayıcı Değerleri)
- **Türkçe Değişkenler:** `musteri_id, lgbm_tahmin_olasiligi, temel_shap_degeri, kolon_katkilari_tl, red_gerekcesi_metni`
- **Kapsam:** Yapay zekanın bir müşteriye neden kredi vermediğini veya neden 'ayrılacak' dediğini TreeSHAP ile matematiksel olarak ayrıştırır; BDDK ve KVKK standartlarında resmi ret gerekçesi yazar.
- **Jupyter Notebook (`gun_281_shap_kredi_churn_karar_aciklayici.ipynb`):**
  1. LightGBM modeliyle tahmin yapılması ve TreeExplainer ile Shapley değerlerinin çıkarılması
  2. Abonenin faturasını ödememe geçmişinin karar üzerindeki negatif/pozitif katkı analizi
  3. Müşterinin anlayacağı 'Son 3 ayda 2 kez gecikmeye girdiğiniz için başvurunuz onaylanmadı' raporu
- **Mülakat Sorusu:** *"Oyun teorisindeki Shapley Değerleri (Shapley Values) ile LIME arasındaki teorik garanti (Doğruluk, Simetri, Toplanabilirlik) farkları nelerdir?"*

### Gün 282: LIME ile Kara Kutu Modelin Yerel Doğrulaması
- **İş Alanı:** Model Teftişi & Denetlenebilir Yapay Zeka Sistemleri
- **Veri Kaynağı:** Şebeke Güvenlik Duvarı Karmaşık Saldırı Tespit Verisi
- **Model:** Scikit-Learn Random Forest + LIME Tabular Explainer
- **Türkçe Değişkenler:** `incelenen_paket_verisi, yerel_vekil_model, ozellik_agirliklari, karar_siniri_mesafesi, aciklama_grafigi`
- **Kapsam:** Son derece karmaşık bir kara kutu yapay zeka modelinin belirli bir IP paketine neden 'Saldırı' dediğini, o paketin etrafında sahte veriler üreterek yerel doğrusal modelle açıklar.
- **Jupyter Notebook (`gun_282_lime_kara_kutu_model_dogrulama.ipynb`):**
  1. Seçilen şüpheli veri noktasının etrafında Gauss gürültüsüyle tedirgin (Perturbed) örnekler üretilmesi
  2. Örneklerin kara kutu modele sorularak yerel bir Ridge Regresyonu uydurulması
  3. Hangi port veya bayt değerinin alarmı tetiklediğinin yerel katsayılarla ispatlanması
- **Mülakat Sorusu:** *"LIME (Local Interpretable Model-agnostic Explanations) metodunda yerel tedirgin örneklerin mesafeye göre üstel ağırlıklandırılmasının (Exponential Kernel) sebebi nedir?"*

### Gün 283: Adil Yapay Zeka (Fairness & Bias) Denetim Motoru
- **İş Alanı:** Etik Yapay Zeka & Sosyal Sorumluluk / Ayrımcılık Denetimi
- **Veri Kaynağı:** İşe Alım veya Kredi Başvurusu Hassas Nitelik Verisi (Cinsiyet, Yaş, Şehir)
- **Model:** Disparate Impact Oranı + Demografik Eşitlik (Demographic Parity) Metrikleri
- **Türkçe Değişkenler:** `korunan_grup_cinsiyet, kabul_orani_grup_a, kabul_orani_grup_b, farklilik_etkisi_orani_08, model_adil_mi`
- **Kapsam:** Eğitilen modellerin belirli yaş gruplarına, kadınlara veya farklı şehirlere karşı gizli ayrımcılık yapıp yapmadığını '4/5 Kuralı' (Disparate Impact) ile denetleyip tarafsızlık karnesi çıkarır.
- **Jupyter Notebook (`gun_283_adil_yapay_zeka_bias_denetimi.ipynb`):**
  1. Hassas kolonların (Protected Attributes) model tahminlerindeki pozitif oranlarının kıyası
  2. Demografik Eşitlik (Demographic Parity) ve Fırsat Eşitliği (Equal Opportunity) metriklerinin hesabı
  3. Disparate Impact oranı 0.8'in altında kalan taraflı modellerin yayına alınmasının engellenmesi
- **Mülakat Sorusu:** *"Adil yapay zekada (Fairness in ML) 'Demographic Parity' ile 'Equalized Odds' kriterleri neden aynı anda sağlanamaz (Fairness Impossibility Theorem)?"*

### Gün 284: Düşmanca (Adversarial) Saldırılara Karşı Model Savunması
- **İş Alanı:** Yapay Zeka Güvenliği & Model Dayanıklılığı (Model Robustness)
- **Veri Kaynağı:** Turkcell Dolandırıcılık Tespit Modeli Sayısal Girdi Vektörleri
- **Model:** Hızlı Gradyan İşaret Metodu (FGSM) + Karşıt Eğitim (Adversarial Training)
- **Türkçe Değişkenler:** `orijinal_girdi, gradyan_isareti_epsilon, saldiriya_ugramis_girdi, model_yaniltildi_mi, savunmali_model_dogrulugu`
- **Kapsam:** Dolandırıcıların sisteme yakalanmamak için faturaya ve arama sürelerine eklediği mikro gürültüleri (FGSM saldırısı) simüle eder; modeli bu saldırı örnekleriyle eğiterek bağışık hale getirir.
- **Jupyter Notebook (`gun_284_adversarial_saldiri_ve_fgsm_savunma.ipynb`):**
  1. Modelin kayıp fonksiyonunun girdiye göre gradyanını alarak FGSM gürültüsü üretilmesi
  2. Modelin temiz veride %95 başarılıyken hafif gürültülü veride %40'a nasıl çöktüğünün testi
  3. Karşıt örneklerin eğitim kümesine katılmasıyla (Adversarial Training) modelin zırhlandırılması
- **Mülakat Sorusu:** *"Adversarial Machine Learning'de FGSM (Fast Gradient Sign Method) ile PGD (Projected Gradient Descent) saldırıları arasındaki fark ve hesaplama maliyeti nedir?"*

### Gün 285: Model Sürüklenmesi (Drift) ve Wasserstein Metrik Takibi
- **İş Alanı:** MLOps & Üretim Hattı Model Yaşam Döngüsü İzleme
- **Veri Kaynağı:** Ocak Ayı (Eğitim) ile Haziran Ayı (Canlı Üretim) Şebeke Trafik Özellikleri
- **Model:** Wasserstein Mesafesi (Earth Mover's Distance) + KS-Testi (Kolmogorov-Smirnov)
- **Türkçe Değişkenler:** `referans_dagilim, canli_veri_dagilimi, wasserstein_mesafesi, p_degeri, model_yeniden_egitilsin_mi`
- **Kapsam:** Yaz tatiliyle birlikte kullanıcı alışkanlıkları değiştiğinde modelin tahmin hatasını beklemeden girdi verisindeki istatistiksel kaymayı (Data Drift) Wasserstein mesafesiyle ölçer.
- **Jupyter Notebook (`gun_285_veri_ve_konsept_suruklenmesi_wasserstein.ipynb`):**
  1. Eğitim ve canlı ortam özellik dağılımlarının kümülatif yoğunluk fonksiyonlarının (CDF) çıkarılması
  2. Scipy ile Kolmogorov-Smirnov ve Wasserstein metriklerinin hesaplanması
  3. Kayma eşiği aşıldığında otomatik 'Veri Sürüklendi, Yeniden Eğitim Başlat' tetikleyicisi
- **Mülakat Sorusu:** *"Veri sürüklenmesi (Data Drift / Covariate Shift) ile Konsept Sürüklenmesi (Concept Drift) arasındaki fark nedir ve etiket gecikmesi olduğunda hangisi takip edilebilir?"*

### Gün 286: Diferansiyel Gizlilik (Differential Privacy) Gürültü Motoru
- **İş Alanı:** Veri Gizliliği, KVKK/GDPR Uyumlu Büyük Veri Paylaşımı
- **Veri Kaynağı:** Müşteri Konum ve Harcama İstatistik Veritabanı Tablosu
- **Model:** Laplace Mekanizması (Epsilon-Diferansiyel Gizlilik)
- **Türkçe Değişkenler:** `gercek_ortalama_harcama, gizlilik_butcesi_epsilon, genel_duyarlilik_delta_f, laplace_gurultusu, anonim_istatistik`
- **Kapsam:** Dışarıya veya üniversitelere anonim veri seti verirken, tek bir müşterinin varlığının geriye dönük çıkarılamaması için verilere matematiksel Laplace gürültüsü enjekte eder.
- **Jupyter Notebook (`gun_286_diferansiyel_gizlilik_laplace.ipynb`):**
  1. Sorgu fonksiyonunun global duyarlılığının (Sensitivity = Max Değişim) hesaplanması
  2. Epsilon gizlilik bütçesine ters orantılı Laplace gürültüsü üretilip sonuca eklenmesi
  3. Veritabanına tek bir zengin müşteri eklense bile sonuç istatistiğinin değişmediğinin kanıtı
- **Mülakat Sorusu:** *"Diferansiyel Gizlilikte (Differential Privacy) Epsilon (ε) parametresi sıfıra yaklaştıkça gizlilik ve veri faydası (Utility) arasındaki ödünleşim (Trade-off) nasıl evrilir?"*

### Gün 287: CTU-13 Gerçek Ağ Trafiği ile Güvensiz SSL/TLS Şifreleme ve Kriptografik Zafiyet Tespiti
- **İş Alanı:** Kriptografik Güvenlik, Ağ Şifreleme Denetimi & SOC Güvenlik Masası
- **Veri Kaynağı:** Stratosphere IPS / CTU-13 Malware & Normal TLS Open Traffic Dataset
- **Model:** Scikit-learn Random Forest + Kuralcı Kriptografik Risk Puanlayıcı
- **Türkçe Değişkenler:** `tls_surumu_1_2_1_3, sifre_paketi_ciphersuite, sertifika_gecerlilik_gun, guvensiz_sifreleme_var_mi, acil_aksiyon_kurali`
- **Kapsam:** Gerçek CTU-13 ağ paket dökümündeki TLS el sıkışmalarını analiz ederek şebekede eski/zayıf şifreleme kullanan savunmasız cihazları tespit edip güvenlik duvarı kuralı üretir.
- **Jupyter Notebook (`gun_287_ctu13_tls_kripto_zafiyet_tespiti.ipynb`):**
  1. CTU-13 açık ağ trafiğinden TLS Client Hello / Server Hello paketlerinin ayrıştırılması
  2. Eski şifreleme algoritmaları (RC4, 3DES, MD5) ve geçersiz sertifikaların taranması
  3. Zafiyetli bağlantıları engelleyen otomatik iptables ve WAF kurallarının çıktılanması
- **Mülakat Sorusu:** *"Şifreli ağ trafiğinde (HTTPS/TLS) paket içeriğini çözmeden Client Hello ve Server Hello paketlerindeki Cipher Suite listesinden cihazın kimliği (JA3 Fingerprinting) nasıl çıkarılır?"*

### Gün 288: OpenCelliD Gerçek Türkiye Hücre Kuleleri Koordinatlarıyla Kapsama ve Voronoi Analizi
- **İş Alanı:** Radyo Şebeke Planlama & Coğrafi Hücre Kapsama Optimizasyonu
- **Veri Kaynağı:** OpenCelliD - Türkiye Hücre Kuleleri Gerçek GPS Koordinat Verisi (Açık Veri)
- **Model:** SciPy Voronoi Poligonları + Scikit-Learn Mekânsal K-Means Kümeleme
- **Türkçe Değişkenler:** `istasyon_enlem, istasyon_boylam, radyo_turu_lte_umts, hucre_voronoi_alani_km2, yuksek_yogunluklu_bolge_mi`
- **Kapsam:** Dünyanın en büyük açık kaynaklı hücresel kule veri tabanı OpenCelliD'den Türkiye'deki gerçek baz istasyonlarını indirir; Voronoi poligonlarıyla her kulenin gerçek kapsama alanını ve aşırı yüklenme riskini modeller.
- **Jupyter Notebook (`gun_288_opencellid_turkiye_voronoi_kapsama.ipynb`):**
  1. OpenCelliD Türkiye CSV dosyasından GSM, UMTS ve LTE baz istasyonu koordinatlarının filtrelenmesi
  2. SciPy Voronoi ile her baz istasyonunun geometrik etki alanının (kapsama poligonu) hesaplanması
  3. Hücre alanı çok geniş kalan kırsal bölgelerin tespiti ve kapasite takviye önerisi
- **Mülakat Sorusu:** *"Baz istasyonu planlamasında Voronoi diyagramlarının sağladığı geometrik kapsama sınırları ile radyo yayılım modelleri (Hata-Okumura / 3GPP) arasındaki fark nedir?"*

### Gün 289: Sentetik Telekom Tablo Verisi Üretimi (CTGAN Prensibi)
- **İş Alanı:** Gizlilik Korumalı Test Verisi & Veri Çoğaltma (Data Augmentation)
- **Veri Kaynağı:** Gerçek Müşteri Profil İstatistikleri (Yaş, Fatura, Tarife, Cihaz)
- **Model:** Koşullu Tablo Dağılım Modeli (Karisim Modelleri Gaussian Mixture + Karar Ağacı)
- **Türkçe Değişkenler:** `gercek_tablo, sentetik_tablo, dagilim_benzerligi_skoru, korelasyon_korunma_orani, gercek_musteri_sizdi_mi`
- **Kapsam:** Yazılımcıların gerçek müşteri verisi görmeden test yapabilmesi için gerçek verinin istatistiksel korelasyonlarını koruyan ancak %100 uydurma olan 50.000 satırlık sentetik müşteri tablosu üretir.
- **Jupyter Notebook (`gun_289_sentetik_tablo_verisi_ureteci.ipynb`):**
  1. Sürekli değişkenlerin Gauss Karışım Modelleriyle (GMM) çok modlu dağılımlarının çıkarılması
  2. Kategorik ve sayısal kolonlar arası korelasyon matrisinin ters dönüşümle korunması
  3. Sentetik veride eğitilen modelin gerçek veride de aynı başarıyı gösterdiğinin kanıtlanması
- **Mülakat Sorusu:** *"Tablosal verilerde sentetik üretim yaparken tek değişkenli (Univariate) dağılımları tutturmak ile çok değişkenli (Multivariate) korelasyonları korumak arasındaki zorluk nedir?"*

### Gün 290: Model Güven Aralığı ve Kalibrasyonu (Platt Scaling)
- **İş Alanı:** Risk Analizi & Karar Güvenilirliği (Uncertainty Estimation)
- **Veri Kaynağı:** Dengesiz Dolandırıcılık ve Ağ Çökmesi Tahmin Çıktıları
- **Model:** Platt Lojistik Kalibrasyonu + Brier Skoru + Güvenilirlik Eğrileri (Reliability Curve)
- **Türkçe Değişkenler:** `ham_model_skoru, kalibre_edilmis_olasilik, brier_skoru_kaybi, beklenen_kalibrasyon_hatasi_ece, guvenilir_tahmin`
- **Kapsam:** Bir model '%99 ihtimalle bu abone ayrılacak' dediğinde gerçekten o gruptaki 100 kişiden 99'unun ayrılıp ayrılmadığını kalibre eder; aşırı özgüvenli hatalı tahminleri dengeler.
- **Jupyter Notebook (`gun_290_model_olasilik_kalibrasyonu_platt.ipynb`):**
  1. Ham model skorları ile gerçek sınıfların güvenilirlik eğrisinde (Reliability Diagram) çizimi
  2. Platt Scaling (Lojistik Sigmoid) ve İzotonik Regresyon ile olasılıkların kalibre edilmesi
  3. Beklenen Kalibrasyon Hatasının (ECE - Expected Calibration Error) %2'nin altına düşürülmesi
- **Mülakat Sorusu:** *"Derin sinir ağlarının ve Karmaşık Ağaç modellerinin ham Softmax çıktılarının 'Aşırı Özgüvenli' (Overconfident) olma eğilimi neden kaynaklanır ve ECE nasıl hesaplanır?"*

### Gün 291: Çok Dilli Model Eşdeğerlik ve Çeviri Güvenliği Testi
- **İş Alanı:** Müşteri Deneyimi & Yabancı Uyruklu Abone Desteği
- **Veri Kaynağı:** Türkçe, İngilizce ve Arapça Müşteri Hizmetleri İfade Çiftleri
- **Model:** FastText Çapraz Dil Hizalama + Karşılıklı Çeviri Doğrulama
- **Türkçe Değişkenler:** `turkce_ifade, ingilizce_ifade, arapca_ifade, anlam_sapma_skoru_0_1, guvenli_cevap_uretildi_mi`
- **Kapsam:** Turkcell çağrı merkezine Arapça veya İngilizce gelen bir şikayetin Türkçe karşılığı ile aynı hassasiyet ve nezaketle yanıtlanıp yanıtlanmadığını semantik vektör farklarıyla denetler.
- **Jupyter Notebook (`gun_291_cok_dilli_ifade_guvenlik_denetimi.ipynb`):**
  1. Farklı dillerdeki müşteri ifadelerinin çok dilli kelime vektör uzayına izdüşürülmesi
  2. Anlamsal mesafe eşiğinin üzerindeki kaymaların (Örn: Yanlış hukuki terim kullanımı) tespiti
  3. Üç dilde de aynı SLA ve çözüm standartlarının sağlandığının raporlanması
- **Mülakat Sorusu:** *"Çok dilli (Multilingual) NLP sistemlerinde 'Geri Çeviri' (Back-Translation) yöntemi veri çoğaltma ve kalite kontrolünde nasıl bir matematiksel denetim aracıdır?"*

### Gün 292: Yapay Zeka Telif Hakkı ve Veri Çıkarma Güvenliği (Membership Inference)
- **İş Alanı:** Model Güvenliği & Özel Hayatın Gizliliği Saldırı Testleri
- **Veri Kaynağı:** Eğitim Kümesinde Olan ve Olmayan Abone Veri Örnekleri
- **Model:** Gölge Model Tekniği (Shadow Model) ile Üyelik Çıkarım Saldırısı
- **Türkçe Değişkenler:** `sorgulanan_musteri, modelin_guven_skoru, egitim_kumesinde_var_miydi_tahmini, uyelik_sizinti_riski`
- **Kapsam:** Kötü niyetli bir kişinin yapay zekaya sürekli soru sorarak 'Ahmet Bey'in verisi bu modelin eğitiminde kullanılmış mı?' sorusunu çözmesini (Modelden veri sızmasını) engelleyen güvenlik testidir.
- **Jupyter Notebook (`gun_292_membership_inference_guvenlik_testi.ipynb`):**
  1. Hedef modelin davranışını taklit eden 3 adet Gölge Model (Shadow Model) eğitilmesi
  2. Modelin ezberlediği (Overfitting) eğitim örneklerindeki yüksek güven skoru anomalisinin analizi
  3. Üyelik çıkarım saldırısına karşı modellerin aşırı öğrenmesini engelleyen kısıtların konulması
- **Mülakat Sorusu:** *"Membership Inference Attack (Üyelik Çıkarım Saldırısı) neden doğrudan modelin aşırı öğrenmesiyle (Overfitting) ilişkilidir?"*

### Gün 293: Otomatik Model Kartı ve Denetim Raporu Üreticisi
- **İş Alanı:** Yapay Zeka Yönetişimi (AI Governance) & Kurumsal Uyum
- **Veri Kaynağı:** Tüm Modellerin Hiperparametreleri, Metrikleri ve Veri Dağılımları
- **Model:** Google Model Cards Standardı + Jinja2 Markdown Şablon Motoru
- **Türkçe Değişkenler:** `model_katalog_bilgisi, egitim_tarihi, dogruluk_metrikleri, kisitlamalar_ve_etik_uyarilar, olusturulan_model_karti_md`
- **Kapsam:** Turkcell bünyesinde canlıya çıkacak her model için mimari, başarım metrikleri, sınırları ve etik kullanım koşullarını içeren uluslararası standartta 'Model Kartı' dokümanını otomatik basar.
- **Jupyter Notebook (`gun_293_otomatik_model_karti_ureteci.ipynb`):**
  1. Model metadata sözlüğünün (Amaç, Girdiler, Çıktılar, Başarım, Önyargı Karnesi) toplanması
  2. Jinja2 şablonuna metrik grafiklerinin ve uyarı maddelerinin otomatik gömülmesi
  3. Kurumsal arşiv ve BTK denetimi için hazır Markdown ve HTML model karnesi üretimi
- **Mülakat Sorusu:** *"Google ve akademik çevreler tarafından önerilen 'Model Cards for Model Reporting' yaklaşımının AI şeffaflığı ve regülasyonlar (EU AI Act) açısından rolü nedir?"*

### Gün 294: Yeşil Bilişim: Model Karbon Ayak İzi ve Enerji Ölçer
- **İş Alanı:** Sürdürülebilirlik & Çevre Dostu Yeşil Yapay Zeka (Green AI)
- **Veri Kaynağı:** Model Eğitimi CPU/GPU Kullanım Süreleri ve Güç Telemetrisi
- **Model:** CodeCarbon Formülasyonu (TDP Watt x Saat x Ülke Karbon Yoğunluğu)
- **Türkçe Değişkenler:** `islemci_turu, harcanan_enerji_kwh, turkiye_sebeke_karbon_faktoru, salinan_co2_kg, dikilmesi_gereken_agac`
- **Kapsam:** Bir yapay zeka modelinin eğitimi veya 1 milyon tahmini sırasında sunucunun tükettiği elektriği ölçerek atmosfere salınan CO2 miktarını ve nötrlemek için gereken ağaç sayısını hesaplar.
- **Jupyter Notebook (`gun_294_green_ai_karbon_ayak_izi_hesaplayici.ipynb`):**
  1. İşlemcinin Termal Tasarım Gücü (TDP) ve aktif çalışma süresinden tüketilen kWh enerjinin hesabı
  2. Türkiye elektrik şebekesi ortalama karbon katsayısıyla (Örn: 0.44 kg CO2/kWh) emisyon hesabı
  3. Aynı işi %80 daha az enerjiyle yapan hafif modellerin çevreye sağladığı faydanın raporlanması
- **Mülakat Sorusu:** *"'Red AI' (Sadece doğruluk artırmak için devasa kaynak harcama) ile 'Green AI' (Enerji ve hesaplama verimliliğini de başarım metriği sayma) arasındaki paradigma farkı nedir?"*

### Gün 295: Yapay Zeka Savunma Hattı Entegre Sağlık Testi
- **İş Alanı:** Sistem Sağlığı & Uçtan Uca Yapay Zeka Güvenlik Denetimi
- **Veri Kaynağı:** 100 Farklı Zorlayıcı, Bozuk ve Kasıtlı Hatalı Test Veri Girdisi
- **Model:** Bütünleşik Sağlık Denetleyicisi (Veri Tipi + Aralık + Dağılım + Güvenlik Filtresi)
- **Türkçe Değişkenler:** `test_girdisi, veri_tipi_dogru_mu, deger_mantikli_mi, saldiri_iz_var_mi, saglik_testi_gecti_mi`
- **Kapsam:** Canlı şebekeye bağlı yapay zeka modellerine gelen veriyi ilk karşılayan güvenlik kapısıdır; NaN değerler, aşırı uç sayılar, SQL enjeksiyonları ve biçimsiz verileri modele ulaşmadan eler.
- **Jupyter Notebook (`gun_295_ai_savunma_hatti_saglik_testi.ipynb`):**
  1. Pydantic benzeri veri tipi ve mantıksal aralık kısıtlarının Python seviyesinde uygulanması
  2. Sıfıra bölme veya matris tekilliği yaratacak zehirli girdilerin ayrıştırılması
  3. Tüm testleri başarıyla geçen 'Zırhlı Model Boru Hattı' (Robust Pipeline) teslimi
- **Mülakat Sorusu:** *"Üretim ortamındaki yapay zeka boru hatlarında (Production AI Pipelines) 'Fail-Safe Default' prensibi nasıl kurgulanır ve model çöktüğünde sistem nasıl davranmalıdır?"*

### Gün 296: BÜYÜK CAPSTONE 1: Otonom Şebeke Operasyon Merkezi (NOC)
- **İş Alanı:** Bütünleşik Şebeke Yönetimi, NOC Otomasyonu & Self-Healing Ağ
- **Veri Kaynağı:** 10.000 Baz İstasyonunun Çok Modlu Alarm, KPI ve Topoloji Verileri
- **Model:** Graf Analitiği + Durum Makinesi Ajanı + Kök Neden Karar Ağacı
- **Türkçe Değişkenler:** `sebeke_durumu, aktif_alarm_listesi, izole_edilen_kok_ariza, uygulanan_otonom_onarım, sebeke_saglik_skoru_100`
- **Kapsam:** Bütün müfredattaki şebeke ve ajan bilgilerini birleştiren devasa proje; Türkiye genelinde elektrik kesintisi ve kablo kopması durumunda insan müdahalesiz rotalama ve onarım yapar.
- **Jupyter Notebook (`gun_296_buyuk_capstone_1_otonom_noc.ipynb`):**
  1. Alarm akışından Netcool benzeri kural tabanlı gürültü eleme ve kök neden kümelemesi
  2. Kopan baz istasyonlarının trafiğini NetworkX ile otomatik yedek mikrodalga rotalara aktarma
  3. Arıza biletini oluşturup sahaya gidecek ekibin rotasını çizen uçtan uca otonom operasyon
- **Mülakat Sorusu:** *"Telekomünikasyonda Seviye 4 Otonom Ağ (Level 4 Autonomous Network - TM Forum) mimarisine ulaşmak için gereken algısal, karar verici ve icracı döngüler nelerdir?"*

### Gün 297: BÜYÜK CAPSTONE 2: Bütünleşik 360° Müşteri Zekası ve Churn Platformu
- **İş Alanı:** Pazarlama, Finans, Müşteri Hizmetleri & Gelir Büyütme Masası
- **Veri Kaynağı:** 100.000 Abonenin Fatura, Çağrı, Paket Kullanımı ve Memnuniyet Kayıtları
- **Model:** XGBoost / LightGBM Sınıflandırıcı + SHAP Karar Açıklayıcı + Kuralcı Teklif
- **Türkçe Değişkenler:** `abone_id, yillik_gelir_katkisi_arpu, ayrilma_riski_yuzde, temel_ayrilma_sebebi_fatura_cekim, onerilen_kisisel_paket`
- **Kapsam:** Bir müşterinin faturasından, müşteri hizmetlerini arama sıklığından ve internet kalitesinden yarın başka operatöre geçip geçmeyeceğini tahmin edip kişiye özel elde tutma paketi önerir.
- **Jupyter Notebook (`gun_297_buyuk_capstone_2_360_musteri_churn.ipynb`):**
  1. Yüz bin abonenin çok boyutlu davranışsal özniteliklerinin çıkarılması ve dengelenmesi
  2. En yüksek AUC skorlu modelin eğitilerek her müşterinin ayrılma olasılığının hesaplanması
  3. SHAP değerleriyle müşterinin neden gideceğinin bulunup karlı bir retention teklifinin atanması
- **Mülakat Sorusu:** *"Gerçek telekom dünyasında müşteri ayrılma tahmininde (Churn Prediction) ROC-AUC skoru yüksek olsa bile F1-Skor ve Precision-Recall dengesi neden iş birimi için daha kritiktir?"*

### Gün 298: BÜYÜK CAPSTONE 3: Uçtan Uca 5G Dilimleme ve Kaynak Orkestratörü
- **İş Alanı:** 5G Core & Radyo Ağı Orkestrasyonu (MANO) & SLA Güvencesi
- **Veri Kaynağı:** Gerçek Zamanlı eMBB, URLLC ve mMTC Trafik Talep Simülasyonu
- **Model:** Dinamik Programlama (Knapsack) + Q-Learning Pekiştirmeli Karar Motoru
- **Türkçe Değişkenler:** `anlik_sebeke_yuku, embb_tahsis_prb, urllc_tahsis_prb, mmtc_tahsis_prb, garanti_edilen_gecikme_ms, toplam_gelir`
- **Kapsam:** Aynı fiziksel baz istasyonu ve çekirdek şebeke üzerinden hem VR video izleyenleri hem ameliyat yapan otonom hastane sistemini milisaniyeler içinde SLA bozmadan yöneten orkestratördür.
- **Jupyter Notebook (`gun_298_buyuk_capstone_3_5g_slicing_orkestratoru.ipynb`):**
  1. Farklı servislerin bant genişliği ve gecikme fonksiyonlarının matematiksel tanımlanması
  2. Fiziksel radyo bloklarının (PRB) dilimler arasında dinamik paylaştırılması
  3. Ani trafik sıçramalarında hastane trafiğinin 1ms altında tutulduğunun canlı grafiği
- **Mülakat Sorusu:** *"5G Network Slicing'de Dinamik Kaynak Tahsisi yapılırken aşırı rezervasyon (Overbooking) oranı ile SLA ceza riski arasındaki optimizasyon dengesi nasıl kurulur?"*

### Gün 299: BÜYÜK CAPSTONE 4: Afet Durumu Hibrit Acil Haberleşme Ajanı
- **İş Alanı:** Deprem/Afet Kriz Masası, Acil İletişim & Hayat Kurtarma Teknolojileri
- **Veri Kaynağı:** Deprem Sonrası GSM Hücre Yıkım Haritası ve Abone S.O.S Sinyalleri
- **Model:** Uzaysal K-Means + Dijkstra Gezgin Satıcı (TSP) + Güç Kaynak Dağıtıcısı
- **Türkçe Değişkenler:** `yikilan_baz_istasyonu_sayisi, gonderilen_mobil_arac_sayisi, tespit_edilen_enkaz_sos_noktalari, kurtarilan_iletisim_orani`
- **Kapsam:** Deprem anında baz istasyonlarının %60'ı yıkıldığında, Turkcell mobil baz istasyonu araçlarını ve drone'larını enkaz altından gelen en çok acil yardım çağrısına göre anında konuşlandırır.
- **Jupyter Notebook (`gun_299_buyuk_capstone_4_afet_acil_haberlesme.ipynb`):**
  1. Çöken baz istasyonlarının coğrafi haritada çıkarılıp kapsama kör noktalarının tespiti
  2. Enkazdan gelen düşük güçlü Bluetooth/Wi-Fi acil durum paketlerinin konum kümelemesi
  3. Mobil baz istasyonlarının sahaya en hızlı varış rotalarının Dijkstra ile çıkarılması
- **Mülakat Sorusu:** *"Büyük doğal afetlerde telekom şebekelerinde 'Sinyalleşme Kilitlenmesi'ni (Congestion Collapse) önlemek için acil durum çağrılarına (Örn: 112) öncelik veren şebeke mimarisi nasıl çalışır?"*

### Gün 300: BÜYÜK CAPSTONE 5: Turkcell Geleceğin Şebekesi Dijital İkizi (Grand Finale)
- **İş Alanı:** Şirket Düzeyinde Bütünleşik Şebeke Simülatörü & Yapay Zeka Stratejisi
- **Veri Kaynağı:** Türkiye Geneli Sentetik Şebeke Telemetrisi, Finansal Gelirler ve Müşteri Akışı
- **Model:** Bütünleşik Ayrık Olay Simülatörü (DES) + Çok Ajanlı Yönetim + XAI Raporlama
- **Türkçe Değişkenler:** `simulasyon_gunu, toplam_turkcell_abonesi, gunluk_veri_tuketimi_petabyte, sebeke_enerji_tuketimi_mwh, net_kar_milyon_tl, mezuniyet_karnesi`
- **Kapsam:** 300 günlük yolculuğun nihai zirvesi; Turkcell'in şebekesini, müşterilerini, finansal akışını ve yeşil enerjisini sanal bir dijital ikiz üzerinde yaşatan ve yöneten devasa simülasyon.
- **Jupyter Notebook (`gun_300_buyuk_capstone_5_dijital_ikiz_grand_finale.ipynb`):**
  1. Tüm fiziksel şebeke katmanlarının (Fiber, 5G, Data Center, Enerji) sanal ikizinin kurulması
  2. Müşteri hareketlerinin, veri kullanımının ve faturalamanın gün gün simülasyonu
  3. Yapay zeka modellerinin şebekeyi otonom yöneterek enerji ve arıza maliyetlerini nasıl düşürdüğünün ve 300 günlük staj başarı sertifikasının yazdırılması
- **Mülakat Sorusu:** *"300 günlük bir yapay zeka mühendisliği stajını tamamlayan bir mühendis olarak, telekom sektöründe yapay zekanın geleceğini ve 'Sıfır İnsan Dokunuşlu Otonom Ağlar' (Zero-Touch Networks) vizyonunu nasıl değerlendirirsiniz?"*


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
