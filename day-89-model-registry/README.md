# Day 89: Model Kayıt Sistemi, Model Sürümleme, Staging/Production Yaşam Döngüsü

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![MLOps: Model Governance & CI/CD](https://img.shields.io/badge/MLOps-Model_Registry_%26_Rollback-darkgreen.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_model_registry.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin sekizinci gününde; Sculley et al. (NeurIPS 2015) *"Hidden Technical Debt in Machine Learning Systems"* ve Zaharia et al. (IEEE Micro 2018) *"Accelerating the Machine Learning Lifecycle with MLflow"* ilkeleri doğrultusunda **Merkezi Model Kayıt Sistemi (Model Registry)**, **Değişmez Sürümleme (Immutable Versioning: v1, v2, v3)**, **Aşama Yaşam Döngüsü (None -> Staging -> Production -> Archived)**, **Otomatik Kalite Kapıları (Quality Gates)** ve **Sıfır Kesintili Geri Alma (Zero-Downtime Rollback)** altyapısını sıfırdan kurup başarıyla doğruluyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Deneysel aşamada eğitilen onlarca `.pt` veya `.onnx` model dosyası tek başına üretim ortamına alınamaz. Model Registry olmadan çalışan sistemlerde:

1. **"Hangi Model Canlıda?" Kaosu (Production Ambiguity):**
   API servislerinin hangi ağırlık dosyasını yüklediği, bu dosyanın hangi eğitim kodu ve hangi veri kümesiyle oluşturulduğu belirsizleşir.
2. **Değişmezlik ve Sürümleme (Immutable Versioning):**
   Bir model sürümü (`v1`, `v2`) sisteme bir kez tescillendiğinde üzerine yazılamaz (immutable); tensör şeması (Input/Output Schema Signature) mühürlenir.
3. **Staging ve Otomatik Kalite Kapısı (Quality Gate):**
   Yeni bir aday model doğrudan canlıya (Production) alınmaz. Önce `STAGING` aşamasında Doğruluk ($Acc \ge 90\%$), Gecikme ($Latency \le 30\text{ ms}$) ve Kalibrasyon ($ECE \le 0.15$) testlerine tabi tutulur.
4. **Sıfır Kesintili Geri Alma (Zero-Downtime Rollback):**
   Canlı ortamda beklenmeyen bir anomali tespit edildiğinde, servis durdurulmadan tek bir veritabanı komutuyla önceki kararlı sürüme (< 10 ms) anında dönülür.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Hatalı/Bozuk Modellerin Canlıya Sızmasını Önleme:**
  Kalite kapısından geçemeyen kusurlu aday modeller (ör. düşük doğruluklu veya yüksek gecikmeli) otomatik olarak bloke edilir.
- **Tek Merkezden Yönetim (Single Source of Truth):**
  Tüm mikroservisler ve API gateway'ler en güncel üretim modelini tek bir sorguyla (`get_production_model("MiniViT")`) dinamik olarak yükler.
- **Model Soykütüğü ve Denetlenebilirlik (Auditability & Lineage):**
  Her model sürümünün hangi geliştirici tarafından, hangi eğitim koşusundan (run_id) ve ne zaman üretildiği denetim günlüğünde (audit log) saklanır.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Depolama Kapasitesi Yönetimi:**
  Arşivlenen eski modeller disk alanı kaplar; belirli bir saklama politikasından (retention policy, ör. son 5 sürümü sakla) sonra eski sürümler bulut soğuk depolamaya (S3 Glacier / GCS Coldline) taşınmalıdır.
- **Dağıtık Senkronizasyon Gecikmesi:**
  Çoklu pod/sunucu kümesinde çalışan inference servislerinin model güncellemesini eşzamanlı algılaması için Redis Pub/Sub veya webhook tetikleyicileri gereklidir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Model Yönetim Yaklaşımı | Versiyonlama | Otomatik Aşama | Kalite Kapısı | Sıfır Kesinti Rollback |
|---|---|---|---|---|
| **Model Registry Motoru (Bizim Sistem)** | **Artan (v1, v2..)** | **None $\to$ Staging $\to$ Prod $\to$ Arch** | **VAR (Acc, Latency, ECE)** | **ANINDA (< 10 ms)** |
| **MLflow Model Registry** | SemVer / Integer | Staging / Production / Archived | Webhook / CI/CD | Var |
| **AWS SageMaker Model Registry** | Versioned Package | Pending / Approved / Rejected | Model Package Group | Var |
| **Hugging Face Model Hub** | Git Commit / Tags | Branch tabanlı | Manuel PR Kontrolü | Git Revert |
| **Dosya Sistemi (Klasör Adlandırma)** | `model_v2_final.pt` | Yok (Manuel kopyalama) | Yok | Yüksek Risk / Kesinti |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        MODEL REGISTRY YAŞAM DÖNGÜSÜ VE KALİTE KAPISI (QUALITY GATE)                       │
│                                                                                                           │
│       Yeni Eğitilen Model (Run Artifact)                                                                  │
│          │                                                                                                │
│          ▼                                                                                                │
│       [ MODEL REGISTRATION (v_k) ] ──> Aşama: NONE                                                        │
│          │                                                                                                │
│          ▼                                                                                                │
│       [ STAGING AŞAMASI ] ──> ModelKaliteKapisi.degerlendir(M)                                            │
│          │                                                                                                │
│          ├── Kriter 1: Doğruluk   (Acc(M) >= 90.0%)                                                       │
│          ├── Kriter 2: Gecikme    (Latency(M) <= 30.0 ms)                                                 │
│          └── Kriter 3: Kalibrasyon(ECE(M) <= 0.15)                                                        │
│          │                                                                                                │
│          ├──────────── [ BAŞARISIZ (Kriter Karşılanamadı) ] ────> [ REDDEDİLDİ (BLOKE EDİLDİ) ]           │
│          │                                                                                                │
│          ▼ [ BAŞARILI (Tüm Kriterler Geçildi) ]                                                           │
│       [ PRODUCTION AŞAMASINA TERFİ ]                                                                      │
│          │                                                                                                │
│          ├──> v_yeni  ──> [ PRODUCTION ] (Aktif Canlı Trafik)                                             │
│          └──> v_eski  ──> [ ARCHIVED ]   (Geri Alma Güvencesi)                                            │
│                                                                                                           │
│       ⚠️ Canlı Anomali Durumunda: motor.geri_al(model_adi) ──> v_eski anında PRODUCTION'a döner!         │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Kalite Kapısı Doğrulama Fonksiyonu
Aday model $M$ için kalite kapısı onay koşulu:

$$\mathcal{Q}(M) = \mathbf{1}\big(\text{Acc}(M) \ge \tau_{\text{acc}}\big) \land \mathbf{1}\big(\text{Latency}(M) \le \tau_{\text{lat}}\big) \land \mathbf{1}\big(\text{ECE}(M) \le \tau_{\text{ece}}\big)$$

$$\text{Aşama Geçişi}(M) = \begin{cases} \text{PRODUCTION} & \text{eğer } \mathcal{Q}(M) = 1 \\ \text{REJECTED / BLOCKED} & \text{eğer } \mathcal{Q}(M) = 0 \end{cases}$$

### 2. Aşama Durum Makinesi (State Machine)
Durum uzayı $\mathcal{S} = \{\text{NONE}, \text{STAGING}, \text{PRODUCTION}, \text{ARCHIVED}\}$ ve aşama geçiş operatörü:

$$T: \mathcal{S} \times \mathcal{A} \longrightarrow \mathcal{S}$$

Tek bir aktif üretim modeli kısıtı (Production Invariant):

$$\sum_{v \in \mathcal{V}(M)} \mathbf{1}\big(\text{Stage}(v) = \text{PRODUCTION}\big) \le 1$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Model Registry** | *Model Kayıt Sistemi* | Eğitilmiş modellerin sürümlerini, metaverilerini, aşamalarını ve onay süreçlerini yöneten merkezi katalog. |
| **Model Version** | *Model Sürümü* | Belirli bir eğitim koşusundan üretilmiş, değiştirilemez (immutable) tekil model paketi. |
| **Staging** | *Hazırlık / Test Aşaması* | Üretime aday modelin otomatik kalite kapılarından ve gölge (shadow) testlerinden geçtiği ara aşama. |
| **Production** | *Canlı / Üretim Aşaması* | Canlı son kullanıcı isteklerini doğrudan yanıtlayan resmi ve aktif model sürümü. |
| **Archived** | *Arşiv Aşaması* | Yeni bir sürüm terfi ettiğinde emekliye ayrılan ancak acil geri alma (rollback) için saklanan sürüm. |
| **Quality Gate** | *Kalite Kapısı* | Modelin canlıya geçebilmesi için doğruluk, gecikme ve kalibrasyon kriterlerini denetleyen otomatik bariyer. |
| **Rollback** | *Geri Alma* | Canlıda hata veren yeni modeli anında arşive çekip önceki kararlı sürümü sıfır kesintiyle devreye alma işlemi. |
| **Model Signature** | *Model Şeması / İmzası* | Modelin beklediği girdi ve ürettiği çıktı tensörlerinin boyut ve veri tipi sözleşmesi. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Hangi modelin canlıda olduğu tek merkezden yönetilir; Kalite kapısı ile hatalı modellerin prod'a geçişi %100 engellenir; Tek komutla sıfır kesintili acil geri alma (Instant Rollback). |
| **Weaknesses (Zayıf Yönler)** | Eski modeller arşivlendiği için disk depolama yönetimi gerekir; Dağıtık sunucularda senkronizasyon için merkezi DB şarttır. |
| **Opportunities (Fırsatlar)** | CI/CD (GitHub Actions / GitLab CI) ile tam otomatik Staging deploy; Canary / Shadow deployment ile A/B trafik testleri. |
| **Threats (Tehditler)** | Kalite kapısı eşikleri gevşek tutulursa hatalı modeller sızabilir. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-89-model-registry/`](.) dizinindedir:

### A. Model Kayıt ve Aşama Geçiş Motoru (SQLite Backed)
Dosya: [`src/kayit_motoru.py`](src/kayit_motoru.py)
```python
class ModelKayitMotoru:
    def asama_degistir(self, model_adi: str, surum_no: int, yeni_asama: str, mevcut_uretimi_arsivle: bool = True) -> None:
        yeni_asama = yeni_asama.upper()
        with self.baglanti:
            if yeni_asama == "PRODUCTION" and mevcut_uretimi_arsivle:
                # Mevcut PRODUCTION modelini otomatik ARCHIVED yap
                self.baglanti.execute(
                    "UPDATE model_surumleri SET asama = 'ARCHIVED' WHERE model_adi = ? AND asama = 'PRODUCTION'",
                    (model_adi,)
                )
            # Hedef sürümü PRODUCTION yap
            self.baglanti.execute(
                "UPDATE model_surumleri SET asama = ? WHERE model_adi = ? AND surum_no = ?",
                (yeni_asama, model_adi, surum_no)
            )

    def geri_al(self, model_adi: str) -> Dict[str, Any]:
        # En son arşivlenen stabil sürümü bul ve tekrar PRODUCTION yap
        imlec = self.baglanti.cursor()
        imlec.execute("SELECT surum_no FROM model_surumleri WHERE model_adi = ? AND asama = 'ARCHIVED' ORDER BY surum_no DESC LIMIT 1", (model_adi,))
        onceki = imlec.fetchone()
        self.asama_degistir(model_adi, int(onceki[0]), "PRODUCTION", mevcut_uretimi_arsivle=True)
        return self.uretim_modelini_getir(model_adi)
```

---

## 📊 Deneysel Sonuçlar ve Doğrulama Çıktıları

`ana_akis.py` koşturularak elde edilen Model Registry yaşam döngüsü doğrulaması:

```text
=====================================================================================
🚀 Day 89: Model Kayıt Sistemi, Sürümleme ve Staging/Production Yaşam Döngüsü Laboratuvarı
=====================================================================================
📌 Çalışma Ortamı Cihazı: CUDA

[1/4] Model v1 (İlk Temel Model) Eğitiliyor ve Registry'ye Ekleniyor...
  ✓ Model v1 Kaydedildi! Doğruluk: %100.00
  🚀 Model v1 Başarıyla PRODUCTION Aşamasına Terfi Etti!

[2/4] Model v2 (Gelişmiş Aday Model) Eğitiliyor ve Kalite Kapısına Alınıyor...
  ✓ Model v2 Kalite Kapısı Değerlendirmesi: GEÇTİ ✅
    • Doğruluk: %100.00 (Eşik: >= %90.0)
    • Gecikme : 0.36 ms (Eşik: <= 30.0 ms)
    • ECE     : 0.0681 (Eşik: <= 0.15)
  🚀 Model v2 PRODUCTION Oldu! (Önceki v1 Otomatik ARCHIVED Yapıldı).

[3/4] Model v3 (Kusurlu Aday Model) Eğitiliyor ve Kalite Kapısı Test Ediliyor...
  ✓ Model v3 Kalite Kapısı Değerlendirmesi: REDDEDİLDİ ❌
    • Doğruluk: %10.00 (Eşik Karşılanamadı! Üretime geçiş BLOKE EDİLDİ).

[4/4] Sıfır Kesintili Acil Geri Alma (Instant Rollback) Simülasyonu...
  ⚠️ Canlı ortamda v2 için alarm tetiklendi! Geri alma (Rollback) başlatılıyor...
  ✓ Geri Alma Tamamlandı! Güncel Aktif Üretim Modeli: v1 (PRODUCTION)

✓ 6 Panelli Teşhis Panosu Kaydedildi: ciktilar/model_registry_paneli.png
```

- **Tam Yaşam Döngüsü Denetimi:** v1 prod oldu $\to$ v2 kalite kapısını geçti ve prod oldu (v1 arşivlendi) $\to$ v3 kalite kapısında reddedildi $\to$ v2 alarmı sonrası v1 tek komutla sıfır kesintiyle geri yüklendi.
- **Birim Test Güvencesi:** [`testler/test_model_registry.py`](testler/test_model_registry.py) altındaki **8/8 birim test %100 PASSED (3.79s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/model_registry_paneli.png`](ciktilar/model_registry_paneli.png) konumundadır:

1. **Model Registry & Yaşam Döngüsü Akışı:** Sürümleme, Staging, Production ve Arşivleme diyagramı.
2. **Model Sürümleri ve Yaşam Döngüsü Durum Matrisi:** v1, v2, v3 sürümlerinin anlık durum tablosu.
3. **Kalite Kapısı (Quality Gate) Karşılaştırması:** v2 (Kabul) vs v3 (Red) kriter grafikleri.
4. **Sürüm Doğruluk vs Gecikme Haritası:** Sürümlerin performans ve hız uzayındaki konumu.
5. **Sıfır Kesintili Geri Alma (Rollback) Akışı:** Acil durumda < 10 ms içinde güvenli geri alma süreci.
6. **Model Registry SWOT Karar Matrisi:** Endüstriyel karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Üretim modelini doğrudan değiştirmek yerine gelen trafiği %90 mevcut modele (v1) ve %10 yeni adaya (v2) yönlendirerek canlı metrik toplayan **Canary / A-B Testing Router** sınıfı yazınız.

```python
import random
import torch

class CanaryModelRouter:
    """Canlı trafiği v1 (%90) ve v2 (%10) arasında dinamik paylaştıran yönlendirici."""
    def __init__(self, prod_model: torch.nn.Module, canary_model: torch.nn.Module, canary_orani: float = 0.10):
        self.prod_model = prod_model
        self.canary_model = canary_model
        self.canary_orani = canary_orani

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, str]:
        if random.random() < self.canary_orani:
            return self.canary_model(x), "CANARY_v2"
        return self.prod_model(x), "PRODUCTION_v1"
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Bir MLOps ekibinde yeni eğitilen bir modelin doğrudan üretim ortamına kopyalanması yerine neden mutlaka bir `STAGING` aşamasından ve `Model Kalite Kapısı`ndan (Quality Gate) geçmesi zorunludur?

> **Mentor Cevabı:**
> 1. **Sessiz Bozulmaların (Silent Regressions) Önlenmesi:** Bir model eğitim kümesinde %99 doğruluk alsa bile, kalibrasyon hatası (ECE) çok yüksek olabilir veya çıkarım gecikmesi (latency) üretim SLA limitlerini (ör. < 30 ms) aşabilir. Kalite Kapısı bu gizli darboğazları canlıya geçmeden önce yakalar.
> 2. **Tensör İmzası ve Şema Uyumluluğu:** Modelin beklediği girdi tensör boyutu veya kanal sayısı değişmişse (ör. 3 kanal yerine 1 kanal bekliyorsa), API sunucusu `ValueError` veya çökme (crash) yaşar. Model Registry şema sözleşmesi (schema signature) ile bu uyumsuzluğu %100 engeller.
> 3. **Acil Durum Geri Alma Güvencesi:** Kalite kapısı ve durum makinesi sayesinde canlıdaki model tekil bir kayıt olarak mühürlenir; beklenmeyen bir durumda tek SQL komutuyla önceki stabil sürüme sıfır kesintiyle dönülür.

---

### 📌 Git & Yol Haritası Güncellemesi:
- **Tamamlanan:** Gün 89 (`day-89-model-registry`) başarıyla tamamlandı.
- **Sıradaki Gün:** **Day 90: GPU Verimliliği İçin Kuyruk Tabanlı Dinamik Batching Çıkarım Motoru (`day-90-dynamic-batching-inference`)**.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
