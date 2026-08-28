# Day 87: MLflow / Weights & Biases ile Merkezi Deney Takibi ve Artefakt Kayıt Sistemi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![MLOps: Experiment Registry & Governance](https://img.shields.io/badge/MLOps-Experiment_Registry-blueviolet.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_experiment_registry.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin altıncı gününde; Zaharia et al. (IEEE Micro 2018) *"Accelerating the Machine Learning Lifecycle with MLflow"* ve Biewald (2020) *"Experiment Tracking with Weights & Biases"* endüstriyel mimari prensipleri doğrultusunda **Merkezi Deney Takibi (Experiment Tracking)**, **Koşu Yaşam Döngüsü (Run Management)**, **Zaman Serisi Metrik Kaydı**, **Artefakt & Model Checkpoint Yönetimi** ve **Liderlik Tablosu (Leaderboard)** sistemini sıfırdan kurup işletiyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Yapay zeka modelleri geliştirilirken yüzlerce farklı hiperparametre kombinasyonu (öğrenme oranı, optimizer, ağırlık sönümü, dropout, katman genişliği, tohum) denenir. Merkezi bir kayıt altyapısı olmadan çalışıldığında:

1. **Teknik Borç ve Deney Kaosu (MLOps Debt):**
   Hangi model ağırlığının (`model_v3_final_really.pt`) hangi hiperparametrelerle, hangi veri kümesi sürümünde ve hangi Git commit'inde eğitildiği kaybolur.
2. **Yeniden Üretilebilirlik Krizi (Reproducibility Crisis):**
   En yüksek doğruluğu veren modelin eğitim dinamikleri, öğrenme oranı çizelgesi ve ara metrikleri geriye dönük incelenemez.
3. **Sistematik Karşılaştırma ve Pareto Seçimi:**
   Sadece doğruluk değil; parametre sayısı, çıkarım gecikmesi (latency) ve eğitim süresi arasındaki Pareto sınırını (Pareto Frontier) belirlemek için merkezi veritabanı sorgusu şarttır.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Kayıp Model Ağırlığı ve Hiperparametre Sorununun Çözümü:**
  Her koşu için parametreler (`params`), epok bazlı zaman serisi metrikleri (`metrics`) ve model artefaktları (`artifacts`) atomik olarak SQLite ve diskte arşivlenir.
- **Takım İçi Şeffaflık ve Liderlik Tablosu (Leaderboard):**
  Tüm modelleri otomatik filtreleyen, en iyi doğrulama başarımı ve en düşük kayıp değerine göre sıralayan dinamik liderlik tablosu üretir.
- **Üretim Dağıtımı Öncesi Denetlenebilirlik (Governance & Lineage):**
  Üretime gidecek modelin başlangıç/bitiş zamanı, cihaz bilgisi ve sürüm soykütüğü (lineage) kayıt altına alınır.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Depolama Alanı Yönetimi (Storage Bloat):**
  Yüzlerce koşu boyunca her epokta büyük model kontrol noktaları (`.pt` / `.bin`) kaydedilirse disk hızla dolabilir; sadece `best_model.pt` veya belirli aralıklarla budama (pruning) yapılmalıdır.
- **Dağıtık Senkronizasyon:**
  Büyük takımlarda yerel SQLite yerine PostgreSQL veya merkezi bir uzaktan sunucu (Remote Tracking Server) kurulmalıdır.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Deney Takip Sistemi | Barındırma | SQL Desteği | Model Registry | Artefakt Yönetimi |
|---|---|---|---|---|
| **Merkezi Takip Motoru (Bizim Sistem)** | **Yerel (SQLite + File Storage)** | **TAM (Gömülü)** | **Var (best_model.pt)** | **Tam (Otomatik Dizinleme)** |
| **MLflow Tracking** | Açık Kaynak / Self-Hosted | Tam (Postgres/MySQL) | Var (Staging/Prod) | S3 / GCS / Local |
| **Weights & Biases (W&B)**| Bulut (SaaS) / Private Cloud | GraphQL / NoSQL | Var | W&B Cloud Artifacts |
| **TensorBoard** | Yerel (Event Files) | Yok (Dosya tabanlı) | Yok | Yalnızca Tensors / Scalars |
| **Neptune.ai** | Bulut (SaaS) | NoSQL Metadata | Var | Neptune Cloud |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       MLFLOW & W&B MERKEZİ DENEY TAKİBİ VE ARTEFAKT MİMARİSİ                              │
│                                                                                                           │
│       Deney Havuzu: Experiment (Ör: "VisionModel_Hiperparametre_Optimizasyonu")                           │
│          │                                                                                                │
│          ├──> Koşu (Run 1): AdamW (lr=1e-3, wd=1e-4)  ──> Params, Metrics(t), best_model.pt               │
│          ├──> Koşu (Run 2): AdamW (lr=3e-4, wd=1e-3)  ──> Params, Metrics(t), best_model.pt               │
│          ├──> Koşu (Run 3): SGD    (lr=1e-2, mom=0.9)  ──> Params, Metrics(t), best_model.pt               │
│          ├──> Koşu (4..N):  ...                                                                           │
│          │                                                                                                │
│          ▼                                                                                                │
│       [ MERKEZİ İLİŞKİSEL VE ARTEFAKT DEPOSU (SQLite + File Storage) ]                                    │
│          │                                                                                                │
│          ├──> SQL Sorgusu & Filtreleme: SELECT * FROM runs WHERE val_acc > 90% ORDER BY val_acc DESC      │
│          ▼                                                                                                │
│       [ LİDERLİK TABLOSU (LEADERBOARD) & PARETO OPTİMAL SEÇİMİ ]                                          │
│       🥇 Şampiyon Model ──> Model Registry ve Üretim Dağıtımına Aktar                                     │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Koşu Karşılaştırma ve Sıralama Fonksiyonu
$N$ adet tamamlanmış koşu kümesi $\mathcal{R} = \{r_1, r_2, \dots, r_N\}$ olsun. Her koşu $r_i$ için amaç fonksiyonu:

$$r^* = \arg\max_{r \in \mathcal{R}} \big( \text{Val\_Acc}(r) \big) \quad \text{veya} \quad r^* = \arg\min_{r \in \mathcal{R}} \big( \text{Val\_Loss}(r) \big)$$

### 2. Pareto Optimal Sınırı (Doğruluk vs Model Karmaşıklığı)
Bir model $r_A$, başka bir model $r_B$'yi ancak ve ancak şu koşulda domine eder:

$$\text{Val\_Acc}(r_A) \ge \text{Val\_Acc}(r_B) \quad \land \quad \text{Param\_Count}(r_A) \le \text{Param\_Count}(r_B)$$

ve en az bir eşitsizlik kesin olmalıdır. Domine edilmeyen tüm modeller **Pareto Optimal** kümesini oluşturur:

$$\mathcal{P} = \{ r \in \mathcal{R} \mid \nexists r' \in \mathcal{R} : r' \succ r \}$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Experiment** | *Deney Havuzu* | Belirli bir araştırma problemi veya model mimarisi için açılan üst düzey mantıksal çalışma alanı. |
| **Run** | *Deney Koşusu* | Tek bir hiperparametre setiyle başlatılan ve bağımsız ID'ye sahip tekil eğitim süreci. |
| **Parameters (Params)** | *Statik Parametreler* | Eğitim başında sabitlenen değişkenler (öğrenme oranı, optimizatör, batch boyutu, tohum). |
| **Metrics** | *Zaman Serisi Metrikleri* | Eğitim boyunca her epok veya adımda kaydedilen sayısal değerler (loss, accuracy, learning rate). |
| **Artifacts** | *Artefaktlar / Çıktılar* | Model ağırlıkları (`.pt`), grafikler, konfigürasyon JSON dosyaları gibi ikili (binary) dosyalar. |
| **Model Lineage** | *Model Soykütüğü* | Bir modelin hangi veriyle, hangi kod revizyonuyla (git commit) ve hangi parametrelerle üretildiğinin tam iz kaydı. |
| **Leaderboard** | *Liderlik Tablosu* | Koşuların performans metriklerine göre sıralandığı dinamik karşılaştırma matrisi. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Sıfır kayıp: Tüm hiperparametreler, metrikler ve modeller kayıtlı; Takım içi şeffaflık, liderlik tablosu ve hızlı model seçimi; Model sürümleme ve artefakt yönetimiyle üretime hazır altyapı. |
| **Weaknesses (Zayıf Yönler)** | Disk ve depolama yönetimi (Her model checkpoint'i yer kaplar); Manuel logging disiplini gerektirir (Otomatik wrapper şart). |
| **Opportunities (Fırsatlar)** | Optuna HPO ve Model Registry aşamalarıyla tam otomatik pipeline; CI/CD süreçlerinde regression test kapıları (Quality Gates). |
| **Threats (Tehditler)** | İzole/untracked yerel script çalıştırma alışkanlığının sürmesi durumunda metadata tutarsızlığı. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-87-experiment-registry/`](.) dizinindedir:

### A. Merkezi Deney Takip Motoru (PyTorch & SQLite)
Dosya: [`src/takip_motoru.py`](src/takip_motoru.py)
```python
class DeneyKosusu:
    def log_param(self, anahtar: str, deger: Any) -> None:
        self.parametreler[anahtar] = deger
        with self.baglanti:
            self.baglanti.execute(
                "INSERT OR REPLACE INTO parametreler (run_id, anahtar, deger) VALUES (?, ?, ?)",
                (self.run_id, anahtar, str(deger))
            )

    def log_metric(self, anahtar: str, deger: float, step: Optional[int] = None) -> None:
        kayit = {"step": step, "value": float(deger), "timestamp": time.time()}
        with self.baglanti:
            self.baglanti.execute(
                "INSERT INTO metrikler (run_id, anahtar, deger, step, timestamp) VALUES (?, ?, ?, ?, ?)",
                (self.run_id, anahtar, float(deger), kayit["step"], kayit["timestamp"])
            )

    def log_model(self, model: torch.nn.Module, model_adi: str = "model.pt") -> str:
        hedef_yol = os.path.join(self.artefakt_dizini, model_adi)
        torch.save(model.state_dict(), hedef_yol)
        return self.log_artifact(hedef_yol)
```

---

## 📊 Deneysel Sonuçlar ve Doğrulama Çıktıları

`ana_akis.py` koşturularak elde edilen deney sonuçları ve Liderlik Tablosu:

```text
=====================================================================================
🚀 Day 87: MLflow / Weights & Biases ile Merkezi Deney Takibi ve Artefakt Kayıt Laboratuvarı
=====================================================================================
📌 Çalışma Ortamı Cihazı: CUDA

[1/3] 5 Farklı Model Koşusu Başlatılıyor ve Merkezi Depoya Loglanıyor...
  ▶ Koşu 1/5: AdamW_lr_1e3_std (Opt: adamw, LR: 0.001) ──> Val Doğruluğu: %100.00
  ▶ Koşu 2/5: AdamW_lr_3e4_decay (Opt: adamw, LR: 0.0003) ──> Val Doğruluğu: %100.00
  ▶ Koşu 3/5: SGD_Momentum_lr_1e2 (Opt: sgd, LR: 0.01) ──> Val Doğruluğu: %100.00
  ▶ Koşu 4/5: Adam_WideNet_lr_1e3 (Opt: adam, LR: 0.001) ──> Val Doğruluğu: %100.00
  ▶ Koşu 5/5: AdamW_lr_5e3_HighLR (Opt: adamw, LR: 0.005) ──> Val Doğruluğu: %100.00

[2/3] Merkezi Veritabanından Tüm Koşular Çekiliyor ve Liderlik Tablosu Oluşturuluyor...

================================================================================
🏆 MERKEZİ DENEY LİDERLİK TABLOSU (LEADERBOARD)
================================================================================
           run_name p_optimizer  p_learning_rate  p_param_count  m_val_acc  m_val_loss  sure_sn
   AdamW_lr_1e3_std       adamw           0.0010        94986.0      100.0    0.225259     2.41
 AdamW_lr_3e4_decay       adamw           0.0003        94986.0      100.0    1.088816     0.63
SGD_Momentum_lr_1e2         sgd           0.0100        94986.0      100.0    0.244488     0.63
Adam_WideNet_lr_1e3        adam           0.0010       211594.0      100.0    0.158735     0.75
AdamW_lr_5e3_HighLR       adamw           0.0050        94986.0      100.0    0.009275     0.71
================================================================================
  🥇 Şampiyon Model: AdamW_lr_1e3_std (Val Acc: %100.00)

✓ 6 Panelli Teşhis Panosu Kaydedildi: ciktilar/experiment_registry_paneli.png
```

- **Tam İzlenebilirlik:** 5 farklı model koşusunun tüm hiperparametreleri, epok epok loss ve accuracy eğrileri ile en iyi model ağırlıkları atomik olarak kaydedilmiştir.
- **Birim Test Güvencesi:** [`testler/test_experiment_registry.py`](testler/test_experiment_registry.py) altındaki **8/8 birim test %100 PASSED (3.89s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/experiment_registry_paneli.png`](ciktilar/experiment_registry_paneli.png) konumundadır:

1. **Merkezi Deney Takip & MLOps Mimarisi:** Experiment, Run, Params, Metrics ve Artifacts hiyerarşisi.
2. **Doğrulama Kayıp (Val Loss) Zaman Serisi:** 5 farklı model koşusunun epok bazında kayıp eğrileri.
3. **Doğrulama Başarım (Val Accuracy %) Zaman Serisi:** Koşuların başarı gelişim eğrileri.
4. **Model Verimliliği & Pareto Cephesi:** Parametre sayısı vs Doğruluk dağılım grafiği.
5. **Model Liderlik Tablosu (Leaderboard):** En yüksek doğruluk ve en düşük kayıptan sıralanan lider modeller.
6. **MLOps Deney Takibi SWOT Karar Matrisi:** Endüstriyel karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Her model koşusunda veri setinin hash özetini (MD5/SHA256) ve aktif Git commit SHA kodunu otomatik tespit edip loglayan **Model Lineage & Data Checksum Logger** yazınız.

```python
import hashlib
import subprocess

def git_commit_sha_al() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "UNKNOWN_COMMIT"

def veri_kumesi_hash_hesapla(veri_tensor: torch.Tensor) -> str:
    """Veri tensörünün byte diziliminden SHA256 checksum üretir."""
    sha = hashlib.sha256()
    sha.update(veri_tensor.numpy().tobytes())
    return sha.hexdigest()[:12]
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** MLOps dünyasında neden ham log dosyaları (log.txt) veya basit print çıktıları yerine MLflow / W&B gibi yapılandırılmış bir "Merkezi Deney Takip Sistemi" (Experiment Registry) kullanılır?

> **Mentor Cevabı:**
> 1. **Yapılandırılmış ve İlişkisel Sorgulama:** Ham metin loglarında *"Öğrenme oranı 1e-3 olup doğruluk değeri %90'ın üzerinde olan ve AdamW ile eğitilen en iyi 3 modeli getir"* sorgusunu yapamazsınız. MLflow/W&B bu verileri ilişkisel (SQLite/Postgres) ve indekslenmiş olarak tutarak anında SQL veya API sorgusu yapılmasına olanak tanır.
> 2. **Artefakt ve Model Eşleşmesi (Lineage Integrity):** Disk üzerinde serbest duran bir `.pt` dosyasının hangi parametrelerle üretildiğini bilemezsiniz. Experiment Registry, model ağırlık dosyasını doğrudan ilgili koşunun UUID'si ve hiperparametreleriyle ilişkilendirir.
> 3. **Ekip İçi İşbirliği ve Karar Alma:** Liderlik tablosu (Leaderboard) ve Pareto analizi sayesinde takım üyeleri en verimli modeli saniyeler içinde belirler.

---

### 📌 Git & Yol Haritası Güncellemesi:
- **Tamamlanan:** Gün 87 (`day-87-experiment-registry`) başarıyla tamamlandı.
- **Sıradaki Gün:** **Day 88: Optuna ile Otomatik Hiperparametre Optimizasyonu (TPE Algoritması, Pruning) (`day-88-optuna-hpo`)**.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
