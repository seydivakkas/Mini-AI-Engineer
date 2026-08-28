# Day 88: Optuna ile Otomatik Hiperparametre Optimizasyonu (TPE Algoritması, Pruning)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Optuna 3.2+](https://img.shields.io/badge/Optuna-TPE_%26_Pruning-blue.svg?style=flat-square)](https://optuna.org/)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_optuna_hpo.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin yedinci gününde; James Bergstra et al. (NeurIPS 2011) *"Algorithms for Hyper-Parameter Optimization"* ve Akiba et al. (KDD 2019) *"Optuna: A Next-generation Hyperparameter Optimization Framework"* makaleleri ışığında derin öğrenme modellerinde **Ağaç Yapılı Parzen Tahmincisi (Tree-structured Parzen Estimator - TPE)** ve **Medyan Erken Budama (Median Pruning)** sistemini hem sıfırdan matematiksel olarak hem de endüstriyel Optuna entegrasyonuyla kuruyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Derin öğrenme modellerinde performans (doğruluk, kayıp, genelleme yeteneği) büyük ölçüde seçilen hiperparametrelere (öğrenme oranı, ağırlık sönümü, optimizatör tipi, dropout, mimari kanal genişliği) bağlıdır. Manuel deneme-yanılma veya ilkel arama algoritmaları ciddi darboğazlara yol açar:

1. **Izgara Arama (Grid Search) ve Boyut Laneti ($O(K^D)$):**
   Parametre sayısı arttıkça arama uzayı üstel büyür; gereksiz ve alakasız kombinasyonlarda günlerce GPU süresi heba olur.
2. **Rastgele Arama (Random Search) Verimsizliği:**
   Önceki denemelerin başarısından hiçbir bilgi öğrenmez (geçmişi görmezden gelir).
3. **Bayesyen TPE Örnekleme (Tree-structured Parzen Estimator):**
   Geleneksel Gauss Süreçleri (Gaussian Processes) yerine $p(x|y)$ koşullu olasılığını iki gruba ($\ell(x)$ iyi hiperparametreler ve $g(x)$ kötü hiperparametreler) ayırarak **Beklenen İyileşme (Expected Improvement - EI)** oranını maksimize eder:
   $$EI_{y^*}(x) \propto \left( \gamma + \frac{g(x)}{\ell(x)} (1 - \gamma) \right)^{-1} \iff \max \frac{\ell(x)}{g(x)}$$
4. **Erken Budama (Automated Pruning / MedianPruner):**
   Önceki başarılı koşuların gerisinde kalan umutsuz denemeler 2. veya 3. epokta otomatik olarak durdurulur (`optuna.TrialPruned()`), böylece GPU bütçesinde %60-%80 net tasarruf sağlanır.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Hesaplama Kaynaklarının İsrafını Önleme:**
  Kötü öğrenme oranı veya aşırı ağırlık sönümü nedeniyle ıraksayan (diverge eden) modellerin saatlerce boşuna eğitilmesini engeller.
- **En Uygun Hiperparametre Setinin Matematiksel Keşfi:**
  Karmaşık ve doğrusal olmayan etkileşimleri (ör. Öğrenme Oranı ile Batch Boyutu veya Dropout ile Ağırlık Sönümü ilişkisi) Bayesyen olasılıkla hızla çözer.
- **Hiperparametre Önem Analizi (fANOVA / Permutation Importance):**
  Hangi parametrenin başarı üzerinde ne kadar kritik olduğunu (ör. LR %65 etkili, Dropout %15 etkili) matematiksel olarak raporlar.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Isınma Fazı (Startup Trials):**
  TPE'nin güvenilir bir Parzen yoğunluk tahmini ($\ell(x), g(x)$) kurabilmesi için ilk 5-10 denemede rastgele arama yapması gerekir.
- **Aşırı Agresif Budama Riski (Aggressive Pruner Trap):**
  Eğer warmup adımı çok küçük tutulursa, yavaş ama istikrarlı öğrenen modeller (ör. Cosine Annealing ile eğitilenler) erkenden budanabilir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| HPO Yaklaşımı | Arama Verimliliği | Erken Budama (Pruning) | Sürekli/Kategorik Destek | Dağıtık Çalışma |
|---|---|---|---|---|
| **Optuna TPE + MedianPruner (Bizim Yöntem)** | **ÇOK YÜKSEK (Bayesian)** | **VAR (Asenkron)** | **TAM (Log, Float, Cat)** | **Postgres / Redis** |
| **Grid Search (Izgara Arama)** | Çok Düşük ($O(K^D)$) | Yok | Yalnızca Ayrık | Basit Paralel |
| **Random Search (Rastgele Arama)** | Orta | Yok | Var | Kolay Paralel |
| **Hyperband / ASHA** | Yüksek (Bandit tabanlı) | Var (Successive Halving) | Var | Ray Tune / Optuna |
| **Gaussian Process BO (Scikit-Optimize)** | Yüksek | Yok | Zor (Yalnızca Sürekli) | Tekil Makine |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           TPE (TREE-STRUCTURED PARZEN ESTIMATOR) VE BUDAMA DÖNGÜSÜ                        │
│                                                                                                           │
│       Geçmiş Denemeler: D = {(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}                                     │
│          │                                                                                                │
│          ├──> Quantile Eşik Değeri: y* = Quantile_γ({y_i})                                                │
│          │                                                                                                │
│          ├──> İki Olasılık Yoğunluk Grubu Oluştur:                                                        │
│          │    • ℓ(x) = p(x | y < y*)   (Başarılı Parametrelerin Gauss KDE'si)                             │
│          │    • g(x) = p(x | y >= y*)  (Başarısız Parametrelerin Gauss KDE'si)                            │
│          │                                                                                                │
│          ▼                                                                                                │
│       [ ÖRNEKLEME VE ORAN MAKSİMİZASYONU ]                                                                │
│       x* = argmax_x ( ℓ(x) / g(x) )                                                                       │
│          │                                                                                                │
│          ▼                                                                                                │
│       [ MODELİ EĞİTMEYE BAŞLA ]                                                                           │
│          │                                                                                                │
│          ├──> Epok t: Validation Loss > Median_t ?                                                        │
│          │       │                                                                                        │
│          │       ├── [ EVET ] ──> [ BUDAMA / PRUNE ] (Denemeyi derhal durdur!)                           │
│          │       │                                                                                        │
│          │       └── [ HAYIR ] ──> Eğitime devam et ve En İyi Skoru Güncelle                             │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Beklenen İyileşme (Expected Improvement) ve TPE Yoğunluk Oranı
Minimizasyon problemi için eşik $y^*$ ve $\gamma = P(y < y^*)$ olmak üzere:

$$p(x|y) = \begin{cases} \ell(x) & \text{eğer } y < y^* \\ g(x) & \text{eğer } y \ge y^* \end{cases}$$

Beklenen iyileşme integrali:

$$EI_{y^*}(x) = \int_{-\infty}^{y^*} (y^* - y) p(y|x) dy = \frac{\gamma y^* \ell(x) - \ell(x) \int_{-\infty}^{y^*} P(y) dy}{\gamma \ell(x) + (1-\gamma) g(x)}$$

Bu ifade sadeleştirildiğinde:

$$EI_{y^*}(x) \propto \left( \gamma + \frac{g(x)}{\ell(x)} (1 - \gamma) \right)^{-1}$$

Dolayısıyla $EI(x)$'i maksimize etmek, doğrudan $\frac{\ell(x)}{g(x)}$ oranını maksimize etmeye denktir.

### 2. Medyan Budama Kuralı (Median Pruning Rule)
Bir denemenin $t$. epoktaki ara metriği $v_t$ olsun. Önceki tüm tamamlanmış koşuların $t$. adımdaki medyan değeri $M_t = \text{median}(\{v_{i, t}\})$ ise:

$$\text{Karar}(t) = \begin{cases} \text{PRUNE (DURDUR)} & \text{eğer } v_t > M_t \quad (\text{Minimizasyon için}) \\ \text{DEVAM ET} & \text{eğer } v_t \le M_t \end{cases}$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Hyperparameter Optimization (HPO)**| *Hiperparametre Optimizasyonu* | Model ağırlıkları dışında kalan ve öğrenme sürecini kontrol eden değişkenlerin otomatik ayarlanması. |
| **Tree-structured Parzen Estimator (TPE)**| *Ağaç Yapılı Parzen Tahmincisi* | Bayesyen optimizasyonda parametre uzayını başarılı ($\ell(x)$) ve başarısız ($g(x)$) çekirdek yoğunluklarıyla modelleyen algoritma. |
| **Median Pruning** | *Medyan Budama* | Ara epoklarda önceki denemelerin medyanından kötü performans gösteren koşuların erken sonlandırılması. |
| **Search Space** | *Arama Uzayı* | Optimizatörün keşfedeceği değişkenlerin alt, üst sınırları ve tipleri (log-uniform, float, categorical). |
| **Study & Trial** | *Çalışma ve Deneme* | Bir Optuna optimizasyon projesine "Study", her bir tekil parametre kombinasyonunun denenmesine "Trial" denir. |
| **Parameter Importance** | *Parametre Önemi* | Model başarımındaki varyansın hangi hiperparametreden kaynaklandığını belirleyen fANOVA/SHAP analizi. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Bayesyen TPE örnekleyici ile hızlı ve akıllı arama; MedianPruner ile umutsuz koşuları erken durdurarak %70 tasarruf; Çoklu parametre tiplerini (kategorik, float, log) doğal destekler. |
| **Weaknesses (Zayıf Yönler)** | İlk 5-10 denemede rastgele örnekleme gerektirir (Isınma fazı); Pruner eşiği çok agresifse yavaş öğrenen iyi modeller budanabilir. |
| **Opportunities (Fırsatlar)** | Dağıtık ortamda (PostgreSQL / Redis) çoklu GPU ile paralel HPO; Model Registry ve CI/CD ile en iyi checkpoint'i doğrudan üretime alma. |
| **Threats (Tehditler)** | Arama uzayı (Search Space) çok geniş seçilirse arama süresi uzar. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-88-optuna-hpo/`](.) dizinindedir:

### A. Sıfırdan Matematiksel TPE Örnekleyicisi
Dosya: [`src/tpe_motoru.py`](src/tpe_motoru.py)
```python
class MatematikselTPESampler:
    def ornekle(self, gecmis_x: List[float], gecmis_y: List[float], alt: float, ust: float, log_olcek: bool = False) -> float:
        if len(gecmis_x) < 5:
            return float(np.exp(self.rng.uniform(np.log(alt), np.log(ust)))) if log_olcek else float(self.rng.uniform(alt, ust))

        x_arr, y_arr = np.array(gecmis_x), np.array(gecmis_y)
        if log_olcek:
            x_arr, alt, ust = np.log(x_arr), np.log(alt), np.log(ust)

        esik_y = np.percentile(y_arr, self.gama * 100.0)
        x_iyi, x_kotu = x_arr[y_arr <= esik_y], x_arr[y_arr > esik_y]

        kde_iyi = gaussian_kde(x_iyi, bw_method="scott")
        kde_kotu = gaussian_kde(x_kotu, bw_method="scott")

        adaylar = np.clip(kde_iyi.resample(self.aday_sayisi)[0], alt, ust)
        oranlar = (kde_iyi.evaluate(adaylar) + 1e-9) / (kde_kotu.evaluate(adaylar) + 1e-9)
        en_iyi_aday = adaylar[np.argmax(oranlar)]

        return float(np.exp(en_iyi_aday)) if log_olcek else float(en_iyi_aday)
```

---

## 📊 Deneysel Sonuçlar ve Doğrulama Çıktıları

`ana_akis.py` koşturularak elde edilen 16 denemelik Optuna TPE & Pruning optimizasyonu:

```text
=====================================================================================
🚀 Day 88: Optuna ile Otomatik Hiperparametre Optimizasyonu (TPE & Pruning) Laboratuvarı
=====================================================================================
📌 Çalışma Ortamı Cihazı: CUDA

[1/3] Optuna TPE Arama Uzayı ve MedianPruner Başlatılıyor...
[2/3] 16 Denemelik TPE Bayesyen Optimizasyonu Koşuluyor...

======================================================================
🏆 OPTUNA HPO ÇALIŞMA SONUÇLARI VE ŞAMPİYON HİPERPARAMETRELER
======================================================================
  Toplam Deneme Sayısı   : 16
  Tamamlanan Denemeler   : 7
  Erken Budanan Denemeler: 9 (Hesaplama Tasarrufu: %56.2)
  En İyi Validation Loss : 0.0226
----------------------------------------------------------------------
  🥇 ŞAMPİYON HİPERPARAMETRELER:
     • lr             : 0.009791
     • optimizer      : adam
     • weight_decay   : 0.000486
     • taban_kanal    : 16
     • dropout        : 0.100000
======================================================================

✓ 6 Panelli Teşhis Panosu Kaydedildi: ciktilar/optuna_hpo_paneli.png
```

- **%56.2 Hesaplama Tasarrufu:** MedianPruner sayesinde 16 denemenin 9'u daha 2. ve 3. epokta otomatik durdurulmuş, GPU süresi doğrudan umut vadeden modellere tahsis edilmiştir.
- **Birim Test Güvencesi:** [`testler/test_optuna_hpo.py`](testler/test_optuna_hpo.py) altındaki **8/8 birim test %100 PASSED (5.92s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/optuna_hpo_paneli.png`](ciktilar/optuna_hpo_paneli.png) konumundadır:

1. **TPE ve Medyan Budama (Pruning) Mimarisi:** Matematiksel $\ell(x)/g(x)$ oranı ve erken budama kuralı.
2. **Optimizasyon Geçmişi (Optimization History):** Her denemenin kaybı ve en iyi kayıp ilerleme eğrisi.
3. **Deneme Durumları Pasta Grafiği:** Tamamlanan (%43.8) vs Erken Budanan (%56.2) deneme dağılımı.
4. **Hiperparametre Önem Dereceleri:** Parametrelerin doğrulama kaybı üzerindeki etki oranları.
5. **Öğrenme Oranı vs Validation Loss:** Log ölçekte LR dağılımı ve optimizer karşılaştırması.
6. **Optuna HPO & TPE SWOT Karar Matrisi:** Endüstriyel karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Birden fazla hedefi (ör. hem Doğrulama Kaybını minimize etmek hem de Model Parametre Sayısını minimize etmek) aynı anda optimize eden **Çok Hedefli (Multi-Objective) Optuna NSGA-II Çalışması** yazınız.

```python
import optuna

def cok_hedefli_objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    taban_kanal = trial.suggest_categorical("taban_kanal", [16, 32, 64])
    
    # 1. Hedef: Validation Loss (Minimize)
    val_loss = 0.05 + (1.0 / taban_kanal) + abs(np.log10(lr) + 3) * 0.1
    # 2. Hedef: Parametre Sayısı (Minimize)
    param_count = taban_kanal * 1500
    
    return val_loss, param_count

# Multi-objective Study (Pareto Cephesi üretir)
study_mo = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler(seed=42)
)
study_mo.optimize(cok_hedefli_objective, n_trials=20)
print(f"Pareto Optimal Deneme Sayısı: {len(study_mo.best_trials)}")
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden klasik Bayesyen Optimizasyondaki Gauss Süreçleri (Gaussian Processes) yerine TPE (Tree-structured Parzen Estimator) algoritması modern derin öğrenme hiperparametre optimizasyonunda çok daha yaygın tercih edilir?

> **Mentor Cevabı:**
> 1. **Kübik Zaman Karmaşıklığı Engeli ($O(N^3)$):** Standart Gauss Süreçleri (GP), $N$ adet deneme için kovaryans matrisinin tersini alırken $O(N^3)$ işlem karmaşıklığına sahiptir. Deneme sayısı 50-100'ü geçtiğinde GP inanılmaz derecede yavaşlar. TPE ise Parzen penceresiyle $O(N)$ doğrusal zamanda örnekleme yapar.
> 2. **Kategorik ve Koşullu Değişken Esnekliği:** Derin öğrenmede hiperparametreler genellikle hiyerarşik veya kategoriktir (ör. `optimizer = 'sgd'` ise `momentum` parametresi aktif olur; `adamw` ise `beta1, beta2` aktif olur). GP bu tip karmaşık arama uzaylarında zorlanırken, TPE ağaç yapısı sayesinde koşullu ve kategorik parametreleri doğal olarak modeller.
> 3. **Erken Budama ile Kusursuz Uyum:** TPE, Optuna'nın Asynchronous Successive Halving (ASHA) ve MedianPruner mekanizmalarıyla tam uyumlu çalışarak kötü denemeleri ilk adımlarda eler.

---

### 📌 Git & Yol Haritası Güncellemesi:
- **Tamamlanan:** Gün 88 (`day-88-optuna-hpo`) başarıyla tamamlandı.
- **Sıradaki Gün:** **Day 89: Model Kayıt Sistemi, Model Sürümleme, Staging/Production Yaşam Döngüsü (`day-89-model-registry`)**.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
