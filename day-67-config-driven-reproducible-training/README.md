# Day 67: YAML/Hydra ile Konfigürasyon Yönetimi, Deterministik & Tekrarlanabilir Eğitim (FAZ 4 Giriş)

[![License: Private All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C.svg)](https://pytorch.org/)
[![PyYAML](https://img.shields.io/badge/PyYAML-6.0.3-CB171E.svg)](https://pyyaml.org/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.13-E92063.svg)](https://docs.pydantic.dev/)
[![Tests: 8 Passed](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)](testler/)

**FAZ 4: İleri Düzey Eğitim, Temsil Öğrenimi ve Sıfırdan Vision Transformer** serimizin 67. gününde; derin öğrenme deneylerinin kod bağımlılığından kurtarılıp **YAML + Pydantic v2** hiyerarşik konfigürasyon dosyalarıyla yönetilmesi, **%100 bit-for-bit deterministik ve tekrarlanabilir (reproducible)** eğitim boru hatlarının kurulması ve rastgelelik kaynaklarının sıfırlanması uçtan uca uygulanmıştır.

---

## 1. 🎯 Günün Konusu & Teorik/Matematiksel Derinlik

### A. Derin Öğrenmede Tekrarlanabilirlik Krizi (Reproducibility Crisis)
Akademik ve endüstriyel yapay zeka projelerinde en büyük zorluklardan biri, aynı kod ve aynı veri setiyle eğitilen modellerin farklı zamanlarda veya farklı makinelerde **farklı doğruluk ve ağırlık değerleri** üretmesidir. Bu durumun temel kaynakları:
1. **Çoklu Rastgelelik Kaynakları (Multiple RNG Sources):** Python `random`, `numpy.random`, `torch.manual_seed` ve `torch.cuda.manual_seed_all` bağımsız psödo-rastgele sayı üreteçlerine (PRNG) sahiptir. Birinin bile tohumlanmaması tüm döngüyü bozar.
2. **Çoklu İşlemcili (Multi-worker) DataLoader Tohumlanması:** PyTorch DataLoader `num_workers > 0` iken ana sürecin tohumunu her worker'a aynen kopyalar. `worker_init_fn` tanımlanmazsa tüm alt iş parçacıkları aynı veri artırma (augmentation) sırasını üreterek veri çeşitliliğini çökertebilir.
3. **Donanım ve CUDA/cuDNN Optimizasyonları:** NVIDIA cuDNN, konvolüsyon algoritmalarını seçerken (`benchmark = True`) en hızlı atomik paralel toplama sırasını dener. Kayan noktalı sayılarda toplama işleminin birleşme özelliği ($\mathbb{R}$ Associativity: $(a+b)+c \neq a+(b+c)$) bulunmadığından paralel toplama sırası değiştikçe sayısal sapmalar birikir.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    DETERMİNİSTİK VE KONFİGÜRASYON ODAKLI EĞİTİM (CONFIG-DRIVEN MLOPS)                     │
│                                                                                                           │
│  [config.yaml] ──► [KonfigurasyonYoneticisi (Pydantic v2)] ──► [Tip Güvenli KokKonfigurasyon]             │
│                              │                                             │                              │
│                              ▼                                             ▼                              │
│               [DeterminizmYoneticisi (Seed Lock)]           [ModulerVisionNet (Dinamik Katman)]           │
│               ├── Python: random.seed(S)                                   │                              │
│               ├── NumPy: np.random.seed(S)                                 ▼                              │
│               ├── PyTorch: torch.manual_seed(S)             [TekrarlanabilirEgitici]                      │
│               ├── CUDA: CUBLAS_WORKSPACE_CONFIG             ├── Generator Shuffling                        │
│               └── cuDNN: deterministic=True, benchmark=False ├── Cosine LR Scheduling                    │
│                                                             └── Gradient Clipping                         │
│                                                                            │                              │
│                                                                            ▼                              │
│  [Run A (Seed=42)] ──────────► [DeterminizmDogrulayici] ◄───────── [Run B (Seed=42)]                      │
│                                            │                                                              │
│                                            ├──► Delta Loss: 0.0000000000 (SIFIR HATA)                     │
│                                            ├──► Delta Val Loss: 0.0000000000 (SIFIR HATA)                 │
│                                            └──► Ağırlık SHA256: %100 BİREBİR EŞİT                         │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### B. Matematiksel Formülasyon: Deterministik Gradyan Güncellemesi

#### 1. Deterministik Ağırlık Yörüngesi (Deterministic Weight Trajectory)
Bir sinir ağının $t$. adımdaki parametre tensörü $W_t$, veri mini-batch'i $\mathcal{B}_t(S)$ ve öğrenme oranı $\eta_t$ olsun:

$$W_{t+1} = W_t - \eta_t \cdot \nabla_W \mathcal{L}(W_t; \mathcal{B}_t(S)) - \eta_t \cdot \lambda W_t$$

Eğer başlangıç ağırlıkları $W_0^{(A)} = W_0^{(B)}$ ve tohum $S_A = S_B = S$ olarak kilitlenirse; deterministik bir hesaplama grafiğinde tüm $t \ge 0$ adımları için:

$$\|W_t^{(A)} - W_t^{(B)}\|_\infty = 0, \quad \mathcal{L}_t^{(A)} = \mathcal{L}_t^{(B)}$$

#### 2. Cosine Annealing Öğrenme Oranı Zamanlayıcısı
$T_{\text{max}}$ toplam periyodu, $\eta_{\text{max}}$ başlangıç ve $\eta_{\text{min}}$ taban öğrenme oranı olmak üzere:

$$\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t}{T_{\text{max}}} \pi\right)\right)$$

Bu zamanlayıcı, gradyan inişinin eğitim sonlarına doğru yerel minimumların tabanına kararlı ve salınımsız biçimde yerleşmesini sağlar.

---

### C. 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Seed / Tohum** | *Random Seed* | Psödo-rastgele sayı üreteçlerinin (PRNG) ürettiği deterministik matematiksel sayı dizisinin başlangıç indisidir. |
| **PRNG** | *Pseudorandom Number Generator* | Deterministik algoritmalarla (ör. Mersenne Twister) rastgele görünümlü ancak tohumu bilindiğinde $\%100$ tekrarlanabilen sayılar üreten matematiksel mekanizma. |
| **cuDNN Determinism** | *Deterministic cuDNN Flag* | NVIDIA cuDNN kütüphanesinin atomik değişken paralel toplama yapan hızlı algoritmaları yerine, her seferinde sabit işlem sırası takip eden deterministik algoritmaları zorlamasıdır (`torch.backends.cudnn.deterministic = True`). |
| **cuDNN Benchmark** | *cuDNN Benchmark Flag* | İlk mini-batch'te mevcut GPU donanımı için en hızlı evrişim algoritmasını arayan profilleme mekanizmasıdır. Determinizm gerektiğinde mutlaka kapatılmalıdır (`benchmark = False`). |
| **CUBLAS Workspace** | *CUBLAS Workspace Config* | CUDA matris çarpımlarında (GEMM) deterministik algoritmaların çalışabilmesi için gereken bellek havuzudur (`os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`). |
| **Worker Init Function** | *Worker Initialization Function* | Çoklu CPU çekirdeğinde çalışan DataLoader alt süreçlerinin her birine farklı ve öngörülebilir bir tohum atayan fonksiyondur. |
| **PyTorch Generator** | *torch.Generator()* | Modelin veya DataLoader'ın genel global rastgelelik durumundan bağımsız, izole bir rastgele sayı üreteci nesnesidir. |
| **Config Overrides** | *Configuration Overrides* | Ana YAML dosyasını değiştirmeden, komut satırından `egitim.tohum=99` şeklinde noktasal parametre ezme işlemidir. |
| **Pydantic Coercion** | *Schema Type Coercion* | String olarak gelen `"42"` veya `"true"` gibi konfigürasyon değerlerinin Pydantic tarafından otomatik olarak `int` veya `bool` tiplerine dönüştürülmesi ve aralık kontrollerinin yapılmasıdır. |
| **State Hash (SHA256)** | *Weight State Hash* | Modelin tüm katmanlarındaki ağırlık tensörlerinin bayt dizisi üzerinden SHA-256 kriptografik özeti çıkarılarak iki modelin bit-for-bit eşitliğini doğrulama tekniğidir. |
| **Gradient Clipping** | *Gradient Norm Clipping* | Patlayan gradyanları (Exploding Gradients) önlemek için gradyan vektörünün $L_2$ normunu maksimum bir $C$ eşiğine sınırlandırmadır: $g \leftarrow g \cdot \min(1, \frac{C}{\|g\|_2})$. |

---

### D. SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | $\%100$ bilimsel tekrarlanabilirlik; hata ayıklama ve model regresyon testlerinde kesin kararlılık; Pydantic v2 ile çalışma zamanı öncesi tam tip güvenliği; YAML ile kod bağımsız deney yönetimi. |
| **Weaknesses (Zayıf Yönler)** | cuDNN deterministik modu sebebiyle GPU eğitim süresinde $\%5-\%10$ seviyesinde hafif hız kaybı; bazı özel CUDA operasyonlarının (ör. scatter/gather) deterministik desteğinin kısıtlı olması. |
| **Opportunities (Fırsatlar)** | Takım içi paylaşılan deneylerde birebir aynı sonuçları elde etme; Optuna / Ray Tune gibi hiperparametre arama araçlarıyla pürüzsüz entegrasyon; MLOps CI/CD boru hatlarında otomatik model doğrulama. |
| **Threats (Tehditler)** | Tohumlanmamış üçüncü parti kütüphanelerin (ör. OpenCV rastgele fonksiyonları) sessizce determinizmi bozması; GPU mimarisi değiştiğinde (ör. Ampere'den Hopper'a) donanımsal kayan nokta farkları. |

---

## 2. 💻 Üretim Seviyesinde Uygulama Mimarisi

Geliştirilen paket [`day-67-config-driven-reproducible-training/`](file:///c:/Users/seydieryilmaz/Desktop/Github%20Mini%20AI%20Engineer/day-67-config-driven-reproducible-training) dizinindedir:

- [`konfigurasyonlar/varsayilan_egitim.yaml`](konfigurasyonlar/varsayilan_egitim.yaml): Üretim ortamı tam eğitim parametreleri.
- [`src/konfigurasyon_semasi.py`](src/konfigurasyon_semasi.py): Pydantic v2 hiyerarşik şemaları (`VeriKonfigurasyonu`, `ModelKonfigurasyonu`, `OptimizerKonfigurasyonu`, `SchedulerKonfigurasyonu`, `EgitimKonfigurasyonu`, `KokKonfigurasyon`).
- [`src/konfigurasyon_yoneticisi.py`](src/konfigurasyon_yoneticisi.py): `KonfigurasyonYoneticisi` (YAML yükleme, noktasal override ayrıştırma, YAML kaydetme).
- [`src/determinizm_motoru.py`](src/determinizm_motoru.py): `DeterminizmYoneticisi` (Python, NumPy, PyTorch, cuDNN, CUBLAS, Worker tohum kilitleri).
- [`src/model_mimari.py`](src/model_mimari.py): `ModulerVisionNet` (Konfigürasyondan dinamik türeyen Residual CNN ve SHA256 ağırlık özeti çıkarıcı).
- [`src/egitim_motoru.py`](src/egitim_motoru.py): `TekrarlanabilirEgitici` (Deterministik eğitim, Cosine Scheduler, Gradient Clipping, metrik izleyici).
- [`src/deney_dogrulayici.py`](src/deney_dogrulayici.py): `DeterminizmDogrulayici` (Run A vs Run B $\Delta \equiv 0$ bit-for-bit matematiksel doğrulayıcı).
- [`src/gorsellestirici.py`](src/gorsellestirici.py): `DeterminizmGorsellestirici` (6 panelli görsel teşhis panosu üreticisi).
- [`ana_akis.py`](ana_akis.py): Uçtan uca konfigürasyon yükleme, determinizm testi ve raporlama betiği.
- [`testler/test_config_ve_determinizm.py`](testler/test_config_ve_determinizm.py): 8 kapsamlı birim testi (%100 Başarı).

---

## 3. 🧪 Günün Alıştırması & Zorlu Görevi (Hands-on Challenge)

**Görev:** Birden fazla deneyi farklı tohumlarla veya farklı hiperparametre kombinasyonlarıyla otomatik olarak yürüten, her deneyin konfigürasyonunu ve sonuç metriklerini JSON/YAML olarak arşivleyen bir `CokluDeneyKosuYurutucu` sınıfı geliştirmek.

**Eksiksiz Kod Çözümü:**
```python
from typing import List, Dict, Any
import os
import json
from src.konfigurasyon_yoneticisi import KonfigurasyonYoneticisi
from src.egitim_motoru import TekrarlanabilirEgitici

class CokluDeneyKosuYurutucu:
    """Farklı tohum veya hiperparametrelerle toplu deney koşturur ve sonuçları arşivler."""

    @classmethod
    def toplu_deney_kos(
        cls,
        ana_yaml_yolu: str,
        tohum_listesi: List[int] = [42, 100, 2026],
        cikti_dizini: str = "deney_arsivi"
    ) -> List[Dict[str, Any]]:
        os.makedirs(cikti_dizini, exist_ok=True)
        sonuclar = []

        for tohum in tohum_listesi:
            cfg = KonfigurasyonYoneticisi.yaml_yukle(ana_yaml_yolu, override_listesi=[f"egitim.tohum={tohum}"])
            egitici = TekrarlanabilirEgitici(cfg)
            res = egitici.egit()

            kayit = {
                "tohum": tohum,
                "son_train_loss": res["son_train_loss"],
                "son_val_loss": res["son_val_loss"],
                "son_val_accuracy": res["son_val_accuracy"],
                "agirlik_hash": res["son_agirlik_hashi"]
            }
            sonuclar.append(kayit)

            dosya_adi = os.path.join(cikti_dizini, f"deney_seed_{tohum}.json")
            with open(dosya_adi, "w", encoding="utf-8") as f:
                json.dump(kayit, f, indent=2)

        return sonuclar
```

---

## 4. 📊 Ölçülen Doğrulama ve Determinizm Metrikleri

`ana_akis.py` koşturularak ölçülen deneysel sonuçlar:

| Koşu | Tohum (Seed) | Son Eğitim Kaybı | Son Doğrulama Kaybı | Doğrulama Başarımı (%) | Ağırlık SHA256 Özeti |
|---|---|---|---|---|---|
| **Run A (Hedef)** | $42$ | **$0.938059$** | **$1.760205$** | **$\%22.00$** | `23513a346a32617264ef7effc2b8...` |
| **Run B (Tekrar)** | $42$ | **$0.938059$** | **$1.760205$** | **$\%22.00$** | `23513a346a32617264ef7effc2b8...` |
| **Run C (Farklı Tohum)**| $99$ | $0.960547$ | $1.800597$ | $\%20.00$ | `7ad2748683e8217088cc79cfc8c6...` |

- **Maksimum Eğitim Kaybı Farkı ($\Delta \mathcal{L}_{\text{train}}$):** **$0.0000000000$ (SIFIR HATA)**
- **Maksimum Doğrulama Kaybı Farkı ($\Delta \mathcal{L}_{\text{val}}$):** **$0.0000000000$ (SIFIR HATA)**
- **Ağırlık Tensörleri SHA256 Eşleşmesi:** **%100 BİREBİR EŞİT**
- **Birim Test Başarımı:** **$8 / 8$ PASSED (%100 Başarı)**

---

## 5. 🚀 Kurulum ve Çalıştırma

```bash
# 1. Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 2. Ana deterministik eğitim ve doğrulama akışını çalıştırın
python ana_akis.py

# 3. Birim testleri koşun
pytest testler -v
```

---

## 6. ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** PyTorch'ta `torch.manual_seed(42)` çağrılmasına rağmen, `DataLoader` içinde `num_workers > 0` ve `shuffle = True` yapıldığında veri artırma adımlarının non-deterministik davranmasının veya tüm worker'ların aynı rastgele diziyi üretmesinin sebebi nedir? Nasıl çözülür?

> **Mentor Cevabı:**
> 1. **Worker Çatallanma (Forking) Problemi:** İşletim sistemi `num_workers` kadar alt işlem (process) başlattığında, her worker ana sürecin o anki RNG durumunu klonlar. Eğer rastgelelik tohumu alt süreç bazında ayrıştırılmazsa, tüm worker'lar eşzamanlı olarak aynı veri artırma dönüşümlerini (örneğin hepsi aynı 30 derecelik açıyla döndürme) uygular.
> 2. **Çözüm (Worker Init Function + PyTorch Generator):**
>    - `DataLoader` oluşturulurken izole bir `torch.Generator` atanır: `DataLoader(..., generator=g, worker_init_fn=worker_init_fn)`.
>    - `worker_init_fn` içinde `worker_seed = torch.initial_seed() % 2**32` hesaplanarak `np.random.seed(worker_seed)` ve `random.seed(worker_seed)` kilitlenir. Böylece her worker deterministik ama birbirinden bağımsız bir rastgelelik akışına sahip olur.

---

## 7. 📜 Lisans & Metaveri

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
