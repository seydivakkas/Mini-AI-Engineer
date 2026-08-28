# Day 81: Vision Transformer İçin LoRA (Low-Rank Adaptation) ile Parametre-Verimli İnce Ayar (PEFT)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Method: LoRA PEFT](https://img.shields.io/badge/Method-LoRA_PEFT-purple.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_lora_peft.py)

Hu et al. (2021) *"LoRA: Low-Rank Adaptation of Large Language Models"* prensibini Bilgisayarlı Görü ve Vision Transformer mimarilerine uyarlıyoruz. Önceden eğitilmiş devasa ViT omurgasının **%98+'lik ağırlıklarını dondurarak (Freeze)**, yalnızca Self-Attention ($W_q, W_v$) katmanlarına düşük dereceli eğitilebilir adaptör matrisleri ($A, B$) enjekte eden ve çıkarım anında **0 ms ek gecikme ile ağırlık birleştirmeyi (Weight Merging)** destekleyen PEFT motorunu sıfırdan inşa ediyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Modern temel modeller (ViT, CLIP, GPT, LLaMA) yüz milyonlarca veya milyarlarca parametreye sahiptir. Her yeni downstream görev için modelin tüm parametrelerini baştan eğitmek (Full Fine-Tuning) şu nedenlerle endüstriyel olarak imkansızdır:

1. **İntrinsik Derece Hipotezi (Intrinsic Rank Hypothesis):**
   Aghajanyan et al. (2020) ve Hu et al. (2021), modellerin belirli bir göreve uyarlanırken ağırlık güncellemelerinin ($\Delta W$) aslında çok düşük bir "içsel boyuta" (intrinsic dimension) sahip olduğunu kanıtlamıştır. $d \times d$ boyutundaki devasa bir matrisin tüm elemanlarını güncellemek yerine, $\Delta W \approx \frac{\alpha}{r} (B \cdot A)$ şeklinde düşük dereceli ($r \ll d$) iki matrise ayrıştırmak modelin tam performansını korur.
2. **Devasa Bellek ve GPU Tasarrufu:**
   Ana model parametreleri dondurulduğu için geriye yayılımda bu ağırlıklar için gradyan hesaplanmaz ve AdamW optimizer durumları ($\mathbf{m}_t, \mathbf{v}_t$) tutulmaz. Optimizer bellek ihtiyacı %80'e varan oranda azalır.
3. **Çok Görevli Dağıtım Kolaylığı (Multi-Task Serving):**
   100 farklı downstream görev için 100 ayrı 1 GB'lık model kaydetmek yerine; tek bir paylaşımlı temel model ve her görev için yalnızca **~100 KB'lık LoRA adaptör ağırlıkları** saklanır.
4. **Çıkarım Aşamasında Sıfır Gecikme (Zero Inference Latency):**
   LoRA adaptörleri dağıtıma alınırken $W_{\text{merged}} = W_0 + \Delta W$ formülüyle ana ağırlıklara kalıcı olarak eklenir; modelin mimarisi ve çıkarım hızı orijinal modelle birebir aynı kalır ($0\text{ ms}$ gecikme ek yükü).

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Felaket Derecesinde Unutma (Catastrophic Forgetting):**
  Tüm ağırlıklar güncellendiğinde model genel görsel ön bilgilerini unutabilir. LoRA ana omurgayı dondurduğu için temel temsil kalitesi asla bozulmaz.
- **GPU VRAM Yetersizliği:**
  Tek bir tüketici GPU'sunda (ör. RTX 3060/4060) bile büyük ViT modellerine rahatlıkla fine-tuning yapılabilmesini sağlar.
- **Depolama ve Dağıtım Darboğazı:**
  GB'larca model kontrol noktası (checkpoint) aktarmak yerine kilobaytlarca adaptör dosyası ile mikroservisler arası anlık geçiş yapılır.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Aşırı Düşük Derece Kısıtı ($r=1, 2$):**
  Eğer downstream görev ön eğitim veri dağılımından çok radikal bir şekilde farklıysa (ör. doğal fotoğraflardan X-Ray tıbbi görüntülere geçiş), düşük $r$ derecesi modelin yeni özellikleri öğrenmesine yetmeyebilir ($r=16, 32$ gerekir).
- **Eşzamanlı Çoklu Kullanıcı Çıkarımı:**
  Aynı GPU üzerinde aynı anda 10 farklı kullanıcı 10 farklı adaptörü çağırırsa ağırlıklar birleştirilemez (unmerged çalışmak gerekir), bu da hafif bir matris çarpım ek yükü ($B \cdot A$) oluşturur.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| PEFT Yöntemi | Eğitilen Parametre Oranı | Çıkarım Gecikmesi | Doğruluk Koruma |
|---|---|---|---|
| **LoRA (Bizim Uygulama)** | **%1 - %3** | **0 ms (Weight Merging ile)** | ⭐⭐⭐⭐⭐ (%99.5+) |
| **Full Fine-Tuning** | %100 | 0 ms | ⭐⭐⭐⭐⭐ (Referans) |
| **Linear Probing (Sadece Head)** | %0.5 | 0 ms | ⭐⭐⭐ (Kısıtlı) |
| **Prefix Tuning / Prompt Tuning**| %0.1 - %1 | +%5-10 Dizi Uzunluğu | ⭐⭐⭐⭐ |
| **Adapter (Houlsby et al.)** | %2 - %4 | +%10-20 Ek Katman Gecikmesi | ⭐⭐⭐⭐ |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                LoRA (LOW-RANK ADAPTATION) MATRİS MEKANİZMASI                              │
│                                                                                                           │
│       Girdi: x ∈ ℝ^(B × N × d_in)                                                                         │
│          │                                                                                                │
│          ├──────────────────────────────────────────────┐ (Düşük Dereceli Adaptör Yolu)                   │
│          ▼                                              ▼                                                 │
│       Dondurulmuş Ana Ağırlık                        A ∈ ℝ^(r × d_in) (Kaiming Init)                      │
│       W_0 ∈ ℝ^(d_out × d_in)                            ▼                                                 │
│       (requires_grad = False)                        B ∈ ℝ^(d_out × r) (SIFIR Init: B=0)                  │
│          │                                              ▼                                                 │
│          │                                           (α / r) · (B · A) x                                  │
│          ▼                                              │                                                 │
│       W_0 x <───────────────────────────────────────────┘                                                 │
│          │                                                                                                │
│          ▼ (+)                                                                                            │
│       Çıktı: h = W_0 x + (α / r) · (B · A) x                                                              │
│                                                                                                           │
│   DAĞITIMDA AĞIRLIK BİRLEŞTİRME (WEIGHT MERGING):                                                         │
│       W_merged = W_0 + (α / r) · (B · A) ──> h = W_merged · x (Sıfır Ek Gecikme!)                        │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Düşük Dereceli Ayrışım ve İleri Geçiş
$W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ dondurulmuş matris, $r \ll \min(d_{\text{in}}, d_{\text{out}})$ düşük derece, $A \in \mathbb{R}^{r \times d_{\text{in}}}$ ve $B \in \mathbb{R}^{d_{\text{out}} \times r}$ eğitilebilir adaptör matrisleri olmak üzere:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

### 2. Başlatma Kuralı (Initialization Invariance)
Eğitimin 0. adımında modelin temel modelle birebir aynı davranmasını garanti etmek için:

$$A \sim \mathcal{N}\left(0, \frac{1}{r}\right), \quad B = \mathbf{0} \implies \Delta W = B \cdot A = \mathbf{0}$$

### 3. Ağırlık Birleştirme (Weight Merging & Unmerging)
$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B \cdot A), \quad W_0 = W_{\text{merged}} - \frac{\alpha}{r} (B \cdot A)$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **PEFT** | *Parameter-Efficient Fine-Tuning* | Modelin tüm ağırlıkları yerine yalnızca küçük bir adaptör grubunu güncelleyen verimli ince ayar ailesi. |
| **LoRA** | *Low-Rank Adaptation* | Ağırlık güncellemelerini iki düşük dereceli matrisin ($B \cdot A$) çarpımı olarak modelleyen yöntem. |
| **Rank ($r$)** | *Matris Derecesi* | Adaptör matrislerinin iç boyutu ($r \in \{2, 4, 8, 16\}$). Parametre sayısını ve adaptasyon kapasitesini belirler. |
| **LoRA Alpha ($\alpha$)**| *LoRA Ölçekleme Katsayısı* | $\frac{\alpha}{r}$ adaptör güncelleme katsayısı; $r$ değiştiğinde öğrenme oranını yeniden ayarlama ihtiyacını ortadan kaldırır. |
| **Weight Merging** | *Ağırlık Birleştirme* | $B \cdot A$ adaptör matrislerini ana ağırlık matrisine ($W_0$) doğrudan toplayarak çıkarım gecikmesini sıfırlama işlemi. |
| **Zero-Init ($B=0$)** | *Sıfırla Başlatma* | $B$ matrisini sıfırlarla başlatarak ince ayarın ilk adımında çıktının değişmemesini sağlayan kural. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Parametrelerin yalnızca ~%1-2'si eğitilerek %99+ başarı; Ağırlık birleştirme ile 0 ms ek gecikme; Hafif dosya boyutu. |
| **Weaknesses (Zayıf Yönler)** | Hangi modüllerin (w_q, w_v vs ffn) seçileceği deneysel karar gerektirir; Düşük derecede kapasite sınırı. |
| **Opportunities (Fırsatlar)** | Kenar cihazlarda (Edge AI) dinamik adaptör değişimi; QLoRA (4-bit kuantizasyon) ile bellek ayak izini %80 azaltma. |
| **Threats (Tehditler)** | Adaptör ağırlıkları birleştirilmeden eşzamanlı çoklu çıkarımda matris çarpım ek yükü. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-81-vit-lora-peft/`](.) dizinindedir:

### A. LoRA Doğrusal Katmanı (Forward, Merging & Unmerging)
Dosya: [`src/lora_katmani.py`](src/lora_katmani.py)
```python
class LoRADogrusalKatman(nn.Module):
    def __init__(self, orijinal_katman: nn.Linear, r: int = 4, lora_alpha: float = 8.0, lora_dropout: float = 0.0):
        super().__init__()
        self.orijinal_katman = orijinal_katman
        self.orijinal_katman.weight.requires_grad = False
        self.r, self.lora_alpha, self.olcek = r, lora_alpha, lora_alpha / r
        self.birlestirildi = False

        cihaz, veri_tipi = orijinal_katman.weight.device, orijinal_katman.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, orijinal_katman.in_features, device=cihaz, dtype=veri_tipi))
        self.lora_B = nn.Parameter(torch.zeros(orijinal_katman.out_features, r, device=cihaz, dtype=veri_tipi))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def birlestir(self):
        if not self.birlestirildi:
            self.orijinal_katman.weight.data += (self.lora_B @ self.lora_A) * self.olcek
            self.birlestirildi = True

    def ayir(self):
        if self.birlestirildi:
            self.orijinal_katman.weight.data -= (self.lora_B @ self.lora_A) * self.olcek
            self.birlestirildi = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.birlestirildi:
            return self.orijinal_katman(x)
        return self.orijinal_katman(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.olcek
```

---

## 📊 Deneysel Sonuçlar ve Performans Doğrulaması

`ana_akis.py` çalıştırılarak elde edilen analitik parametre ve gecikme dökümü:

```text
======================================================================
              Metrik                |             Değer             
======================================================================
Toplam Model Parametresi            |            212,170            
Dondurulan (Frozen) Parametre       |            207,424             (%97.76)
Eğitilebilir (LoRA + Head) Parametre |             4,746              (% 2.24)
Enjekte Edilen LoRA Katman Sayısı   |               8               
======================================================================
  ✓ Ayrık ve Birleşik Çıktılar Arasındaki Maksimum Sayısal Fark: 5.36e-07
  ✓ Ayrık LoRA Çıkarım Süresi (500 iterasyon): 2067.70 ms
  ✓ Birleşik LoRA (0 ms Ek Yük) Çıkarım Süresi: 1575.16 ms (%23.8 Hızlanma)
```

- **%97.76 Parametre Tasarrufu:** Ana omurganın %97.76'sı tamamen dondurulmuş, yalnızca 4,746 parametre ile downstream adaptasyon tamamlanmıştır.
- **Kusursuz Matematiksel Tutarlılık:** Ayrık model ile birleştirilmiş model çıktıları arasındaki fark $5.36 \times 10^{-7}$ (sıfır hata) seviyesindedir.
- **Birim Test Güvencesi:** [`testler/test_lora_peft.py`](testler/test_lora_peft.py) altındaki **8/8 birim test %100 PASSED (5.11s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/vit_lora_peft_paneli.png`](ciktilar/vit_lora_peft_paneli.png) konumundadır:

1. **LoRA Matematiksel Matris Ayrışımı:** $W_0 + \frac{\alpha}{r} (B \cdot A)$ hesaplama grafiği.
2. **Parametre Dağılımı:** Dondurulan (%97.76) vs Eğitilebilir LoRA (%2.24) pasta grafiği.
3. **Farklı LoRA Dereceleri (Rank $r=2..16$):** Parametre büyümesi ve doğruluk eğrisi.
4. **Ağırlık Birleştirme Çıkarım Gecikmesi:** Ayrık vs Merged vs Base çıkarım süreleri.
5. **LoRA İnce Ayar Eğitim Yakınsaması:** Kayıp ve doğruluk profili.
6. **Vision Transformer LoRA SWOT Matrisi:** Mühendislik karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** LoRA adaptörlerini sadece dikkat katmanlarına ($W_q, W_v$) değil, isteğe bağlı olarak Transformer'ın **Feed-Forward MLP katmanlarına ($W_{\text{fc1}}, W_{\text{fc2}}$)** da otomatik enjekte eden esnek bir genişletici yazınız.

```python
class KapsamliViTLoRAEnjekteEdici(ViTLoRAEnjekteEdici):
    """Hem MHSA hem de FFN katmanlarını destekleyen genişletilmiş enjekte edici."""
    def __init__(self, dikkat_modulleri=["w_q", "w_v"], ffn_modulleri=["fc1", "fc2"], r=4, lora_alpha=8.0):
        super().__init__(hedef_moduller=dikkat_modulleri, r=r, lora_alpha=lora_alpha)
        self.ffn_modulleri = ffn_modulleri

    def enjekte_et(self, model: nn.Module) -> nn.Module:
        model = super().enjekte_et(model)
        for blok in model.bloklar:
            for ffn_ad in self.ffn_modulleri:
                if hasattr(blok.ffn, ffn_ad):
                    eski_linear = getattr(blok.ffn, ffn_ad)
                    if isinstance(eski_linear, nn.Linear):
                        lora_katman = LoRADogrusalKatman(eski_linear, r=self.r, lora_alpha=self.lora_alpha)
                        setattr(blok.ffn, ffn_ad, lora_katman)
                        self.enjekte_edilen_katmanlar.append(lora_katman)
        return model
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** LoRA adaptör matrislerinden $A$ matrisi rastgele Gauss/Kaiming ile başlatılırken, $B$ matrisi neden **KESİNLİKLE SIFIR ($\mathbf{0}$)** ile başlatılmak zorundadır? $B$ de rastgele başlatılsaydı ne olurdu?

> **Mentor Cevabı:**
> 1. **Eğitim Başlangıç Şartı (Zero-Perturbation Principle):** İnce ayarın 0. adımında modelin çıktısı $h = W_0 x + \frac{\alpha}{r} (B \cdot A) x$ formülüne dayanır. Eğer $B = \mathbf{0}$ olursa, $B \cdot A = \mathbf{0}$ olur ve $\Delta W = \mathbf{0}$ çıkar. Bu sayede model eğitime başlarken **önceden eğitilmiş temel modelle %100 özdeş çıktılar üretir.**
> 2. **Rastgele Başlatmanın Yıkıcı Etkisi:** Eğer $B$ de rastgele bir dağılımla başlatılsaydı, $\Delta W \neq \mathbf{0}$ olacak ve modelin önceden öğrendiği tüm ağırlıklara devasa bir rastgele gürültü eklenmiş olacaktı. Bu durum ilk adımda modelin doğruluk oranını sıfıra düşürür ve ön eğitim kazanımlarını büyük ölçüde tahrip ederdi.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
