# Day 71: Çökmeye Dayanıklı Checkpoint, State Restoration ve Devam Edebilir Eğitim Motoru

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passed-brightgreen?style=flat-square)

## 🎯 Proje Özeti & Mühendislik Hedefi

Bulut sunucularında (AWS EC2, GCP Compute Engine, Kubernetes K8s) günlerce veya haftalarca süren büyük ölçekli derin öğrenme eğitimleri; donanım arızaları, bellek yetersizliği (**OOM - Out Of Memory**), çekirdek kesintileri (**SIGKILL / SIGTERM**) ve özellikle maliyet avantajı sağlayan **Spot / Preemptible GPU** kesintileri nedeniyle her an aniden sonlanabilir.

Yalnızca `torch.save(model.state_dict(), "model.pt")` çağrısı yapmak, eğitim yeniden başlatıldığında **büyük kayıp sıçramalarına (loss spikes)**, optimizer momentum kaybına, öğrenme oranı çizelgesinin sıfırlanmasına ve determinizmin bozulmasına yol açar.

Bu projede; tam durum restorasyonunu garanti eden, atomik dosya yazımı (**Atomic I/O**) ile bozuk dosya oluşumunu sıfıra indiren ve disk taşmasını önleyen **Top-K Checkpoint Saklama Politikasına** sahip kurumsal sınıf bir **Devam Edebilir Eğitim Motoru (Resumable Training Engine)** geliştirilmiştir.

---

## 🔬 Teorik & Matematiksel Derinlik

### 1. Tam Eğitim Durum Matrisi (Complete Training State Tuple)
Eğitimin bir $t$ anından sıfır kayıp sıçraması ve %100 determinizm ile devam edebilmesi için saklanması zorunlu olan durum demeti $\mathcal{S}_t$:

$$\mathcal{S}_t = \Big\langle t, \; \theta_t, \; \mathcal{S}_{\text{opt}}(t), \; \mathcal{S}_{\text{sched}}(t), \; \mathcal{S}_{\text{scaler}}(t), \; \mathcal{S}_{\text{rng}}, \; \mathcal{M}_{\text{val}} \Big\rangle$$

- $t$: Tamamlanan global epoch / adım indeksi.
- $\theta_t$: Model ağırlıkları (`model.state_dict()`).
- $\mathcal{S}_{\text{opt}}(t)$: Optimizer durum tensörleri (`optimizer.state_dict()`). AdamW için 1. moment $m_t$, 2. moment $v_t$ ve adım sayacı $t_{\text{opt}}$.
- $\mathcal{S}_{\text{sched}}(t)$: Öğrenme oranı zamanlayıcısının mevcut adımı (`scheduler.state_dict()`).
- $\mathcal{S}_{\text{rng}}$: Rastgele sayı üreteçleri (`torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, `numpy`, `random`).
- $\mathcal{M}_{\text{val}}$: Doğrulama kaybı ve doğruluğu (`val_loss`, `val_acc`).

---

### 2. Atomik Dosya Yazma Mekanizması (Atomic I/O & Crash-Safe Storage)
Doğrudan `checkpoint.pt` dosyasına yazılırken sistem çökerse veya elektrik kesilirse dosya yarıda kalır ve bozulur (**Corrupted Checkpoint**). Çözüm:

$$\text{Bellek Tensörleri} \xrightarrow{\text{torch.save()}} \text{checkpoint.pt.tmp} \xrightarrow{\text{os.replace() [Atomic Rename]}} \text{checkpoint.pt}$$

`os.replace()` işletim sistemi çekirdeği seviyesinde (POSIX ve Windows NTFS) atomiktir; dosya ya tamamen ve hatasız hedefte var olur ya da eski dosya korunur.

---

### 3. Optimizer Momentum Kaybı ve Kayıp Sıçraması (Loss Spike) Mekaniği
Eğer eğitim sadece $\theta_t$ ile başlatılırsa:
1. $m_t \leftarrow \mathbf{0}$ ve $v_t \leftarrow \mathbf{0}$ (Soğuk Başlangıç / Cold Start).
2. Adaptif öğrenme oranı terimi $\frac{1}{\sqrt{v_t} + \epsilon} \approx \frac{1}{\epsilon} = 10^8$ mertebesine fırlar!
3. İlk optimizasyon adımında parametreler aşırı savrulur ve eğitim kaybı $1.0 \to 15.0$ seviyesine fırlar (Loss Spike).

---

## 🛠️ Neden Bu Yöntem Seçildi? (Mühendislik Gerekçesi & Kaçınılan Tuzaklar)

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    STANDART VS KURUMSAL ÇÖKMEYE DAYANIKLI CHECKPOINT MİMARİSİ                             │
│                                                                                                           │
│  [1. Standart Acemi Yaklaşım (Tehlikeli)]:                                                                 │
│  • torch.save(model.state_dict(), "model.pt")                                                             │
│  • Çökme anında dosya bozulur (0 byte), optimizer momentumu sıfırlanır, eğitim patlar.                   │
│                                                                                                           │
│  [2. Kurumsal Sınıf Çökmeye Dayanıklı Motor (Uygulanan)]:                                                 │
│  • .tmp dosyası + os.replace() ile ATOMİK I/O (Bozulma riski = %0).                                       │
│  • Model + Optimizer + Cosine Scheduler + RNG durumları tek pakette saklanır.                            │
│  • Top-K budama ile disk dolması (Disk Full Outage) engellenir.                                           │
│  • Çökme noktasından devam edildiğinde kayıp eğrisi kesintisiz ve pürüzsüz akar.                         │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Spot / Preemptible GPU Güvenliği:** AWS/GCP üzerinde %70 daha ucuz olan Spot sunucular kullanılabilir; sunucu kapandığında yeni makineye geçilip son checkpoint'ten devam edilir.
2. **Determinizm ve Veri Karıştırma Devamlılığı:** RNG durumu korunduğu için veri yükleyicinin shuffle döngüsü ve dropout maskeleri kaldığı yerden tutarlı devam eder.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Checkpointing** | *Model & State Checkpointing* | Modelin ağırlıklarının ve eğitim yürütücüsü bileşenlerinin periyodik olarak kalıcı depolama birimine yazılması işlemi. |
| **State Restoration** | *Complete State Restoration* | Kayıtlı bir checkpoint dosyasından model, optimizer, scheduler ve rastgele durumları belleğe yükleyip eğitimi tam kaldığı adımdan devam ettirme. |
| **Atomic I/O** | *Atomic File Write & Replace* | Dosya yazımının yarım kalmasını önlemek için önce geçici dosyaya yazıp ardından atomik `os.replace` ile hedefi güncelleme mimarisi. |
| **Loss Spike** | *Optimizer Cold-Start Loss Spike* | Optimizer durumları ($m_t, v_t$) olmadan eğitim başlatıldığında ilk adımlarda yaşanan ani ve yıkıcı kayıp artışı. |
| **RNG State** | *Random Number Generator State* | PyTorch, NumPy ve Python'ın rastgele sayı üreticilerinin içsel durum tensörleri; deterministik eğitim devamlılığı için şarttır. |
| **Top-K Retention** | *Top-K Checkpoint Pruning* | Disk alanını tüketmemek için yalnızca en iyi $K$ adet checkpoint dosyasını ve `last.pt` dosyasını saklayıp eskileri silen politika. |
| **Spot Instance** | *Preemptible / Spot GPU Instance* | Bulut sağlayıcıların boşta kalan GPU'ları %70-90 indirimle sunduğu ancak her an geri çağırabildiği sunucu türü. |
| **OOM (Out Of Memory)** | *GPU Out Of Memory Crash* | Grafik belleğinin taşması sonucu eğitimin işletim sistemi tarafından aniden sonlandırılması (SIGKILL) durumu. |
| **last.pt / best.pt** | *Latest and Best Checkpoint Pointers* | En son tamamlanan epoch'u ve en yüksek doğrulama başarımı gösteren modeli işaret eden sabit referans dosyalar. |
| **Graceful Shutdown** | *Signal-Driven Safe Termination* | SIGTERM veya SIGINT sinyali alındığında motorun mevcut batch'i tamamlayıp güvenli checkpoint alarak kapanması. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Atomik I/O ile sıfır bozuk dosya riski; Optimizer momentum koruması ile sıfır kayıp sıçraması; RNG saklama ile %100 determinizm; Top-K ile kontrollü disk kullanımı. |
| **Weaknesses (Zayıf Yönler)** | Büyük modellerde (örn. 70B LLM) her checkpoint yazımında disk I/O süresi (SSD/NVMe gereksinimi). |
| **Opportunities (Fırsatlar)** | Spot GPU kümelerinde yüz binlerce dolarlık bulut maliyeti tasarrufu; haftalarca süren Vision Transformer eğitimlerinde tam iş sürekliliği. |
| **Threats (Tehditler)** | Çok sık checkpoint alma aralığı seçildiğinde disk I/O darboğazı sebebiyle eğitim süresinin uzaması. |

---

## 📈 Deneysel Çöküş & Kesintisiz Devam Sonuçları

Epoch 1-5 arasında normal eğitim koşturulmuş, Epoch 5 sonunda **donanım çöküşü (RuntimeError / SIGKILL)** simüle edilmiş, ardından `last.pt` dosyasından yeni bir motor örneği ile Epoch 6-10 kesintisiz tamamlanmıştır:

| Epoch | Train Loss | Val Loss | Val Acc (%) | Öğrenme Oranı (LR) | Durum Açıklaması |
|---|---|---|---|---|---|
| **1** | $1.6747$ | $1.6081$ | %23.00 | $0.000976$ | Faz 1 (Normal Çalışma) |
| **2** | $1.4725$ | $1.6322$ | %18.50 | $0.000905$ | Faz 1 (Normal Çalışma) |
| **3** | $1.3614$ | $1.6325$ | %21.00 | $0.000796$ | Faz 1 (Normal Çalışma) |
| **4** | $1.2654$ | $1.6382$ | %21.00 | $0.000658$ | Faz 1 (Normal Çalışma) |
| **5** | **$1.1705$** | $1.6445$ | %22.00 | $0.000505$ | **Faz 1 (Son Kayıt - Çöküş Gerçekleşti)** |
| **6** | **$1.0842$** | $1.6480$ | %20.50 | $0.000352$ | **Faz 2 (Geri Yüklendi - SIFIR KAYIP SIÇRAMASI)** |
| **7** | $1.0151$ | $1.6666$ | %21.50 | $0.000214$ | Faz 2 (Devam Ediyor) |
| **8** | $0.9596$ | $1.6583$ | %19.50 | $0.000105$ | Faz 2 (Devam Ediyor) |
| **9** | $0.9260$ | $1.6559$ | %20.50 | $0.000034$ | Faz 2 (Devam Ediyor) |
| **10**| **$0.9075$** | $1.6597$ | %19.00 | $0.000010$ | Faz 2 (Eğitim Tamamlandı) |

- **Kayıp Devamlılığı (No Loss Spike):** Epoch 5 ($1.1705$) $\to$ Epoch 6 ($1.0842$). Sıfır sıçrama ile pürüzsüz yakınsama devam etti.
- **LR Çizelgesi Koruması:** Cosine Annealing eğrisi sıfırlanmadı, $0.000505 \to 0.000352$ olarak devam etti.
- **Birim Test Başarımı:** **$8 / 8$ PASSED (%100 Başarı, 7.75s)**

---

## 🖼️ Görsel Çıktı: 6 Panelli Teşhis Panosu

Laboratuvar sonuçları [`ciktilar/resumable_training_paneli.png`](file:///c:/Users/seydieryilmaz/Desktop/Github%20Mini%20AI%20Engineer/day-71-fault-tolerant-resumable-training-engine/ciktilar/resumable_training_paneli.png) dosyasında üretilmiştir:
1. **Çöküş & Restorasyon Özeti**: Faz 1 ve Faz 2 geçiş metrikleri kartı.
2. **Kesintisiz Eğitim Kaybı Trajektorisi**: Çöküş çizgisinde sıçrama olmadan devam eden yeşil eğri.
3. **Doğrulama Başarımı**: 10 epoch'luk doğruluk grafiği.
4. **Scheduler Durum Devamlılığı**: Cosine LR çizelgesinin sıfırlanmadan akışı.
5. **Top-K Disk Yönetimi**: `best.pt`, `last.pt` ve saklanan checkpoint boyutları.
6. **SWOT Karar Matrisi**: Mimari tercihlerin endüstriyel sentezi.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Checkpoint yüklerken model mimarisinde küçük bir değişiklik yapılmışsa (örneğin son sınıflandırıcı katmanı değiştirilmişse) `strict=False` mantığı ile uyumlu ağırlıkları yükleyip eksik olanları loglayan esnek bir yükleyici yazınız.

**Eksiksiz Çözüm:**
```python
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List

def esnek_checkpoint_yukle(
    model: nn.Module,
    checkpoint_yolu: str
) -> Tuple[List[str], List[str]]:
    """Eksik veya uyuşmayan katmanları filtreleyerek güvenle ağırlık yükler."""
    paket = torch.load(checkpoint_yolu, map_location="cpu", weights_only=False)
    kayitli_durum = paket.get("model_state_dict", paket)

    model_durumu = model.state_dict()
    uyumlu_durum = {}
    atlanan_katmanlar = []

    for k, v in kayitli_durum.items():
        if k in model_durumu and model_durumu[k].shape == v.shape:
            uyumlu_durum[k] = v
        else:
            atlanan_katmanlar.append(k)

    model_durumu.update(uyumlu_durum)
    model.load_state_dict(model_durumu)
    
    eksik_katmanlar = [k for k in model.state_dict().keys() if k not in kayitli_durum]
    return atlanan_katmanlar, eksik_katmanlar
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Derin bir modeli eğitirken neden doğrudan `torch.save(..., "checkpoint.pt")` yazmak üretim ortamında ölümcül bir hatadır? `os.replace` kullanarak geçici dosyadan hedef dosyaya taşıma (Atomic Replace) mekanizması işletim sistemi seviyesinde bu problemi nasıl çözer?

> **Mentor Cevabı:**
> 1. **Doğrudan Yazmanın Tehlikesi (Kısmi / Bozuk Dosya):** `torch.save` çağrısı gigabaytlarca veriyi diske parça parça yazar (I/O akışı). Eğer bu işlem sürerken sunucuya `SIGKILL` gelirse, elektrik kesilirse veya Spot GPU geri çağrılırsa hedef dosya yarım kalır (örneğin %40'ı yazılmış 0 byte veya bozuk dosya). Eğitim yeniden başlatılmaya çalışıldığında `EOFError: Ran out of input` veya `PytorchStreamReader failed` hatasıyla eğitim tamamen çöker ve önceki tüm ilerleme kaybedilir!
> 2. **Atomik İsim Değiştirme (Atomic Rename / Replace):** Modern dosya sistemlerinde (Linux ext4/xfs, Windows NTFS) bir dosyanın adını değiştirmek veya dizin kaydını güncellemek tek bir inode/metadata operasyonudur ve atomiktir (ya %100 gerçekleşir ya da hiç gerçekleşmez).
> 3. **Çözüm Algoritması:** Veri önce `checkpoint.pt.tmp` geçici dosyasına tamamen yazılır. Yazma başarıyla bittiğinde `os.replace("checkpoint.pt.tmp", "checkpoint.pt")` çağrılır. Eğer yazma sırasında elektrik kesilirse sadece `.tmp` dosyası zarar görür; diskteki önceki çalışan `checkpoint.pt` dosyasına asla dokunulmamış olur!

---

## 📜 Lisans & Telif Hakkı

```text
/*
 * Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101-Day AI, Computer Vision & MLOps Master Series
 * License: Private - All Rights Reserved
 */
```
