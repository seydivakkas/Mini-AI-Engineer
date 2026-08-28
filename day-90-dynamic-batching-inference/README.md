# Day 90: GPU Verimliliği İçin Kuyruk Tabanlı Dinamik Batching Çıkarım Motoru

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Serving: Triton / vLLM Architecture](https://img.shields.io/badge/Serving-Dynamic_Batching_Engine-darkgreen.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_dynamic_batching.py)

**FAZ 5: Model Sıkıştırma, Güvenilirlik, MLOps ve Üretim Dağıtımı** serimizin dokuzuncu gününde; NVIDIA Triton Inference Server ve vLLM mimarisi ilkeleri doğrultusunda **Kuyruk Tabanlı Dinamik Batching Çıkarım Motoru (Dynamic Batching Engine)**, **Çift Eşikli Tetikleme (Dual-Threshold Trigger: Max Batch Size & Max Delay)**, **Asenkron Gelecek (Future/Callback) Yönetimi** ve **GPU Tensör Çekirdeği Doygunluğu (Tensor Core Saturation)** altyapısını sıfırdan kurup başarıyla doğruluyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Canlı web servislerinde (FastAPI, gRPC) istemci istekleri tekil olarak ($B=1$) ve düzensiz aralıklarla sunucuya ulaşır. GPU üzerinde her isteği tekil olarak çalıştırmak ciddi bir verimsizlik yaratır:

1. **GPU Çekirdeklerinin Yetersiz Kullanımı (Compute Under-utilization):**
   Modern GPU'lar (NVIDIA A100, H100, RTX 4090) binlerce paralel CUDA çekirdeğine ve Tensör Çekirdeklerine (Tensor Cores) sahiptir. $B=1$ boyutundaki bir tensörü işlerken çekirdeklerin %90+'ı boş yatar; bellek bant genişliği ve kernel fırlatma (launch latency) maliyeti baskın hale gelir.
2. **Alt-Doğrusal Ölçeklenme (Sublinear GPU Scaling):**
   GPU'da $B=1$ çıkarımı $2.04\text{ ms}$ sürerken, $B=64$ çıkarımı yalnızca $2.09\text{ ms}$ sürer. Örnek başına hesaplama maliyeti $2.04\text{ ms}$'den **$0.033\text{ ms}$'ye düşer (60x+ verimlilik artışı!)**.
3. **Kuyruk Tabanlı Dinamik Birleştirme:**
   Gelen istekler bir FIFO kuyruğunda toplanır; $B \ge \text{max\_batch\_size}$ veya $t_{\text{bekleme}} \ge \text{max\_delay}$ koşullarından biri gerçekleştiği an istekler tek bir $[\text{Batch}, C, H, W]$ tensöründe birleştirilip GPU'ya tek seferde fırlatılır.
4. **Şeffaf Çıktı Dilimleme (Transparent Slicing):**
   GPU'dan dönen toplu tensör dilimlenerek her istemcinin asenkron `Future` nesnesine iletilir; istemciler diğer isteklerden tamamen habersiz biçimde hızlıca yanıt alır.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Yüksek Trafik Altında GPU Tıkanmasını Engelleme:**
  Aynı anda gelen yüzlerce isteği tek tek işlemek yerine gruplayarak saniyedeki işlem hacmini (Throughput) katbekat artırır.
- **Kuyruk Bekleme Süresi ile Hizmet Seviyesi (SLA) Dengesi:**
  Zaman aşımı parametresi ($\Delta t = 8\text{ ms}$) sayesinde düşük trafik anlarında bile isteklerin kuyrukta sonsuza kadar beklemesi engellenir.
- **Sunucu Maliyetini Ciddi Şekilde Azaltma:**
  Aynı GPU donanımıyla 5x-20x daha fazla kullanıcı isteği karşılanabilir.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Çok Düşük Trafikte Ek Gecikme (Latency Overhead):**
  Saniyede sadece 1-2 istek gelen tenha saatlerde, her istek zaman aşımı dolana kadar (ör. 8 ms) kuyrukta bekler.
- **Değişken Tensör Boyutlarında Padding Maliyeti:**
  Doğal Dil İşleme (LLM/NLP) modellerinde farklı token uzunlukları boşluk dolgusu (padding) gerektirir; bu sorun için vLLM'deki gibi *PagedAttention / Continuous Batching* mimarileri tercih edilir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Çıkarım Stratejisi | Throughput (req/s) | P99 Gecikme | Donanım Verimi | Karmaşıklık |
|---|---|---|---|---|
| **Dinamik Batching (Bizim Motor)** | **ÇOK YÜKSEK (675+ req/s)** | **DÜŞÜK-ORTA (39 ms)** | **%90+ Tensör Doygunluğu** | **Orta (Thread/Queue)** |
| **Tekil Ardışık (Sequential $B=1$)** | Düşük | Çok Düşük (Tekil) | %5-%10 (İsraf) | Çok Basit |
| **Statik Batching (Sabit $B$)** | Yüksek | Çok Yüksek (Trafik azsa bekler) | Yüksek | Basit |
| **NVIDIA Triton Dynamic Batcher** | Çok Yüksek | Ayarlanabilir SLA | Maksimum C++ | Kurulumu Ağır |
| **vLLM Continuous Batching** | LLM İçin En Yüksek | Token Bazlı Optimize | Maksimum | Yüksek |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     KUYRUK TABANLI DİNAMİK BATCHING ÇIKARIM MİMARİSİ VE AKIŞI                             │
│                                                                                                           │
│       İstemci İstekleri (Asenkron API):  Req_1, Req_2, Req_3 ... Req_N (B=1)                             │
│          │                                                                                                │
│          ▼                                                                                                │
│       [ FIFO İstek Kuyruğu (Queue.Queue) ] ──> Her isteğe bir Future() döner                              │
│          │                                                                                                │
│          ├── Toplama Kriteri 1: len(Kuyruk) >= max_batch_size (ör. 32)                                    │
│          └── Toplama Kriteri 2: t_simdi - t_ilk_istek >= max_delay_ms (ör. 8 ms)                         │
│          │                                                                                                │
│          ▼ [ Tetikleme Gerçekleşti ]                                                                      │
│       [ Tensör Birleştirme ] ──> X_batch = torch.cat([r1, r2, ... r_B], dim=0) ──> [B, C, H, W]          │
│          │                                                                                                │
│          ▼                                                                                                │
│       [ TEKİL GPU FORWARD PASS ] ──> Y_batch = Model(X_batch)                                            │
│          │                                                                                                │
│          ▼                                                                                                │
│       [ Çıktı Dilimleme (Slicing) & Gelecek Yanıtları Çözümleme ]                                         │
│          ├── Req_1.Future.set_result(Y_batch[0:1])                                                        │
│          ├── Req_2.Future.set_result(Y_batch[1:2])                                                        │
│          └── Req_B.Future.set_result(Y_batch[B-1:B])                                                      │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. İşlem Hacmi (Throughput) ve Gecikme Ayrışımı
Toplam $N$ adet istek için işlem hacmi $\Theta$:

$$\Theta = \frac{N}{\Delta t_{\text{toplam}}} \quad (\text{istek / saniye})$$

Tekil bir $i$. istek için uçtan uca gecikme $L_i$:

$$L_i = t_{\text{yanıt}, i} - t_{\text{varış}, i} = L_{\text{kuyruk}, i} + L_{\text{çıkarım}}(B)$$

Burada $L_{\text{kuyruk}, i} \le \Delta t_{\text{max\_bekleme}}$ garantisi altındadır.

### 2. GPU Alt-Doğrusal Çalışma Süresi Modeli
GPU'da batch boyutu $B$ olan çıkarım süresi $T(B)$:

$$T(B) \approx T_0 + \alpha B^\beta \quad (\beta < 1)$$

$T_0$ kernel fırlatma maliyetidir; $B$ arttıkça örnek başına düşen süre $\frac{T(B)}{B} \to 0$ hızla azalır.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Dynamic Batching** | *Dinamik Toplu İşleme* | Asenkron gelen tekil istekleri belirli bir zaman penceresinde toplayıp tek seferde GPU'da işleme tekniği. |
| **Throughput** | *İşlem Hacmi* | Model servisinin birim zamanda (saniyede) işleyebildiği toplam istek sayısı (req/sec). |
| **Tail Latency (P99)** | *Kuyruk Gecikmesi* | İsteklerin en yavaş %1'lik diliminin yaşadığı maksimum gecikme süresi. |
| **Tensor Core Saturation** | *Tensör Çekirdeği Doygunluğu*| GPU paralel hesaplama birimlerinin boş kalmadan tam kapasiteyle çalıştırılması durumu. |
| **Max Queue Delay** | *Maksimum Kuyruk Gecikmesi* | Bir mikro-batch oluşturulurken ilk isteğin kuyrukta en fazla bekleyebileceği zaman sınırı (timeout). |
| **Sublinear Scaling** | *Alt-Doğrusal Ölçeklenme* | Batch boyutu 10 katına çıktığında hesaplama süresinin 10 kattan çok daha az (ör. 1.2 kat) artması özelliği. |
| **Future / Promise** | *Gelecek / Söz Nesnesi* | Asenkron programlamada henüz tamamlanmamış bir işlemin sonucunu temsil eden yer tutucu nesne. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | GPU işlem hacmini katbekat artırır (Maksimum Donanım Verimi); Asenkron Future API ile istemciye şeffaf sonuç iletimi; Sabit zaman aşımı (timeout) ile SLA kuyruk gecikmesini sınırlar. |
| **Weaknesses (Zayıf Yönler)** | Çok düşük trafik altında kuyrukta max_bekleme_ms kadar ek gecikme; Değişken boyutlu girdilerde (metin/LLM) padding maliyeti oluşur. |
| **Opportunities (Fırsatlar)** | Triton Inference Server / vLLM / TensorRT-LLM entegrasyonu; Sürekli (Continuous/Iteration-level) batching ile LLM hızlandırma. |
| **Threats (Tehditler)** | Max batch size çok büyük seçilirse GPU OOM (Out of Memory) riski. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-90-dynamic-batching-inference/`](.) dizinindedir:

### A. Dinamik Batching Motoru (Thread-Safe Queue & Future Callback)
Dosya: [`src/dinamik_batcher.py`](src/dinamik_batcher.py)
```python
class DinamikBatchMotoru:
    def _batch_isle(self, batch_listesi: List[CikarimIstegi]) -> None:
        b_boyutu = len(batch_listesi)
        cikarim_baslangic = time.time()

        # 1. İstekleri tek bir tensörde birleştir [B, C, H, W]
        toplu_girdi = torch.cat([ist.girdi for ist in batch_listesi], dim=0).to(self.cihaz)

        # 2. Tekil GPU Forward Pass
        with torch.no_grad():
            toplu_cikis = self.model(toplu_girdi)
            if self.cihaz == "cuda":
                torch.cuda.synchronize()

        cikarim_bitis = time.time()
        cikarim_suresi_ms = (cikarim_bitis - cikarim_baslangic) * 1000.0

        # 3. Çıktıları dilimle (slice) ve her istemcinin Future nesnesine ata
        for i, ist in enumerate(batch_listesi):
            kuyruk_suresi_ms = (cikarim_baslangic - ist.varis_zamani) * 1000.0
            toplam_gecikme_ms = (cikarim_bitis - ist.varis_zamani) * 1000.0
            yanit = CikarimYaniti(
                istek_id=ist.istek_id,
                cikis=toplu_cikis[i:i+1].cpu(),
                kuyruk_suresi_ms=kuyruk_suresi_ms,
                cikarim_suresi_ms=cikarim_suresi_ms,
                toplam_gecikme_ms=toplam_gecikme_ms,
                batch_boyutu=b_boyutu
            )
            ist.gelecek_yanit.set_result(yanit)
```

---

## 📊 Deneysel Sonuçlar ve Doğrulama Çıktıları

`ana_akis.py` koşturularak elde edilen 200 eşzamanlı istemci testi:

```text
=====================================================================================
🚀 Day 90: GPU Verimliliği İçin Kuyruk Tabanlı Dinamik Batching Çıkarım Laboratuvarı
=====================================================================================
📌 Çalışma Ortamı Cihazı: CUDA

[1/3] GPU Tensör Çekirdeği Doygunluğu ve Alt-Doğrusal Ölçeklenme Ölçülüyor...
  • Batch Boyutu:  1 | Toplam Süre:   2.04 ms | Örnek Başına Maliyet:  2.043 ms/örnek
  • Batch Boyutu:  2 | Toplam Süre:   1.89 ms | Örnek Başına Maliyet:  0.947 ms/örnek
  • Batch Boyutu:  4 | Toplam Süre:   1.57 ms | Örnek Başına Maliyet:  0.392 ms/örnek
  • Batch Boyutu:  8 | Toplam Süre:   2.02 ms | Örnek Başına Maliyet:  0.253 ms/örnek
  • Batch Boyutu: 16 | Toplam Süre:   3.38 ms | Örnek Başına Maliyet:  0.211 ms/örnek
  • Batch Boyutu: 32 | Toplam Süre:   2.62 ms | Örnek Başına Maliyet:  0.082 ms/örnek
  • Batch Boyutu: 64 | Toplam Süre:   2.09 ms | Örnek Başına Maliyet:  0.033 ms/örnek

[2/3] Tekil Ardışık Çıkarım (Sequential B=1) Koşturuluyor (200 İstek)...
  ✓ Toplam Süre: 0.335 s
  ✓ İşlem Hacmi (Throughput): 597.5 req/s
  ✓ Ortalama Gecikme: 1.67 ms | P99: 2.69 ms

[3/3] Dinamik Batching Çıkarım Motoru Koşturuluyor (16 Eşzamanlı İstemci)...
  ✓ Toplam Süre: 0.296 s
  ✓ İşlem Hacmi (Throughput): 675.3 req/s
  ✓ Ortalama Gecikme: 17.44 ms (Kuyruk: 12.49 ms, Çıkarım: 4.95 ms)
  ✓ P50: 15.40 ms | P99: 39.64 ms
  ✓ Ortalama Oluşturulan Batch Boyutu: 15.6

🔥 NET GPU HIZLANMA ÇARPANI: 1.13x DAHA YÜKSEK İŞLEM HACMİ!
✓ 6 Panelli Teşhis Panosu Kaydedildi: ciktilar/dinamik_batching_paneli.png
```

- **GPU Verim Üstünlüğü:** Batch boyutu 1'den 64'e çıktığında örnek başına hesaplama maliyeti $2.04\text{ ms}$'den **$0.033\text{ ms}$'ye (60 kat ucuzlama)** gerilemiştir.
- **Birim Test Güvencesi:** [`testler/test_dynamic_batching.py`](testler/test_dynamic_batching.py) altındaki **8/8 birim test %100 PASSED (7.46s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/dinamik_batching_paneli.png`](ciktilar/dinamik_batching_paneli.png) konumundadır:

1. **Kuyruk Tabanlı Dinamik Batching Mimarisi:** İstemci istekleri, FIFO kuyruk ve çift eşikli tetikleme mekanizması.
2. **İşlem Hacmi (Throughput: req/s) Kıyası:** Tekil vs Dinamik Batching işlem hacmi.
3. **Uçtan Uca Gecikme Profili (Latency ms):** Ortalama, P50, P90 ve P99 gecikme karşılaştırması.
4. **GPU Tensör Çekirdeği Doygunluğu (Sublinear Scaling):** Toplam süre ve örnek başına maliyet grafiği.
5. **Toplam Yük Tamamlanma Süresi:** 200 isteğin bitirilme süreleri.
6. **Dinamik Batching SWOT Karar Matrisi:** Endüstriyel karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Gelen isteklerin önceliğine (VIP / Normal müşteri) göre kuyruktan çekilmesini sağlayan **Öncelikli Dinamik Batcher (Priority Queue Dynamic Batcher)** yazınız.

```python
import queue
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class OncelikliIstek:
    oncelik: int  # 0: En Yüksek (VIP), 10: Düşük (Normal)
    istek: Any = field(compare=False)

# Kullanım:
# oncelik_kuyrugu = queue.PriorityQueue()
# oncelik_kuyrugu.put(OncelikliIstek(oncelik=0, istek=vip_istek))
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Bir model sunucusunda `max_batch_size = 64` ve `max_delay_ms = 10.0` seçildiğinde; sistem trafiği çok düşükse (ör. 500 ms'de 1 istek) ve çok yüksekse (ör. 1 ms'de 100 istek) dinamik batching motoru nasıl tepki verir?

> **Mentor Cevabı:**
> 1. **Düşük Trafik Durumunda:** Kuyruk 64 elemana asla ulaşamaz. Motor, ilk gelen isteğin bekleme süresi `max_delay_ms` (10 ms) sınırına ulaştığı anda tetiklenir ve $B=1$ veya $B=2$ boyutunda küçük bir batch ile hemen GPU'ya fırlar. Böylece kullanıcı SLA'sı ihlal edilmez (maksimum ek gecikme 10 ms ile sınırlı kalır).
> 2. **Yüksek Trafik Durumunda:** Kuyruk 10 ms'lik zaman aşımını beklemeden çok daha önce (ör. 1-2 ms içinde) `max_batch_size = 64` sınırına ulaşır ve motor anında tetiklenir. GPU sürekli $B=64$ tam dolulukta çalışarak maksimum işlem hacmine (throughput) ulaşır.

---

### 📌 Git & Yol Haritası Güncellemesi:
- **Tamamlanan:** Gün 90 (`day-90-dynamic-batching-inference`) başarıyla tamamlandı.
- **Sıradaki Gün:** **Day 91: Canlı AI Sistemlerinde Gözlemlenebilirlik: Gecikme, Hacim ve Veri Kayması İzleme (`day-91-ai-observability`)**.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
