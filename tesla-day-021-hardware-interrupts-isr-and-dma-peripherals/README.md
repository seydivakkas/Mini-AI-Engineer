# 🚗 Tesla Gömülü Donanım | Gün 21: Donanım Kesmeleri (ISR) ve DMA Sürücüleri (SPI/I2C/UART)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Architecture-ARM%20Cortex--M%20%2F%20NVIC-blue.svg?style=flat-square)](https://www.arm.com/)
[![DMA](https://img.shields.io/badge/DMA-Zero--Copy%20Ping--Pong-green.svg?style=flat-square)](https://www.st.com/)
[![Safety](https://img.shields.io/badge/Safety-Deterministic%20ISR-red.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"21. günümüze hoş geldin stajyer!  
> Bir Tesla'nın ivmeölçer, jiroskop, batarya voltaj okuma çipleri (AFE - Analog Front End) veya lidar/radar alıcıları saniyede on binlerce kez veri üretir.  
> Eğer ana mikrodenetleyici CPU'su sürekli bir döngüde `while(!SPI_RX_FLAG)` diyerek yoklama (Polling) yaparsa, CPU'nun $\%98$'i boşa harcanır ve motor kontrol algoritmaları gecikir!  
> Çözüm: **Donanım Kesmeleri (Interrupt Service Routines - ISR)** ve **Doğrudan Bellek Erişimi (DMA - Direct Memory Access)** mimarisidir:  
> 1. **NVIC (Nested Vectored Interrupt Controller):** Donanım kesmelerini öncelik sırasına göre yuvalar (Nesting). Örneğin kaza anında hava yastığı kesmesi, SPI veri okuma kesmesini anında bölüp icra edilir.  
> 2. **Sıfır-Kopyalı Ping-Pong DMA:** Donanım çevre birimi veriyi doğrudan RAM'deki tampona yazar; CPU sadece transfer tamamlandığında (Transfer Complete ISR) haberdar olur. CPU yükü $\%98$'den $\%1.8$'e düşer!  
> 3. **ISR Altın Kuralı:** Bir Kesme Servis Rutini içinde asla `sleep`/`delay` yapılmaz, dinamik bellek ayrılmaz ve ağır matematik hesaplanmaz. Sadece bayrak set edilir ve veri tamponu takas edilir.  
> Bugün gömülü donanım sürücülerinin kalbini oluşturan kesme ve DMA motorunu yazıyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Donanımsal Kesme Gecikmesi (Interrupt Latency)
Olayın donanımda meydana gelmesi ile ISR'ın ilk komutunun çalışması arasındaki toplam süre $T_{\text{latency}}$:

$$T_{\text{latency}} = T_{\text{detect}} + T_{\text{stack\_push}} + T_{\text{vector\_fetch}} + T_{\text{pipeline\_flush}}$$

ARM Cortex-M mimarisinde tipik olarak $12\text{ clock cycle}$ (168 MHz işlemcide yaklaşık $71.4\text{ ns}$) sürer.

### 2. DMA Çift Tamponlama (Ping-Pong) Veri Akışı
Bant genişliği $BW_{\text{DMA}}$ ve CPU işlem süresi $T_{\text{proc}}$ olmak üzere işlemcinin veri kaçırmadan (No Overrun) çalışması için:

$$T_{\text{proc}}(\text{Buffer}_0) \le T_{\text{fill}}(\text{Buffer}_1) = \frac{\text{Buffer\_Size\_Bytes}}{BW_{\text{peripheral}}}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
CPU'yu çevre birimlerinin yavaş saat hızlarında döngüsel bekletmelerden kurtarmak ve acil donanımsal olaylara mikrosaniye altında deterministik tepki vermek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **CPU Meşguliyetini Bitirdi:** Yoklama (Polling) yerine DMA kullanılarak CPU yükü $\%98$'den $\%1.8$'e indirildi ($54\times$ verim artışı).
- **Veri Kaybını Önledi:** Ping-Pong çift tamponlama ile yüksek hızlı 50 MHz SPI akışlarında donanımsal veri taşması (Overrun) sıfırlandı.
- **Öncelik Hiyerarşisi:** NVIC öncelik gruplaması ile güvenlik-kritik kesmelerin anında devreye girmesi sağlandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Önbellek Tutarlılığı (Cache Coherency):** DMA doğrudan RAM'e yazdığında CPU L1/L2 Cache ile RAM arasında uyumsuzluk olabilir (Cache Invalidation gerektirir).
- **Yarış Durumları (Race Conditions):** Ana döngü ve ISR aynı bayrağa erişirken atomik işlemler (`std::atomic` / `__disable_irq()`) kullanılmazsa veri bozulur.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Polling (Yoklama):** Basit sistemlerde kullanılabilir fakat çoklu sensör ve gerçek zamanlı araç kontrolünde tamamen yetersizdir.
- **Dedicated Hardware Coprocessors:** Veri transferini ana MCU yerine özel FPGA veya AXI-Stream donanımlarına devretmek.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **ISR (Interrupt Service Routine)** | Donanım kesmesi tetiklendiğinde otomatik çalışan özel fonksiyon. |
| **NVIC** | ARM Cortex çekirdeklerinde kesme önceliklerini ve vektörlerini yöneten donanım modülü. |
| **DMA (Direct Memory Access)** | CPU müdahalesi olmaksızın çevre birimleri ile RAM arasında veri taşıyan donanım kanalı. |
| **Ping-Pong Buffer** | DMA bir tamponu doldururken CPU'nun diğer tamponu işlediği çift tamponlama yapısı. |
| **Preemption / Nesting** | Yüksek öncelikli bir kesmenin, çalışmakta olan daha düşük öncelikli kesmeyi durdurup önce çalışması. |
| **Vector Table** | Belleğin başlangıcında tüm kesme servis fonksiyonlarının adreslerini tutan işaretçi tablosu. |
| **SPI (Serial Peripheral Interface)** | 4 telli (MOSI, MISO, SCK, CS) tam çift yönlü (Full-Duplex) yüksek hızlı seri veri yolu. |
| **I2C (Inter-Integrated Circuit)** | 2 telli (SDA, SCL) çok master'lı düşük hızlı çevre birimi veri yolu. |
| **Half-Transfer Interrupt** | DMA tamponunun yarısı dolduğunda tetiklenen kesme sinyali. |
| **Atomic Operation** | Kesintiye uğramadan tek bir döngüde tamamlanan bölünemez işlem. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %96+ CPU yük tasarrufu sağlayan sıfır-kopyalı DMA   | • Hata ayıklamanın (Debugging) karmaşık olması        |
| • Donanım seviyesinde nanometre gecikmeli NVIC        | • Cache tutarlılığı (Coherency) yönetimi gerekliliği  |
| • 50 MHz yüksek hızlı SPI sensör veri pompası         |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tesla FSD ve BMB modüllerinde mikro-gecikmeli       | • ISR içinde uzun işlem yapılması durumunda sistemin  |
|   sensör füzyonu verimliliği                          |   diğer kesmeleri kaçırması (Interrupt Starvation)   |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & Donanım Veri Akışı

```
+------------------------+                        +------------------------+
|   50 MHz SPI Sensör    | -- Hardware Stream --> |   DMA Controller       |
|   (İvmeölçer / Jiroskop|                        |   (Channel 0)          |
+------------------------+                        +------------------------+
                                                              |
                                               +--------------+--------------+
                                               |                             |
                                               v                             v
                                     +-------------------+         +-------------------+
                                     |   Memory Buffer 0 |         |   Memory Buffer 1 |
                                     |   (DMA Dolduruyor)| <=====> |   (CPU İşliyor)   |
                                     +-------------------+         +-------------------+
                                               |
                                     (Transfer Complete)
                                               |
                                               v
                                     +-------------------+
                                     |   NVIC Controller | ---> CPU Core Wakeup
                                     |   (Trigger ISR)   |      (Zero-Copy Read)
                                     +-------------------+
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana kesme akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
