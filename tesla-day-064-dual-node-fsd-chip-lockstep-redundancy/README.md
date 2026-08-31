# 🚗 Tesla FSD Otonom Sürüş | Gün 64: Çift Kanallı Güvenlik ve FSD HW Çip Yedekliliği (Redundancy & Lockstep)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Hardware Redundancy](https://img.shields.io/badge/Hardware-Dual%20SoC%20%28Node%20A%20%2B%20Node%20B%29-red.svg?style=flat-square)](https://www.tesla.com/)
[![Lockstep](https://img.shields.io/badge/Architecture-Lockstep%20Voting%20Arbiter-blue.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/ASIL--D-Failover%20%26%20Safe%20Stop-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"64. günümüze hoş geldin stajyer!  
> Tesla FSD Bilgisayarı (HW3 ve HW4) anakartını açtığınızda göreceğiniz en çarpıcı şey, yan yana duran iki devasa çiptir: **Node A ve Node B**.  
> Bu çipler ana-yedek (Master-Slave) değil, **Tam Eşzamanlı Çift Bağımsız Bilgisayar (Dual Independent Redundant Compute)** olarak çalışır:  
> 1. **Bağımsız Güç ve Girdi Hatları:** Her iki çip de tüm 8 kameradan aynı görüntüleri ayrı veri yollarıyla alır ve kendi bağımsız NPU çekirdeklerinde çalıştırır.  
> 2. **Karar Arabulucusu (Hardware Arbiter & Voting):** İki çip çıkardıkları direksiyon ve ivme komutlarını karşılaştırır ($|\delta_A - \delta_B| \le 0.05\text{ rad}$, $|a_A - a_B| \le 0.5\text{ m/s}^2$).  
> 3. **Tam Uzlaşı (Full Consensus):** İki çip anlaştığında iki komutun ortalaması aktüatörlere gönderilir.  
> 4. **Anlık Failover:** Bir çip donarsa veya watchdog sinyali kesilirse diğer çip tek bir mikrosaniye bile kaybetmeden kontrolü devralır.  
> 5. **Ayrışma Kalkanı (Discrepancy Safe Stop):** Eğer iki çip de çalışıyor fakat farklı kararlar veriyorsa (örneğin biri sağa biri sola kırmaya çalışıyorsa) sistem direksiyonu düz hatta kilitler ve yumuşakça emniyet şeridine durur.  
> Bugün Tesla'nın donanımsal çip güvenliğinin kalbi olan Karar Arabulucusunu inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Çift Düğüm Uzlaşı Kriteri (Consensus Criteria)

$$\text{Uzlaşı Var} \iff \left( |\delta_A - \delta_B| \le 0.05\text{ rad} \right) \land \left( |a_A - a_B| \le 0.50\text{ m/s}^2 \right)$$

### 2. Uygulanan Nihai Kontrol Vektörü

$$\mathbf{u}_{\text{applied}} = \begin{cases} \frac{\mathbf{u}_A + \mathbf{u}_B}{2}, & \text{Node A ve Node B Sağlıklı \& Uzlaşı Var} \\ \mathbf{u}_A, & \text{Node A Sağlıklı, Node B Arızalı (Failover)} \\ \mathbf{u}_B, & \text{Node B Sağlıklı, Node A Arızalı (Failover)} \\ \begin{bmatrix} -1.50\text{ m/s}^2 \\ 0.0\text{ rad} \end{bmatrix}, & \text{Her İki Düğüm Sağlıklı fakat Karar Ayrışması Var (Safe Stop)} \end{cases}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
FSD NPU çipinde oluşabilecek silikon seviyesindeki donanım hataları, bit çevrilmeleri (Single Event Upset) veya çip yanmalarında sürüş güvenliğini kesintisiz sürdürmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Çip Donmalarında Sıfır Gecikme:** Master-Slave geçişlerindeki 500 ms'lik devir gecikmesini sıfıra indirdi (Lockstep anında paralel).
- **Hatalı Çıkarım İzolasyonu:** Bir NPU'nun ağırlık tensöründe bozulma olduğunda ayrışma tespit edilerek yanlış manevra yapması engellendi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Güç Tüketimi:** İki tam SoC çipinin aynı anda çalışması anakartta ~72 Watt sürekli güç tüketimi oluşturur.
- **Yazılım Hatası Ortaklığı:** Eğer sinir ağı modelinde ortak bir yazılım mantık hatası varsa her iki çip de aynı yanlış kararı verebilir (Shadow Mode ve Model Distillation ile çözülür).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Tek Çip + Harici Güvenlik MCU'su (Örn. Aurix TC397):** Maliyet düşüktür ancak Seviye 4 Robotaksi için tam NPU yedekliliği sağlayamaz.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Dual-Node FSD** | Tesla HW3/HW4 anakartında yer alan iki özdeş ve bağımsız System-on-Chip (SoC). |
| **Hardware Arbiter** | İki bağımsız işlemcinin kararlarını karşılaştırıp nihai komutu aktüatörlere ileten donanımsal arabulucu. |
| **Lockstep Compute** | İki işlemcinin aynı girdileri aynı saat çevriminde paralel işleyerek sonuçları karşılaştırması. |
| **Consensus Voting** | Çoklu işlemcilerin ortak bir kararda birleşmesi oylama algoritması. |
| **Failover** | Birincil donanım çöktüğünde ikincil donanımın görevi kesintisiz devralması. |
| **Discrepancy** | İki bağımsız düğümün aynı sahneye zıt kontrol komutları üretmesi durumu. |
| **Watchdog Heartbeat** | İşlemcinin donmadığını kanıtlamak için periyodik olarak yayınladığı canlılık sinyali. |
| **Safe Stop Mode** | Ayrışma durumunda aracı düz hatta tutup kontrollü bir şekilde yavaşlatan güvenlik kalkanı. |
| **CRC32 Checksum** | Çipler arası veri aktarımında paket bozulmalarını tespit eden döngüsel artıklık denetimi. |
| **NPU Redundancy** | Yapay zeka çıkarım motorunun donanımsal olarak çiftlenmesi mimarisi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Donanım seviyesinde %100 sıfır tek nokta hatası     | • Anakart güç tüketimi ve termal soğutma maliyeti     |
| • 1 µs ultra hızlı RTOS karar oylaması                | • Yazılımsal model hatalarının her iki çipe           |
| • Anında kesintisiz failover kabiliyeti               |   ortak yansıması riski                               |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Sürücüsüz Robotaksi ve Cybercab filolarında tam     | • Fiziksel kaza anında anakartın ezilerek her iki     |
|   donanım güvenilirliği ile ticari lisans alma        |   çipin birden devre dışı kalması                     |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Çift Düğüm Karar Arabulucusu Akış Şeması

```
[ Kameralardan Gelen Görüntüler (Ayrı Veri Yolları) ]
                 |                        |
                 v                        v
        [ FSD NPU Node A ]       [ FSD NPU Node B ]
        - Steer A, Acc A         - Steer B, Acc B
                 \                        /
                  \                      /
                   v                    v
              [ FSD Hardware Arbiter (Oylama) ]
                            |
        +-------------------+-------------------+
        |                                       |
        v                                       v
[ Uzlaşı Var: |Steer_diff| <= 0.05 ]    [ Ayrışma: |Steer_diff| > 0.05 ]
        |                                       |
        v                                       v
[ Ortalamayı Uygula: (A+B)/2 ]          [ DISCREPANCY SAFE STOP ]
- Nominal Otonom Sürüş                  - Direksiyon: 0.0 (Düz Hat)
                                        - İvme: -1.5 m/s² Güvenli Duruş
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Çift Düğüm Arabulucu simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
