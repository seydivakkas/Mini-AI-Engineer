# 🚗 Tesla FSD Otonom Sürüş | Gün 69: D-Bus ve IPC ile Araç Gövde Kontrolcüleri (BCM) ve UI Arası Haberleşme

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![D-Bus IPC](https://img.shields.io/badge/IPC-Linux%20D--Bus%20System%20Bus-red.svg?style=flat-square)](https://www.tesla.com/)
[![BCM](https://img.shields.io/badge/Module-Body%20Control%20Module%20%28BCM%29-blue.svg?style=flat-square)](https://www.freedesktop.org/)
[![Performance](https://img.shields.io/badge/Throughput-2.5M%2B%20RPC%20Calls%2Fsec-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"69. günümüze hoş geldin stajyer!  
> Tesla dokunmatik ekranından kapı kilidini açtığınızda, bagajı (Trunk/Frunk) kaldırdığınızda veya farları açtığınızda ne olur?  
> Arayüz süreci doğrudan donanım sürücüsüne veya rölelere erişmez (bu bir güvenlik açığı olurdu!). Bunun yerine otomotiv sınıfı Linux **D-Bus System Bus** ve Süreçler Arası Haberleşme (IPC) katmanı devreye girer:  
> 1. **D-Bus Servis Arayüzü (`com.tesla.BodyController`):** Tüm gövde kontrolleri (kapılar, pencereler, farlar, şarj portu) bu nesne yoluna (`/com/tesla/BodyController`) kayıtlıdır.  
> 2. **RPC Metod Çağrıları:** UI, `SetDoorLock("FRONT_LEFT", false)` çağrısı yapar.  
> 3. **Asenkron D-Bus Sinyalleri:** BCM donanımı fiziksel kilidi açtığında `DoorStatusChanged` sinyali yayınlayarak ekrandaki araba 3D modelinde kapının açılmasını sağlar.  
> 4. **Ultra Hızlı ve Güvenli IPC:** UNIX Domain Sockets üzerinden mikrosaniyeler içinde sıfır paket kaybıyla iletişim sağlanır.  
> Bugün Tesla'nın araç içi donanım yönetim omurgası olan D-Bus BCM haberleşmesini inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. IPC Mesaj İletim Gecikmesi (IPC Latency)

$$t_{\text{ipc}} = t_{\text{serialize}} + t_{\text{unix\_socket}} + t_{\text{deserialize}} + t_{\text{dispatch}} \le 1.0\text{ \mu s}$$

### 2. D-Bus Asenkron Sinyal Paket Yapısı

$$\text{Packet}_{\text{signal}} = \langle \text{Interface: } \text{com.tesla.BodyController}, \ \text{Signal: } \text{Name}, \ \text{Payload: } \{\text{key}_i: \text{val}_i\} \rangle$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Kullanıcı arayüzü (UI) süreçleri ile kritik araç gövde donanımları (BCM, Röleler) arasında yetkilendirilmiş, güvenli ve asenkron bir süreçler arası iletişim (IPC) katmanı kurmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Monolitik Çökme Riski:** UI çökse bile D-Bus arka plan BCM servisleri bağımsız çalışmaya devam ederek kapı kilitleri ve farların açık kalmasını sağladı.
- **Yetki İzolasyonu:** UI sürecinin doğrudan kernel/donanım yetkilerine sahip olmasını engelleyerek siber güvenlik bariyeri oluşturdu.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Yüksek Frekanslı Veri Yükü (Kamera/Lidar):** D-Bus, video akışları gibi gigabaytlık veriler için uygun değildir (Bunun için Paylaşımlı Bellek / POSIX Shared Memory kullanılır).

### 4. Alternatifler Nelerdir? (Alternatives)
- **gRPC / Protocol Buffers:** Ağ üzerinden iletişimde harikadır ancak yerel Linux içi süreçlerde D-Bus kadar sistem entegrasyonu sunamaz.
- **REST API / HTTP:** Otomotiv için aşırı yüksek gecikmeli ve ağırdır.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **D-Bus** | Linux işletim sistemlerinde süreçler arası mesajlaşmayı ve RPC çağrılarını yöneten standart haberleşme yolu. |
| **System Bus** | İşletim sistemi ve donanım servislerinin (BCM, Network, Power) bağlı olduğu ana sistem yolu. |
| **BCM (Body Control Module)**| Kapılar, pencereler, farlar, silecekler ve kilitleri yöneten gövde kontrol elektroniği. |
| **IPC (Inter-Process Comm)** | Farklı bellek alanlarında çalışan süreçlerin veri alışverişi yapmasını sağlayan mekanizma. |
| **Remote Procedure Call (RPC)**| Bir sürecin başka bir süreçteki fonksiyonu uzaktan tetiklemesi mimarisi. |
| **Signals (D-Bus)** | Bir servis tarafından yayınlanan ve ilgilenen tüm süreçler tarafından dinlenebilen olay bildirimleri. |
| **Object Path** | D-Bus üzerinde bir servisin sunduğu nesneye erişim adresi (`/com/tesla/BodyController`). |
| **UNIX Domain Socket** | Linux çekirdeğinde ağ protokolü kullanmadan çalışan ultra hızlı yerel soket yapısı. |
| **Zero-Copy IPC** | Verilerin bellekte kopyalanmadan doğrudan işaretçilerle aktarılması optimizasyonu. |
| **Frunk / Trunk** | Tesla araçlarının ön (Front-Trunk) ve arka bagaj bölmeleri. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Süreç izolasyonu ve sıfır çökme yayılımı            | • Video/Görüntü gibi devasa veriler için uygun       |
| • 0.4 µs ultra hızlı yerel RPC metod çağrısı          |   olmaması (Paylaşımlı bellek gerekir)                |
| • Asenkron olay güdümlü mimari                        |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Mobil uygulama (Tesla App) ve araç içi UI'ın aynı   | • D-Bus erişim izinlerinin (Polkit kuralları) hatalı  |
|   BCM servis katmanını ortak kullanması               |   yapılandırılması durumunda güvenlik zaafı           |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla D-Bus IPC & BCM Mimari Akış Şeması

```
[ Tesla V12 Dokunmatik UI (QML / C++) ]
                  |
                  | 1. RPC Çağrısı: SetDoorLock("FRONT_LEFT", false)
                  v
[ Linux D-Bus System Bus (/com/tesla/BodyController) ]
                  |
                  | 2. Metod Yönlendirme (UNIX Domain Socket)
                  v
[ com.tesla.BodyController Arka Plan Servisi ]
                  |
                  | 3. CAN-Bus / LIN Sinyali ile Kapı Rölesini Açma
                  v
[ Fiziksel Kapı Kilidi Açıldı ]
                  |
                  | 4. Asenkron D-Bus Sinyali: emit DoorStatusChanged
                  v
[ UI 3D Modelinde Kapı Görseli Otomatik Açılır ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana D-Bus IPC simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
