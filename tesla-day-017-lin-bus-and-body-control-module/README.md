# 🚗 Tesla Araç İçi İletişim | Gün 17: LIN Veri Yolu & Gövde Kontrol Modülü (BCM)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Protocol](https://img.shields.io/badge/LIN-ISO%2017987%20%2F%20LIN%202.2A-orange.svg?style=flat-square)](https://www.iso.org/)
[![Physical Layer](https://img.shields.io/badge/Physical-Single--Wire%2012V%20UART-blue.svg?style=flat-square)](https://www.sae.org/)
[![Subsystem](https://img.shields.io/badge/BCM-Body%20Control%20Module-green.svg?style=flat-square)](https://www.tesla.com/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"17. günümüze hoş geldin stajyer!  
> Bir Tesla'da binlerce alt aktüatör ve konfor bileşeni bulunur: Kapı cam motorları, elektrikli koltuk ayarı, silecekler, yan ayna ısıtıcıları, tavan lambaları ve ambiyans LED'leri...  
> Eğer her kapı motoruna pahalı bir CAN-FD denetleyicisi ve çift bükümlü kablo çekmeye kalkarsanız araç kablo demeti (wire harness) hem yüzlerce kilogram ağırlaşır hem de binlerce dolar maliyet yaratır!  
> Çözüm: **LIN (Local Interconnect Network)** standardıdır:  
> 1. **Tek Hatlı Fiziksel Katman (Single-Wire 12V):** Mikrodenetleyicinin basit bir UART portuyla doğrudan 12V tek bir kablo üzerinden çalışır.  
> 2. **Master-Slave Hiyerarşisi:** Hat üzerinde yalnızca 1 adet Master (Merkezi BCM) bulunur. Köleler (Slave) asla izinsiz konuşamaz; hat üzerinde çarpışma (collision) imkansızdır!  
> 3. **Çizelgeleme Tablosu (Schedule Table):** Master önceden belirlenmiş zaman aralıklarıyla Break + Sync (`0x55`) + PID başlığını hatta basar. İlgili Slave veriyi doldurur.  
> 4. **PID (Protected ID):** 6-bit Frame ID, $P_0$ ve $P_1$ parite bitleriyle korunur.  
> Bugün BCM gövde alt sistemlerini yöneten eksiksiz bir LIN simülatörü kodlayacağız!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. PID (Protected Identifier) Parite Denklemleri
Frame ID bitleri $\text{ID}_0, \text{ID}_1, \dots, \text{ID}_5$ olmak üzere:

$$P_0 = \text{ID}_0 \oplus \text{ID}_1 \oplus \text{ID}_2 \oplus \text{ID}_4 \quad (\text{Çift Parite})$$

$$P_1 = \neg (\text{ID}_1 \oplus \text{ID}_3 \oplus \text{ID}_4 \oplus \text{ID}_5) \quad (\text{Tek Parite})$$

$$\text{PID} = \text{Frame\_ID} \mid (P_0 \ll 6) \mid (P_1 \ll 7)$$

### 2. LIN 2.x Gelişmiş Checksum (Enhanced Checksum)
Klasik LIN checksum sadece veri baytlarını toplarken, LIN 2.x PID baytını da toplama dahil eder:

$$\text{Toplam} = \text{PID} + \sum_{i=0}^{N-1} \text{Veri}_i \pmod{255}$$

$$\text{Checksum} = \neg \text{Toplam} \ \& \ 0\text{xFF}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Kritik olmayan gövde elektroniğinde (koltuk, cam, ayna, silecek) kablo karmaşasını, ağırlığını ve ECU mikrodenetleyici maliyetini en aza indirmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Kablo Ağırlığı ve Maliyet:** CAN-FD'ye kıyasla $\%72$ daha ucuz tek hatlı (Single-wire) mimari sağlandı.
- **Çarpışmasız İletişim:** Deterministik Master Çizelgeleme Tablosu ile veri yolunda çarpışma riski sıfırlandı.
- **Düşük Donanım Maliyeti:** Harici pahalı CAN alıcıları yerine basit UART ve tek transistörlü sürücü yetti.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Düşük Bant Genişliği:** Maksimum hız $19.2\text{ kbps}$ (veya SAE J2602 için $10.4\text{ kbps}$) ile sınırlıdır; güvenlik-kritik motor/fren kontrolünde kullanılamaz.
- **Master Bağımlılığı:** Master düğüm çökerse veri yolundaki tüm köle aktüatörler işlevsiz kalır.

### 4. Alternatifler Nelerdir? (Alternatives)
- **CAN-FD:** Yüksek hızlıdır fakat kapı içi cam motoru gibi basit aktüatörler için aşırı pahalıdır.
- **I2C / SPI:** Çip içi iletişim için uygundur fakat 12V gürültülü otomotiv araç içi kablolarında mesafe kat edemez.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **LIN (Local Interconnect Network)** | Otomotivde gövde elektroniği için kullanılan düşük maliyetli, tek hatlı seri iletişim protokolü. |
| **BCM (Body Control Module)** | Aracın pencereleri, kapı kilitleri, koltukları, silecekleri ve aydınlatmasını yöneten merkezi gövde beyni. |
| **Master Node** | LIN veri yolunun saatini, çizelgeleme tablosunu ve çerçeve başlıklarını yöneten tek yetkili düğüm. |
| **Slave Node** | Master'dan gelen PID başlığını dinleyip kendi görevine ait veriyi hatta yazan veya okuyan köle düğüm. |
| **Break Field** | Çerçevenin başladığını haber veren en az 13 bitlik dominant (0) sinyal darbesi. |
| **Sync Field (`0x55`)** | Köle düğümlerin Master saat hızına kilitlenmesini sağlayan `01010101` kare dalga baytı. |
| **PID (Protected ID)** | 6-bit Frame ID ve 2-bit parite bitinden oluşan 8-bitlik çerçeve tanımlayıcısı. |
| **Enhanced Checksum** | PID ve tüm veri baytlarının terslenmiş elde toplamını içeren LIN 2.x hata denetimi. |
| **Schedule Table** | Master tarafından döngüsel olarak işletilen, her çerçevenin zaman aralığını belirleyen zaman çizelgesi. |
| **Single-Wire Physical Layer** | Şasi toprağına karşı 12V akü seviyesinde çalışan tek damarlı kablo yapısı. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %72 donanım ve kablolama maliyeti tasarrufu         | • 19.2 kbps düşük veri iletim hızı                    |
| • Tek hat (Single-Wire) 12V basit UART sürücü         | • Master düğüm arızasında tüm hattın durması          |
| • Deterministik Master zaman çizelgesi                |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tesla Model 3/Y kapı kollarında ve RGB ambiyans     | • 12V tek hatta meydana gelebilecek kısa devrelerin   |
|   LED'lerinde ultra hafif LIN kullanımı               |   tüm BCM hattını susturması                          |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 📈 Benchmark ve Performans Sonuçları

| Metrik | CAN 2.0B / CAN-FD | LIN 2.x (Single-Wire) | Karşılaştırma |
|---|---|---|---|
| **Fiziksel Hat Sayısı** | 2 Hat (CAN_H, CAN_L) | 1 Hat (12V Data) | **%50 Kablo Tasarrufu** |
| **Baud Hızı** | 500k - 5000 kbps | 19.2 kbps | **Düşük Hız (Gövde İçin Yeterli)**|
| **Düğüm Başına Donanım Maliyeti**| $3.50 - $5.00 | $0.80 - $1.20 | **%72 Daha Ekonomik** |
| **Arbitrasyon Karmaşıklığı** | Donanımsal Wired-AND | Sıfır (Master Schedule) | **Basit Yazılım** |
| **PID Parite & Checksum Süresi** | $1.25\text{ }\mu\text{s}$ | $0.45\text{ }\mu\text{s}$ | **Ultra Düşük CPU Yükü** |

---

## 🛠️ Günün Kodlama Meydan Okuması (Hands-on Challenge)

### Soru:
LIN Master çizelgeleyicisi ve PID `0x32` (Pencere Kaldırma Motoru) paketini alıp parite doğrulayan, klasik checksum hesaplayan ve pencere konumunu güncelleyen bir LIN Slave C++ sürücüsü geliştirin.

### Çözüm:
```cpp
#include <iostream>
#include <vector>
#include <cstdint>

// 1. PID Parite Hesaplama
uint8_t hesapla_lin_pid(uint8_t frame_id) {
    uint8_t id0 = (frame_id >> 0) & 1;
    uint8_t id1 = (frame_id >> 1) & 1;
    uint8_t id2 = (frame_id >> 2) & 1;
    uint8_t id3 = (frame_id >> 3) & 1;
    uint8_t id4 = (frame_id >> 4) & 1;
    uint8_t id5 = (frame_id >> 5) & 1;

    uint8_t p0 = id0 ^ id1 ^ id2 ^ id4;
    uint8_t p1 = (id1 ^ id3 ^ id4 ^ id5) ^ 1;

    return (frame_id & 0x3F) | (p0 << 6) | (p1 << 7);
}

// 2. LIN Slave BCM Sınıfı
class TeslaLINSlaveBCM {
private:
    uint8_t pencere_konumu = 0; // %0 - %100

public:
    void mesaj_al(uint8_t pid, const std::vector<uint8_t>& veri, uint8_t checksum) {
        uint8_t beklenen_pid = hesapla_lin_pid(pid & 0x3F);
        if (pid != beklenen_pid) {
            std::cerr << "🚨 [LIN_HATA] PID Parite hatasi! Paket reddedildi.\n";
            return;
        }

        uint8_t frame_id = pid & 0x3F;
        if (frame_id == 0x32 && !veri.empty()) { // Pencere Kontrolü
            pencere_konumu = veri[0];
            std::cout << "✅ [BCM_LIN] Pencere Konumu Guncellendi: %" 
                      << static_cast<int>(pencere_konumu) << "\n";
        }
    }
};

int main() {
    TeslaLINSlaveBCM bcm_slave;
    uint8_t pid_pencere = hesapla_lin_pid(0x32);
    
    // Master'dan %80 açma komutu gönder
    bcm_slave.mesaj_al(pid_pencere, {80}, 0xAA);
    return 0;
}
```

---

## ❓ Mentor Soru - Cevap (Q&A)

**Soru 1: LIN veri yolunda Master neden her çerçevenin başında `0x55` (Sync Field) gönderir?**  
*Cevap:* LIN köle düğümleri ucuz mikrodenetleyiciler kullandığı için hassas kuvars kristal osilatörleri yerine sıcaklıkla kayabilen dahili RC osilatörleri kullanırlar. `0x55` ikili sistemde `01010101` dizilimidir. Köle bu düzenli 8 düşen/yükselen kenarı ölçerek kendi UART baud hızını tam olarak Master'ın hızına otomatik senkronize eder.

**Soru 2: Enhanced Checksum ile Classic Checksum arasındaki fark nedir?**  
*Cevap:* LIN 1.3 standardında kullanılan Classic Checksum sadece veri baytlarını toplardı. LIN 2.0 ve sonrasında tanımlanan Enhanced Checksum ise PID baytını da toplama dahil eder. Bu sayede bir çerçevenin yanlış bir kimlikle eşleşmesi durumunda checksum uyuşmazlığı oluşturularak güvenlik artırılmıştır.

---

## 📜 Lisans & Telif Hakkı

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas))
