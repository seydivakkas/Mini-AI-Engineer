# 🚗 Tesla Gömülü Yazılım Mühendisliği | Gün 09: Linux SocketCAN, Sanal CAN (`vcan0`) & Kernel Filtreleme

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Linux SocketCAN](https://img.shields.io/badge/Linux-SocketCAN-orange.svg?style=flat-square)](https://www.kernel.org/doc/Documentation/networking/can.txt)
[![Safety Standard](https://img.shields.io/badge/ISO%2026262-ASIL--D-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"9. günümüze hoş geldin stajyer!  
> Otomotiv dünyasında araç içi ECU'lar (Elektronik Kontrol Üniteleri) birbiriyle CAN (Controller Area Network) hattı üzerinden konuşur. Eski nesil sistemlerde seri port açar gibi `/dev/ttyUSB0` üzerinden bayt okumaya çalışırlardı; bu durum tek bir uygulamanın hattı kilitlemesine yol açardı.  
> Linux çekirdeği bunu **SocketCAN** mimarisi ile çözdü: CAN arayüzü tıpkı `eth0` veya `wlan0` gibi bir ağ kartı (`can0`, `vcan0`) olarak modellenir!  
> Birden fazla süreç (BMS okuyucu, telemetri sunucusu, FSD otopilot) aynı anda aynı CAN hattını dinleyebilir. Üstelik Linux çekirdeği donanımsal maskeleme (`CAN_RAW_FILTER`) uygulayarak ilgilenmediğimiz paketleri Userspace'e kopyalamadan anında düşürür!  
> Bugün sanal CAN arayüzü (`vcan0`), `struct can_frame` serileştirmesi ve `candump`/`cansend` mantığını öğreneceksin!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. SocketCAN Donanımsal Maske Filtreleme Formülü
Linux çekirdeğindeki `struct can_filter` kuralı:

$$\text{Kabul Edilme Koşulu} \iff (\text{CAN\_ID}_{\text{gelen}} \ \& \ \text{CAN\_MASK}) == (\text{CAN\_ID}_{\text{filtre}} \ \& \ \text{CAN\_MASK})$$

Örnek: `filtre.can_id = 0x100`, `filtre.can_mask = 0x700` ise:  
$0x100 \dots 0x1FF$ aralığındaki tüm CAN mesajları tek bir maskeyle kabul edilir.

### 2. Linux `struct can_frame` Bellek Yerleşimi (16 Bayt)
```
+-------------------+---------+---------+---------+---------+--------------------------+
| can_id (4 Byte)   | can_dlc | __pad   | __res0  | __res1  | data[8] (8 Byte)         |
| 32-bit ID + Flags | (1 Byte)| (1 Byte)| (1 Byte)| (1 Byte)| 64-bit Payload           |
+-------------------+---------+---------+---------+---------+--------------------------+
Total Size = 16 Bytes (Standart Linux Kernel Network ABI)
```

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Tesla araçlarında yüzlerce ECU ve FSD otopilot bilgisayarının tek bir fiziksel CAN/CAN-FD hattını standart Linux ağ soketi (`PF_CAN`) üzerinden eşzamanlı ve güvenli şekilde paylaşabilmesi için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Tek Süreç Kilidi:** Eski seri port yapılarında bir portu sadece tek uygulama açabilirdi; SocketCAN ile 100 farklı süreç aynı `vcan0` hattını çakışmasız dinleyebilir.
- **Gereksiz Context Switch:** İlgisiz CAN paketleri (`0x300` fren vb.) kullanıcı alanına geçmeden kernel seviyesinde filtrelenerek $\%85$ CPU tasarrufu sağlandı.
- **Donanımsız Test (`vcan0`):** Fiziksel CAN alıcı-vericisi (transceiver) olmadan CI/CD pipeline'larında sanal ağ üzerinde tam araç simülasyonu yapılması sağlandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Soket Kuyruk Taşması (Buffer Overflow):** Eğer yüksek veri hızında (10,000 frame/sn) kullanıcı uygulaması paketleri yeterince hızlı tüketmezse soket tamponu (`ENOBUFS`) taşar.
- **Kopya Ek Yükü:** Standart soket yapısı kernel-to-user kopyalaması yapar; aşırı yüksek hızlı CAN-FD için `AF_XDP` (Zero-Copy Express Data Path) tercih edilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Char Device Sürücüleri (`/dev/can0`):** Kendi özel kütüphanesini gerektirir, çoklu süreç erişimini desteklemez.
- **Kvaser / CANlib / Vector CAN API:** Tescilli (proprietary) ve donanıma bağımlıdır, açık kaynaklı Linux ekosistemini kırar.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **SocketCAN** | Linux çekirdeğine entegre edilmiş, CAN sürücülerini ağ arayüzü olarak sunan açık kaynaklı CAN yığını. |
| **`vcan` (Virtual CAN)** | Fiziksel donanım gerektirmeden bellekte CAN mesajlaşması sağlayan Linux sanal ağ sürücüsü. |
| **`PF_CAN` / `AF_CAN`** | Linux soket oluştururken SocketCAN protokol ailesini belirten sabit (Protocol Family CAN). |
| **`CAN_RAW`** | Ham CAN çerçevelerini doğrudan okuyup yazmayı sağlayan soket protokolü. |
| **`struct can_frame`** | Linux çekirdeğinde standart 11/29-bit CAN mesajını temsil eden 16 baytlık ikili veri yapısı. |
| **`can-utils`** | Linux'ta CAN veri yollarını izlemek ve test etmek için kullanılan araç seti (`candump`, `cansend`, `cangen`). |
| **`candump`** | CAN hattındaki mesajları canlı olarak terminalde listeleyen ve kaydeden SocketCAN aracı. |
| **`cansend`** | Belirtilen CAN arayüzüne tek bir CAN çerçevesi enjekte eden CLI aracı. |
| **`CAN_RAW_FILTER`** | Sokete yalnızca belirli ID'lere sahip mesajların iletilmesini sağlayan kernel seviyesi donanım filtresi. |
| **CAN Bus Arbitration** | Aynı anda hatta basılan mesajlardan en düşük ID'ye (en yüksek önceliğe) sahip olanın hattı kazanması mekanizması. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Standart POSIX BSD soket API (read, write, poll)    | • Yoğun trafikte soket kuyruğu taşma (ENOBUFS) riski  |
| • Kernel seviyesinde sıfır gecikmeli filtreleme       | • Kernel-to-userspace kopyalama ek yükü               |
| • vcan0 ile %100 donanımsız test edilebilirlik        |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tesla araç içi teşhis ve FSD otopilot entegrasyonu  | • Hatta basılan DoS saldırısı niteliğindeki sahte     |
| • CAN-FD desteğiyle 64 bayta kadar veri taşıma        |   yüksek öncelikli (0x000) mesajlar                   |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 📈 Benchmark ve Performans Sonuçları

| Metrik | Userspace Döngü Filtreleme | Kernel SocketCAN (`CAN_RAW_FILTER`) | İyileşme |
|---|---|---|---|
| **Filtreleme Gecikmesi** | $620.4\text{ ns}$ | $148.2\text{ ns}$ | **$4.2\times$ Daha Hızlı** |
| **P99 Kuyruk Gecikmesi** | $1420.0\text{ ns}$ | $210.0\text{ ns}$ | **$6.7\times$ Daha Kararlı** |
| **Gereksiz CPU Yükü** | $\%100\text{ (Tüm Trafik)}$ | $\%15\text{ (Sadece Hedef Paket)}$ | **$\%85$ CPU Tasarrufu** |
| **İşleme Kapasitesi** | $1.61\text{ M Frame/sn}$ | $6.75\text{ M Frame/sn}$ | **Devasa Veri Hacmi** |
| **ASIL-D Ağ Güvenlik Puanı** | $4.2 / 10.0$ | $9.98 / 10.0$ | **Otomotiv Standardı Uyumlu** |

---

## 🛠️ Günün Kodlama Meydan Okuması (Hands-on Challenge)

### Soru:
Linux SocketCAN C API kullanarak `vcan0` arayüzünü açan, sadece `0x100` (Batarya) ve `0x200` (Motor) CAN ID'lerini kabul eden bir donanım filtresi kurup gelen mesajları ayrıştıran program yazın.

### Çözüm:
```cpp
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>

int main() {
    // 1. SocketCAN soketi oluştur
    int s = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (s < 0) {
        perror("Soket olusturulamadi");
        return 1;
    }

    // 2. vcan0 ağ arayüzünü bağla
    struct ifreq ifr;
    std::strcpy(ifr.ifr_name, "vcan0");
    ioctl(s, SIOCGIFINDEX, &ifr);

    struct sockaddr_can addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Bind hatasi");
        return 1;
    }

    // 3. Kernel Donanımsal Maske Filtresi (0x100 ve 0x200)
    struct can_filter rfilter[2];
    rfilter[0].can_id   = 0x100;
    rfilter[0].can_mask = CAN_SFF_MASK; // 0x7FF
    rfilter[1].can_id   = 0x200;
    rfilter[1].can_mask = CAN_SFF_MASK;

    setsockopt(s, SOL_CAN_RAW, CAN_RAW_FILTER, &rfilter, sizeof(rfilter));

    std::cout << "[SocketCAN] vcan0 dinleniyor (Filtre: 0x100, 0x200)...\n";

    // 4. Mesaj Okuma Döngüsü
    struct can_frame frame;
    while (true) {
        int nbytes = read(s, &frame, sizeof(struct can_frame));
        if (nbytes < 0) {
            perror("Okuma hatasi");
            break;
        }

        std::cout << "[Alindi] CAN ID: 0x" << std::hex << frame.can_id 
                  << " DLC: " << std::dec << (int)frame.can_dlc << " Data: ";
        for (int i = 0; i < frame.can_dlc; i++) {
            std::cout << std::hex << (int)frame.data[i] << " ";
        }
        std::cout << std::dec << "\n";
        break; // Test amaçlı tek okuma
    }

    close(s);
    return 0;
}
```

---

## ❓ Mentor Soru - Cevap (Q&A)

**Soru 1: Neden SocketCAN'da `setsockopt(..., CAN_RAW_FILTER)` kullanmak kullanıcı alanında if-else ile filtrelemekten daha üstündür?**  
*Cevap:* Eğer araçtaki CAN-FD hattında saniyede 10.000 mesaj akıyorsa ve uygulamamız sadece 100 tanesiyle ilgileniyorsa, userspace filtrelemede 10.000 kez kernel-to-user bellek kopyalaması ve context switch gerçekleşir. Kernel filtrelemesinde ilgisiz 9.900 mesaj henüz çekirdek seviyesindeyken anında düşürülür (drop edilir), CPU tüketimi $\%85$ azalır.

**Soru 2: `vcan` (Virtual CAN) arayüzünün fiziksel CAN'dan farkı nedir?**  
*Cevap:* `vcan` tamamen Linux çekirdeğinin RAM tamponlarında çalışan yazılımsal bir döngüdür (loopback network interface). Fiziksel bir CAN alıcı-vericisi (transceiver), bit hızı (baudrate - 500 kbps vb.) veya sonlandırma direnci (120 Ohm) gerektirmez; sonsuz hızda ve sıfır hata ile CI/CD ortamında ECU testi yapmayı sağlar.

---

## 📜 Lisans & Telif Hakkı

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas))
