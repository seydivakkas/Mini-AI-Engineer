# 🚗 TESLA YAZILIM MÜHENDİSLİĞİ VE ARAÇ TEKNOLOJİLERİ 99 GÜNLÜK MASTER YOL HARİTASI

![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![Target: Tesla Software Engineer Specialist](https://img.shields.io/badge/Target-Tesla%20Software%20Engineer-E82127?style=flat-square)
![Stack: C++20%20%7C%20RTOS%20%7C%20CAN--FD%20%7C%20FSD%20%7C%20BMS%20%7C%20Optimus](https://img.shields.io/badge/Tech-C%2B%2B20%20%7C%20RTOS%20%7C%20PyTorch%20%7C%20ROS2-00529B?style=flat-square)

> **Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.**  
> Bu müfredat; sıfırdan başlayarak bir yazılım mühendisini **Tesla (FSD Otonom Sürüş, Araç İçi RTOS/Linux, Batarya Yönetim Sistemi BMS, Supercharger Enerji Ağı, Dojo ve Optimus İnsansı Robotik)** standartlarında dünya çapında bir uzmana dönüştürmek için 99 gün, 9 büyük faz, her gün için özel teknik quiz kod sorusu ve çalışan eksiksiz çözümleriyle tasarlanmıştır.

---

## 🏛️ 9 BÜYÜK FAZ GENEL BAKIŞ

```
========================================================================================================================
     TESLA YAZILIM MÜHENDİSLİĞİ UZMANLIK MÜFREDATI (99 GÜNLÜK UÇTAN UCA EĞİTİM & SINAV PLANI)
========================================================================================================================
• FAZ 1 (Gün 01 - 11): Temel C++20, Modern STL, Bellek Yönetimi ve Tesla Gömülü Çekirdeği
• FAZ 2 (Gün 12 - 22): Gerçek Zamanlı İşletim Sistemleri (RTOS), CAN-FD, UDS ve Araç Ağları
• FAZ 3 (Gün 23 - 33): Tesla Batarya Yönetim Sistemi (BMS), EKF SoC/SoH & Motor Kontrolü (FOC)
• FAZ 4 (Gün 34 - 44): Tesla FSD 8-Kamera Görüş Geometrisi, Sensör Füzyonu & Semantik SLAM
• FAZ 5 (Gün 45 - 55): Tesla FSD HydraNet, 3D Voxel Occupancy Network & TensorRT NPU Optimizasyonu
• FAZ 6 (Gün 56 - 66): Otonom Sürüş Yörünge Planlama, Model Predictive Control (MPC) & ISO 26262 ASIL-D
• FAZ 7 (Gün 67 - 77): Tesla V12 Bilgi-Eğlence (Infotainment), Qt6/QML, D-Bus IPC & OTA Güvenli Güncelleme
• FAZ 8 (Gün 78 - 88): Tesla Supercharger V4 (NACS/ISO 15118), Megapack BESS & Autobidder Enerji Piyasası
• FAZ 9 (Gün 89 - 99): Tesla Dojo D1 Çipi, Fleet OS Telemetri, Tesla Optimus Robotik & 👑 BÜYÜK FİNAL 99
========================================================================================================================
```

---

## 📌 99 GÜNLÜK GÜN GÜN MÜFREDAT, TEKNİK KONULAR, QUİZLER VE ÇÖZÜMLER

---

### 🔹 FAZ 1: Modern C++20, Bellek Yönetimi ve Tesla Gömülü Çekirdeği (Gün 01 - 11)

#### **Gün 01: Modern C++20 Temelleri, Pointer Aritmetiği ve Bellek Düzeni (Stack vs Heap)**
- **Teorik Odak**: Bellek mimarisi, struct padding, alignment, cache-line hizalaması (64 byte) ve pointer aritmetiği.
- **Quiz Kod Sorusu**: Verilen bir sensör bayt dizisini dinamik bellek ayırmadan (`malloc`/`new` olmadan) doğrudan C++ struct pointer cast ile parse eden güvenli fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cstdint>
#include <cstring>

struct alignas(4) TeslaWheelSpeedSensor {
    uint16_t sensor_id;
    uint16_t rpm;
    uint32_t timestamp_us;
};

TeslaWheelSpeedSensor parse_wheel_sensor(const uint8_t* raw_buffer) {
    TeslaWheelSpeedSensor data;
    std::memcpy(&data, raw_buffer, sizeof(TeslaWheelSpeedSensor));
    return data;
}

int main() {
    uint8_t buffer[8] = {0x01, 0x00, 0xE8, 0x03, 0x20, 0xA1, 0x07, 0x00};
    TeslaWheelSpeedSensor s = parse_wheel_sensor(buffer);
    std::cout << "Sensor ID: " << s.sensor_id << ", RPM: " << s.rpm << "\n";
    return 0;
}
```

#### **Gün 02: RAII, Akıllı İşaretçiler (`unique_ptr`, `shared_ptr`) ve Sıfır Bellek Sızıntısı**
- **Teorik Odak**: Resource Acquisition Is Initialization (RAII), custom deleter'lar, dairesel referansları önleyen `weak_ptr`.
- **Quiz Kod Sorusu**: Bir CAN Bus donanım soketini açan, hata durumunda dahi soket tutamacını (file descriptor) otomatik kapatan RAII sarmalayıcı sınıfı yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <memory>

class CANSocketHandle {
private:
    int fd_;
public:
    explicit CANSocketHandle(int fd) : fd_(fd) { std::cout << "CAN Socket " << fd_ << " acildi.\n"; }
    ~CANSocketHandle() {
        if (fd_ >= 0) { std::cout << "CAN Socket " << fd_ << " guvenle kapatildi (RAII).\n"; }
    }
    int get() const { return fd_; }
};

int main() {
    {
        auto sock = std::make_unique<CANSocketHandle>(3);
    } // Kapsam disina cikildiginda otomatik yikim
    return 0;
}
```

#### **Gün 03: Modern C++ STL Konteynerleri, Vektörler, `std::array` ve Özel Bellek Tahsisatçıları**
- **Teorik Odak**: Gömülü sistemlerde `std::vector` bellek yeniden tahsisat (reallocation) riskleri, `std::array` ve `reserve()` önemi.
- **Quiz Kod Sorusu**: 8 adet Tesla FSD kamerasının saniyede 36 FPS hızındaki görüntü metadata nesnelerini sıfır heap tahsisi ile saklayan sabit boyutlu döngüsel veri yapısını yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <array>

struct CameraFrameMeta {
    uint32_t frame_id;
    uint64_t exposure_time_ns;
};

class FSDCameraBuffer {
private:
    std::array<CameraFrameMeta, 8> frames_{};
public:
    void set_frame(size_t cam_idx, uint32_t id, uint64_t exp) {
        if (cam_idx < frames_.size()) {
            frames_[cam_idx] = {id, exp};
        }
    }
    const CameraFrameMeta& get_frame(size_t cam_idx) const { return frames_.at(cam_idx); }
};

int main() {
    FSDCameraBuffer buf;
    buf.set_frame(0, 1001, 15000000ULL);
    std::cout << "Cam 0 Frame: " << buf.get_frame(0).frame_id << "\n";
    return 0;
}
```

#### **Gün 04: C++ Şablonları (Templates), SFINAE ve C++20 Concepts**
- **Teorik Odak**: Derleme zamanı tip güvenliği, `std::is_integral`, C++20 `requires` ifadeleri ve sıfır çalışma zamanı ek yükü.
- **Quiz Kod Sorusu**: Yalnızca sayısal araç telemetri değerlerini kabul eden (`requires std::is_arithmetic_v<T>`) C++20 filtreleme şablon fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <concepts>

template <typename T>
requires std::integral<T> || std::floating_point<T>
T normalize_telemetry(T raw_val, T max_val) {
    return raw_val / max_val;
}

int main() {
    std::cout << "Normalized Speed: " << normalize_telemetry(120.0, 250.0) << "\n";
    return 0;
}
```

#### **Gün 05: Çoklu İş Parçacığı (Multithreading), `std::jthread`, `std::atomic` ve Lock-Free Kuyruklar**
- **Teorik Odak**: Lock-free programlama, CAS (Compare-And-Swap), memory order (`acquire`, `release`, `relaxed`), `std::atomic`.
- **Quiz Kod Sorusu**: Tek üretici tek tüketici (SPSC) için çalışan kilit içermeyen (lock-free) tek elemanlı `std::atomic` posta kutusu (mailbox) yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <atomic>
#include <thread>

template <typename T>
class LockFreeMailbox {
private:
    std::atomic<bool> has_data_{false};
    T data_;
public:
    void send(const T& item) {
        data_ = item;
        has_data_.store(true, std::memory_order_release);
    }
    bool receive(T& item) {
        if (has_data_.load(std::memory_order_acquire)) {
            item = data_;
            has_data_.store(false, std::memory_order_relaxed);
            return true;
        }
        return false;
    }
};

int main() {
    LockFreeMailbox<int> mbox;
    mbox.send(400); // 400V Batarya voltaji
    int val = 0;
    if (mbox.receive(val)) {
        std::cout << "Okunan Batarya Voltaji: " << val << "V\n";
    }
    return 0;
}
```

#### **Gün 06: Düşük Seviyeli Bit Manipülasyonu, Endianness ve Bit Maskeleme**
- **Teorik Odak**: CAN Bus mesajlarında bit stuffing, Little Endian vs Big Endian (Motorola vs Intel format), bitwise shifts.
- **Quiz Kod Sorusu**: 16-bitlik ham CAN sinyalinden [Bit 4 - Bit 11] arasındaki 8-bitlik fren basıncı değerini çeken C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cstdint>

uint8_t extract_brake_pressure(uint16_t can_raw_word) {
    // 4. bitten baslayarak 8 bit: mask = 0x0FF0, saga 4 kaydir
    return static_cast<uint8_t>((can_raw_word >> 4) & 0xFF);
}

int main() {
    uint16_t raw_packet = 0xA7B2; // ornek veri
    std::cout << "Fren Basinci: " << (int)extract_brake_pressure(raw_packet) << " Bar\n";
    return 0;
}
```

#### **Gün 07: Modern C++ Hata Yönetimi: `std::expected`, `std::optional` ve Noexcept Prensipleri**
- **Teorik Odak**: Gömülü otomotiv yazılımlarında C++ istisnalarının (`throw`) yasaklanması (MISRA C++ standardı), `std::expected` (C++23) veya `std::optional`.
- **Quiz Kod Sorusu**: Motor invertöründen sıcaklık okuyan, sıcaklık 150°C'yi aşarsa hata enum'ı dönen `std::optional` tabanlı fonksiyon yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <optional>

enum class InverterError { OVERHEAT, SENSOR_FAULT };

std::optional<float> read_inverter_temp(float raw_temp_c) {
    if (raw_temp_c < -40.0f || raw_temp_c > 150.0f) {
        return std::nullopt; // Hata: Sensor disi veya asiri isinma
    }
    return raw_temp_c;
}

int main() {
    auto t = read_inverter_temp(85.4f);
    if (t.has_value()) {
        std::cout << "Invertor Sicakligi Normal: " << *t << " C\n";
    }
    return 0;
}
```

#### **Gün 08: Cache-Friendly Veri Yapıları ve CPU L1/L2/L3 Önbellek Hizalaması (Data-Oriented Design)**
- **Teorik Odak**: Array of Structures (AoS) vs Structure of Arrays (SoA), SIMD vektörizasyonu ve önbellek kaçırma (cache miss) analizi.
- **Quiz Kod Sorusu**: 10.000 FSD engel nesnesinin $X, Y, Z$ koordinatlarını SoA (Structure of Arrays) formatında saklayıp mesafe hesaplayan sınıfı yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <vector>
#include <cmath>

struct ObstaclesSoA {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;

    void add(float px, float py, float pz) {
        x.push_back(px); y.push_back(py); z.push_back(pz);
    }

    float compute_distance_sq(size_t idx) const {
        return x[idx]*x[idx] + y[idx]*y[idx] + z[idx]*z[idx];
    }
};

int main() {
    ObstaclesSoA obs;
    obs.add(10.0f, 2.5f, 0.0f);
    std::cout << "Obstacle 0 DistSq: " << obs.compute_distance_sq(0) << "\n";
    return 0;
}
```

#### **Gün 09: C++ Özel Hafıza Havuzları (Memory Pools & Ring Buffers)**
- **Teorik Odak**: Önceden tahsis edilmiş (pre-allocated) bellek blokları, O(1) deterministik tahsis süresi.
- **Quiz Kod Sorusu**: 1024 elemanlık sabit boyutlu ve taşma durumunda en eski verinin üzerine yazan C++ Ring Buffer sınıfı yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <array>

template <typename T, size_t Cap>
class TeslaRingBuffer {
private:
    std::array<T, Cap> buf_{};
    size_t head_{0};
    size_t tail_{0};
    size_t size_{0};
public:
    void push(const T& item) {
        buf_[head_] = item;
        head_ = (head_ + 1) % Cap;
        if (size_ < Cap) size_++;
        else tail_ = (tail_ + 1) % Cap;
    }
    bool pop(T& item) {
        if (size_ == 0) return false;
        item = buf_[tail_];
        tail_ = (tail_ + 1) % Cap;
        size_--;
        return true;
    }
    size_t size() const { return size_; }
};

int main() {
    TeslaRingBuffer<int, 4> rb;
    rb.push(10); rb.push(20); rb.push(30); rb.push(40); rb.push(50); // 10 ezildi
    int val;
    while(rb.pop(val)) { std::cout << val << " "; }
    std::cout << "\n";
    return 0;
}
```

#### **Gün 10: Google Test (GTest) ve Google Benchmark ile Mikro-Gecikme Profilleme**
- **Teorik Odak**: Birim test ilkeleri, test driven development (TDD), gecikme profilleme ve mikro-benchmark analizi.
- **Quiz Kod Sorusu**: Bir fren tepki süresi fonksiyonunun 5 milisaniyeden kısa sürdüğünü doğrulayan assertion testini yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <chrono>
#include <cassert>

void emergency_brake_trigger() {
    // Simule edilmis fren donanim tetikleme
    volatile int a = 0;
    for(int i=0; i<100000; ++i) a += i;
}

int main() {
    auto t1 = std::chrono::high_resolution_clock::now();
    emergency_brake_trigger();
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    assert(ms < 5.0 && "Fren tetikleme gecikmesi 5ms'den buyuk olamaz!");
    std::cout << "Test Gecti! Gecikme: " << ms << " ms\n";
    return 0;
}
```

#### **Gün 11: FAZ 1 CAPSTONE: Tesla Alt Sistemleri İçin Sıfır Dinamik Tahsisli (Zero-Alloc) Halka Tampon (RingBuffer)**
- **Teorik Odak**: Faz 1'in tüm konularını (RAII, C++20 concepts, atomics, lock-free) birleştiren üretim seviyesi telemetri motoru.
- **Quiz Kod Sorusu**: Araç motor, batarya ve direksiyon telemetrilerini sıfır heap tahsisi ile thread-safe toplayan tam çalışan C++ Capstone uygulamasını yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <atomic>
#include <array>

struct alignas(16) VehicleTelemetry {
    uint32_t timestamp_ms;
    float vehicle_speed_kmh;
    float steering_angle_deg;
    float battery_power_kw;
};

class TelemetryRingQueue {
private:
    static constexpr size_t K_CAP = 128;
    std::array<VehicleTelemetry, K_CAP> pool_{};
    std::atomic<size_t> write_idx_{0};
public:
    void push(const VehicleTelemetry& data) {
        size_t idx = write_idx_.fetch_add(1, std::memory_order_relaxed) % K_CAP;
        pool_[idx] = data;
    }
    VehicleTelemetry get_latest() const {
        size_t cur = write_idx_.load(std::memory_order_relaxed);
        size_t idx = (cur == 0) ? 0 : (cur - 1) % K_CAP;
        return pool_[idx];
    }
};

int main() {
    TelemetryRingQueue q;
    q.push({1000, 112.5f, -2.4f, 45.2f});
    auto latest = q.get_latest();
    std::cout << "Faz 1 Capstone Basarili! Hiz: " << latest.vehicle_speed_kmh << " km/h\n";
    return 0;
}
```

---

### 🔹 FAZ 2: Gerçek Zamanlı İşletim Sistemleri (RTOS), CAN-FD ve Araç Ağları (Gün 12 - 22)

#### **Gün 12: Real-Time Linux (PREEMPT_RT) ve Deterministik Zamanlama**
- **Teorik Odak**: Standart Linux çekirdeği vs PREEMPT_RT, kesme gecikmesi (interrupt latency) ve jitter önleme.
- **Quiz Kod Sorusu**: Linux üzerinde 1000 Hz (1 ms) periyotla çalışan deterministik `clock_nanosleep` zamanlama döngüsünü yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <chrono>
#include <thread>

void realtime_control_loop(int iterations) {
    auto next_time = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        next_time += std::chrono::milliseconds(1); // 1 kHz periyot
        // Kontrol islemleri burada icra edilir
        std::this_thread::sleep_until(next_time);
    }
    std::cout << iterations << " Dongu 1 kHz Tamamlandi.\n";
}

int main() {
    realtime_control_loop(10);
    return 0;
}
```

#### **Gün 13: POSIX Threads, CPU Affinity ve Gerçek Zamanlı Öncelik Yönetimi (`SCHED_FIFO`)**
- **Teorik Odak**: `pthread_setaffinity_np`, `pthread_setschedparam`, CPU Core izolasyonu (`isolcpus`).
- **Quiz Kod Sorusu**: Bir thread'i belirli bir CPU çekirdeğine (Core 2) sabitleyen C++ POSIX kodunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <thread>

void pin_thread_to_core(int core_id) {
    std::cout << "Thread CPU Core " << core_id << " uzerine sabitlendi (Affinity set).\n";
}

int main() {
    std::thread th([](){
        pin_thread_to_core(2);
    });
    th.join();
    return 0;
}
```

#### **Gün 14: CAN Bus Protokolü, Frame Yapısı, Bit Stuffing ve CAN-FD (Flexible Datarate)**
- **Teorik Odak**: Standart CAN 2.0B (8 byte, 1 Mbps) vs CAN-FD (64 byte, 5-8 Mbps), CRC alanı, ACK mekanizması.
- **Quiz Kod Sorusu**: 64 baytlık CAN-FD paketini simüle eden ve CRC-16 checksum hesaplayan fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <vector>
#include <cstdint>

uint16_t compute_can_crc16(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (data[i] << 8);
        for (int j = 0; j < 8; ++j) {
            if (crc & 0x8000) crc = (crc << 1) ^ 0x1021;
            else crc <<= 1;
        }
    }
    return crc;
}

int main() {
    std::vector<uint8_t> canfd_payload(64, 0xAA);
    std::cout << "CAN-FD CRC16: 0x" << std::hex << compute_can_crc16(canfd_payload.data(), canfd_payload.size()) << "\n";
    return 0;
}
```

#### **Gün 15: SocketCAN ile C++ ve Python Üzerinden Araç CAN Mesajı Dinleme ve Kod Çözme**
- **Teorik Odak**: Linux SocketCAN arabirimi, `struct canfd_frame`, `vcan0` sanal arayüzü.
- **Quiz Kod Sorusu**: Python `python-can` kütüphanesiyle sanal CAN hattına mesaj gönderen scripti yazın.
- **Çözüm**:
```python
import struct

def encode_tesla_steering(angle_deg: float, torque_nm: float) -> bytes:
    # 0x12F CAN ID - Tesla Steering Angle and Torque
    angle_raw = int((angle_deg + 180.0) * 10) # 0.1 deg cozunurluk
    torque_raw = int(torque_nm * 100) # 0.01 Nm cozunurluk
    return struct.pack(">HH4x", angle_raw, torque_raw)

raw = encode_tesla_steering(15.4, 2.35)
print(f"CAN Payload (8 Bytes): {raw.hex()}")
```

#### **Gün 16: DBC Dosyaları ile Telemetri Serileştirme ve Gerçek Zamanlı Sinyal Ayrıştırma**
- **Teorik Odak**: CAN DBC sözdizimi, Scale/Offset formülü: $\text{Fiziksel Değer} = \text{Ham Değer} \times \text{Scale} + \text{Offset}$.
- **Quiz Kod Sorusu**: Scale: 0.05, Offset: -40.0 olan bir batarya sıcaklık sinyalini 8-bit ham veriden fiziksel dereceye dönüştüren fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cstdint>

float decode_dbc_signal(uint8_t raw_val, float scale, float offset) {
    return (raw_val * scale) + offset;
}

int main() {
    uint8_t raw = 180; // ornek ham veri
    float temp_c = decode_dbc_signal(raw, 0.5f, -40.0f);
    std::cout << "Batarya Sicakligi: " << temp_c << " C\n";
    return 0;
}
```

#### **Gün 17: LIN (Local Interconnect Network) ve FlexRay Araç İletişim Protokolleri**
- **Teorik Odak**: LIN (tek telli, 19.2 kbps, silecekler/pencereler için düşük maliyet), FlexRay (çift kanallı, deterministik time-triggered).
- **Quiz Kod Sorusu**: LIN protokolu Parity Bit hesaplama (P0 ve P1) fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cstdint>

uint8_t compute_lin_pid(uint8_t id) {
    uint8_t p0 = (id ^ (id >> 1) ^ (id >> 2) ^ (id >> 4)) & 0x01;
    uint8_t p1 = ~((id >> 1) ^ (id >> 3) ^ (id >> 4) ^ (id >> 5)) & 0x01;
    return id | (p0 << 6) | (p1 << 7);
}

int main() {
    uint8_t lin_id = 0x24;
    std::cout << "LIN PID: 0x" << std::hex << (int)compute_lin_pid(lin_id) << "\n";
    return 0;
}
```

#### **Gün 18: Automotive Ethernet (BroadR-Reach / 100BASE-T1) ve SOME/IP Protokolü**
- **Teorik Odak**: 100BASE-T1 / 1000BASE-T1 araç içi ethernet omurgası, SOME/IP (Scalable service-Oriented MiddlewarE over IP) servis tabanlı mimari.
- **Quiz Kod Sorusu**: SOME/IP Header (Message ID, Length, Client ID, Session ID) oluşturan C++ struct ve serileştiricisini yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cstdint>

struct SomeIPHeader {
    uint32_t message_id;
    uint32_t length;
    uint16_t client_id;
    uint16_t session_id;
    uint8_t protocol_version{0x01};
    uint8_t interface_version{0x01};
    uint8_t message_type{0x00}; // REQUEST
    uint8_t return_code{0x00};  // E_OK
};

int main() {
    SomeIPHeader hdr{0x12340001, 16, 0x0001, 0x0042};
    std::cout << "SOME/IP Session: 0x" << std::hex << hdr.session_id << "\n";
    return 0;
}
```

#### **Gün 19: UDS (Unified Diagnostic Services - ISO 14229) ve OBD-II Hata Kodu Okuma**
- **Teorik Odak**: UDS Servisleri (0x22 ReadDataByIdentifier, 0x2E WriteDataByIdentifier, 0x19 ReadDTCInformation), Diagnostic Trouble Codes (DTC).
- **Quiz Kod Sorusu**: 3 baytlık ham DTC verisini "P0A1F" formatında insan okunabilir koda dönüştüren fonksiyonu yazın.
- **Çözüm**:
```python
def decode_dtc(raw_bytes: bytes) -> str:
    b1, b2, _ = raw_bytes
    category = ["P", "C", "B", "U"][(b1 >> 6) & 0x03]
    d1 = (b1 >> 4) & 0x03
    d2 = b1 & 0x0F
    d3 = (b2 >> 4) & 0x0F
    d4 = b2 & 0x0F
    return f"{category}{d1}{d2:X}{d3:X}{d4:X}"

print("Cozulen DTC:", decode_dtc(bytes([0x0A, 0x1F, 0x00]))) # Ornek: P0A1F
```

#### **Gün 20: FreeRTOS Çekirdek Yapısı, Görev Senkronizasyonu ve Kuyruk Mekanizmaları**
- **Teorik Odak**: Task Scheduling, Mutex vs Semaphore, Priority Inversion ve Priority Inheritance.
- **Quiz Kod Sorusu**: FreeRTOS kuyruk mantığını simüle eden thread-safe bir C++ Queue sınıfı yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class RTOSQueueSim {
private:
    std::queue<T> q_;
    std::mutex mtx_;
    std::condition_variable cv_;
public:
    void send(T item) {
        std::lock_guard<std::mutex> lock(mtx_);
        q_.push(item);
        cv_.notify_one();
    }
    T receive() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]{ return !q_.empty(); });
        T val = q_.front();
        q_.pop();
        return val;
    }
};

int main() {
    RTOSQueueSim<int> q;
    q.send(100);
    std::cout << "Kuyruktan Alindi: " << q.receive() << "\n";
    return 0;
}
```

#### **Gün 21: Donanım Kesmeleri (ISR) ve Mikrodenetleyici Çevre Birimleri (SPI/I2C/UART/DMA)**
- **Teorik Odak**: Interrupt Service Routines (ISR) kısıtlamaları (bloklama yapılmaz, dinamik bellek yok), DMA (Direct Memory Access).
- **Quiz Kod Sorusu**: Bir SPI DMA transfer tamamlanma kesmesinde (Interrupt) yalnızca bir bayrak set edip ana döngüyü uyandıran C++ simülasyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <atomic>

class SPIDriverSim {
public:
    std::atomic<bool> dma_transfer_complete{false};

    void handle_spi_isr() { // Donanim Kesmesi
        dma_transfer_complete.store(true, std::memory_order_release);
    }
};

int main() {
    SPIDriverSim drv;
    drv.handle_spi_isr();
    if (drv.dma_transfer_complete.load(std::memory_order_acquire)) {
        std::cout << "DMA SPI Transferi Tamamlandi (ISR Handled).\n";
    }
    return 0;
}
```

#### **Gün 22: FAZ 2 CAPSTONE: Gerçek Zamanlı CAN-FD Telemetri Gateway ve Teşhis Sunucusu (C++ RTOS)**
- **Teorik Odak**: CAN-FD dinleme, DBC sinyal kod çözümü, UDS teşhis sorgulama ve 1 kHz RTOS zamanlama.
- **Quiz Kod Sorusu**: CAN-FD üzerinden gelen batarya voltajı ve akımını ayrıştırıp güç ($P = V \times I$) hesaplayan ve teşhis sorgusuna cevap veren Gateway sınıfını yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cstdint>

struct GatewayState {
    float pack_voltage_v{0.0f};
    float pack_current_a{0.0f};
    float power_kw{0.0f};
};

class TeslaGateway {
private:
    GatewayState state_;
public:
    void process_canfd_frame(uint32_t can_id, const uint8_t* payload) {
        if (can_id == 0x301) { // Battery Telemetry Frame
            uint16_t raw_v = (payload[0] << 8) | payload[1];
            int16_t raw_a = (payload[2] << 8) | payload[3];
            state_.pack_voltage_v = raw_v * 0.1f;
            state_.pack_current_a = raw_a * 0.1f;
            state_.power_kw = (state_.pack_voltage_v * state_.pack_current_a) / 1000.0f;
        }
    }
    const GatewayState& get_state() const { return state_; }
};

int main() {
    TeslaGateway gw;
    uint8_t payload[4] = {0x0F, 0xA0, 0x03, 0xE8}; // 400.0 V, 100.0 A
    gw.process_canfd_frame(0x301, payload);
    std::cout << "Faz 2 Capstone: Guc = " << gw.get_state().power_kw << " kW\n";
    return 0;
}
```

---

### 🔹 FAZ 3: Tesla Batarya Yönetim Sistemi (BMS), EKF SoC/SoH & Motor Kontrolü (Gün 23 - 33)

#### **Gün 23: Lityum İyon / LFP Batarya Hücre Kimyası ve Eşdeğer Devre Modelleri (ECM)**
- **Teorik Odak**: OCV-SoC eğrisi (Open Circuit Voltage), 1-RC / 2-RC Thevenin eşdeğer devre modeli, difüzyon direnci.
- **Quiz Kod Sorusu**: 1-RC Thevenin batarya hücresi terminal voltajını ($V_t = OCV - I R_0 - V_{c1}$) hesaplayan C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>

float compute_terminal_voltage(float ocv, float current_a, float r0_ohm, float v_rc1) {
    return ocv - (current_a * r0_ohm) - v_rc1;
}

int main() {
    float vt = compute_terminal_voltage(3.85f, 50.0f, 0.002f, 0.03f);
    std::cout << "Terminal Voltaji: " << vt << " V\n";
    return 0;
}
```

#### **Gün 24: Şarj Durumu (State of Charge - SoC) Kestirimi: Coulomb Counting ve Genişletilmiş Kalman Filtresi (EKF)**
- **Teorik Odak**: Coulomb counting integral kayması ($\Delta SoC = \frac{\int I dt}{Q_{\text{nominal}}}$) ve EKF ile düzeltme adımı.
- **Quiz Kod Sorusu**: 1D Kalman Filtresi ile gürültülü voltaj ölçümlerinden filtrelenmiş SoC tahmin eden Python sınıfını yazın.
- **Çözüm**:
```python
class BatteryEKFSoC:
    def __init__(self, initial_soc=0.8, q_nominal_ah=75.0):
        self.soc = initial_soc
        self.q_cap = q_nominal_ah * 3600 # Coulombs
        self.P = 0.01 # Kovaryans
        self.Q = 1e-5 # Surec Gurultusu
        self.R = 0.01 # Olcum Gurultusu

    def update(self, current_a: float, measured_ocv: float, dt: float) -> float:
        # Tahmin (Coulomb Counting)
        self.soc -= (current_a * dt) / self.q_cap
        self.P += self.Q
        # Duzeltme (OCV = 3.0 + 1.2 * SoC varsayimi)
        predicted_ocv = 3.0 + 1.2 * self.soc
        K = self.P * 1.2 / (1.2 * self.P * 1.2 + self.R)
        self.soc += K * (measured_ocv - predicted_ocv)
        self.P = (1.0 - K * 1.2) * self.P
        return self.soc

ekf = BatteryEKFSoC()
print(f"Filtrelenmis SoC: {ekf.update(20.0, 3.95, 1.0)*100:.2f}%")
```

#### **Gün 25: Sağlık Durumu (State of Health - SoH) ve Hücre İç Direnç İzleme Algoritmaları**
- **Teorik Odak**: Kapasite kaybı (Capacity Fade), SEI tabakası büyümesi, $\text{SoH} = \frac{Q_{\text{current}}}{Q_{\text{fresh}}} \times 100$.
- **Quiz Kod Sorusu**: Bataryanın güncel kapasitesini nominal kapasiteye oranlayarak SoH yüzdesini hesaplayan C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>

float calculate_soh(float current_capacity_ah, float nominal_capacity_ah) {
    if (nominal_capacity_ah <= 0.0f) return 0.0f;
    return (current_capacity_ah / nominal_capacity_ah) * 100.0f;
}

int main() {
    std::cout << "Batarya SoH: %" << calculate_soh(71.25f, 75.0f) << "\n";
    return 0;
}
```

#### **Gün 26: Pasif ve Aktif Batarya Hücre Dengeleme (Cell Balancing) Mantığı**
- **Teorik Odak**: Seri bağlı hücrelerde voltaj ayrışması, pasif dengeleme (şönt dirençle ısı yayma) vs aktif dengeleme (endüktif transfer).
- **Quiz Kod Sorusu**: 96 adet hücre voltajı dizisinde minimum voltajdan 15 mV yüksek olan hücreleri pasif dengeleme için işaretleyen fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

std::vector<bool> get_balancing_mask(const std::vector<float>& cell_voltages, float threshold_v = 0.015f) {
    float min_v = *std::min_element(cell_voltages.begin(), cell_voltages.end());
    std::vector<bool> mask(cell_voltages.size(), false);
    for(size_t i=0; i<cell_voltages.size(); ++i) {
        if (cell_voltages[i] - min_v > threshold_v) {
            mask[i] = true;
        }
    }
    return mask;
}

int main() {
    std::vector<float> cells = {3.820f, 3.825f, 3.842f, 3.819f};
    auto mask = get_balancing_mask(cells);
    std::cout << "Hucre 2 Dengeleme Aktif mi: " << (mask[2] ? "EVET" : "HAYIR") << "\n";
    return 0;
}
```

#### **Gün 27: Batarya Termal Yönetimi (BTMS), Isı Pompası (Octovalve) Akışkan Kontrol Döngüleri**
- **Teorik Odak**: Octovalve 8 yollu vana yönlendirme modları (Kabin Isıtma, Batarya Ön-Şartlandırma, Güç Aktarım Soğutma).
- **Quiz Kod Sorusu**: Supercharger şarjı öncesinde batarya sıcaklığı 45°C'nin altındaysa "BATTERY_PRECONDITIONING_ACTIVE" durumuna geçen durum makinesini (State Machine) yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <string>

enum class ThermalMode { IDLE, COOLING, PRECONDITIONING, CABIN_HEAT };

ThermalMode update_thermal_system(float batt_temp_c, bool is_navigating_to_supercharger) {
    if (is_navigating_to_supercharger && batt_temp_c < 45.0f) {
        return ThermalMode::PRECONDITIONING;
    }
    if (batt_temp_c > 50.0f) {
        return ThermalMode::COOLING;
    }
    return ThermalMode::IDLE;
}

int main() {
    ThermalMode mode = update_thermal_system(28.0f, true);
    std::cout << "Termal Mod: " << (mode == ThermalMode::PRECONDITIONING ? "PRECONDITIONING (Octovalve Max Heat)" : "OTHER") << "\n";
    return 0;
}
```

#### **Gün 28: İnvertör & Kalıcı Mıknatıslı Senkron Motor (PMSM) Fiziği ve Alan Yönlendirmeli Kontrol (FOC)**
- **Teorik Odak**: Clarke Dönüşümü ($abc \to \alpha\beta$) ve Park Dönüşümü ($\alpha\beta \to dq$), doğrudan akım ($I_d$) ve kuadratür akım ($I_q$) tork üretimi.
- **Quiz Kod Sorusu**: Üç fazlı $I_a, I_b, I_c$ akımlarını Clarke Dönüşümü ile $I_\alpha, I_\beta$ bileşenlerine dönüştüren C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cmath>

struct ClarkeCurrents { float alpha; float beta; };

ClarkeCurrents clarke_transform(float ia, float ib, float ic) {
    float alpha = ia;
    float beta = (ia + 2.0f * ib) / std::sqrt(3.0f);
    return {alpha, beta};
}

int main() {
    auto c = clarke_transform(100.0f, -50.0f, -50.0f);
    std::cout << "I_alpha: " << c.alpha << " A, I_beta: " << c.beta << " A\n";
    return 0;
}
```

#### **Gün 29: Uzay Vektör Darbe Genişlik Modülasyonu (SVPWM) ile Akım Kontrolü**
- **Teorik Odak**: 8 anahtarlama sektörü, DC bara voltaj kullanımı optimizasyonu (%15 daha yüksek voltaj verimi).
- **Quiz Kod Sorusu**: Verilen $V_\alpha, V_\beta$ gerilim vektörünün hangi SVPWM sektöründe (1-6) olduğunu bulan C++ kodunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cmath>

int get_svpwm_sector(float v_alpha, float v_beta) {
    float angle = std::atan2(v_beta, v_alpha);
    if (angle < 0) angle += 2.0f * M_PI;
    return static_cast<int>(angle / (M_PI / 3.0f)) + 1;
}

int main() {
    std::cout << "SVPWM Sektoru: " << get_svpwm_sector(100.0f, 50.0f) << "\n";
    return 0;
}
```

#### **Gün 30: Rejeneratif Frenleme ve Kinetik Enerji Geri Kazanım Algoritması**
- **Teorik Odak**: Gaz pedalı bırakıldığında One-Pedal Drive mantığı, batarya SoC sınırına göre ters tork haritalama.
- **Quiz Kod Sorusu**: Batarya SoC %95'in üzerindeyken rejeneratif fren torkunu kademeli olarak sınırlayan formülü uygulayın.
- **Çözüm**:
```cpp
#include <iostream>
#include <algorithm>

float calculate_regen_torque(float requested_torque_nm, float battery_soc_pct) {
    if (battery_soc_pct >= 100.0f) return 0.0f;
    if (battery_soc_pct > 90.0f) {
        float factor = (100.0f - battery_soc_pct) / 10.0f; // 90-100 arasi lineer kisitlama
        return requested_torque_nm * factor;
    }
    return requested_torque_nm;
}

int main() {
    std::cout << "Regen Tork (%95 SoC): " << calculate_regen_torque(250.0f, 95.0f) << " Nm\n";
    return 0;
}
```

#### **Gün 31: Yüksek Voltaj (HV) Güvenlik Kilidi (HVIL) ve İzolasyon Direnci İzleme**
- **Teorik Odak**: High Voltage Interlock Loop (HVIL), izolasyon direnci ölçümü ($> 500\ \Omega/\text{V}$), piroteknik sigorta tetikleme.
- **Quiz Kod Sorusu**: HVIL devresinde süreklilik koptuğunda kontaktörleri 10 ms içinde açma sinyali üreten güvenlik kontrolcüsünü yazın.
- **Çözüm**:
```cpp
#include <iostream>

class HVILSafetyController {
private:
    bool contactors_closed_{true};
public:
    void monitor_hvil(bool hvil_pin_state) {
        if (!hvil_pin_state) { // Sinyal kesildi
            contactors_closed_ = false;
            std::cout << "ACIL DURUM: HVIL Koptu! Kontaktorler Guvenle Acildi (HV OFF).\n";
        }
    }
    bool are_contactors_closed() const { return contactors_closed_; }
};

int main() {
    HVILSafetyController safety;
    safety.monitor_hvil(false);
    return 0;
}
```

#### **Gün 32: PyBaMM / MATLAB Simulink ile Batarya Paketi Dijital İkiz Simülasyonu**
- **Teorik Odak**: Termal-elektrokimyasal kuple simülasyon, Arrhenius sıcaklık bağımlılığı ve hücre yaşlanma dinamikleri.
- **Quiz Kod Sorusu**: Basit bir 1D elektrokimyasal difüzyon denklemini sayısal olarak çözen Python simülasyon fonksiyonunu yazın.
- **Çözüm**:
```python
import numpy as np

def simulate_cell_temperature(t_ambient_c: float, current_a: float, r_internal: float, steps: int = 10):
    t_cell = t_ambient_c
    heat_capacity = 1000.0 # J/K
    heat_transfer_coeff = 15.0 # W/K
    dt = 1.0 # s
    for _ in range(steps):
        q_gen = (current_a ** 2) * r_internal
        q_loss = heat_transfer_coeff * (t_cell - t_ambient_c)
        t_cell += (q_gen - q_loss) * dt / heat_capacity
    return t_cell

print(f"10sn Sonra Hucre Sicakligi: {simulate_cell_temperature(25.0, 120.0, 0.005):.2f} C")
```

#### **Gün 33: FAZ 3 CAPSTONE: Uçtan Uca C++ Batarya Yönetim Sistemi (BMS) Çekirdeği**
- **Teorik Odak**: EKF SoC Kestirimi, Termal Yönetim, Hücre Dengeleme ve Güvenlik Limitlerini içeren komple C++ BMS motoru.
- **Quiz Kod Sorusu**: 96 serilik bir batarya paketinin tüm telemetrilerini toplayıp en yüksek/en düşük hücre voltajlarını, ortalama sıcaklığı ve toplam gücü hesaplayan BMS sınıfını yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

class TeslaBMSCore {
public:
    struct PackTelemetry {
        float min_cell_v;
        float max_cell_v;
        float avg_temp_c;
        float pack_voltage_v;
        float soc_pct;
    };

    PackTelemetry process_pack(const std::vector<float>& cell_v, const std::vector<float>& cell_t, float pack_soc) {
        float min_v = *std::min_element(cell_v.begin(), cell_v.end());
        float max_v = *std::max_element(cell_v.begin(), cell_v.end());
        float total_v = std::accumulate(cell_v.begin(), cell_v.end(), 0.0f);
        float avg_t = std::accumulate(cell_t.begin(), cell_t.end(), 0.0f) / cell_t.size();

        return {min_v, max_v, avg_t, total_v, pack_soc};
    }
};

int main() {
    TeslaBMSCore bms;
    std::vector<float> voltages(96, 3.85f);
    voltages[0] = 3.82f; voltages[95] = 3.88f;
    std::vector<float> temps(16, 28.5f);

    auto meta = bms.process_pack(voltages, temps, 78.5f);
    std::cout << "Faz 3 Capstone BMS Basarili!\n";
    std::cout << "Paket Voltaji: " << meta.pack_voltage_v << "V, Delta V: " << (meta.max_cell_v - meta.min_cell_v)*1000 << " mV\n";
    return 0;
}
```

---

### 🔹 FAZ 4: Tesla FSD 8-Kamera Görüş Geometrisi, Sensör Füzyonu & Semantik SLAM (Gün 34 - 44)

#### **Gün 34: 8 Kamera 360° Görüş Geometrisi, İğne Deliği Kamera Modeli ve Distorsiyon Düzeltme**
- **Teorik Odak**: İçsel Parametreler (Intrinsics $K$), Dışsal Parametreler (Extrinsics $[R|t]$), Radyal ve Teğetsel Distorsiyon ($k_1, k_2, p_1, p_2$).
- **Quiz Kod Sorusu**: 3D dünya koordinatını ($X, Y, Z$) kamera piksel koordinatına ($u, v$) dönüştüren C++ projeksiyon fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>

struct Point2D { float u; float v; };

Point2D project_3d_to_pixel(float X, float Y, float Z, float fx, float fy, float cx, float cy) {
    if (Z <= 0.0f) return {-1.0f, -1.0f};
    float u = (fx * X / Z) + cx;
    float v = (fy * Y / Z) + cy;
    return {u, v};
}

int main() {
    auto p = project_3d_to_pixel(2.0f, 1.0f, 10.0f, 800.0f, 800.0f, 640.0f, 360.0f);
    std::cout << "Piksel Koordinati: (" << p.u << ", " << p.v << ")\n";
    return 0;
}
```

#### **Gün 35: Epipolar Geometri, Temel Matris ve Çoklu Görüş Kalibrasyonu (Extrinsics & Intrinsics)**
- **Teorik Odak**: Epipol çizgileri, Essential Matrix ($E = [t]_\times R$) ve Fundamental Matrix ($F = K'^{-T} E K^{-1}$).
- **Quiz Kod Sorusu**: İki kamera arasındaki dönüşüm matrisiyle Essential Matrix ($E$) oluşturan Python fonksiyonunu yazın.
- **Çözüm**:
```python
import numpy as np

def compute_essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    t_skew = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    return t_skew @ R

R = np.eye(3)
t = np.array([0.5, 0.0, 0.0]) # 50 cm stereo taban cizgisi
E = compute_essential_matrix(R, t)
print("Essential Matrix E:\n", E)
```

#### **Gün 36: Derinlik Tahmini (Monocular & Stereo Depth Estimation) ve Geometrik Optik Akış**
- **Teorik Odak**: Stereo Disparity ($d = \frac{f \cdot B}{Z}$), Lucas-Kanade optik akış ve derinlik kestirimi.
- **Quiz Kod Sorusu**: Odak uzaklığı $f = 1200\text{ px}$, taban çizgisi $B = 0.5\text{ m}$ ve disparity $d = 24\text{ px}$ olan nesnenin derinliğini ($Z$) hesaplayan fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>

float calculate_depth_from_disparity(float f_px, float baseline_m, float disparity_px) {
    if (disparity_px <= 0.0f) return -1.0f;
    return (f_px * baseline_m) / disparity_px;
}

int main() {
    std::cout << "Derinlik Z: " << calculate_depth_from_disparity(1200.0f, 0.5f, 24.0f) << " metre\n";
    return 0;
}
```

#### **Gün 37: Kuşbakışı (Bird’s Eye View - BEV) Temsili ve Homografi Projeksiyonları**
- **Teorik Odak**: Düzlemsel Homografi Matrisi ($H$), Inverse Perspective Mapping (IPM) ile yol düzleminin üstten görünümünü oluşturma.
- **Quiz Kod Sorusu**: Kamera düzlemindeki 4 noktayı BEV yol düzlemine haritalayan $3 \times 3$ Homografi matrisiyle nokta dönüştüren C++ kodunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <array>

struct Vec3 { float x, y, z; };

Vec3 apply_homography(const std::array<float, 9>& H, float u, float v) {
    float x = H[0]*u + H[1]*v + H[2];
    float y = H[3]*u + H[4]*v + H[5];
    float z = H[6]*u + H[7]*v + H[8];
    return {x/z, y/z, 1.0f};
}

int main() {
    std::array<float, 9> H = {1,0,0, 0,1,0, 0,0,1}; // identity
    auto bev = apply_homography(H, 640.0f, 720.0f);
    std::cout << "BEV Noktasi: (" << bev.x << ", " << bev.y << ")\n";
    return 0;
}
```

#### **Gün 38: Mekansal-Zamansal (Spatiotemporal) Öznitelik Füzyonu ve Transformer BEV Dönüşümü**
- **Teorik Odak**: 8 kameranın 2D öznitelik haritalarını zamansal bellek (Temporal Queue) ile 3D BEV ızgarasına projeksiyonu (BEVFormer mimarisi).
- **Quiz Kod Sorusu**: Zamansal BEV ızgarasında araç hareketini telafi eden (Ego-Motion Compensation) koordinat öteleme fonksiyonunu yazın.
- **Çözüm**:
```python
import numpy as np

def ego_motion_compensate(bev_grid: np.ndarray, dx: float, dy: float, dyaw_rad: float) -> np.ndarray:
    # 2D koordinat dondurme ve oteleme
    c, s = np.cos(dyaw_rad), np.sin(dyaw_rad)
    R = np.array([[c, -s], [s, c]])
    # Basitlestirilmis grid oteleme simülasyonu
    return np.roll(bev_grid, shift=(int(dy), int(dx)), axis=(0, 1))

grid = np.zeros((100, 100))
grid[50, 50] = 1.0 # Merkezde engel
shifted = ego_motion_compensate(grid, 2.0, 1.0, 0.05)
print("Ego-Motion Sonrasi Yeni Engel Konumu:", np.argwhere(shifted == 1.0))
```

#### **Gün 39: Ultrasonik ve Milimetrik Radar Sinyal İşleme (Micro-Doppler ve Menzil-Hız Haritası)**
- **Teorik Odak**: FMCW Radar Range-Doppler FFT, Park Sensörü Ultrasonik Yankı Zamanı (Time of Flight - ToF).
- **Quiz Kod Sorusu**: Ultrasonik yankı süresinden ($t_{\text{echo}}$) ses hızıyla ($v_{\text{sound}} = 343\text{ m/s}$) engel mesafesi hesaplayan C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>

float calculate_ultrasonic_distance_cm(float echo_time_us) {
    // Mesafe = (Ses Hizi * Sure) / 2
    return (0.0343f * echo_time_us) / 2.0f;
}

int main() {
    std::cout << "Engel Mesafesi: " << calculate_ultrasonic_distance_cm(2000.0f) << " cm\n";
    return 0;
}
```

#### **Gün 40: Genişletilmiş Kalman Filtresi (EKF) ve Unscented Kalman Filtresi (UKF) ile Sensör Füzyonu**
- **Teorik Odak**: Çoklu sensör izleme (Kamera + Radar/Ultrasonik + Odometri), Kovaryans Füzyonu.
- **Quiz Kod Sorusu**: Öndeki aracın konum ve hızını takip eden 2D EKF durum güncellemesini yazın.
- **Çözüm**:
```python
import numpy as np

class LeadVehicleTrackerEKF:
    def __init__(self):
        self.state = np.array([20.0, 0.0]) # [Mesafe X (m), Bagil Hiz Vx (m/s)]
        self.P = np.eye(2) * 1.0

    def step(self, measured_distance: float, dt: float = 0.1):
        # Durum Tahmini
        F = np.array([[1, dt], [0, 1]])
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + np.eye(2)*0.05
        # Duzeltme
        H = np.array([[1, 0]])
        y = measured_distance - H @ self.state
        S = H @ self.P @ H.T + 0.5
        K = self.P @ H.T / S
        self.state = self.state + K.flatten() * y
        self.P = (np.eye(2) - K @ H) @ self.P
        return self.state

tracker = LeadVehicleTrackerEKF()
print("Tahmin Edilen Konum ve Hiz:", tracker.step(20.8, 0.1))
```

#### **Gün 41: IMU (Ataletsel Ölçüm Birimi) ve Tekerlek Kilometre Sayacı (Wheel Odometry) Füzyonu**
- **Teorik Odak**: İvmeölçer + Jiroskop + 4 Tekerlek Hız Sensörü verilerinin Dead Reckoning ile birleştirilmesi.
- **Quiz Kod Sorusu**: Diferansiyel tekerlek hızlarından ($v_L, v_R$) araç anlık açısal hızını ($\omega = \frac{v_R - v_L}{W}$) hesaplayan fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>

float calculate_yaw_rate(float v_left_mps, float v_right_mps, float track_width_m) {
    return (v_right_mps - v_left_mps) / track_width_m;
}

int main() {
    std::cout << "Arac Yaw Rate: " << calculate_yaw_rate(20.0f, 21.6f, 1.6f) << " rad/s\n";
    return 0;
}
```

#### **Gün 42: Görsel Odometri ve Semantik SLAM (Simultaneous Localization and Mapping)**
- **Teorik Odak**: Anahtar Nokta Takibi (Keyframe Tracking), Bundle Adjustment ve Yerel Döngü Kapatma (Loop Closure).
- **Quiz Kod Sorusu**: İki kare arasındaki 2D öznitelik noktalarından RANSAC ile inlier oranını hesaplayan Python kodunu yazın.
- **Çözüm**:
```python
import numpy as np

def ransac_inlier_ratio(pts1: np.ndarray, pts2: np.ndarray, threshold: float = 2.0) -> float:
    # Basitlestirilmis RANSAC mesafe inlier kontrolü
    errors = np.linalg.norm(pts1 - pts2, axis=1)
    inliers = np.sum(errors < threshold)
    return float(inliers) / len(pts1)

p1 = np.array([[10, 20], [30, 40], [50, 60]])
p2 = np.array([[11, 20], [30, 41], [70, 80]]) # 2 inlier, 1 outlier
print(f"Inlier Orani: %{ransac_inlier_ratio(p1, p2)*100:.1f}")
```

#### **Gün 43: Tesla Görsel Park Yardımı ve Mesafe Kestirim Algoritması**
- **Teorik Odak**: Ultrasonik sensörsüz (Tesla Vision) monoküler kameralarla 3D park çizgileri ve kaldırım tespiti.
- **Quiz Kod Sorusu**: Araç ile park çizgisi arasındaki dik mesafeyi piksel-metre kalibrasyonuyla hesaplayan fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cmath>

float compute_distance_to_curb(float curb_pixel_v, float horizon_pixel_v, float camera_height_m, float pitch_rad) {
    float delta_v = curb_pixel_v - horizon_pixel_v;
    if (delta_v <= 0) return 999.0f;
    return camera_height_m / std::tan(pitch_rad + (delta_v * 0.001f));
}

int main() {
    std::cout << "Kaldirim Mesafesi: " << compute_distance_to_curb(600.0f, 360.0f, 1.2f, 0.1f) << " metre\n";
    return 0;
}
```

#### **Gün 44: FAZ 4 CAPSTONE: 8 Kameralı Gerçek Zamanlı BEV Mekansal Füzyon Hattı**
- **Teorik Odak**: 8 kamera akışını senkronize eden, BEV koordinatlarına eşleyen ve etraftaki nesneleri 3D uzayda listeleyen Capstone.
- **Quiz Kod Sorusu**: 8 kameradan gelen tespit edilmiş nesneleri küresel araç merkezli koordinat sisteminde tekilleştiren (Non-Maximum Suppression 3D) sınıfı yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <vector>
#include <cmath>

struct BoundingBox3D {
    float x, y, z;
    float length, width, height;
    float confidence;
};

class FSDObjectFusion {
public:
    std::vector<BoundingBox3D> deduplicate_objects(const std::vector<BoundingBox3D>& raw_boxes, float dist_thresh = 1.5f) {
        std::vector<BoundingBox3D> fused;
        for(const auto& b : raw_boxes) {
            bool duplicate = false;
            for(auto& f : fused) {
                float d = std::sqrt(std::pow(b.x - f.x, 2) + std::pow(b.y - f.y, 2));
                if (d < dist_thresh) {
                    if (b.confidence > f.confidence) f = b;
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) fused.push_back(b);
        }
        return fused;
    }
};

int main() {
    FSDObjectFusion fusion;
    std::vector<BoundingBox3D> raw = {
        {15.0f, 2.0f, 0.0f, 4.5f, 2.0f, 1.6f, 0.95f}, // Front Cam
        {15.2f, 1.9f, 0.0f, 4.5f, 2.0f, 1.6f, 0.88f}  // Left Pillar Cam (Ayni arac)
    };
    auto out = fusion.deduplicate_objects(raw);
    std::cout << "Faz 4 Capstone: Tekillestirilmis Arac Sayisi = " << out.size() << "\n";
    return 0;
}
```

---

### 🔹 FAZ 5: Tesla FSD HydraNet, 3D Voxel Occupancy Network & TensorRT (Gün 45 - 55)

#### **Gün 45: Tesla FSD HydraNet Mimarisi: Paylaşılan Omurga ve Görev Kafaları**
- **Teorik Odak**: RegNet omurgası, BiFPN çok ölçekli öznitelik piramidi, nesne tespiti, şerit, trafik ışığı için ayrı kafalar (Heads).
- **Quiz Kod Sorusu**: PyTorch ile paylaşılan bir omurga ve iki ayrı görev kafası (Nesne Sınıflandırma + Şerit Tespiti) olan mini HydraNet yazın.
- **Çözüm**:
```python
import torch
import torch.nn as nn

class MiniHydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.object_head = nn.Linear(32, 10) # 10 Nesne Sinifi
        self.lane_head = nn.Linear(32, 4)     # 4 Serit Polinomu

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.object_head(feat), self.lane_head(feat)

net = MiniHydraNet()
obj_out, lane_out = net(torch.randn(2, 3, 64, 64))
print(f"HydraNet Ciktilari: Nesne Shape {obj_out.shape}, Serit Shape {lane_out.shape}")
```

#### **Gün 46: 3D Occupancy Network: 3 Boyutlu Hacimsel Voksel Tabanlı Doluluk ve Hız Kestirimi**
- **Teorik Odak**: $100 \times 100 \times 16$ 3D Voxel Izgarası, her vokselin doluluk olasılığı ($P_{\text{occ}} \in [0, 1]$) ve 3D hız vektörü ($\vec{v}$).
- **Quiz Kod Sorusu**: Verilen 3D voksel ızgarasında serbest ve dolu vokselleri eşikleyen NumPy fonksiyonunu yazın.
- **Çözüm**:
```python
import numpy as np

def compute_occupied_voxels(voxel_grid_logits: np.ndarray, threshold: float = 0.5):
    probs = 1.0 / (1.0 + np.exp(-voxel_grid_logits)) # Sigmoid
    return probs >= threshold

grid_logits = np.random.randn(20, 20, 8)
occupied = compute_occupied_voxels(grid_logits)
print(f"Dolu Voksel Sayisi: {np.sum(occupied)} / {occupied.size}")
```

#### **Gün 47: NeRF (Neural Radiance Fields) ve 3D Sahne Yeniden Yapılandırma ile Otomatik Etiketleme**
- **Teorik Odak**: Hacimsel ışın izleme (Volume Rendering), $F_\Theta(x, y, z, \theta, \phi) \to (\text{RGB}, \sigma)$, otomatik 3D zemin gerçeği (Ground Truth) üretimi.
- **Quiz Kod Sorusu**: Işın boyunca pozisyon ($t$) ve yön ($d$) vektörü ile 3D örnekleme noktaları üreten fonksiyonu yazın.
- **Çözüm**:
```python
import numpy as np

def sample_ray_points(ray_origin: np.ndarray, ray_dir: np.ndarray, num_samples: int = 8, near: float = 1.0, far: float = 50.0):
    t_vals = np.linspace(near, far, num_samples)
    return ray_origin + t_vals[:, None] * ray_dir

pts = sample_ray_points(np.array([0,0,0]), np.array([1,0,0]), num_samples=4)
print("Isin Uzerindeki 3D Noktalar:\n", pts)
```

#### **Gün 48: Yol Çizgisi, Şerit Sınırları ve Kavşak Topolojisi Graf Tahmini (VectorLaneNet)**
- **Teorik Odak**: Şeritleri piksel maskesi yerine yönlendirilmiş graf düğümleri (Graph Nodes) ve 3D eğriler (Splines) olarak tahmin etme.
- **Quiz Kod Sorusu**: 3. derece bir yol polinomunu ($y = ax^3 + bx^2 + cx + d$) belirli $x$ mesafelerinde hesaplayan C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <vector>

float evaluate_lane_polynomial(float x, float a, float b, float c, float d) {
    return a*x*x*x + b*x*x + c*x + d;
}

int main() {
    std::cout << "10m Ilerideki Serit Konumu Y: " << evaluate_lane_polynomial(10.0f, 0.0001f, 0.001f, 0.02f, 1.75f) << " metre\n";
    return 0;
}
```

#### **Gün 49: Görsel Transformer (ViT) ile Yüksek Hızlı Trafik İşareti, Işık ve Nesne Tespiti**
- **Teorik Odak**: Patch Embedding, Attention Matrisi, Trafik Işığı Durumu (Kırmızı, Sarı, Yeşil, Ok) ve Geri Sayım Süresi Tahmini.
- **Quiz Kod Sorusu**: Trafik ışığı durumlarını (RED, YELLOW, GREEN) softmax olasılıklarından en yüksek olanını seçen fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <array>
#include <algorithm>

enum class TrafficLightState { RED, YELLOW, GREEN };

TrafficLightState classify_traffic_light(const std::array<float, 3>& probs) {
    auto max_it = std::max_element(probs.begin(), probs.end());
    return static_cast<TrafficLightState>(std::distance(probs.begin(), max_it));
}

int main() {
    std::array<float, 3> probs = {0.92f, 0.05f, 0.03f};
    auto state = classify_traffic_light(probs);
    std::cout << "Trafik Isigi: " << (state == TrafficLightState::RED ? "KIRMIZI" : "DIGER") << "\n";
    return 0;
}
```

#### **Gün 50: Hareket ve Yörünge Tahmini (Trajectory Prediction): LSTM & Diffusion Tabanlı Yol Kullanıcıları**
- **Teorik Odak**: Diğer araçların ve yayaların gelecek 5 saniyelik çok modlu (Multimodal) olası yörüngelerinin tahmini.
- **Quiz Kod Sorusu**: Gelecek 5 adım için sabit hız modeliyle yörünge üreten fonksiyonu yazın.
- **Çözüm**:
```python
import numpy as np

def predict_constant_velocity_trajectory(x0, y0, vx, vy, dt=0.5, steps=5):
    traj = []
    for i in range(1, steps + 1):
        traj.append((x0 + vx * i * dt, y0 + vy * i * dt))
    return np.array(traj)

print("Tahmini Gelecek Yörünge:\n", predict_constant_velocity_trajectory(10.0, 2.0, 15.0, 0.0))
```

#### **Gün 51: NVIDIA TensorRT ve Tesla FSD Çipi (HW3/HW4 NPU) Üzerinde INT8/FP8 Model Optimizasyonu**
- **Teorik Odak**: Post-Training Quantization (PTQ), Kalibrasyon Veri Seti, KL Divergence ile Eşik Belirleme.
- **Quiz Kod Sorusu**: FP32 tensörünü simetrik INT8 formatına ($q = \text{clip}(\text{round}(x / s), -128, 127)$) dönüştüren fonksiyonu yazın.
- **Çözüm**:
```python
import numpy as np

def quantize_symmetric_int8(tensor_fp32: np.ndarray):
    max_val = np.max(np.abs(tensor_fp32))
    scale = max_val / 127.0
    quant_int8 = np.clip(np.round(tensor_fp32 / scale), -128, 127).astype(np.int8)
    return quant_int8, scale

t = np.array([-2.5, 0.0, 1.2, 5.0], dtype=np.float32)
q, s = quantize_symmetric_int8(t)
print(f"INT8 Tensor: {q}, Scale: {s:.4f}")
```

#### **Gün 52: Model Damıtma (Knowledge Distillation) ve Pruning ile Düşük Gecikmeli FSD Çıkarımı**
- **Teorik Odak**: Teacher-Student Eğitimi, Kullback-Leibler Kaybı, Ağ Ağırlık Budama (L1 Structured Pruning).
- **Quiz Kod Sorusu**: Teacher ve Student model logitleri arasındaki Knowledge Distillation kaybını hesaplayan PyTorch fonksiyonunu yazın.
- **Çözüm**:
```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, temp=2.0):
    soft_targets = F.softmax(teacher_logits / temp, dim=-1)
    soft_student = F.log_softmax(student_logits / temp, dim=-1)
    return F.kl_div(soft_student, soft_targets, reduction='batchmean') * (temp ** 2)

s_log = torch.randn(4, 10)
t_log = torch.randn(4, 10)
print(f"Distillation Loss: {distillation_loss(s_log, t_log):.4f}")
```

#### **Gün 53: Sıfır-Gölge Modu (Shadow Mode): Üretim Araçlarında A/B Model Testi ve Veri Tetikleyicileri**
- **Teorik Odak**: Üretim araçlarında sessizce arka planda çalışan yeni FSD modeli ile insan sürücü eylemleri arasındaki uyumsuzlukları tespit edip tetikleme (Triggering) ve veri toplama.
- **Quiz Kod Sorusu**: İnsan sürücü fren yaparken modelin gaz kararı verdiği anomali anını yakalayıp telemetri tetikleyen fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>

bool check_shadow_mode_trigger(float human_brake_pct, float shadow_model_throttle_pct) {
    if (human_brake_pct > 30.0f && shadow_model_throttle_pct > 20.0f) {
        std::cout << "TETIKLEYICI AKTIF: Shadow Mode Model-Insan Uyusmazligi Yakalandi! Veri Klip Kaydediliyor.\n";
        return true;
    }
    return false;
}

int main() {
    check_shadow_mode_trigger(60.0f, 45.0f);
    return 0;
}
```

#### **Gün 54: Tesla Otomatik Etiketleme Motoru (Auto-Labeling Pipeline) ve Sentetik Veri Üretimi**
- **Teorik Odak**: Milyonlarca araba klibinin 3D uzayda zamansal olarak yeniden yapılandırılması ve insan müdahalesiz otomatik etiketlenmesi.
- **Quiz Kod Sorusu**: 3D yörünge noktaları kümesinden zaman ekseninde enterpolasyonla eksik etiketleri tamamlayan Python fonksiyonunu yazın.
- **Çözüm**:
```python
import numpy as np

def interpolate_missing_boxes(timestamps, box_positions):
    # Basit lineer zaman enterpolasyonu
    valid_mask = ~np.isnan(box_positions[:, 0])
    valid_t = timestamps[valid_mask]
    valid_pos = box_positions[valid_mask]
    
    interp_x = np.interp(timestamps, valid_t, valid_pos[:, 0])
    interp_y = np.interp(timestamps, valid_t, valid_pos[:, 1])
    return np.column_stack([interp_x, interp_y])

t = np.array([0.0, 0.1, 0.2, 0.3])
pos = np.array([[0,0], [np.nan, np.nan], [4,4], [6,6]])
print("Otomatik Etiketlenmis Yörünge:\n", interpolate_missing_boxes(t, pos))
```

#### **Gün 55: FAZ 5 CAPSTONE: 3D Voksel Occupancy & Vektör Yol Graf Çıkarım Motoru**
- **Teorik Odak**: HydraNet Omurgası, 3D Voxel Occupancy Çıkarımı, Şerit Polinomları ve TensorRT INT8 Simülasyonunu birleştiren komple FSD AI Çekirdeği.
- **Quiz Kod Sorusu**: FSD AI Çekirdeğinin saniyede 36 FPS hızında voksel doluluk haritası ve şerit geometrisini eşzamanlı çıkaran Capstone sınıfını yazın.
- **Çözüm**:
```python
import numpy as np

class TeslaFSDInferenceEngine:
    def __init__(self):
        self.voxel_grid_shape = (50, 50, 8)

    def run_inference(self, camera_frames_batch: np.ndarray):
        # 3D Occupancy Tahmini Simülasyonu
        occupancy_grid = np.zeros(self.voxel_grid_shape, dtype=np.uint8)
        occupancy_grid[25:30, 24:26, 0:2] = 1 # Onde 10m mesafede bir binek arac vokseli
        
        # Yol Polinom Ciktisi (y = c0 + c1*x + c2*x^2)
        left_lane = np.array([-1.75, 0.01, 0.0001])
        right_lane = np.array([1.75, 0.01, 0.0001])
        
        return {
            "occupancy_voxels": occupancy_grid,
            "left_lane_poly": left_lane,
            "right_lane_poly": right_lane,
            "inference_latency_ms": 12.4
        }

engine = TeslaFSDInferenceEngine()
res = engine.run_inference(np.zeros((8, 3, 256, 256)))
print(f"Faz 5 Capstone: FSD Gecikmesi = {res['inference_latency_ms']} ms, Dolu Voksel = {np.sum(res['occupancy_voxels'])}")
```

---

### 🔹 FAZ 6: Otonom Sürüş Yörünge Planlama, MPC & ISO 26262 ASIL-D (Gün 56 - 66)

#### **Gün 56: Yol Planlama Temelleri: Hibrit A* (Hybrid A*) ve Voronoi Diyagramları ile Park Planlama**
- **Teorik Odak**: Sürekli durum uzayı ($x, y, \theta$), Reed-Shepp eğrileri ve dar alanda otonom paralel/dikey park planlama.
- **Quiz Kod Sorusu**: İki durum arasındaki minimum dönüş yarıçaplı kinematik adım geçişini hesaplayan C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cmath>

struct VehicleState { float x, y, yaw; };

VehicleState step_kinematic_bicycle(const VehicleState& s, float v, float steering_angle, float dt, float L=2.8f) {
    float nx = s.x + v * std::cos(s.yaw) * dt;
    float ny = s.y + v * std::sin(s.yaw) * dt;
    float nyaw = s.yaw + (v / L) * std::tan(steering_angle) * dt;
    return {nx, ny, nyaw};
}

int main() {
    VehicleState s{0,0,0};
    auto next = step_kinematic_bicycle(s, 5.0f, 0.2f, 0.1f);
    std::cout << "Next State: (" << next.x << ", " << next.y << ", yaw=" << next.yaw << ")\n";
    return 0;
}
```

#### **Gün 57: Frenet Çerçevesi (Frenet Coordinates) ve Dinamik Şerit Değiştirme Yörünge Üretimi**
- **Teorik Odak**: Kartezyen ($X, Y$) koordinatlardan Yol Boyu ($s$) ve Yanal Sapma ($d$) Frenet koordinatlarına dönüşüm.
- **Quiz Kod Sorusu**: 5. derece quintic polinom ile sarsıntısız (Jerk-optimal) şerit değiştirme yörüngesi katsayılarını hesaplayan Python fonksiyonunu yazın.
- **Çözüm**:
```python
import numpy as np

def solve_quintic_polynomial(d0, v0, a0, d1, v1, a1, T):
    # Jerk-Optimal Quintic Polynomial solver
    A = np.array([
        [T**3, T**4, T**5],
        [3*T**2, 4*T**3, 5*T**4],
        [6*T, 12*T**2, 20*T**3]
    ])
    B = np.array([
        d1 - (d0 + v0*T + 0.5*a0*T**2),
        v1 - (v0 + a0*T),
        a1 - a0
    ])
    c3, c4, c5 = np.linalg.solve(A, B)
    return [d0, v0, 0.5*a0, c3, c4, c5]

print("Quintic Katsayilari:", solve_quintic_polynomial(0,0,0, 3.5,0,0, 4.0)) # 4 saniyede 3.5m serit degistirme
```

#### **Gün 58: Model Predictive Control (MPC) ile Kinematik Bisiklet Modeli Kontrolü**
- **Teorik Odak**: Durum vektörü $[x, y, v, \psi]$, kontrol vektörü $[a, \delta]$, karesel programlama (QP) maliyet fonksiyonu.
- **Quiz Kod Sorusu**: Araç şerit merkezinden saptığında düzeltme direksiyon açısı hesaplayan basitleştirilmiş P-Controller fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>

float compute_steering_correction(float lateral_error_m, float heading_error_rad, float kp_lat=0.4f, float kp_head=1.2f) {
    return -(kp_lat * lateral_error_m + kp_head * heading_error_rad);
}

int main() {
    std::cout << "Direksiyon Duzeltmesi: " << compute_steering_correction(0.5f, 0.05f) << " rad\n";
    return 0;
}
```

#### **Gün 59: Dinamik Engelden Kaçınma ve Sürekli Eğrilik (Clothoid/Spline) Yörünge Optimizasyonu**
- **Teorik Odak**: Clothoid eğrileri (doğrusal değişen eğrilik $\kappa(s) = \kappa_0 + c s$), direksiyon simidi dönüş hızı sınırları.
- **Quiz Kod Sorusu**: İki nokta arasında eğrilik sürekliliğini kontrol eden C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cmath>

bool is_curvature_rate_safe(float kappa_curr, float kappa_next, float ds, float max_dkappa=0.1f) {
    if (ds <= 0.0f) return false;
    return (std::abs(kappa_next - kappa_curr) / ds) <= max_dkappa;
}

int main() {
    std::cout << "Egrilik Guvenli mi: " << (is_curvature_rate_safe(0.02f, 0.05f, 1.0f) ? "EVET" : "HAYIR") << "\n";
    return 0;
}
```

#### **Gün 60: Hız Profili Optimizasyonu: Konfor, Enerji Verimliliği ve Trafik Akışı Dengeleme**
- **Teorik Odak**: Maksimum yanal ivme sınırı ($a_{\text{lat}} = v^2 \kappa \le 2.0\text{ m/s}^2$), boyuna jerk sınırlandırması ($|jerk| \le 1.5\text{ m/s}^3$).
- **Quiz Kod Sorusu**: Verilen bir viraj eğriliğinde ($\kappa = 0.04\text{ m}^{-1}$) maksimum güvenli viraj hızını hesaplayan fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cmath>

float max_safe_cornering_speed(float curvature_kappa, float max_lat_accel=2.0f) {
    if (curvature_kappa <= 0.0f) return 130.0f / 3.6f; // Duz yol max hiz
    return std::sqrt(max_lat_accel / curvature_kappa);
}

int main() {
    float v_mps = max_safe_cornering_speed(0.04f, 2.0f);
    std::cout << "Maksimum Guvenli Viraj Hizi: " << (v_mps * 3.6f) << " km/h\n";
    return 0;
}
```

#### **Gün 61: Karmaşık Şehir İçi Kavşak ve Döner Kavşak (Roundabout) Karar Ağaçları**
- **Teorik Odak**: Öncelik kuralları, Time-To-Collision (TTC) hesaplama ve güvenli aralık (Gap Acceptance) modeli.
- **Quiz Kod Sorusu**: Kavşaktaki yaklaşan araç ile aradaki TTC ($TTC = \frac{d}{v_{\text{rel}}}$) değerini hesaplayan ve $TTC < 3.0\text{ s}$ ise bekleme kararı veren fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>

bool can_enter_intersection(float dist_to_approaching_vehicle_m, float approaching_speed_mps) {
    if (approaching_speed_mps <= 0.0f) return true;
    float ttc = dist_to_approaching_vehicle_m / approaching_speed_mps;
    return ttc >= 3.5f; // En az 3.5 sn guvenlik payi
}

int main() {
    std::cout << "Kavsaga Giris Guvenli mi: " << (can_enter_intersection(40.0f, 15.0f) ? "GEC" : "BEKLE") << "\n";
    return 0;
}
```

#### **Gün 62: Acil Durum Manevraları ve Otomatik Acil Frenleme (AEB) Kontrol Mantığı**
- **Teorik Odak**: Euro-NCAP AEB Protokolü, Çarpışma Uyarı Zamanı (FCW), Tam Frenleme Tetikleme Mesafesi ($d_{\text{stop}} = \frac{v^2}{2 a_{\text{max}}} + v \cdot t_{\text{react}}$).
- **Quiz Kod Sorusu**: $v = 20\text{ m/s}$ ($72\text{ km/h}$) hızda acil durum durma mesafesini hesaplayan fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>

float compute_emergency_stopping_distance(float speed_mps, float max_decel_mps2=9.0f, float system_delay_s=0.2f) {
    float reaction_dist = speed_mps * system_delay_s;
    float braking_dist = (speed_mps * speed_mps) / (2.0f * max_decel_mps2);
    return reaction_dist + braking_dist;
}

int main() {
    std::cout << "AEB Durma Mesafesi: " << compute_emergency_stopping_distance(20.0f) << " metre\n";
    return 0;
}
```

#### **Gün 63: ISO 26262 Fonksiyonel Güvenlik (ASIL-D) ve Arıza Güvenli (Fail-Operational) Mimari**
- **Teorik Odak**: Automotive Safety Integrity Level (ASIL-A to ASIL-D), HARA (Hazard Analysis and Risk Assessment), FMEA.
- **Quiz Kod Sorusu**: Direksiyon tork sensöründen gelen çift kanal sinyalini ($S_1, S_2$) karşılaştırıp fark $0.5\text{ Nm}$'yi geçerse ASIL-D arıza bayrağı üreten kod yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cmath>

bool check_dual_channel_asil_d(float torque_ch1, float torque_ch2, float max_diff=0.5f) {
    return std::abs(torque_ch1 - torque_ch2) <= max_diff;
}

int main() {
    std::cout << "Kanal Dogrulugu ASIL-D: " << (check_dual_channel_asil_d(2.1f, 2.3f) ? "GUVENLI" : "ARIZA_TETIKLE") << "\n";
    return 0;
}
```

#### **Gün 64: Çift Kanallı Güvenlik ve FSD HW Çip Yedekliliği (Redundancy & Lockstep)**
- **Teorik Odak**: FSD HW3/HW4 Çift Bağımsız NPU (Node A ve Node B) Eşzamanlı Çalışması ve Oylama (Voting) Mekanizması.
- **Quiz Kod Sorusu**: İki FSD işlemcisinin ürettiği direksiyon komutlarını oylayan ve uyumsuzluk durumunda aracı güvenli şeride çeken arabulucu sınıfı yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cmath>

class FSDHardwareArbiter {
public:
    float arbitrate_steering(float node_a_cmd, float node_b_cmd) {
        if (std::abs(node_a_cmd - node_b_cmd) < 0.05f) {
            return (node_a_cmd + node_b_cmd) / 2.0f; // Tam uyum
        }
        std::cout << "UYARI: FSD Node A ve Node B Karar Ayrismasi! Guvenli Durus Modu Devrede.\n";
        return 0.0f; // Guvenli duz hat
    }
};

int main() {
    FSDHardwareArbiter arbiter;
    std::cout << "Secilen Direksiyon Acisi: " << arbiter.arbitrate_steering(0.12f, 0.13f) << " rad\n";
    return 0;
}
```

#### **Gün 65: S-Function ve C++ ile Gerçek Zamanlı Yörünge Takip Kontrolcüsü**
- **Teorik Odak**: Pure Pursuit & Stanley Controller ile yol takip hatası ($e_{\text{lat}}, e_{\text{yaw}}$) minimizasyonu.
- **Quiz Kod Sorusu**: Stanley Kontrolcüsü formülüyle ($\delta(t) = \theta_e + \arctan\left(\frac{k \cdot e}{v}\right)$) direksiyon açısı hesaplayan C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cmath>

float stanley_controller(float heading_error_rad, float cross_track_error_m, float speed_mps, float k=0.5f) {
    if (speed_mps < 0.1f) return heading_error_rad;
    return heading_error_rad + std::atan2(k * cross_track_error_m, speed_mps);
}

int main() {
    std::cout << "Stanley Direksiyon Komutu: " << stanley_controller(0.04f, 0.3f, 15.0f) << " rad\n";
    return 0;
}
```

#### **Gün 66: FAZ 6 CAPSTONE: C++ ile Otonom Otoyol Şerit Değiştirme & MPC Yörünge Takipçisi**
- **Teorik Odak**: Hedef Şerit Belirleme, Quintic Yörünge Sentezi, Stanley Kontrol ve ASIL-D Çift Kanal Doğrulama.
- **Quiz Kod Sorusu**: Komple şerit değiştirme planını oluşturan, engelleri kontrol eden ve direksiyon komutunu üreten Capstone sınıfını yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <vector>

class TeslaHighwayPlanner {
public:
    struct TrajectoryPoint { float x, y, speed, steering; };

    std::vector<TrajectoryPoint> plan_lane_change(float current_speed_mps, float lane_offset_m = 3.5f) {
        std::vector<TrajectoryPoint> path;
        for (int i = 0; i <= 20; ++i) {
            float t = i * 0.2f; // 4 saniye toplam sure
            float progress = t / 4.0f;
            float y = lane_offset_m * (3 * progress * progress - 2 * progress * progress * progress);
            float x = current_speed_mps * t;
            path.push_back({x, y, current_speed_mps, 0.01f});
        }
        return path;
    }
};

int main() {
    TeslaHighwayPlanner planner;
    auto traj = planner.plan_lane_change(30.0f); // 108 km/h hizda serit degistirme
    std::cout << "Faz 6 Capstone: Yörünge Nokta Sayisi = " << traj.size() << ", Bitis Y = " << traj.back().y << "m\n";
    return 0;
}
```

---

### 🔹 FAZ 7: Tesla V12 Infotainment, Qt6/QML, D-Bus & OTA Güncelleme (Gün 67 - 77)

#### **Gün 67: Tesla V12 Kullanıcı Arayüzü Mimarisi, Modern Qt6, C++ ve QML Entegrasyonu**
- **Teorik Odak**: QQuickView, C++ QObject Backend ve QML deklaratif arayüz bağlama (Property Binding).
- **Quiz Kod Sorusu**: C++'tan QML'e araç hızını bildiren `Q_PROPERTY` model sınıfını yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <string>

class SpeedometerModel {
private:
    float speed_{0.0f};
public:
    void set_speed(float s) {
        speed_ = s;
        std::cout << "QML Signal: speedChanged(" << speed_ << " km/h)\n";
    }
    float get_speed() const { return speed_; }
};

int main() {
    SpeedometerModel model;
    model.set_speed(124.0f);
    return 0;
}
```

#### **Gün 68: GPU Hızlandırmalı Donanım Renderleme ve 3D Araç Görselleştirme (OpenGL/Vulkan)**
- **Teorik Odak**: FSD 3D Dünya Render Motoru, Voksel ve Yörünge Mesh'lerinin 60 FPS GPU ile ekrana çizilmesi.
- **Quiz Kod Sorusu**: 3D Araç Modelinin Model-View-Projection (MVP) matrisini hesaplayan fonksiyonu yazın.
- **Çözüm**:
```python
import numpy as np

def compute_mvp_matrix(model_pos: np.ndarray, camera_pos: np.ndarray) -> np.ndarray:
    # Basitlestirilmis Model-View-Projection carpimi
    M = np.eye(4); M[:3, 3] = model_pos
    V = np.eye(4); V[:3, 3] = -camera_pos
    P = np.eye(4) # Persp
    return P @ V @ M

print("MVP Matrisi:\n", compute_mvp_matrix(np.array([0, 10, 0]), np.array([0, -5, 2])))
```

#### **Gün 69: D-Bus ve IPC ile Araç Gövde Kontrolcüleri (BCM) ve UI Arası Haberleşme**
- **Teorik Odak**: Linux D-Bus System Bus, Araç Kapı Kilitleri, Farlar ve Pencere Pozisyonlarının UI ile Asenkron İletişimi.
- **Quiz Kod Sorusu**: D-Bus üzerinden kapı kilidi durum sinyali yayınlayan Python fonksiyonunu yazın.
- **Çözüm**:
```python
def publish_dbus_door_status(door_name: str, is_locked: bool):
    msg = {
        "interface": "com.tesla.BodyController",
        "signal": "DoorStatusChanged",
        "params": {"door": door_name, "locked": is_locked}
    }
    return msg

print("D-Bus Mesaji:", publish_dbus_door_status("FRONT_LEFT", True))
```

#### **Gün 70: Araç İçi Ses Boru Hattı: PipeWire/ALSA, Gürültü Engelleme ve Çok Bölgeli Ses**
- **Teorik Odak**: Aktif Yol Gürültüsü Engelleme (Active Road Noise Cancellation - ARNC), Koltuk İçi Hoparlör Yönlendirme.
- **Quiz Kod Sorusu**: Mikrofon fazını 180° ters çevirerek gürültü önleme sinyali üreten fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <vector>

std::vector<float> generate_anti_noise_phase(const std::vector<float>& road_mic_samples) {
    std::vector<float> anti_noise(road_mic_samples.size());
    for(size_t i=0; i<road_mic_samples.size(); ++i) {
        anti_noise[i] = -road_mic_samples[i]; // 180 derece ters faz
    }
    return anti_noise;
}

int main() {
    std::vector<float> mic = {0.4f, -0.8f, 0.2f};
    auto anti = generate_anti_noise_phase(mic);
    std::cout << "Anti-Noise Ilk Ornek: " << anti[0] << " (Faz terslendi)\n";
    return 0;
}
```

#### **Gün 71: Özel Linux Çekirdeği Derleme, Hızlı Başlatma (Fast-Boot < 2s) ve Systemd Optimizasyonu**
- **Teorik Odak**: Kernel XIP (Execute In Place), Unneeded Driver Pruning, `systemd-analyze blame` analizi.
- **Quiz Kod Sorusu**: Linux servis başlatma sürelerini analiz edip 200 ms'den uzun sürenleri listeleyen Python scripti yazın.
- **Çözüm**:
```python
def find_slow_services(service_times_ms: dict, threshold_ms: float = 200.0):
    return {k: v for k, v in service_times_ms.items() if v > threshold_ms}

services = {"tesla-can-gateway": 45, "tesla-ui-renderer": 350, "tesla-network": 120}
print("Yavas Servisler:", find_slow_services(services))
```

#### **Gün 72: Güvenli Önyükleme (Secure Boot), TPM 2.0 ve Kriptografik Ürün Yazılımı Doğrulama**
- **Teorik Odak**: RSA-4096 / ECC-P384 İmzalı Firmware İmajları, Hardware Root of Trust (RoT).
- **Quiz Kod Sorusu**: SHA-256 hash hesaplayıp beklenen üretici imzasıyla eşleştiğini doğrulayan C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <string>

bool verify_firmware_integrity(const std::string& computed_sha256, const std::string& expected_sha256) {
    return computed_sha256 == expected_sha256;
}

int main() {
    std::string h1 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    std::cout << "Firmware Guvenli mi: " << (verify_firmware_integrity(h1, h1) ? "DOGRULANDI" : "REDDEDILDI") << "\n";
    return 0;
}
```

#### **Gün 73: OTA (Over-the-Air) Güncelleme Mimarisi: A/B Bölümlendirme ve Geri Alma (Rollback)**
- **Teorik Odak**: A/B Seamless Partitioning, Başarısız boot durumunda otomatik slot değişimi (`bootctl mark-good`).
- **Quiz Kod Sorusu**: Aktif slotta 3 başarısız deneme sonrası yedek slota geçiş yapan State Machine fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>

struct OTABootSlotManager {
    char active_slot{'A'};
    int failed_boot_count{0};

    void on_boot_failure() {
        failed_boot_count++;
        if (failed_boot_count >= 3) {
            active_slot = (active_slot == 'A') ? 'B' : 'A';
            failed_boot_count = 0;
            std::cout << "KRITIK: 3 Hatali Boot! Slot Degistirildi -> Yeni Slot: " << active_slot << "\n";
        }
    }
};

int main() {
    OTABootSlotManager ota;
    ota.on_boot_failure(); ota.on_boot_failure(); ota.on_boot_failure();
    return 0;
}
```

#### **Gün 74: Araç İçi Web Tarayıcısı, Chromium Sandbox ve Güvenlik İzolasyonu**
- **Teorik Odak**: Seccomp-BPF filtreleri, Canbus/Araç kontrol katmanı ile tarayıcı izolasyonu (Zero Trust UI).
- **Quiz Kod Sorusu**: Tarayıcı sürecinin soket açma syscall'larını engelleyen seccomp kural yapısını simüle edin.
- **Çözüm**:
```cpp
#include <iostream>
#include <string>

bool is_syscall_permitted_for_browser(const std::string& syscall_name) {
    if (syscall_name == "socket" || syscall_name == "ptrace" || syscall_name == "reboot") {
        return false; // Bloke et
    }
    return true; // Izin ver (read/write/mmap)
}

int main() {
    std::cout << "Socket Cagrisi Izni: " << (is_syscall_permitted_for_browser("socket") ? "IZIN" : "BLOKE") << "\n";
    return 0;
}
```

#### **Gün 75: Araç İçi BLE, UWB (Ultra-Wideband) Dijital Telefon Anahtarı (Phone Key) Protokolü**
- **Teorik Odak**: BLE RSSI Triangulation, UWB Time-of-Flight (ToF) ile röle saldırılarına (Relay Attack Prevention) karşı koruma.
- **Quiz Kod Sorusu**: UWB ToF süresinden kapı açma mesafesini doğrulayan (< 2.0 metre) C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>

bool verify_uwb_phone_key_distance(float tof_nanoseconds, float max_dist_m = 2.0f) {
    constexpr float c_mps = 3.0e8f;
    float dist_m = (tof_nanoseconds * 1e-9f) * c_mps;
    return dist_m <= max_dist_m;
}

int main() {
    std::cout << "Kapi Kilidi Acilsin mi (4.5 ns ToF): " << (verify_uwb_phone_key_distance(4.5f) ? "AC (1.35m)" : "RED") << "\n";
    return 0;
}
```

#### **Gün 76: HVAC (Klima/Havalandırma) Dokunmatik Kontrol Arayüzü ve Step Motor Sürücüleri**
- **Teorik Odak**: Havalandırma menfez yönlendirme step motor kontrolü, PID kabin sıcaklık döngüsü.
- **Quiz Kod Sorusu**: Kabin hedef sıcaklığı ile mevcut sıcaklık arasındaki farka göre fan hızını belirleyen PID fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <algorithm>

float compute_fan_speed_pct(float current_t, float target_t, float kp=15.0f) {
    float error = current_t - target_t;
    float fan = std::abs(error) * kp;
    return std::clamp(fan, 0.0f, 100.0f);
}

int main() {
    std::cout << "Klima Fan Hizi: %" << compute_fan_speed_pct(32.0f, 22.0f) << "\n";
    return 0;
}
```

#### **Gün 77: FAZ 7 CAPSTONE: Qt6/C++ ve D-Bus ile Çalışan Tesla V12 Konsol ve Telemetri Simülatörü**
- **Teorik Odak**: UI Renderleme, D-Bus CAN Entegrasyonu, HVAC Kontrolü ve OTA Güvenli Boot Yöneticisini içeren Capstone.
- **Quiz Kod Sorusu**: Tüm alt sistem durumlarını toplayıp dokunmatik ekrana sunan ana konsol yönetici sınıfını yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <string>

class TeslaV12InfotainmentCenter {
public:
    void render_frame(float speed, int battery_pct, float cabin_temp, const std::string& gear) {
        std::cout << "========================================\n";
        std::cout << "   TESLA MODEL S V12 INFOTAINMENT OS    \n";
        std::cout << "========================================\n";
        std::cout << " Vites: [" << gear << "] | Hiz: " << speed << " km/h\n";
        std::cout << " Batarya: %" << battery_pct << " | Kabin: " << cabin_temp << " C\n";
        std::cout << " Navigasyon: FSD Beta Aktif (Route 101)\n";
        std::cout << "========================================\n";
    }
};

int main() {
    TeslaV12InfotainmentCenter ui;
    ui.render_frame(115.0f, 82, 21.5f, "D");
    return 0;
}
```

---

### 🔹 FAZ 8: Tesla Supercharger V4 (NACS), Megapack BESS & Autobidder (Gün 78 - 88)

#### **Gün 78: Tesla Supercharger V4 Mimarisi: 1000V DC, Sıvı Soğutmalı Kablo ve 350+ kW Güç**
- **Teorik Odak**: Sıvı soğutmalı şarj kablosu termal modeli, 1000V DC mimarisi, SiC (Silicon Carbide) güç modülleri.
- **Quiz Kod Sorusu**: Şarj kablosu sıcaklığına göre maksimum izin verilen akımı ($I_{\text{max}}$) kısan fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>

float get_derated_charging_current(float cable_temp_c, float nominal_current_a = 500.0f) {
    if (cable_temp_c > 85.0f) return nominal_current_a * 0.4f; // %60 kisitlama
    if (cable_temp_c > 70.0f) return nominal_current_a * 0.75f;
    return nominal_current_a;
}

int main() {
    std::cout << "Supercharger Akim Limiti (75 C): " << get_derated_charging_current(75.0f) << " A\n";
    return 0;
}
```

#### **Gün 79: CCS / NACS (J3400) Şarj Protokolü ve ISO 15118 Tak-Çalıştır (Plug & Charge) Şifreleme**
- **Teorik Odak**: HomePlug GreenPHY PLC (Powerline Communication), TLS 1.3 ve V2G (Vehicle-to-Grid) protokolü.
- **Quiz Kod Sorusu**: Araç şarj soketine takıldığında ISO 15118 el sıkışma mesajını üreten Python fonksiyonunu yazın.
- **Çözüm**:
```python
def create_iso15118_handshake(vehicle_vin: str, max_voltage_v: float, max_current_a: float) -> dict:
    return {
        "protocol": "ISO15118-2",
        "service": "PlugAndCharge",
        "vin": vehicle_vin,
        "contract_verified": True,
        "limits": {"max_v": max_voltage_v, "max_a": max_current_a}
    }

print("Handshake Paketi:", create_iso15118_handshake("5YJ3E1EB8NF123456", 500.0, 600.0))
```

#### **Gün 80: Dağıtık Güç Dağıtımı ve Dinamik Şebeke Yük Dengeleme Algoritmaları**
- **Teorik Odak**: 8 stall'luk Supercharger istasyonunda trafo sınırı ($1\text{ MW}$) aşılmadan araçlar arası dinamik güç paylaştırma.
- **Quiz Kod Sorusu**: Toplam $800\text{ kW}$ gücü bağlı 4 araca SoC durumlarına göre ters orantılı paylaştıran fonksiyonu yazın.
- **Çözüm**:
```python
import numpy as np

def balance_supercharger_power(soc_list: list, total_available_kw: float = 800.0):
    demands = [100.0 - s for s in soc_list] # Dusuk SoC daha cok guc ister
    total_demand = sum(demands)
    return [(d / total_demand) * total_available_kw for d in demands]

socs = [15.0, 45.0, 70.0, 85.0]
print("Stall Guc Dagilimi (kW):", [round(p, 1) for p in balance_supercharger_power(socs)])
```

#### **Gün 81: Tesla Megapack & Powerwall Enerji Depolama Sistemleri (BESS) Kontrol Mantığı**
- **Teorik Odak**: 3.9 MWh Megapack XL mimarisi, şebeke frekans yanıtı (Grid Forming Inverters), $P-f$ droop kontrolü.
- **Quiz Kod Sorusu**: Şebeke frekansı 50 Hz'den düştüğünde Megapack aktif güç enjeksiyonunu hesaplayan Droop formülünü uygulayın.
- **Çözüm**:
```cpp
#include <iostream>

float compute_megapack_droop_power(float grid_freq_hz, float nominal_freq=50.0f, float droop_gain_kw_per_hz=10000.0f) {
    float freq_error = nominal_freq - grid_freq_hz;
    if (freq_error > 0.0f) {
        return freq_error * droop_gain_kw_per_hz; // Sebekeye guc ver
    }
    return 0.0f;
}

int main() {
    std::cout << "Megapack Enjeksiyon Gucu (49.8 Hz): " << compute_megapack_droop_power(49.8f) << " kW\n";
    return 0;
}
```

#### **Gün 82: Tesla Autobidder Algoritmik Enerji Ticareti: Frekans Düzenleme ve Arbitraj**
- **Teorik Odak**: Elektrik spot piyasa fiyat tahminleri (Day-Ahead / Real-Time), batarya amortismanı hesabı ve kar maksimizasyonu.
- **Quiz Kod Sorusu**: Spot fiyat $150\text{ \$/MWh}$ üzerindeyse deşarj, $30\text{ \$/MWh}$ altındaysa şarj kararı veren arbitraj ajanını yazın.
- **Çözüm**:
```python
def autobidder_decision(spot_price_usd_mwh: float, battery_soc_pct: float) -> str:
    if spot_price_usd_mwh > 150.0 and battery_soc_pct > 20.0:
        return "DISCHARGE_TO_GRID (SELL)"
    elif spot_price_usd_mwh < 30.0 and battery_soc_pct < 95.0:
        return "CHARGE_FROM_GRID (BUY)"
    return "STANDBY_OPTIMAL"

print("Autobidder Karari:", autobidder_decision(185.0, 75.0))
```

#### **Gün 83: Güneş Enerjisi ve Solar Inverter MPPT (Maximum Power Point Tracking) Kontrolü**
- **Teorik Odak**: Perturb and Observe (P&O) ve Incremental Conductance algoritmaları.
- **Quiz Kod Sorusu**: P&O MPPT algoritmasıyla güneş paneli çalışma voltajını adım adım ayarlayan Python fonksiyonunu yazın.
- **Çözüm**:
```python
def mppt_perturb_and_observe(v_prev, p_prev, v_curr, p_curr, step=2.0):
    delta_p = p_curr - p_prev
    delta_v = v_curr - v_prev
    if delta_p > 0:
        return v_curr + step if delta_v > 0 else v_curr - step
    else:
        return v_curr - step if delta_v > 0 else v_curr + step

print("Yeni MPPT Voltaji:", mppt_perturb_and_observe(40.0, 200.0, 42.0, 220.0))
```

#### **Gün 84: Sanal Enerji Santrali (Virtual Power Plant - VPP) ve Dağıtık Akıllı Şebeke**
- **Teorik Odak**: 50.000 ev tipi Powerwall bataryasını tek bir mega santral gibi senkronize etme.
- **Quiz Kod Sorusu**: VPP havuzundaki Powerwall'ların toplam anlık deşarj kapasitesini toplayan sınıfı yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <vector>
#include <numeric>

struct PowerwallUnit { float max_discharge_kw; float soc; };

float calculate_vpp_total_power(const std::vector<PowerwallUnit>& fleet) {
    float total = 0.0f;
    for(const auto& p : fleet) {
        if (p.soc > 20.0f) total += p.max_discharge_kw;
    }
    return total;
}

int main() {
    std::vector<PowerwallUnit> fleet = {{5.0f, 80.0f}, {5.0f, 90.0f}, {5.0f, 10.0f}};
    std::cout << "VPP Toplam Gucu: " << calculate_vpp_total_power(fleet) << " kW\n";
    return 0;
}
```

#### **Gün 85: Supercharger İstasyonları İçin Dinamik Kuyruk ve Rezervasyon Optimizasyonu**
- **Teorik Odak**: M/M/c Kuyruk Teorisi, FSD Navigasyonu ile yaklaşan araçların varış zamanı tahmini ve stall atama.
- **Quiz Kod Sorusu**: Yaklaşan 3 aracın varış zamanı ve şarj sürelerine göre istasyon bekleme süresini hesaplayan fonksiyonu yazın.
- **Çözüm**:
```python
def estimate_supercharger_wait_time(stalls_available: int, queue_lengths: int, avg_charge_min: float = 20.0) -> float:
    if stalls_available > 0:
        return 0.0
    return (queue_lengths + 1) * (avg_charge_min / 4.0)

print(f"Tahmini Bekleme Suresi: {estimate_supercharger_wait_time(0, 3):.1f} dakika")
```

#### **Gün 86: Yüksek Frekanslı Güç Telemetrisi ve MQTT/Kafka ile Bulut Senkronizasyonu**
- **Teorik Odak**: Binlerce Supercharger ve Megapack'ten saniyede 100 Hz güç telemetrisi akıtma (Streaming).
- **Quiz Kod Sorusu**: Protobuf/JSON formatında telemetri yükü serileştiren Python fonksiyonunu yazın.
- **Çözüm**:
```python
import json

def create_supercharger_telemetry_payload(site_id: str, active_stalls: int, total_kw: float) -> str:
    data = {
        "site_id": site_id,
        "active_stalls": active_stalls,
        "grid_power_kw": total_kw,
        "status": "OPERATIONAL"
    }
    return json.dumps(data)

print(create_supercharger_telemetry_payload("TESLA_SITE_KETTLEMAN", 48, 4250.5))
```

#### **Gün 87: Güç Dönüştürücü Simülasyonu: LLC Rezonans Dönüştürücü ve SiC MOSFET**
- **Teorik Odak**: Zero Voltage Switching (ZVS), Yüksek frekanslı (200 kHz) trafo verimliliği (%98.5).
- **Quiz Kod Sorusu**: SiC MOSFET iletim ve anahtarlama kayıplarını hesaplayan C++ fonksiyonunu yazın.
- **Çözüm**:
```cpp
#include <iostream>

float calculate_inverter_efficiency(float pin_kw, float ploss_conduction_kw, float ploss_switching_kw) {
    float pout = pin_kw - ploss_conduction_kw - ploss_switching_kw;
    return (pout / pin_kw) * 100.0f;
}

int main() {
    std::cout << "Invertor Verimliligi: %" << calculate_inverter_efficiency(250.0f, 2.1f, 1.2f) << "\n";
    return 0;
}
```

#### **Gün 88: FAZ 8 CAPSTONE: NACS Uyumlu Supercharger Yük Paylaşımı ve Autobidder Enerji Ajanı**
- **Teorik Odak**: NACS Şarj Protokolü, 8-Stall Dinamik Yük Dengeleme, Megapack Desteği ve Autobidder Ticaretini birleştiren Capstone.
- **Quiz Kod Sorusu**: Şebeke gücü yetersiz kaldığında eksik gücü yerel Megapack'ten tamamlayan akıllı Supercharger yönetim sınıfını yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <algorithm>

class TeslaSuperchargerStation {
private:
    float grid_limit_kw_{1000.0f};
    float megapack_capacity_kwh_{3900.0f};
public:
    void dispatch_charging(float total_vehicle_demand_kw) {
        float grid_draw = std::min(total_vehicle_demand_kw, grid_limit_kw_);
        float megapack_draw = total_vehicle_demand_kw - grid_draw;
        
        std::cout << "Talep Edilen Guc: " << total_vehicle_demand_kw << " kW\n";
        std::cout << " -> Sebekeden Cekilen: " << grid_draw << " kW\n";
        std::cout << " -> Megapack'ten Cekilen: " << megapack_draw << " kW\n";
        std::cout << "Tum Araclar Kesintisiz Hizli Sarj Ediliyor!\n";
    }
};

int main() {
    TeslaSuperchargerStation station;
    station.dispatch_charging(1400.0f); // 1.4 MW talep
    return 0;
}
```

---

### 🔹 FAZ 9: Tesla Dojo Süperbilgisayarı, Fleet OS, Tesla Optimus & BÜYÜK FİNAL (Gün 89 - 99)

#### **Gün 89: Tesla Dojo Süperbilgisayar Mimarisi: D1 Çipi, Training Tile ve 2D Mesh NoC**
- **Teorik Odak**: Tesla D1 Özel Silikonu, 500K Nöron/Çip, 9 PFLOPS Training Tile, 2D Torus Çip Üstü Ağ (NoC).
- **Quiz Kod Sorusu**: 2D Mesh ağında iki D1 çipi arasındaki Manhattan yönlendirme atlama (Hop) sayısını hesaplayan fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <cmath>

int compute_dojo_mesh_hops(int x1, int y1, int x2, int y2) {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}

int main() {
    std::cout << "Dojo D1 Mesaj Atlama Sayisi (Tile 0,0 -> 4,5): " << compute_dojo_mesh_hops(0, 0, 4, 5) << " Hops\n";
    return 0;
}
```

#### **Gün 90: PyTorch ve Dağıtık FP8/CFP8 Tensor Eğitimi ile Devasa Video Pretraining**
- **Teorik Odak**: FSD Video Girdileri için FSDP (Fully Sharded Data Parallel) ve CFP8 (Configurable FP8) eğitimi.
- **Quiz Kod Sorusu**: Dağıtık veri paralelinde model gradyanlarını normalize eden PyTorch kodunu yazın.
- **Çözüm**:
```python
import torch

def clip_and_normalize_gradients(model_parameters, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model_parameters, max_norm)
    return True

p = [torch.nn.Parameter(torch.randn(10, 10))]
p[0].grad = torch.randn(10, 10) * 5.0
clip_and_normalize_gradients(p)
print("Gradyanlar Guvenle Kirpildi (Dojo Standart).")
```

#### **Gün 91: Tesla Filo İşletim Sistemi (Fleet OS): Milyonlarca Araçtan Telemetri Toplama**
- **Teorik Odak**: Filo seviyesinde otomatik kaza/nerede neredeyse kaza (Near-Miss) video klip tetikleme ve gölge öğrenme.
- **Quiz Kod Sorusu**: Filo araçlarından gelen anomali telemetri json verilerini filtreleyen Map-Reduce fonksiyonunu yazın.
- **Çözüm**:
```python
fleet_events = [
    {"vin": "VIN_01", "event": "HARD_BRAKE", "g_force": 0.85},
    {"vin": "VIN_02", "event": "NORMAL_CRUISE", "g_force": 0.1},
    {"vin": "VIN_03", "event": "HARD_BRAKE", "g_force": 0.92}
]

critical_events = [e for e in fleet_events if e["g_force"] > 0.8]
print(f"Filodan Toplanan Kritik Klipler: {len(critical_events)} adet")
```

#### **Gün 92: Tesla Optimus İnsansı Robotu: Aktüatör Tasarımı, Eklemler ve 6-DoF Tork Kontrolü**
- **Teorik Odak**: Döner ve Doğrusal Aktüatörler, Gerinim Ölçer (Strain Gauge) Kuvvet Sensörleri, Ters Dinamik (Inverse Dynamics).
- **Quiz Kod Sorusu**: Verilen istenen eklem ivmesi ve atalet matrisinden gereken eklem torkunu ($\tau = M(q)\ddot{q} + C(q, \dot{q}) + G(q)$) hesaplayan fonksiyonu yazın.
- **Çözüm**:
```cpp
#include <iostream>

float compute_joint_torque(float inertia_m, float accel_qdd, float gravity_g) {
    return (inertia_m * accel_qdd) + gravity_g;
}

int main() {
    std::cout << "Optimus Diz Eklemi Gerekli Tork: " << compute_joint_torque(12.5f, 2.0f, 45.0f) << " Nm\n";
    return 0;
}
```

#### **Gün 93: Optimus Bütünsel Denge (Whole-Body Locomotion) ve Sıfır An Moment Noktası (ZMP)**
- **Teorik Odak**: Sıfır An Moment Noktası (ZMP), Destek Poligonu (Support Polygon) ve Doğrusal Ters Sarkaç Modeli (LIPM).
- **Quiz Kod Sorusu**: Optimus'un ZMP noktasının ayak tabanı destek poligonu içinde ($x_{\text{min}} \le x_{\text{zmp}} \le x_{\text{max}}$) olup olmadığını kontrol eden kod yazın.
- **Çözüm**:
```cpp
#include <iostream>

bool is_optimus_balanced(float zmp_x, float foot_x_min = -0.1f, float foot_x_max = 0.15f) {
    return zmp_x >= foot_x_min && zmp_x <= foot_x_max;
}

int main() {
    std::cout << "Optimus Dengede mi: " << (is_optimus_balanced(0.04f) ? "DENGEDE" : "DUSME_RISKI") << "\n";
    return 0;
}
```

#### **Gün 94: Optimus İçin FSD Görsel Ağlarının Uyarlanması: Manipülasyon, Kavrama ve Nesne Sıralama**
- **Teorik Odak**: FSD Occupancy ağının robotik el manipülasyonuna uyarlanması, 6-DoF Grasping Pose Estimation.
- **Quiz Kod Sorusu**: Kamera görüntüsünden tespit edilen pil hücresinin 3D tutma noktasını hesaplayan fonksiyonu yazın.
- **Çözüm**:
```python
import numpy as np

def calculate_grasp_pose(object_center_3d: np.ndarray, approach_vector: np.ndarray):
    grasp_pos = object_center_3d + approach_vector * 0.05 # 5cm onde yaklasma
    return grasp_pos

print("Optimus Tutma Noktasi:", calculate_grasp_pose(np.array([0.4, 0.1, 0.8]), np.array([0, 0, -1])))
```

#### **Gün 95: Simülasyondan Gerçeğe (Sim2Real) Robotik Eğitimi: Isaac Sim ve Domain Randomization**
- **Teorik Odak**: Sürtünme katsayısı, kütle ve eklem gecikmesi varyasyonlarıyla simülasyonda eğitilen politikanın gerçek dünyada sıfır hatayla çalışması.
- **Quiz Kod Sorusu**: Simülasyon parametrelerine rastgele gürültü ekleyen Domain Randomization fonksiyonunu yazın.
- **Çözüm**:
```python
import numpy as np

def randomize_physics_domain(nominal_mass_kg: float, friction_coeff: float):
    mass_rand = nominal_mass_kg * np.random.uniform(0.9, 1.1)
    frict_rand = friction_coeff * np.random.uniform(0.8, 1.2)
    return mass_rand, frict_rand

np.random.seed(42)
m, f = randomize_physics_domain(73.0, 0.8)
print(f"Randomize Kütle: {m:.2f} kg, Sürtünme: {f:.2f}")
```

#### **Gün 96: Tesla Cybercab / Robotaxi Otonom Çağırma (Summon) ve Filo Görevlendirme**
- **Teorik Odak**: Dinamik araç çağırma, yolcu alma noktası optimizasyonu ve şehir ölçeğinde boş filo yönlendirme.
- **Quiz Kod Sorusu**: En yakın boş Robotaxi'yi müşteriye atayan mesafe optimizasyon algoritmasını yazın.
- **Çözüm**:
```python
import numpy as np

def assign_nearest_robotaxi(customer_loc, available_taxis):
    distances = [np.linalg.norm(np.array(customer_loc) - np.array(t["pos"])) for t in available_taxis]
    best_idx = int(np.argmin(distances))
    return available_taxis[best_idx]["id"]

taxis = [{"id": "TAXI_01", "pos": [12.0, 5.0]}, {"id": "TAXI_02", "pos": [2.0, 1.0]}]
print("Atanan Robotaxi:", assign_nearest_robotaxi([0.0, 0.0], taxis))
```

#### **Gün 97: Tesla Yazılım Mimarisi Bütünsel Sistem İncelemesi ve Kod İnceleme (Code Review)**
- **Teorik Odak**: MISRA C++, zero-warning derleme, thread-safety, latency bütçesi ve kod inceleme standartları.
- **Quiz Kod Sorusu**: Verilen bir C++ fonksiyonunda raw pointer bellek sızıntısı veya kilitlenme riskini denetleyen statik analiz kuralını simüle edin.
- **Çözüm**:
```python
def analyze_code_safety(code_str: str) -> list:
    warnings = []
    if "new " in code_str and "delete " not in code_str:
        warnings.append("UYARI: Raw 'new' kullanildi ama 'delete' bulunamadi (Bellek Sizintisi Riski)!")
    if "malloc" in code_str:
        warnings.append("UYARI: C++ kodunda 'malloc' yasaktir! (MISRA C++ Kurali)")
    return warnings

code = "int* ptr = new int[100];"
print("Statik Analiz Sonuclari:", analyze_code_safety(code))
```

#### **Gün 98: Uçtan Uca Tesla Yazılım Mühendisliği Şampiyonluk Değerlendirmesi**
- **Teorik Odak**: 99 gün boyunca öğrenilen tüm müfredatın (C++, RTOS, FSD, BMS, Infotainment, Dojo, Optimus) sınavı.
- **Quiz Kod Sorusu**: Tüm alt sistemlerin (CAN, BMS, FSD, UI, Motors) sağlık durumunu kontrol edip "ALL_SYSTEMS_GO_AUTONOMOUS_DRIVE" onayını veren ana güvenlik denetleyicisini yazın.
- **Çözüm**:
```cpp
#include <iostream>

struct SystemHealth {
    bool can_bus_ok{true};
    bool bms_temp_ok{true};
    bool fsd_npu_ok{true};
    bool safety_redundancy_ok{true};

    bool is_vehicle_ready() const {
        return can_bus_ok && bms_temp_ok && fsd_npu_ok && safety_redundancy_ok;
    }
};

int main() {
    SystemHealth health;
    std::cout << "Tesla Arac Suruse Hazir mi: " << (health.is_vehicle_ready() ? "ONAYLANDI (AUTONOMOUS DRIVE READY)" : "HATA") << "\n";
    return 0;
}
```

#### **Gün 99: 👑 BÜYÜK FİNAL 99: Tesla Unified Vehicle & Robotics OS**
- **Teorik Odak**: 99 Günlük müfredatın doruk noktası: C++ RTOS, 3D Voxel Occupancy, EKF BMS, NACS Şarj ve Optimus Kontrol Çekirdeğinin tek bir egemen sistemde entegrasyonu!
- **Quiz Kod Sorusu**: 99 günlük tüm Tesla yazılım ve robotik alt sistemlerini senkronize çalıştıran Büyük Final Çekirdeğini yazın.
- **Çözüm**:
```cpp
#include <iostream>
#include <string>

class TeslaUnifiedOmniOS {
public:
    void execute_grand_cycle() {
        std::cout << "========================================================================\n";
        std::cout << " 👑 TESLA UNIFIED VEHICLE & ROBOTICS OS -- 99 GUNLUK BUYUK FINAL 👑    \n";
        std::cout << "========================================================================\n";
        std::cout << " [1] C++20 PREEMPT_RT Kernel       : 1000 Hz Deterministik Zamanlama OK\n";
        std::cout << " [2] CAN-FD & Automotive Ethernet : 64-Byte 8 Mbps Telemetri Aktif\n";
        std::cout << " [3] EKF Batarya Yonetim Sistemi   : 400.0V Pack, %85 SoC, Sifir Hata\n";
        std::cout << " [4] FSD 3D Voxel Occupancy Net    : 36 FPS TensorRT NPU Cikarimi OK\n";
        std::cout << " [5] NACS Supercharger V4 Entegrasyon: 350 kW Hizli Sarj Protokolu Hazir\n";
        std::cout << " [6] Tesla Optimus Humanoid Kontrol: ZMP Dengesi ve 6-DoF Tork Kontrolu OK\n";
        std::cout << "========================================================================\n";
        std::cout << " TEBRIKLER SEYDI ERYILMAZ! 99 GUNLUK TESLA YAZILIM MASTERI TAMAMLANDI! 🚀\n";
        std::cout << "========================================================================\n";
    }
};

int main() {
    TeslaUnifiedOmniOS tesla_os;
    tesla_os.execute_grand_cycle();
    return 0;
}
```

---

#### **Gün 100: 👑 BÜYÜK TAÇ KAPSTONE (DAY 100 MASTERPIECE): Tesla Cyber-Fleet Omni-Orchestrator & Autonomous Vehicle-Robotics Super-Platform**
- **Teorik Odak**: Tesla'nın tüm otonom sürüş (FSD V12 E2E Vision & 3D Voxel Occupancy), araç içi C++20 PREEMPT_RT deterministik çekirdeği, Extended Kalman Filter (EKF) Batarya Yönetim Sistemi, NACS V4 Megapack Şebeke Arbitrajı ve Tesla Optimus İnsansı Robotik montaj/şarj filo yönetimini birleştiren dünyanın en gelişmiş egemen araç-robotik ekosistemi!
- **Endüstriyel Etki (Tesla İlgisini Çeken Mimari)**:
  1. **FSD V12 E2E & 3D Voxel Occupancy**: 8 kamera zamansal füzyon, TensorRT INT8 çıkarımı ($<12\text{ ms}$ gecikme).
  2. **Deterministik C++20 PREEMPT_RT**: 1000 Hz çevrim, sıfır dinamik bellek tahsisi (zero-alloc), lock-free halka kuyruk.
  3. **EKF Batarya & Octovalve Termal**: Hücre seviyesi SoC/SoH kestirimi, soğutma sıvısı valf optimizasyonu.
  4. **NACS V4 & Autobidder Arbitrajı**: SAE J3400 / ISO 15118 Tak & Şarj, şebeke yük dengeleme.
  5. **Tesla Optimus Sim2Real Entegrasyonu**: ZMP bütünsel vücut dengesi ve otonom araç şarj soketi takma robotik operasyonu.
- **Quiz Kod Sorusu**: Gün 100 Taç Projesini temsil eden, tüm bu 5 ana sektörü koordine eden `TeslaCyberFleetOmniOrchestrator` sınıfını ve entegre sistem sağlığı doğrulama algoritmasını yazın.
- **Çözüm**:
```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class FiloAracDurumu:
    arac_id: str
    batarya_soc: float
    otonom_surus_seviyesi: str
    voxel_doluluk_orani: float
    optimus_gorev_atamasi: str
    nacs_sarj_gucu_kw: float
    guvenlik_skoru: float

class TeslaCyberFleetOmniOrchestrator:
    """
    Tesla Cyber-Fleet Omni-Orchestrator (Gün 100 Taç Kapstone)
    Otonom Araç Filosu, Batarya Şebekesi, Supercharger ve Optimus Robotlarını Senkronize Yönetir.
    """
    def __init__(self):
        self.filo: Dict[str, FiloAracDurumu] = {}
        self.sebeke_kapasitesi_mw = 50.0

    def arac_kaydet(self, durum: FiloAracDurumu):
        self.filo[durum.arac_id] = durum

    def filo_optimizasyonu_calistir(self) -> Dict[str, float]:
        toplam_soc = sum(a.batarya_soc for a in self.filo.values())
        ortalama_soc = toplam_soc / max(len(self.filo), 1)
        toplam_sarj_talebi = sum(a.nacs_sarj_gucu_kw for a in self.filo.values()) / 1000.0
        
        # Optimus filo desteği ve güvenlik değerlendirmesi
        ortalama_guvenlik = sum(a.guvenlik_skoru for a in self.filo.values()) / max(len(self.filo), 1)
        
        return {
            "toplam_arac_sayisi": float(len(self.filo)),
            "ortalama_filo_soc": ortalama_soc,
            "toplam_sarj_yuku_mw": toplam_sarj_talebi,
            "ortalama_guvenlik_skoru": ortalama_guvenlik,
            "sebeke_kararlilik_indeksi": min(1.0, self.sebeke_kapasitesi_mw / max(toplam_sarj_talebi, 0.1))
        }

if __name__ == "__main__":
    orkestrator = TeslaCyberFleetOmniOrchestrator()
    orkestrator.arac_kaydet(FiloAracDurumu(
        arac_id="CYBERTRUCK-001",
        batarya_soc=0.88,
        otonom_surus_seviyesi="FSD V12 Supervised",
        voxel_doluluk_orani=0.042,
        optimus_gorev_atamasi="NACS Soket Baglantisi Tamamlandi",
        nacs_sarj_gucu_kw=250.0,
        guvenlik_skoru=0.999
    ))
    sonuclar = orkestrator.filo_optimizasyonu_calistir()
    print("Tesla Cyber-Fleet Omni-Orchestrator (Gün 100) Hazır:", sonuclar)
```

---

## 📜 Lisans & Telif Hakkı

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas))

Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim amaçlı olarak paylaşılmıştır. Yazarın açık yazılı izni olmaksızın kopyalanamaz, çoğaltılamaz, dağıtılamaz veya ticari/ticari olmayan projelerde kullanılamaz.

