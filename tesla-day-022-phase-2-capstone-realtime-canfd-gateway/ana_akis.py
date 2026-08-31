"""
Tesla Gün 22 Ana Akış (Tesla Day 22 Main Pipeline)
===================================================
👑 FAZ 2 BÜYÜK CAPSTONE: Merkezi Araç Telemetri Gateway & UDS Teşhis Motoru
Uçtan Uca Çoklu Ağ Yönlendirme ve Tanı Paneli Üretim Scripti.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import sys
import os
import struct

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_faz2_capstone_gateway import TeslaCentralGateway
from src.tesla_capstone_profilleyici import TeslaCapstoneProfilleyici
from src.tesla_capstone_gorsellestirici import TeslaCapstoneGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("👑 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 22: FAZ 2 BÜYÜK CAPSTONE GATEWAY 👑")
    print("================================================================================")
    print("Stajyer Görevi: CAN-FD, LIN BCM, SOME/IP Ethernet, UDS ISO 14229 & RTOS Pipeline")
    print("--------------------------------------------------------------------------------\n")

    gw = TeslaCentralGateway()

    # 1. CAN-FD Powertrain & Chassis Telemetri Beslemesi
    print(" [1] CAN-FD Ağlarından Güç Aktarımı ve Şasi Telemetrisi İşleniyor...")
    payload_pwr = struct.pack(">HHh2x", 4000, 1500, 345)  # 400.0V, 150.0A, 34.5°C
    payload_chs = struct.pack(">HH4x", 2400, 1850)        # 120.0 km/h, +5.0 deg
    gw.decode_canfd_powertrain(0x301, payload_pwr)
    gw.decode_canfd_chassis(0x12F, payload_chs)

    print(f"     -> Batarya Paketi Gerilimi : {gw.state.pack_voltage_v:.1f} V")
    print(f"     -> Çekilen Anlık Akım     : {gw.state.pack_current_a:.1f} A")
    print(f"     -> Hesaplanan Anlık Güç   : {gw.state.power_kw:.1f} kW")
    print(f"     -> Araç Hızı              : {gw.state.vehicle_speed_kmh:.1f} km/h")
    print(f"     -> Direksiyon Açısı       : {gw.state.steering_angle_deg:+.1f} Derece")

    # 2. LIN BCM ve SOME/IP Otopilot RPC
    print("\n [2] LIN BCM Gövde Durumu ve SOME/IP Otopilot Durum Yönetimi...")
    gw.decode_lin_bcm(0x24, bytes([0x01, 0x00]))
    gw.process_someip_rpc(0x1234, 0x0001, bytes([0x01]))

    print(f"     -> LIN BCM Kapı Durumu    : {'🔒 KİLİTLİ' if gw.state.door_lock_status else '🔓 AÇIK'}")
    print(f"     -> SOME/IP FSD Durumu     : {'🚀 OTOPİLOT AKTİF (ENGAGED)' if gw.state.fsd_engaged else 'MANUEL'}")

    # 3. UDS 0x22 Teşhis Sorgulama
    print("\n [3] UDS (ISO 14229) DoIP Üzerinden Telemetri ve VIN Sorgulaması...")
    resp_vin = gw.handle_uds_request(bytes([0x22, 0xF1, 0x90]))
    resp_pwr = gw.handle_uds_request(bytes([0x22, 0x01, 0x04]))

    vin_str = resp_vin[3:].decode('utf-8') if len(resp_vin) > 3 else "N/A"
    pwr_f = struct.unpack(">f", resp_pwr[3:])[0] if len(resp_pwr) >= 7 else 0.0
    print(f"     -> UDS DID 0xF190 (VIN)   : {vin_str}")
    print(f"     -> UDS DID 0x0104 (Güç)   : {pwr_f:.1f} kW")

    # 4. Performans ve Yönlendirme Benchmark'ı
    print("\n [4] Çoklu Ağ Yönlendirme ve Gateway Verim Analizi...")
    profilleyici = TeslaCapstoneProfilleyici(ornek_sayisi=5000)
    metrikler = profilleyici.benchmark_gateway_pipeline()

    print(f"     -> Ortalama Gateway Gecikmesi : {metrikler['gateway_ortalama_us']:.3f} µs (P99: {metrikler['gateway_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Gateway Kapasitesi: {metrikler['saniyelik_gateway_hacmi']:,} Frame/sn")
    print(f"     -> İşlenen Toplam Çerçeve     : {metrikler['islenen_toplam_frame']:,}")

    # 5. Tanı Paneli Görselleştirme
    print("\n [5] 6 Panelli Faz 2 Büyük Capstone Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaCapstoneGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_faz2_capstone_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 👑 TEBRİKLER! FAZ 2 (GÜN 12 - 22) BÜYÜK CAPSTONE BAŞARIYLA TAMAMLANDI! 👑")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
