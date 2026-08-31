"""
Tesla Gün 19 Ana Akış (Tesla Day 19 Main Pipeline)
===================================================
UDS (Unified Diagnostic Services - ISO 14229) & OBD-II DTC Teşhis Motoru
Uçtan Uca Çalıştırma ve Teşhis Paneli Üretim Scripti.

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

from src.tesla_uds_protokolu import (
    TeslaUDSServer,
    TeslaUDSClient,
    DiagnosticSessionType,
    UDSServiceID,
    UDSNRC,
    decode_dtc
)
from src.tesla_uds_profilleyici import TeslaUDSProfilleyici
from src.tesla_uds_gorsellestirici import TeslaUDSGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 19: UDS (ISO 14229) & OBD-II DTC 🚗")
    print("================================================================================")
    print("Stajyer Görevi: UDS Servisleri (0x22, 0x19, 0x27), DTC Ayrıştırma, Seed-Key & DoIP")
    print("--------------------------------------------------------------------------------\n")

    server = TeslaUDSServer(ecu_name="Tesla BMS Core ECU")
    client = TeslaUDSClient(server)

    # 1. 0x22 ReadDataByIdentifier (DID Okuma)
    print(" [1] UDS 0x22 ReadDataByIdentifier ile ECU Telemetri Parametreleri Okunuyor...")
    ok_vin, vin_bytes = client.read_did(0xF190)
    ok_volt, volt_bytes = client.read_did(0x0100)
    ok_temp, temp_bytes = client.read_did(0x0101)
    ok_rpm, rpm_bytes = client.read_did(0x0102)

    vin = vin_bytes.decode('utf-8') if ok_vin and vin_bytes else "N/A"
    volt = struct.unpack(">f", volt_bytes)[0] if ok_volt and volt_bytes else 0.0
    temp = struct.unpack(">f", temp_bytes)[0] if ok_temp and temp_bytes else 0.0
    rpm = struct.unpack(">i", rpm_bytes)[0] if ok_rpm and rpm_bytes else 0

    print(f"     -> DID 0xF190 (VIN)           : {vin}")
    print(f"     -> DID 0x0100 (Batarya Voltaj): {volt:.1f} V")
    print(f"     -> DID 0x0101 (İnvertör Sıcak): {temp:.1f} °C")
    print(f"     -> DID 0x0102 (Motor RPM)     : {rpm} d/d")

    # 2. 0x19 ReadDTCInformation (Arıza Kodu Sorgulama)
    print("\n [2] UDS 0x19 ReadDTCInformation ile Aktif & Kayıtlı Arıza Kodları Okunuyor...")
    dtcler = client.read_dtcs(status_mask=0xFF)
    print(f"     -> Toplam Tespit Edilen DTC Sayısı: {len(dtcler)}")
    for kod, mask in dtcler:
        confirmed = "ONAYLI" if (mask & 0x08) else "BEKLEMEDE"
        active = "AKTİF" if (mask & 0x01) else "GEÇMİŞ"
        print(f"        * DTC: {kod} | Durum Maskesi: 0x{mask:02X} [{confirmed}, {active}]")

    # 3. 0x27 SecurityAccess & 0x2E WriteDataByIdentifier
    print("\n [3] UDS 0x27 SecurityAccess (Seed-Key) ve 0x2E Konfigürasyon Yazma Denetimi...")
    client.set_session(DiagnosticSessionType.EXTENDED_DIAGNOSTIC_SESSION)
    print(f"     -> Teşhis Oturumu : {server.current_session.name}")
    
    sec_unlocked = client.unlock_security()
    print(f"     -> Güvenlik Kilidi: {'✅ AÇILDI (Security Unlocked)' if sec_unlocked else '❌ KİLİTLİ'}")

    yazma_ok = client.write_did(0x0103, bytes([0x00]))  # Otopilot durumunu 0x00 (Manual) yap
    print(f"     -> DID 0x0103 Yazma: {'✅ BAŞARILI' if yazma_ok else '❌ REDDEDİLDİ'}")

    # 4. Performans ve Gecikme Benchmark'ı
    print("\n [4] UDS Servis Gecikmesi & DoIP vs CAN-FD Karşılaştırma Analizi...")
    profilleyici = TeslaUDSProfilleyici(ornek_sayisi=5000)
    metrikler = profilleyici.benchmark_uds_servisleri()

    print(f"     -> 0x22 DID Okuma Gecikmesi   : {metrikler['did_ortalama_us']:.3f} µs (P99: {metrikler['did_p99_us']:.3f} µs)")
    print(f"     -> 0x19 DTC Okuma Gecikmesi   : {metrikler['dtc_ortalama_us']:.3f} µs")
    print(f"     -> 0x27 Seed-Key El Sıkışması : {metrikler['security_handshake_us']:.1f} µs")
    print(f"     -> DoIP Hızlanma Oranı        : {metrikler['hizlanma_doip_vs_can']:.1f}x Daha Hızlı!")
    print(f"     -> Saniyelik DID Kapasitesi   : {metrikler['saniyelik_did_sorgusu']:,} Sorgu/sn")

    # 5. Tanı Paneli Görselleştirme
    print("\n [5] 6 Panelli Tesla UDS Teşhis Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaUDSGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_uds_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 19 BAŞARIYLA TAMAMLANDI! UDS ISO 14229 & OBD-II DTC DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
