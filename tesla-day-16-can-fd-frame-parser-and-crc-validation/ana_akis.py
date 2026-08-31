"""
Tesla Gun 16 Ana Akis (Tesla Day 16 Main Pipeline)
===================================================
CAN-FD Frame Parser & CRC-17 / CRC-21 Dogrulama
Uctan Uca Calistirma ve Teshis Paneli Uretim Scripti.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_can_fd_parser import (
    TeslaCANFDFrameParser,
    hesapla_crc17,
    hesapla_crc21
)
from src.tesla_crc_profilleyici import TeslaCRCProfilleyici
from src.tesla_crc_gorsellestirici import TeslaCRCGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 16: CAN-FD FRAME PARSER & CRC 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: CRC-17 (<=16B), CRC-21 (>16B) Dogrulama & Bit-Flip Hata Ayiklama")
    print("--------------------------------------------------------------------------------\n")

    parser = TeslaCANFDFrameParser()

    # 1. 16-Byte Payload ve CRC-17 Doğrulaması
    print(" [1] 16-Byte Payload İçeren Batarya Sıcaklık Çerçevesi Ayrıştırılıyor (CRC-17)...")
    veri_16b = b'BATTERY_TEMP_OK!'
    paket16 = parser.cerceve_serilestir(can_id=0x120, veri=veri_16b)
    sonuc16 = parser.cerceve_ayristir(paket16)

    print(f"     -> CAN ID: {hex(sonuc16.can_id)} | DLC: {sonuc16.dlc} | Algoritma: {sonuc16.crc_turu}")
    print(f"     -> Alınan CRC: {hex(sonuc16.alinan_crc)} | Hesaplanan CRC: {hex(sonuc16.hesaplanan_crc)}")
    print(f"     -> Doğrulama Durumu: {'✅ GEÇERLİ' if sonuc16.gecerli_mi else '❌ GEÇERSİZ'} ({sonuc16.hata_kodu})")

    # 2. 64-Byte Payload ve CRC-21 Doğrulaması
    print("\n [2] 64-Byte Payload İçeren Otopilot Vektör Çerçevesi Ayrıştırılıyor (CRC-21)...")
    veri_64b = b'AUTOPILOT_TRAJECTORY_VECTOR_DATA_64_BYTES_STREAM_TESLA_HW4_FSD_CORE'
    paket64 = parser.cerceve_serilestir(can_id=0x080, veri=veri_64b)
    sonuc64 = parser.cerceve_ayristir(paket64)

    print(f"     -> CAN ID: {hex(sonuc64.can_id)} | DLC: {sonuc64.dlc} | Algoritma: {sonuc64.crc_turu}")
    print(f"     -> Alınan CRC: {hex(sonuc64.alinan_crc)} | Hesaplanan CRC: {hex(sonuc64.hesaplanan_crc)}")
    print(f"     -> Doğrulama Durumu: {'✅ GEÇERLİ' if sonuc64.gecerli_mi else '❌ GEÇERSİZ'} ({sonuc64.hata_kodu})")

    # 3. Bit-Flip Bozuk Veri Enjeksiyonu Testi
    print("\n [3] Veri Yolunda Elektriksel Gürültü / Bit-Flip Hatası Simülasyonu...")
    paket_bozuk = bytearray(paket64)
    paket_bozuk[12] ^= 0x01  # 12. baytın 1 bitini tersine çevir
    sonuc_bozuk = parser.cerceve_ayristir(bytes(paket_bozuk))

    print(f"     -> Alınan CRC: {hex(sonuc_bozuk.alinan_crc)} | Hesaplanan CRC: {hex(sonuc_bozuk.hesaplanan_crc)}")
    print(f"     -> Güvenlik Yanıtı: {'✅ GÜVENLİ ŞEKİLDE REDDEDİLDİ' if not sonuc_bozuk.gecerli_mi else '❌ GÜVENLİK AÇIĞI'}")
    print(f"     -> Hata Kodu: {sonuc_bozuk.hata_kodu}")

    # 4. Performans ve Hız Benchmark'ı
    print("\n [4] CRC Polinom Hesaplama ve Çerçeve Ayrıştırma Benchmark'ı...")
    profilleyici = TeslaCRCProfilleyici(ornek_sayisi=5000)
    metrikler = profilleyici.benchmark_crc_ve_ayristirma()

    print(f"     -> CRC-17 Hesaplama Gecikmesi (16B): {metrikler['crc17_ortalama_us']:.3f} µs")
    print(f"     -> CRC-21 Hesaplama Gecikmesi (64B): {metrikler['crc21_ortalama_us']:.3f} µs (P99: {metrikler['crc21_p99_us']:.3f} µs)")
    print(f"     -> Saniyedeki Ayrıştırma Kapasitesi: {metrikler['saniyede_islenen_cerceve']:,} Çerçeve/sn")

    # 5. Tanı Paneli Görselleştirme
    print("\n [5] 6 Panelli Tesla CRC Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaCRCGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_crc_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 16 BAŞARIYLA TAMAMLANDI! CAN-FD PARSER VE CRC DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
