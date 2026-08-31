"""
Tesla Gun 18 Ana Akis (Tesla Day 18 Main Pipeline)
===================================================
Automotive Ethernet & SOME/IP (Scalable service-Oriented MiddlewarE over IP)
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

from src.tesla_someip_protokolu import (
    TeslaSOMEIPHeader,
    TeslaSOMEIPPaket,
    TeslaSOMEIPServer,
    TeslaSOMEIPClient,
    SOMEIPMessageType,
    SOMEIPReturnCode
)
from src.tesla_someip_profilleyici import TeslaSOMEIPProfilleyici
from src.tesla_someip_gorsellestirici import TeslaSOMEIPGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 18: AUTOMOTIVE ETHERNET & SOME/IP 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: 16-Byte Header, Service Discovery, RPC Request/Response & SOA")
    print("--------------------------------------------------------------------------------\n")

    server = TeslaSOMEIPServer(service_id=0x1234)
    client = TeslaSOMEIPClient(client_id=0x0042)

    # 1. Başarılı SOME/IP RPC Çağrısı
    print(" [1] Otopilot Hedef Hız Güncelleme RPC Çağrısı Başlatılıyor (Service 0x1234, Method 0x0001)...")
    istenen_hiz = 120.0  # km/h
    basarili, onaylanan_hiz = client.rpc_hedef_hiz_cagir(server, istenen_hiz)
    print(f"     -> İstemci İstek Gönderdi  : Hedef Hız = {istenen_hiz:.1f} km/h")
    print(f"     -> Sunucu Yanıtı Aldı     : Durum = {'✅ BAŞARILI (E_OK)' if basarili else '❌ HATA'}")
    print(f"     -> Sunucuda Aktif Hız     : {onaylanan_hiz:.1f} km/h")

    # 2. Hatalı Servis Çağrısı (Error Handling)
    print("\n [2] Geçersiz Servis Çağrısı ve SOME/IP Hata Yönetimi Denetimi...")
    sahte_baslik = TeslaSOMEIPHeader(
        service_id=0x9999, method_id=0x0001, uzunluk=8,
        client_id=0x0042, session_id=99,
        message_type=SOMEIPMessageType.REQUEST, return_code=SOMEIPReturnCode.E_OK
    )
    hata_yaniti = server.istek_isle(TeslaSOMEIPPaket(baslik=sahte_baslik, payload=b''))
    print(f"     -> Gönderilen Servis ID    : 0x9999 (Kayıtsız)")
    print(f"     -> Sunucu Dönüş Kodu       : {hata_yaniti.baslik.return_code.name} (0x{int(hata_yaniti.baslik.return_code):02x})")
    print(f"     -> Mesaj Tipi              : {hata_yaniti.baslik.message_type.name}")

    # 3. Performans ve Hız Benchmark'ı
    print("\n [3] SOME/IP RPC vs CAN-FD Gecikme ve Bant Genişliği Benchmark Analizi...")
    profilleyici = TeslaSOMEIPProfilleyici(ornek_sayisi=5000)
    metrikler = profilleyici.benchmark_someip_rpc()

    print(f"     -> SOME/IP Ethernet Gecikmesi: {metrikler['someip_ortalama_us']:.3f} µs (P99: {metrikler['someip_p99_us']:.3f} µs)")
    print(f"     -> CAN-FD Eşdeğer RPC Süresi : {metrikler['can_fd_rpc_us']:.1f} µs")
    print(f"     -> Hızlanma Çarpanı          : {metrikler['hizlanma_carpani']:.1f}x Daha Hızlı!")
    print(f"     -> Saniyelik RPC Kapasitesi  : {metrikler['saniyelik_rpc_kapasitesi']:,} Çağrı/sn")

    # 4. Tanı Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla SOME/IP Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaSOMEIPGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_someip_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 18 BAŞARIYLA TAMAMLANDI! SOME/IP & AUTOMOTIVE ETHERNET DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
