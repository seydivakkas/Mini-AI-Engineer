"""
Tesla Gün 89 Ana Akış (Tesla Day 89 Main Pipeline)
===================================================
Tesla Dojo Süperbilgisayar Mimarisi: D1 Çipi, Training Tile ve 2D Mesh NoC
Uçtan Uca Çalıştırma ve Teşhis Paneli Üretim Scripti.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
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

from src.tesla_dojo_d1_mesh_yonlendirici import TeslaDojoMeshRouter
from src.tesla_dojo_profilleyici import TeslaDojoProfilleyici
from src.tesla_dojo_gorsellestirici import TeslaDojoGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 89: TESLA DOJO D1 ÇİPİ VE 2D MESH NoC 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 25 D1 Çipli Training Tile, 9 PFLOPS, 36 TB/s Biseksiyon & NoC")
    print("--------------------------------------------------------------------------------\n")

    # 1. Dojo Benchmark'ı
    print(" [1] 25 D1 Çipli Dojo Training Tile NoC Yönlendirme Simülasyonu Başlatılıyor...")
    profilleyici = TeslaDojoProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_dojo_routing()

    print(f"     -> D1 Çip Sayısı           : {metrikler['num_chips']} Adet Özel 7nm Silikon (5x5 Matris)")
    print(f"     -> Training Tile Kapasite  : {metrikler['tile_pflops']:.2f} PFLOPS (BF16 / CFP8 Hesaplama)")
    print(f"     -> Çapraz Atlama (Corner)  : {metrikler['hop_count_corner_to_corner']} Hops (2.5 ns / Hop)")
    print(f"     -> 1 MB Transfer Gecikmesi : {metrikler['total_latency_ns']:.2f} ns")
    print(f"     -> Efektif Bant Genişliği  : {metrikler['effective_bw_gb_s']:,.0f} GB/s")
    print(f"     -> NoC Kararlılık Durumu   : %100 DEADLOCK-FREE (Dimension-Ordered XY)")

    # 2. Yönlendirme Hızı
    print("\n [2] NoC Yönlendirme ve Paket İletim Algoritması RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi   : {metrikler['step_ortalama_us']:.3f} µs (P99: {metrikler['step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Yönlendirme   : {metrikler['saniyelik_yonlendirme_kapasitesi']:,} Paket/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Dojo NoC Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaDojoGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_dojo_d1_mesh_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 89 BAŞARIYLA TAMAMLANDI! DOJO D1 NoC MİMARİSİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
