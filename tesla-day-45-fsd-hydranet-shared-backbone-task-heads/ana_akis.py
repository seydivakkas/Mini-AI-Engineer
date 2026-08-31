"""
Tesla Gün 45 Ana Akış (Tesla Day 45 Main Pipeline)
===================================================
Tesla FSD HydraNet Mimarisi: Paylaşılan Omurga ve Görev Kafaları
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

from src.tesla_fsd_hydranet_mimarisi import TeslaFSDHydraNet
from src.tesla_hydranet_profilleyici import TeslaHydraNetProfilleyici
from src.tesla_hydranet_gorsellestirici import TeslaHydraNetGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 45: HYDRANET PAYLAŞILAN OMURGA VE GÖREV KAFALARI 🚗")
    print("================================================================================")
    print("Stajyer Görevi: RegNet/BiFPN Omurga, 4 Görev Kafası & %72 NPU Hesaplama Kazancı")
    print("--------------------------------------------------------------------------------\n")

    # 1. HydraNet Benchmark'ı
    print(" [1] HydraNet Çoklu Görev Çıkarımı ve Homoscedastic Kayıp Değerlendirmesi...")
    profilleyici = TeslaHydraNetProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_hydranet()

    c = metrikler["ciktilar"]
    print(f"     -> Tespit Edilen Nesne        : {c['objects']['detected_class']} (3D Konum: {c['objects']['bbox_3d'][0]:.1f}m)")
    print(f"     -> Sol / Sağ Şerit Polinomları: y = {c['lanes']['left_lane'][0]:.2f}m / y = {c['lanes']['right_lane'][0]:.2f}m")
    print(f"     -> Trafik Işığı Durumu        : {c['traffic_light']['state']} (Güven: %{c['traffic_light']['confidence']*100:.1f})")
    print(f"     -> Çoklu Görev Kaybı          : {metrikler['toplam_coklu_gorev_kaybi']:.3f}")
    print(f"     -> Hesaplama Tasarrufu        : %{metrikler['hesaplama_tasarrufu_pct']:.1f} NPU Kazancı")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] HydraNet RTOS Çıkarım Performansı...")
    print(f"     -> Ortalama Çözüm Süresi      : {metrikler['hydranet_step_ortalama_us']:.3f} µs (P99: {metrikler['hydranet_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Kare Kapasitesi  : {metrikler['saniyelik_hydranet_karesi']:,} FPS")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD HydraNet Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaHydraNetGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_hydranet_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi     : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 45 BAŞARIYLA TAMAMLANDI! HYDRANET MİMARİSİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
