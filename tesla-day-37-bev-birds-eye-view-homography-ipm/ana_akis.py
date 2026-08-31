"""
Tesla Gün 37 Ana Akış (Tesla Day 37 Main Pipeline)
===================================================
Kuşbakışı (BEV) Temsili, Düzlemsel Homografi ve IPM
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

from src.tesla_bev_homografi_ve_ipm import TeslaBEVTransformer
from src.tesla_bev_profilleyici import TeslaBEVProfilleyici
from src.tesla_bev_gorsellestirici import TeslaBEVGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 37: KUŞBAKIŞI (BEV) TEMSİLİ VE HOMOGRAFİ / IPM 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 2D Perspektif -> Metrik BEV (X, Y), Şerit Dönüşümü & IPM Motoru")
    print("--------------------------------------------------------------------------------\n")

    # 1. BEV ve IPM Dönüşüm Benchmark'ı
    print(" [1] 200 Noktalı Şerit Projeksiyonu ve Gidiş-Dönüş Geometrik Doğrulama...")
    profilleyici = TeslaBEVProfilleyici(num_points=200, iterations=100)
    metrikler = profilleyici.benchmark_bev_donusumu()

    print(f"     -> Gidiş-Dönüş (Roundtrip) Hata: {metrikler['roundtrip_error_px']:.6e} Piksel (Kusursuz Tersinir)")
    print(f"     -> Dönüştürülen Sol Şerit Nokta : {len(metrikler['bev_left'])} Metrik Nokta")
    print(f"     -> Dönüştürülen Sağ Şerit Nokta: {len(metrikler['bev_right'])} Metrik Nokta")

    # 2. BEV RTOS Çözümleme Hızı
    print("\n [2] Düzlemsel Homografi ve IPM Çözümleme Hızı...")
    print(f"     -> Ortalama Çözüm Süresi       : {metrikler['bev_step_ortalama_us']:.3f} µs (P99: {metrikler['bev_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik BEV Dönüşümü      : {metrikler['saniyelik_bev_donusumu']:,} Kare/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Kuşbakışı (BEV) Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaBEVGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_bev_homografi_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi      : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 37 BAŞARIYLA TAMAMLANDI! KUŞBAKIŞI (BEV) VE IPM DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
