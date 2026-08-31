"""
Tesla Gün 54 Ana Akış (Tesla Day 54 Main Pipeline)
===================================================
Tesla Veri Fabrikası: Otomatik 3D Yörünge ve Bounding Box Etiketleme
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

from src.tesla_otomatik_etiketleme_ve_sentetik_veri import TeslaAutoLabelingPipeline
from src.tesla_otomatik_etiketleme_profilleyici import TeslaOtomatikEtiketlemeProfilleyici
from src.tesla_otomatik_etiketleme_gorsellestirici import TeslaOtomatikEtiketlemeGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 54: OTO-ETİKETLEME VE SENTETİK VERİ FABRİKASI 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Çift Yönlü Düzeltme, Çoklu Sürüş Eşleme & Sentetik Varyasyonlar")
    print("--------------------------------------------------------------------------------\n")

    # 1. Otomatik Etiketleme Benchmark'ı
    print(" [1] Dojo Çift Yönlü Yörünge Düzeltme ve 3D IoU Doğrulaması Çözümleniyor...")
    profilleyici = TeslaOtomatikEtiketlemeProfilleyici(trajectory_len=100, iterations=100)
    metrikler = profilleyici.benchmark_auto_labeling_pipeline()

    print(f"     -> 3D Kutu IoU Kalitesi     : {metrikler['3d_bbox_iou']:.4f} (> 0.95 Standart)")
    print(f"     -> Çevrimdışı Gürültü Azaltma: %{metrikler['noise_reduction_pct']:.1f} İyileşme")
    print(f"     -> Çoklu Sürüş Eşleşme RMSE : {metrikler['alignment_rmse_cm']:.1f} cm")
    print(f"     -> Birleştirilen Nokta Sayısı: {metrikler['total_points']} 3D Nokta")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Dojo Otomatik Etiketleme RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['autolabel_step_ortalama_us']:.3f} µs (P99: {metrikler['autolabel_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Kare Kapasitesi: {metrikler['saniyelik_klip_karesi']:,} Kare/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Otomatik Etiketleme Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaOtomatikEtiketlemeGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_auto_labeling_pipeline_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 54 BAŞARIYLA TAMAMLANDI! OTO-ETİKETLEME FABRİKASI DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
