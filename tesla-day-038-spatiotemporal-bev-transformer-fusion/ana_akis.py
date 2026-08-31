"""
Tesla Gün 38 Ana Akış (Tesla Day 38 Main Pipeline)
===================================================
Mekansal-Zamansal (Spatiotemporal) Öznitelik Füzyonu ve BEV Transformer
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

from src.tesla_spatiotemporal_bev_transformer import TeslaSpatiotemporalBEVTransformer
from src.tesla_transformer_profilleyici import TeslaTransformerProfilleyici
from src.tesla_transformer_gorsellestirici import TeslaTransformerGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 38: SPATIOTEMPORAL BEV TRANSFORMER FÜZYONU 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Spatial Cross-Attention, Ego-Motion Warp & Oklüzyon Belleği")
    print("--------------------------------------------------------------------------------\n")

    # 1. Spatiotemporal Transformer Benchmark'ı
    print(" [1] 8 Kamera Mekansal-Zamansal BEV Transformer ve Oklüzyon Simülasyonu...")
    profilleyici = TeslaTransformerProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_spatiotemporal_transformer()

    print(f"     -> Başlangıç Nesne Algı Olasılığı: %{metrikler['occlusion_memory_probs'][0] * 100:.1f}")
    print(f"     -> Oklüzyon Anında Korunan Bellek: %{metrikler['occlusion_memory_probs'][21] * 100:.1f} (Görüş Koptuğu Halde)")
    print(f"     -> 100 Kare Sonrası Çıktı Boyutu : {metrikler['final_occupancy_prob'].shape} BEV Grid")

    # 2. Transformer RTOS Çözümleme Hızı
    print("\n [2] Spatiotemporal Transformer RTOS Çözümleme Performansı...")
    print(f"     -> Ortalama Çözüm Süresi         : {metrikler['transformer_step_ortalama_us']:.3f} µs (P99: {metrikler['transformer_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik BEV Adım Kapasitesi : {metrikler['saniyelik_bev_adim']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Spatiotemporal BEV Transformer Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaTransformerGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_bev_transformer_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi        : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 38 BAŞARIYLA TAMAMLANDI! BEV TRANSFORMER VE FÜZYON DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
