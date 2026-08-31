"""
Tesla Gün 55 Ana Akış (Tesla Day 55 Main Pipeline)
===================================================
FAZ 5 BÜYÜK CAPSTONE: Tesla FSD Yapay Zeka Çıkarım Motoru
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

from src.tesla_fsd_ai_cikarim_motoru_capstone import TeslaFSDAIInferenceEngineCapstone
from src.tesla_capstone_profilleyici import TeslaCapstoneProfilleyici
from src.tesla_capstone_gorsellestirici import TeslaCapstoneGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 55: FAZ 5 BÜYÜK CAPSTONE: FSD AI ÇIKARIM MOTORU 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Occupancy, DAG Grafı, ViT Işık/Levha, 5s Yörünge, INT8 & Gölge")
    print("--------------------------------------------------------------------------------\n")

    # 1. Faz 5 Büyük Capstone Benchmark'ı
    print(" [1] 10 Derin Öğrenme Modeli Tek Bir FSD AI Çıkarım Motorunda Birleştiriliyor...")
    profilleyici = TeslaCapstoneProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_capstone_engine()
    c = metrikler["ciktilar"]

    print(f"     -> 3D Voksel Doluluk        : {c['occupied_voxels']:,} Hücre (Doluluk: %{c['occupancy_ratio_pct']:.1f})")
    print(f"     -> Öncü Araç 3D Voxel Flow  : Vx = {c['lead_voxel_flow_vx']:.1f} m/s")
    print(f"     -> VectorLaneNet Yol Grafı  : 10m Eğrilik: {c['lane_curvature_10m']:.6f} 1/m, DAG Geçişleri: {c['legal_dag_lanes']}")
    print(f"     -> Vision Transformer (ViT) : {c['traffic_light']} (%{c['tl_confidence']*100:.1f}), {c['tl_countdown_sec']:.1f}s Geri Sayım")
    print(f"     -> Trafik Levhası OCR       : {c['traffic_sign']} (%{c['sign_confidence']*100:.1f})")
    print(f"     -> 5s Dinamik Yörünge & TTC : TTC = {c['ttc_seconds']:.1f} Saniye")
    print(f"     -> INT8 NPU SRAM Tasarrufu  : %{c['int8_memory_saving_pct']:.1f} Tasarruf")
    print(f"     -> Bilgi Damıtma Doğruluğu  : %{c['distillation_accuracy_retention']:.1f} Korundu")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Faz 5 Büyük Capstone RTOS Çıkarım Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['capstone_step_ortalama_us']:.3f} µs (P99: {metrikler['capstone_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik FSD AI Hacmi   : {metrikler['saniyelik_fsd_ai_karesi']:,} Kare/sn (10,000+ FPS Hedefi)")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Faz 5 Büyük Capstone Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaCapstoneGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_phase_5_capstone_fsd_ai_engine_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🏆 FAZ 5 BÜYÜK CAPSTONE %100 BAŞARIYLA TAMAMLANDI! FSD AI MOTORU HAZIR! 🏆")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
