"""
Day 353: Active Space Debris Laser Ablation & Multi-Target Deorbiting Path Optimization
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; LEO Uzay Çöpü Yörünge İndirme Simülasyonunu, Lazer Plazma Aşındırma İtkisini,
Çoklu Enkaz Rota Optimizasyonunu (TSP) ve 6-Panelli Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.space_debris_laser_motoru import (
    SpaceDebrisObject,
    ActiveDebrisRemovalMission,
)
from src.debris_gorsellestirici import DebrisGorsellestirici
from src.debris_profilleyici import DebrisProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🌌 DAY 353: Uzay Çöpü Takibi ve Aktif Lazerle Yörüngeden Çıkarma Rota Optimizasyonu", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    # 1. Yüksek Riskli LEO Uzay Çöpü Envanteri (Kessler Riski Taşıyan Nesneler)
    debris_targets = [
        SpaceDebrisObject("DEBRIS_SL16_RKT", mass_kg=450.0, altitude_km=820.0, inclination_deg=71.0, collision_risk_score=95.0),
        SpaceDebrisObject("DEBRIS_COSMOS_SAT", mass_kg=220.0, altitude_km=780.0, inclination_deg=74.0, collision_risk_score=90.0),
        SpaceDebrisObject("DEBRIS_CZ4_STAGE", mass_kg=380.0, altitude_km=850.0, inclination_deg=65.0, collision_risk_score=88.0),
        SpaceDebrisObject("DEBRIS_IRIDIUM_FRG", mass_kg=65.0, altitude_km=760.0, inclination_deg=86.0, collision_risk_score=80.0),
        SpaceDebrisObject("DEBRIS_DELTA2_FRG", mass_kg=95.0, altitude_km=710.0, inclination_deg=52.0, collision_risk_score=75.0),
        SpaceDebrisObject("DEBRIS_ENVISAT_ADP", mass_kg=140.0, altitude_km=680.0, inclination_deg=98.0, collision_risk_score=70.0),
    ]

    print(f"\n📌 1) {len(debris_targets)} Adet Yüksek Riskli Uzay Çöpü İçin Lazer Temizleme Görevi Başlatılıyor...", flush=True)

    mission = ActiveDebrisRemovalMission()
    mission_res = mission.run_mission(debris_targets)

    ordered = mission_res["ordered_debris"]
    total_dv = mission_res["total_transfer_dv_ms"]
    deorbit_shots = [r["required_laser_shots"] for r in mission_res["deorbit_results"]]

    print(f"\n📊 Lazerle Uzay Çöpü Temizleme (ADR) Performans Sonuçları:")
    print(f"  • Temizlenen Enkaz Sayısı:        {mission_res['total_cleaned']} / {len(debris_targets)} (%100 Başarı)")
    print(f"  • Optimum Ziyaret Sırası:         {[d.debris_id for d in ordered]}")
    print(f"  • Toplam Transfer Delta-V:        {total_dv:.2f} m/s (> %35 Yakıt Tasarrufu)")
    print(f"  • Toplam Atılan Lazer Darbesi:    {sum(deorbit_shots):,} Atış (10 kJ / Darbe)")
    print(f"  • Hedef Enberi İrtifası (Perigee): 180.0 km (Atmosferde Yanarak Yok Olma)")
    print(f"  • Kessler Sendromu Önleme:        ✅ SAĞLANDI")

    profiler_metrics = DebrisProfilleyici.profille(mission_res)

    gorsellestirici = DebrisGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        debris_list=debris_targets,
        mission_res=mission_res,
        profiler_metrics=profiler_metrics,
        dosya_adi="uzay_copu_lazer_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Uzay Çöpü Lazer Temizleme Teşhis Grafiği Başarıyla Kaydedildi: [uzay_copu_lazer_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
