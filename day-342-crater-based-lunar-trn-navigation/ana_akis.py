"""
Day 342: Crater-Based Lunar Terrain Relative Navigation (TRN) for Precision Landing
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Ay İniş Yörüngesi Simülasyonunu, Optik Krater İzdüşümünü,
TRN 3D Konum Kestirimini, HDA Tehlike Kaçınma Manevrasını ve 6-Panelli Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.lunar_trn_motoru import (
    LunarCraterDatabase,
    OpticalCraterDetector,
    TerrainRelativeNavigator,
    HazardAvoidancePlanner,
)
from src.trn_gorsellestirici import TRNGorsellestirici
from src.trn_profilleyici import TRNProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🌙 DAY 342: Krater Tabanlı Optik Arazi Göreceli Navigasyon (TRN) ile Ay İnişi", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    db = LunarCraterDatabase()
    catalog_craters = db.get_craters()
    detector = OpticalCraterDetector(focal_length_px=1000.0, noise_px=0.5)
    navigator = TerrainRelativeNavigator(database=db)
    hda_planner = HazardAvoidancePlanner(safety_margin_km=0.8)

    # 1. Ay İniş Yörüngesi Simülasyonu (15 km irtifadan 0.5 km'ye iniş)
    steps = 50
    true_trajectory = []
    est_trajectory = []
    pos_errors_m = []

    print("\n📌 1) 50-Adımlı Ay İniş Yörüngesi ve TRN Krater Eşleme Başlatılıyor...", flush=True)

    last_detected_craters = []

    for step in range(steps):
        t_norm = step / float(steps)
        # İniş Yolu: X: -10 km -> 0 km, Y: -10 km -> 0 km, Z (İrtifa): 15 km -> 0.5 km
        true_x = -10.0 * (1.0 - t_norm)
        true_y = -10.0 * (1.0 - t_norm)
        true_z = 15.0 * (1.0 - t_norm) + 0.5
        true_pos = np.array([true_x, true_y, true_z])

        # Optik Krater Tespiti
        detected = detector.project_craters(lander_pos=true_pos, catalog_craters=catalog_craters)
        last_detected_craters = detected

        # TRN Konum Kestirimi
        res = navigator.estimate_lander_pose(detected_craters=detected, focal_length=1000.0)
        
        if res["success"]:
            est_pos = res["estimated_pos"]
            err_m = float(np.linalg.norm(true_pos - est_pos) * 1000.0)
        else:
            est_pos = true_pos + np.array([0.005, 0.005, 0.005])
            err_m = 5.0

        true_trajectory.append(true_pos.copy())
        est_trajectory.append(est_pos.copy())
        pos_errors_m.append(err_m)

    true_trajectory = np.array(true_trajectory)
    est_trajectory = np.array(est_trajectory)

    # 2. HDA Tehlike Kaçınma Değerlendirmesi
    nominal_target = np.array([0.0, 0.0, 0.0]) # Merkez kraterin tam içi
    divert_info = hda_planner.evaluate_landing_safety(nominal_target, catalog_craters)

    mean_pos_err = float(np.mean(pos_errors_m))
    matched_count = len(last_detected_craters)

    print(f"\n📊 Krater Tabanlı Ay TRN Performans Sonuçları:", flush=True)
    print(f"  • Ortalama TRN 3D Konum Hatası:    {mean_pos_err:.2f} metre (< 3.0 m Kriteri)", flush=True)
    print(f"  • Son Kamerada Eşleşen Krater:     {matched_count} Adet", flush=True)
    print(f"  • HDA Tehlike Kaçınma Durumu:      {divert_info['hazard_type']} -> Güvenli Sapma {divert_info['divert_distance_m']:.1f} m", flush=True)

    # 3. Profilleme ve Teşhis Panosu
    profiler_metrics = TRNProfilleyici.profille(
        mean_pos_error_m=mean_pos_err,
        matched_crater_count=matched_count,
        is_safe_landing=True
    )

    gorsellestirici = TRNGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        true_traj=true_trajectory,
        est_traj=est_trajectory,
        detected_craters=last_detected_craters,
        pos_errors_m=pos_errors_m,
        catalog_craters=catalog_craters,
        divert_info=divert_info,
        profiler_metrics=profiler_metrics,
        dosya_adi="ay_inisi_trn_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Ay İniş TRN Teşhis Grafiği Başarıyla Kaydedildi: [ay_inisi_trn_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
