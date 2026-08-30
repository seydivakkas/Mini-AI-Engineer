"""
Day 341: Spacecraft Autonomous GNC (Guidance, Navigation & Control) under Zero-GNSS
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Sıfır-GNSS Yörünge Simülasyonunu, Optik Yıldız Takipçisi TRIAD Çözümünü,
J2 Pertürbasyonlu EKF Navigasyonunu ve 6-Panelli Teşhis Panosunu çalıştırır.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.spacecraft_gnc_motoru import (
    OpticalStarTracker,
    OrbitalEKFNavigator,
    AutonomousGNCController,
)
from src.gnc_gorsellestirici import GNCGorsellestirici
from src.gnc_profilleyici import GNCProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🛰️ DAY 341: Sıfır GNSS ile Uzay Aracı Otonom Rehberlik, Navigasyon ve Kontrol (GNC)", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    # 1. Başlangıç Yörünge Durumu (500 km LEO Dairesel Yörünge)
    r_orbit = 6378.137 + 500.0  # km
    v_orbit = np.sqrt(OrbitalEKFNavigator.MU / r_orbit)  # km/s
    
    true_state = np.array([r_orbit, 0.0, 0.0, 0.0, v_orbit, 0.0])  # [rx, ry, rz, vx, vy, vz]
    initial_estimate = true_state + np.array([0.005, -0.003, 0.002, 0.0001, -0.0001, 0.0001])

    star_tracker = OpticalStarTracker(noise_std=0.001)
    ekf = OrbitalEKFNavigator(initial_state=initial_estimate, dt=1.0)
    controller = AutonomousGNCController(kp_pos=0.01, kd_pos=0.05)

    true_orbit_history = []
    est_orbit_history = []
    attitude_errors_deg = []
    pos_errors_m = []
    thrust_profiles = []

    print("\n📌 1) 60-Saniyelik Sıfır-GNSS Yörünge ve Yönelim Simülasyonu Başlatılıyor...", flush=True)

    for step in range(60):
        # 1. Gerçek Yörünge Hareketi (İki Cisim + J2)
        r_true = true_state[:3]
        v_true = true_state[3:]
        a_true = ekf.gravitational_acceleration(r_true)
        
        true_state[:3] += v_true * 1.0 + 0.5 * a_true * 1.0
        true_state[3:] += a_true * 1.0

        # 2. Optik Yıldız Takipçisi Yönelim Belirleme (TRIAD)
        # Basit rotasyon matrisi simülasyonu
        theta = step * 0.01
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        v1_b, v2_b = star_tracker.measure_body_vectors(R_true)
        R_est = star_tracker.triad_attitude_estimation(v1_b, v2_b)
        
        # Yönelim Hatası (Euler Açı Farkı / Frobenious norm)
        att_err_rad = np.linalg.norm(R_true - R_est) / np.sqrt(8.0)
        att_err_deg = float(np.degrees(att_err_rad))
        attitude_errors_deg.append(att_err_deg)

        # 3. EKF Navigasyon Tahmini ve Gözlem Güncellemesi (Optik Ufuk / Nirengi)
        ekf.propagate_state()
        z_measured = true_state[:3] + np.random.normal(0, 0.0005, 3)  # 0.5 metre gürültülü gözlem
        ekf.measurement_update(z_measured)

        # 4. Otonom GNC İtki Rehberliği
        gnc_cmd = controller.compute_gnc_commands(ekf.state, true_state)

        pos_err_m = float(np.linalg.norm(true_state[:3] - ekf.state[:3]) * 1000.0)
        
        true_orbit_history.append(true_state[:3].copy())
        est_orbit_history.append(ekf.state[:3].copy())
        pos_errors_m.append(pos_err_m)
        thrust_profiles.append(gnc_cmd["thrust_magnitude_m_s2"])

    true_orbit_history = np.array(true_orbit_history)
    est_orbit_history = np.array(est_orbit_history)

    mean_pos_err = float(np.mean(pos_errors_m))
    mean_att_err = float(np.mean(attitude_errors_deg))
    avg_thrust = float(np.mean(thrust_profiles))

    print(f"\n📊 Sıfır-GNSS Uzay Aracı GNC Performans Sonuçları:", flush=True)
    print(f"  • Ortalama Yörünge Konum Hatası:   {mean_pos_err:.2f} metre (< 1.5 m Kriteri)", flush=True)
    print(f"  • Ortalama TRIAD Yönelim Hatası:   {mean_att_err:.4f}° (< 0.05° Kriteri)", flush=True)
    print(f"  • Ortalama GNC İtki Şiddeti:        {avg_thrust:.4f} m/s²", flush=True)
    print(f"  • Yerçekimi J2 Pertürbasyon Telafisi: ✅ AKTİF", flush=True)

    # 5. Profilleme ve Teşhis Panosu
    profiler_metrics = GNCProfilleyici.profille(
        mean_pos_error_m=mean_pos_err,
        mean_attitude_error_deg=mean_att_err,
        thrust_command_avg_m_s2=avg_thrust
    )

    gorsellestirici = GNCGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        true_orbit=true_orbit_history,
        est_orbit=est_orbit_history,
        attitude_errors=attitude_errors_deg,
        pos_errors_m=pos_errors_m,
        thrust_profiles=thrust_profiles,
        profiler_metrics=profiler_metrics,
        dosya_adi="uzay_araci_gnc_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Sıfır-GNSS GNC Teşhis Grafiği Başarıyla Kaydedildi: [uzay_araci_gnc_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
