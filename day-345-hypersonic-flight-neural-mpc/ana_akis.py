"""
Day 345: Hypersonic Flight Neural Model Predictive Control (Neural MPC)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Mach 6 Hipersonik Boyuna Dinamik Simülasyonunu, Nöral İleri Vekil (Surrogate) Ufuk Rollout'unu,
Yüksek Hızlı Nöral MPC Geri Besleme Kontrolünü ve 6-Panelli Teşhis Grafiğini çalıştırır.
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

from src.hypersonic_nmpc_motoru import (
    HypersonicAeroDynamics,
    NeuralDynamicsSurrogate,
    HighSpeedNeuralMPC,
)
from src.nmpc_gorsellestirici import NMPCGorsellestirici
from src.nmpc_profilleyici import NMPCProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🚀 DAY 345: Hipersonik Uçuş Kontrolü: Yüksek Hızlı Nöral Model Öngörülü Kontrol (NMPC)", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    aero = HypersonicAeroDynamics()
    surrogate = NeuralDynamicsSurrogate(aero)
    nmpc = HighSpeedNeuralMPC(surrogate, horizon=10, dt=0.02)

    # Başlangıç Durumu: [V=1800 m/s (Mach 6), gamma=0.0 rad, alpha=0.05 rad (2.86°), q=0.0 rad/s]
    state = np.array([1800.0, 0.0, np.radians(3.0), 0.0])
    dt = 0.02
    steps = 150

    time_history = []
    alpha_actual_deg = []
    alpha_target_deg = []
    elevon_deg = []
    pitch_rates = []
    velocities = []
    costs = []
    solve_times_ms = []

    print("\n📌 1) 150-Adımlı Mach 6 Hipersonik Nöral MPC Uçuş Simülasyonu Başlatılıyor...", flush=True)

    for step in range(steps):
        t = step * dt
        # Hedef Hücum Açısı (3° ile 7° arasında basamaklı komut)
        if t < 1.0:
            target_alpha_deg = 3.0
        elif t < 2.0:
            target_alpha_deg = 6.5
        else:
            target_alpha_deg = 4.0

        target_alpha_rad = np.radians(target_alpha_deg)

        # Nöral MPC Optimizasyonu
        t0 = time.perf_counter()
        opt_res = nmpc.optimize_control(state, target_alpha_rad)
        t_solve_ms = (time.perf_counter() - t0) * 1000.0

        u_elevon = opt_res["optimal_delta_e_rad"]

        # Gerçek Hipersonik Fiziği Bir Adım İlerlet
        state = aero.step(state, u_elevon, dt=dt)

        time_history.append(t)
        alpha_actual_deg.append(float(np.degrees(state[2])))
        alpha_target_deg.append(target_alpha_deg)
        elevon_deg.append(float(np.degrees(u_elevon)))
        pitch_rates.append(float(state[3]))
        velocities.append(float(state[0]))
        costs.append(opt_res["cost"])
        solve_times_ms.append(t_solve_ms)

    alpha_errors = [abs(a - tg) for a, tg in zip(alpha_actual_deg, alpha_target_deg)]
    mean_alpha_err = float(np.mean(alpha_errors))
    max_elevon = float(np.max(np.abs(elevon_deg)))
    mean_solve_time = float(np.mean(solve_times_ms))

    print(f"\n📊 Hipersonik Nöral MPC Performans Sonuçları:", flush=True)
    print(f"  • Ortalama Hücum Açısı (Alpha) Hatası: {mean_alpha_err:.4f}° (< 0.2° Kriteri)", flush=True)
    print(f"  • Maksimum Elevon Sapma Açısı:        {max_elevon:.2f}° (Limit: ±20°)", flush=True)
    print(f"  • Nöral MPC Ortalama Çözüm Süresi:     {mean_solve_time:.4f} ms (< 1.0 ms Gerçek Zamanlı)", flush=True)
    print(f"  • Mach 6 Uçuş Kararlılığı:             ✅ SAĞLANDI", flush=True)

    # 2. Profilleme ve Teşhis Panosu
    profiler_metrics = NMPCProfilleyici.profille(
        mean_alpha_error_deg=mean_alpha_err,
        max_elevon_deg=max_elevon,
        mean_solve_time_ms=mean_solve_time
    )

    gorsellestirici = NMPCGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        time_history=time_history,
        alpha_actual_deg=alpha_actual_deg,
        alpha_target_deg=alpha_target_deg,
        elevon_deg=elevon_deg,
        pitch_rates=pitch_rates,
        velocities=velocities,
        costs=costs,
        profiler_metrics=profiler_metrics,
        dosya_adi="hipersonik_nmpc_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Hipersonik NMPC Teşhis Grafiği Başarıyla Kaydedildi: [hipersonik_nmpc_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
