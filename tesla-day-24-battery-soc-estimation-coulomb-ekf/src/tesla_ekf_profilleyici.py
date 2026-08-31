"""
Tesla EKF SoC Kestirici Profilleyici Modülü
===========================================
Bu modül; Saf Coulomb Counting ile EKF (Extended Kalman Filter) arasındaki
doğruluk farkını (RMSE, Drift Dayanımı) ve EKF matris çözücü hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_ekf_soc_kestirici import BatteryEKFSoCEstimator, CoulombCounter


class TeslaEKFProfilleyici:
    """
    EKF ve Coulomb Counting Karşılaştırmalı Performans Profilleyicisi.
    """
    def __init__(self, sim_adimlari: int = 1500):
        self.sim_adimlari = sim_adimlari

    def benchmark_ekf_soc(self) -> Dict[str, Any]:
        # Gerçek Hücre Başlangıcı: %85 SoC
        true_soc_init = 0.85
        # Kötü Başlangıç Tahmini: %50 SoC (BMS yeniden başladı varsayımı)
        initial_guess_soc = 0.50

        # Akım Sensörü Hatası (+1.5 Amper DC Bias)
        current_sensor_bias_a = 1.5

        ekf = BatteryEKFSoCEstimator(initial_soc_guess=initial_guess_soc, capacity_ah=75.0)
        coulomb = CoulombCounter(initial_soc=initial_guess_soc, capacity_ah=75.0)

        # Gerçek Hücre Durum Değişkenleri
        true_soc = true_soc_init
        true_v_rc1 = 0.0
        true_v_rc2 = 0.0

        true_soc_history = []
        coulomb_soc_history = []
        ekf_soc_history = []
        soc_std_history = []
        gecikmeler_ekf_us: List[float] = []

        for i in range(self.sim_adimlari):
            # Gerçek Dinamik Akım (A)
            i_true = float(50.0 + 40.0 * np.sin(i * 0.02) + 20.0 * np.cos(i * 0.1))
            # Sensörün okuduğu gürültülü ve yanlı akım
            i_measured = i_true + current_sensor_bias_a + np.random.normal(0, 0.2)

            # 1. Gerçek Hücre Simülasyonu
            true_soc -= (i_true * 0.1) / (75.0 * 3600.0)
            tau1 = 0.0010 * 2500.0
            tau2 = 0.0008 * 20000.0
            true_v_rc1 = np.exp(-0.1 / tau1) * true_v_rc1 + 0.0010 * (1.0 - np.exp(-0.1 / tau1)) * i_true
            true_v_rc2 = np.exp(-0.1 / tau2) * true_v_rc2 + 0.0008 * (1.0 - np.exp(-0.1 / tau2)) * i_true
            ocv_true, _ = ekf._compute_ocv_and_derivative(true_soc)
            v_terminal_true = ocv_true - (i_true * 0.0015) - true_v_rc1 - true_v_rc2
            v_measured = v_terminal_true + np.random.normal(0, 0.005)  # 5 mV gürültü

            # 2. Coulomb Counting Adımı
            coulomb_soc = coulomb.step(i_measured, dt_s=0.1)

            # 3. EKF Adımı
            t0 = time.perf_counter_ns()
            out_ekf = ekf.step(current_a=i_measured, measured_terminal_v=v_measured, dt_s=0.1)
            t1 = time.perf_counter_ns()
            gecikmeler_ekf_us.append(float(t1 - t0) / 1000.0)

            true_soc_history.append(true_soc * 100.0)
            coulomb_soc_history.append(coulomb_soc * 100.0)
            ekf_soc_history.append(out_ekf["estimated_soc"] * 100.0)
            soc_std_history.append(out_ekf["soc_uncertainty_std"] * 100.0)

        ekf_dizi = np.array(gecikmeler_ekf_us)
        t_ekf_avg_us = float(np.mean(ekf_dizi))

        # Hata Metrikleri (Yakınsama sonrası son 1000 adım)
        true_arr = np.array(true_soc_history[500:])
        ekf_arr = np.array(ekf_soc_history[500:])
        coulomb_arr = np.array(coulomb_soc_history[500:])

        rmse_ekf = float(np.sqrt(np.mean((ekf_arr - true_arr) ** 2)))
        rmse_coulomb = float(np.sqrt(np.mean((coulomb_arr - true_arr) ** 2)))

        return {
            "ekf_step_ortalama_us": t_ekf_avg_us,
            "ekf_step_p99_us": float(np.percentile(ekf_dizi, 99)),
            "saniyelik_ekf_adimi": int(1e6 / max(t_ekf_avg_us, 1e-4)),
            "rmse_ekf_pct": rmse_ekf,
            "rmse_coulomb_pct": rmse_coulomb,
            "hata_iyilesme_orani": rmse_coulomb / max(rmse_ekf, 1e-4),
            "true_soc": true_soc_history,
            "coulomb_soc": coulomb_soc_history,
            "ekf_soc": ekf_soc_history,
            "soc_std": soc_std_history,
            "ekf_gecikmeler": gecikmeler_ekf_us[:200]
        }
