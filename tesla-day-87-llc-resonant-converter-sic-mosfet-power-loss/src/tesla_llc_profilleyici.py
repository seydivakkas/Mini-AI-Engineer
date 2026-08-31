"""
Tesla LLC Profilleyici Modülü
==============================
Bu modül; LLC rezonans dönüştürücü ve SiC MOSFET kayıp hesaplama
algoritmalarının RTOS çözümleme hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_llc_donusturucu import TeslaLLCResonantConverter


class TeslaLLCProfilleyici:
    """
    Tesla LLC Dönüştürücü Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_llc_converter(self) -> Dict[str, Any]:
        conv = TeslaLLCResonantConverter()

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            c_inst = TeslaLLCResonantConverter()
            t0 = time.perf_counter_ns()
            _ = c_inst.calculate_losses(i_rms_a=40.0, junction_temp_c=75.0, enable_zvs=True)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # Farklı akımlara göre verimlilik eğrisi
        currents = np.linspace(5.0, 50.0, 30)
        eff_curve_zvs = []
        eff_curve_hard = []
        for i_cur in currents:
            res_z = conv.calculate_losses(i_rms_a=i_cur, enable_zvs=True)
            res_h = conv.calculate_losses(i_rms_a=i_cur, enable_zvs=False)
            eff_curve_zvs.append(res_z["efficiency_pct"])
            eff_curve_hard.append(res_h["efficiency_pct"])

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        nom_res = conv.calculate_losses(i_rms_a=40.0, junction_temp_c=75.0, enable_zvs=True)

        return {
            "resonant_freq_khz": nom_res["resonant_freq_hz"] / 1000.0,
            "nominal_efficiency": nom_res["efficiency_pct"],
            "total_loss_w": nom_res["total_loss_w"],
            "p_cond_w": nom_res["p_conduction_w"],
            "p_sw_w": nom_res["p_switching_w"],
            "p_mag_w": nom_res["p_magnetic_w"],
            "p_out_w": nom_res["p_out_w"],
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_cozumleme_hizi": int(1e6 / max(t_avg_us, 1e-4)),
            "currents": list(currents),
            "eff_zvs": eff_curve_zvs,
            "eff_hard": eff_curve_hard,
            "gecikmeler": gecikmeler_us[:200]
        }
