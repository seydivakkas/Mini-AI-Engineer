"""
Tesla Batarya ECM Profilleyici Modülü
======================================
Bu modül; LFP ve NMC hücre modellerinin WLTP dinamik sürüş profillerindeki
voltaj çökmesini (Voltage Sag), sıcaklık artışını ve 2-RC simülasyon hızını analiz eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_batarya_ecm_modeli import (
    TeslaBatteryECM,
    BatteryCellParameters,
    BatteryChemistry
)


class TeslaECMProfilleyici:
    """
    Batarya Kimyası ve ECM Performans Profilleyicisi.
    """
    def __init__(self, sim_adimlari: int = 1000):
        self.sim_adimlari = sim_adimlari

    def benchmark_batarya_ecm(self) -> Dict[str, Any]:
        params_lfp = BatteryCellParameters(chemistry=BatteryChemistry.LFP)
        params_nmc = BatteryCellParameters(chemistry=BatteryChemistry.NMC)

        bms_lfp = TeslaBatteryECM(params_lfp, initial_soc=0.85, initial_temp_c=25.0)
        bms_nmc = TeslaBatteryECM(params_nmc, initial_soc=0.85, initial_temp_c=25.0)

        # 1. 200A Hızlanma Darbesinde (Acceleration Pulse) Voltaj Çökmesi
        lfp_voltajlar = []
        nmc_voltajlar = []
        lfp_soc_list = []
        nmc_soc_list = []
        lfp_sicaklik = []
        nmc_sicaklik = []

        gecikmeler_step_us: List[float] = []

        # Simülasyon: Dinamik İvmelenme + Rejenerasyon Profili
        for i in range(self.sim_adimlari):
            # Dinamik akım profili: -50A (Regen) ile +180A (Tam Gaz) arasında
            akim = float(60.0 + 80.0 * np.sin(i * 0.05) + 30.0 * np.sin(i * 0.2))

            t0 = time.perf_counter_ns()
            out_lfp = bms_lfp.step(current_a=akim, dt_s=0.1)
            t1 = time.perf_counter_ns()
            gecikmeler_step_us.append(float(t1 - t0) / 1000.0)

            out_nmc = bms_nmc.step(current_a=akim, dt_s=0.1)

            lfp_voltajlar.append(out_lfp["v_terminal"])
            nmc_voltajlar.append(out_nmc["v_terminal"])
            lfp_soc_list.append(out_lfp["soc"] * 100.0)
            nmc_soc_list.append(out_nmc["soc"] * 100.0)
            lfp_sicaklik.append(out_lfp["temp_c"])
            nmc_sicaklik.append(out_nmc["temp_c"])

        step_dizi = np.array(gecikmeler_step_us)
        t_step_avg_us = float(np.mean(step_dizi))

        # 2. Soğuk Hava (-10°C) İç Direnç Artış Çarpanı
        bms_cold = TeslaBatteryECM(params_nmc, initial_soc=0.85, initial_temp_c=-10.0)
        r0_cold = bms_cold.get_temperature_adjusted_r0()
        r0_warm = bms_nmc.params.r0_ohmic_ohm
        cold_increase_ratio = r0_cold / r0_warm

        return {
            "ecm_step_ortalama_us": t_step_avg_us,
            "ecm_step_p99_us": float(np.percentile(step_dizi, 99)),
            "saniyelik_ecm_adimi": int(1e6 / max(t_step_avg_us, 1e-4)),
            "lfp_son_voltaj": lfp_voltajlar[-1],
            "nmc_son_voltaj": nmc_voltajlar[-1],
            "lfp_son_soc": lfp_soc_list[-1],
            "nmc_son_soc": nmc_soc_list[-1],
            "cold_r0_ratio": cold_increase_ratio,
            "lfp_voltajlar": lfp_voltajlar,
            "nmc_voltajlar": nmc_voltajlar,
            "lfp_soc": lfp_soc_list,
            "nmc_soc": nmc_soc_list,
            "lfp_temp": lfp_sicaklik,
            "nmc_temp": nmc_sicaklik,
            "step_gecikmeler": gecikmeler_step_us[:200]
        }
