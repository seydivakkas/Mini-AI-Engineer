"""
Day 359: Extreme-Temperature Adaptive Neural Scaling & Dynamic Voltage/Frequency Scaling (DVFS)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Ekstrem Çevre Sıcaklığı ve Çip Isıl (Thermal RC) Dinamiğini,
Elastik Nöral Ağ Boyut Ölçeklemesini ve Termal Acil Durum DVFS Valisi (Governor) Ajanını içerir.
"""

from enum import Enum
from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class ThermalOperatingMode(str, Enum):
    """Termal Yönetim Çalışma Modları."""
    FULL_POWER_HIGH_PERF = "FULL_POWER_HIGH_PERF" # 100% Model, 1.2 GHz Clock
    WARM_BALANCED = "WARM_BALANCED" # 50% Model, 800 MHz Clock
    CRITICAL_HEAT_SURVIVAL = "CRITICAL_HEAT_SURVIVAL" # 25% Model, 400 MHz Clock


class AvionicsThermalDieSimulator:
    """
    Aviyonik İşlemci Çip Sıcaklığı (Die Temp) ve Güç Tüketimi Simülatörü.
    Dinamik anahtarlama gücü (P_dyn = C V^2 f) ve sıcaklığa bağlı kaçak akım (P_leak) modelini çözer.
    """
    def __init__(self, c_th: float = 12.0, r_th: float = 2.5):
        self.c_th = c_th # Isıl kapasitans (J / °C)
        self.r_th = r_th # Isıl direnç (°C / W)
        self.t_die = 35.0 # Başlangıç çip sıcaklığı (°C)

    def step_thermal(
        self,
        clock_ghz: float,
        model_load_ratio: float,
        t_ambient: float,
        dt: float = 0.5
    ) -> Tuple[float, float]:
        """
        Bir zaman adımı termal RC diferansiyel denklemini çözer.
        """
        # Dinamik Güç (W) = 15.0 * (f/1.2)^2 * load
        p_dyn = 14.0 * (clock_ghz / 1.2)**2 * model_load_ratio
        # Kaçak Güç (T_die arttıkça üstel artar)
        p_leak = 1.5 * np.exp((self.t_die - 40.0) / 35.0)
        p_total = p_dyn + p_leak

        # dT_die / dt = (P_total - (T_die - T_amb) / R_th) / C_th
        cooling_power = (self.t_die - t_ambient) / self.r_th
        dt_die = ((p_total - cooling_power) / self.c_th) * dt
        self.t_die += dt_die

        return self.t_die, p_total


class ElasticNeuralScalingModel:
    """
    Elastik Nöral Ağ Modeli (Dynamic Width & Depth Scaling).
    Sıcaklığa göre inference kanal genişliğini dinamik olarak budar (%100 -> %50 -> %25).
    """
    @staticmethod
    def infer(input_vec: np.ndarray, mode: ThermalOperatingMode) -> Tuple[float, float]:
        """Mode'a göre çıkarım doğruluğu ve işlem yükü döner."""
        if mode == ThermalOperatingMode.FULL_POWER_HIGH_PERF:
            accuracy = 0.985
            load_ratio = 1.0 # 100% Compute
        elif mode == ThermalOperatingMode.WARM_BALANCED:
            accuracy = 0.942
            load_ratio = 0.48 # 48% Compute
        else: # CRITICAL_HEAT_SURVIVAL
            accuracy = 0.885
            load_ratio = 0.22 # 22% Compute (Temel Hayatta Kalma)
        return accuracy, load_ratio


class DynamicThermalGovernorAgent:
    """
    Termal Uyumlu Otonom DVFS ve Nöral Ölçekleme Yöneticisi (Governor).
    Çip sıcaklığını T_die < 95°C güvenli bölgesinde tutar.
    """
    def __init__(self, t_warm: float = 70.0, t_critical: float = 88.0):
        self.t_warm = t_warm
        self.t_critical = t_critical

    def select_mode_and_clock(self, current_t_die: float) -> Tuple[ThermalOperatingMode, float]:
        """Sıcaklığa göre çalışma modu ve saat frekansı (GHz) seçer."""
        if current_t_die >= self.t_critical:
            return ThermalOperatingMode.CRITICAL_HEAT_SURVIVAL, 0.40 # 400 MHz
        elif current_t_die >= self.t_warm:
            return ThermalOperatingMode.WARM_BALANCED, 0.80 # 800 MHz
        else:
            return ThermalOperatingMode.FULL_POWER_HIGH_PERF, 1.20 # 1.2 GHz


class ExtremeTemperatureFlightMission:
    """
    Uçtan Uca Ekstrem Sıcaklık Uçuş Görevi Simülatörü.
    """
    def __init__(self):
        self.governor = DynamicThermalGovernorAgent()

    def run_reentry_thermal_profile(self, steps: int = 120) -> Dict[str, Any]:
        """Hipersonik atmosfere giriş sıcaklık darbesi simülasyonunu icra eder."""
        time_axis = np.linspace(0, 60.0, steps) # 60 saniye
        
        # Dış Ortam Sıcaklık Profili (25°C -> 110°C Hipersonik Isınma Zirvesi -> 40°C Soğuma)
        t_ambient_profile = 25.0 + 85.0 * np.exp(-((time_axis - 28.0) / 12.0)**2)

        # 1. Yönetilmeyen Sabit Sistem (Unmanaged Baseline)
        sim_unmanaged = AvionicsThermalDieSimulator()
        unmanaged_t_die = []
        unmanaged_shutdown = False
        shutdown_step = -1

        for step in range(steps):
            t_amb = t_ambient_profile[step]
            t_die, p_tot = sim_unmanaged.step_thermal(clock_ghz=1.2, model_load_ratio=1.0, t_ambient=t_amb)
            unmanaged_t_die.append(t_die)
            if t_die >= 105.0 and not unmanaged_shutdown:
                unmanaged_shutdown = True
                shutdown_step = step

        # 2. Yapay Zeka Termal Ölçeklemeli Sistem (Bizim Sistemimiz)
        sim_ai = AvionicsThermalDieSimulator()
        ai_t_die = []
        ai_power = []
        ai_modes = []
        ai_clocks = []
        ai_accuracies = []

        for step in range(steps):
            t_amb = t_ambient_profile[step]
            mode, clock_ghz = self.governor.select_mode_and_clock(sim_ai.t_die)
            acc, load = ElasticNeuralScalingModel.infer(np.zeros(5), mode)
            
            t_die, p_tot = sim_ai.step_thermal(clock_ghz=clock_ghz, model_load_ratio=load, t_ambient=t_amb)
            
            ai_t_die.append(t_die)
            ai_power.append(p_tot)
            ai_modes.append(mode.value)
            ai_clocks.append(clock_ghz)
            ai_accuracies.append(acc)

        return {
            "time_axis": time_axis,
            "t_ambient_profile": t_ambient_profile,
            "unmanaged_t_die": np.array(unmanaged_t_die),
            "unmanaged_shutdown": unmanaged_shutdown,
            "shutdown_step": shutdown_step,
            "ai_t_die": np.array(ai_t_die),
            "ai_power": np.array(ai_power),
            "ai_clocks": np.array(ai_clocks),
            "ai_accuracies": np.array(ai_accuracies),
            "max_ai_temp": float(np.max(ai_t_die)),
            "survived_mission": bool(np.max(ai_t_die) < 95.0)
        }
