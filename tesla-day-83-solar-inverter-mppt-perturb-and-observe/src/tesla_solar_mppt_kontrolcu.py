r"""
Tesla Solar Inverter ve MPPT (Perturb & Observe) Çekirdeği
==========================================================
Bu modül; Tesla Solar Roof ve Powerwall 3 dahili solar invertörünün
Fotovoltaik (PV) P-V ve I-V eğrisi modellemesini, Perturb and Observe (P&O)
Maksimum Güç Noktası Takip (MPPT) algoritmasını ve %99+ verimlilik
optimizasyonunu gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaSolarMPPTController:
    """
    Tesla Solar Roof & Inverter MPPT (P&O) Kontrolcüsü.
    """
    def __init__(
        self,
        v_oc: float = 50.0,
        i_sc: float = 10.0,
        optimal_v_mpp: float = 33.44
    ):
        self.v_oc = v_oc
        self.i_sc = i_sc
        self.optimal_v_mpp = optimal_v_mpp
        self.optimal_p_mpp = self.calculate_pv_power(optimal_v_mpp)

        self.prev_v = 0.0
        self.prev_p = 0.0
        self.step_dir = 1.0  # +1: Artır, -1: Azalt

    def calculate_pv_current(self, v: float) -> float:
        """PV Panel I-V Karakteristiği (Diyot Modeli Yaklaşımı)."""
        if v < 0:
            return self.i_sc
        if v >= self.v_oc:
            return 0.0
        # Basitleştirilmiş non-lineer PV akım modeli
        return float(self.i_sc * max(0.0, 1.0 - (v / self.v_oc) ** 4))

    def calculate_pv_power(self, v: float) -> float:
        """P = V * I(V) Güç Hesabı."""
        i = self.calculate_pv_current(v)
        return float(v * i)

    def mppt_step_perturb_and_observe(
        self,
        v_curr: float,
        p_curr: float,
        step_v: float = 0.5
    ) -> float:
        """
        Perturb and Observe (P&O) MPPT Algoritması.
        """
        delta_p = p_curr - self.prev_p
        delta_v = v_curr - self.prev_v

        if abs(delta_v) > 1e-4:
            if delta_p > 0:
                # Güç arttı -> Aynı yönde ilerle
                self.step_dir = 1.0 if delta_v > 0 else -1.0
            else:
                # Güç düştü -> Yönü tersine çevir
                self.step_dir = -1.0 if delta_v > 0 else 1.0

        v_next = v_curr + self.step_dir * step_v
        v_next_clamped = float(np.clip(v_next, 5.0, self.v_oc - 1.0))

        self.prev_v = v_curr
        self.prev_p = p_curr

        return v_next_clamped

    def simulate_mppt_tracking(
        self,
        initial_v: float = 20.0,
        iterations: int = 50,
        step_v: float = 0.5
    ) -> Dict[str, Any]:
        """MPPT algoritmasının maksimum güç noktasına kilitlenmesini simüle eder."""
        v_history = []
        p_history = []

        v_curr = initial_v
        self.prev_v = initial_v - 0.1
        self.prev_p = self.calculate_pv_power(self.prev_v)

        for _ in range(iterations):
            p_curr = self.calculate_pv_power(v_curr)
            v_history.append(v_curr)
            p_history.append(p_curr)

            v_curr = self.mppt_step_perturb_and_observe(v_curr, p_curr, step_v=step_v)

        final_p = p_history[-1]
        efficiency_pct = (final_p / self.optimal_p_mpp) * 100.0

        return {
            "iterations": iterations,
            "v_history": v_history,
            "p_history": p_history,
            "optimal_v_mpp": self.optimal_v_mpp,
            "optimal_p_mpp": self.optimal_p_mpp,
            "final_tracked_v": v_history[-1],
            "final_tracked_p": final_p,
            "mppt_efficiency_pct": float(np.round(efficiency_pct, 2)),
            "locked_on_mpp": bool(efficiency_pct >= 99.0)
        }
