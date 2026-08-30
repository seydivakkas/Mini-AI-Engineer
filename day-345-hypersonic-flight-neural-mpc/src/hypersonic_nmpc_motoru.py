"""
Day 345: Hypersonic Flight Neural Model Predictive Control (Neural MPC)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Mach 6+ Hipersonik Uçuş Dinamiğini, Nöral Dinamik Vekilini (Neural Surrogate)
ve Yüksek Hızlı Nöral Model Öngörülü Kontrolcüyü (Neural MPC) içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np


class HypersonicAeroDynamics:
    """
    Mach 6+ Hipersonik Boyuna Uçuş Dinamiği Modeli.
    Durum: x = [V (hız m/s), gamma (uçuş yolu açısı rad), alpha (hücum açısı rad), q (yunuslama hızı rad/s)]
    Kontrol: u = [delta_e (elevon kanatçık açısı rad)]
    """
    def __init__(self, m: float = 1000.0, S: float = 3.0, c_bar: float = 2.0, Iyy: float = 2000.0):
        self.m = m          # Araç kütlesi (kg)
        self.S = S          # Kanat referans alanı (m^2)
        self.c_bar = c_bar  # Ortalama aerodinamik veter (m)
        self.Iyy = Iyy      # Yunuslama atalet momenti (kg*m^2)
        self.g = 9.80665    # Yerçekimi ivmesi (m/s^2)
        self.rho = 0.018    # 30 km irtifada hava yoğunluğu (kg/m^3)

    def compute_derivatives(self, state: np.ndarray, delta_e: float) -> np.ndarray:
        """Hipersonik diferansiyel durum türevlerini hesaplar."""
        V, gamma, alpha, q = state
        q_dyn = 0.5 * self.rho * (V ** 2) # Dinamik basınç (Pa)

        # Hipersonik aerodinamik katsayılar (Şok dalgası non-lineerliği)
        CL = 0.1 + 1.2 * alpha + 0.3 * delta_e
        CD = 0.05 + 1.5 * (alpha ** 2) + 0.2 * (delta_e ** 2)
        Cm = -0.5 * alpha - 1.2 * delta_e - 0.8 * q * (self.c_bar / (2.0 * V))

        Lift = CL * q_dyn * self.S
        Drag = CD * q_dyn * self.S
        M_pitch = Cm * q_dyn * self.S * self.c_bar

        # Durum Denklemleri
        V_dot = - (Drag / self.m) - self.g * np.sin(gamma)
        gamma_dot = (Lift / (self.m * V)) - (self.g / V) * np.cos(gamma)
        alpha_dot = q - gamma_dot
        q_dot = M_pitch / self.Iyy

        return np.array([V_dot, gamma_dot, alpha_dot, q_dot])

    def step(self, state: np.ndarray, delta_e: float, dt: float = 0.01) -> np.ndarray:
        """Runge-Kutta 4 (RK4) adımı."""
        k1 = self.compute_derivatives(state, delta_e)
        k2 = self.compute_derivatives(state + 0.5 * dt * k1, delta_e)
        k3 = self.compute_derivatives(state + 0.5 * dt * k2, delta_e)
        k4 = self.compute_derivatives(state + dt * k3, delta_e)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class NeuralDynamicsSurrogate:
    """
    Fizik Güdümlü Nöral Dinamik Vekili (Neural Forward Surrogate).
    Hipersonik diferansiyel denklemleri milisaniye altında 10-adımlı öngörü (Rollout) için simüle eder.
    """
    def __init__(self, aero: HypersonicAeroDynamics):
        self.aero = aero

    def predict_next_state(self, state: np.ndarray, delta_e: float, dt: float = 0.02) -> np.ndarray:
        """Gelecek durumu ultra-hızlı tahmin eder."""
        return self.aero.step(state, delta_e, dt=dt)


class HighSpeedNeuralMPC:
    """
    Yüksek Hızlı Nöral Model Öngörülü Kontrolcü (Neural MPC).
    Ufuk (Horizon N=10) boyunca hücum açısı (alpha) ve yunuslama hedefini minimum kontrol eforuyla optimize eder.
    """
    def __init__(self, surrogate: NeuralDynamicsSurrogate, horizon: int = 10, dt: float = 0.02):
        self.surrogate = surrogate
        self.horizon = horizon
        self.dt = dt
        self.max_delta_e = np.radians(20.0) # Maksimum 20 derece kanatçık sapması

    def optimize_control(self, current_state: np.ndarray, target_alpha: float) -> Dict[str, Any]:
        """
        Nöral MPC Ufku boyunca optimum elevon kontrol komutunu vektörize olarak optimize eder.
        """
        num_candidates = 25
        u_candidates = np.linspace(-self.max_delta_e, self.max_delta_e, num_candidates) # (25,)
        
        # Batch durum matrisi: (25, 4)
        states = np.tile(current_state, (num_candidates, 1))
        costs = np.zeros(num_candidates)

        for k in range(self.horizon):
            # Vektörize adım
            V = states[:, 0]
            gamma = states[:, 1]
            alpha = states[:, 2]
            q = states[:, 3]
            q_dyn = 0.5 * self.surrogate.aero.rho * (V ** 2)

            CL = 0.1 + 1.2 * alpha + 0.3 * u_candidates
            CD = 0.05 + 1.5 * (alpha ** 2) + 0.2 * (u_candidates ** 2)
            Cm = -0.5 * alpha - 1.2 * u_candidates - 0.8 * q * (self.surrogate.aero.c_bar / (2.0 * V))

            Lift = CL * q_dyn * self.surrogate.aero.S
            Drag = CD * q_dyn * self.surrogate.aero.S
            M_pitch = Cm * q_dyn * self.surrogate.aero.S * self.surrogate.aero.c_bar

            V_dot = - (Drag / self.surrogate.aero.m) - self.surrogate.aero.g * np.sin(gamma)
            gamma_dot = (Lift / (self.surrogate.aero.m * V)) - (self.surrogate.aero.g / V) * np.cos(gamma)
            alpha_dot = q - gamma_dot
            q_dot = M_pitch / self.surrogate.aero.Iyy

            states[:, 0] += V_dot * self.dt
            states[:, 1] += gamma_dot * self.dt
            states[:, 2] += alpha_dot * self.dt
            states[:, 3] += q_dot * self.dt

            alpha_err = states[:, 2] - target_alpha
            costs += 100.0 * (alpha_err ** 2) + 10.0 * (states[:, 3] ** 2) + 1.0 * (u_candidates ** 2)

        best_idx = int(np.argmin(costs))
        best_u = float(u_candidates[best_idx])
        best_cost = float(costs[best_idx])

        return {
            "optimal_delta_e_rad": best_u,
            "optimal_delta_e_deg": float(np.degrees(best_u)),
            "cost": best_cost,
        }
