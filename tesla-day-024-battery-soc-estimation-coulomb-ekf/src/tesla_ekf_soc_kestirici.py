"""
Tesla Batarya SoC Kestirimi: Coulomb Counting ve Genişletilmiş Kalman Filtresi (EKF)
====================================================================================
Bu modül; akım sensörü kaymalarına (Sensor Drift/Bias) dayanıklı 3-Durumlu
Genişletilmiş Kalman Filtresi (Extended Kalman Filter - EKF) ile
Tesla batarya hücresi Şarj Durumu (State of Charge - SoC) kestirimini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class CoulombCounter:
    """
    Saf Coulomb Counting (Amper-Saat İntegrasyonu).
    Akım sensöründeki DC ofset veya gürültü sebebiyle zamanla sınırsız kayar!
    """
    def __init__(self, initial_soc: float, capacity_ah: float = 75.0):
        self.soc = float(initial_soc)
        self.total_coulombs = capacity_ah * 3600.0

    def step(self, current_a: float, dt_s: float = 0.1) -> float:
        self.soc -= (current_a * dt_s) / self.total_coulombs
        self.soc = float(np.clip(self.soc, 0.0, 1.0))
        return self.soc


class BatteryEKFSoCEstimator:
    """
    3-Durumlu Genişletilmiş Kalman Filtresi (EKF).
    Durum Vektörü: x = [SoC, V_RC1, V_RC2]^T
    Ölçüm: V_terminal = OCV(SoC) - I*R0 - V_RC1 - V_RC2 + v_meas
    """
    def __init__(
        self,
        initial_soc_guess: float = 0.50,
        capacity_ah: float = 75.0,
        r0_ohm: float = 0.0015,
        r1_ohm: float = 0.0010,
        c1_f: float = 2500.0,
        r2_ohm: float = 0.0008,
        c2_f: float = 20000.0
    ):
        self.capacity_coulombs = capacity_ah * 3600.0
        self.r0 = r0_ohm
        self.r1 = r1_ohm
        self.c1 = c1_f
        self.r2 = r2_ohm
        self.c2 = c2_f

        # Durum Vektörü: [SoC, V_RC1, V_RC2]
        self.x = np.array([initial_soc_guess, 0.0, 0.0], dtype=np.float64)

        # Durum Kovaryans Matrisi P
        self.P = np.diag([1e-2, 1e-4, 1e-4])

        # Süreç Gürültüsü Kovaryansı Q
        self.Q = np.diag([1e-6, 1e-6, 1e-6])

        # Ölçüm Gürültüsü Kovaryansı R (Voltaj sensörü varyansı: ~10 mV RMS)
        self.R = np.array([[1e-4]])

    def _compute_ocv_and_derivative(self, soc: float) -> Tuple[float, float]:
        """OCV ve Jacobian için d(OCV)/d(SoC) türevini analitik hesaplar."""
        soc_c = float(np.clip(soc, 0.001, 0.999))
        # NMC Polinomu: OCV = 3.0 + 1.2*SoC + 0.05*ln(SoC) - 0.02*exp(-15*SoC)
        ocv = 3.0 + 1.20 * soc_c + 0.05 * np.log(soc_c) - 0.02 * np.exp(-15.0 * soc_c)
        # Türev d(OCV)/d(SoC)
        docv_dsoc = 1.20 + (0.05 / soc_c) + 0.30 * np.exp(-15.0 * soc_c)
        return float(ocv), float(docv_dsoc)

    def step(self, current_a: float, measured_terminal_v: float, dt_s: float = 0.1) -> Dict[str, float]:
        """
        EKF Zaman Güncellemesi (Tahmin) ve Ölçüm Güncellemesi (Düzeltme).
        """
        # --- 1. TAHMİN ADIMI (Time Update / Prediction) ---
        tau1 = self.r1 * self.c1
        tau2 = self.r2 * self.c2
        exp1 = np.exp(-dt_s / tau1)
        exp2 = np.exp(-dt_s / tau2)

        # Durum Geçiş Matrisi A
        A = np.array([
            [1.0, 0.0,  0.0],
            [0.0, exp1, 0.0],
            [0.0, 0.0,  exp2]
        ])

        # Durum Tahmini x_hat_minus
        soc_pred = self.x[0] - (current_a * dt_s) / self.capacity_coulombs
        v_rc1_pred = exp1 * self.x[1] + self.r1 * (1.0 - exp1) * current_a
        v_rc2_pred = exp2 * self.x[2] + self.r2 * (1.0 - exp2) * current_a

        self.x = np.array([np.clip(soc_pred, 0.0, 1.0), v_rc1_pred, v_rc2_pred])

        # Kovaryans Tahmini P_minus = A * P * A^T + Q
        self.P = A @ self.P @ A.T + self.Q

        # --- 2. DÜZELTME ADIMI (Measurement Update / Correction) ---
        ocv_pred, docv_dsoc = self._compute_ocv_and_derivative(self.x[0])
        v_pred = ocv_pred - (current_a * self.r0) - self.x[1] - self.x[2]

        # Ölçüm Jacobian Matrisi C = [d(OCV)/d(SoC), -1, -1]
        C = np.array([[docv_dsoc, -1.0, -1.0]])

        # İnovasyon (Artık Hata)
        y_residual = measured_terminal_v - v_pred

        # İnovasyon Kovaryansı S = C * P * C^T + R
        S = C @ self.P @ C.T + self.R

        # Kalman Kazancı K = P * C^T * S^(-1)
        K = self.P @ C.T @ np.linalg.inv(S)

        # Durum Güncellemesi x_hat_plus = x_hat_minus + K * y
        self.x = self.x + (K.flatten() * y_residual)
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)

        # Kovaryans Güncellemesi P_plus = (I - K * C) * P
        I_mat = np.eye(3)
        self.P = (I_mat - K @ C) @ self.P

        return {
            "estimated_soc": float(self.x[0]),
            "estimated_v_rc1": float(self.x[1]),
            "estimated_v_rc2": float(self.x[2]),
            "predicted_voltage": float(v_pred),
            "voltage_residual": float(y_residual),
            "soc_uncertainty_std": float(np.sqrt(self.P[0, 0]))
        }
