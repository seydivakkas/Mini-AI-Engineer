r"""
Tesla Optimus İnsansı Robot Aktüatör ve 6-DoF Tork Kontrol Çekirdeği
=====================================================================
Bu modül; Tesla Optimus (Gen 2) insansı robotunun 28 aktüatörlü yapısal
eklem mimarisini, Euler-Lagrange ters dinamik tork hesabını
($\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$),
yerçekimi kompanzasyonunu ve 1000 Hz empedans tork kontrolcüsünü gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaOptimusJointController:
    """
    Tesla Optimus 6-DoF Eklem ve Tork Kontrolcüsü.
    """
    def __init__(
        self,
        num_dof: int = 6,
        inertia_diag: Optional[List[float]] = None,
        link_masses: Optional[List[float]] = None,
        link_lengths: Optional[List[float]] = None
    ):
        self.num_dof = num_dof
        # 6-DoF Kol/Bacak Atalet Matrisi Köşegen Değerleri (kg*m^2)
        self.inertia_diag = np.array(inertia_diag or [12.5, 8.0, 5.2, 2.0, 1.1, 0.5])
        # Uzuv kütleleri (kg) ve boyları (m)
        self.masses = np.array(link_masses or [4.5, 3.8, 2.5, 1.2, 0.8, 0.4])
        self.lengths = np.array(link_lengths or [0.35, 0.32, 0.28, 0.15, 0.10, 0.08])
        self.g_acc = 9.81  # Yerçekimi ivmesi (m/s^2)

    def compute_gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Her eklem üzerindeki yerçekimi torkunu g(q) hesaplar."""
        # g_i(q) = m_i * g * l_i * cos(q_i)
        return self.masses * self.g_acc * self.lengths * np.cos(q)

    def compute_coriolis_damping(self, q_dot: np.ndarray) -> np.ndarray:
        """Coriolis ve viskoz sönümleme kuvvetleri C(q, q_dot) * q_dot."""
        damping_coeffs = np.array([2.5, 2.0, 1.5, 0.8, 0.5, 0.2])
        return damping_coeffs * q_dot

    def compute_inverse_dynamics_torque(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ddot_des: np.ndarray
    ) -> np.ndarray:
        """
        Ters Dinamik (Inverse Dynamics) Tork Hesabı:
        tau = M(q)*q_ddot + C(q, q_dot)*q_dot + g(q)
        """
        tau_inertial = self.inertia_diag * q_ddot_des
        tau_coriolis = self.compute_coriolis_damping(q_dot)
        tau_gravity = self.compute_gravity_vector(q)

        return tau_inertial + tau_coriolis + tau_gravity

    def compute_impedance_torque(
        self,
        q_curr: np.ndarray,
        q_dot_curr: np.ndarray,
        q_des: np.ndarray,
        q_dot_des: np.ndarray,
        kp: float = 200.0,
        kd: float = 30.0
    ) -> np.ndarray:
        """
        1000 Hz Empedans ve Yerçekimi Kompanzasyonlu Tork Kontrolü.
        tau_cmd = Kp*(q_des - q) + Kd*(q_dot_des - q_dot) + g(q)
        """
        pos_err = q_des - q_curr
        vel_err = q_dot_des - q_dot_curr
        g_comp = self.compute_gravity_vector(q_curr)

        tau_feedback = kp * pos_err + kd * vel_err
        tau_cmd = tau_feedback + g_comp

        # Aktüatör Tork Doyumu (-150 Nm ile +150 Nm arası)
        return np.clip(tau_cmd, -150.0, 150.0)

    def simulate_joint_step(
        self,
        q_curr: np.ndarray,
        q_dot_curr: np.ndarray,
        q_des: np.ndarray,
        dt_s: float = 0.001
    ) -> Dict[str, Any]:
        """1 ms (1000 Hz) RTOS eklem kontrol adımı simülasyonu."""
        tau_cmd = self.compute_impedance_torque(
            q_curr=q_curr,
            q_dot_curr=q_dot_curr,
            q_des=q_des,
            q_dot_des=np.zeros(self.num_dof)
        )

        # İleri Dinamik İvme: q_ddot = (tau - C*q_dot - g) / M
        g_vec = self.compute_gravity_vector(q_curr)
        c_vec = self.compute_coriolis_damping(q_dot_curr)
        q_ddot = (tau_cmd - c_vec - g_vec) / self.inertia_diag

        # Euler İntegrasyonu
        q_dot_next = q_dot_curr + q_ddot * dt_s
        q_next = q_curr + q_dot_next * dt_s

        pos_error_norm = float(np.linalg.norm(q_des - q_next))

        return {
            "tau_cmd_nm": list(np.round(tau_cmd, 2)),
            "q_next_rad": list(np.round(q_next, 4)),
            "q_dot_next_rad_s": list(np.round(q_dot_next, 4)),
            "pos_error_norm_rad": float(np.round(pos_error_norm, 4)),
            "max_joint_torque_nm": float(np.round(np.max(np.abs(tau_cmd)), 2))
        }
