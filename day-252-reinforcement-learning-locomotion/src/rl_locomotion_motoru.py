"""
Pekiştirmeli Öğrenme ile Robotik Yürüme (RL Locomotion - PPO) Motoru (Day 252).
12-DoF Quadruped MDP, Çok Bileşenli Ödül Şekillendirme ve PPO Aktör-Kritik Politikası.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardShaper:
    """Robotik Yürüme Çok Bileşenli Ödül ve Maliyet Hesaplayıcısı."""

    @classmethod
    def compute_step_reward(
        cls,
        lin_vel: np.ndarray,
        target_lin_vel: np.ndarray,
        ang_vel: float,
        target_ang_vel: float,
        joint_torques: np.ndarray,
        joint_acc: np.ndarray,
        base_z: float,
        target_z: float = 0.35,
    ) -> Dict[str, float]:
        """Tüm alt ödül terimlerini ve Cost of Transport (COT) değerini hesaplar."""
        # 1. Doğrusal Hız Takip Ödülü
        lin_err = np.sum((lin_vel[:2] - target_lin_vel[:2]) ** 2)
        r_lin = float(np.exp(-lin_err / 0.25))

        # 2. Açısal Hız Takip Ödülü
        ang_err = (ang_vel - target_ang_vel) ** 2
        r_ang = float(np.exp(-ang_err / 0.25))

        # 3. Gövde Yüksekliği ve Dikey Hız Cezası
        z_err = (base_z - target_z) ** 2
        r_height = -2.0 * float(z_err)

        # 4. Enerji / Tork Tüketim Cezası
        r_torque = -0.0001 * float(np.sum(joint_torques ** 2))

        # 5. Eklem İvme / Yumuşaklık Cezası
        r_smooth = -1e-6 * float(np.sum(joint_acc ** 2))

        # 6. Hayatta Kalma Bonusu
        r_alive = 1.0

        toplam_odul = round(r_lin + 0.5 * r_ang + r_height + r_torque + r_smooth + r_alive, 4)

        return {
            "toplam_odul": toplam_odul,
            "r_lin_tracking": round(r_lin, 4),
            "r_ang_tracking": round(r_ang, 4),
            "r_height_penalty": round(r_height, 4),
            "r_torque_penalty": round(r_torque, 4),
        }

    @classmethod
    def compute_cost_of_transport(
        cls,
        joint_torques: np.ndarray,
        joint_vel: np.ndarray,
        mass_kg: float = 12.0,
        lin_vel_mag: float = 1.0,
        g: float = 9.81,
    ) -> float:
        """Taşıma Maliyeti (Cost of Transport - COT = Güç / (m * g * v))."""
        if lin_vel_mag < 0.05:
            return 10.0
        guc = float(np.sum(np.abs(joint_torques * joint_vel)))
        cot = guc / (mass_kg * g * lin_vel_mag)
        return round(float(cot), 3)


class PPOActorCritic(nn.Module):
    """12-DoF Robotik Yürüme Aktör-Kritik PyTorch Ağı."""

    def __init__(self, obs_dim: int = 48, act_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Aktör Ağı: s -> mean(a)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Kritik Ağı: s -> V(s)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gözlemden deterministik eylem ortalamasını ve standart sapmayı döner."""
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        return mean, std

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """Gözlemden durum değerini V(s) döner."""
        return self.critic(obs)

    def get_action(self, obs_np: np.ndarray) -> np.ndarray:
        """Numpy girdi için çıkarım yapar."""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            mean, _ = self.forward_actor(obs_tensor)
            return mean.squeeze(0).numpy()


class LocomotionEnvironment:
    """12-DoF Dört Bacaklı (Quadruped) Basitleştirilmiş Dinamik Çevre Simülasyonu."""

    def __init__(self):
        self.dof = 12
        self.nominal_q = np.array([0.0, 0.7, -1.4] * 4, dtype=np.float64)
        self.q = self.nominal_q.copy()
        self.q_dot = np.zeros(12, dtype=np.float64)
        self.base_pos = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        self.base_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.kp = 40.0
        self.kd = 2.0

    def step(self, action_offset: np.ndarray, dt: float = 0.02) -> Dict[str, Any]:
        """Eylem ötelemesi uygulayarak tork ve sonraki durumu hesaplar."""
        q_des = self.nominal_q + 0.3 * action_offset
        torques = self.kp * (q_des - self.q) - self.kd * self.q_dot
        torques = np.clip(torques, -30.0, 30.0)

        # Basit durum entegrasyonu
        acc = torques / 0.5  # Basit bacak eylemsizliği
        self.q_dot += acc * dt
        self.q += self.q_dot * dt

        # Gövde ileri hareketi
        self.base_vel[0] = 0.9 * self.base_vel[0] + 0.1 * (1.2 - 0.1 * np.mean(np.abs(action_offset)))
        self.base_pos += self.base_vel * dt

        return {
            "q": self.q.tolist(),
            "q_dot": self.q_dot.tolist(),
            "torques": torques.tolist(),
            "base_vel": self.base_vel.tolist(),
            "base_pos": self.base_pos.tolist(),
        }
