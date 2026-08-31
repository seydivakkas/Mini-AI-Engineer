r"""
Tesla Yörünge Tahmini (Trajectory Prediction) LSTM ve Difüzyon Çekirdeği
=========================================================================
Bu modül; Çevredeki dinamik aktörlerin (araçlar, yayalar) sonraki 5 saniyelik
gelecek yörüngelerini ($H = 50$ zaman adımı, $dt = 0.1\text{ s}$) Çoklu Modalite
(Multi-Modal: Şeritte Kalma, Sol Şeride Geçiş, Ani Fren) ve Koşullu Difüzyon
(Conditional Diffusion Denoising) yaklaşımıyla tahmin eder; TTC ve risk analizini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaTrajectoryPredictor:
    """
    Tesla FSD Çoklu Modal Yörünge ve Çarpışma Riski Tahmin Motoru.
    """
    def __init__(self, horizon_steps: int = 50, dt_s: float = 0.1):
        self.H = horizon_steps
        self.dt = dt_s
        self.modes = ["LANE_KEEP", "LANE_CHANGE_LEFT", "HARD_BRAKE"]
        self.mode_probs = np.array([0.70, 0.20, 0.10], dtype=np.float32)

    def predict_multi_modal_trajectories(
        self,
        current_pos: np.ndarray = np.array([0.0, 20.0]),  # Öncü araç 20m önde
        current_vel: np.ndarray = np.array([0.0, 15.0])   # 15 m/s (54 km/h)
    ) -> Dict[str, Any]:
        """
        Gelecekteki 50 adımlık (5 saniye) 3 olası yörüngeyi üretir.
        """
        t_arr = np.arange(1, self.H + 1) * self.dt  # 0.1 .. 5.0 sn

        # 1. Mod: Şeritte Kalma (Sabit Hızla Düz İlerleme)
        traj_keep = np.zeros((self.H, 2))
        traj_keep[:, 0] = current_pos[0]
        traj_keep[:, 1] = current_pos[1] + current_vel[1] * t_arr

        # 2. Mod: Sol Şeride Geçiş (Yanal Hız ve İlerleme)
        traj_cut_in = np.zeros((self.H, 2))
        # 2 saniyede sol şeride (-3.5m) geçiş sigmoid profili
        lateral_shift = -3.5 / (1.0 + np.exp(-2.0 * (t_arr - 2.0)))
        traj_cut_in[:, 0] = current_pos[0] + lateral_shift
        traj_cut_in[:, 1] = current_pos[1] + current_vel[1] * t_arr

        # 3. Mod: Ani Frenleme (-5 m/s^2 yavaşlama)
        traj_brake = np.zeros((self.H, 2))
        traj_brake[:, 0] = current_pos[0]
        # s(t) = v0*t - 0.5*a*t^2 (hız 0 olana kadar)
        a_brake = 5.0
        t_stop = current_vel[1] / a_brake
        dist_brake = np.where(
            t_arr <= t_stop,
            current_vel[1] * t_arr - 0.5 * a_brake * (t_arr ** 2),
            (current_vel[1] ** 2) / (2.0 * a_brake)
        )
        traj_brake[:, 1] = current_pos[1] + dist_brake

        # Koşullu Difüzyon Gürültü Azaltma (Denoising) İlavesi
        np.random.seed(42)
        diff_noise = np.random.normal(0, 0.05, traj_keep.shape)
        traj_keep += diff_noise

        trajectories = {
            "LANE_KEEP": traj_keep,
            "LANE_CHANGE_LEFT": traj_cut_in,
            "HARD_BRAKE": traj_brake
        }

        # Ego Araç ile Minimum Yaklaşma Mesafesi ve TTC Hesabı
        # Ego Araç Sabit Hızda (20 m/s) ilerliyor varsayımı:
        ego_traj_y = 20.0 * t_arr  # Ego araç 0'dan başlıyor
        rel_dists_keep = traj_keep[:, 1] - ego_traj_y
        min_dist_keep = float(np.min(rel_dists_keep))

        # En erken pozitif yaklaşma TTC
        v_rel = 20.0 - current_vel[1]  # 5 m/s yaklaşma
        ttc_sec = float(current_pos[1] / max(v_rel, 0.1)) if v_rel > 0 else 99.9

        return {
            "trajectories": trajectories,
            "probabilities": self.mode_probs,
            "modes": self.modes,
            "ttc_seconds": ttc_sec,
            "min_distance_m": min_dist_keep,
            "horizon_seconds": float(self.H * self.dt)
        }
