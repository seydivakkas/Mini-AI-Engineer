"""
Çift Kollu (Bimanual) Robot Koordinasyon ve Bağıl Jakoben Motoru (Day 250).
Kapalı Kinematik Zincir, Mutlak/Bağıl Hareket Ayrışımı ve Eşzamanlı Nesne Taşıma.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class SingleArmKinematics:
    """Tekil 7-DoF Robot Kolu İleri ve Ters Kinematik Modeli."""

    def __init__(self, taban_konumu: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.taban = np.array(taban_konumu, dtype=np.float64)
        self.linkler = [0.12, 0.15, 0.15, 0.12, 0.12, 0.08, 0.05]
        self.dof = len(self.linkler)

    def forward_kinematics(self, eklemler: np.ndarray) -> np.ndarray:
        """Eklem açılarından dünya koordinat sisteminde uç nokta (EEF) konumunu bulur."""
        x = self.taban[0]
        y = self.taban[1]
        z = self.taban[2] + 0.1  # Taban yüksekliği

        kumulatif_yaw = 0.0
        kumulatif_pitch = 0.0

        for i in range(min(len(eklemler), self.dof)):
            aci = eklemler[i]
            if i % 2 == 0:
                kumulatif_yaw += aci
            else:
                kumulatif_pitch += aci

            L = self.linkler[i]
            x += L * np.cos(kumulatif_yaw) * np.cos(kumulatif_pitch)
            y += L * np.sin(kumulatif_yaw) * np.cos(kumulatif_pitch)
            z += L * np.sin(kumulatif_pitch)

        return np.array([x, y, z], dtype=np.float64)

    def inverse_kinematics(self, hedef_pos: np.ndarray, q0: np.ndarray = None, max_iter: int = 40) -> np.ndarray:
        """Ters Kinematik çözücüsü (L-BFGS-B Sayısal Optimizasyon)."""
        from scipy.optimize import minimize

        if q0 is None or np.all(q0 == 0) or np.linalg.norm(q0) < 0.05:
            q_init = np.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.2, 0.0])
        else:
            q_init = q0.copy().astype(np.float64)

        def loss_fn(q):
            pos = self.forward_kinematics(q)
            return float(np.sum((pos - hedef_pos) ** 2))

        res = minimize(
            loss_fn,
            q_init,
            method="SLSQP",
            bounds=[(-np.pi, np.pi)] * self.dof,
            options={"maxiter": max_iter},
        )
        return res.x


class BimanualDualArmSystem:
    """Çift Kollu (Sol + Sağ) Kapalı Kinematik Zincir Sistemi."""

    def __init__(self, nesne_genisligi_m: float = 0.30, taban_mesafesi_m: float = 0.50):
        self.d_obj = nesne_genisligi_m
        # Sol Kol: (-0.25, 0, 0), Sağ Kol: (+0.25, 0, 0)
        yarim_taban = taban_mesafesi_m / 2.0
        self.left_arm = SingleArmKinematics(taban_konumu=(-yarim_taban, 0.0, 0.0))
        self.right_arm = SingleArmKinematics(taban_konumu=(yarim_taban, 0.0, 0.0))
        self.k_stiffness = 500.0  # N/m esneklik katsayısı

    def compute_bimanual_metrics(self, q_left: np.ndarray, q_right: np.ndarray) -> Dict[str, Any]:
        """Mutlak nesne konumu, bağıl mesafe ve iç yıkıcı kuvveti hesaplar."""
        p_L = self.left_arm.forward_kinematics(q_left)
        p_R = self.right_arm.forward_kinematics(q_right)

        # Mutlak Nesne Konumu: x_abs = 0.5 * (p_L + p_R)
        x_abs = 0.5 * (p_L + p_R)

        # Bağıl Vektör: x_rel = p_L - p_R
        x_rel = p_L - p_R
        mevcut_mesafe = float(np.linalg.norm(x_rel))

        # Mesafe Sapması ve İç Gerilim Kuvveti (Internal Stress Force)
        mesafe_sapmasi_m = abs(mevcut_mesafe - self.d_obj)
        ic_gerilim_kuvveti_N = round(float(self.k_stiffness * mesafe_sapmasi_m), 2)

        return {
            "p_left": p_L.tolist(),
            "p_right": p_R.tolist(),
            "x_abs_nesne": x_abs.round(4).tolist(),
            "mevcut_mesafe_m": round(mevcut_mesafe, 4),
            "mesafe_sapmasi_mm": round(mesafe_sapmasi_m * 1000.0, 2),
            "ic_gerilim_kuvveti_N": ic_gerilim_kuvveti_N,
            "senkronize_mi": mesafe_sapmasi_m < 0.01,  # 10mm tolerans
        }


class BimanualTrajectoryPlanner:
    """Çift Kol İçin Bağıl Jakoben Tabanlı Senkronize Yörünge Üreticisi."""

    @classmethod
    def generate_coordinated_trajectory(
        cls,
        dual_system: BimanualDualArmSystem,
        start_obj_pos: np.ndarray,
        goal_obj_pos: np.ndarray,
        steps: int = 10,
    ) -> List[Dict[str, Any]]:
        """Nesneyi ezmeden ve düşürmeden iki kolla senkron taşıma adımlarını üretir."""
        half_w = dual_system.d_obj / 2.0
        trajectory = []

        q_L = np.zeros(7)
        q_R = np.zeros(7)

        for step in range(steps + 1):
            alpha = step / float(steps)
            # İnterpole Edilen Nesne Konumu
            current_obj = (1.0 - alpha) * start_obj_pos + alpha * goal_obj_pos

            # Sol ve Sağ Kol Hedef Uç Noktaları (Sabit d_obj mesafesi)
            target_L = current_obj + np.array([-half_w, 0.0, 0.0])
            target_R = current_obj + np.array([half_w, 0.0, 0.0])

            # İki Kolun IK Çözümleri
            q_L = dual_system.left_arm.inverse_kinematics(target_L, q0=q_L)
            q_R = dual_system.right_arm.inverse_kinematics(target_R, q0=q_R)

            metrikler = dual_system.compute_bimanual_metrics(q_L, q_R)
            trajectory.append({
                "adim": step,
                "hedef_obj_pos": current_obj.round(3).tolist(),
                "q_left": q_L.round(3).tolist(),
                "q_right": q_R.round(3).tolist(),
                "metrikler": metrikler,
            })

        return trajectory
