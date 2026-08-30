"""
Day 385: Sub-Millimeter Precision Microsurgery Robot (Vascular Anastomosis & Tremor Cancellation)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Mikro-Cerrahi Robotunun İnsan El Titremesi Sönümleme Filtresini (8-12 Hz Dijital Çentik / Kalman),
0.8 mm Damar Dikiş (Vasküler Anastomoz) Yörünge Planlayıcısını,
ve Doku Yırtılmasını Önleyen Empedans Kuvvet Geri Besleme Kontrolcüsünü içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class NeedleState:
    """Mikro-Cerrahi İğnesi ve Uç Elemanı Durum Modeli."""
    pos_mm: np.ndarray
    vel_mm_s: np.ndarray
    force_n: float = 0.0
    needle_angle_rad: float = 0.0
    is_punctured: bool = False
    tissue_stress_kpa: float = 0.0


class TremorCancellationKalmanFilter:
    """
    Cerrah El Titremesi Aktif Sönümleme Filtresi (8-12 Hz Biyomekanik Çentik ve Durum Kestirimi).
    """
    def __init__(self, dt_s: float = 0.02, f_tremor_hz: float = 10.0, r_pole: float = 0.60):
        self.dt = dt_s
        self.f0 = f_tremor_hz
        self.r = r_pole
        
        w0 = 2.0 * np.pi * self.f0 * self.dt
        self.b0 = 1.0
        self.b1 = -2.0 * np.cos(w0)
        self.b2 = 1.0

        self.a1 = -2.0 * self.r * np.cos(w0)
        self.a2 = self.r ** 2

        gain_dc = (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2)
        if abs(gain_dc) > 1e-5:
            self.b0 /= gain_dc
            self.b1 /= gain_dc
            self.b2 /= gain_dc

        self.x_hist = [0.0, 0.0]
        self.y_hist = [0.0, 0.0]
        self.initialized = False

    def reset(self, initial_pos_mm: float):
        self.x_hist = [initial_pos_mm, initial_pos_mm]
        self.y_hist = [initial_pos_mm, initial_pos_mm]
        self.initialized = True

    def filter_position(self, raw_pos_mm: float) -> float:
        if not self.initialized:
            self.reset(raw_pos_mm)
            return float(raw_pos_mm)

        y = self.b0 * raw_pos_mm + self.b1 * self.x_hist[0] + self.b2 * self.x_hist[1] - self.a1 * self.y_hist[0] - self.a2 * self.y_hist[1]

        self.x_hist[1] = self.x_hist[0]
        self.x_hist[0] = raw_pos_mm
        self.y_hist[1] = self.y_hist[0]
        self.y_hist[0] = y

        return float(y)


class VascularAnastomosisPlanner:
    """
    Milimetre-Altı (0.8 mm) Vasküler Anastomoz Dairesel İğne Yörünge Planlayıcısı.
    """
    def __init__(self, vessel_radius_mm: float = 0.4, needle_radius_mm: float = 1.2):
        self.v_rad = vessel_radius_mm
        self.n_rad = needle_radius_mm

    def generate_stitch_trajectory(self, num_points: int = 100) -> List[np.ndarray]:
        thetas = np.linspace(-np.pi * 0.6, np.pi * 0.6, num_points)
        path = []
        for th in thetas:
            x = self.n_rad * np.sin(th)
            y = self.v_rad + (self.n_rad * (1.0 - np.cos(th))) - 0.2
            z = 0.05 * th
            path.append(np.array([x, y, z], dtype=np.float64))
        return path


class ImpedanceForceFeedbackController:
    """
    Doku Koruyucu Empedans ve Kuvvet Geri Besleme Kontrolcüsü.
    """
    def __init__(self, k_tissue: float = 0.35, d_tissue: float = 0.05, max_safe_force_n: float = 0.25):
        self.k_e = k_tissue
        self.d_e = d_tissue
        self.max_force = max_safe_force_n
        self.puncture_threshold_n = 0.085

    def compute_interaction(self, current_pos: np.ndarray, target_pos: np.ndarray, vel: np.ndarray) -> Tuple[float, float, bool]:
        disp = float(np.linalg.norm(current_pos - target_pos))
        speed = float(np.linalg.norm(vel))

        raw_force = self.k_e * disp + self.d_e * speed
        is_punctured = bool(raw_force > self.puncture_threshold_n)

        if is_punctured:
            contact_force = self.puncture_threshold_n * 0.4 + 0.02 * speed
        else:
            contact_force = min(self.max_force, raw_force)

        needle_tip_area_mm2 = np.pi * (0.025 ** 2)
        tissue_stress_kpa = (contact_force / needle_tip_area_mm2) * 1e-3

        return float(contact_force), float(tissue_stress_kpa), bool(is_punctured)


class MicrosurgeryBenchmark:
    """
    Mikro-Cerrahi Robotik Dikiş ve Titreme Sönümleme Başarım Paketi.
    """
    def __init__(self):
        self.filter_x = TremorCancellationKalmanFilter(dt_s=0.02, f_tremor_hz=10.0, r_pole=0.55)
        self.filter_y = TremorCancellationKalmanFilter(dt_s=0.02, f_tremor_hz=10.0, r_pole=0.55)
        self.planner = VascularAnastomosisPlanner()
        self.impedance = ImpedanceForceFeedbackController()

    def run_benchmark(self, num_steps: int = 100) -> Dict[str, Any]:
        np.random.seed(42)
        ideal_path = self.planner.generate_stitch_trajectory(num_points=num_steps)

        self.filter_x.reset(ideal_path[0][0])
        self.filter_y.reset(ideal_path[0][1])

        raw_hand_path = []
        filtered_robot_path = []
        forces = []
        stresses = []
        tracking_errors_um = []

        for i in range(num_steps):
            target = ideal_path[i]
            t = i * 0.02

            # 10 Hz titreme (~80 mikrometre)
            tremor_x = 0.08 * np.sin(2.0 * np.pi * 10.0 * t) + np.random.normal(0, 0.002)
            tremor_y = 0.07 * np.cos(2.0 * np.pi * 10.0 * t) + np.random.normal(0, 0.002)

            raw_pos = target + np.array([tremor_x, tremor_y, 0.0])
            raw_hand_path.append(raw_pos)

            clean_x = self.filter_x.filter_position(raw_pos[0])
            clean_y = self.filter_y.filter_position(raw_pos[1])
            robot_pos = np.array([clean_x, clean_y, target[2]])
            filtered_robot_path.append(robot_pos)

            err_um = np.linalg.norm(robot_pos - target) * 1000.0
            tracking_errors_um.append(err_um)

            vel = (robot_pos - filtered_robot_path[-2]) / 0.02 if i > 0 else np.zeros(3)
            f_n, stress_kpa, _ = self.impedance.compute_interaction(robot_pos, target, vel)
            forces.append(f_n)
            stresses.append(stress_kpa)

        raw_err_mean = np.mean([np.linalg.norm(raw_hand_path[k] - ideal_path[k]) * 1000.0 for k in range(num_steps)])
        robot_err_mean = np.mean(tracking_errors_um)
        tremor_attenuation_pct = max(0.0, (1.0 - (robot_err_mean / raw_err_mean)) * 100.0)

        max_force = float(np.max(forces))
        safe_tissue = bool(max_force < 0.25)

        return {
            "num_steps": num_steps,
            "avg_positioning_error_um": round(float(robot_err_mean), 2),
            "raw_hand_error_um": round(float(raw_err_mean), 2),
            "tremor_attenuation_pct": round(float(tremor_attenuation_pct), 2),
            "max_contact_force_n": round(max_force, 4),
            "tissue_integrity_safe": safe_tissue,
            "submillimeter_precision_pass": bool(robot_err_mean < 25.0),
            "ideal_path": np.array(ideal_path),
            "raw_hand_path": np.array(raw_hand_path),
            "filtered_robot_path": np.array(filtered_robot_path),
            "forces": forces,
            "stresses": stresses,
            "tracking_errors_um": tracking_errors_um
        }

    def kos(self, num_steps: int = 100) -> Dict[str, Any]:
        return self.run_benchmark(num_steps)
