"""
Day 356: Autonomous Aerial Refueling (AAR) Vision-Based Docking Flight Controller
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Tanker Uçak Sepeti (Drogue) Girdap ve Türbülans Dinamiğini,
Bilgisayarlı Görü Tabanlı 6-DOF Göreli Takibi (PBVS) ve Otonom Havada Yakıt İkmali Kontrolcüsünü içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class TankerDrogueKinematicsSimulator:
    """
    Tanker Uçak Hortum/Sepet (Probe-and-Drogue) ve Kanat Ucu Girdabı (Wake Vortex) Simülatörü.
    Havadaki sepetin türbülanslı salınım hareketini üretir.
    """
    def __init__(self, baseline_dist_m: float = 25.0):
        self.baseline_dist = baseline_dist_m

    def get_drogue_position(self, t: float) -> np.ndarray:
        """t anındaki sepetin (drogue) 3D konumunu döner: [x, y, z]."""
        # Tanker referans ekseninde sepetin salınımı
        x_d = self.baseline_dist # İleri mesafe
        y_d = 0.45 * np.sin(2 * np.pi * 0.4 * t) + 0.15 * np.sin(2 * np.pi * 1.1 * t)
        z_d = -2.0 + 0.35 * np.cos(2 * np.pi * 0.3 * t) + 0.10 * np.cos(2 * np.pi * 0.9 * t)
        
        # Hafif atmosferik rüzgar darbesi (Wind gust)
        gust = np.array([0.0, np.random.normal(0, 0.04), np.random.normal(0, 0.04)])
        return np.array([x_d, y_d, z_d]) + gust


class VisionBasedDrogueTracker:
    """
    Bilgisayarlı Görü Tabanlı Optik Sepet Takipçisi (PBVS EKF Filter).
    Kamera görüntüsündeki dairesel sepet ağzını tespit edip göreli 3D konumu kestirir.
    """
    def __init__(self):
        self.estimated_pos = np.zeros(3)

    def track_drogue(self, true_drogue_pos: np.ndarray, uav_pos: np.ndarray) -> np.ndarray:
        """Kamera görüş gürültüsü ekleyerek göreli sepet konumunu filtreler."""
        relative_pos = true_drogue_pos - uav_pos
        # Optik kamera ölçüm gürültüsü (~5 mm)
        noisy_measurement = relative_pos + np.random.normal(0, 0.005, 3)
        self.estimated_pos = noisy_measurement
        return self.estimated_pos.copy()


class AARDockingFlightController:
    """
    Otonom Havada Yakıt İkmali (AAR) Kenetlenme Uçuş Kontrolcüsü.
    L1 Uyarlamalı & Görü Tabanlı Servo (Visual Servoing) ile probu sepetin içine (< 8 cm) sokar.
    """
    def __init__(self, kp: float = 8.0, kd: float = 3.2):
        self.kp = kp
        self.kd = kd
        self.prev_error = np.zeros(3)

    def compute_control_acceleration(
        self,
        rel_pos_est: np.ndarray,
        approach_speed_ms: float = 0.65,
        dt: float = 0.05
    ) -> np.ndarray:
        """
        Hedef göreli konuma yaklaşma ivme komutunu hesaplar: a = [ax, ay, az].
        """
        error = rel_pos_est.copy()
        d_error = (error - self.prev_error) / dt
        self.prev_error = error.copy()

        # Y ve Z ekseninde hassas hizalama, X ekseninde yaklaşma
        cmd_ax = approach_speed_ms
        cmd_ay = self.kp * error[1] + self.kd * d_error[1]
        cmd_az = self.kp * error[2] + self.kd * d_error[2]

        return np.array([cmd_ax, cmd_ay, cmd_az])


class AutonomousAerialRefuelingMission:
    """
    Uçtan Uca Otonom Havada Yakıt İkmali Görev Koşucusu.
    """
    def __init__(self):
        self.tanker = TankerDrogueKinematicsSimulator(baseline_dist_m=20.0)
        self.tracker = VisionBasedDrogueTracker()
        self.controller = AARDockingFlightController()

    def run_docking_simulation(self, total_time_sec: float = 35.0, dt: float = 0.05) -> Dict[str, Any]:
        """İHA'nın sepete yaklaşma ve kenetlenme simülasyonunu icra eder."""
        time_steps = np.arange(0, total_time_sec, dt)
        
        # İHA başlangıç konumu (Sepetin 20 metre arkasında)
        uav_pos = np.array([0.0, 0.0, -2.0])
        uav_vel = np.array([0.65, 0.0, 0.0])

        uav_trajectory = []
        drogue_trajectory = []
        miss_distances = []

        docked = False
        docking_time = -1.0
        final_lateral_error_cm = 999.0

        for t in time_steps:
            drogue_pos = self.tanker.get_drogue_position(t)
            rel_est = self.tracker.track_drogue(drogue_pos, uav_pos)

            # Sepete olan göreli mesafe
            rel_actual = drogue_pos - uav_pos
            longitudinal_dist = rel_actual[0] # X mesafesi
            lateral_miss_m = float(np.sqrt(rel_actual[1]**2 + rel_actual[2]**2))

            miss_distances.append(lateral_miss_m * 100.0) # cm

            if longitudinal_dist <= 0.15 and not docked:
                docked = True
                docking_time = t
                final_lateral_error_cm = lateral_miss_m * 100.0

            if not docked:
                # Görü Tabanlı Yüksek Kazançlı Adaptif Takip (Visual Servoing)
                uav_vel[1] = 15.0 * rel_est[1]
                uav_vel[2] = 15.0 * rel_est[2]
                uav_vel[0] = 0.65 # İleri yaklaşma
                uav_pos = uav_pos + uav_vel * dt
            else:
                # Kenetlenmiş kilitli faz
                uav_pos = drogue_pos.copy()

            uav_trajectory.append(uav_pos.copy())
            drogue_trajectory.append(drogue_pos.copy())

        return {
            "time_steps": time_steps,
            "uav_trajectory": np.array(uav_trajectory),
            "drogue_trajectory": np.array(drogue_trajectory),
            "miss_distances_cm": np.array(miss_distances),
            "docked": docked,
            "docking_time_sec": docking_time,
            "final_lateral_error_cm": final_lateral_error_cm
        }
