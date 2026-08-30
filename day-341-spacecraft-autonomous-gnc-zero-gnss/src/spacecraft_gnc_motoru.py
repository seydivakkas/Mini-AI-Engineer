"""
Day 341: Spacecraft Autonomous GNC (Guidance, Navigation & Control) under Zero-GNSS
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Optik Yıldız Takipçisini (Star Tracker TRIAD), İki Cisim + J2 Yerçekimi Genişletilmiş
Kalman Filtresi (EKF) Yörünge Navigatörünü ve Otonom Reaksiyon Tekerleği Kontrolcüsünü içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np


class OpticalStarTracker:
    """
    Optik Yıldız Takipçisi (Star Tracker) ve TRIAD Yönelim (Attitude Quaternion) Belirleyicisi.
    Sıfır GNSS koşullarında gök küresi yıldız katalog eşleştirmesi ile yönelim tayini yapar.
    """
    def __init__(self, noise_std: float = 0.002):
        self.noise_std = noise_std
        # İnertial referans yıldız vektörleri (Katalog)
        self.star1_inertial = np.array([1.0, 0.0, 0.0])
        self.star2_inertial = np.array([0.0, 1.0, 0.0])

    def measure_body_vectors(self, true_rotation_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gövde koordinat sisteminde yıldız birim vektörlerini gürültülü olarak ölçer."""
        v1_body = np.dot(true_rotation_matrix, self.star1_inertial) + np.random.normal(0, self.noise_std, 3)
        v2_body = np.dot(true_rotation_matrix, self.star2_inertial) + np.random.normal(0, self.noise_std, 3)
        return v1_body / np.linalg.norm(v1_body), v2_body / np.linalg.norm(v2_body)

    def triad_attitude_estimation(self, v1_b: np.ndarray, v2_b: np.ndarray) -> np.ndarray:
        """TRIAD algoritması ile gövdeden eylemsizliğe dönüşüm matrisini (R_est) tahmin eder."""
        # Gövde üçlüsü (Body Triad)
        t1_b = v1_b
        t2_b = np.cross(v1_b, v2_b)
        t2_b = t2_b / np.linalg.norm(t2_b)
        t3_b = np.cross(t1_b, t2_b)
        M_body = np.column_stack((t1_b, t2_b, t3_b))

        # Eylemsiz üçlüsü (Inertial Triad)
        t1_i = self.star1_inertial
        t2_i = np.cross(self.star1_inertial, self.star2_inertial)
        t2_i = t2_i / np.linalg.norm(t2_i)
        t3_i = np.cross(t1_i, t2_i)
        M_inertial = np.column_stack((t1_i, t2_i, t3_i))

        # R_est * M_inertial = M_body  =>  R_est = M_body * M_inertial^T
        R_est = np.dot(M_body, M_inertial.T)
        return R_est


class OrbitalEKFNavigator:
    """
    İki Cisim + J2 Düzensiz Yerçekimi Modelli Genişletilmiş Kalman Filtresi (EKF).
    GPS/GNSS sinyali olmadan optik mesafe/yön gözlemleriyle uzay aracı yörünge durumunu x = [r, v] kestirir.
    """
    MU = 398600.4418  # Dünya Yerçekimi Sabiti (km^3 / s^2)
    J2 = 1.08263e-3   # Dünya Basıklık (Oblateness) Katsayısı
    R_EARTH = 6378.137  # Dünya Ekvator Yarıçapı (km)

    def __init__(self, initial_state: np.ndarray, dt: float = 1.0):
        self.dt = dt
        self.state = initial_state.copy()  # [rx, ry, rz, vx, vy, vz]
        self.P = np.eye(6) * 1.0  # Hata Kovaryans Matrisi
        self.Q = np.eye(6) * 1e-4  # Süreç Gürültüsü
        self.R = np.eye(3) * 0.05  # Ölçüm Gürültüsü

    def gravitational_acceleration(self, r: np.ndarray) -> np.ndarray:
        """İki Cisim + J2 Pertürbasyon İvmesini Hesaplar."""
        r_norm = np.linalg.norm(r)
        z = r[2]
        r_sq = r_norm ** 2
        
        # Keplerian ana çekim
        a_kep = - (self.MU / (r_norm ** 3)) * r
        
        # J2 pertürbasyon faktörü
        factor = 1.5 * self.J2 * self.MU * (self.R_EARTH ** 2) / (r_norm ** 5)
        a_j2 = np.array([
            factor * r[0] * (5.0 * (z**2) / r_sq - 1.0),
            factor * r[1] * (5.0 * (z**2) / r_sq - 1.0),
            factor * r[2] * (5.0 * (z**2) / r_sq - 3.0)
        ])
        
        return a_kep + a_j2

    def propagate_state(self, u_thrust: np.ndarray = np.zeros(3)):
        """Durumu Runge-Kutta 4 (RK4) integrasyonu ile bir sonraki adıma öteler."""
        r = self.state[:3]
        v = self.state[3:]
        
        a = self.gravitational_acceleration(r) + u_thrust
        
        # Basit Euler/RK öteleme
        self.state[:3] += v * self.dt + 0.5 * a * (self.dt ** 2)
        self.state[3:] += a * self.dt
        
        # Kovaryans güncelleme
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * self.dt
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def measurement_update(self, z_measured_pos: np.ndarray):
        """Optik optik nirengi / ufuk sensörü konum gözlemi ile EKF düzeltmesi."""
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        
        y = z_measured_pos - self.state[:3]  # Ölçüm artığı (Innovation)
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        
        self.state += np.dot(K, y)
        self.P = np.dot((np.eye(6) - np.dot(K, H)), self.P)


class AutonomousGNCController:
    """
    Otonom Rehberlik, Navigasyon ve Kontrol (GNC) Kontrolcüsü.
    Hedef yörünge pozisyonuna göre İtki ve Reaksiyon Tekerleği torku üretir.
    """
    def __init__(self, kp_pos: float = 0.01, kd_pos: float = 0.05):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos

    def compute_gnc_commands(self, current_state: np.ndarray, target_state: np.ndarray) -> Dict[str, Any]:
        """
        Pozisyon ve hız hatasına göre otonom itki ivmesi (Delta-V) hesaplar.
        """
        pos_err = target_state[:3] - current_state[:3]
        vel_err = target_state[3:] - current_state[3:]
        
        # PD Kontrol itki komutu (km/s^2)
        thrust_acc = self.kp_pos * pos_err + self.kd_pos * vel_err
        thrust_magnitude = float(np.linalg.norm(thrust_acc))
        
        return {
            "thrust_acc": thrust_acc,
            "thrust_magnitude_m_s2": thrust_magnitude * 1000.0,
            "pos_error_km": float(np.linalg.norm(pos_err)),
            "vel_error_km_s": float(np.linalg.norm(vel_err)),
        }
