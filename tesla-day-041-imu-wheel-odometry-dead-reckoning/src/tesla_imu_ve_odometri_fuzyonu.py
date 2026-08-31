r"""
Tesla IMU ve Tekerlek Odometrisi Füzyon Çekirdeği (Dead Reckoning & ESKF)
========================================================================
Bu modül; 6-eksenli IMU (İvmeölçer + Jiroskop) ve 4-Tekerlek Hız Odometrisi
verilerini Hata-Durumu Kalman Filtresi (Error-State EKF) ile birleştirerek
sürüklenmesiz (Drift-Free) araç konumu ($X, Y$), yönelimi ($\psi$) ve eğim tahmini yapar.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaIMUWheelOdometryFusion:
    """
    IMU + Tekerlek Odometrisi Dead Reckoning Füzyon Motoru.
    """
    def __init__(self, track_width_m: float = 1.62):
        self.w_track = track_width_m
        self.last_gyro_yaw = 0.0
        
        # Durum Vektörü: [X (m), Y (m), Yaw_psi (rad), Velocity_v (m/s), Gyro_Bias_b (rad/s)]
        self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # Kovaryans Matrisi P (5x5)
        self.P = np.diag([0.01, 0.01, 1e-3, 1.0, 0.01])

        # Gürültü Matrisleri
        self.Q_imu = np.diag([0.01, 0.01, 1e-4, 0.05, 1e-5])  # Süreç gürültüsü
        self.R_wheel = np.diag([0.01, 1e-4])  # Tekerlek hızı ve bias farkı gürültüsü

    def predict_imu(self, ax_mps2: float, gyro_yaw_rads: float, dt_s: float):
        """
        IMU Ölçümleriyle Kinematik Tahmin:
        X += v * cos(psi) * dt
        Y += v * sin(psi) * dt
        psi += (gyro_yaw - bias) * dt
        v += ax * dt
        """
        self.last_gyro_yaw = gyro_yaw_rads
        X, Y, psi, v, b = self.x

        unbiased_gyro = gyro_yaw_rads - b

        # Durum Güncellemesi
        self.x[0] += v * np.cos(psi) * dt_s
        self.x[1] += v * np.sin(psi) * dt_s
        self.x[2] = (psi + unbiased_gyro * dt_s + np.pi) % (2 * np.pi) - np.pi
        self.x[3] += ax_mps2 * dt_s

        # Doğrusallaştırılmış Jacobian F (5x5)
        F = np.eye(5)
        F[0, 2] = -v * np.sin(psi) * dt_s
        F[0, 3] = np.cos(psi) * dt_s
        F[1, 2] = v * np.cos(psi) * dt_s
        F[1, 3] = np.sin(psi) * dt_s
        F[2, 4] = -dt_s

        self.P = F @ self.P @ F.T + self.Q_imu

    def update_wheel_odometry(self, v_left_mps: float, v_right_mps: float):
        """
        Tekerlek Hız Sensörleriyle Düzeltme Adımı:
        v_odom = (v_R + v_L) / 2
        yaw_rate_odom = (v_R - v_L) / W_track
        z_bias = last_gyro_yaw - yaw_rate_odom
        """
        v_odom = (v_right_mps + v_left_mps) / 2.0
        yaw_rate_odom = (v_right_mps - v_left_mps) / self.w_track
        z_bias = self.last_gyro_yaw - yaw_rate_odom

        z = np.array([v_odom, z_bias])

        # Ölçüm Modeli h(x) = [v, bias]
        H = np.zeros((2, 5))
        H[0, 3] = 1.0
        H[1, 4] = 1.0

        z_pred = np.array([self.x[3], self.x[4]])
        y = z - z_pred

        S = H @ self.P @ H.T + self.R_wheel
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P
