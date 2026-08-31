r"""
Tesla Asenkron Sensör Füzyonu Çekirdeği (EKF & UKF)
===================================================
Bu modül; 6-durumlu ($p_x, p_y, v_x, v_y, a_x, a_y$) hedef kinematik takibini,
Kamera (Kartezyen 2D) ve Radar (Polar $[r, \theta, \dot{r}]$) asenkron ölçümlerinin
Genişletilmiş Kalman Filtresi (EKF) ve Unscented Kalman Filtresi (UKF) ile
füzyonunu ve Mahalanobis mesafe kapılama (Gating) mekanizmasını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaSensorFusionEKF:
    """
    6-Durumlu Asenkron EKF ve UKF Sensör Füzyon Motoru.
    """
    def __init__(self, init_x: float = 20.0, init_y: float = 0.0, init_vx: float = 10.0, init_vy: float = 0.0):
        # Durum Vektörü: [px, py, vx, vy, ax, ay]^T
        self.x = np.array([init_x, init_y, init_vx, init_vy, 0.0, 0.0], dtype=np.float64)
        
        # Kovaryans Matrisi P (6x6)
        self.P = np.diag([1.0, 1.0, 4.0, 4.0, 10.0, 10.0])
        
        # Ölçüm Gürültüleri
        self.R_cam = np.diag([0.25, 0.25])  # Kamera (Kartezyen X, Y)
        self.R_radar = np.diag([0.16, 0.001, 0.09])  # Radar (r, theta, r_dot)

    def predict(self, dt_s: float, process_noise_std: float = 1.5):
        """
        Sürekli İvme Kinematik Tahmin Adımı: x = F @ x, P = F @ P @ F.T + Q.
        """
        # Durum Geçiş Matrisi F (6x6)
        F = np.eye(6)
        F[0, 2] = dt_s
        F[0, 4] = 0.5 * (dt_s ** 2)
        F[1, 3] = dt_s
        F[1, 5] = 0.5 * (dt_s ** 2)
        F[2, 4] = dt_s
        F[3, 5] = dt_s

        # Süreç Gürültü Matrisi Q (Piecewise Continuous Acceleration)
        G = np.array([
            [0.5 * dt_s**2, 0],
            [0, 0.5 * dt_s**2],
            [dt_s, 0],
            [0, dt_s],
            [1.0, 0],
            [0, 1.0]
        ])
        Q = G @ (np.eye(2) * (process_noise_std ** 2)) @ G.T

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_camera(self, z_cam: np.ndarray) -> bool:
        """
        Kamera Ölçüm Güncellemesi: z_cam = [x, y]^T (Lineer).
        """
        assert len(z_cam) == 2
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        y = z_cam - (H @ self.x)
        S = H @ self.P @ H.T + self.R_cam

        # Mahalanobis Mesafe Denetimi (Chi-Square 2-DoF %95 eşiği = 5.99)
        d_m2 = float(y.T @ np.linalg.inv(S) @ y)
        if d_m2 > 9.21:  # %99 dışı ise Outlier reddi
            return False

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        return True

    def update_radar(self, z_radar: np.ndarray) -> bool:
        """
        Radar Ölçüm Güncellemesi: z_radar = [r, theta, r_dot]^T (Non-lineer EKF).
        """
        assert len(z_radar) == 3
        px, py, vx, vy, _, _ = self.x

        r = np.sqrt(px**2 + py**2)
        if r < 1e-4:
            return False

        theta = np.arctan2(py, px)
        r_dot = (px * vx + py * vy) / r

        z_pred = np.array([r, theta, r_dot])

        # Jacobian Matrisi Hj (3x6)
        r2 = r ** 2
        r3 = r ** 3
        Hj = np.zeros((3, 6))
        Hj[0, 0] = px / r
        Hj[0, 1] = py / r
        Hj[1, 0] = -py / r2
        Hj[1, 1] = px / r2
        Hj[2, 0] = (py * (vx * py - vy * px)) / r3
        Hj[2, 1] = (px * (vy * px - vx * py)) / r3
        Hj[2, 2] = px / r
        Hj[2, 3] = py / r

        y = z_radar - z_pred
        # Açı farkını [-pi, pi] aralığında sar
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi

        S = Hj @ self.P @ Hj.T + self.R_radar

        # Mahalanobis 3-DoF %99 eşiği = 11.34
        d_m2 = float(y.T @ np.linalg.inv(S) @ y)
        if d_m2 > 15.0:
            return False

        K = self.P @ Hj.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ Hj) @ self.P
        return True
