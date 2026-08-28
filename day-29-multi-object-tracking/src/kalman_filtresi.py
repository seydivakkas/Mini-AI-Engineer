"""
8 Boyutlu Kalman Filtresi Durum Kestirimi ve Mahalanobis Kapılama (Gating).
State Vector: [u, v, gamma, h, u_dot, v_dot, gamma_dot, h_dot]
"""

from typing import Tuple
import numpy as np
import scipy.linalg


class KalmanKutuFiltresi:
    """
    Kutunun merkez (u, v), en-boy oranı gamma (w/h), yükseklik (h) ve hızlarını
    doğrusal sabit hızlı (Constant Velocity) hareket modeliyle takip eder.
    """

    # 4 serbestlik dereceli ki-kare %95 güven aralığı eşiği
    MAHALANOBIS_ESIK_095 = 9.4877

    def __init__(self):
        self._dt = 1.0

        # Durum Geçiş Matrisi F (8x8)
        self._F = np.eye(8, 8)
        for i in range(4):
            self._F[i, i + 4] = self._dt

        # Ölçüm Matrisi H (4x8)
        self._H = np.eye(4, 8)

        # Standart sapma katsayıları
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def ilklendir(self, olcum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        İlk tespit [x1, y1, x2, y2] veya [u, v, gamma, h] ile durum vektörü x ve kovaryans P başlatır.
        """
        u, v, gamma, h = self.kutu_to_uvgh(olcum)

        x = np.zeros(8)
        x[:4] = [u, v, gamma, h]

        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        P = np.diag(np.square(std))
        return x, P

    def tahmin(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tahmin Adımı (Predict):
            x' = F * x
            P' = F * P * F^T + Q
        """
        h = x[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        x_tahmin = np.dot(self._F, x)
        P_tahmin = np.linalg.multi_dot([self._F, P, self._F.T]) + Q
        return x_tahmin, P_tahmin

    def guncelle(self, x: np.ndarray, P: np.ndarray, olcum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ölçüm Güncelleme Adımı (Update):
            y = z - H * x
            S = H * P * H^T + R
            K = P * H^T * S^-1
            x = x + K * y
            P = (I - K * H) * P
        """
        z = self.kutu_to_uvgh(olcum)
        h = x[3]
        std_r = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        R = np.diag(np.square(std_r))

        y = z - np.dot(self._H, x)
        S = np.linalg.multi_dot([self._H, P, self._H.T]) + R

        # Kararlı Kalman Kazancı Hesabı (Cholesky Faktörizasyonu)
        chol_factor, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve((chol_factor, lower), np.dot(self._H, P), check_finite=False).T

        x_yeni = x + np.dot(K, y)
        P_yeni = P - np.linalg.multi_dot([K, S, K.T])
        return x_yeni, P_yeni

    def mahalanobis_mesafesi(self, x: np.ndarray, P: np.ndarray, olcumler: np.ndarray) -> np.ndarray:
        """
        Durum x ile bir dizi tespit z_j arasındaki Mahalanobis kapılama mesafesini hesaplar:
        d^2 = (z - H*x)^T * S^-1 * (z - H*x)
        """
        h = x[3]
        std_r = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        R = np.diag(np.square(std_r))
        S = np.linalg.multi_dot([self._H, P, self._H.T]) + R
        chol_factor, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)

        z_matrisi = np.array([self.kutu_to_uvgh(b) for b in olcumler])
        fark = z_matrisi - np.dot(self._H, x)

        # S^-1 * fark^T
        sol_fark = scipy.linalg.cho_solve((chol_factor, lower), fark.T, check_finite=False)
        kare_mesafeler = np.sum(fark.T * sol_fark, axis=0)
        return kare_mesafeler

    @staticmethod
    def kutu_to_uvgh(kutu: np.ndarray) -> np.ndarray:
        """[x1, y1, x2, y2] -> [u, v, gamma, h]"""
        if len(kutu) == 4 and kutu[2] > kutu[0] and kutu[3] > kutu[1]:
            w = kutu[2] - kutu[0]
            h = kutu[3] - kutu[1]
            u = kutu[0] + w / 2.0
            v = kutu[1] + h / 2.0
            gamma = w / (h + 1e-6)
            return np.array([u, v, gamma, h])
        return kutu.copy()

    @staticmethod
    def uvgh_to_kutu(uvgh: np.ndarray) -> np.ndarray:
        """[u, v, gamma, h] -> [x1, y1, x2, y2]"""
        u, v, gamma, h = uvgh[:4]
        w = gamma * h
        x1 = u - w / 2.0
        y1 = v - h / 2.0
        x2 = u + w / 2.0
        y2 = v + h / 2.0
        return np.array([x1, y1, x2, y2])
