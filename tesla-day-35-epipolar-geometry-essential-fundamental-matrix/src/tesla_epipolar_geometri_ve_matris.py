"""
Tesla Epipolar Geometri ve Çoklu Görüş Kalibrasyon Çekirdeği
============================================================
Bu modül; iki kamera arasındaki Göreli Rotasyon (R) ve Öteleme (t) vektörlerinden
Essential Matris (E) ve Fundamental Matris (F) hesaplamasını, 8-nokta algoritmasını
ve epipolar kısıt hata doğrulamasını (Sampson Distance) gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaEpipolarCalibrator:
    """
    Epipolar Geometri ve Çoklu Görüş Kalibratörü.
    """
    @staticmethod
    def skew_symmetric(t: np.ndarray) -> np.ndarray:
        """3D öteleme vektörünün çarpraz çarpım matrisi: [t]x."""
        return np.array([
            [0.0, -t[2], t[1]],
            [t[2], 0.0, -t[0]],
            [-t[1], t[0], 0.0]
        ], dtype=np.float64)

    @staticmethod
    def compute_essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """E = [t]x @ R."""
        t_skew = TeslaEpipolarCalibrator.skew_symmetric(t)
        return t_skew @ R

    @staticmethod
    def compute_fundamental_matrix(K1: np.ndarray, K2: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """F = inv(K2).T @ E @ inv(K1)."""
        E = TeslaEpipolarCalibrator.compute_essential_matrix(R, t)
        inv_K2_T = np.linalg.inv(K2).T
        inv_K1 = np.linalg.inv(K1)
        F = inv_K2_T @ E @ inv_K1
        return F / np.linalg.norm(F)  # Ölçek normalizasyonu

    @staticmethod
    def compute_epipolar_line(F: np.ndarray, pt_cam1: np.ndarray) -> np.ndarray:
        """Kamera 1'deki bir noktanın Kamera 2'deki epipolar doğrusu: l2 = F @ x1."""
        x1_homog = np.array([pt_cam1[0], pt_cam1[1], 1.0], dtype=np.float64)
        l2 = F @ x1_homog
        # Çizgi katsayılarını normalleştir: sqrt(a^2 + b^2) = 1
        norm = np.hypot(l2[0], l2[1])
        return l2 / max(norm, 1e-12)

    @staticmethod
    def compute_sampson_distance(F: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> float:
        """
        Sampson Epipolar Geometrik Hata Mesafesi (Piksel cinsinden).
        d = (x2^T F x1)^2 / ((F x1)[0]^2 + (F x1)[1]^2 + (F^T x2)[0]^2 + (F^T x2)[1]^2)
        """
        x1 = np.array([pt1[0], pt1[1], 1.0])
        x2 = np.array([pt2[0], pt2[1], 1.0])

        Fx1 = F @ x1
        FTx2 = F.T @ x2
        numerator = (x2.T @ Fx1) ** 2
        denominator = Fx1[0]**2 + Fx1[1]**2 + FTx2[0]**2 + FTx2[1]**2
        return float(np.sqrt(numerator / max(denominator, 1e-12)))

    @staticmethod
    def estimate_fundamental_8point(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Normalized 8-Nokta Algoritması ile F matrisi kestirimi ve Rank-2 zorlaması.
        """
        assert len(pts1) >= 8 and len(pts2) >= 8

        # A matrisi oluşturma: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
        A = []
        for (u1, v1), (u2, v2) in zip(pts1, pts2):
            A.append([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1.0])
        A = np.array(A)

        # SVD çözümü
        _, _, Vt = np.linalg.svd(A)
        F_raw = Vt[-1].reshape(3, 3)

        # Rank-2 Zorlaması: En küçük tekil değeri sıfırla
        U, S, Vt_f = np.linalg.svd(F_raw)
        S[-1] = 0.0
        F_rank2 = U @ np.diag(S) @ Vt_f
        return F_rank2 / np.linalg.norm(F_rank2)
