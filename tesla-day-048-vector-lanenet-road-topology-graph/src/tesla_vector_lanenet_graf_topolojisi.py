r"""
Tesla VectorLaneNet Yol Graf Topolojisi ve Şerit Polinom Çekirdeği
==================================================================
Bu modül; 2D piksel maskesi yerine parametrik 3. Derece Şerit Polinomlarını,
Eğrilik ($\kappa(x)$) analizini ve Kavşak Şerit Bağlantılarını Yönlendirilmiş
Asiklik Graf (DAG) ve Komşuluk Matrisi ($A_{N \times N}$) olarak gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaVectorLaneNet:
    """
    Vektörel Şerit ve Kavşak Topolojisi Graf Motoru.
    """
    def __init__(self):
        # Şerit Düğümleri (Nodes): List of Dicts
        self.lane_nodes: List[Dict[str, Any]] = []
        # Komşuluk Matrisi (Adjacency Matrix): N x N
        self.adjacency_matrix: np.ndarray = np.zeros((0, 0), dtype=np.uint8)

    def evaluate_lane_polynomial(self, coeffs: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
        """
        y(x) = c0 + c1*x + c2*x^2 + c3*x^3
        """
        c0, c1, c2, c3 = coeffs
        return c0 + c1 * x_vals + c2 * (x_vals ** 2) + c3 * (x_vals ** 3)

    def compute_lane_curvature(self, coeffs: np.ndarray, x_val: float) -> float:
        """
        Şerit Eğriliği: kappa = |y''(x)| / (1 + (y'(x))^2)^(3/2)
        y'(x) = c1 + 2*c2*x + 3*c3*x^2
        y''(x) = 2*c2 + 6*c3*x
        """
        _, c1, c2, c3 = coeffs
        y_prime = c1 + 2.0 * c2 * x_val + 3.0 * c3 * (x_val ** 2)
        y_double_prime = 2.0 * c2 + 6.0 * c3 * x_val
        kappa = abs(y_double_prime) / ((1.0 + (y_prime ** 2)) ** 1.5)
        return float(kappa)

    def construct_synthetic_intersection_graph(self) -> Dict[str, Any]:
        """
        Kavşak Topolojisi:
        0: Yaklaşan Sol Şerit
        1: Yaklaşan Sağ Şerit
        2: Sola Dönüş Yayı
        3: Düz İlerleme Yolu
        4: Sağa Dönüş Yayı
        """
        self.lane_nodes = [
            {"id": 0, "name": "Approach_Left", "poly": np.array([-1.85, 0.0, 0.0, 0.0]), "type": "STRAIGHT"},
            {"id": 1, "name": "Approach_Right", "poly": np.array([1.85, 0.0, 0.0, 0.0]), "type": "STRAIGHT"},
            {"id": 2, "name": "Turn_Left", "poly": np.array([-1.85, -0.1, -0.01, 0.0]), "type": "LEFT_TURN"},
            {"id": 3, "name": "Go_Straight", "poly": np.array([0.0, 0.0, 0.0, 0.0]), "type": "STRAIGHT"},
            {"id": 4, "name": "Turn_Right", "poly": np.array([1.85, 0.1, 0.01, 0.0]), "type": "RIGHT_TURN"},
        ]

        n = len(self.lane_nodes)
        self.adjacency_matrix = np.zeros((n, n), dtype=np.uint8)

        # 0. Şerit -> Sola Dönüş (2) ve Düz (3) gidebilir
        self.adjacency_matrix[0, 2] = 1
        self.adjacency_matrix[0, 3] = 1

        # 1. Şerit -> Düz (3) ve Sağa Dönüş (4) gidebilir
        self.adjacency_matrix[1, 3] = 1
        self.adjacency_matrix[1, 4] = 1

        # Şerit Değişimi: 0 <-> 1
        self.adjacency_matrix[0, 1] = 1
        self.adjacency_matrix[1, 0] = 1

        return {
            "node_count": n,
            "nodes": self.lane_nodes,
            "adjacency_matrix": self.adjacency_matrix
        }

    def get_legal_next_lanes(self, current_lane_id: int) -> List[int]:
        """
        Mevcut şeritten geçilebilecek yasal sonraki şerit ID'lerini döndürür.
        """
        if 0 <= current_lane_id < len(self.adjacency_matrix):
            return list(np.where(self.adjacency_matrix[current_lane_id] == 1)[0])
        return []
