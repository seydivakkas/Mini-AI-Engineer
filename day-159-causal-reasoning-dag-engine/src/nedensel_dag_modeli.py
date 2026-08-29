"""
Nedensel Graf ve Yapısal Model (Causal DAG & SCM) Modülü (Day 159 - Faz 8).
Değişkenler arasındaki neden-sonuç ilişkilerini (DAG) ve olasılık dağılımlarını tanımlar.
"""

from typing import Dict, Any, List, Tuple


class NedenselDAGModeli:
    """Neden-Sonuç Grafı (DAG) ve Olasılık Dağılım Modeli."""

    def __init__(self):
        # Düğümler: Z (Konfondör - Yaş), X (Müdahale - İlaç), Y (Sonuç - İyileşme)
        self.dugumler = ["Z", "X", "Y"]
        self.kenarlar = [("Z", "X"), ("Z", "Y"), ("X", "Y")]

        # P(Z) Dağılımı: 0 -> Genç (%50), 1 -> Yaşlı (%50)
        self.p_z = {0: 0.50, 1: 0.50}

        # P(X | Z): İlaç Alma Olasılığı (Gençler daha çok ilaç alıyor)
        self.p_x_given_z = {
            0: {1: 0.80, 0: 0.20},  # Gençler: %80 ilaç aldı
            1: {1: 0.20, 0: 0.80},  # Yaşlılar: %20 ilaç aldı
        }

        # P(Y | X, Z): İyileşme Olasılığı
        self.p_y_given_xz = {
            (0, 0): 0.80,  # Genç, İlaç Almadı -> %80 İyileşme
            (1, 0): 0.90,  # Genç, İlaç Aldı    -> %90 İyileşme (+%10 İlaç Etkisi)
            (0, 1): 0.40,  # Yaşlı, İlaç Almadı -> %40 İyileşme
            (1, 1): 0.50,  # Yaşlı, İlaç Aldı   -> %50 İyileşme (+%10 İlaç Etkisi)
        }

    def gozlemsel_olasilik_hesapla(self) -> Dict[str, float]:
        """
        1. Basamak: Gözlemsel Korelasyon P(Y=1 | X=1) vs P(Y=1 | X=0).
        """
        # Bayes Kuralı ile P(Z | X)
        # P(X=1) = P(X=1|Z=0)*P(Z=0) + P(X=1|Z=1)*P(Z=1) = 0.8*0.5 + 0.2*0.5 = 0.50
        # P(X=0) = P(X=0|Z=0)*P(Z=0) + P(X=0|Z=1)*P(Z=1) = 0.2*0.5 + 0.8*0.5 = 0.50
        p_z0_given_x1 = (0.80 * 0.50) / 0.50  # 0.80
        p_z1_given_x1 = (0.20 * 0.50) / 0.50  # 0.20

        p_z0_given_x0 = (0.20 * 0.50) / 0.50  # 0.20
        p_z1_given_x0 = (0.80 * 0.50) / 0.50  # 0.80

        # P(Y=1 | X=1) = P(Y=1|X=1,Z=0)*P(Z=0|X=1) + P(Y=1|X=1,Z=1)*P(Z=1|X=1)
        p_y1_given_x1 = 0.90 * p_z0_given_x1 + 0.50 * p_z1_given_x1
        p_y1_given_x0 = 0.80 * p_z0_given_x0 + 0.40 * p_z1_given_x0

        return {
            "p_y1_given_x1": round(p_y1_given_x1, 3),  # 0.820
            "p_y1_given_x0": round(p_y1_given_x0, 3),  # 0.480
            "gozlemsel_fark": round(p_y1_given_x1 - p_y1_given_x0, 3),  # 0.340 (Yanıltıcı/Şişirilmiş)
        }
