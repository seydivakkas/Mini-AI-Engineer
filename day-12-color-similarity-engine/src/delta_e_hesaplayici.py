"""CIELAB Uzayında Algısal Delta-E Renk Mesafesi Hesaplayıcı.

Bu modül; RGB renklerini uluslararası standart CIELAB float uzayına dönüştürür
ve insan gözünün renk algısına göre CIE76 ve modern CIEDE2000 Delta-E mesafelerini hesaplar.
"""

from typing import Tuple
import cv2
import numpy as np


class DeltaEHesaplayici:
    """CIELAB renk uzayında algısal farkları (Delta-E) hesaplayan sınıf."""

    @staticmethod
    def rgb_to_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
        """RGB tamsayı [0, 255] rengini standart CIELAB (L*, a*, b*) koordinatlarına çevirir.

        Döndürür:
            np.ndarray: [L*, a*, b*] float32 vektörü.
                        L* in [0, 100], a* in [-128, 127], b* in [-128, 127]
        """
        r, g, b = rgb
        piksel_norm = np.float32([[[r / 255.0, g / 255.0, b / 255.0]]])
        lab = cv2.cvtColor(piksel_norm, cv2.COLOR_RGB2LAB)[0, 0]
        return lab.astype(np.float64)

    @staticmethod
    def cie76_mesafesi(lab1: np.ndarray, lab2: np.ndarray) -> float:
        """CIE76 Algısal Öklid Renk Mesafesi (Delta E 76).

        Formül:
            Delta_E_76 = sqrt((L1 - L2)^2 + (a1 - a2)^2 + (b1 - b2)^2)
        """
        fark = np.asarray(lab1, dtype=np.float64) - np.asarray(lab2, dtype=np.float64)
        return float(np.sqrt(np.sum(fark**2)))

    @staticmethod
    def ciede2000_mesafesi(lab1: np.ndarray, lab2: np.ndarray) -> float:
        """CIEDE2000 (Delta E 00) — İnsan gözünün ışık, doygunluk ve renk tonu

        farklarına olan non-lineer hassasiyetini modelleyen en gelişmiş standart.
        """
        L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
        L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

        if np.allclose([L1, a1, b1], [L2, a2, b2], atol=1e-5):
            return 0.0

        ortalama_L = (L1 + L2) / 2.0
        C1 = np.hypot(a1, b1)
        C2 = np.hypot(a2, b2)
        ortalama_C = (C1 + C2) / 2.0

        G = 0.5 * (1.0 - np.sqrt((ortalama_C**7) / (ortalama_C**7 + 25.0**7 + 1e-12)))
        a1_us = (1.0 + G) * a1
        a2_us = (1.0 + G) * a2
        C1_us = np.hypot(a1_us, b1)
        C2_us = np.hypot(a2_us, b2)
        ortalama_C_us = (C1_us + C2_us) / 2.0

        h1_us = np.degrees(np.arctan2(b1, a1_us)) % 360.0
        h2_us = np.degrees(np.arctan2(b2, a2_us)) % 360.0

        fark_h_us = h2_us - h1_us
        if abs(fark_h_us) > 180.0:
            fark_h_us -= 360.0 * np.sign(fark_h_us)

        delta_L_us = L2 - L1
        delta_C_us = C2_us - C1_us
        delta_H_us = 2.0 * np.sqrt(C1_us * C2_us) * np.sin(np.radians(fark_h_us / 2.0))

        if abs(h1_us - h2_us) > 180.0:
            ortalama_h_us = (h1_us + h2_us + 360.0) / 2.0
        else:
            ortalama_h_us = (h1_us + h2_us) / 2.0

        T = (
            1.0
            - 0.17 * np.cos(np.radians(ortalama_h_us - 30.0))
            + 0.24 * np.cos(np.radians(2.0 * ortalama_h_us))
            + 0.32 * np.cos(np.radians(3.0 * ortalama_h_us + 6.0))
            - 0.20 * np.cos(np.radians(4.0 * ortalama_h_us - 63.0))
        )

        S_L = 1.0 + (0.015 * (ortalama_L - 50.0)**2) / np.sqrt(20.0 + (ortalama_L - 50.0)**2)
        S_C = 1.0 + 0.045 * ortalama_C_us
        S_H = 1.0 + 0.015 * ortalama_C_us * T

        delta_theta = 30.0 * np.exp(-(((ortalama_h_us - 275.0) / 25.0)**2))
        R_C = 2.0 * np.sqrt((ortalama_C_us**7) / (ortalama_C_us**7 + 25.0**7 + 1e-12))
        R_T = -np.sin(np.radians(2.0 * delta_theta)) * R_C

        terim_L = delta_L_us / S_L
        terim_C = delta_C_us / S_C
        terim_H = delta_H_us / S_H

        delta_e = np.sqrt(terim_L**2 + terim_C**2 + terim_H**2 + R_T * terim_C * terim_H)
        return float(delta_e)
