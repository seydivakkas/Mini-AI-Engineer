"""
CIE Delta-E 2000 (Delta-E 00) Hassas Renk Farkı Hesaplayıcı.
"""

from typing import Union
import math
import numpy as np


def delta_e_2000(
    lab1: Union[np.ndarray, list],
    lab2: Union[np.ndarray, list],
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0
) -> float:
    """
    İki CIELAB renk vektörü arasındaki CIE Delta-E 2000 renk farkını hesaplar.
    lab1, lab2: [L*, a*, b*]
    """
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    C_bar = (C1 + C2) / 2.0

    C_bar_7 = C_bar ** 7
    G = 0.5 * (1.0 - math.sqrt(C_bar_7 / (C_bar_7 + 25.0 ** 7 + 1e-12)))

    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2

    C1_prime = math.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = math.sqrt(a2_prime ** 2 + b2 ** 2)
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360.0
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360.0

    # Delta L', Delta C'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Delta h'
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0.0
    else:
        diff = h2_prime - h1_prime
        if abs(diff) <= 180.0:
            delta_h_prime = diff
        elif diff > 180.0:
            delta_h_prime = diff - 360.0
        else:
            delta_h_prime = diff + 360.0

    delta_H_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2.0))

    # Bar L', Bar h'
    L_bar_prime = (L1 + L2) / 2.0

    if C1_prime * C2_prime == 0:
        h_bar_prime = h1_prime + h2_prime
    else:
        diff_h = abs(h1_prime - h2_prime)
        sum_h = h1_prime + h2_prime
        if diff_h <= 180.0:
            h_bar_prime = sum_h / 2.0
        elif sum_h < 360.0:
            h_bar_prime = (sum_h + 360.0) / 2.0
        else:
            h_bar_prime = (sum_h - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_prime - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_prime))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_prime - 63.0))
    )

    S_L = 1.0 + (0.015 * (L_bar_prime - 50.0) ** 2) / math.sqrt(20.0 + (L_bar_prime - 50.0) ** 2)
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T

    C_bar_prime_7 = C_bar_prime ** 7
    R_C = 2.0 * math.sqrt(C_bar_prime_7 / (C_bar_prime_7 + 25.0 ** 7 + 1e-12))
    delta_theta = 30.0 * math.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2))
    R_T = -math.sin(math.radians(2.0 * delta_theta)) * R_C

    dE2 = (
        (delta_L_prime / (kL * S_L)) ** 2
        + (delta_C_prime / (kC * S_C)) ** 2
        + (delta_H_prime / (kH * S_H)) ** 2
        + R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H))
    )

    return float(math.sqrt(max(dE2, 0.0)))
