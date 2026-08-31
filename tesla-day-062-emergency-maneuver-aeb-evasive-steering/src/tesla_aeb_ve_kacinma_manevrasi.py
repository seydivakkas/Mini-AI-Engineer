r"""
Tesla Acil Durum Manevraları ve Otomatik Acil Frenleme (AEB) Çekirdeği
========================================================================
Bu modül; Euro-NCAP AEB Protokolünü, Çarpışma Uyarı Zamanını (FCW),
Tam Acil Frenleme Durma Mesafesini ($d_{\text{stop}} = v \cdot t_{\text{react}} + \frac{v^2}{2 a_{\max}}$)
ve Acil Kaçınma Direksiyonu (AES) Karar Mantığını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np


class AEBLevel(Enum):
    NORMAL = "NORMAL"
    FCW_WARNING = "FCW_WARNING"
    PARTIAL_BRAKE = "PARTIAL_BRAKE"
    FULL_AEB = "FULL_AEB"
    EVASIVE_STEER = "EVASIVE_STEER"


class TeslaAEBController:
    """
    Tesla FSD Euro-NCAP Otonom Acil Frenleme (AEB) ve Kaçınma Kontrolcüsü.
    """
    def __init__(
        self,
        max_aeb_decel_mps2: float = 9.0,     # Maksimum 0.92g acil fren ivmesi
        partial_decel_mps2: float = 4.0,     # Kısmi frenleme ivmesi
        system_delay_s: float = 0.20,        # Hidrolik / CAN veri yolu gecikmesi
        fcw_ttc_s: float = 2.4,              # Çarpışma uyarısı TTC eşiği
        partial_ttc_s: float = 1.6,          # Kısmi frenleme TTC eşiği
        full_aeb_ttc_s: float = 1.0          # Tam acil frenleme TTC eşiği
    ):
        self.a_max = max_aeb_decel_mps2
        self.a_partial = partial_decel_mps2
        self.t_delay = system_delay_s
        self.ttc_fcw = fcw_ttc_s
        self.ttc_partial = partial_ttc_s
        self.ttc_full = full_aeb_ttc_s

    def compute_emergency_stopping_distance(self, speed_mps: float) -> float:
        """
        Acil Durum Durma Mesafesi:
        d_stop = v * t_delay + v^2 / (2 * a_max)
        """
        reaction_dist = speed_mps * self.t_delay
        braking_dist = (speed_mps ** 2) / (2.0 * self.a_max)
        return float(reaction_dist + braking_dist)

    def evaluate_aeb_trigger(
        self,
        ego_speed_mps: float,
        dist_to_obstacle_m: float,
        rel_speed_mps: float,
        is_adjacent_lane_clear: bool = False
    ) -> Dict[str, Any]:
        """
        Anlık TTC ve durma mesafesine göre AEB / FCW / AES kararını üretir.
        """
        d_stop = self.compute_emergency_stopping_distance(ego_speed_mps)

        # TTC Hesabı
        if rel_speed_mps <= 0.1:
            ttc = 999.0
        else:
            ttc = dist_to_obstacle_m / rel_speed_mps

        # AEB Hiyerarşik Karar Mantığı
        if dist_to_obstacle_m < (d_stop * 0.75) and is_adjacent_lane_clear:
            # Yalnızca frenleme yetersiz kalıyorsa ve yan şerit boşsa -> Acil Kaçınma Direksiyonu (AES)
            level = AEBLevel.EVASIVE_STEER
            target_acc = -self.a_partial
            target_steer = 0.40  # Acil kaçış açısı
            action_desc = "ACİL KAÇINMA DİREKSİYONU (AES) DEVREDE"
        elif ttc <= self.ttc_full or dist_to_obstacle_m <= d_stop:
            level = AEBLevel.FULL_AEB
            target_acc = -self.a_max
            target_steer = 0.0
            action_desc = "TAM ACİL FRENLEME (AEB -9.0 m/s²)"
        elif ttc <= self.ttc_partial:
            level = AEBLevel.PARTIAL_BRAKE
            target_acc = -self.a_partial
            target_steer = 0.0
            action_desc = "KISMİ FRENLEME (-4.0 m/s²)"
        elif ttc <= self.ttc_fcw:
            level = AEBLevel.FCW_WARNING
            target_acc = 0.0
            target_steer = 0.0
            action_desc = "ÇARPIŞMA UYARISI (FCW SESLİ/GÖRSEL)"
        else:
            level = AEBLevel.NORMAL
            target_acc = 0.0
            target_steer = 0.0
            action_desc = "GÜVENLİ TAKİP MESAFESİ"

        return {
            "aeb_level": level.value,
            "ttc_s": float(ttc),
            "stopping_dist_m": d_stop,
            "dist_to_obstacle_m": dist_to_obstacle_m,
            "target_acc_mps2": float(target_acc),
            "target_steer_rad": float(target_steer),
            "action_desc": action_desc,
            "is_emergency": bool(level in [AEBLevel.FULL_AEB, AEBLevel.EVASIVE_STEER])
        }
