r"""
Tesla Şehir İçi Kavşak ve Döner Kavşak (Roundabout) Karar Ağacı Çekirdeği
==========================================================================
Bu modül; Döner Kavşak Geçiş Önceliği (Right-of-Way Rules),
Time-To-Collision (TTC) & Güvenli Aralık Kabul Modeli (Gap Acceptance Model)
ve Hiyerarşik Sonlu Durum Makinesi (HFSM) Karar Motorunu gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np


class RoundaboutState(Enum):
    APPROACHING = "APPROACHING"
    YIELDING = "YIELDING"
    ENTERING = "ENTERING"
    CIRCULATING = "CIRCULATING"
    EXITING = "EXITING"


class TeslaIntersectionDecisionTree:
    """
    Tesla FSD Şehir İçi Kavşak ve Döner Kavşak Karar Ağacı.
    """
    def __init__(self, min_ttc_safe_s: float = 3.5, yield_dist_threshold_m: float = 15.0):
        self.min_ttc = min_ttc_safe_s
        self.yield_dist = yield_dist_threshold_m

    def compute_ttc(self, dist_to_approaching_m: float, approaching_speed_mps: float) -> float:
        """
        Time-To-Collision (TTC):
        TTC = d / v_rel
        """
        if approaching_speed_mps <= 0.1:
            return 999.0  # Duran araç veya yaklaşmayan araç
        return float(dist_to_approaching_m / approaching_speed_mps)

    def can_enter_intersection(self, dist_to_approaching_m: float, approaching_speed_mps: float) -> bool:
        """
        Kavşağa / Döner Kavşağa Giriş Güvenlik Kararı (Gap Acceptance):
        TTC >= 3.5 sn ise GİRİŞ (True), değilse BEKLE (False).
        """
        ttc = self.compute_ttc(dist_to_approaching_m, approaching_speed_mps)
        return bool(ttc >= self.min_ttc)

    def evaluate_roundabout_scenario(
        self,
        ego_dist_to_yield_line_m: float,
        circulating_vehicles: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Döner kavşak içindeki araçların konum ve hızlarına göre anlık karar üretir.
        circulating_vehicles: [{"id": 1, "dist_m": 40.0, "speed_mps": 15.0}, ...]
        """
        # En kritik (en düşük TTC'ye sahip) aracı bul
        min_ttc = 999.0
        critical_veh_id = None

        for veh in circulating_vehicles:
            ttc = self.compute_ttc(veh["dist_m"], veh["speed_mps"])
            if ttc < min_ttc:
                min_ttc = ttc
                critical_veh_id = veh.get("id", -1)

        # Hiyerarşik Karar Mantığı
        if ego_dist_to_yield_line_m > self.yield_dist:
            state = RoundaboutState.APPROACHING
            action = "KAVŞAĞA YAKLAŞILIYOR (HIZ DÜŞÜR)"
            target_acc = -1.0
        else:
            # Yol verme çizgisindeyiz
            if min_ttc >= self.min_ttc:
                state = RoundaboutState.ENTERING
                action = "GÜVENLİ ARALIK BULUNDU (KAVŞAĞA GİR)"
                target_acc = 1.5
            else:
                state = RoundaboutState.YIELDING
                action = f"YOL VERİLİYOR (Araç {critical_veh_id}, TTC: {min_ttc:.1f}s < 3.5s)"
                target_acc = -2.0

        return {
            "state": state.value,
            "action": action,
            "min_ttc_s": min_ttc,
            "critical_vehicle_id": critical_veh_id,
            "target_acc_mps2": target_acc,
            "can_enter": bool(min_ttc >= self.min_ttc)
        }
