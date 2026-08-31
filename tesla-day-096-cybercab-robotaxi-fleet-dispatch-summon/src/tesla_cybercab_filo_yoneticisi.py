r"""
Tesla Cybercab / Robotaxi Otonom Filo Görevlendirme ve Çağırma Çekirdeği
========================================================================
Bu modül; pedalsız/direksiyonsuz Tesla Cybercab filosunun dinamik yolcu
eşleştirmesini, asgari bekleme süresi (ETA $< 3\text{ dk}$) optimizasyonunu,
batarya durumuna göre otonom endüktif şarj yönlendirmesini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class CybercabVehicle:
    cab_id: str
    x_km: float
    y_km: float
    soc_pct: float
    status: str  # "AVAILABLE", "ON_TRIP", "CHARGING"


@dataclass
class PassengerRequest:
    req_id: str
    pickup_x_km: float
    pickup_y_km: float
    dest_x_km: float
    dest_y_km: float


class TeslaCybercabFleetDispatcher:
    """
    Tesla Cybercab Filo ve Otonom Çağırma (Summon) Görevlendiricisi.
    """
    def __init__(self, avg_speed_kmh: float = 45.0, min_soc_for_trip: float = 20.0):
        self.speed_kmh = avg_speed_kmh
        self.min_soc = min_soc_for_trip

    def dispatch_trip(
        self,
        req: PassengerRequest,
        fleet: List[CybercabVehicle]
    ) -> Dict[str, Any]:
        """
        Gelen yolcu çağrısına en uygun müsait Cybercab'i eşleştirir.
        """
        available_cabs = [
            c for c in fleet
            if c.status == "AVAILABLE" and c.soc_pct >= self.min_soc
        ]

        if not available_cabs:
            return {
                "matched": False,
                "reason": "NO_AVAILABLE_CAB_WITH_ENOUGH_CHARGE",
                "assigned_cab_id": None,
                "eta_minutes": 999.0
            }

        # En yakın aracı bul (Öklid Mesafesi)
        best_cab = None
        min_dist_km = float('inf')

        for cab in available_cabs:
            dist = np.sqrt((cab.x_km - req.pickup_x_km)**2 + (cab.y_km - req.pickup_y_km)**2)
            if dist < min_dist_km:
                min_dist_km = dist
                best_cab = cab

        # ETA Hesabı: (Mesafe / Hız) * 60 dakika
        eta_min = (min_dist_km / self.speed_kmh) * 60.0

        # Yolculuk mesafesi
        trip_dist_km = float(np.sqrt((req.dest_x_km - req.pickup_x_km)**2 + (req.dest_y_km - req.pickup_y_km)**2))

        return {
            "matched": True,
            "assigned_cab_id": best_cab.cab_id,
            "pickup_distance_km": round(float(min_dist_km), 2),
            "eta_minutes": round(float(eta_min), 2),
            "trip_distance_km": round(trip_dist_km, 2),
            "cab_soc_pct": best_cab.soc_pct,
            "autonomous_summon_active": True
        }

    def auto_supercharge_rebalancing(self, fleet: List[CybercabVehicle]) -> List[str]:
        """
        Bataryası %20'nin altına düşen araçları otomatik kablosuz şarj pedlerine yönlendirir.
        """
        routed_cabs = []
        for cab in fleet:
            if cab.soc_pct < self.min_soc and cab.status != "CHARGING":
                cab.status = "CHARGING"
                routed_cabs.append(cab.cab_id)
        return routed_cabs
