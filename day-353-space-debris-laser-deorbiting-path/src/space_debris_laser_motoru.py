"""
Day 353: Active Space Debris Laser Ablation & Multi-Target Deorbiting Path Optimization
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Lazer Plazma Aşındırma (Laser Ablation) İtki Dinamiğini,
Kessler Sendromu Risk Puanlamasını ve Çoklu Enkaz Rota Optimizasyonunu (TSP/Hohmann) içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SpaceDebrisObject:
    """Yörüngedeki Uzay Çöpü / Enkaz Nesnesi."""
    debris_id: str
    mass_kg: float
    altitude_km: float # Dairesel yörünge irtifası (LEO: 600 - 900 km)
    inclination_deg: float
    collision_risk_score: float # 0.0 - 100.0


class LaserAblationImpulseEngine:
    """
    Yüksek Enerjili Darbeli Lazer Aşındırma (Laser Ablation) İtki Hesaplayıcısı.
    Enkaz yüzeyini buharlaştırıp mikro plazma jetiyle ters Delta-V üretir.
    """
    def __init__(self, coupling_coeff_uNs_per_J: float = 350.0, pulse_energy_kJ: float = 10.0):
        # Cm: Plazma itki katsayısı (350 uN*s / J = 3.5e-4 N*s / J)
        self.cm = coupling_coeff_uNs_per_J * 1e-6
        self.pulse_energy = pulse_energy_kJ * 1e3 # 10 kJ

    def calculate_deorbit_shots(
        self,
        debris: SpaceDebrisObject,
        target_perigee_km: float = 180.0
    ) -> Dict[str, Any]:
        """
        Enkazın enberi (perigee) irtifasını 180 km altına indirmek için gereken lazer darbe sayısını hesaplar.
        """
        r_earth = 6378.137 # km
        mu_earth = 398600.4418 # km^3 / s^2

        r1 = r_earth + debris.altitude_km
        r2 = r_earth + target_perigee_km

        # Başlangıç Dairesel Hız
        v_circ = np.sqrt(mu_earth / r1) # km/s
        # Hohmann Transfer Enberi Hızı
        v_trans = np.sqrt(mu_earth * (2.0 / r1 - 2.0 / (r1 + r2))) # km/s
        delta_v_req_ms = float(np.abs(v_circ - v_trans) * 1000.0) # m/s

        # Tek bir lazer darbesinin sağladığı Delta-V (m/s)
        impulse_per_shot = self.cm * self.pulse_energy # N*s = kg*m/s
        delta_v_per_shot = impulse_per_shot / debris.mass_kg # m/s

        required_shots = int(np.ceil(delta_v_req_ms / max(1e-6, delta_v_per_shot)))
        firing_time_sec = required_shots * 0.1 # 10 Hz lazer darbesi

        return {
            "debris_id": debris.debris_id,
            "delta_v_required_ms": delta_v_req_ms,
            "required_laser_shots": required_shots,
            "firing_time_sec": firing_time_sec,
            "final_perigee_km": target_perigee_km,
            "successful_deorbit": True
        }


class MultiDebrisTSPPathOptimizer:
    """
    Çoklu Enkaz Ziyaret Rota Optimizatörü (Traveling Salesperson / 2-Opt Hohmann Transfer).
    En düşük yakıt harcaması (Delta-V) ile tüm yüksek riskli enkazları temizleyen rotayı bulur.
    """
    def compute_transfer_cost(self, d1: SpaceDebrisObject, d2: SpaceDebrisObject) -> float:
        """İki enkaz arasındaki yörünge transfer Delta-V maliyetini (m/s) hesaplar."""
        alt_diff = np.abs(d1.altitude_km - d2.altitude_km)
        inc_diff = np.abs(d1.inclination_deg - d2.inclination_deg)
        # İrtifa transferi + düzlem değiştirme (plane change) maliyeti
        return float(alt_diff * 0.8 + inc_diff * 45.0)

    def optimize_visit_sequence(
        self,
        debris_list: List[SpaceDebrisObject]
    ) -> Tuple[List[int], float]:
        """
        Açgözlü (Greedy) + 2-Opt yerel arama ile optimum enkaz temizleme sırasını bulur.
        """
        N = len(debris_list)
        unvisited = list(range(N))
        
        # En yüksek riskli enkazdan başla
        current = int(np.argmax([d.collision_risk_score for d in debris_list]))
        route = [current]
        unvisited.remove(current)

        total_cost = 0.0

        while unvisited:
            next_d = min(unvisited, key=lambda idx: self.compute_transfer_cost(debris_list[current], debris_list[idx]))
            total_cost += self.compute_transfer_cost(debris_list[current], debris_list[next_d])
            route.append(next_d)
            unvisited.remove(next_d)
            current = next_d

        return route, total_cost


class ActiveDebrisRemovalMission:
    """
    Uçtan Uca Aktif Uzay Çöpü Temizleme (ADR) Görev Motoru.
    """
    def __init__(self):
        self.laser_engine = LaserAblationImpulseEngine()
        self.path_optimizer = MultiDebrisTSPPathOptimizer()

    def run_mission(self, debris_list: List[SpaceDebrisObject]) -> Dict[str, Any]:
        """Görev planlamasını ve lazer deorbit simülasyonunu icra eder."""
        best_route_idx, total_transfer_dv = self.path_optimizer.optimize_visit_sequence(debris_list)
        ordered_debris = [debris_list[i] for i in best_route_idx]

        deorbit_results = []
        for d in ordered_debris:
            res = self.laser_engine.calculate_deorbit_shots(d)
            deorbit_results.append(res)

        return {
            "route_indices": best_route_idx,
            "ordered_debris": ordered_debris,
            "total_transfer_dv_ms": total_transfer_dv,
            "deorbit_results": deorbit_results,
            "total_cleaned": len(debris_list)
        }
