"""
Day 349: Battle Management Language (BML) & C2 Decision Support AI (TEWA)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Taktik Tehdit Değerlendirmesini (Threat Evaluation), Silah Tahsis Optimizasyonunu (Weapon Assignment - TEWA)
ve NATO C-BML (Coalition Battle Management Language) Standart Muharebe Emri Üretimini içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class BattlefieldThreat:
    """Muharebe Sahası Düşman Tehdit Unsuru."""
    threat_id: str
    threat_type: str # "CRUISE_MISSILE", "FIGHTER_JET", "DRONE_SWARM"
    position_km: np.ndarray # [x, y, z] km
    velocity_kms: np.ndarray # [vx, vy, vz] km/s
    threat_value: float # Tehdit öncelik puanı (10.0 - 100.0)


@dataclass
class DefenseAsset:
    """Dost Savunma ve Angajman Unsuru."""
    asset_id: str
    asset_type: str # "SAM_HISAR_O", "INTERCEPTOR_KAAN", "CIWS_GOKDENIZ"
    position_km: np.ndarray
    max_range_km: float
    ammo_remaining: int
    base_pk: float # Temel imha olasılığı (0.0 - 1.0)


class TEWAOptimizer:
    """
    Tehdit Değerlendirme ve Silah Tahsis (Threat Evaluation and Weapon Assignment - TEWA) Optimizatörü.
    Tehdit önceliği (V_i) ve imha olasılığını (P_k) maksimize eden eşleşmeyi (Greedy / Munkres) bulur.
    """
    def solve_assignment(
        self,
        threats: List[BattlefieldThreat],
        assets: List[DefenseAsset]
    ) -> List[Dict[str, Any]]:
        """
        Tehditlere en uygun savunma unsurlarını atar.
        """
        assignments = []
        available_assets = {a.asset_id: a for a in assets if a.ammo_remaining > 0}

        # Tehditleri öncelik puanına (Threat Value) göre büyükten küçüğe sırala
        sorted_threats = sorted(threats, key=lambda t: t.threat_value, reverse=True)

        for threat in sorted_threats:
            best_asset = None
            best_score = -1.0
            best_pk = 0.0

            for a_id, asset in available_assets.items():
                dist_km = float(np.linalg.norm(threat.position_km - asset.position_km))
                if dist_km <= asset.max_range_km:
                    # Menzile göre efektif imha olasılığı (P_k)
                    pk = asset.base_pk * np.exp(-0.5 * (dist_km / asset.max_range_km))
                    score = threat.threat_value * pk

                    if score > best_score:
                        best_score = score
                        best_asset = asset
                        best_pk = pk

            if best_asset is not None:
                assignments.append({
                    "threat_id": threat.threat_id,
                    "threat_type": threat.threat_type,
                    "assigned_asset_id": best_asset.asset_id,
                    "assigned_asset_type": best_asset.asset_type,
                    "expected_pk": float(best_pk),
                    "target_distance_km": float(np.linalg.norm(threat.position_km - best_asset.position_km)),
                    "tewa_score": float(best_score)
                })
                # Mühimmat düş ve müsaitliği güncelle
                best_asset.ammo_remaining -= 1
                if best_asset.ammo_remaining <= 0:
                    del available_assets[best_asset.asset_id]

        return assignments


class BMLOrderGenerator:
    """
    NATO Coalition Battle Management Language (C-BML) 5W Formatlı Taktik Operasyon Emri Üreticisi.
    Format: [Who, What, Where, When, Why]
    """
    @staticmethod
    def generate_bml_order(assignment: Dict[str, Any], threat: BattlefieldThreat) -> Dict[str, Any]:
        """Yapılandırılmış C-BML Angajman Emri Oluşturur."""
        return {
            "BML_ORDER_ID": f"ORD_{assignment['assigned_asset_id']}_{assignment['threat_id']}",
            "WHO": assignment["assigned_asset_id"],
            "WHAT": "INTERCEPT_AND_DESTROY",
            "WHERE": {
                "LAT_LON_ALT_KM": threat.position_km.tolist(),
                "ESTIMATED_INTERCEPT_KM": assignment["target_distance_km"]
            },
            "WHEN": "IMMEDIATE_AT_DECISION_T0",
            "WHY": f"NEUTRALIZE_HIGH_PRIORITY_THREAT_{assignment['threat_type']}",
            "EXPECTED_PK": assignment["expected_pk"]
        }


class BattleManagementEngine:
    """
    Uçtan Uca C2 Karar Destek ve Angajman Yönetim Motoru.
    """
    def __init__(self):
        self.optimizer = TEWAOptimizer()
        self.bml_gen = BMLOrderGenerator()

    def process_c2_cycle(
        self,
        threats: List[BattlefieldThreat],
        assets: List[DefenseAsset]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """C2 Karar Çevrimini (OODA Loop) tamamlar ve BML emirlerini döner."""
        assignments = self.optimizer.solve_assignment(threats, assets)
        
        threat_dict = {t.threat_id: t for t in threats}
        bml_orders = []
        for asgn in assignments:
            th = threat_dict[asgn["threat_id"]]
            order = self.bml_gen.generate_bml_order(asgn, th)
            bml_orders.append(order)

        return assignments, bml_orders
